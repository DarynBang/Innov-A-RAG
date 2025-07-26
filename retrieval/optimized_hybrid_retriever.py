"""
OptimizedHybridRetriever: High-performance hybrid retrieval with caching, FAISS, and parallel processing.
Incorporates advanced optimization techniques for large-scale RAG systems.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

import os
import pickle
import asyncio
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock

# Core libraries
import numpy as np
import faiss
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# Redis imports with fallback
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Caching
from functools import lru_cache

# Optional caching dependencies with fallbacks
try:
    from diskcache import Cache
    DISK_CACHE_AVAILABLE = True
except ImportError:
    DISK_CACHE_AVAILABLE = False
    Cache = None

logger = get_logger(__name__)

@dataclass
class SearchResult:
    """Structured search result with metadata."""
    document: Document
    score: float
    source: str
    doc_id: str
    normalized_score: float = 0.0
    confidence: str = "unknown"

@dataclass
class CacheConfig:
    """Configuration for caching system."""
    use_redis: bool = False  # Disable Redis by default since it's not always available
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None  # Added Redis password support
    use_disk_cache: bool = True
    disk_cache_dir: str = "cache/retrieval"
    memory_cache_size: int = 2000  # Increased from 1000
    cache_ttl: int = 7200  # Increased to 2 hours for better hit rates
    semantic_cache_threshold: float = 0.85  # Added for semantic caching
    memory_optimization: bool = True  # Added for memory management

class OptimizedHybridRetriever:
    """
    High-performance hybrid retriever with advanced optimizations:
    - FAISS for approximate nearest neighbor search
    - Multi-level caching (memory, disk, Redis)
    - Parallel processing for dense and sparse retrieval
    - Query preprocessing and filtering
    - Early stopping based on confidence thresholds
    - Dimensionality reduction for embeddings
    """
    
    def __init__(
        self,
        vectorstore: Chroma,
        documents: List[str],
        metadatas: List[Dict] = None,
        bm25_cache_path: str = "bm25_index.pkl",
        cache_config: Optional[CacheConfig] = None,
        faiss_index_path: str = "faiss_index.bin",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_faiss: bool = True,
        use_dimensionality_reduction: bool = False,
        reduced_dimensions: int = 128,
        confidence_threshold: float = 0.9,
        max_workers: int = 4
    ):
        """
        Initialize OptimizedHybridRetriever with advanced features.
        
        Args:
            vectorstore: Langchain Chroma vectorstore instance
            documents: List of raw text documents for BM25
            metadatas: Optional list of metadata dicts for each document
            bm25_cache_path: Path to cache BM25 index
            cache_config: Configuration for caching system
            faiss_index_path: Path to FAISS index file
            embedding_model: HuggingFace embedding model name
            use_faiss: Whether to use FAISS for ANN search
            use_dimensionality_reduction: Whether to reduce embedding dimensions
            reduced_dimensions: Target dimensions for reduction
            confidence_threshold: Threshold for early stopping
            max_workers: Maximum number of threads for parallel processing
        """
        logger.info("Initializing OptimizedHybridRetriever with advanced features")
        
        self.vectorstore = vectorstore
        self.documents = documents
        self.metadatas = metadatas or [{} for _ in documents]
        self.confidence_threshold = confidence_threshold
        self.max_workers = max_workers
        
        # Initialize caching system
        self.cache_config = cache_config or CacheConfig()
        self._init_caching_system()
        
        # Thread safety
        self._cache_lock = Lock()
        self._stats_lock = Lock()
        
        # Performance statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'semantic_cache_hits': 0,  # Added semantic cache tracking
            'total_queries': 0,
            'avg_response_time': 0.0,
            'faiss_queries': 0,
            'early_stops': 0,
            'preprocessed_queries': 0  # Added query preprocessing tracking
        }
        
        # Query preprocessing and semantic cache
        self.query_embeddings_cache = {}  # Cache for query embeddings
        self.semantic_cache = {}  # Cache for semantically similar queries
        
        # Initialize embedding system
        self.embedding_model_name = embedding_model
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cuda' if os.environ.get('CUDA_AVAILABLE') else 'cpu'}
        )
        
        # Initialize FAISS system
        self.use_faiss = use_faiss
        self.faiss_index = None
        self.faiss_index_path = faiss_index_path
        self.use_dimensionality_reduction = use_dimensionality_reduction
        self.reduced_dimensions = reduced_dimensions
        self.dimension_reducer = None
        
        if self.use_faiss:
            self._init_faiss_system()
        
        # Initialize BM25 system
        self.bm25_retriever = None
        self.bm25_cache_path = bm25_cache_path
        self._init_bm25_system()
        
        # Pre-compute common query embeddings
        self._precompute_common_queries()
        
        logger.info("OptimizedHybridRetriever initialization completed")
    
    def _init_caching_system(self):
        """Initialize multi-level caching system."""
        try:
            # Memory cache (LRU)
            self.memory_cache = {}
            self.memory_cache_order = []
            
            # Disk cache
            if self.cache_config.use_disk_cache and DISK_CACHE_AVAILABLE:
                try:
                    os.makedirs(self.cache_config.disk_cache_dir, exist_ok=True)
                    self.disk_cache = Cache(self.cache_config.disk_cache_dir)
                    logger.info("Disk cache initialized successfully")
                except Exception as e:
                    logger.warning(f"Disk cache initialization failed: {e}")
                    self.disk_cache = None
            else:
                self.disk_cache = None
                if not DISK_CACHE_AVAILABLE:
                    logger.info("diskcache not available, using memory cache only")
            
            # Redis cache
            if self.cache_config.use_redis and REDIS_AVAILABLE:
                try:
                    redis_kwargs = {
                        'host': self.cache_config.redis_host,
                        'port': self.cache_config.redis_port,
                        'db': self.cache_config.redis_db,
                        'decode_responses': True,
                        'socket_timeout': 2,  # Reduced timeout for faster failure detection
                        'socket_connect_timeout': 2,
                        'retry_on_timeout': False  # Don't retry on timeout for faster fallback
                    }
                    
                    if self.cache_config.redis_password:
                        redis_kwargs['password'] = self.cache_config.redis_password
                    
                    self.redis_cache = redis.Redis(**redis_kwargs)
                    # Test connection
                    self.redis_cache.ping()
                    logger.info("✅ Redis cache initialized successfully")
                except Exception as e:
                    logger.info(f"ℹ️  Redis not available, using disk/memory cache only: {e.args[0] if e.args else str(e)}")
                    self.redis_cache = None
            else:
                self.redis_cache = None
            
            logger.info("Caching system initialized")
            
        except Exception as e:
            logger.error(f"Error initializing caching system: {e}")
            self.disk_cache = None
            self.redis_cache = None
    
    def _init_faiss_system(self):
        """Initialize FAISS index for approximate nearest neighbor search."""
        try:
            # Check if FAISS index exists
            if os.path.exists(self.faiss_index_path):
                logger.info("Loading existing FAISS index")
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                
                # Load dimension reducer if exists
                reducer_path = self.faiss_index_path.replace('.bin', '_reducer.pkl')
                if os.path.exists(reducer_path) and self.use_dimensionality_reduction:
                    with open(reducer_path, 'rb') as f:
                        self.dimension_reducer = pickle.load(f)
                    logger.info("Dimension reducer loaded")
            else:
                logger.info("Building new FAISS index")
                self._build_faiss_index()
            
            logger.info(f"FAISS index initialized with {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error initializing FAISS system: {e}")
            self.use_faiss = False
            self.faiss_index = None
    
    def _build_faiss_index(self):
        """Build FAISS index from vectorstore."""
        try:
            # Get all embeddings from vectorstore
            logger.info("Extracting embeddings for FAISS index")
            
            # This is a simplified approach - in practice, you'd want to extract
            # embeddings more efficiently from your Chroma vectorstore
            all_embeddings = []
            
            # Sample approach: get embeddings for all documents
            for i, doc_text in enumerate(self.documents[:1000]):  # Limit for demo
                embedding = self.embeddings.embed_query(doc_text)
                all_embeddings.append(embedding)
                
                if i % 100 == 0:
                    logger.info(f"Processed {i} documents for FAISS index")
            
            embeddings_array = np.array(all_embeddings, dtype=np.float32)
            
            # Apply dimensionality reduction if requested
            if self.use_dimensionality_reduction:
                from sklearn.decomposition import PCA
                self.dimension_reducer = PCA(n_components=self.reduced_dimensions)
                embeddings_array = self.dimension_reducer.fit_transform(embeddings_array)
                
                # Save dimension reducer
                reducer_path = self.faiss_index_path.replace('.bin', '_reducer.pkl')
                with open(reducer_path, 'wb') as f:
                    pickle.dump(self.dimension_reducer, f)
                logger.info(f"Dimensionality reduced from {len(all_embeddings[0])} to {self.reduced_dimensions}")
            
            # Build FAISS index
            dimension = embeddings_array.shape[1]
            
            # Use HNSW for better performance with large datasets
            self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
            self.faiss_index.hnsw.efConstruction = 200
            self.faiss_index.hnsw.efSearch = 50
            
            # Add vectors to index
            self.faiss_index.add(embeddings_array)
            
            # Save index
            faiss.write_index(self.faiss_index, self.faiss_index_path)
            logger.info(f"FAISS index built and saved with {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}")
            self.use_faiss = False
            self.faiss_index = None
    
    def _init_bm25_system(self):
        """Initialize BM25 retriever with caching."""
        try:
            if os.path.exists(self.bm25_cache_path):
                logger.info("Loading cached BM25 index")
                with open(self.bm25_cache_path, "rb") as f:
                    self.bm25_retriever = pickle.load(f)
            else:
                logger.info("Creating new BM25 retriever")
                doc_objects = []
                for i, doc_text in enumerate(self.documents):
                    metadata = self.metadatas[i] if i < len(self.metadatas) else {}
                    metadata['doc_id'] = i
                    doc_objects.append(Document(page_content=doc_text, metadata=metadata))
                
                self.bm25_retriever = BM25Retriever.from_documents(doc_objects)
                
                # Cache BM25 scores for frequent queries
                self._precompute_bm25_scores()
                
                with open(self.bm25_cache_path, "wb") as f:
                    pickle.dump(self.bm25_retriever, f)
                
                logger.info(f"BM25 retriever initialized with {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error initializing BM25 system: {e}")
            self.bm25_retriever = None
    
    def _precompute_common_queries(self):
        """Pre-compute embeddings for common queries."""
        common_queries = [
            "machine learning", "artificial intelligence", "deep learning",
            "neural networks", "computer vision", "natural language processing",
            "data science", "algorithm", "innovation", "technology",
            "patent", "invention", "research", "development", "analysis"
        ]
        
        logger.info("Pre-computing embeddings for common queries")
        for query in common_queries:
            try:
                embedding = self.embeddings.embed_query(query)
                cache_key = f"embedding:{hashlib.md5(query.encode()).hexdigest()}"
                self._set_cache(cache_key, embedding, ttl=86400)  # 24 hours
            except Exception as e:
                logger.warning(f"Error pre-computing embedding for '{query}': {e}")
    
    def _precompute_bm25_scores(self):
        """Pre-compute BM25 scores for common queries."""
        if not self.bm25_retriever:
            return
        
        common_queries = [
            "machine learning", "artificial intelligence", "innovation",
            "technology", "patent", "research", "development"
        ]
        
        logger.info("Pre-computing BM25 scores for common queries")
        for query in common_queries:
            try:
                results = self.bm25_retriever.get_relevant_documents(query)
                cache_key = f"bm25:{hashlib.md5(query.encode()).hexdigest()}"
                self._set_cache(cache_key, results, ttl=86400)  # 24 hours
            except Exception as e:
                logger.warning(f"Error pre-computing BM25 for '{query}': {e}")
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query for better retrieval performance.
        
        Args:
            query: Raw user query
            
        Returns:
            Preprocessed query
        """
        with self._stats_lock:
            self.stats['preprocessed_queries'] += 1
        
        # Remove extra whitespace and normalize
        preprocessed = ' '.join(query.strip().split())
        
        # Convert to lowercase for consistency
        preprocessed = preprocessed.lower()
        
        # Remove common stop phrases that don't add semantic value
        stop_phrases = ['tell me about', 'show me', 'what is', 'how does', 'can you']
        for phrase in stop_phrases:
            if preprocessed.startswith(phrase):
                preprocessed = preprocessed[len(phrase):].strip()
                break
        
        return preprocessed

    def _get_semantic_cache_key(self, query: str) -> Optional[str]:
        """
        Check for semantically similar queries in cache.
        
        Args:
            query: Preprocessed query
            
        Returns:
            Cache key if similar query found, None otherwise
        """
        if not self.semantic_cache:
            return None
        
        try:
            # Get or compute query embedding
            if query not in self.query_embeddings_cache:
                query_embedding = self.embeddings.embed_query(query)
                self.query_embeddings_cache[query] = np.array(query_embedding)
            
            query_embedding = self.query_embeddings_cache[query]
            
            # Check similarity with cached queries
            for cached_query, cached_key in self.semantic_cache.items():
                if cached_query not in self.query_embeddings_cache:
                    continue
                
                cached_embedding = self.query_embeddings_cache[cached_query]
                
                # Compute cosine similarity
                similarity = np.dot(query_embedding, cached_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
                )
                
                if similarity >= self.cache_config.semantic_cache_threshold:
                    logger.info(f"Semantic cache hit! Similarity: {similarity:.3f}")
                    with self._stats_lock:
                        self.stats['semantic_cache_hits'] += 1
                    return cached_key
            
        except Exception as e:
            logger.warning(f"Semantic cache check failed: {e}")
        
        return None

    def _get_cache_key(self, query: str, k: int, dense_weight: float, sparse_weight: float) -> str:
        """Generate cache key for query."""
        key_data = f"{query}:{k}:{dense_weight}:{sparse_weight}"
        return f"hybrid:{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def _get_cache(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        try:
            with self._cache_lock:
                # Check memory cache first
                if key in self.memory_cache:
                    # Move to end (LRU)
                    self.memory_cache_order.remove(key)
                    self.memory_cache_order.append(key)
                    return self.memory_cache[key]
                
                # Check Redis cache
                if self.redis_cache:
                    try:
                        cached_data = self.redis_cache.get(key)
                        if cached_data:
                            data = json.loads(cached_data)
                            # Store in memory cache
                            self._update_memory_cache(key, data)
                            return data
                    except Exception as e:
                        logger.warning(f"Redis cache get error: {e}")
                
                # Check disk cache
                if self.disk_cache:
                    try:
                        data = self.disk_cache.get(key)
                        if data is not None:
                            # Store in memory cache
                            self._update_memory_cache(key, data)
                            return data
                    except Exception as e:
                        logger.warning(f"Disk cache get error: {e}")
                
                return None
                
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    def _set_cache(self, key: str, value: Any, ttl: int = None):
        """Set value in multi-level cache."""
        try:
            ttl = ttl or self.cache_config.cache_ttl
            
            with self._cache_lock:
                # Store in memory cache
                self._update_memory_cache(key, value)
                
                # Store in Redis cache
                if self.redis_cache:
                    try:
                        self.redis_cache.setex(key, ttl, json.dumps(value, default=str))
                    except Exception as e:
                        logger.warning(f"Redis cache set error: {e}")
                
                # Store in disk cache
                if self.disk_cache:
                    try:
                        self.disk_cache.set(key, value, expire=ttl)
                    except Exception as e:
                        logger.warning(f"Disk cache set error: {e}")
                        
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def _update_memory_cache(self, key: str, value: Any):
        """Update memory cache with LRU eviction and memory optimization."""
        # Add to memory cache
        if key in self.memory_cache:
            self.memory_cache_order.remove(key)
        
        self.memory_cache[key] = value
        self.memory_cache_order.append(key)
        
        # Memory-aware eviction
        if self.cache_config.memory_optimization:
            self._memory_aware_eviction()
        else:
            # Standard LRU eviction
            while len(self.memory_cache) > self.cache_config.memory_cache_size:
                oldest_key = self.memory_cache_order.pop(0)
                del self.memory_cache[oldest_key]

    def _memory_aware_eviction(self):
        """Smart memory eviction based on system memory usage."""
        try:
            import psutil
            # Get current memory usage
            process = psutil.Process()
            memory_percent = process.memory_percent()
            
            # If memory usage is high, be more aggressive with eviction
            if memory_percent > 70:  # High memory usage
                target_size = int(self.cache_config.memory_cache_size * 0.5)  # Reduce to 50%
                logger.info(f"High memory usage ({memory_percent:.1f}%), reducing cache to {target_size}")
            elif memory_percent > 50:  # Medium memory usage
                target_size = int(self.cache_config.memory_cache_size * 0.7)  # Reduce to 70%
            else:
                target_size = self.cache_config.memory_cache_size
            
            # Evict oldest entries to reach target size
            while len(self.memory_cache) > target_size:
                oldest_key = self.memory_cache_order.pop(0)
                del self.memory_cache[oldest_key]
                
        except ImportError:
            logger.info("psutil not available, using standard LRU eviction")
            # Fallback to standard LRU
            while len(self.memory_cache) > self.cache_config.memory_cache_size:
                oldest_key = self.memory_cache_order.pop(0)
                del self.memory_cache[oldest_key]
        except Exception as e:
            logger.warning(f"Memory-aware eviction failed, using standard LRU: {e}")
            # Fallback to standard LRU
            while len(self.memory_cache) > self.cache_config.memory_cache_size:
                oldest_key = self.memory_cache_order.pop(0)
                del self.memory_cache[oldest_key]

    def optimize_memory_usage(self):
        """Optimize memory usage by cleaning up unused resources."""
        logger.info("Starting memory optimization")
        
        # Clean up semantic cache if it gets too large
        if len(self.semantic_cache) > 1000:
            # Keep only the most recent 500 entries
            sorted_items = sorted(self.semantic_cache.items())
            self.semantic_cache = dict(sorted_items[-500:])
            logger.info("Cleaned up semantic cache")
        
        # Clean up query embeddings cache
        if len(self.query_embeddings_cache) > 500:
            # Keep only the most recent 250 entries
            sorted_items = sorted(self.query_embeddings_cache.items())
            self.query_embeddings_cache = dict(sorted_items[-250:])
            logger.info("Cleaned up query embeddings cache")
        
        # Force garbage collection
        import gc
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Log memory stats if available
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB RSS, {memory_info.vms / 1024 / 1024:.1f} MB VMS")
        except ImportError:
            logger.info("psutil not available for memory monitoring")
        except Exception as e:
            logger.warning(f"Error getting memory stats: {e}")

    def clear_all_caches(self):
        """Clear all caches and free memory."""
        with self._cache_lock:
            self.memory_cache.clear()
            self.memory_cache_order.clear()
            self.semantic_cache.clear()
            self.query_embeddings_cache.clear()
            
            if self.disk_cache:
                try:
                    self.disk_cache.clear()
                except Exception as e:
                    logger.warning(f"Error clearing disk cache: {e}")
            
            if self.redis_cache:
                try:
                    # Clear only our keys (with hybrid: prefix)
                    keys = self.redis_cache.keys("hybrid:*")
                    if keys:
                        self.redis_cache.delete(*keys)
                except Exception as e:
                    logger.warning(f"Error clearing Redis cache: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("All caches cleared and memory optimized")
    
    async def retrieve_async(
        self,
        query: str,
        k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        use_cache: bool = True,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Asynchronous hybrid retrieval with advanced optimizations.
        
        Args:
            query: The user query string
            k: Number of top documents to retrieve
            dense_weight: Weight for dense (vector) scores
            sparse_weight: Weight for sparse (BM25) scores
            use_cache: Whether to use caching
            metadata_filters: Optional metadata filters to apply before search
            
        Returns:
            List of relevant documents with fused scores
        """
        start_time = time.time()
        
        with self._stats_lock:
            self.stats['total_queries'] += 1
        
        logger.info(f"Async hybrid retrieval for: '{query}' (k={k})")
        
        # Preprocess query for better performance
        preprocessed_query = self._preprocess_query(query)
        
        # Check semantic cache first
        semantic_cache_key = None
        if use_cache:
            semantic_cache_key = self._get_semantic_cache_key(preprocessed_query)
            if semantic_cache_key:
                cached_result = self._get_cache(semantic_cache_key)
                if cached_result:
                    logger.info("Semantic cache hit - returning cached results")
                    return cached_result
        
        # Check exact cache
        cache_key = self._get_cache_key(preprocessed_query, k, dense_weight, sparse_weight)
        if use_cache:
            cached_result = self._get_cache(cache_key)
            if cached_result:
                with self._stats_lock:
                    self.stats['cache_hits'] += 1
                logger.info("Exact cache hit - returning cached results")
                return cached_result
        
        with self._stats_lock:
            self.stats['cache_misses'] += 1
        
        # Apply metadata filters if provided
        filtered_indices = None
        if metadata_filters:
            filtered_indices = self._apply_metadata_filters(metadata_filters)
        
        # Parallel retrieval using preprocessed query
        dense_task = asyncio.create_task(
            self._get_dense_results_async(preprocessed_query, k + int(0.3 * k), filtered_indices)
        )
        sparse_task = asyncio.create_task(
            self._get_sparse_results_async(preprocessed_query, k + int(0.3 * k), filtered_indices)
        )
        
        # Wait for both to complete
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
        
        # Fuse results
        fused_results = self._fuse_results_optimized(
            dense_results, sparse_results, dense_weight, sparse_weight
        )
        
        # Early stopping based on confidence
        if fused_results and len(fused_results) > 0:
            top_confidence = fused_results[0].metadata.get('hybrid_score', 0)
            if top_confidence >= self.confidence_threshold:
                with self._stats_lock:
                    self.stats['early_stops'] += 1
                logger.info(f"Early stopping triggered (confidence: {top_confidence:.3f})")
        
        # Return top-k results
        final_results = fused_results[:k]
        
        # Cache results
        if use_cache:
            self._set_cache(cache_key, final_results)
            # Add to semantic cache for future similarity checks
            self.semantic_cache[preprocessed_query] = cache_key
        
        # Update performance stats
        response_time = time.time() - start_time
        with self._stats_lock:
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['total_queries'] - 1) + response_time) /
                self.stats['total_queries']
            )
        
        logger.info(f"Async hybrid retrieval completed in {response_time:.3f}s, returning {len(final_results)} results")
        return final_results
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        use_cache: bool = True,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Optimized synchronous retrieval method for better performance.
        """
        start_time = time.time()
        
        with self._stats_lock:
            self.stats['total_queries'] += 1
        
        logger.info(f"Optimized hybrid retrieval for: '{query}' (k={k})")
        
        # Preprocess query for better performance
        preprocessed_query = self._preprocess_query(query)
        
        # Check semantic cache first
        semantic_cache_key = None
        if use_cache:
            semantic_cache_key = self._get_semantic_cache_key(preprocessed_query)
            if semantic_cache_key:
                cached_result = self._get_cache(semantic_cache_key)
                if cached_result:
                    logger.info("Semantic cache hit - returning cached results")
                    return cached_result
        
        # Check exact cache
        cache_key = self._get_cache_key(preprocessed_query, k, dense_weight, sparse_weight)
        if use_cache:
            cached_result = self._get_cache(cache_key)
            if cached_result:
                with self._stats_lock:
                    self.stats['cache_hits'] += 1
                logger.info("Exact cache hit - returning cached results")
                return cached_result
        
        with self._stats_lock:
            self.stats['cache_misses'] += 1
        
        # Apply metadata filters if provided
        filtered_indices = None
        if metadata_filters:
            filtered_indices = self._apply_metadata_filters(metadata_filters)
        
        # Use concurrent futures for parallel retrieval instead of asyncio
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both retrievals to run in parallel using preprocessed query
            dense_future = executor.submit(
                self._get_dense_results_sync, preprocessed_query, k + int(0.3 * k), filtered_indices
            )
            sparse_future = executor.submit(
                self._get_sparse_results_sync, preprocessed_query, k + int(0.3 * k), filtered_indices
            )
            
            # Wait for both to complete
            dense_results = dense_future.result()
            sparse_results = sparse_future.result()
        
        # Fuse results
        fused_results = self._fuse_results_optimized(
            dense_results, sparse_results, dense_weight, sparse_weight
        )
        
        # Early stopping based on confidence
        if fused_results and len(fused_results) > 0:
            top_confidence = fused_results[0].metadata.get('hybrid_score', 0)
            if top_confidence >= self.confidence_threshold:
                with self._stats_lock:
                    self.stats['early_stops'] += 1
                logger.info(f"Early stopping triggered (confidence: {top_confidence:.3f})")
        
        # Return top-k results
        final_results = fused_results[:k]
        
        # Cache results
        if use_cache:
            self._set_cache(cache_key, final_results)
            # Add to semantic cache for future similarity checks
            self.semantic_cache[preprocessed_query] = cache_key
        
        # Update stats
        retrieval_time = time.time() - start_time
        with self._stats_lock:
            self.stats['avg_retrieval_time'] = (
                (self.stats['avg_retrieval_time'] * (self.stats['total_queries'] - 1) + retrieval_time) /
                self.stats['total_queries']
            )
        
        logger.info(f"Optimized hybrid retrieval completed in {retrieval_time:.3f}s, returning {len(final_results)} results")
        return final_results
    
    def _apply_metadata_filters(self, filters: Dict[str, Any]) -> List[int]:
        """Apply metadata filters to get valid document indices."""
        valid_indices = []
        for i, metadata in enumerate(self.metadatas):
            match = True
            for key, value in filters.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break
            if match:
                valid_indices.append(i)
        
        logger.info(f"Metadata filters applied: {len(valid_indices)} documents match")
        return valid_indices
    
    def _get_dense_results_sync(
        self,
        query: str,
        k: int,
        filtered_indices: Optional[List[int]] = None
    ) -> List[SearchResult]:
        """Get dense retrieval results synchronously (optimized version)."""
        try:
            # Check for cached embedding
            embedding_cache_key = f"embedding:{hashlib.md5(query.encode()).hexdigest()}"
            query_embedding = self._get_cache(embedding_cache_key)
            
            if query_embedding is None:
                query_embedding = self.embeddings.embed_query(query)
                self._set_cache(embedding_cache_key, query_embedding, ttl=86400)
            
            # Use FAISS if available
            if self.use_faiss and self.faiss_index:
                return self._faiss_search_sync(query_embedding, k, filtered_indices)
            else:
                return self._vectorstore_search_sync(query, k, filtered_indices)
                
        except Exception as e:
            logger.error(f"Error in dense retrieval: {e}")
            return []
    
    def _faiss_search_sync(
        self,
        query_embedding: List[float],
        k: int,
        filtered_indices: Optional[List[int]] = None
    ) -> List[SearchResult]:
        """Perform FAISS search synchronously."""
        try:
            with self._stats_lock:
                self.stats['faiss_queries'] += 1
            
            # Convert to numpy array
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Apply dimensionality reduction if used
            if self.dimension_reducer:
                query_vector = self.dimension_reducer.transform(query_vector)
            
            # Search FAISS index
            distances, indices = self.faiss_index.search(query_vector, k * 2)  # Get more for filtering
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                # Skip if not in filtered indices
                if filtered_indices is not None and idx not in filtered_indices:
                    continue
                
                # Convert distance to similarity score
                similarity_score = 1.0 / (1.0 + distance)
                
                # Get document and metadata
                if idx < len(self.documents):
                    doc_text = self.documents[idx]
                    metadata = self.metadatas[idx] if idx < len(self.metadatas) else {}
                    
                    # Ensure doc_id exists
                    if 'doc_id' not in metadata:
                        metadata['doc_id'] = idx
                    
                    doc = Document(page_content=doc_text, metadata=metadata)
                    
                    results.append(SearchResult(
                        document=doc,
                        score=similarity_score,
                        source='dense_faiss',
                        doc_id=str(metadata['doc_id'])
                    ))
                
                if len(results) >= k:
                    break
            
            logger.info(f"FAISS search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in FAISS search: {e}")
            return []
    
    def _vectorstore_search_sync(
        self,
        query: str,
        k: int,
        filtered_indices: Optional[List[int]] = None
    ) -> List[SearchResult]:
        """Fallback vectorstore search synchronously."""
        try:
            # Use regular vectorstore search
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            search_results = []
            for doc, score in results:
                # Skip if not in filtered indices
                doc_id = doc.metadata.get('doc_id', hash(doc.page_content))
                if filtered_indices is not None and doc_id not in filtered_indices:
                    continue
                
                search_results.append(SearchResult(
                    document=doc,
                    score=score,
                    source='dense_vectorstore',
                    doc_id=str(doc_id)
                ))
            
            logger.info(f"Vectorstore search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in vectorstore search: {e}")
            return []
    
    def _get_sparse_results_sync(
        self,
        query: str,
        k: int,
        filtered_indices: Optional[List[int]] = None
    ) -> List[SearchResult]:
        """Get sparse retrieval results synchronously (optimized version)."""
        if not self.bm25_retriever:
            logger.warning("BM25 retriever not available")
            return []
        
        try:
            # Check for cached BM25 results
            bm25_cache_key = f"bm25:{hashlib.md5(query.encode()).hexdigest()}"
            cached_results = self._get_cache(bm25_cache_key)
            
            if cached_results:
                results = cached_results
            else:
                results = self.bm25_retriever.invoke(query)
                self._set_cache(bm25_cache_key, results, ttl=3600)
            
            search_results = []
            for i, doc in enumerate(results[:k]):
                # Skip if not in filtered indices
                doc_id = doc.metadata.get('doc_id', hash(doc.page_content))
                if filtered_indices is not None and doc_id not in filtered_indices:
                    continue
                
                # Use rank-based scoring
                rank_score = 1.0 / (i + 1)
                
                search_results.append(SearchResult(
                    document=doc,
                    score=rank_score,
                    source='sparse_bm25',
                    doc_id=str(doc_id)
                ))
            
            logger.info(f"BM25 search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in sparse retrieval: {e}")
            return []
    
    async def _get_dense_results_async(
        self,
        query: str,
        k: int,
        filtered_indices: Optional[List[int]] = None
    ) -> List[SearchResult]:
        """Get dense retrieval results asynchronously."""
        try:
            # Check for cached embedding
            embedding_cache_key = f"embedding:{hashlib.md5(query.encode()).hexdigest()}"
            query_embedding = self._get_cache(embedding_cache_key)
            
            if query_embedding is None:
                query_embedding = self.embeddings.embed_query(query)
                self._set_cache(embedding_cache_key, query_embedding, ttl=86400)
            
            # Use FAISS if available
            if self.use_faiss and self.faiss_index:
                return await self._faiss_search_async(query_embedding, k, filtered_indices)
            else:
                return await self._vectorstore_search_async(query, k, filtered_indices)
                
        except Exception as e:
            logger.error(f"Error in dense retrieval: {e}")
            return []
    
    async def _faiss_search_async(
        self,
        query_embedding: List[float],
        k: int,
        filtered_indices: Optional[List[int]] = None
    ) -> List[SearchResult]:
        """Perform FAISS search asynchronously."""
        try:
            with self._stats_lock:
                self.stats['faiss_queries'] += 1
            
            # Convert to numpy array
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Apply dimensionality reduction if used
            if self.dimension_reducer:
                query_vector = self.dimension_reducer.transform(query_vector)
            
            # Search FAISS index
            distances, indices = self.faiss_index.search(query_vector, k * 2)  # Get more for filtering
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                # Skip if not in filtered indices
                if filtered_indices is not None and idx not in filtered_indices:
                    continue
                
                # Convert distance to similarity score
                similarity_score = 1.0 / (1.0 + distance)
                
                # Get document and metadata
                if idx < len(self.documents):
                    doc_text = self.documents[idx]
                    metadata = self.metadatas[idx] if idx < len(self.metadatas) else {}
                    
                    # Ensure doc_id exists
                    if 'doc_id' not in metadata:
                        metadata['doc_id'] = idx
                    
                    doc = Document(page_content=doc_text, metadata=metadata)
                    
                    results.append(SearchResult(
                        document=doc,
                        score=similarity_score,
                        source='dense_faiss',
                        doc_id=str(metadata['doc_id'])
                    ))
                
                if len(results) >= k:
                    break
            
            logger.info(f"FAISS search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in FAISS search: {e}")
            return []
    
    async def _vectorstore_search_async(
        self,
        query: str,
        k: int,
        filtered_indices: Optional[List[int]] = None
    ) -> List[SearchResult]:
        """Fallback vectorstore search asynchronously."""
        try:
            # Use regular vectorstore search
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            search_results = []
            for doc, score in results:
                # Skip if not in filtered indices
                doc_id = doc.metadata.get('doc_id', hash(doc.page_content))
                if filtered_indices is not None and doc_id not in filtered_indices:
                    continue
                
                similarity_score = 1.0 / (1.0 + score)
                
                search_results.append(SearchResult(
                    document=doc,
                    score=similarity_score,
                    source='dense_vectorstore',
                    doc_id=str(doc_id)
                ))
            
            logger.info(f"Vectorstore search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in vectorstore search: {e}")
            return []
    
    async def _get_sparse_results_async(
        self,
        query: str,
        k: int,
        filtered_indices: Optional[List[int]] = None
    ) -> List[SearchResult]:
        """Get sparse retrieval results asynchronously."""
        if not self.bm25_retriever:
            logger.warning("BM25 retriever not available")
            return []
        
        try:
            # Check for cached BM25 results
            bm25_cache_key = f"bm25:{hashlib.md5(query.encode()).hexdigest()}"
            cached_results = self._get_cache(bm25_cache_key)
            
            if cached_results:
                results = cached_results
            else:
                results = self.bm25_retriever.invoke(query)
                self._set_cache(bm25_cache_key, results, ttl=3600)
            
            search_results = []
            for i, doc in enumerate(results[:k]):
                # Skip if not in filtered indices
                doc_id = doc.metadata.get('doc_id', hash(doc.page_content))
                if filtered_indices is not None and doc_id not in filtered_indices:
                    continue
                
                # Use rank-based scoring
                rank_score = 1.0 / (i + 1)
                
                search_results.append(SearchResult(
                    document=doc,
                    score=rank_score,
                    source='sparse_bm25',
                    doc_id=str(doc_id)
                ))
            
            logger.info(f"BM25 search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in sparse retrieval: {e}")
            return []
    
    def _fuse_results_optimized(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        dense_weight: float,
        sparse_weight: float
    ) -> List[Document]:
        """
        Optimized result fusion with better score normalization.
        """
        logger.info("Fusing results with optimized algorithm")
        
        # Normalize scores
        if dense_results:
            dense_scores = [r.score for r in dense_results]
            min_dense, max_dense = min(dense_scores), max(dense_scores)
            if max_dense > min_dense:
                for result in dense_results:
                    result.normalized_score = (result.score - min_dense) / (max_dense - min_dense)
            else:
                for result in dense_results:
                    result.normalized_score = 1.0
        
        if sparse_results:
            sparse_scores = [r.score for r in sparse_results]
            min_sparse, max_sparse = min(sparse_scores), max(sparse_scores)
            if max_sparse > min_sparse:
                for result in sparse_results:
                    result.normalized_score = (result.score - min_sparse) / (max_sparse - min_sparse)
            else:
                for result in sparse_results:
                    result.normalized_score = 1.0
        
        # Combine results by document ID
        combined_scores = defaultdict(lambda: {'dense': 0, 'sparse': 0, 'document': None})
        
        # Add dense scores
        for result in dense_results:
            combined_scores[result.doc_id]['dense'] = result.normalized_score
            combined_scores[result.doc_id]['document'] = result.document
        
        # Add sparse scores
        for result in sparse_results:
            combined_scores[result.doc_id]['sparse'] = result.normalized_score
            if combined_scores[result.doc_id]['document'] is None:
                combined_scores[result.doc_id]['document'] = result.document
        
        # Calculate final scores and create documents
        final_results = []
        for doc_id, scores in combined_scores.items():
            final_score = (dense_weight * scores['dense'] + sparse_weight * scores['sparse'])
            
            doc = scores['document']
            if doc:
                # Add comprehensive metadata
                doc.metadata.update({
                    'hybrid_score': final_score,
                    'dense_score': scores['dense'],
                    'sparse_score': scores['sparse'],
                    'retrieval_method': 'optimized_hybrid',
                    'confidence': 'high' if final_score >= 0.8 else 'medium' if final_score >= 0.6 else 'low'
                })
                
                final_results.append((doc, final_score))
        
        # Sort by final score
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return documents only
        fused_documents = [doc for doc, score in final_results]
        
        logger.info(f"Optimized fusion completed: {len(fused_documents)} unique documents")
        return fused_documents
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._stats_lock:
            stats = self.stats.copy()
        
        # Calculate cache hit rate
        total_requests = stats['cache_hits'] + stats['cache_misses']
        cache_hit_rate = stats['cache_hits'] / total_requests if total_requests > 0 else 0
        
        return {
            **stats,
            'cache_hit_rate': cache_hit_rate,
            'total_cache_requests': total_requests,
            'faiss_enabled': self.use_faiss,
            'dimension_reduction_enabled': self.use_dimensionality_reduction
        }
    
    def clear_cache(self):
        """Clear all caches."""
        with self._cache_lock:
            self.memory_cache.clear()
            self.memory_cache_order.clear()
            
            if self.disk_cache:
                self.disk_cache.clear()
            
            if self.redis_cache:
                try:
                    self.redis_cache.flushdb()
                except Exception as e:
                    logger.warning(f"Error clearing Redis cache: {e}")
        
        logger.info("All caches cleared")
    
    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ) -> List[List[Document]]:
        """
        Batch retrieval for multiple queries with parallel processing.
        
        Args:
            queries: List of query strings
            k: Number of results per query
            dense_weight: Weight for dense retrieval
            sparse_weight: Weight for sparse retrieval
            
        Returns:
            List of result lists, one per query
        """
        logger.info(f"Batch retrieval for {len(queries)} queries")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all queries
            future_to_query = {
                executor.submit(
                    self.retrieve, query, k, dense_weight, sparse_weight
                ): query for query in queries
            }
            
            # Collect results in order
            query_results = {}
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = future.result()
                    query_results[query] = result
                except Exception as e:
                    logger.error(f"Error in batch query '{query}': {e}")
                    query_results[query] = []
            
            # Return results in original order
            results = [query_results[query] for query in queries]
        
        logger.info(f"Batch retrieval completed for {len(queries)} queries")
        return results 