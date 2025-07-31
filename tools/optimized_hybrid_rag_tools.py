"""
Optimized Hybrid RAG tools with advanced performance optimizations.
Integrates OptimizedHybridRetriever with caching, FAISS, and parallel processing.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import pandas as pd
import os

from retrieval.optimized_hybrid_retriever import OptimizedHybridRetriever, CacheConfig
from tools.company_tools import get_company_tools
from tools.patent_tools import get_patent_tools
from utils.data_mapping import create_mapping_manager

logger = get_logger(__name__)

@dataclass
class SearchConfig:
    """Configuration for optimized search."""
    use_faiss: bool = True
    use_cache: bool = True
    use_dimensionality_reduction: bool = False
    reduced_dimensions: int = 128
    confidence_threshold: float = 0.7  # Reduced for faster early stopping
    max_workers: int = 12  # Increased from 4 to 12 for better parallelization
    enable_metadata_filtering: bool = True
    early_stopping: bool = True
    batch_size: int = 50  # Added for batch processing
    memory_optimization: bool = True  # Added for memory management

class OptimizedHybridRAGTools:
    """
    Advanced RAG tools with multiple performance optimizations:
    - FAISS-based approximate nearest neighbor search
    - Multi-level caching (memory, disk, Redis)
    - Parallel processing for dense and sparse retrieval
    - Metadata-based filtering
    - Query preprocessing and optimization
    - Performance monitoring and analytics
    """
    
    def __init__(
        self,
        index_dir: str = "RAG_INDEX",
        search_config: Optional[SearchConfig] = None,
        cache_config: Optional[CacheConfig] = None
    ):
        self.index_dir = index_dir
        self.search_config = search_config or SearchConfig()
        self.cache_config = cache_config or CacheConfig()
        
        # Initialize components
        self.data_mapping_manager = None
        self.company_hybrid_retriever = None
        self.patent_hybrid_retriever = None
        
        # Performance tracking
        self.performance_stats = {
            'total_queries': 0,
            'company_queries': 0,
            'patent_queries': 0,
            'cache_hits': 0,
            'avg_response_time': 0.0,
            'parallel_queries': 0,
            'metadata_filtered_queries': 0
        }
        
        logger.info("OptimizedHybridRAGTools initialized with advanced features")
        
        # Add indexing status tracking
        self.indexing_status = {
            'company_faiss_indexed': False,
            'patent_faiss_indexed': False,
            'company_bm25_indexed': False,
            'patent_bm25_indexed': False,
            'total_company_docs': 0,
            'total_patent_docs': 0,
            'faiss_enabled': search_config.use_faiss if search_config else True
        }
    
    def initialize_all_components(self) -> bool:
        """Initialize all components with optimizations."""
        try:
            logger.info("Initializing all optimized components...")
            
            # Initialize data mapping
            success = self.initialize_data_mapping()
            if not success:
                logger.warning("Data mapping initialization failed")
            
            # Initialize optimized retrievers
            success = self.initialize_optimized_retrievers()
            if not success:
                logger.error("Failed to initialize optimized retrievers")
                return False
            
            logger.info("All optimized components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False
    
    def initialize_data_mapping(self) -> bool:
        """Initialize the data mapping manager."""
        try:
            self.data_mapping_manager = create_mapping_manager(self.index_dir)
            logger.info("Data mapping manager initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing data mapping: {e}")
            return False
    
    def initialize_optimized_retrievers(self) -> bool:
        """Initialize optimized hybrid retrievers with advanced features."""
        try:
            logger.info("Initializing optimized hybrid retrievers...")
            
            # Get tool instances and data
            company_tools = get_company_tools()
            patent_tools = get_patent_tools()
            
            # Initialize company retriever
            if company_tools:
                try:
                    # Ensure chunks are built first
                    logger.info("Ensuring company chunks are built...")
                    company_tools.firm_rag.build_chunks(force_reindex=False)
                    
                    company_collection = company_tools.vectorstore
                    company_docs_data = company_tools.get_all_documents_and_metadatas()
                    
                    if company_docs_data and company_docs_data.get('documents') and len(company_docs_data['documents']) > 0:
                        company_vectorstore_wrapper = self._create_vectorstore_wrapper(
                            company_collection, 
                            company_docs_data['documents'], 
                            company_docs_data['metadatas']
                        )
                        
                        self.company_hybrid_retriever = OptimizedHybridRetriever(
                            vectorstore=company_vectorstore_wrapper,
                            documents=company_docs_data['documents'],
                            metadatas=company_docs_data['metadatas'],
                            bm25_cache_path=os.path.join(self.index_dir, "company_bm25.pkl"),
                            cache_config=self.cache_config,
                            faiss_index_path=os.path.join(self.index_dir, "company_faiss.bin"),
                            use_faiss=self.search_config.use_faiss,
                            use_dimensionality_reduction=self.search_config.use_dimensionality_reduction,
                            reduced_dimensions=self.search_config.reduced_dimensions,
                            confidence_threshold=self.search_config.confidence_threshold,
                            max_workers=self.search_config.max_workers,
                            faiss_sample_ratio=1.0,  # Index ALL company documents for FAISS
                            is_patent_data=False  # Mark as company data
                        )
                        logger.info(f"Company optimized retriever initialized with {len(company_docs_data['documents'])} documents")
                        
                        # Update indexing status
                        self.indexing_status['company_faiss_indexed'] = self.search_config.use_faiss
                        self.indexing_status['company_bm25_indexed'] = True
                        self.indexing_status['total_company_docs'] = len(company_docs_data['documents'])
                    else:
                        logger.warning("No company documents found for optimized retriever")
                except Exception as e:
                    logger.error(f"Error initializing company optimized retriever: {e}")
            
            # Initialize patent retriever
            if patent_tools:
                try:
                    # Ensure chunks are built first
                    logger.info("Ensuring patent chunks are built...")
                    patent_tools.patent_rag.build_chunks(force_reindex=False)
                    
                    patent_collection = patent_tools.vectorstore
                    patent_docs_data = patent_tools.get_all_documents_and_metadatas()
                    
                    if patent_docs_data and patent_docs_data.get('documents') and len(patent_docs_data['documents']) > 0:
                        patent_vectorstore_wrapper = self._create_vectorstore_wrapper(
                            patent_collection, 
                            patent_docs_data['documents'], 
                            patent_docs_data['metadatas']
                        )
                        
                        self.patent_hybrid_retriever = OptimizedHybridRetriever(
                            vectorstore=patent_vectorstore_wrapper,
                            documents=patent_docs_data['documents'],
                            metadatas=patent_docs_data['metadatas'],
                            bm25_cache_path=os.path.join(self.index_dir, "patent_bm25.pkl"),
                            cache_config=self.cache_config,
                            faiss_index_path=os.path.join(self.index_dir, "patent_faiss.bin"),
                            use_faiss=self.search_config.use_faiss,
                            use_dimensionality_reduction=self.search_config.use_dimensionality_reduction,
                            reduced_dimensions=self.search_config.reduced_dimensions,
                            confidence_threshold=self.search_config.confidence_threshold,
                            max_workers=self.search_config.max_workers,
                            faiss_sample_ratio=0.01,  # Only index 1% of patents for FAISS (16k out of 1.6M)
                            is_patent_data=True  # Mark as patent data for special handling
                        )
                        logger.info(f"Patent optimized retriever initialized with {len(patent_docs_data['documents'])} documents")
                        
                        # Update indexing status
                        self.indexing_status['patent_faiss_indexed'] = self.search_config.use_faiss
                        self.indexing_status['patent_bm25_indexed'] = True
                        self.indexing_status['total_patent_docs'] = len(patent_docs_data['documents'])
                    else:
                        logger.warning("No patent documents found for optimized retriever")
                except Exception as e:
                    logger.error(f"Error initializing patent optimized retriever: {e}")
            
            if not self.company_hybrid_retriever and not self.patent_hybrid_retriever:
                logger.error("No optimized retrievers could be initialized - check if data ingestion was completed")
                return False
            
            logger.info("Optimized hybrid retrievers initialized successfully")
            
            # Log comprehensive indexing status
            self._log_indexing_status()
            return True
            
        except Exception as e:
            logger.error(f"Error initializing optimized retrievers: {e}")
            return False
    
    def _create_vectorstore_wrapper(self, collection, documents, metadatas):
        """Create optimized vectorstore wrapper."""
        class OptimizedChromaDBWrapper:
            def __init__(self, collection, documents, metadatas):
                self.collection = collection
                self.documents = documents
                self.metadatas = metadatas
            
            def similarity_search_with_score(self, query: str, k: int = 5):
                """Optimized similarity search with better error handling."""
                try:
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=min(k, len(self.documents)),  # Don't exceed available docs
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    documents_and_scores = []
                    if results["documents"] and results["documents"][0]:
                        for i, (doc_text, metadata, distance) in enumerate(zip(
                            results["documents"][0],
                            results["metadatas"][0] if results["metadatas"] else [{}] * len(results["documents"][0]),
                            results["distances"][0] if results["distances"] else [0] * len(results["documents"][0])
                        )):
                            # Enhanced metadata processing
                            if metadata is None:
                                metadata = {}
                            
                            # Ensure required fields
                            if 'doc_id' not in metadata:
                                if 'company_id' in metadata and 'chunk_index' in metadata:
                                    metadata['doc_id'] = f"company_{metadata['company_id']}_{metadata['chunk_index']}"
                                elif 'patent_id' in metadata and 'chunk_index' in metadata:
                                    metadata['doc_id'] = f"patent_{metadata['patent_id']}_{metadata['chunk_index']}"
                                else:
                                    metadata['doc_id'] = f"doc_{i}_{hash(doc_text) % 10000}"
                            
                            # Enhanced source naming
                            if 'source_name' not in metadata:
                                if 'company_name' in metadata:
                                    metadata['source_name'] = metadata['company_name']
                                elif 'patent_id' in metadata:
                                    metadata['source_name'] = f"Patent {metadata['patent_id']}"
                                else:
                                    metadata['source_name'] = "Unknown Source"
                            
                            # Add performance metadata
                            metadata['search_timestamp'] = time.time()
                            metadata['search_method'] = 'optimized_chroma'
                            
                            doc = type('Document', (), {
                                'page_content': doc_text,
                                'metadata': metadata
                            })()
                            
                            documents_and_scores.append((doc, distance))
                    
                    return documents_and_scores
                    
                except Exception as e:
                    logger.error(f"Error in optimized ChromaDB wrapper: {e}")
                    return []
        
        return OptimizedChromaDBWrapper(collection, documents, metadatas)
    
    async def optimized_hybrid_search_async(
        self,
        query: str,
        top_k: int = 3,
        search_type: str = "both",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Advanced asynchronous hybrid search with all optimizations.
        
        Args:
            query: Search query
            top_k: Number of results per type
            search_type: "company", "patent", or "both"
            dense_weight: Weight for dense retrieval
            sparse_weight: Weight for sparse retrieval
            metadata_filters: Optional metadata filters
            use_cache: Whether to use caching
            
        Returns:
            Optimized search results
        """
        start_time = time.time()
        self.performance_stats['total_queries'] += 1
        
        logger.info(f"Optimized async hybrid search: {query} (type: {search_type}, top_k: {top_k})")
        
        # Initialize if needed
        if not self.company_hybrid_retriever or not self.patent_hybrid_retriever:
            await asyncio.get_event_loop().run_in_executor(None, self.initialize_optimized_retrievers)
        
        result = {
            "company_contexts": [],
            "patent_contexts": [],
            "optimization_used": True,
            "search_metadata": {
                "query": query,
                "search_type": search_type,
                "top_k": top_k,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "metadata_filters": metadata_filters,
                "retrieval_method": "optimized_async_hybrid",
                "timestamp": time.time()
            },
            "performance_info": {},
            "success": True
        }
        
        # Track metadata filtering
        if metadata_filters and self.search_config.enable_metadata_filtering:
            self.performance_stats['metadata_filtered_queries'] += 1
        
        # Parallel search tasks
        tasks = []
        
        if search_type in ["company", "both"] and self.company_hybrid_retriever:
            self.performance_stats['company_queries'] += 1
            tasks.append(self._search_company_async(
                query, top_k, dense_weight, sparse_weight, metadata_filters, use_cache
            ))
        
        if search_type in ["patent", "both"] and self.patent_hybrid_retriever:
            self.performance_stats['patent_queries'] += 1
            tasks.append(self._search_patent_async(
                query, top_k, dense_weight, sparse_weight, metadata_filters, use_cache
            ))
        
        # Execute searches in parallel
        if tasks:
            self.performance_stats['parallel_queries'] += 1
            search_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, search_result in enumerate(search_results):
                if isinstance(search_result, Exception):
                    logger.error(f"Search task {i} failed: {search_result}")
                    result["success"] = False
                elif isinstance(search_result, dict):
                    if "company_contexts" in search_result:
                        result["company_contexts"] = search_result["company_contexts"]
                    if "patent_contexts" in search_result:
                        result["patent_contexts"] = search_result["patent_contexts"]
        
        # Add performance information
        response_time = time.time() - start_time
        self.performance_stats['avg_response_time'] = (
            (self.performance_stats['avg_response_time'] * (self.performance_stats['total_queries'] - 1) + response_time) /
            self.performance_stats['total_queries']
        )
        
        result["performance_info"] = {
            "response_time": response_time,
            "cache_used": use_cache,
            "metadata_filtered": metadata_filters is not None,
            "parallel_execution": len(tasks) > 1
        }
        
        # Add retriever performance stats
        if self.company_hybrid_retriever:
            result["performance_info"]["company_retriever_stats"] = self.company_hybrid_retriever.get_performance_stats()
        if self.patent_hybrid_retriever:
            result["performance_info"]["patent_retriever_stats"] = self.patent_hybrid_retriever.get_performance_stats()
        
        logger.info(f"Optimized search completed in {response_time:.3f}s")
        return result
    
    async def _search_company_async(
        self,
        query: str,
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
        metadata_filters: Optional[Dict[str, Any]],
        use_cache: bool
    ) -> Dict[str, Any]:
        """Asynchronous company search."""
        try:
            # Apply company-specific filters
            company_filters = metadata_filters.copy() if metadata_filters else {}
            if 'source_type' not in company_filters:
                company_filters['source_type'] = 'company'
            
            docs = await self.company_hybrid_retriever.retrieve_async(
                query=query,
                k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                use_cache=use_cache,
                metadata_filters=company_filters if self.search_config.enable_metadata_filtering else None
            )
            
            company_contexts = []
            for i, doc in enumerate(docs):
                company_contexts.append({
                    'company_name': doc.metadata.get('company_name', 'Unknown Company'),
                    'company_id': doc.metadata.get('company_id', 'unknown'),
                    'hojin_id': doc.metadata.get('hojin_id', doc.metadata.get('company_id', 'unknown')),
                    'chunk': doc.page_content,
                    'score': doc.metadata.get('hybrid_score', 0),
                    'confidence': doc.metadata.get('confidence', 'unknown'),
                    'retrieval_method': 'optimized_hybrid',
                    'rank': i + 1,
                    'chunk_index': doc.metadata.get('chunk_index', 0),
                    'doc_id': doc.metadata.get('doc_id', f"company_{i}"),
                    'search_timestamp': doc.metadata.get('search_timestamp', time.time())
                })
            
            logger.info(f"Optimized company search returned {len(company_contexts)} results")
            return {"company_contexts": company_contexts}
            
        except Exception as e:
            logger.error(f"Error in optimized company search: {e}")
            return {"company_contexts": [], "error": str(e)}
    
    async def _search_patent_async(
        self,
        query: str,
        top_k: int,
        dense_weight: float,
        sparse_weight: float,
        metadata_filters: Optional[Dict[str, Any]],
        use_cache: bool
    ) -> Dict[str, Any]:
        """Asynchronous patent search."""
        try:
            # Apply patent-specific filters
            patent_filters = metadata_filters.copy() if metadata_filters else {}
            if 'source_type' not in patent_filters:
                patent_filters['source_type'] = 'patent'
            
            docs = await self.patent_hybrid_retriever.retrieve_async(
                query=query,
                k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                use_cache=use_cache,
                metadata_filters=patent_filters if self.search_config.enable_metadata_filtering else None
            )
            
            patent_contexts = []
            for i, doc in enumerate(docs):
                patent_contexts.append({
                    'patent_id': doc.metadata.get('patent_id', 'unknown'),
                    'company_name': doc.metadata.get('company_name', 'Unknown Company'),
                    'company_id': doc.metadata.get('company_id', doc.metadata.get('hojin_id', 'unknown')),
                    'chunk': doc.page_content,
                    'score': doc.metadata.get('hybrid_score', 0),
                    'confidence': doc.metadata.get('confidence', 'unknown'),
                    'retrieval_method': 'optimized_hybrid',
                    'rank': i + 1,
                    'chunk_index': doc.metadata.get('chunk_index', 0),
                    'doc_id': doc.metadata.get('doc_id', f"patent_{i}"),
                    'search_timestamp': doc.metadata.get('search_timestamp', time.time())
                })
            
            logger.info(f"Optimized patent search returned {len(patent_contexts)} results")
            return {"patent_contexts": patent_contexts}
            
        except Exception as e:
            logger.error(f"Error in optimized patent search: {e}")
            return {"patent_contexts": [], "error": str(e)}
    
    def optimized_hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        search_type: str = "both",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Synchronous wrapper for async optimized search."""
        return asyncio.run(self.optimized_hybrid_search_async(
            query, top_k, search_type, dense_weight, sparse_weight, metadata_filters, use_cache
        ))
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 3,
        search_type: str = "both",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Optimized batch search for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            search_type: "company", "patent", or "both"
            dense_weight: Weight for dense retrieval
            sparse_weight: Weight for sparse retrieval
            
        Returns:
            List of search results, one per query
        """
        logger.info(f"Optimized batch search for {len(queries)} queries")
        start_time = time.time()
        
        # Use company retriever for batch processing if available
        if search_type == "company" and self.company_hybrid_retriever:
            results = self.company_hybrid_retriever.batch_retrieve(
                queries, top_k, dense_weight, sparse_weight
            )
            # Convert to expected format
            formatted_results = []
            for i, docs in enumerate(results):
                formatted_results.append({
                    "company_contexts": [
                        {
                            'company_name': doc.metadata.get('company_name', 'Unknown'),
                            'chunk': doc.page_content,
                            'score': doc.metadata.get('hybrid_score', 0),
                            'rank': j + 1
                        } for j, doc in enumerate(docs)
                    ],
                    "patent_contexts": [],
                    "query": queries[i],
                    "optimization_used": True
                })
            return formatted_results
        
        # Use patent retriever for batch processing if available
        elif search_type == "patent" and self.patent_hybrid_retriever:
            results = self.patent_hybrid_retriever.batch_retrieve(
                queries, top_k, dense_weight, sparse_weight
            )
            # Convert to expected format
            formatted_results = []
            for i, docs in enumerate(results):
                formatted_results.append({
                    "company_contexts": [],
                    "patent_contexts": [
                        {
                            'patent_id': doc.metadata.get('patent_id', 'unknown'),
                            'chunk': doc.page_content,
                            'score': doc.metadata.get('hybrid_score', 0),
                            'rank': j + 1
                        } for j, doc in enumerate(docs)
                    ],
                    "query": queries[i],
                    "optimization_used": True
                })
            return formatted_results
        
        # Fallback to individual searches with parallel processing
        else:
            results = []
            with ThreadPoolExecutor(max_workers=self.search_config.max_workers) as executor:
                future_to_query = {
                    executor.submit(
                        self.optimized_hybrid_search,
                        query, top_k, search_type, dense_weight, sparse_weight
                    ): query for query in queries
                }
                
                query_results = {}
                for future in as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        result = future.result()
                        query_results[query] = result
                    except Exception as e:
                        logger.error(f"Error in batch query '{query}': {e}")
                        query_results[query] = {
                            "company_contexts": [],
                            "patent_contexts": [],
                            "error": str(e)
                        }
                
                # Return results in original order
                results = [query_results[query] for query in queries]
        
        batch_time = time.time() - start_time
        logger.info(f"Batch search completed in {batch_time:.3f}s for {len(queries)} queries")
        
        return results

    async def async_batch_search(
        self,
        queries: List[str],
        top_k: int = 5,
        search_type: str = "both",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Asynchronous batch search with improved parallelization and memory management.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            search_type: "company", "patent", or "both"
            dense_weight: Weight for dense search
            sparse_weight: Weight for sparse search
            use_cache: Whether to use caching
            
        Returns:
            List of search results for each query
        """
        if not queries:
            return []
        
        start_time = time.time()
        logger.info(f"Starting async batch search for {len(queries)} queries")
        
        # Process in batches to manage memory usage
        batch_size = self.search_config.batch_size
        all_results = []
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(queries) + batch_size - 1)//batch_size}")
            
            # Create async tasks for this batch
            tasks = []
            for query in batch_queries:
                if search_type in ["company", "both"] and self.company_hybrid_retriever:
                    tasks.append(self._async_search_single(
                        query, top_k, "company", dense_weight, sparse_weight, use_cache
                    ))
                elif search_type in ["patent", "both"] and self.patent_hybrid_retriever:
                    tasks.append(self._async_search_single(
                        query, top_k, "patent", dense_weight, sparse_weight, use_cache
                    ))
                else:
                    # Fallback to hybrid search
                    tasks.append(self._async_search_single(
                        query, top_k, "both", dense_weight, sparse_weight, use_cache
                    ))
            
            # Execute batch tasks concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process batch results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error in batch query '{batch_queries[j]}': {result}")
                    all_results.append({
                        "company_contexts": [],
                        "patent_contexts": [],
                        "query": batch_queries[j],
                        "error": str(result),
                        "optimization_used": True
                    })
                else:
                    result["query"] = batch_queries[j]
                    result["optimization_used"] = True
                    all_results.append(result)
            
            # Memory cleanup between batches
            if self.search_config.memory_optimization:
                import gc
                gc.collect()
        
        batch_time = time.time() - start_time
        logger.info(f"Async batch search completed in {batch_time:.3f}s for {len(queries)} queries")
        
        return all_results

    async def _async_search_single(
        self,
        query: str,
        top_k: int,
        search_type: str,
        dense_weight: float,
        sparse_weight: float,
        use_cache: bool
    ) -> Dict[str, Any]:
        """
        Single async search operation.
        
        Returns:
            Search result dictionary
        """
        try:
            if search_type == "company" and self.company_hybrid_retriever:
                docs = await self.company_hybrid_retriever.retrieve_async(
                    query=query,
                    k=top_k,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    use_cache=use_cache
                )
                return {
                    "company_contexts": [
                        {
                            'company_name': doc.metadata.get('company_name', 'Unknown'),
                            'chunk': doc.page_content,
                            'score': doc.metadata.get('hybrid_score', 0),
                            'rank': j + 1,
                            'confidence': doc.metadata.get('confidence', 'unknown')
                        } for j, doc in enumerate(docs)
                    ],
                    "patent_contexts": []
                }
            
            elif search_type == "patent" and self.patent_hybrid_retriever:
                docs = await self.patent_hybrid_retriever.retrieve_async(
                    query=query,
                    k=top_k,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    use_cache=use_cache
                )
                return {
                    "company_contexts": [],
                    "patent_contexts": [
                        {
                            'patent_id': doc.metadata.get('patent_id', 'unknown'),
                            'chunk': doc.page_content,
                            'score': doc.metadata.get('hybrid_score', 0),
                            'rank': j + 1,
                            'confidence': doc.metadata.get('confidence', 'unknown')
                        } for j, doc in enumerate(docs)
                    ]
                }
            
            else:  # Both or fallback
                # Run both searches in parallel
                tasks = []
                if self.company_hybrid_retriever:
                    tasks.append(self.company_hybrid_retriever.retrieve_async(
                        query=query, k=top_k, dense_weight=dense_weight,
                        sparse_weight=sparse_weight, use_cache=use_cache
                    ))
                if self.patent_hybrid_retriever:
                    tasks.append(self.patent_hybrid_retriever.retrieve_async(
                        query=query, k=top_k, dense_weight=dense_weight,
                        sparse_weight=sparse_weight, use_cache=use_cache
                    ))
                
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    company_contexts = []
                    patent_contexts = []
                    
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            logger.warning(f"Search task {i} failed: {result}")
                            continue
                        
                        if i == 0 and self.company_hybrid_retriever:  # Company results
                            company_contexts = [
                                {
                                    'company_name': doc.metadata.get('company_name', 'Unknown'),
                                    'chunk': doc.page_content,
                                    'score': doc.metadata.get('hybrid_score', 0),
                                    'rank': j + 1,
                                    'confidence': doc.metadata.get('confidence', 'unknown')
                                } for j, doc in enumerate(result)
                            ]
                        elif self.patent_hybrid_retriever:  # Patent results
                            patent_contexts = [
                                {
                                    'patent_id': doc.metadata.get('patent_id', 'unknown'),
                                    'chunk': doc.page_content,
                                    'score': doc.metadata.get('hybrid_score', 0),
                                    'rank': j + 1,
                                    'confidence': doc.metadata.get('confidence', 'unknown')
                                } for j, doc in enumerate(result)
                            ]
                    
                    return {
                        "company_contexts": company_contexts,
                        "patent_contexts": patent_contexts
                    }
                
                # Fallback if no retrievers available
                return {
                    "company_contexts": [],
                    "patent_contexts": [],
                    "error": "No retrievers available"
                }
                
        except Exception as e:
            logger.error(f"Error in async single search for '{query}': {e}")
            return {
                "company_contexts": [],
                "patent_contexts": [],
                "error": str(e)
            }
    
    def get_comprehensive_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "tool_stats": self.performance_stats.copy(),
            "company_retriever_stats": None,
            "patent_retriever_stats": None,
            "optimization_features": {
                "faiss_enabled": self.search_config.use_faiss,
                "caching_enabled": self.search_config.use_cache,
                "dimensionality_reduction": self.search_config.use_dimensionality_reduction,
                "metadata_filtering": self.search_config.enable_metadata_filtering,
                "early_stopping": self.search_config.early_stopping,
                "parallel_processing": True
            }
        }
        
        if self.company_hybrid_retriever:
            stats["company_retriever_stats"] = self.company_hybrid_retriever.get_performance_stats()
        
        if self.patent_hybrid_retriever:
            stats["patent_retriever_stats"] = self.patent_hybrid_retriever.get_performance_stats()
        
        return stats
    
    def clear_all_caches(self):
        """Clear all caches across all retrievers."""
        if self.company_hybrid_retriever:
            self.company_hybrid_retriever.clear_cache()
        
        if self.patent_hybrid_retriever:
            self.patent_hybrid_retriever.clear_cache()
        
        logger.info("All caches cleared across all retrievers")
    
    def optimize_for_query_pattern(self, common_queries: List[str]):
        """Optimize system for common query patterns."""
        logger.info(f"Optimizing for {len(common_queries)} common query patterns")
        
        # Pre-warm caches with common queries
        for query in common_queries:
            try:
                self.optimized_hybrid_search(query, top_k=3, use_cache=True)
                logger.info(f"Pre-warmed cache for query: {query[:50]}...")
            except Exception as e:
                logger.warning(f"Error pre-warming query '{query[:50]}...': {e}")
        
        logger.info("Query pattern optimization completed")
    
    def _log_indexing_status(self):
        """Log comprehensive indexing status."""
        logger.info("\n" + "="*60)
        logger.info("INDEXING STATUS REPORT")
        logger.info("="*60)
        
        # FAISS Status
        logger.info(f"FAISS Indexing: {'ENABLED' if self.indexing_status['faiss_enabled'] else 'DISABLED'}")
        if self.indexing_status['faiss_enabled']:
            logger.info(f"  Company FAISS: {'INDEXED' if self.indexing_status['company_faiss_indexed'] else 'NOT INDEXED'}")
            logger.info(f"  Patent FAISS:  {'INDEXED' if self.indexing_status['patent_faiss_indexed'] else 'NOT INDEXED'}")
        
        # BM25 Status
        logger.info(f"\nBM25 Indexing:")
        logger.info(f"  Company BM25: {'INDEXED' if self.indexing_status['company_bm25_indexed'] else 'NOT INDEXED'}")
        logger.info(f"  Patent BM25:  {'INDEXED' if self.indexing_status['patent_bm25_indexed'] else 'NOT INDEXED'}")
        
        # Document Counts
        logger.info(f"\nDocument Counts:")
        logger.info(f"  Company Documents: {self.indexing_status['total_company_docs']:,}")
        logger.info(f"  Patent Documents:  {self.indexing_status['total_patent_docs']:,}")
        logger.info(f"  Total Documents:   {self.indexing_status['total_company_docs'] + self.indexing_status['total_patent_docs']:,}")
        
        # Performance Estimates with sampling info
        total_docs = self.indexing_status['total_company_docs'] + self.indexing_status['total_patent_docs']
        if self.indexing_status['faiss_enabled'] and total_docs > 0:
            est_speed_improvement = min(total_docs / 1000, 30)  # Cap at 30x
            logger.info(f"\nPerformance Estimates:")
            logger.info(f"  Estimated Speed Improvement: {est_speed_improvement:.1f}x faster with FAISS")
            logger.info(f"  Expected Query Time: {200/est_speed_improvement:.0f}ms (vs {200}ms without FAISS)")
            
            # Show sampling info for patents
            patent_docs = self.indexing_status['total_patent_docs']
            if patent_docs > 100000:  # Large patent dataset
                sampled_patents = int(patent_docs * 0.01)  # 1% sampling
                time_saved_hours = (patent_docs - sampled_patents) * 3 / 100 / 3600
                logger.info(f"\nPatent Sampling Optimization:")
                logger.info(f"  Total Patents: {patent_docs:,}")
                logger.info(f"  FAISS Indexed: {sampled_patents:,} (1% sample)")
                logger.info(f"  Time Saved: ~{time_saved_hours:.1f} hours vs full indexing")
                logger.info(f"  BM25 Still Uses: ALL {patent_docs:,} patents (full keyword search)")
        
        logger.info("="*60 + "\n")
    
    def get_indexing_status(self) -> Dict[str, Any]:
        """Get current indexing status."""
        return self.indexing_status.copy()
    
    def force_reindex(self, index_type: str = "all"):
        """Force reindexing of FAISS indexes.
        
        Args:
            index_type: "company", "patent", or "all"
        """
        logger.info(f"Force reindexing: {index_type}")
        
        if index_type in ["company", "all"] and self.company_hybrid_retriever:
            logger.info("Clearing company FAISS index for rebuild...")
            if hasattr(self.company_hybrid_retriever, 'faiss_index_path'):
                import os
                if os.path.exists(self.company_hybrid_retriever.faiss_index_path):
                    os.remove(self.company_hybrid_retriever.faiss_index_path)
            self.company_hybrid_retriever._init_faiss_system()
            self.indexing_status['company_faiss_indexed'] = True
            
        if index_type in ["patent", "all"] and self.patent_hybrid_retriever:
            logger.info("Clearing patent FAISS index for rebuild...")
            if hasattr(self.patent_hybrid_retriever, 'faiss_index_path'):
                import os
                if os.path.exists(self.patent_hybrid_retriever.faiss_index_path):
                    os.remove(self.patent_hybrid_retriever.faiss_index_path)
            self.patent_hybrid_retriever._init_faiss_system()
            self.indexing_status['patent_faiss_indexed'] = True
        
        self._log_indexing_status()
        logger.info("Force FAISS reindexing completed")
    
    def force_bm25_reindex(self, index_type: str = "all"):
        """Force reindexing of BM25 indexes.
        
        Args:
            index_type: "company", "patent", or "all"
        """
        logger.info(f"Force BM25 reindexing: {index_type}")
        
        if index_type in ["company", "all"] and self.company_hybrid_retriever:
            logger.info("Clearing company BM25 index for rebuild...")
            if hasattr(self.company_hybrid_retriever, 'bm25_cache_path'):
                import os
                if os.path.exists(self.company_hybrid_retriever.bm25_cache_path):
                    os.remove(self.company_hybrid_retriever.bm25_cache_path)
                    logger.info(f"Removed: {self.company_hybrid_retriever.bm25_cache_path}")
            self.company_hybrid_retriever._init_bm25_system()
            self.indexing_status['company_bm25_indexed'] = True
            
        if index_type in ["patent", "all"] and self.patent_hybrid_retriever:
            logger.info("Clearing patent BM25 index for rebuild...")
            if hasattr(self.patent_hybrid_retriever, 'bm25_cache_path'):
                import os
                if os.path.exists(self.patent_hybrid_retriever.bm25_cache_path):
                    os.remove(self.patent_hybrid_retriever.bm25_cache_path)
                    logger.info(f"Removed: {self.patent_hybrid_retriever.bm25_cache_path}")
            self.patent_hybrid_retriever._init_bm25_system()
            self.indexing_status['patent_bm25_indexed'] = True
        
        self._log_indexing_status()
        logger.info("Force BM25 reindexing completed")
    
    def force_all_indexes_reindex(self, index_type: str = "all"):
        """Force reindexing of ALL indexes (BM25 + FAISS) without touching ChromaDB.
        
        Args:
            index_type: "company", "patent", or "all"
        """
        logger.info(f"\nREBUILDING ALL INDEXES: {index_type.upper()}")
        logger.info("="*60)
        logger.info("This will rebuild BM25 + FAISS indexes without touching ChromaDB data")
        logger.info("="*60)
        
        # Step 1: Rebuild BM25 indexes
        logger.info("STEP 1: Rebuilding BM25 indexes...")
        self.force_bm25_reindex(index_type)
        
        # Step 2: Rebuild FAISS indexes  
        logger.info("STEP 2: Rebuilding FAISS indexes...")
        self.force_reindex(index_type)  # This is the FAISS reindex method
        
        logger.info("\nALL INDEXES REBUILT SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("Your system is now fully optimized and ready for fast searches!")
        logger.info("="*60)

# Global instance management
_optimized_hybrid_tools = None

def get_optimized_hybrid_tools(
    index_dir: str = "RAG_INDEX",
    search_config: Optional[SearchConfig] = None,
    cache_config: Optional[CacheConfig] = None
) -> OptimizedHybridRAGTools:
    """Get the global optimized hybrid tools instance."""
    global _optimized_hybrid_tools
    if _optimized_hybrid_tools is None:
        logger.info("Creating new optimized hybrid tools instance")
        _optimized_hybrid_tools = OptimizedHybridRAGTools(index_dir, search_config, cache_config)
        
        # Initialize all components
        init_success = _optimized_hybrid_tools.initialize_all_components()
        if not init_success:
            logger.warning("Some components failed to initialize")
        else:
            logger.info("Optimized hybrid tools instance created successfully")
    
    return _optimized_hybrid_tools

def reset_optimized_hybrid_tools():
    """Reset the global optimized hybrid tools instance."""
    global _optimized_hybrid_tools
    _optimized_hybrid_tools = None
    logger.info("Optimized hybrid tools instance reset")

# Tool functions for external use
def optimized_hybrid_rag_retrieval_tool(
    query: str,
    top_k: int = 3,
    search_type: str = "both",
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5
) -> Dict[str, Any]:
    """
    Optimized hybrid RAG retrieval tool with advanced performance features.
    
    Args:
        query: Search query
        top_k: Number of results per type
        search_type: "company", "patent", or "both"
        dense_weight: Weight for dense retrieval
        sparse_weight: Weight for sparse retrieval
        
    Returns:
        Optimized search results
    """
    logger.info(f"Optimized hybrid RAG tool called: query='{query}', top_k={top_k}, type='{search_type}'")
    
    tools = get_optimized_hybrid_tools()
    result = tools.optimized_hybrid_search(
        query, top_k, search_type, dense_weight, sparse_weight
    )
    
    # Add debug information
    result["debug_info"] = {
        "optimized_retrievers_used": True,
        "company_retriever_initialized": tools.company_hybrid_retriever is not None,
        "patent_retriever_initialized": tools.patent_hybrid_retriever is not None,
        "performance_features_enabled": tools.search_config.__dict__
    }
    
    return result

def batch_optimized_retrieval_tool(
    queries: List[str],
    top_k: int = 3,
    search_type: str = "both"
) -> List[Dict[str, Any]]:
    """
    Batch optimized retrieval tool for multiple queries.
    
    Args:
        queries: List of search queries
        top_k: Number of results per query
        search_type: "company", "patent", or "both"
        
    Returns:
        List of search results
    """
    logger.info(f"Batch optimized retrieval called for {len(queries)} queries")
    
    tools = get_optimized_hybrid_tools()
    return tools.batch_search(queries, top_k, search_type)

def get_performance_analytics_tool(*args, **kwargs) -> Dict[str, Any]:
    """
    Get comprehensive performance analytics.
    
    Args:
        *args: Optional arguments (ignored for compatibility)
        **kwargs: Optional keyword arguments (ignored for compatibility)
    
    Returns:
        Performance statistics and analytics
    """
    tools = get_optimized_hybrid_tools()
    return tools.get_comprehensive_performance_stats()

def clear_caches_tool(*args, **kwargs):
    """
    Clear all caches across the system.
    
    Args:
        *args: Optional arguments (ignored for compatibility)
        **kwargs: Optional keyword arguments (ignored for compatibility)
    """
    tools = get_optimized_hybrid_tools()
    tools.clear_all_caches()
    return {"message": "All caches cleared successfully"}

def optimize_for_queries_tool(common_queries: List[str]):
    """Optimize system for common query patterns."""
    tools = get_optimized_hybrid_tools()
    tools.optimize_for_query_pattern(common_queries)
    return {"message": f"System optimized for {len(common_queries)} query patterns"} 