"""
Configuration for optimization features in the InnovARAG system.
Centralized settings for caching, FAISS, parallel processing, and other performance optimizations.
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class CacheOptimizationConfig:
    """Configuration for caching optimizations."""
    # Memory cache settings
    use_memory_cache: bool = True
    memory_cache_size: int = 1000
    
    # Disk cache settings
    use_disk_cache: bool = True
    disk_cache_dir: str = "cache/retrieval"
    disk_cache_size_mb: int = 1024  # 1GB
    
    # Redis cache settings
    use_redis: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Cache TTL settings (in seconds)
    query_cache_ttl: int = 3600  # 1 hour
    embedding_cache_ttl: int = 86400  # 24 hours
    bm25_cache_ttl: int = 3600  # 1 hour
    
    # Cache warming
    enable_cache_warming: bool = True
    common_queries_for_warming: List[str] = None
    
    def __post_init__(self):
        if self.common_queries_for_warming is None:
            self.common_queries_for_warming = [
                "machine learning", "artificial intelligence", "deep learning",
                "neural networks", "computer vision", "natural language processing",
                "data science", "algorithm", "innovation", "technology",
                "patent", "invention", "research", "development", "analysis"
            ]

@dataclass
class FAISSOptimizationConfig:
    """Configuration for FAISS-based optimizations."""
    # FAISS usage
    use_faiss: bool = True
    faiss_index_type: str = "HNSW"  # Options: "HNSW", "IVF", "Flat"
    
    # HNSW parameters
    hnsw_m: int = 32  # Number of connections per node
    hnsw_ef_construction: int = 200  # Size of dynamic candidate list
    hnsw_ef_search: int = 50  # Size of dynamic candidate list during search
    
    # IVF parameters (if using IVF)
    ivf_nlist: int = 100  # Number of clusters
    ivf_nprobe: int = 10  # Number of clusters to search
    
    # Index file paths
    company_faiss_index_path: str = "RAG_INDEX/company_faiss.bin"
    patent_faiss_index_path: str = "RAG_INDEX/patent_faiss.bin"
    
    # Dimensionality reduction
    use_dimensionality_reduction: bool = False
    target_dimensions: int = 128
    reduction_method: str = "PCA"  # Options: "PCA", "UMAP", "t-SNE"
    
    # Index building
    force_rebuild_index: bool = False
    batch_size_for_building: int = 1000

@dataclass
class ParallelProcessingConfig:
    """Configuration for parallel processing optimizations."""
    # Thread pool settings
    max_workers: int = 4
    use_async_processing: bool = True
    
    # Batch processing
    enable_batch_processing: bool = True
    batch_size: int = 10
    batch_timeout_seconds: int = 30
    
    # Parallel retrieval
    parallel_dense_sparse: bool = True
    parallel_company_patent: bool = True
    
    # Resource management
    max_memory_usage_gb: float = 8.0
    enable_memory_monitoring: bool = True

@dataclass
class SearchOptimizationConfig:
    """Configuration for search optimizations."""
    # Early stopping
    enable_early_stopping: bool = True
    confidence_threshold: float = 0.9
    min_score_threshold: float = 0.1
    
    # Metadata filtering
    enable_metadata_filtering: bool = True
    prefilter_by_source_type: bool = True
    
    # Query preprocessing
    enable_query_preprocessing: bool = True
    remove_stop_words: bool = True
    enable_query_expansion: bool = False
    
    # Score fusion optimization
    adaptive_weight_adjustment: bool = False
    use_rank_fusion: bool = False
    
    # Result optimization
    enable_result_deduplication: bool = True
    similarity_threshold_for_dedup: float = 0.95

@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring."""
    # Performance tracking
    enable_performance_tracking: bool = True
    track_query_patterns: bool = True
    track_cache_performance: bool = True
    
    # Logging
    log_slow_queries: bool = True
    slow_query_threshold_seconds: float = 2.0
    
    # Metrics export
    enable_metrics_export: bool = False
    metrics_export_interval_seconds: int = 60
    
    # Analytics
    enable_query_analytics: bool = True
    store_query_history: bool = True
    max_query_history_size: int = 10000

@dataclass
class OptimizationConfig:
    """Master configuration for all optimizations."""
    # Sub-configurations
    cache: CacheOptimizationConfig
    faiss: FAISSOptimizationConfig
    parallel: ParallelProcessingConfig
    search: SearchOptimizationConfig
    monitoring: MonitoringConfig
    
    # Global settings
    optimization_level: str = "high"  # Options: "low", "medium", "high", "maximum"
    enable_all_optimizations: bool = True
    
    # Data size considerations
    company_data_size: int = 6000
    patent_data_size: int = 1600000
    
    # Resource constraints
    available_memory_gb: float = 16.0
    available_cpu_cores: int = 8
    use_gpu: bool = True
    
    def __post_init__(self):
        """Adjust settings based on optimization level and data size."""
        if self.optimization_level == "maximum":
            self._apply_maximum_optimizations()
        elif self.optimization_level == "high":
            self._apply_high_optimizations()
        elif self.optimization_level == "medium":
            self._apply_medium_optimizations()
        elif self.optimization_level == "low":
            self._apply_low_optimizations()
    
    def _apply_maximum_optimizations(self):
        """Apply maximum optimization settings for best performance."""
        # Cache optimizations
        self.cache.use_memory_cache = True
        self.cache.memory_cache_size = 2000
        self.cache.use_disk_cache = True
        self.cache.use_redis = True
        self.cache.enable_cache_warming = True
        
        # FAISS optimizations
        self.faiss.use_faiss = True
        self.faiss.hnsw_ef_construction = 400
        self.faiss.hnsw_ef_search = 100
        self.faiss.use_dimensionality_reduction = True
        
        # Parallel processing
        self.parallel.max_workers = min(8, self.available_cpu_cores)
        self.parallel.use_async_processing = True
        self.parallel.enable_batch_processing = True
        
        # Search optimizations
        self.search.enable_early_stopping = True
        self.search.enable_metadata_filtering = True
        self.search.enable_query_preprocessing = True
        self.search.adaptive_weight_adjustment = True
    
    def _apply_high_optimizations(self):
        """Apply high optimization settings (default)."""
        # Cache optimizations
        self.cache.use_memory_cache = True
        self.cache.memory_cache_size = 1000
        self.cache.use_disk_cache = True
        self.cache.use_redis = False  # Optional for high level
        
        # FAISS optimizations
        self.faiss.use_faiss = True
        self.faiss.use_dimensionality_reduction = False  # Optional for high level
        
        # Parallel processing
        self.parallel.max_workers = min(4, self.available_cpu_cores)
        self.parallel.use_async_processing = True
        
        # Search optimizations
        self.search.enable_early_stopping = True
        self.search.enable_metadata_filtering = True
    
    def _apply_medium_optimizations(self):
        """Apply medium optimization settings."""
        # Cache optimizations
        self.cache.use_memory_cache = True
        self.cache.memory_cache_size = 500
        self.cache.use_disk_cache = True
        self.cache.use_redis = False
        
        # FAISS optimizations
        self.faiss.use_faiss = True
        self.faiss.use_dimensionality_reduction = False
        
        # Parallel processing
        self.parallel.max_workers = 2
        self.parallel.use_async_processing = False
        
        # Search optimizations
        self.search.enable_early_stopping = False
        self.search.enable_metadata_filtering = True
    
    def _apply_low_optimizations(self):
        """Apply minimal optimization settings."""
        # Cache optimizations
        self.cache.use_memory_cache = True
        self.cache.memory_cache_size = 100
        self.cache.use_disk_cache = False
        self.cache.use_redis = False
        
        # FAISS optimizations
        self.faiss.use_faiss = False
        
        # Parallel processing
        self.parallel.max_workers = 1
        self.parallel.use_async_processing = False
        
        # Search optimizations
        self.search.enable_early_stopping = False
        self.search.enable_metadata_filtering = False

def create_optimization_config(
    optimization_level: str = "high",
    custom_settings: Optional[Dict[str, Any]] = None
) -> OptimizationConfig:
    """
    Create an optimization configuration with the specified level and custom settings.
    
    Args:
        optimization_level: "low", "medium", "high", or "maximum"
        custom_settings: Dictionary of custom settings to override defaults
        
    Returns:
        OptimizationConfig instance
    """
    # Create default sub-configurations
    cache_config = CacheOptimizationConfig()
    faiss_config = FAISSOptimizationConfig()
    parallel_config = ParallelProcessingConfig()
    search_config = SearchOptimizationConfig()
    monitoring_config = MonitoringConfig()
    
    # Apply custom settings if provided
    if custom_settings:
        for key, value in custom_settings.items():
            if key.startswith('cache.'):
                setattr(cache_config, key[6:], value)
            elif key.startswith('faiss.'):
                setattr(faiss_config, key[6:], value)
            elif key.startswith('parallel.'):
                setattr(parallel_config, key[9:], value)
            elif key.startswith('search.'):
                setattr(search_config, key[7:], value)
            elif key.startswith('monitoring.'):
                setattr(monitoring_config, key[11:], value)
    
    # Create master configuration
    config = OptimizationConfig(
        cache=cache_config,
        faiss=faiss_config,
        parallel=parallel_config,
        search=search_config,
        monitoring=monitoring_config,
        optimization_level=optimization_level
    )
    
    return config

def get_recommended_config_for_data_size(
    company_count: int,
    patent_count: int,
    available_memory_gb: float = 16.0,
    available_cpu_cores: int = 8
) -> OptimizationConfig:
    """
    Get recommended optimization configuration based on data size and available resources.
    
    Args:
        company_count: Number of company records
        patent_count: Number of patent records
        available_memory_gb: Available memory in GB
        available_cpu_cores: Number of available CPU cores
        
    Returns:
        Recommended OptimizationConfig
    """
    total_records = company_count + patent_count
    
    # Determine optimization level based on data size
    if total_records > 1000000:  # Very large dataset
        optimization_level = "maximum"
    elif total_records > 100000:  # Large dataset
        optimization_level = "high"
    elif total_records > 10000:  # Medium dataset
        optimization_level = "medium"
    else:  # Small dataset
        optimization_level = "low"
    
    # Custom settings based on data characteristics
    custom_settings = {}
    
    # Adjust FAISS settings for patent data size
    if patent_count > 500000:
        custom_settings.update({
            'faiss.use_dimensionality_reduction': True,
            'faiss.target_dimensions': 128,
            'faiss.hnsw_ef_construction': 400,
            'faiss.hnsw_ef_search': 100
        })
    
    # Adjust cache settings based on memory
    if available_memory_gb >= 32:
        custom_settings.update({
            'cache.memory_cache_size': 5000,
            'cache.disk_cache_size_mb': 4096,
            'cache.use_redis': True
        })
    elif available_memory_gb >= 16:
        custom_settings.update({
            'cache.memory_cache_size': 2000,
            'cache.disk_cache_size_mb': 2048
        })
    else:
        custom_settings.update({
            'cache.memory_cache_size': 500,
            'cache.disk_cache_size_mb': 512
        })
    
    # Adjust parallel processing based on CPU cores
    custom_settings['parallel.max_workers'] = min(available_cpu_cores, 8)
    
    config = create_optimization_config(optimization_level, custom_settings)
    config.company_data_size = company_count
    config.patent_data_size = patent_count
    config.available_memory_gb = available_memory_gb
    config.available_cpu_cores = available_cpu_cores
    
    return config

# Default configurations for different scenarios
DEFAULT_DEVELOPMENT_CONFIG = create_optimization_config("medium")
DEFAULT_PRODUCTION_CONFIG = create_optimization_config("high")
DEFAULT_HIGH_PERFORMANCE_CONFIG = create_optimization_config("maximum")

# Recommended configuration for the current dataset (6K companies, 1.6M patents)
RECOMMENDED_CONFIG_FOR_CURRENT_DATASET = get_recommended_config_for_data_size(
    company_count=6000,
    patent_count=1600000,
    available_memory_gb=16.0,
    available_cpu_cores=8
) 


