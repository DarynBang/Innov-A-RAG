"""
optimization_utils.py: Utility functions for optimization system setup and performance monitoring.

This module contains utility functions that were moved from main.py to better organize
the codebase and separate optimization-related functionality.
"""
import time
from utils.logging_utils import get_logger
from tools.optimized_hybrid_rag_tools import (
    get_performance_analytics_tool,
    optimize_for_queries_tool,
    batch_optimized_retrieval_tool,
    get_optimized_hybrid_tools
)
from config.optimization_config import RECOMMENDED_CONFIG_FOR_CURRENT_DATASET

logger = get_logger(__name__)

def setup_optimization_system():
    """Set up the optimization system with recommended configuration."""
    logger.info("Setting up optimization system...")
    
    try:
        # Get recommended configuration for the dataset
        config = RECOMMENDED_CONFIG_FOR_CURRENT_DATASET
        
        logger.info("Optimization Configuration:")
        logger.info(f"   - Companies: {config.company_data_size:,}")
        logger.info(f"   - Patents: {config.patent_data_size:,}")
        logger.info(f"   - Optimization level: {config.optimization_level}")
        logger.info(f"   - FAISS enabled: {config.faiss.use_faiss}")
        logger.info(f"   - Memory cache enabled: {config.cache.use_memory_cache}")
        logger.info(f"   - Memory cache size: {config.cache.memory_cache_size}")
        logger.info(f"   - Disk cache enabled: {config.cache.use_disk_cache}")
        logger.info(f"   - Redis cache enabled: {config.cache.use_redis}")
        logger.info(f"   - Max parallel workers: {config.parallel.max_workers}")
        logger.info(f"   - Early stopping enabled: {config.search.enable_early_stopping}")
        logger.info(f"   - Metadata filtering enabled: {config.search.enable_metadata_filtering}")
        
        # Initialize optimized tools (this will now work because RAG systems are ready)
        logger.info("Initializing optimized hybrid tools...")
        tools = get_optimized_hybrid_tools()
        
        # Only pre-warm cache if initialization was successful
        if tools and hasattr(tools, 'company_hybrid_retriever') and tools.company_hybrid_retriever:
            logger.info("Pre-warming cache with common queries...")
            common_queries = [
                "artificial intelligence",
                "machine learning", 
                "innovation technology",
                "patent analysis",
                "company research"
            ]
            try:
                optimize_for_queries_tool(common_queries)
                logger.info("Cache pre-warming completed successfully")
            except Exception as e:
                logger.warning(f"Cache pre-warming failed (not critical): {e}")
        else:
            logger.warning("Skipping cache pre-warming due to retriever initialization issues")
        
        logger.info("Optimization system setup completed successfully")
        return config, tools
        
    except Exception as e:
        logger.error(f"Error setting up optimization system: {e}")
        logger.warning("Falling back to standard system without optimizations")
        return None, None

def log_performance_analytics(optimization_tools=None):
    """Log current performance analytics."""
    try:
        logger.info("Getting performance analytics...")
        
        # Use existing tools if available, otherwise create new ones
        if optimization_tools:
            logger.info("Using existing optimization tools for analytics")
            stats = optimization_tools.get_performance_analytics() if hasattr(optimization_tools, 'get_performance_analytics') else {}
        else:
            logger.info("Creating new tools instance for analytics")
            stats = get_performance_analytics_tool()
        
        if 'tool_stats' in stats:
            tool_stats = stats['tool_stats']
            logger.info("Tool Performance Statistics:")
            logger.info(f"   - Total queries processed: {tool_stats.get('total_queries', 0)}")
            logger.info(f"   - Company queries: {tool_stats.get('company_queries', 0)}")
            logger.info(f"   - Patent queries: {tool_stats.get('patent_queries', 0)}")
            logger.info(f"   - Parallel queries: {tool_stats.get('parallel_queries', 0)}")
            logger.info(f"   - Average response time: {tool_stats.get('avg_response_time', 0):.3f}s")
        
        # Log retriever statistics
        for retriever_type in ['company_retriever_stats', 'patent_retriever_stats']:
            if retriever_type in stats and stats[retriever_type]:
                retriever_stats = stats[retriever_type]
                retriever_name = retriever_type.replace('_', ' ').title()
                logger.info(f"{retriever_name}:")
                logger.info(f"   - Cache hit rate: {retriever_stats.get('cache_hit_rate', 0):.2%}")
                logger.info(f"   - Total queries: {retriever_stats.get('total_queries', 0)}")
                logger.info(f"   - FAISS queries: {retriever_stats.get('faiss_queries', 0)}")
                logger.info(f"   - Early stops: {retriever_stats.get('early_stops', 0)}")
        
        # Log optimization features status
        if 'optimization_features' in stats:
            features = stats['optimization_features']
            logger.info("Optimization Features Status:")
            logger.info(f"   - FAISS enabled: {features.get('faiss_enabled', False)}")
            logger.info(f"   - Caching enabled: {features.get('caching_enabled', False)}")
            logger.info(f"   - Dimensionality reduction: {features.get('dimensionality_reduction', False)}")
            logger.info(f"   - Metadata filtering: {features.get('metadata_filtering', False)}")
            logger.info(f"   - Parallel processing: {features.get('parallel_processing', False)}")
        else:
            logger.info("Basic analytics - optimization features not available")
            
    except Exception as e:
        logger.warning(f"Could not retrieve performance analytics: {e}")

def process_query_with_optimization(runner, question, optimization_tools=None, use_batch=False, product_suggestion_mode=False):
    """Process query using optimization features with detailed logging."""
    mode_desc = "product suggestion" if product_suggestion_mode else "market analysis"
    logger.info(f"Processing query with optimizations ({mode_desc} mode): '{question}'")
    
    start_time = time.time()
    
    try:
        if use_batch:
            # Demonstrate batch processing for multiple related queries
            logger.info("Using batch processing for related queries...")
            if product_suggestion_mode:
                related_queries = [
                    question,
                    f"What products or technologies are mentioned related to {question}?",
                    f"What innovations are described in {question}?"
                ]
            else:
                related_queries = [
                    question,
                    f"What are the market opportunities related to {question}?",
                    f"What are the risks associated with {question}?"
                ]
            
            batch_results = batch_optimized_retrieval_tool(
                queries=related_queries,
                top_k=3,
                search_type="both"
            )
            
            logger.info(f"Batch processing completed: {len(batch_results)} queries processed")
        
        # Run the enhanced workflow
        workflow_mode = "product suggestion" if product_suggestion_mode else "market analysis"
        logger.info(f"Running enhanced multi-agent workflow in {workflow_mode} mode...")
        workflow_start = time.time()
        
        results = runner.run_enhanced_workflow(question, product_suggestion_mode=product_suggestion_mode)
        
        workflow_time = time.time() - workflow_start
        logger.info(f"Workflow completed in {workflow_time:.3f}s")
        
        if "error" in results:
            logger.error(f"Workflow failed: {results['error']}")
        else:
            if product_suggestion_mode:
                logger.info("Product suggestion workflow completed successfully")
                
                # Log product suggestion specific results
                product_suggestions = results.get('product_suggestions', '')
                if product_suggestions:
                    logger.info(f"Product suggestions generated (length: {len(product_suggestions)} characters)")
                else:
                    logger.warning("No product suggestions generated")
                    
                # Log validation results for product suggestions
                fact_checking = results.get('fact_checking', {})
                validation_score = fact_checking.get('overall_score', 0)
                flagged_issues = fact_checking.get('flagged_issues', [])
                logger.info(f"Product suggestion validation score: {validation_score}/10")
                if flagged_issues:
                    logger.warning(f"Validation issues found: {len(flagged_issues)}")
                    for issue in flagged_issues[:3]:  # Log first 3 issues
                        logger.warning(f"  - {issue}")
            else:
                logger.info("Market analysis workflow completed successfully")
            
            # Log additional optimization metrics
            total_time = time.time() - start_time
            logger.info(f"Total processing time: {total_time:.3f}s")
            
            # Get and log performance analytics after the query
            log_performance_analytics(optimization_tools)
        
        return results
        
    except Exception as e:
        logger.error(f"Error during optimized query processing: {e}")
        return {"error": str(e)}