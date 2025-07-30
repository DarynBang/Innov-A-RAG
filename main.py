"""
main.py: Entry point for InnovARAG Multi-Agent RAG System with Performance Optimizations
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

import argparse
import time
from agents.multi_agent_runner import MultiAgentRunner
from config.agent_config import agent_config, DEFAULT_LLM_TYPE
from config.rag_config import patent_config, firm_config
from utils.firm_summary_rag import FirmSummaryRAG
from utils.patent_rag import PatentRAG
from tools.company_tools import init_company_tools
from tools.patent_tools import init_patent_tools

# OLD ENHANCED HYBRID TOOLS (commented out for comparison)
# from tools.enhanced_hybrid_rag_tools import (
#     enhanced_hybrid_rag_retrieval_tool,
#     company_data_with_mapping_tool,
#     mapping_key_search_tool
# )

# NEW OPTIMIZED HYBRID TOOLS
from tools.optimized_hybrid_rag_tools import (
    optimized_hybrid_rag_retrieval_tool,
    batch_optimized_retrieval_tool,
    get_performance_analytics_tool,
    clear_caches_tool,
    optimize_for_queries_tool,
    get_optimized_hybrid_tools,
    reset_optimized_hybrid_tools
)

# NEW OPTIMIZATION CONFIGURATION
from config.optimization_config import (
    RECOMMENDED_CONFIG_FOR_CURRENT_DATASET,
    create_optimization_config,
    get_recommended_config_for_data_size
)

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
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

def process_query_with_optimization(runner, question, optimization_tools=None, use_batch=False):
    """Process query using optimization features with detailed logging."""
    logger.info(f"Processing query with optimizations: '{question}'")
    
    start_time = time.time()
    
    try:
        if use_batch:
            # Demonstrate batch processing for multiple related queries
            logger.info("Using batch processing for related queries...")
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
        logger.info("Running enhanced multi-agent workflow...")
        workflow_start = time.time()
        
        results = runner.run_enhanced_workflow(question)
        
        workflow_time = time.time() - workflow_start
        logger.info(f"Workflow completed in {workflow_time:.3f}s")
        
        if "error" in results:
            logger.error(f"Workflow failed: {results['error']}")
        else:
            logger.info("Enhanced workflow completed successfully")
            
            # Log additional optimization metrics
            total_time = time.time() - start_time
            logger.info(f"Total processing time: {total_time:.3f}s")
            
            # Get and log performance analytics after the query
            log_performance_analytics(optimization_tools)
        
        return results
        
    except Exception as e:
        logger.error(f"Error during optimized query processing: {e}")
        return {"error": str(e)}

def main():
    logger.info("=== Starting InnovARAG Multi-Agent RAG System (OPTIMIZED) ===")
    
    parser = argparse.ArgumentParser(description="InnovARAG Multi-Agent RAG System with Optimizations")
    parser.add_argument('--mode', choices=['query', 'test', 'ingest', 'chat'], required=True, help='Workflow mode to run')
    parser.add_argument('--query', type=str, help='User query (required for query mode)')
    parser.add_argument('--force_reindex', action='store_true', help='Force reindex during ingestion')
    parser.add_argument('--optimization_level', choices=['low', 'medium', 'high', 'maximum'], 
                       default='high', help='Optimization level to use')
    parser.add_argument('--use_batch', action='store_true', help='Use batch processing for queries')
    parser.add_argument('--clear_cache', action='store_true', help='Clear all caches before starting')
    args = parser.parse_args()
    
    logger.info(f"Running in {args.mode} mode with {DEFAULT_LLM_TYPE} LLM")
    logger.info(f"Optimization level: {args.optimization_level}")
    
    if args.query:
        logger.info(f"Query: {args.query}")
    
    # Clear caches if requested
    if args.clear_cache:
        logger.info("Clearing all caches...")
        clear_caches_tool()
        logger.info("Caches cleared")
    
    # Load data and configs FIRST
    index_dir = "RAG_INDEX"
    logger.info("Loading patent and firm data...")
    
    try:
        patent_df = pd.read_csv(patent_config.get("patent_csv"))
        firm_df = pd.read_csv(firm_config.get("firm_csv"))
        logger.info(f"Loaded {len(patent_df):,} patents and {len(firm_df):,} firms")
        
        # Log data size for optimization reference
        total_records = len(patent_df) + len(firm_df)
        logger.info(f"Total records: {total_records:,} (Patents: {len(patent_df):,}, Companies: {len(firm_df):,})")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Initialize RAG systems SECOND
    try:
        logger.info("Initializing RAG systems...")
        global firm_rag, patent_rag  # Make them globally accessible
        firm_rag = FirmSummaryRAG(df=firm_df, index_dir=index_dir, config=firm_config)
        patent_rag = PatentRAG(df=patent_df, index_dir=index_dir, config=patent_config)
        logger.info("RAG systems initialized")
    except Exception as e:
        logger.error(f"Error initializing RAG systems: {e}")
        return
    
    # Initialize tools THIRD
    try:
        logger.info("Initializing tools...")
        init_company_tools(firm_df, index_dir)
        init_patent_tools(patent_df, index_dir)
        logger.info("Tools initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing tools: {e}")
        return
    
    # Setup optimization system LAST (after everything is ready)
    logger.info("Setting up optimization system after RAG initialization...")
    optimization_config, optimization_tools = setup_optimization_system()

    if args.mode == 'ingest':
        logger.info("Starting data ingestion...")
        ingestion_start = time.time()
        
        try:
            logger.info("Ingesting firm summaries...")
            firm_start = time.time()
            firm_rag.ingest_all(force_reindex=args.force_reindex)
            firm_time = time.time() - firm_start
            logger.info(f"Firm ingestion completed in {firm_time:.2f}s")
            
            logger.info("Ingesting patents...")
            patent_start = time.time()
            patent_rag.ingest_all(force_reindex=args.force_reindex)
            patent_time = time.time() - patent_start
            logger.info(f"Patent ingestion completed in {patent_time:.2f}s")
            
            total_ingestion_time = time.time() - ingestion_start
            logger.info(f"Total ingestion completed in {total_ingestion_time:.2f}s")
            logger.info(f"Ingestion rate: {total_records/total_ingestion_time:.1f} records/second")
            
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
        return

    # Set up multi-agent runner
    try:
        logger.info("Initializing Multi-Agent Runner...")
        runner = MultiAgentRunner()
        
        # OLD ENHANCED HYBRID TOOLS RESET (commented out)
        # from tools.enhanced_hybrid_rag_tools import reset_enhanced_hybrid_tools
        # reset_enhanced_hybrid_tools()
        
        # NEW OPTIMIZED HYBRID TOOLS RESET
        reset_optimized_hybrid_tools()
        
        # Initialize and register tools
        company_tools = init_company_tools(firm_df, index_dir)
        patent_tools = init_patent_tools(patent_df, index_dir)
        
        all_tools = {**company_tools, **patent_tools}
        
        # OLD ENHANCED HYBRID TOOLS (commented out)
        # all_tools['enhanced_hybrid_rag_retrieval'] = enhanced_hybrid_rag_retrieval_tool
        # all_tools['company_data_with_mapping'] = company_data_with_mapping_tool
        # all_tools['mapping_key_search'] = mapping_key_search_tool
        
        # NEW OPTIMIZED HYBRID TOOLS
        all_tools['optimized_hybrid_rag_retrieval'] = optimized_hybrid_rag_retrieval_tool
        all_tools['batch_optimized_retrieval'] = batch_optimized_retrieval_tool
        all_tools['get_performance_analytics'] = get_performance_analytics_tool
        
        runner.register_tools(all_tools)
        logger.info(f"Registered {len(all_tools)} tools (including optimized tools)")
        logger.info("Multi-agent runner initialized successfully with optimizations")
        
    except Exception as e:
        logger.error(f"Error setting up multi-agent runner: {e}")
        return

    if args.mode in ['query', 'test']:
        if not args.query and args.mode != 'test':
            logger.error("Please provide a query with --query for query mode.")
            return
            
        # Use default query for test mode
        question = args.query or "Tell me about Intel's business focus and market opportunities with optimization analysis. NOTE THAT INTEL IS THE WHOLE COMPANY NAME ALREADY."
        
        logger.info("Starting optimized query processing...")
        
        try:
            # Process query with optimization features
            results = process_query_with_optimization(runner, question, optimization_tools, use_batch=args.use_batch)
            
            if "error" in results:
                logger.error(f"Optimized workflow failed: {results['error']}")
            else:
                logger.info("Optimized workflow completed successfully")
            
        except Exception as e:
            logger.error(f"Error during optimized query processing: {e}")
    
    elif args.mode == 'chat':
        # Interactive chat mode with optimizations
        logger.info("Starting interactive chat mode with optimizations...")
        logger.info("\n" + "="*60)
        logger.info("InnovARAG Interactive Chat Mode (OPTIMIZED)")
        logger.info("="*60)
        logger.info("Optimizations enabled: caching, parallel processing, FAISS indexing")
        logger.info("Enter your queries about companies, patents, or innovation topics.")
        logger.info("Type 'stats' to see performance analytics")
        logger.info("Type 'clear' to clear caches")
        logger.info("Type 'quit', 'exit', or press Ctrl+C to stop.\n")
        
        query_count = 0
        session_start_time = time.time()
        
        try:
            while True:
                try:
                    question = input("Ask me anything: ").strip()
                    
                    if not question:
                        continue
                        
                    if question.lower() in ['quit', 'exit', 'q']:
                        break
                    elif question.lower() == 'stats':
                        logger.info("\n" + "="*40)
                        logger.info("PERFORMANCE ANALYTICS")
                        logger.info("="*40)
                        log_performance_analytics(optimization_tools)
                        
                        session_time = time.time() - session_start_time
                        if query_count > 0:
                            avg_time_per_query = session_time / query_count
                            logger.info(f"Session Statistics:")
                            logger.info(f"   - Queries processed: {query_count}")
                            logger.info(f"   - Session duration: {session_time:.1f}s")
                            logger.info(f"   - Average time per query: {avg_time_per_query:.3f}s")
                        logger.info("="*40 + "\n")
                        continue
                    elif question.lower() == 'clear':
                        logger.info("Clearing all caches...")
                        clear_caches_tool()
                        logger.info("Caches cleared\n")
                        continue
                    
                    query_count += 1
                    logger.info(f"\n" + "="*40)
                    logger.info(f"Processing query #{query_count}...")
                    logger.info("="*40)
                    
                    # Process with optimizations
                    results = process_query_with_optimization(runner, question, optimization_tools)
                    
                    if "error" in results:
                        logger.error(f"Error: {results['error']}")
                    else:
                        logger.info("Analysis complete!")
                        
                    logger.info("\n" + "-"*60)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in optimized chat mode: {e}")
                    
        except KeyboardInterrupt:
            pass
        
        # Final session statistics
        session_time = time.time() - session_start_time
        logger.info(f"\nChat session completed!")
        logger.info(f"Final session statistics:")
        logger.info(f"   - Total queries processed: {query_count}")
        logger.info(f"   - Total session time: {session_time:.1f}s")
        if query_count > 0:
            logger.info(f"   - Average time per query: {session_time/query_count:.3f}s")
        
        # Show final performance analytics
        logger.info("\nFinal Performance Analytics:")
        log_performance_analytics(optimization_tools)
            
        logger.info("\nThanks for using InnovARAG with optimizations! Goodbye!")

if __name__ == "__main__":
    main() 
    
    
