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

# OPTIMIZATION UTILITIES
from utils.optimization_utils import (
    setup_optimization_system,
    log_performance_analytics,
    process_query_with_optimization
)

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)



def main():
    logger.info("=== Starting InnovARAG Multi-Agent RAG System (OPTIMIZED) ===")
    
    parser = argparse.ArgumentParser(description="InnovARAG Multi-Agent RAG System with Optimizations")
    parser.add_argument('--mode', choices=['query', 'test', 'ingest', 'chat'], required=True, help='Workflow mode to run')
    parser.add_argument('--query', type=str, help='User query (required for query mode)')
    parser.add_argument('--product_suggestion', action='store_true', 
                       help='Enable product suggestion mode (extracts products from contexts with citations)')
    parser.add_argument('--force_reindex', action='store_true', help='Force reindex during ingestion')
    parser.add_argument('--force_faiss_reindex', choices=['company', 'patent', 'all'], 
                       help='Force rebuild FAISS indexes for better performance')
    parser.add_argument('--force_bm25_reindex', choices=['company', 'patent', 'all'],
                       help='Force rebuild BM25 indexes for better keyword search')
    parser.add_argument('--force_all_indexes', choices=['company', 'patent', 'all'],
                       help='Force rebuild ALL indexes (BM25 + FAISS) without touching ChromaDB data')
    parser.add_argument('--optimization_level', choices=['low', 'medium', 'high', 'maximum'], 
                       default='high', help='Optimization level to use')
    parser.add_argument('--use_batch', action='store_true', help='Use batch processing for queries')
    parser.add_argument('--clear_cache', action='store_true', help='Clear all caches before starting')
    args = parser.parse_args()
    
    logger.info(f"Running in {args.mode} mode with {DEFAULT_LLM_TYPE} LLM")
    logger.info(f"Optimization level: {args.optimization_level}")
    
    # Log product suggestion mode status
    if args.product_suggestion:
        logger.info("Product Suggestion Mode: ENABLED - Will extract products from retrieved contexts")
    else:
        logger.info("Market Analysis Mode: ENABLED - Full multi-agent workflow with strategic analysis")
    
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
    
    # Handle FAISS reindexing if requested
    if args.force_faiss_reindex:
        logger.info(f"Force FAISS reindexing requested: {args.force_faiss_reindex}")
        if optimization_tools:
            # Reset tools before reindexing to ensure clean state
            reset_optimized_hybrid_tools()
            optimization_tools = get_optimized_hybrid_tools()  # Get fresh instance
            optimization_tools.force_reindex(args.force_faiss_reindex)
        else:
            logger.warning("Optimization tools not available for FAISS reindexing")
    
    # Handle BM25 reindexing if requested
    if args.force_bm25_reindex:
        logger.info(f"Force BM25 reindexing requested: {args.force_bm25_reindex}")
        if optimization_tools:
            # Reset tools before reindexing to ensure clean state
            reset_optimized_hybrid_tools()
            optimization_tools = get_optimized_hybrid_tools()  # Get fresh instance
            optimization_tools.force_bm25_reindex(args.force_bm25_reindex)
        else:
            logger.warning("Optimization tools not available for BM25 reindexing")
    
    # Handle ALL indexes reindexing if requested (BM25 + FAISS)
    if args.force_all_indexes:
        logger.info(f"Force ALL indexes reindexing requested: {args.force_all_indexes}")
        if optimization_tools:
            # Reset tools before reindexing to ensure clean state
            reset_optimized_hybrid_tools()
            optimization_tools = get_optimized_hybrid_tools()  # Get fresh instance
            optimization_tools.force_all_indexes_reindex(args.force_all_indexes)
        else:
            logger.warning("Optimization tools not available for index reindexing")

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
        
        # NEW OPTIMIZED HYBRID TOOLS RESET - ONLY for force reindex operations
        # DO NOT reset unnecessarily - it destroys singleton pattern and causes 53s re-init delay
        # reset_optimized_hybrid_tools()  # Commented out to prevent redundant re-initialization
        
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
            
        # Use default query for test mode - different for product suggestion vs market analysis
        if args.product_suggestion:
            question = args.query or "artificial intelligence applications and machine learning products"
        else:
            question = args.query or "Tell me about Intel's business focus and market opportunities with optimization analysis. NOTE THAT INTEL IS THE WHOLE COMPANY NAME ALREADY."
        
        mode_desc = "product suggestion" if args.product_suggestion else "market analysis"
        logger.info(f"Starting optimized query processing in {mode_desc} mode...")
        
        try:
            # Process query with optimization features
            results = process_query_with_optimization(
                runner, 
                question, 
                optimization_tools, 
                use_batch=args.use_batch,
                product_suggestion_mode=args.product_suggestion
            )
            
            if "error" in results:
                logger.error(f"Optimized workflow failed: {results['error']}")
            else:
                logger.info(f"Optimized {mode_desc} workflow completed successfully")
            
        except Exception as e:
            logger.error(f"Error during optimized query processing: {e}")
    
    elif args.mode == 'chat':
        # Interactive chat mode with optimizations
        mode_desc = "Product Suggestion" if args.product_suggestion else "Market Analysis"
        logger.info(f"Starting interactive chat mode with optimizations ({mode_desc})...")
        logger.info("\n" + "="*60)
        logger.info(f"InnovARAG Interactive Chat Mode (OPTIMIZED - {mode_desc.upper()})")
        logger.info("="*60)
        logger.info("Optimizations enabled: caching, parallel processing, FAISS indexing")
        
        if args.product_suggestion:
            logger.info("PRODUCT SUGGESTION MODE ACTIVE:")
            logger.info("   - Enter queries about technologies, patents, or innovations")
            logger.info("   - System will extract products from retrieved contexts")
            logger.info("   - All suggestions will be properly cited")
            logger.info("   - No external knowledge will be used")
        else:
            logger.info("MARKET ANALYSIS MODE ACTIVE:")
            logger.info("   - Enter queries about companies, patents, or innovation topics")
            logger.info("   - Full multi-agent workflow with strategic analysis")
            logger.info("   - Includes opportunity and risk assessment")
        
        logger.info("Type 'mode' to toggle between product suggestion and market analysis")
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
                    elif question.lower() == 'mode':
                        # Toggle between product suggestion and market analysis mode
                        args.product_suggestion = not args.product_suggestion
                        mode_desc = "Product Suggestion" if args.product_suggestion else "Market Analysis"
                        logger.info(f"\nMode switched to: {mode_desc.upper()}")
                        if args.product_suggestion:
                            logger.info("Now in Product Suggestion Mode - will extract products from contexts")
                        else:
                            logger.info("Now in Market Analysis Mode - full workflow with strategic analysis")
                        logger.info("")
                        continue
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
                        
                        # Show current mode
                        current_mode = "Product Suggestion" if args.product_suggestion else "Market Analysis"
                        logger.info(f"   - Current mode: {current_mode}")
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
                    results = process_query_with_optimization(
                        runner, 
                        question, 
                        optimization_tools,
                        product_suggestion_mode=args.product_suggestion
                    )
                    
                    if "error" in results:
                        logger.error(f"Error: {results['error']}")
                    else:
                        mode_desc = "product suggestion" if args.product_suggestion else "market analysis"
                        logger.info(f"{mode_desc.capitalize()} complete!")
                        
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
    
    
