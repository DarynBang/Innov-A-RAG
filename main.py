"""
main.py: Entry point for InnovARAG Multi-Agent RAG System
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

import argparse
from agents.multi_agent_runner import MultiAgentRunner
from config.agent_config import agent_config, DEFAULT_LLM_TYPE
from config.rag_config import patent_config, firm_config
from firm_summary_rag import FirmSummaryRAG
from patent_rag import PatentRAG
from tools.company_tools import init_company_tools
from tools.patent_tools import init_patent_tools
from tools.hybrid_rag_tools import hybrid_rag_retrieval_tool_wrapper
from tools.enhanced_hybrid_rag_tools import (
    enhanced_hybrid_rag_retrieval_tool,
    company_data_with_mapping_tool,
    mapping_key_search_tool
)
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)

def main():
    logger.info("=== Starting InnovARAG Multi-Agent RAG System ===")
    
    parser = argparse.ArgumentParser(description="InnovARAG Multi-Agent RAG System")
    parser.add_argument('--mode', choices=['query', 'test', 'ingest', 'chat'], required=True, help='Workflow mode to run')
    parser.add_argument('--query', type=str, help='User query (required for query mode)')
    parser.add_argument('--force_reindex', action='store_true', help='Force reindex during ingestion')
    parser.add_argument('--legacy', action='store_true', help='Use legacy workflow for backward compatibility')
    args = parser.parse_args()
    
    logger.info(f"Running in {args.mode} mode with {DEFAULT_LLM_TYPE} LLM (configured in agent_config.py)")
    if args.query:
        logger.info(f"Query: {args.query}")
    if args.legacy:
        logger.info("Using legacy workflow mode")

    # Load data and configs
    index_dir = "RAG_INDEX"
    logger.info("Loading patent and firm data...")
    
    try:
        patent_df = pd.read_csv(patent_config.get("patent_csv"))
        firm_df = pd.read_csv(firm_config.get("firm_csv"))
        logger.info(f"Loaded {len(patent_df)} patents and {len(firm_df)} firms")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Initialize RAG systems
    try:
        firm_rag = FirmSummaryRAG(df=firm_df, index_dir=index_dir, config=firm_config)
        patent_rag = PatentRAG(df=patent_df, index_dir=index_dir, config=patent_config)
        logger.info("RAG systems initialized")
    except Exception as e:
        logger.error(f"Error initializing RAG systems: {e}")
        return
    
    # Initialize tools
    try:
        init_company_tools(firm_df, index_dir)
        init_patent_tools(patent_df, index_dir)
        logger.info("Tools initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing tools: {e}")
        return

    if args.mode == 'ingest':
        logger.info("Starting data ingestion...")
        try:
            logger.info("Ingesting firm summaries...")
            firm_rag.ingest_all(force_reindex=args.force_reindex)
            logger.info("Ingesting patents...")
            patent_rag.ingest_all(force_reindex=args.force_reindex)
            logger.info("Ingestion completed successfully")
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
        return

    # Set up multi-agent runner
    try:
        logger.info("Initializing Multi-Agent Runner...")
        runner = MultiAgentRunner()
        
        # Initialize and register tools
        company_tools = init_company_tools(firm_df, index_dir)
        patent_tools = init_patent_tools(patent_df, index_dir)
        
        all_tools = {**company_tools, **patent_tools}
        all_tools['hybrid_rag_retrieval'] = hybrid_rag_retrieval_tool_wrapper
        
        # Add enhanced hybrid tools
        all_tools['enhanced_hybrid_rag_retrieval'] = enhanced_hybrid_rag_retrieval_tool
        all_tools['company_data_with_mapping'] = company_data_with_mapping_tool
        all_tools['mapping_key_search'] = mapping_key_search_tool
        
        runner.register_tools(all_tools)
        logger.info(f"Registered {len(all_tools)} tools")
        logger.info("Multi-agent runner initialized successfully")
        
    except Exception as e:
        logger.error(f"Error setting up multi-agent runner: {e}")
        return

    if args.mode in ['query', 'test']:
        if not args.query and args.mode != 'test':
            logger.error("Please provide a query with --query for query mode.")
            return
            
        # Use default query for test mode
        question = args.query or "Tell me about TechNova's business focus and market opportunities"
        logger.info(f"Processing query: {question}")
        
        try:
            if args.legacy:
                # Run legacy workflow for backward compatibility
                logger.info("Using legacy workflow...")
                
                # Retrieve contexts using RAG systems
                logger.info("Retrieving patent contexts...")
                patent_contexts = patent_rag.retrieve_patent_contexts(question, top_k=3)
                logger.info(f"Retrieved {len(patent_contexts)} patent contexts")
                
                logger.info("Retrieving firm contexts...")
                firm_contexts = firm_rag.retrieve_firm_contexts(question, top_k=3)
                logger.info(f"Retrieved {len(firm_contexts)} firm contexts")
                
                # Run legacy workflow
                result = runner.run_legacy_workflow(
                    {"question": question}, 
                    patent_contexts=patent_contexts, 
                    firm_summary_contexts=firm_contexts
                )
                
                logger.info("Legacy workflow completed successfully")
                logger.info("\n" + "="*80)
                logger.info("FINAL RESULT:")
                logger.info("="*80)
                logger.info(result)
                logger.info("="*80)
                
            else:
                # Run enhanced workflow
                logger.info("Using enhanced workflow...")
                results = runner.run_enhanced_workflow(question)
                
                if "error" in results:
                    logger.error(f"Workflow failed: {results['error']}")
                else:
                    logger.info("Enhanced workflow completed successfully")
                    # Summary already printed by the workflow
            
        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            logger.error(f"Error: {e}")
    
    elif args.mode == 'chat':
        # Interactive chat mode
        logger.info("Starting interactive chat mode...")
        logger.info("\n" + "="*60)
        logger.info("InnovARAG Interactive Chat Mode")
        logger.info("="*60)
        logger.info("Enter your queries about companies, patents, or general innovation topics.")
        logger.info("Type 'quit', 'exit', or press Ctrl+C to stop.\n")
        
        try:
            while True:
                try:
                    question = input("🤖 Ask me anything: ").strip()
                    
                    if not question:
                        continue
                        
                    if question.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    logger.info("\n" + "="*40)
                    logger.info("Processing your query...")
                    logger.info("="*40)
                    
                    # Run enhanced workflow
                    results = runner.run_enhanced_workflow(question)
                    
                    if "error" in results:
                        logger.error(f"Error: {results['error']}")
                    else:
                        logger.info("Analysis complete!")
                        
                    logger.info("\n" + "-"*60)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Error in chat mode: {e}")
                    logger.error(f"Error processing query: {e}")
                    
        except KeyboardInterrupt:
            pass
            
        logger.info("\nThanks for using InnovARAG! Goodbye!")

if __name__ == "__main__":
    main() 
    
    
