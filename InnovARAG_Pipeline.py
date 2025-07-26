import logging

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.multi_agent_runner import MultiAgentRunner
from patent_rag import PatentRAG
from firm_summary_rag import FirmSummaryRAG
from config.agent_config import agent_config
from config.rag_config import patent_config, firm_config
from tools.company_tools import init_company_tools
from tools.patent_tools import init_patent_tools
from tools.hybrid_rag_tools import hybrid_rag_retrieval_tool_wrapper
from tools.enhanced_hybrid_rag_tools import (
    enhanced_hybrid_rag_retrieval_tool,
    company_data_with_mapping_tool,
    mapping_key_search_tool
)
import os
import gc
import warnings
import pandas as pd

# --- new code --- 
warnings.filterwarnings("ignore")


# import torch
# torch.set_default_dtype(torch.bfloat16)

class InnovARAG_Pipeline:
    def __init__(self, index_dir, patent_config, firm_config, agent_config, ingest_only=False):
        """
        Initialize the InnovARAG_Pipeline with enhanced multi-agent capabilities.
        """
        # Initialize RAG indexer and multi-agent QA system
        os.makedirs(index_dir, exist_ok=True)
        self.index_dir = index_dir
        self.multi_agent = None

        patent_df = pd.read_csv(patent_config.get("patent_csv"))
        firm_df = pd.read_csv(firm_config.get("firm_csv"))

        # Initialize RAG
        self.patent_rag = PatentRAG(df=patent_df,
                                    index_dir=index_dir,
                                    config=patent_config)

        self.firm_rag = FirmSummaryRAG(df=firm_df,
                                       index_dir=index_dir,
                                       config=firm_config)

        if not ingest_only:
            # Initialize enhanced multi-agent runner
            self.multi_agent = MultiAgentRunner()
            
            # Initialize and register tools
            company_tools = init_company_tools(firm_df, index_dir)
            patent_tools = init_patent_tools(patent_df, index_dir)
            
            all_tools = {**company_tools, **patent_tools}
            all_tools['hybrid_rag_retrieval'] = hybrid_rag_retrieval_tool_wrapper
            
            # Add enhanced hybrid tools
            all_tools['enhanced_hybrid_rag_retrieval'] = enhanced_hybrid_rag_retrieval_tool
            all_tools['company_data_with_mapping'] = company_data_with_mapping_tool
            all_tools['mapping_key_search'] = mapping_key_search_tool
            
            self.multi_agent.register_tools(all_tools)
            logger.info("Enhanced multi-agent system initialized with tools")

    def ingest_patent(self, force_reindex=False) -> None:
        self.patent_rag.ingest_all(force_reindex=force_reindex)


    def ingest_firm(self, force_reindex=False) -> None:
        self.firm_rag.ingest_all(force_reindex=force_reindex)


    def add_patent_to_index(self,
                            patent_id: str,
                            company_id: str,
                            company_name: str,
                            full_text: str):

        self.patent_rag.add_one(patent_id=patent_id,
                                company_id=company_id,
                                company_name=company_name,
                                full_text=full_text)


    def add_firm_summary_to_index(self,
                                  company_id: str,
                                  company_name: str,
                                  company_keywords: str,
                                  summary_text: str
                                  ):
        self.firm_rag.add_one(company_id=company_id,
                              company_name=company_name,
                              company_keywords=company_keywords,
                              summary_text=summary_text)


    def process_query(self, question: str, patent_abstract: str = None):
        """
        Process a query using the enhanced multi-agent workflow.
        
        Args:
            question: User query
            patent_abstract: Optional patent abstract (legacy parameter)
            
        Returns:
            Enhanced workflow results
        """
        logger.info(f"Processing query with enhanced workflow: {question}")
        
        if not self.multi_agent:
            logger.error("Multi-agent system not initialized. Set ingest_only=False during initialization.")
            return None
        
        try:
            # Use the enhanced workflow
            results = self.multi_agent.run_enhanced_workflow(question)
            
            # Clean up GPU memory if available
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info("Freed GPU memory after processing")
            except Exception as e:
                logger.error(f"Error freeing memory: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return None

def main():
    # Initialize pipeline with configs

    # Consistent index folder under PDF dir
    index_dir = r"RAG_INDEX"
    query = r"What are the market opportunities for AI companies in healthcare?"
    
    logger.info("Initializing InnovARAG Pipeline...")
    pipeline = InnovARAG_Pipeline(
        index_dir=index_dir,
        agent_config=agent_config,
        patent_config=patent_config,
        firm_config=firm_config,
        ingest_only=False
    )

    # Run ingestion
    logger.info("Running data ingestion...")
    pipeline.ingest_firm(False)
    pipeline.ingest_patent(False)

    # Add patent/firm summary to index
    # pipeline.add_patent_to_index(
    #     patent_id="JP2024001234A",
    #     company_name="Random Company idk",
    #     company_id="1235678",
    #     full_text="Here is the full, cleaned patent text ...")
    #
    # pipeline.add_firm_summary_to_index(
    #     company_id="HOJIN_1234",
    #     company_name="Acme Co.",
    #     company_keywords="robotics|ai",
    #     summary_text="Acme develops advanced robots...")

    logger.info("Processing query with enhanced multi-agent workflow...")
    results = pipeline.process_query(query)
    
    if results:
        logger.info("Query processing completed successfully!")
        logger.info(f"Analysis team used: {results.get('metadata', {}).get('analysis_team_used', 'Unknown')}")
        logger.info(f"Total contexts retrieved: {results.get('total_contexts', 0)}")
    else:
        logger.error("Query processing failed")


if __name__ == '__main__':
    main()
