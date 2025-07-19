import logging

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.multi_agent_runner import MultiAgentRunner
from patent_rag import PatentRAG
from firm_summary_rag import FirmSummaryRAG
from config.agent_config import agent_config
from config.rag_config import patent_config, firm_config
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
        Initialize the InnovARAG_Pipeline.
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

        if ingest_only is False:
            # one global qa_model for all agents (you could customize per‐agent too)
            qa_generalize = agent_config.get("qa_generalize", "openai")
            qa_market_opportunity = agent_config.get("qa_market_opportunity", "openai")
            qa_market_risk = agent_config.get("qa_market_risk", "openai")
            qa_market_manager = agent_config.get("qa_market_manager", 'openai')

            self.multi_agent = MultiAgentRunner()

            self.multi_agent.register_agent("GeneralizeAgent", qa_model=qa_generalize)

            self.multi_agent.register_agent("MarketOpportunityAgent", qa_model=qa_market_opportunity)
            self.multi_agent.register_agent("MarketRiskAgent", qa_model=qa_market_risk)
            self.multi_agent.register_agent("MarketManagerAgent", qa_model=qa_market_manager)

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
        with torch.inference_mode():
            patent_results = self.patent_rag.retrieve_patent_contexts(question, top_k=3)
            firm_summary_results = self.firm_rag.retrieve_firm_contexts(question, top_k=3)

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print("Freed space from patent and firm Retriever")

        except Exception as e:
            print(f"Error freeing space for Retriever due to {e}")

        # Pass context + question into the multi-agent system
        self.multi_agent.run({"question": question,
                            "patent_abstract": patent_abstract},
                            patent_contexts=patent_results,
                            firm_summary_contexts=firm_summary_results)

def main():
    # Initialize pipeline with configs

    # Consistent index folder under PDF dir
    index_dir = r"RAG_INDEX"
    query = r"Machine Learning and Computer Vision"
    patent_abstract = """"""

    pipeline = InnovARAG_Pipeline(
        index_dir=index_dir,
        agent_config=agent_config,
        patent_config=patent_config,
        firm_config=firm_config,
        ingest_only=False
    )

    # Run ingestion
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

    pipeline.process_query(query)


if __name__ == '__main__':
    main()
