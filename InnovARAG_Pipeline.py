import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.multi_agent_runner import MultiAgentRunner
from patent_rag import PatentRAG
from firm_summary_rag import FirmSummaryRAG
from config.rag_config import patent_config, firm_config
import os
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
            qa_text = agent_config.get("qa_text", "openai")
            qa_generalize = agent_config.get("qa_generalize", "openai")
            qa_planning = agent_config.get("qa_planning", "openai")
            qa_merge = agent_config.get("qa_merge", "openai")
            qa_verifier = agent_config.get("qa_verifier", "openai")


            self.multi_agent = MultiAgentRunner(patent_rag=self.patent_rag, firm_rag=self.firm_rag, config=agent_config)

            self.multi_agent.register_agent("TextAgent", qa_model=qa_text)

            self.multi_agent.register_agent("GeneralizeAgent", qa_model=qa_generalize)

            # --- new code ---
            self.multi_agent.register_agent("PlanningAgent", qa_model=qa_planning)
            self.multi_agent.register_agent("MergeAgent", qa_model=qa_merge)
            self.multi_agent.register_agent("VerifierAgent", qa_model=qa_verifier)
            # --- end new code ---

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

    def process_query_patent(self, question: str):
        return self.multi_agent.run(question)

    def process_query_firm(self, question: str):
        return self.multi_agent.run(question)
#         print("\n👋 Chat interrupted. Goodbye!")

# if __name__ == '__main__':
#     main()
