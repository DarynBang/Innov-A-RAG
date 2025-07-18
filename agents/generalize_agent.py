"""
agents/generalize_agent.py
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from typing import Optional, List
from text_generation.base_runner import get_text_captioning_runner
import torch
import gc

class GeneralizeAgent(BaseAgent):
    def __init__(self, name: str = "TextAgent", qa_model = "qwen"):
        super().__init__(name)
        self.qa_model = qa_model
        logger.info(f"Initializing GeneralizeAgent with backend model: {qa_model}")

        # Dynamically load the correct runner and wrap in RunnableLambda
        self.caption_with_llm = get_text_captioning_runner(qa_model)

    def run(self,
            input_data: dict,
            patent_contexts: Optional[List[dict]] = None,
            firm_summary_contexts: Optional[List[dict]] = None) -> str:
        question = input_data.get("question", "")
        
        if not question or not patent_contexts or not firm_summary_contexts:
            logger.warning("Missing question or contexts. Skipping generation.")
            return "No answer found."

        # Extract raw text chunks for Firm Summary
        firm_chunks = [c["chunk"] for c in firm_summary_contexts]
        patent_chunks = [c["chunk"] for c in patent_contexts]
        firm_context_str = "\n- ".join(firm_chunks)
        patent_context_str = "\n- ".join(patent_chunks)

        # Free memory before LLM call
        del firm_summary_contexts, patent_contexts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Generate answer using the selected model
        return self.caption_with_llm.invoke({"query": question,
                                             "firm_summary_context": firm_context_str,
                                             "patent_context": patent_context_str})


