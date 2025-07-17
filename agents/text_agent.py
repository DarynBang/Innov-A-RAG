
# agents/text_agent.py
from M3ARAG.agents.base import BaseAgent
from typing import Optional, List

from langchain_core.runnables import RunnableLambda
import logging
import torch
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextAgent(BaseAgent):
    def __init__(self, name: str = "TextAgent", qa_model = "qwen"):
        super().__init__(name)

        # === Load runner ===
        if qa_model == "openai":
            logger.info("Using OpenAI GPT-4o-mini for Question Answering.")
            from M3ARAG.text_captioning.openai_runner import generate_caption_with_openai
            self.caption_with_llm = RunnableLambda(generate_caption_with_openai)

        elif qa_model == "gemini":
            logger.info("Using Gemini for Question Answering.")
            from M3ARAG.text_captioning.gemini_runner import generate_caption_with_gemini
            self.caption_with_llm = RunnableLambda(generate_caption_with_gemini)

        elif qa_model == "qwen":
            logger.info("Using Qwen2.5-VL for Question Answering.")
            from M3ARAG.text_captioning.qwen_runner import generate_caption_batch
            self.caption_with_llm = RunnableLambda(generate_caption_batch)


    def run(self, input_data: dict, contexts: Optional[List[dict]] = None) -> str:
        question = input_data.get("question", "")

        contexts = [ctx['chunk'] for ctx in contexts]
        contexts_str = "\n- ".join(contexts)

        # Memory cleanup
        del contexts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self.caption_with_llm.invoke({"query": question, "texts": contexts_str})
