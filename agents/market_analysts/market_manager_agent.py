"""
agents/market_manager_agent.py

This module defines the `VerifierAgent`, responsible for evaluating the final merged answer
in a multi-agent RAG system. It assesses answer quality based on relevance, completeness,
correctness, and clarity, then returns a score and feedback.

Core Features:
- Uses a configurable LLM (OpenAI or HuggingFace-based) to evaluate generated answers.
- Parses response to extract evaluation text, score (1-10), and follow-up questions.
- Determines whether the answer needs improvement based on a score threshold.

Typical Usage:
    agent = VerifierAgent(qa_model="openai", threshold=7)
    feedback = agent.run({
        "question": "What is Tesla's expansion plan?",
        "merged_answer": "Tesla will open a factory in Singapore..."
    })
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from config.prompt import MARKET_MANAGER_PROMPT_NO_PATENT, MARKET_MANAGER_PROMPT_WITH_PATENT
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.model_utils import get_qwen_vl_model_and_processor
from transformers import pipeline

class MarketManagerAgent(BaseAgent):
    def __init__(self, name="MarketManagerAgent", qa_model="openai"):
        super().__init__(name)
        self.qa_model = qa_model
        logger.info(f"Initializing MarketManagerAgent with model: {qa_model}")

        # Instantiate the raw LLM (no prompt bound yet)
        if qa_model == "openai":
            self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

        elif "gemini" in qa_model:
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

        elif "qwen" in qa_model:
            model, processor = get_qwen_vl_model_and_processor()
            self.llm = HuggingFacePipeline(
                pipeline=pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=processor.tokenizer,
                    device_map="auto",
                    return_full_text=False,  # <-- key change
                    max_new_tokens=1024,  # or whatever you need
                    clean_up_tokenization_spaces=True
                )
            )
        else:
            raise ValueError(f"Unknown qa_model: {qa_model!r}")

        # Load both templates
        self.prompt_with_patent = PromptTemplate.from_template(
            MARKET_MANAGER_PROMPT_WITH_PATENT
        )
        self.prompt_without_patent = PromptTemplate.from_template(
            MARKET_MANAGER_PROMPT_NO_PATENT
        )

        # Parser for the output
        self.parser = StrOutputParser()

    def run(self, input_data: dict) -> str:
        """
        Evaluates the merged answer against the original user question using a scoring rubric.

        Args:
            input_data (dict): Dictionary with the following keys:
                - 'question' (str): The original user question.
                - 'merged_answer' (str): The answer to evaluate.

        Returns:
            dict: A dictionary containing:
                - 'score' (int): Score between 1 and 10.
                - 'needs_retry' (bool): True if score < threshold.
                - 'evaluation' (str): Explanation of the score.
                - 'merged_answer' (str): The original answer evaluated.
                - 'follow_up_questions' (list[str], optional): Suggestions for improvement if needed.
        """
        query = input_data.get("question")
        patent_abstract = input_data.get('patent_abstract', None)

        market_opportunities = input_data.get("MarketOpportunityAgent", None)
        market_risks = input_data.get("MarketRiskAgent", None)

        logger.info("Running MarketManagerAgent system")

        if not market_opportunities or not market_risks or not query:
            logger.warning("⚠️ Missing contexts or question")
            return "No context to synthesize final output"

        if patent_abstract:
            logger.info("Found patent abstract for MarketManagerAgent")
            template = self.prompt_with_patent
            prompt_input = {
                "market_opportunities": market_opportunities,
                "market_risks": market_risks,
                "query": query,
                "patent_abstract": patent_abstract
            }
        else:
            logger.info("Patent abstract not found for MarketManagerAgent")
            template = self.prompt_without_patent
            prompt_input = {
                "market_opportunities": market_opportunities,
                "market_risks": market_risks,
                "query": query
            }

        prompt_text = template.format(**prompt_input)
        # Render and invoke the LLM
        try:
            if isinstance(self.llm, HuggingFacePipeline):
                llm_output = self.llm(prompt_text)
                final = self.parser.parse(llm_output)
            else:
                messages = [
                    HumanMessage(content=prompt_text)
                ]
                llm_response = self.llm(messages)
                final = self.parser.parse(llm_response.content)

            return final

        except Exception as e:
            logger.error("MarketManagerAgent failed:", exc_info=e)
            return "Failed to synthesize final suggestion in the Market."

