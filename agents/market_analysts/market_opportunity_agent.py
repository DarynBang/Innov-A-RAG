"""
agents/market_analysts/market_opportunity_agent.py
"""

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agents.base import BaseAgent
from config.prompt import MARKET_OPPORTUNITY_PROMPT_NO_PATENT, MARKET_OPPORTUNITY_PROMPT_WITH_PATENT
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.model_utils import get_qwen_vl_model_and_processor
from transformers import pipeline
import pprint

class MarketOpportunityAgent(BaseAgent):
    def __init__(self, name="MarketOpportunityAgent", qa_model="openai", ):
        super().__init__(name)
        logger.info(f"Initializing MarketOpportunityAgent with model: {qa_model}")

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
            MARKET_OPPORTUNITY_PROMPT_WITH_PATENT
        )
        self.prompt_without_patent = PromptTemplate.from_template(
            MARKET_OPPORTUNITY_PROMPT_NO_PATENT
        )

        # Parser for the output
        self.parser = StrOutputParser()

    def run(self,
            input_data: dict,
            ) -> str:
        generalized_answers = input_data.get("GeneralizeAgent", "").strip()
        query = input_data.get("question")
        patent_abstract = input_data.get('patent_abstract', None)
        logger.info("Running MarketOpportunityAgent")

        if not generalized_answers or not query:
            logger.warning("⚠️ Missing context or question")
            return "No context to generate market opportunities."

        if patent_abstract:
            logger.info("Found patent abstract for MarketOpportunityAgent")
            template = self.prompt_with_patent
            prompt_input = {
                "general_summary": generalized_answers,
                "query": query,
                "patent_abstract": patent_abstract
            }
        else:
            logger.info("Patent abstract not found for MarketOpportunityAgent")
            template = self.prompt_without_patent
            prompt_input = {
                "general_summary": generalized_answers,
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
            logger.error("MarketOpportunityAgent failed:", exc_info=e)
            return "Failed to generate market opportunities."


