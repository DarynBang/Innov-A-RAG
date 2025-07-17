# agents/generalize_agent.py
from M3ARAG.agents.base import BaseAgent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from M3ARAG.config.agent_config import GENERALIZED_PROMPT
from M3ARAG.utils.model_utils import get_qwen_vl_model_and_processor
from typing import Optional, List
from transformers import pipeline
from langchain import HuggingFacePipeline


class GeneralizeAgent(BaseAgent):
    def __init__(self, name: str = "GeneralizeAgent", qa_model: str = "qwen"):
        super().__init__(name)
        prompt = PromptTemplate.from_template(GENERALIZED_PROMPT)

        if qa_model == "gpt":
            self.chain = prompt | ChatOpenAI(model_name="gpt-4o-mini", temperature=0) | StrOutputParser()

        elif "qwen" in qa_model:
            model, processor = get_qwen_vl_model_and_processor()

            qwen_pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=processor.tokenizer,
                device_map="auto",
                max_length=1024,
                truncation=True,
            )
            qwen_llm = HuggingFacePipeline(pipeline=qwen_pipeline)

            self.chain = prompt | qwen_llm | StrOutputParser()


    def run(self, input_data: dict, contexts: Optional[List[dict]] = None) -> str:
        # return self.chain.invoke({
        #     "text_answer": input_data.get("text_answer", ""),
        #     "image_answer": input_data.get("image_answer", "")
        # })

        return self.chain.invoke({
            "text_answer": input_data.get("TextAgent", ""),
            "image_answer": input_data.get("ImageAgent", "")
        })
