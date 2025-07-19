"""
MarketManagerAgent: Synthesizes the final answer using the selected LLM and outputs from opportunity/risk agents.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from agents.base import BaseAgent
from config.prompts import (
    MARKET_MANAGER_AGENT_SYSTEM_PROMPT,
    MARKET_MANAGER_AGENT_USER_PROMPT,
    DEFAULT_MODELS
)
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = get_logger(__name__)

class MarketManagerAgent(BaseAgent):
    def __init__(self, name="MarketManagerAgent", qa_model="openai"):
        super().__init__(name, qa_model)
        logger.info(f"Initializing MarketManagerAgent with model: {qa_model}")
        
        self.llm_type = qa_model
        self.llm = self._initialize_llm()
        
        # Initialize tools dictionary for future extensibility
        self.tools = {}
        logger.info("MarketManagerAgent initialization completed")
    
    def _initialize_llm(self):
        """Initialize the language model based on the specified type."""
        try:
            if self.llm_type == "openai":
                return ChatOpenAI(
                    model=DEFAULT_MODELS["openai"],
                    temperature=0
                )
            elif self.llm_type == "gemini":
                return ChatGoogleGenerativeAI(
                    model=DEFAULT_MODELS["gemini"],
                    temperature=0
                )
            elif self.llm_type == "qwen":
                return Ollama(
                    model=DEFAULT_MODELS["qwen"],
                    temperature=0
                )
            else:
                raise ValueError(f"Unsupported LLM type: {self.llm_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.llm_type} LLM: {str(e)}")
            raise

    def register_tools(self, tools: dict):
        """Register tools for potential future use."""
        self.tools.update(tools)
        logger.info(f"Registered {len(tools)} tools: {list(tools.keys())}")

    def run(self, input_data: dict) -> str:
        """
        Synthesize the final answer using the selected LLM and outputs from opportunity/risk agents.
        
        Args:
            input_data: dict with outputs from previous agents
            
        Returns:
            str: final answer
        """
        opportunities = input_data.get("opportunity_analysis", "")
        risks = input_data.get("risk_analysis", "")
        question = input_data.get("question")
        synthesis_result = input_data.get("synthesis_result", "")
        
        logger.info(f"MarketManagerAgent processing query: {question}")
        logger.info(f"Opportunities length: {len(opportunities)} characters")
        logger.info(f"Risks length: {len(risks)} characters")
        
        if not opportunities or not risks or not question:
            logger.warning("Missing contexts or question")
            return "No context to synthesize final output"
        
        try:
            # Create the prompt
            user_prompt = MARKET_MANAGER_AGENT_USER_PROMPT.format(
                query=question,
                synthesis_result=synthesis_result,
                opportunity_analysis=opportunities,
                risk_analysis=risks,
                contexts=input_data.get("contexts", [])
            )
            
            # Prepare messages
            messages = [
                SystemMessage(content=MARKET_MANAGER_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            logger.info("Invoking LLM for final synthesis")
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Extract content
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            logger.info(f"Generated final synthesis length: {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error in final synthesis: {e}")
            return f"Error synthesizing final output: {str(e)}"

