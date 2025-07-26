"""
MarketRiskAgent: Analyzes market risks from the summary using the selected LLM.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from agents.base import BaseAgent
from config.prompts import (
    MARKET_RISK_AGENT_SYSTEM_PROMPT,
    MARKET_RISK_AGENT_USER_PROMPT,
    DEFAULT_MODELS
)
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = get_logger(__name__)

class MarketRiskAgent(BaseAgent):
    def __init__(self, name="MarketRiskAgent", qa_model="openai"):
        super().__init__(name, qa_model)
        logger.info(f"Initializing MarketRiskAgent with model: {qa_model}")
        
        self.llm_type = qa_model
        self.llm = self._initialize_llm()
        
        # Initialize tools dictionary for future extensibility
        self.tools = {}
        logger.info("MarketRiskAgent initialization completed")
    
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

    def run(self, input_data: dict) -> str:
        """
        Analyze the context for market risks using the selected LLM.
        
        Args:
            input_data: dict with context and question
            
        Returns:
            str: risks summary
        """
        context = self._extract_context(input_data)
        question = input_data.get("question", "")
        
        logger.info(f"MarketRiskAgent processing query: {question}")
        logger.info(f"Context length: {len(context)} characters")
        
        if not context or not question:
            logger.warning("Missing context or question")
            return "No context to assess market risks."
        
        try:
            # Create the prompt
            user_prompt = MARKET_RISK_AGENT_USER_PROMPT.format(
                context=context,
                query=question
            )
            
            # Prepare messages
            messages = [
                SystemMessage(content=MARKET_RISK_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            logger.info("Invoking LLM for risk analysis")
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Extract content
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            logger.info(f"Generated risk analysis length: {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {e}")
            return f"Error assessing market risks: {str(e)}"
    
    def _extract_context(self, input_data: dict) -> str:
        """
        Extract and format context from various input sources.
        
        Args:
            input_data: Dictionary containing various context sources
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Check for synthesis result
        if "synthesis_result" in input_data:
            context_parts.append(f"Synthesis Result:\n{input_data['synthesis_result']}")
        
        # Check for generalize agent output (legacy)
        if "GeneralizeAgent" in input_data:
            context_parts.append(f"Summary:\n{input_data['GeneralizeAgent']}")
        
        # Check for contexts array
        if "contexts" in input_data:
            contexts = input_data["contexts"]
            if contexts:
                context_parts.append("Retrieved Contexts:")
                for i, ctx in enumerate(contexts[:3], 1):  # Limit to first 3 contexts
                    tool = ctx.get('tool', 'unknown')
                    result = ctx.get('result', '')
                    if isinstance(result, dict):
                        result = str(result)
                    context_parts.append(f"Context {i} (Tool: {tool}):\n{result}")
        
        return "\n\n".join(context_parts) if context_parts else "No context available"
