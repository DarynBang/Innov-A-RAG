"""
MarketOpportunityAgent: Analyzes market opportunities with comprehensive scoring and source attribution.
Enhanced with proper prompting, confidence scoring, and detailed opportunity assessment.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from agents.base import BaseAgent
from config.prompts import (
    MARKET_OPPORTUNITY_AGENT_SYSTEM_PROMPT,
    MARKET_OPPORTUNITY_AGENT_USER_PROMPT,
    DEFAULT_MODELS
)
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Dict, Any

logger = get_logger(__name__)

class MarketOpportunityAgent(BaseAgent):
    def __init__(self, name="MarketOpportunityAgent", qa_model="openai"):
        super().__init__(name, qa_model)
        logger.info(f"Initializing MarketOpportunityAgent with model: {qa_model}")
        
        self.llm_type = qa_model
        self.llm = self._initialize_llm()
        
        # Initialize tools dictionary for future extensibility
        self.tools = {}
        logger.info("MarketOpportunityAgent initialization completed")
    
    def _initialize_llm(self):
        """Initialize the language model based on the specified type."""
        try:
            if self.llm_type == "openai":
                return ChatOpenAI(
                    model=DEFAULT_MODELS["openai"],
                    temperature=0.1
                )
            elif self.llm_type == "gemini":
                return ChatGoogleGenerativeAI(
                    model=DEFAULT_MODELS["gemini"],
                    temperature=0.1
                )
            elif self.llm_type == "qwen":
                return Ollama(
                    model=DEFAULT_MODELS["qwen"],
                    temperature=0.1
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
        Analyze the context for market opportunities with comprehensive scoring.
        
        Args:
            input_data: dict with context, question, and analysis results
            
        Returns:
            str: detailed opportunities analysis with sources and scores
        """
        # Extract context from various possible sources
        context = self._extract_context(input_data)
        question = input_data.get("question", "")
        
        logger.info(f"MarketOpportunityAgent processing query: {question}")
        logger.info(f"Context length: {len(context)} characters")
        
        if not context or not question:
            logger.warning("Missing context or question")
            return "No context available to generate market opportunities analysis."
        
        try:
            # Create the prompt
            user_prompt = MARKET_OPPORTUNITY_AGENT_USER_PROMPT.format(
                context=context,
                query=question
            )
            
            # Prepare messages
            messages = [
                SystemMessage(content=MARKET_OPPORTUNITY_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            logger.info(f"Invoking {self.llm_type} LLM for opportunity analysis")
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Extract content
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            logger.info(f"Generated opportunities analysis length: {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error in opportunity analysis: {e}")
            return f"Error generating market opportunities analysis: {str(e)}"
    
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
