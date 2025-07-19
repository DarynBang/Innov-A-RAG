"""
GeneralizeAgent: Synthesizes information from multiple sources and subquestions to provide comprehensive answers.
Enhanced with proper prompting, source attribution, and confidence scoring.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from agents.base import BaseAgent
from text_generation.base_runner import get_text_captioning_runner
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage, HumanMessage
from config.prompts import (
    GENERALIZE_AGENT_SYSTEM_PROMPT,
    GENERALIZE_AGENT_USER_PROMPT,
    DEFAULT_MODELS
)
from typing import Dict, List, Any, Optional

logger = get_logger(__name__)

class GeneralizeAgent(BaseAgent):
    def __init__(self, name="GeneralizeAgent", qa_model="openai"):
        super().__init__(name, qa_model)
        logger.info(f"Initializing GeneralizeAgent with backend model: {qa_model}")
        
        self.llm_type = qa_model
        try:
            self.llm = self._initialize_llm()
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
        
        # Initialize tools dictionary for future extensibility
        self.tools = {}
        logger.info("GeneralizeAgent initialization completed")
    
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

    def synthesize_information(
        self, 
        original_query: str, 
        subquestions: List[str], 
        contexts: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesize information from multiple sources to answer user queries comprehensively.
        
        Args:
            original_query: The original user query
            subquestions: List of subquestions derived from the original query
            contexts: List of context dictionaries with retrieval results
            
        Returns:
            Comprehensive answer with source attribution
        """
        logger.info(f"Synthesizing information for query: {original_query}")
        logger.info(f"Processing {len(subquestions)} subquestions and {len(contexts)} contexts")
        
        try:
            # Format contexts for the prompt
            contexts_text = self._format_contexts(contexts)
            subquestions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(subquestions)])
            
            # Create the prompt
            user_prompt = GENERALIZE_AGENT_USER_PROMPT.format(
                original_query=original_query,
                subquestions=subquestions_text,
                contexts=contexts_text
            )
            
            # Prepare messages
            messages = [
                SystemMessage(content=GENERALIZE_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            logger.debug(f"Sending synthesis request to {self.llm_type} LLM")
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Extract content
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            logger.info(f"Generated synthesis response length: {len(content)} characters")
            return content
            
        except Exception as e:
            logger.error(f"Error during information synthesis: {e}")
            return f"Error synthesizing information: {str(e)}"
    
    def _format_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Format contexts for inclusion in the prompt.
        
        Args:
            contexts: List of context dictionaries
            
        Returns:
            Formatted context string
        """
        formatted_contexts = []
        
        for i, context in enumerate(contexts, 1):
            tool_name = context.get('tool', 'unknown')
            result = context.get('result', '')
            
            if isinstance(result, dict):
                # Handle structured results
                if 'chunks' in result:
                    chunks = result['chunks']
                    sources = []
                    content_parts = []
                    
                    for chunk in chunks:
                        if isinstance(chunk, dict):
                            content_parts.append(chunk.get('chunk', ''))
                            if 'source' in chunk:
                                sources.append(chunk['source'])
                    
                    content = '\n'.join(content_parts)
                    source_info = f"Sources: {', '.join(set(sources))}" if sources else ""
                else:
                    content = str(result)
                    source_info = ""
            else:
                content = str(result)
                source_info = ""
            
            formatted_context = f"Context {i} (Tool: {tool_name}):\n{content}"
            if source_info:
                formatted_context += f"\n{source_info}"
            
            formatted_contexts.append(formatted_context)
        
        return "\n\n".join(formatted_contexts)

    def run(self, input_data: dict, patent_contexts=None, firm_summary_contexts=None) -> str:
        """
        Backward compatibility method - summarize merged contexts using the selected LLM.
        
        Args:
            input_data: dict with question and other info
            patent_contexts: list of patent context dicts
            firm_summary_contexts: list of company context dicts
            
        Returns:
            str: summary string
        """
        question = input_data.get("question", "")
        logger.info(f"GeneralizeAgent processing query: {question}")
        
        if not question:
            logger.warning("No question provided to GeneralizeAgent")
            return "No question provided for processing."
        
        # Convert legacy contexts to new format
        contexts = []
        
        if firm_summary_contexts:
            contexts.append({
                "tool": "company_rag_retrieval",
                "result": {"chunks": firm_summary_contexts}
            })
            logger.info(f"Processing {len(firm_summary_contexts)} firm summary chunks")
        
        if patent_contexts:
            contexts.append({
                "tool": "patent_rag_retrieval", 
                "result": {"chunks": patent_contexts}
            })
            logger.info(f"Processing {len(patent_contexts)} patent chunks")
        
        # Use new synthesis method
        return self.synthesize_information(question, [question], contexts)


