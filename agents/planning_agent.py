"""
Planning Agent for InnovARAG System

This agent is responsible for analyzing user queries and determining whether they need
to be split into multiple focused subquestions for better processing and analysis.
"""

import json
import logging
from typing import Dict, List, Any

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage, HumanMessage

from config.prompts import (
    PLANNING_AGENT_SYSTEM_PROMPT,
    PLANNING_AGENT_USER_PROMPT,
    DEFAULT_MODELS
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class PlanningAgent:
    """
    Planning Agent that analyzes user queries and breaks them down into subquestions.
    
    This agent determines whether a complex query should be split into multiple
    focused subquestions for better information retrieval and analysis.
    """
    
    def __init__(self, llm_type: str = "openai"):
        """
        Initialize the Planning Agent.
        
        Args:
            llm_type: Type of LLM to use ("openai", "gemini", or "qwen")
        """
        self.llm_type = llm_type
        self.llm = self._initialize_llm()
        
        logger.info(f"PlanningAgent initialized with {llm_type} LLM")
    
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
    
    def plan_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a user query and determine if it needs to be split into subquestions.
        
        Args:
            query: The original user query to analyze
            
        Returns:
            Dictionary containing analysis results, whether splitting is needed,
            and the list of subquestions (or original query if no splitting needed)
        """
        logger.info(f"Planning query analysis for: {query[:100]}...")
        
        try:
            # Create the prompt
            user_prompt = PLANNING_AGENT_USER_PROMPT.format(query=query)
            
            # Prepare messages
            messages = [
                SystemMessage(content=PLANNING_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            logger.debug(f"Sending query to {self.llm_type} LLM for planning analysis")
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Extract content based on LLM type
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            logger.debug(f"Raw LLM response: {content}")
            
            # Parse JSON response
            planning_result = self._parse_planning_response(content)
            
            # Log the results
            needs_splitting = planning_result.get('needs_splitting', False)
            num_subquestions = len(planning_result.get('subquestions', []))
            
            logger.info(
                f"Planning analysis complete. Needs splitting: {needs_splitting}, "
                f"Subquestions: {num_subquestions}"
            )
            
            if needs_splitting:
                logger.info("Query will be split into subquestions:")
                for i, subq in enumerate(planning_result.get('subquestions', []), 1):
                    logger.info(f"  {i}. {subq}")
            else:
                self.logger.info("Query will be processed as a single question")
            
            return planning_result
            
        except Exception as e:
            logger.error(f"Error during query planning: {str(e)}")
            # Return fallback result
            return {
                "analysis": f"Error during planning analysis: {str(e)}",
                "needs_splitting": False,
                "subquestions": [query],
                "error": str(e)
            }
    
    def _parse_planning_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the JSON response from the planning agent.
        
        Args:
            response: Raw response string from the LLM
            
        Returns:
            Parsed planning result dictionary
        """
        try:
            # Clean the response - remove markdown code blocks if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON
            result = json.loads(cleaned_response)
            
            # Validate required fields
            required_fields = ['analysis', 'needs_splitting', 'subquestions']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure subquestions is a list
            if not isinstance(result['subquestions'], list):
                raise ValueError("subquestions must be a list")
            
            # Ensure needs_splitting is boolean
            if not isinstance(result['needs_splitting'], bool):
                result['needs_splitting'] = bool(result['needs_splitting'])
            
            logger.debug("Successfully parsed planning response")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response}")
            raise ValueError(f"Invalid JSON response from planning agent: {e}")
        
        except Exception as e:
            logger.error(f"Error parsing planning response: {e}")
            raise
    
    # This is defined but not used yet
    def get_subquestions(self, query: str) -> List[str]:
        """
        Get the list of subquestions for a given query.
        
        This is a convenience method that returns just the subquestions list.
        
        Args:
            query: The original user query
            
        Returns:
            List of subquestions (or single original query if no splitting needed)
        """
        try:
            planning_result = self.plan_query(query)
            return planning_result.get('subquestions', [query])
        
        except Exception as e:
            logger.error(f"Error getting subquestions: {e}")
            return [query]  # Return original query as fallback
    
    # This is defined but not used yet
    def should_split_query(self, query: str) -> bool:
        """
        Determine if a query should be split into subquestions.
        
        Args:
            query: The original user query
            
        Returns:
            Boolean indicating whether the query should be split
        """
        try:
            planning_result = self.plan_query(query)
            return planning_result.get('needs_splitting', False)
        
        except Exception as e:
            logger.error(f"Error determining if query should be split: {e}")
            return False  # Conservative fallback - don't split if unsure 
        
        
