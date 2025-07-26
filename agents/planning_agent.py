"""
Planning Agent for InnovARAG System

This agent is responsible for analyzing user queries and determining whether they need
to be split into multiple focused subquestions for better processing and analysis.
It also determines if market analysis team should be involved.
"""

import json
import logging
from typing import Dict, List, Any

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage, HumanMessage

from config.prompts import (
    DEFAULT_MODELS,
    PLANNING_AGENT_SYSTEM_PROMPT,
    PLANNING_AGENT_USER_PROMPT
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class PlanningAgent:
    """
    Planning Agent that analyzes user queries and determines the processing workflow.
    
    This agent determines whether a complex query should be split into multiple
    focused subquestions and whether market analysis team should be involved.
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
    
    def plan_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a user query and determine the processing workflow.
        
        Args:
            query: The original user query to analyze
            
        Returns:
            Dictionary containing analysis results, whether splitting is needed,
            whether analysis team is needed, and the list of subquestions
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
            needs_analysis_team = planning_result.get('needs_analysis_team', False)
            num_subquestions = len(planning_result.get('subquestions', []))
            
            logger.info(
                f"Planning analysis complete. Needs splitting: {needs_splitting}, "
                f"Needs analysis team: {needs_analysis_team}, "
                f"Subquestions: {num_subquestions}"
            )
            
            return planning_result
            
        except Exception as e:
            logger.error(f"Error during planning analysis: {str(e)}")
            # Return fallback result
            return {
                "analysis": f"Error during planning: {str(e)}",
                "needs_splitting": False,
                "needs_analysis_team": False,
                "analysis_reasoning": "Error occurred, defaulting to basic processing",
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
            required_fields = ['analysis', 'needs_splitting', 'needs_analysis_team', 'analysis_reasoning', 'subquestions']
            for field in required_fields:
                if field not in result:
                    if field == 'needs_analysis_team':
                        result[field] = False  # Default to not needing analysis team
                    elif field == 'analysis_reasoning':
                        result[field] = "No reasoning provided"
                    elif field == 'analysis':
                        result[field] = "Query analyzed"
                    elif field == 'needs_splitting':
                        result[field] = False
                    elif field == 'subquestions':
                        result[field] = [result.get('query', 'Unknown query')]
            
            # Ensure subquestions is a list
            if not isinstance(result['subquestions'], list):
                result['subquestions'] = [str(result['subquestions'])]
            
            # Ensure boolean fields are boolean
            result['needs_splitting'] = bool(result.get('needs_splitting', False))
            result['needs_analysis_team'] = bool(result.get('needs_analysis_team', False))
            
            logger.debug("Successfully parsed planning response")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response}")
            raise ValueError(f"Invalid JSON response from planning agent: {e}")
        
        except Exception as e:
            logger.error(f"Error parsing planning response: {e}")
            raise 
        
        
