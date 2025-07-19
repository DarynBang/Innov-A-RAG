"""
Fact Checking Agent for InnovARAG System

This agent is responsible for validating the accuracy and consistency of responses
from other agents, flagging potential issues and providing confidence scores.
"""

import json
import logging
from typing import Dict, List, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage, HumanMessage

from config.prompts import (
    FACT_CHECKING_AGENT_SYSTEM_PROMPT,
    FACT_CHECKING_AGENT_USER_PROMPT,
    DEFAULT_MODELS,
    CONFIDENCE_THRESHOLDS
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class FactCheckingAgent:
    """
    Fact Checking Agent that validates response accuracy and flags potential issues.
    
    This agent performs comprehensive validation of market analysis responses,
    checking for source attribution, consistency, evidence support, and potential
    hallucinations or unsupported claims.
    """
    
    def __init__(self, llm_type: str = "openai"):
        """
        Initialize the Fact Checking Agent.
        
        Args:
            llm_type: Type of LLM to use ("openai", "gemini", or "qwen")
        """
        self.llm_type = llm_type
        self.llm = self._initialize_llm()
        
        logger.info(f"FactCheckingAgent initialized with {llm_type} LLM")
    
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
    
    def validate_response(
        self, 
        query: str, 
        response: str, 
        sources: List[str]
    ) -> Dict[str, Any]:
        """
        Validate a market analysis response for accuracy and consistency.
        
        Args:
            query: The original user query
            response: The market analysis response to validate
            sources: List of available sources for verification
            
        Returns:
            Dictionary containing validation results, scores, and recommendations
        """
        logger.info(f"Starting fact-checking validation for query: {query[:100]}...")
        
        try:
            # Prepare sources string
            sources_text = "\n".join([f"- {source}" for source in sources])
            
            # Create the prompt
            user_prompt = FACT_CHECKING_AGENT_USER_PROMPT.format(
                query=query,
                response=response,
                sources=sources_text
            )
            
            # Prepare messages
            messages = [
                SystemMessage(content=FACT_CHECKING_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            logger.debug(f"Sending validation request to {self.llm_type} LLM")
            
            # Get response from LLM
            llm_response = self.llm.invoke(messages)
            
            # Extract content based on LLM type
            if hasattr(llm_response, 'content'):
                content = llm_response.content
            else:
                content = str(llm_response)
            
            logger.debug(f"Raw fact-checking response: {content}")
            
            # Parse validation response
            validation_result = self._parse_validation_response(content)
            
            # Log validation summary
            overall_score = validation_result.get('overall_score', 0)
            flagged_issues = validation_result.get('flagged_issues', [])
            
            logger.info(
                f"Fact-checking complete. Overall score: {overall_score}/10, "
                f"Issues flagged: {len(flagged_issues)}"
            )
            
            if flagged_issues:
                logger.warning("Issues identified during fact-checking:")
                for issue in flagged_issues:
                    logger.warning(f"  - {issue}")
            
            # Add confidence level based on score
            validation_result['confidence_level'] = self._get_confidence_level(overall_score)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error during fact-checking validation: {str(e)}")
            # Return fallback validation result
            return {
                "overall_score": 5,
                "validation_results": {
                    "source_attribution": f"Could not validate due to error: {str(e)}",
                    "consistency": "Unable to assess",
                    "evidence_support": "Unable to assess", 
                    "speculation_level": "Unable to assess",
                    "completeness": "Unable to assess"
                },
                "flagged_issues": [f"Validation error: {str(e)}"],
                "recommendations": "Manual review recommended due to validation error",
                "confidence_assessment": "Low confidence due to validation failure",
                "confidence_level": "low",
                "error": str(e)
            }
    
    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the JSON response from the fact-checking agent.
        
        Args:
            response: Raw response string from the LLM
            
        Returns:
            Parsed validation result dictionary
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
            required_fields = [
                'overall_score', 'validation_results', 'flagged_issues',
                'recommendations', 'confidence_assessment'
            ]
            for field in required_fields:
                if field not in result:
                    logger.warning(f"Missing field in validation response: {field}")
            
            # Ensure overall_score is valid
            if 'overall_score' in result:
                try:
                    result['overall_score'] = max(1, min(10, int(result['overall_score'])))
                except (ValueError, TypeError):
                    result['overall_score'] = 5
            else:
                result['overall_score'] = 5
            
            # Ensure flagged_issues is a list
            if 'flagged_issues' not in result or not isinstance(result['flagged_issues'], list):
                result['flagged_issues'] = []
            
            logger.debug("Successfully parsed fact-checking response")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response}")
            raise ValueError(f"Invalid JSON response from fact-checking agent: {e}")
        
        except Exception as e:
            logger.error(f"Error parsing fact-checking response: {e}")
            raise
    
    def _get_confidence_level(self, score: int) -> str:
        """
        Convert numerical score to confidence level.
        
        Args:
            score: Numerical score from 1-10
            
        Returns:
            Confidence level string ("high", "medium", "low")
        """
        if score >= CONFIDENCE_THRESHOLDS["high"]:
            return "high"
        elif score >= CONFIDENCE_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "low"
    
    def is_response_reliable(self, validation_result: Dict[str, Any]) -> bool:
        """
        Determine if a response is reliable based on validation results.
        
        Args:
            validation_result: Result from validate_response method
            
        Returns:
            Boolean indicating if the response is considered reliable
        """
        try:
            overall_score = validation_result.get('overall_score', 0)
            flagged_issues = validation_result.get('flagged_issues', [])
            
            # Consider reliable if score is medium or above and no critical issues
            is_reliable = (
                overall_score >= CONFIDENCE_THRESHOLDS["medium"] and
                len(flagged_issues) <= 2  # Allow minor issues
            )
            
            logger.debug(
                f"Reliability assessment: {is_reliable} "
                f"(score: {overall_score}, issues: {len(flagged_issues)})"
            )
            
            return is_reliable
            
        except Exception as e:
            logger.error(f"Error assessing response reliability: {e}")
            return False  # Conservative approach
    
    def get_validation_summary(self, validation_result: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of validation results.
        
        Args:
            validation_result: Result from validate_response method
            
        Returns:
            String summary of validation results
        """
        try:
            overall_score = validation_result.get('overall_score', 0)
            confidence_level = validation_result.get('confidence_level', 'unknown')
            flagged_issues = validation_result.get('flagged_issues', [])
            
            summary = f"Validation Score: {overall_score}/10 ({confidence_level.upper()} confidence)\n"
            
            if flagged_issues:
                summary += f"Issues Identified ({len(flagged_issues)}):\n"
                for i, issue in enumerate(flagged_issues, 1):
                    summary += f"  {i}. {issue}\n"
            else:
                summary += "No significant issues identified.\n"
            
            recommendations = validation_result.get('recommendations', '')
            if recommendations:
                summary += f"\nRecommendations: {recommendations}"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating validation summary: {e}")
            return f"Validation summary unavailable due to error: {e}"
    
    def validate_multiple_responses(
        self, 
        query: str, 
        responses: Dict[str, str], 
        sources: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Validate multiple responses for comparison.
        
        Args:
            query: The original user query
            responses: Dictionary of response_name -> response_text
            sources: List of available sources for verification
            
        Returns:
            Dictionary of response_name -> validation_result
        """
        logger.info(f"Validating {len(responses)} responses for comparison")
        
        validation_results = {}
        
        for response_name, response_text in responses.items():
            try:
                logger.debug(f"Validating response: {response_name}")
                validation_result = self.validate_response(query, response_text, sources)
                validation_results[response_name] = validation_result
                
            except Exception as e:
                logger.error(f"Error validating response {response_name}: {e}")
                validation_results[response_name] = {
                    "overall_score": 0,
                    "error": str(e),
                    "confidence_level": "low"
                }
        
        return validation_results 