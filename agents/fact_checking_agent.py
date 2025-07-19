"""
Fact Checking Agent for InnovARAG System

This agent is responsible for validating the accuracy and consistency of responses
from other agents, flagging potential issues and providing confidence scores.
"""

import json
import logging
from typing import Dict, List, Any

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
    
    def validate_response(
        self, 
        query: str, 
        response: str, 
        sources: List[str],
        contexts: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate a market analysis response for accuracy and consistency.
        
        Args:
            query: The original user query
            response: The market analysis response to validate
            sources: List of available sources for verification
            contexts: Optional list of actual context data for deep verification
            
        Returns:
            Dictionary containing validation results, scores, and recommendations
        """
        logger.info(f"Starting enhanced fact-checking validation for query: {query[:100]}...")
        
        try:
            # If we have actual contexts, perform enhanced validation
            if contexts:
                return self._validate_with_contexts(query, response, sources, contexts)
            else:
                # Fallback to basic validation
                return self._validate_basic(query, response, sources)
            
        except Exception as e:
            logger.error(f"Error during fact-checking validation: {str(e)}")
            return self._create_fallback_result(str(e))

    def _validate_with_contexts(
        self, 
        query: str, 
        response: str, 
        sources: List[str], 
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Enhanced validation using actual source contexts.
        
        Args:
            query: Original user query
            response: Response to validate
            sources: Source names/references
            contexts: Actual context data from tools
            
        Returns:
            Enhanced validation results
        """
        logger.info("Performing enhanced fact-checking with actual source verification")
        
        # Extract and organize source content
        source_content = self._extract_source_content(contexts)
        
        # Perform source-specific fact checking
        source_verification = self._verify_claims_against_sources(response, source_content)
        
        # Check for hallucinations and unsupported claims
        unsupported_claims = self._identify_unsupported_claims(response, source_content)
        
        # Prepare enhanced prompt with actual source data
        enhanced_sources_text = self._format_source_content_for_prompt(source_content)
        
        # Create the enhanced prompt
        user_prompt = FACT_CHECKING_AGENT_USER_PROMPT.format(
            query=query,
            response=response,
            sources=enhanced_sources_text
        )
        
        # Add source verification details
        user_prompt += f"\n\nSOURCE VERIFICATION ANALYSIS:\n{source_verification}"
        
        if unsupported_claims:
            user_prompt += f"\n\nPOTENTIAL UNSUPPORTED CLAIMS:\n" + "\n".join([f"- {claim}" for claim in unsupported_claims])
        
        # Get LLM validation
        messages = [
            SystemMessage(content=FACT_CHECKING_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        logger.debug(f"Sending enhanced validation request to {self.llm_type} LLM")
        
        llm_response = self.llm.invoke(messages)
        content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        
        # Parse and enhance validation result
        validation_result = self._parse_validation_response(content)
        
        # Add source verification details
        validation_result['source_verification'] = {
            'verified_claims': source_verification,
            'unsupported_claims': unsupported_claims,
            'source_coverage': len(source_content),
            'total_contexts': len(contexts)
        }
        
        # Adjust scores based on source verification
        if unsupported_claims:
            validation_result['overall_score'] = max(1, validation_result.get('overall_score', 5) - len(unsupported_claims))
            validation_result['flagged_issues'].extend([f"Unsupported claim: {claim}" for claim in unsupported_claims])
        
        # Add confidence level
        validation_result['confidence_level'] = self._get_confidence_level(validation_result.get('overall_score', 0))
        
        logger.info(f"Enhanced fact-checking complete. Score: {validation_result.get('overall_score', 0)}/10, "
                   f"Unsupported claims: {len(unsupported_claims)}")
        
        return validation_result

    def _validate_basic(self, query: str, response: str, sources: List[str]) -> Dict[str, Any]:
        """
        Basic validation without deep source verification.
        
        Args:
            query: Original user query
            response: Response to validate
            sources: Source names/references
            
        Returns:
            Basic validation results
        """
        logger.info("Performing basic fact-checking validation")
        
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
        
        logger.debug(f"Sending basic validation request to {self.llm_type} LLM")
        
        # Get response from LLM
        llm_response = self.llm.invoke(messages)
        content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        
        logger.debug(f"Raw fact-checking response: {content}")
        
        # Parse validation response
        validation_result = self._parse_validation_response(content)
        
        # Log validation summary
        overall_score = validation_result.get('overall_score', 0)
        flagged_issues = validation_result.get('flagged_issues', [])
        
        logger.info(f"Basic fact-checking complete. Score: {overall_score}/10, Issues: {len(flagged_issues)}")
        
        if flagged_issues:
            logger.warning("Issues identified during fact-checking:")
            for issue in flagged_issues:
                logger.warning(f"  - {issue}")
        
        # Add confidence level based on score
        validation_result['confidence_level'] = self._get_confidence_level(overall_score)
        
        return validation_result

    def _extract_source_content(self, contexts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Extract and organize source content from contexts."""
        source_content = {}
        
        for i, context in enumerate(contexts):
            if isinstance(context, dict):
                tool_name = context.get('tool', f'context_{i}')
                result = context.get('result', context)
                
                source_content[tool_name] = {
                    'content': str(result),
                    'type': tool_name,
                    'length': len(str(result))
                }
        
        logger.debug(f"Extracted content from {len(source_content)} sources")
        return source_content

    def _verify_claims_against_sources(self, response: str, source_content: Dict[str, Dict[str, Any]]) -> str:
        """Verify specific claims against source content."""
        verification_results = []
        
        # Extract potential claims from response (simple heuristic)
        sentences = response.split('.')
        claims_to_verify = [s.strip() for s in sentences if len(s.strip()) > 20 and any(keyword in s.lower() for keyword in ['according to', 'based on', 'shows that', 'indicates', 'reveals'])]
        
        for claim in claims_to_verify[:5]:  # Limit to first 5 claims for performance
            verification = self._verify_single_claim(claim, source_content)
            verification_results.append(f"Claim: '{claim[:100]}...' - {verification}")
        
        return "\n".join(verification_results) if verification_results else "No specific claims identified for verification"

    def _verify_single_claim(self, claim: str, source_content: Dict[str, Dict[str, Any]]) -> str:
        """Verify a single claim against source content."""
        # Simple keyword matching for verification
        claim_lower = claim.lower()
        matches = []
        
        for source_name, content_info in source_content.items():
            content_lower = content_info['content'].lower()
            # Check for keyword overlap
            claim_words = set(claim_lower.split())
            content_words = set(content_lower.split())
            overlap = len(claim_words.intersection(content_words))
            
            if overlap > 3:  # Arbitrary threshold
                matches.append(f"{source_name} (overlap: {overlap} words)")
        
        if matches:
            return f"Supported by: {', '.join(matches)}"
        else:
            return "No clear source support found"

    def _identify_unsupported_claims(self, response: str, source_content: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify claims that appear unsupported by source content."""
        unsupported = []
        
        # Look for specific numeric claims or definitive statements
        sentences = response.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
                
            # Check for definitive statements without clear source support
            if any(keyword in sentence.lower() for keyword in ['will increase by', 'expected to grow', 'market size of', '% of']):
                if not self._has_source_support(sentence, source_content):
                    unsupported.append(sentence)
        
        return unsupported[:3]  # Limit to first 3 for readability

    def _has_source_support(self, claim: str, source_content: Dict[str, Dict[str, Any]]) -> bool:
        """Check if a claim has reasonable source support."""
        claim_lower = claim.lower()
        
        for content_info in source_content.values():
            content_lower = content_info['content'].lower()
            # Simple check for related content
            if any(word in content_lower for word in claim_lower.split() if len(word) > 4):
                return True
        
        return False

    def _format_source_content_for_prompt(self, source_content: Dict[str, Dict[str, Any]]) -> str:
        """Format source content for inclusion in prompts."""
        formatted_sources = []
        
        for source_name, content_info in source_content.items():
            content = content_info['content']
            # Truncate very long content
            if len(content) > 1000:
                content = content[:1000] + "... [TRUNCATED]"
            
            formatted_sources.append(f"SOURCE: {source_name}\nCONTENT: {content}\n")
        
        return "\n".join(formatted_sources)

    def _create_fallback_result(self, error_msg: str) -> Dict[str, Any]:
        """Create fallback validation result on error."""
        return {
            "overall_score": 5,
            "validation_results": {
                "source_attribution": f"Could not validate due to error: {error_msg}",
                "consistency": "Unable to assess",
                "evidence_support": "Unable to assess", 
                "speculation_level": "Unable to assess",
                "completeness": "Unable to assess"
            },
            "flagged_issues": [f"Validation error: {error_msg}"],
            "recommendations": "Manual review recommended due to validation error",
            "confidence_assessment": "Low confidence due to validation failure",
            "confidence_level": "low",
            "error": error_msg
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
    
    # This is defined but not used yet
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
    
    # This is defined but not used yet
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
    
    # This is defined but not used yet
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
    

