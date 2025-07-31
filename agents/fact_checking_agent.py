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
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from config.prompts import (
    FACT_CHECKING_AGENT_SYSTEM_PROMPT,
    FACT_CHECKING_AGENT_USER_PROMPT,
    PRODUCT_SUGGESTION_FACT_CHECK_SYSTEM_PROMPT,
    PRODUCT_SUGGESTION_FACT_CHECK_USER_PROMPT,
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
            elif self.llm_type == "ollama":
                return ChatOllama(
                    model=DEFAULT_MODELS["ollama"],
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
        contexts: List[Dict[str, Any]] = None,
        validation_mode: str = "market_analysis",
        production_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Validate a response for accuracy and consistency.
        
        Args:
            query: The original user query
            response: The response to validate
            sources: List of available sources for verification
            contexts: Optional list of actual context data for deep verification
            validation_mode: "market_analysis" or "product_suggestion"
            
        Returns:
            Dictionary containing validation results, scores, and recommendations
        """
        logger.info(f"Starting {validation_mode} fact-checking validation for query: {query[:100]}...")
        
        try:
            if validation_mode == "product_suggestion":
                return self.validate_product_suggestions(query, response, sources, contexts, production_mode=production_mode)
            else:
                # Default to market analysis validation
                if contexts:
                    return self._validate_with_contexts(query, response, sources, contexts)
                else:
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
    
    def _create_production_fallback_result(self, error_msg: str) -> Dict[str, Any]:
        """Create fallback validation result for production mode on error."""
        return {
            "overall_score": 5.0,
            "confidence_level": "low",
            "production_criteria": {
                "robustness": {"score": 5, "issues": [f"Validation error: {error_msg}"]},
                "standardization": {"score": 5, "issues": ["Unable to assess"]},
                "detail_level": {"score": 5, "issues": ["Unable to assess"]},
                "citation_quality": {"score": 5, "issues": ["Unable to assess"]}
            },
            "flagged_issues": [f"Validation error: {error_msg}"],
            "recommendations": ["Manual review recommended due to validation error"],
            "validation_mode": "production",
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
    
    def _parse_production_validation_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the JSON response from the production mode fact-checking validation.
        Expected format: {overall_score, confidence_level, production_criteria, flagged_issues, recommendations}
        
        Args:
            response: Raw response string from LLM
            
        Returns:
            Parsed validation result dictionary
        """
        try:
            # Clean response - remove markdown code blocks if present
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
            
            # Validate required fields for production mode
            required_fields = [
                'overall_score', 'confidence_level', 'production_criteria',
                'flagged_issues', 'recommendations'
            ]
            for field in required_fields:
                if field not in result:
                    logger.warning(f"Missing field in production validation response: {field}")
            
            # Ensure overall_score is valid
            if 'overall_score' in result:
                try:
                    result['overall_score'] = max(1, min(10, float(result['overall_score'])))
                except (ValueError, TypeError):
                    result['overall_score'] = 5.0
            else:
                result['overall_score'] = 5.0
            
            # Ensure flagged_issues is a list
            if 'flagged_issues' not in result or not isinstance(result['flagged_issues'], list):
                result['flagged_issues'] = []
            
            # Ensure recommendations is a list
            if 'recommendations' not in result or not isinstance(result['recommendations'], list):
                result['recommendations'] = []
            
            # Ensure production_criteria exists and has the expected structure
            if 'production_criteria' not in result or not isinstance(result['production_criteria'], dict):
                result['production_criteria'] = {
                    'robustness': {'score': 5, 'issues': []},
                    'standardization': {'score': 5, 'issues': []},
                    'detail_level': {'score': 5, 'issues': []},
                    'citation_quality': {'score': 5, 'issues': []}
                }
            
            logger.debug("Successfully parsed production validation response")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response}")
            raise ValueError(f"Invalid JSON response from production fact-checking agent: {e}")
        
        except Exception as e:
            logger.error(f"Error parsing production validation response: {e}")
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
    
    def validate_product_suggestions(
        self, 
        query: str, 
        response: str, 
        sources: List[str],
        contexts: List[Dict[str, Any]] = None,
        production_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Validate product suggestions for accuracy and proper citation.
        
        Args:
            query: The original user query
            response: The product suggestions response to validate
            sources: List of available sources for verification
            contexts: Optional list of actual context data for verification
            
        Returns:
            Dictionary containing validation results specific to product suggestions
        """
        logger.info(f"Starting product suggestion validation for query: {query[:100]}...")
        
        try:
            if production_mode:
                return self._validate_production_mode(query, response, sources, contexts)
            else:
                # Enhanced validation with contexts if available
                if contexts:
                    return self._validate_product_suggestions_with_contexts(query, response, sources, contexts)
                else:
                    return self._validate_product_suggestions_basic(query, response, sources)
            
        except Exception as e:
            logger.error(f"Error during product suggestion validation: {str(e)}")
            if production_mode:
                return self._create_production_fallback_result(str(e))
            else:
                return self._create_fallback_result(str(e))
    
    def _validate_production_mode(
        self, 
        query: str, 
        response: str, 
        sources: List[str],
        contexts: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Production mode validation: Check if answers are robust, standard, detailed and properly cited.
        
        Args:
            query: Original user query
            response: Product suggestions response to validate
            sources: Source names/references
            contexts: Actual context data from tools
            
        Returns:
            Production mode validation results focusing on robustness, standardization, detail, and citations
        """
        logger.info("Performing production mode validation focusing on robustness, standardization, detail, and citations")
        
        try:
            # Production mode validation criteria
            validation_criteria = {
                'robustness': self._check_robustness(response),
                'standardization': self._check_standardization(response),
                'detail_level': self._check_detail_level(response),
                'citation_quality': self._check_citation_quality(response, sources, contexts)
            }
            
            # Calculate overall score based on production criteria
            scores = [criteria['score'] for criteria in validation_criteria.values()]
            overall_score = sum(scores) / len(scores) if scores else 0
            
            # Collect all issues
            flagged_issues = []
            for criterion, result in validation_criteria.items():
                flagged_issues.extend(result.get('issues', []))
            
            # Use the proper production fact-checking prompt
            try:
                from config.prompts import PRODUCTION_FACT_CHECK_PROMPT
                
                # Format sources for validation
                sources_text = "\n".join(sources) if sources else "No sources provided"
                
                messages = [
                    SystemMessage(content="You are a validation specialist for production mode."),
                    HumanMessage(content=PRODUCTION_FACT_CHECK_PROMPT.format(
                        query=query,
                        response=response,
                        sources=sources_text
                    ))
                ]
            except ImportError:
                # Fallback to custom prompt if import fails
                production_prompt = f"""
You are validating a product suggestion response in PRODUCTION MODE. Focus on these critical criteria:

1. ROBUSTNESS: Is the response well-structured and comprehensive?
2. STANDARDIZATION: Does it follow a consistent, professional format?
3. DETAIL: Are the suggestions sufficiently detailed and informative?
4. CITATIONS: Are sources properly cited and verifiable?

Query: {query}

Response to validate:
{response}

Available sources: {sources}

Provide a JSON validation result focusing on production readiness.
                """
                
                messages = [
                    SystemMessage(content="You are a production-ready response validator focusing on robustness, standardization, detail level, and proper citations."),
                    HumanMessage(content=production_prompt)
                ]
            
            logger.debug(f"Sending production mode validation request to {self.llm_type} LLM")
            
            llm_response = self.llm.invoke(messages)
            content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
            
            # Parse LLM validation result
            llm_validation = self._parse_production_validation_response(content)
            
            # Combine automated checks with LLM validation
            final_result = {
                'overall_score': min(overall_score, llm_validation.get('overall_score', 10)),
                'confidence_level': self._get_confidence_level(overall_score),
                'flagged_issues': flagged_issues + llm_validation.get('flagged_issues', []),
                'recommendations': llm_validation.get('recommendations', []),
                'production_criteria': validation_criteria,
                'validation_mode': 'production',
                'source_verification': {
                    'available_sources': len(sources),
                    'contexts_analyzed': len(contexts) if contexts else 0
                }
            }
            
            logger.info(f"Production mode validation complete. Score: {final_result['overall_score']:.1f}/10, Issues: {len(final_result['flagged_issues'])}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in production mode validation: {str(e)}")
            return self._create_production_fallback_result(str(e))
    
    def _check_robustness(self, response: str) -> Dict[str, Any]:
        """
        Check if the response is robust and well-structured.
        
        Args:
            response: Response to check
            
        Returns:
            Dictionary with robustness score and issues
        """
        issues = []
        score = 10
        
        # Check response length (should be substantial)
        if len(response) < 100:
            issues.append("Response too short - lacks robustness")
            score -= 3
        
        # Check for structured format (looking for product suggestions format)
        if not any(pattern in response.lower() for pattern in ['product', 'suggestion', 'technology', 'innovation']):
            issues.append("Response lacks clear product/technology focus")
            score -= 2
        
        # Check for logical structure
        if response.count('.') < 3:  # Very basic check for sentence structure
            issues.append("Response lacks detailed structure")
            score -= 2
        
        return {'score': max(0, score), 'issues': issues}
    
    def _check_standardization(self, response: str) -> Dict[str, Any]:
        """
        Check if the response follows a standard format.
        
        Args:
            response: Response to check
            
        Returns:
            Dictionary with standardization score and issues
        """
        issues = []
        score = 10
        
        # Check for consistent formatting patterns
        expected_patterns = [':', '(', ')', ';']  # Basic formatting elements
        pattern_count = sum(1 for pattern in expected_patterns if pattern in response)
        
        if pattern_count < 2:
            issues.append("Response lacks consistent formatting patterns")
            score -= 3
        
        # Check for proper capitalization (basic check)
        sentences = response.split('.')
        uncapitalized = sum(1 for sentence in sentences if sentence.strip() and not sentence.strip()[0].isupper())
        
        if uncapitalized > len(sentences) * 0.3:  # More than 30% uncapitalized
            issues.append("Response has inconsistent capitalization")
            score -= 2
        
        return {'score': max(0, score), 'issues': issues}
    
    def _check_detail_level(self, response: str) -> Dict[str, Any]:
        """
        Check if the response has sufficient detail.
        
        Args:
            response: Response to check
            
        Returns:
            Dictionary with detail level score and issues
        """
        issues = []
        score = 10
        
        # Check word count (should be detailed)
        word_count = len(response.split())
        if word_count < 50:
            issues.append("Response lacks sufficient detail (too few words)")
            score -= 4
        elif word_count < 100:
            issues.append("Response could be more detailed")
            score -= 2
        
        # Check for descriptive elements
        descriptive_words = ['description', 'details', 'features', 'capabilities', 'benefits', 'applications']
        descriptive_count = sum(1 for word in descriptive_words if word.lower() in response.lower())
        
        if descriptive_count == 0:
            issues.append("Response lacks descriptive details")
            score -= 3
        
        return {'score': max(0, score), 'issues': issues}
    
    def _check_citation_quality(self, response: str, sources: List[str], contexts: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check if the response has proper citations.
        
        Args:
            response: Response to check
            sources: Available sources
            contexts: Context data
            
        Returns:
            Dictionary with citation quality score and issues
        """
        issues = []
        score = 10
        
        # Check for citation patterns
        citation_patterns = ['(Source:', '(Company', '(Patent', 'Reason:', 'source:', 'company', 'patent']
        citation_count = sum(1 for pattern in citation_patterns if pattern.lower() in response.lower())
        
        if citation_count == 0:
            issues.append("Response lacks proper source citations")
            score -= 5
        elif citation_count < 2:
            issues.append("Response has insufficient citations")
            score -= 2
        
        # Check if sources are referenced
        if sources:
            referenced_sources = sum(1 for source in sources if source.lower() in response.lower())
            if referenced_sources == 0:
                issues.append("Response doesn't reference available sources")
                score -= 3
        
        # Check for explanation patterns (Reason:, because, etc.)
        explanation_patterns = ['reason:', 'because', 'since', 'due to', 'based on']
        explanation_count = sum(1 for pattern in explanation_patterns if pattern.lower() in response.lower())
        
        if explanation_count == 0:
            issues.append("Response lacks explanations for suggestions")
            score -= 2
        
        return {'score': max(0, score), 'issues': issues}

    def _validate_product_suggestions_with_contexts(
        self, 
        query: str, 
        response: str, 
        sources: List[str], 
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Enhanced validation for product suggestions using actual source contexts.
        
        Args:
            query: Original user query
            response: Product suggestions response to validate
            sources: Source names/references
            contexts: Actual context data from tools
            
        Returns:
            Enhanced validation results for product suggestions
        """
        logger.info("Performing enhanced product suggestion validation with source verification")
        
        # Extract and organize source content
        source_content = self._extract_source_content(contexts)
        
        # Verify product suggestions against sources
        citation_verification = self._verify_product_citations(response, source_content)
        
        # Check for unsupported product claims
        unsupported_products = self._identify_unsupported_products(response, source_content)
        
        # Prepare enhanced prompt with actual source data
        enhanced_sources_text = self._format_source_content_for_prompt(source_content)
        
        # Create the enhanced prompt for product suggestions
        user_prompt = PRODUCT_SUGGESTION_FACT_CHECK_USER_PROMPT.format(
            query=query,
            response=response,
            sources=enhanced_sources_text
        )
        
        # Add verification details
        user_prompt += f"\n\nCITATION VERIFICATION ANALYSIS:\n{citation_verification}"
        
        if unsupported_products:
            user_prompt += f"\n\nPOTENTIAL UNSUPPORTED PRODUCTS:\n" + "\n".join([f"- {product}" for product in unsupported_products])
        
        # Get LLM validation
        messages = [
            SystemMessage(content=PRODUCT_SUGGESTION_FACT_CHECK_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        logger.debug(f"Sending enhanced product suggestion validation request to {self.llm_type} LLM")
        
        llm_response = self.llm.invoke(messages)
        content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        
        # Parse and enhance validation result
        validation_result = self._parse_product_suggestion_validation_response(content)
        
        # Add verification details
        validation_result['citation_verification'] = {
            'verified_citations': citation_verification,
            'unsupported_products': unsupported_products,
            'source_coverage': len(source_content),
            'total_contexts': len(contexts)
        }
        
        # Adjust scores based on citation verification
        if unsupported_products:
            validation_result['overall_score'] = max(1, validation_result.get('overall_score', 5) - len(unsupported_products))
            validation_result['flagged_issues'].extend([f"Unsupported product: {product}" for product in unsupported_products])
        
        # Add confidence level
        validation_result['confidence_level'] = self._get_confidence_level(validation_result.get('overall_score', 0))
        
        logger.info(f"Enhanced product suggestion validation complete. Score: {validation_result.get('overall_score', 0)}/10, "
                   f"Unsupported products: {len(unsupported_products)}")
        
        return validation_result

    def _validate_product_suggestions_production(self, query: str, response: str, sources: List[str], contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Production mode validation: Check if answer is robust, standard, detailed, and properly cited using COT.
        
        Args:
            query: Original user query
            response: Generated product suggestions
            sources: List of source references
            contexts: Retrieved contexts for validation
            
        Returns:
            Detailed validation results with production criteria as JSON
        """
        logger.info("Running production mode validation with COT and JSON output")
        
        try:
            from config.prompts import PRODUCTION_FACT_CHECK_PROMPT
            
            # Format sources for validation
            sources_text = "\n".join(sources) if sources else "No sources provided"
            
            messages = [
                SystemMessage(content="You are a validation specialist for production mode."),
                HumanMessage(content=PRODUCTION_FACT_CHECK_PROMPT.format(
                    query=query,
                    response=response,
                    sources=sources_text
                ))
            ]
            
            llm_response = self.llm.invoke(messages)
            
            # Parse JSON response
            
            try:
                # Clean the response - remove markdown code blocks if present
                cleaned_response = llm_response.content.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                
                cleaned_response = cleaned_response.strip()
                
                # Additional cleaning for common LLM issues
                if cleaned_response == '"overall_score"' or cleaned_response.strip() == '"overall_score"':
                    # LLM returned exactly just the field name in quotes
                    logger.warning(f"LLM returned just field name: {cleaned_response}")
                    return self._validate_production_fallback(query, response, sources, contexts)
                elif cleaned_response.startswith('"overall_score"') and len(cleaned_response) < 50:
                    # LLM sometimes returns just the field name without value or incomplete
                    logger.warning(f"LLM returned incomplete JSON: {cleaned_response}")
                    return self._validate_production_fallback(query, response, sources, contexts)
                elif cleaned_response.startswith('"') and cleaned_response.endswith('"') and '"' not in cleaned_response[1:-1]:
                    # LLM sometimes returns just the field name in quotes
                    logger.warning(f"LLM returned just field name: {cleaned_response}")
                    return self._validate_production_fallback(query, response, sources, contexts)
                else:
                    result = json.loads(cleaned_response)
                    logger.info(f"Production validation completed with JSON. Overall score: {result.get('overall_score', 0)}/10")
                    return result
                    
            except json.JSONDecodeError as e:
                # Log the actual response for debugging
                logger.warning(f"JSON parsing failed: {e}")
                logger.warning(f"Raw response: {llm_response.content.strip()}")
                # Fallback if JSON parsing fails
                logger.warning("Using fallback validation")
                return self._validate_production_fallback(query, response, sources, contexts)
                
        except Exception as e:
            logger.error(f"Error in production mode validation: {e}")
            return {
                "overall_score": 1.0,
                "confidence_level": "very_low",
                "error": str(e),
                "validation_mode": "production"
            }
    
    def _validate_production_fallback(self, query: str, response: str, sources: List[str], contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fallback production mode validation without JSON parsing.
        
        Args:
            query: Original user query
            response: Generated product suggestions
            sources: List of source references
            contexts: Retrieved contexts for validation
            
        Returns:
            Detailed validation results with production criteria
        """
        logger.info("Running production mode validation fallback")
        
        try:
            # Use production-specific validation criteria
            validation_criteria = {
                'robustness': self._check_robustness(response),
                'standardization': self._check_standardization(response),
                'detail_level': self._check_detail_level(response),
                'citation_quality': self._check_citation_quality(response, sources)
            }
            
            # Calculate overall score
            scores = [criteria['score'] for criteria in validation_criteria.values()]
            overall_score = sum(scores) / len(scores) if scores else 0
            
            # Determine confidence level
            if overall_score >= 8.5:
                confidence_level = 'very_high'
            elif overall_score >= 7.0:
                confidence_level = 'high'
            elif overall_score >= 5.0:
                confidence_level = 'medium'
            elif overall_score >= 3.0:
                confidence_level = 'low'
            else:
                confidence_level = 'very_low'
            
            # Collect flagged issues
            flagged_issues = []
            for criterion, result in validation_criteria.items():
                flagged_issues.extend(result.get('issues', []))
            
            # Generate recommendations
            recommendations = self._generate_production_recommendations(validation_criteria)
            
            result = {
                "overall_score": round(overall_score, 1),
                "confidence_level": confidence_level,
                "production_criteria": validation_criteria,
                "flagged_issues": flagged_issues,
                "recommendations": recommendations,
                "validation_mode": "production",
                "criteria_met": {
                    "robust": validation_criteria['robustness']['score'] >= 7,
                    "standard": validation_criteria['standardization']['score'] >= 7,
                    "detailed": validation_criteria['detail_level']['score'] >= 7,
                    "cited": validation_criteria['citation_quality']['score'] >= 7
                }
            }
            
            logger.info(f"Production validation fallback completed. Overall score: {overall_score}/10")
            return result
            
        except Exception as e:
            logger.error(f"Error in production mode validation fallback: {e}")
            return {
                "overall_score": 1.0,
                "confidence_level": "very_low",
                "error": str(e),
                "validation_mode": "production"
            }

    def _validate_product_suggestions_basic(self, query: str, response: str, sources: List[str]) -> Dict[str, Any]:
        """
        Basic validation for product suggestions without deep source verification.
        
        Args:
            query: Original user query
            response: Product suggestions response to validate
            sources: Source names/references
            
        Returns:
            Basic validation results for product suggestions
        """
        logger.info("Performing basic product suggestion validation")
        
        # Prepare sources string
        sources_text = "\n".join([f"- {source}" for source in sources])
        
        # Create the prompt
        user_prompt = PRODUCT_SUGGESTION_FACT_CHECK_USER_PROMPT.format(
            query=query,
            response=response,
            sources=sources_text
        )
        
        # Prepare messages
        messages = [
            SystemMessage(content=PRODUCT_SUGGESTION_FACT_CHECK_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        logger.debug(f"Sending basic product suggestion validation request to {self.llm_type} LLM")
        
        # Get response from LLM
        llm_response = self.llm.invoke(messages)
        content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
        
        logger.debug(f"Raw product suggestion validation response: {content}")
        
        # Parse validation response
        validation_result = self._parse_product_suggestion_validation_response(content)
        
        # Log validation summary
        overall_score = validation_result.get('overall_score', 0)
        flagged_issues = validation_result.get('flagged_issues', [])
        
        logger.info(f"Basic product suggestion validation complete. Score: {overall_score}/10, Issues: {len(flagged_issues)}")
        
        if flagged_issues:
            logger.warning("Issues identified during product suggestion validation:")
            for issue in flagged_issues:
                logger.warning(f"  - {issue}")
        
        # Add confidence level based on score
        validation_result['confidence_level'] = self._get_confidence_level(overall_score)
        
        return validation_result

    def _verify_product_citations(self, response: str, source_content: Dict[str, Dict[str, Any]]) -> str:
        """Verify product citations against source content."""
        verification_results = []
        
        # Look for citation patterns in the response
        import re
        citation_pattern = r'\[Context from ([^\]]+)\]'
        citations = re.findall(citation_pattern, response)
        
        # Extract product mentions (simple heuristic)
        product_pattern = r'\*\*([^*]+)\*\*'
        products = re.findall(product_pattern, response)
        
        for i, citation in enumerate(citations):
            verification = f"Citation {i+1}: '{citation}' - "
            
            # Check if citation corresponds to available sources
            citation_found = False
            for source_name, content_info in source_content.items():
                if citation.lower() in source_name.lower() or source_name.lower() in citation.lower():
                    verification += f"Verified in {source_name}"
                    citation_found = True
                    break
            
            if not citation_found:
                verification += "Source not clearly identified"
            
            verification_results.append(verification)
        
        # Check if products have citations
        products_without_citations = []
        for product in products:
            # Simple check if product appears near a citation
            product_index = response.find(product)
            if product_index != -1:
                # Look for citations within 200 characters
                surrounding_text = response[max(0, product_index-100):product_index+100]
                if not re.search(citation_pattern, surrounding_text):
                    products_without_citations.append(product)
        
        if products_without_citations:
            verification_results.append(f"Products without clear citations: {', '.join(products_without_citations[:3])}")
        
        return "\n".join(verification_results) if verification_results else "No specific citations to verify"

    def _identify_unsupported_products(self, response: str, source_content: Dict[str, Dict[str, Any]]) -> List[str]:
        """Identify product suggestions that appear unsupported by source content."""
        import re
        unsupported = []
        
        # Extract product names from response
        product_pattern = r'\*\*([^*]+)\*\*'
        products = re.findall(product_pattern, response)
        
        for product in products[:5]:  # Limit for performance
            product_lower = product.lower()
            
            # Check if product is mentioned in any source content
            found_in_sources = False
            for content_info in source_content.values():
                content_lower = content_info['content'].lower()
                
                # Check for product name or related keywords
                product_words = set(product_lower.split())
                content_words = set(content_lower.split())
                overlap = len(product_words.intersection(content_words))
                
                if overlap >= 2:  # Require at least 2 word overlap
                    found_in_sources = True
                    break
                    
                # Also check for direct substring match
                if product_lower in content_lower or any(word in content_lower for word in product_words if len(word) > 3):
                    found_in_sources = True
                    break
            
            if not found_in_sources:
                unsupported.append(product)
        
        return unsupported[:3]  # Limit to first 3 for readability

    def _parse_product_suggestion_validation_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the JSON response from the product suggestion fact-checking agent.
        
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
            
            # Validate required fields for product suggestions
            required_fields = [
                'overall_score', 'validation_results', 'flagged_issues',
                'recommendations', 'confidence_assessment'
            ]
            for field in required_fields:
                if field not in result:
                    logger.warning(f"Missing field in product suggestion validation response: {field}")
            
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
            
            logger.debug("Successfully parsed product suggestion validation response")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response}")
            # Fallback for product suggestions
            return {
                "overall_score": 5,
                "validation_results": {
                    "citation_quality": "Could not parse response",
                    "data_fidelity": "Could not parse response",
                    "relevance": "Could not parse response",
                    "clarity": "Could not parse response",
                    "accuracy": "Could not parse response"
                },
                "flagged_issues": [f"JSON parsing error: {e}"],
                "recommendations": "Manual review recommended due to parsing error",
                "confidence_assessment": "Low confidence due to parsing failure"
            }
        
        except Exception as e:
            logger.error(f"Error parsing product suggestion validation response: {e}")
            return self._create_fallback_result(str(e))


