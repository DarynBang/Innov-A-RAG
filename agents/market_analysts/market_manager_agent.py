"""
MarketManagerAgent: Synthesizes the final answer using the selected LLM and outputs from opportunity/risk agents.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from agents.base import BaseAgent
from config.prompts import (
    MARKET_MANAGER_AGENT_SYSTEM_PROMPT,
    MARKET_MANAGER_AGENT_USER_PROMPT,
    PRODUCT_SUGGESTION_MANAGER_SYSTEM_PROMPT,
    PRODUCT_SUGGESTION_MANAGER_USER_PROMPT,
    DEFAULT_MODELS
)
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
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
        
        # Product suggestion mode configuration
        self.product_suggestion_mode = False
        
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

    def run(self, input_data: dict, product_suggestion_mode: bool = False) -> str:
        """
        Synthesize the final answer using the selected LLM.
        
        Args:
            input_data: dict with outputs from previous agents
            product_suggestion_mode: if True, run in product suggestion mode
            
        Returns:
            str: final answer
        """
        if product_suggestion_mode:
            return self.run_product_suggestion(input_data, production_mode=True)
        else:
            return self.run_market_analysis(input_data)
    
    def run_market_analysis(self, input_data: dict) -> str:
        """
        Run the traditional market analysis workflow.
        
        Args:
            input_data: dict with outputs from previous agents
            
        Returns:
            str: market analysis result
        """
        opportunities = input_data.get("opportunity_analysis", "")
        risks = input_data.get("risk_analysis", "")
        question = input_data.get("question")
        synthesis_result = input_data.get("synthesis_result", "")
        
        logger.info(f"MarketManagerAgent processing market analysis for query: {question}")
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
            
            logger.info("Invoking LLM for market analysis synthesis")
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Extract content
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            logger.info(f"Generated market analysis synthesis length: {len(result)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Error in market analysis synthesis: {e}")
            return f"Error synthesizing market analysis output: {str(e)}"
    
    def run_product_suggestion(self, input_data: dict, production_mode: bool = False) -> str:
        """
        Run the product suggestion workflow - identify products from retrieved contexts.
        
        Args:
            input_data: dict with retrieved contexts and query
            production_mode: If True, use only given information with structured citations
            
        Returns:
            str: product suggestions based on retrieved contexts
        """
        question = input_data.get("question")
        contexts = input_data.get("contexts", [])
        synthesis_result = input_data.get("synthesis_result", "")
        
        mode_desc = "production" if production_mode else "development"
        logger.info(f"MarketManagerAgent processing product suggestions ({mode_desc} mode) for query: {question}")
        logger.info(f"Available contexts: {len(contexts)}")
        
        if not question:
            logger.warning("Missing question for product suggestion")
            return "No query provided for product suggestions"
        
        if not contexts:
            logger.warning("No contexts available for product suggestions")
            return "No contexts available to extract product suggestions from"
        
        try:
            if production_mode:
                return self._run_production_mode_suggestions(question, synthesis_result, contexts)
            else:
                return self._run_development_mode_suggestions(question, contexts)
            
        except Exception as e:
            logger.error(f"Error in product suggestion analysis: {e}")
            return f"Error generating product suggestions: {str(e)}"
    
    def _run_production_mode_suggestions(self, question: str, synthesis_result: str, contexts: list) -> str:
        """
        Production mode: Generate product suggestions based on synthesis result AND all retrieved contexts.
        
        Args:
            question: Original user query
            synthesis_result: Structured information from generalize agent
            contexts: All retrieved contexts (60 chunks)
            
        Returns:
            Product suggestions with proper source citations
        """
        logger.info("Running production mode product suggestions with structured citations")
        logger.info(f"Available contexts for product suggestions: {len(contexts)}")
        
        if not synthesis_result and not contexts:
            logger.warning("No synthesis result or contexts available for production mode")
            return "No information available to generate product suggestions."
        
        # Format all contexts for product suggestion analysis
        formatted_contexts = self._format_contexts_for_product_suggestion(contexts) if contexts else ""
        
        try:
            from config.prompts import PRODUCTION_PRODUCT_SUGGESTION_PROMPT
            
            # Prepare messages using the enhanced production prompt from config
            messages = [
                SystemMessage(content="You are a product suggestion specialist for production mode. Use ALL available contexts to generate comprehensive product suggestions."),
                HumanMessage(content=PRODUCTION_PRODUCT_SUGGESTION_PROMPT.format(
                    query=question,
                    synthesis_result=synthesis_result,
                    formatted_contexts=formatted_contexts
                ))
            ]
            
            logger.info("Invoking LLM for production mode product suggestions with COT and JSON")
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            logger.debug(f"MarketManagerAgent LLM response: {type(response)}")
            if hasattr(response, 'content'):
                logger.debug(f"MarketManagerAgent response content repr: {repr(response.content)}")
            
            # Parse JSON response
            import json
            product_suggestions = None  # Initialize to avoid NameError
            try:
                # Clean the response - remove markdown code blocks if present
                cleaned_response = response.content.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                
                cleaned_response = cleaned_response.strip()
                
                # Additional cleaning for common LLM issues - handle field names with whitespace
                cleaned_response_stripped = cleaned_response.strip()
                if cleaned_response_stripped == '"product_suggestions"':
                    # LLM returned exactly just the field name in quotes
                    logger.warning(f"LLM returned just field name: {repr(cleaned_response)}")
                    return "No product suggestions available due to parsing error"
                elif cleaned_response_stripped.startswith('"product_suggestions"') and len(cleaned_response_stripped) < 50:
                    # LLM sometimes returns just the field name without value or incomplete
                    logger.warning(f"LLM returned incomplete JSON: {repr(cleaned_response)}")
                    return "No product suggestions available due to parsing error"
                elif cleaned_response.startswith('"') and cleaned_response.endswith('"') and '"' not in cleaned_response[1:-1]:
                    # LLM sometimes returns just the field name in quotes
                    logger.warning(f"LLM returned just field name: {cleaned_response}")
                    return "No product suggestions available due to parsing error"
                else:
                    result = json.loads(cleaned_response)
                    product_suggestions = result.get('product_suggestions', [])
            
                # Ensure product_suggestions is always set
                if product_suggestions is None:
                    logger.warning("product_suggestions was never assigned, using empty list")
                    product_suggestions = []
                
                # Convert to formatted string for compatibility
                if product_suggestions:
                    formatted_suggestions = "\n".join([f"• {suggestion}" for suggestion in product_suggestions])
                else:
                    formatted_suggestions = "No specific products found in the provided information."
                
                logger.info(f"Generated {len(product_suggestions)} production mode product suggestions")
                return formatted_suggestions
                    
            except json.JSONDecodeError as e:
                # Log the actual response for debugging
                logger.warning(f"JSON parsing failed: {e}")
                logger.warning(f"Raw response: {response.content.strip()}")
                # Fallback if JSON parsing fails
                logger.warning("Using raw response as fallback")
                return "No product suggestions available due to parsing error"
            
        except Exception as e:
            logger.error(f"Error in production mode product suggestions: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error generating production mode product suggestions: {str(e)}"
    
    def _run_development_mode_suggestions(self, question: str, contexts: list) -> str:
        """
        Development mode: Full LLM-based product suggestions (original behavior).
        
        Args:
            question: Original user query
            contexts: List of context dictionaries
            
        Returns:
            Comprehensive product suggestions
        """
        logger.info("Running development mode product suggestions with full LLM analysis")
        
        # Format contexts for product suggestion analysis
        formatted_contexts = self._format_contexts_for_product_suggestion(contexts)
        
        # Create the product suggestion prompt
        user_prompt = PRODUCT_SUGGESTION_MANAGER_USER_PROMPT.format(
            query=question,
            contexts=formatted_contexts
        )
        
        # Prepare messages
        messages = [
            SystemMessage(content=PRODUCT_SUGGESTION_MANAGER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        logger.info("Invoking LLM for development mode product suggestion analysis")
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        
        # Extract content
        if hasattr(response, 'content'):
            result = response.content
        else:
            result = str(response)
        
        logger.info(f"Generated development mode product suggestions length: {len(result)} characters")
        return result
    
    def _extract_source_info(self, context: dict, tool_name: str) -> str:
        """
        Extract detailed source information for enhanced citation purposes.
        
        Args:
            context: context dictionary
            tool_name: name of the tool that generated the context
            
        Returns:
            str: detailed source information string for proper citations
        """
        result = context.get('result', '')
        
        # Enhanced patent source extraction
        if 'patent' in tool_name.lower():
            patent_id = "Unknown"
            company_name = "Unknown Company"
            content_type = "content"
            
            if isinstance(result, str):
                import re
                # Extract patent ID - look for patterns like "1. PATENT 50911877"
                patent_match = re.search(r'patent\s+(\d+)', result, re.IGNORECASE)
                if patent_match:
                    patent_id = patent_match.group(1)
                
                # Extract company name - look for "   Company: Halozyme Therapeutics"
                company_match = re.search(r'company:\s*([^\n\r]+)', result, re.IGNORECASE)
                if company_match:
                    company_name = company_match.group(1).strip()
                
                # Determine content type based on content - be more specific
                content_lower = result.lower()
                if 'devices' in content_lower and 'systems' in content_lower:
                    content_type = "device and system specifications"
                elif 'methods' in content_lower and 'formulations' in content_lower:
                    content_type = "method and formulation details"
                elif 'composition' in content_lower and 'delivery' in content_lower:
                    content_type = "composition and delivery mechanisms"
                elif 'transdermal' in content_lower or 'transmucosal' in content_lower:
                    content_type = "transdermal delivery technology"
                elif 'device' in content_lower or 'system' in content_lower:
                    content_type = "device/system section"
                elif 'method' in content_lower or 'composition' in content_lower:
                    content_type = "method/composition section"
                elif 'claim' in content_lower:
                    content_type = "claims section"  
                else:
                    content_type = "patent technical content"
            
            return f"PATENT_SOURCE: Patent US{patent_id} - {company_name} - {content_type}"
        
        # Enhanced company source extraction  
        elif 'company' in tool_name.lower():
            company_name = "Unknown Company"
            content_type = "company profile"
            
            if isinstance(result, str):
                # Extract company name from various patterns - try multiple approaches
                lines = result.split('\n')
                
                # Method 1: Look for company names in first few lines
                for line in lines[:10]:
                    line = line.strip()
                    if line and len(line) < 100 and len(line) > 3:
                        # Check if this looks like a company name
                        if any(keyword in line.lower() for keyword in ['inc', 'corp', 'ltd', 'llc', 'company', 'therapeutics', 'technologies', 'pharma', 'biotech']):
                            company_name = line
                            break
                        # Check if it's a standalone company name (common pattern)
                        if not any(char in line for char in [':','=','.','{','[']) and line[0].isupper():
                            potential_name = line
                            # Verify it's not a generic term
                            if not any(generic in line.lower() for generic in ['relevant', 'information', 'company', 'profile', 'description']):
                                company_name = potential_name
                                break
                
                # Determine content section with more specificity
                content_lower = result.lower()
                if 'product' in content_lower and 'portfolio' in content_lower:
                    content_type = "product portfolio and offerings"
                elif 'business' in content_lower and ('focus' in content_lower or 'model' in content_lower):
                    content_type = "business strategy and focus areas"
                elif 'technology' in content_lower or 'innovation' in content_lower:
                    content_type = "technology and innovation details"
                elif 'drug' in content_lower and 'development' in content_lower:
                    content_type = "drug development and pipeline"
                elif 'therapeutic' in content_lower or 'pharma' in content_lower:
                    content_type = "therapeutic focus and capabilities"
                else:
                    content_type = "company profile and business overview"
            
            return f"COMPANY_SOURCE: Company Profile: {company_name} - {content_type}"
        
        else:
            # Generic enhanced source info
            return f"SOURCE: {tool_name} retrieval"

    def _format_contexts_for_product_suggestion(self, contexts: list) -> str:
        """
        Format contexts specifically for product suggestion analysis.
        
        Args:
            contexts: List of context dictionaries
            
        Returns:
            Formatted string of all contexts for comprehensive product analysis
        """
        if not contexts:
            return "No contexts available"
        
        formatted_sections = []
        
        for i, context in enumerate(contexts, 1):
            if isinstance(context, dict):
                # Extract key information from context
                tool_name = context.get('tool_name', 'Unknown Tool')
                result = context.get('result', 'No content')
                
                # Create detailed context section
                section = f"\n--- Context {i} ({tool_name}) ---\n"
                
                # Add enhanced source info using the correct method and parameter order
                source_info = self._extract_source_info(context, tool_name)
                section += f"Source: {source_info}\n"
                
                # Add content with truncation for very long results
                if isinstance(result, str):
                    # Keep full content for comprehensive analysis
                    content = result.strip()
                    if len(content) > 3000:  # Truncate extremely long content but keep substantial amount
                        content = content[:3000] + "... [truncated for length]"
                    section += f"Content: {content}\n"
                else:
                    section += f"Content: {str(result)}\n"
                
                formatted_sections.append(section)
            else:
                # Handle string contexts
                formatted_sections.append(f"\n--- Context {i} ---\n{str(context)}\n")
        
        result = "\n".join(formatted_sections)
        logger.info(f"Formatted {len(contexts)} contexts for product suggestion analysis (total length: {len(result)} chars)")
        
        return result

