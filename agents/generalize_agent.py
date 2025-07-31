"""
GeneralizeAgent: Synthesizes information from multiple sources and subquestions to provide comprehensive answers.
Enhanced with proper prompting, source attribution, and confidence scoring.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

import json
from agents.base import BaseAgent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from config.prompts import (
    GENERALIZE_AGENT_SYSTEM_PROMPT,
    GENERALIZE_AGENT_USER_PROMPT,
    DEFAULT_MODELS
)
from typing import Dict, List, Any

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

    def register_tools(self, tools: dict):
        """Register tools for potential future use."""
        self.tools.update(tools)
        logger.info(f"Registered {len(tools)} tools: {list(tools.keys())}")

    def run(self, input_data: dict) -> str:
        """
        Run the GeneralizeAgent with input data and return synthesis results.
        
        Args:
            input_data: dict containing:
                - original_query: the original user query
                - subquestions: list of subquestions to process
                - contexts: list of retrieved contexts
                - accumulated_context: optional accumulated context from previous steps
                - product_suggestion_mode: optional boolean for product suggestion mode
                
        Returns:
            str: Synthesized information response
        """
        logger.info("GeneralizeAgent.run() called")
        
        # Extract required parameters from input_data
        original_query = input_data.get("original_query", "")
        subquestions = input_data.get("subquestions", [])
        contexts = input_data.get("contexts", [])
        accumulated_context = input_data.get("accumulated_context")
        product_suggestion_mode = input_data.get("product_suggestion_mode", False)
        
        if not original_query:
            logger.warning("No original_query provided in input_data")
            return "Error: No original query provided for synthesis."
        
        if not contexts:
            logger.warning("No contexts provided for synthesis")
            return "Error: No contexts available for information synthesis."
        
        logger.info(f"Synthesizing information for query: {original_query}")
        logger.info(f"Processing {len(subquestions)} subquestions with {len(contexts)} contexts")
        logger.info(f"Product suggestion mode: {product_suggestion_mode}")
        
        try:
            # Call the main synthesis functionality
            result = self.synthesize_information(
                original_query=original_query,
                subquestions=subquestions,
                contexts=contexts,
                accumulated_context=accumulated_context,
                product_suggestion_mode=product_suggestion_mode
            )
            
            logger.info("GeneralizeAgent.run() completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in GeneralizeAgent.run(): {e}")
            return f"Error during information synthesis: {str(e)}"

    def synthesize_information(
        self, 
        original_query: str, 
        subquestions: List[str], 
        contexts: List[Dict[str, Any]],
        accumulated_context: str = None,
        product_suggestion_mode: bool = False
    ) -> str:
        """
        Synthesize information from multiple sources to answer user queries comprehensively.
        
        Args:
            original_query: The original user query
            subquestions: List of subquestions derived from the original query
            contexts: List of context dictionaries with retrieval results
            accumulated_context: Optional accumulated context from previous subquestions
            
        Returns:
            Comprehensive answer with source attribution
        """
        mode_desc = "production" if product_suggestion_mode else "development"
        logger.info(f"Synthesizing information ({mode_desc} mode) for query: {original_query}")
        logger.info(f"Processing {len(subquestions)} subquestions and {len(contexts)} contexts")
        
        if accumulated_context:
            logger.info(f"Using accumulated context (length: {len(accumulated_context)}) for enhanced synthesis")
        
        try:
            if product_suggestion_mode:
                return self._synthesize_production_mode(original_query, contexts)
            else:
                return self._synthesize_development_mode(original_query, subquestions, contexts, accumulated_context)

        except Exception as e:
            logger.error(f"Error during information synthesis: {e}")
            return f"Error synthesizing information: {str(e)}"
    
    def _synthesize_production_mode(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """
        Production mode synthesis: Return structured dict format with company/patent IDs using COT prompt.
        
        Args:
            query: Original user query
            contexts: List of context dictionaries
            
        Returns:
            Structured dictionary information as JSON string
        """
        logger.info("Running production mode synthesis with structured output and COT")
        
        try:
            from config.prompts import PRODUCTION_SYNTHESIS_PROMPT
            
            # Format contexts for the prompt
            contexts_text = self._format_contexts(contexts)
            
            messages = [
                SystemMessage(content="You are an information synthesis specialist for production mode."),
                HumanMessage(content=PRODUCTION_SYNTHESIS_PROMPT.format(query=query, contexts=contexts_text))
            ]
            
            response = self.llm.invoke(messages)
            logger.debug(f"GeneralizeAgent LLM response: {type(response)}")
            if hasattr(response, 'content'):
                logger.debug(f"GeneralizeAgent response content repr: {repr(response.content)}")
            
            # Parse JSON response
            structured_info = None  # Initialize to avoid NameError
            
            # Clean the response - remove markdown code blocks if present
            cleaned_response = response.content.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            cleaned_response = cleaned_response.strip()
            
            # Additional cleaning for common LLM issues
            if cleaned_response == '"structured_info"' or cleaned_response.strip() == '"structured_info"':
                # LLM returned exactly just the field name in quotes
                logger.warning(f"LLM returned just field name: {cleaned_response}")
                return self._synthesize_production_mode_fallback(query, contexts)
            elif cleaned_response.startswith('"structured_info"') and len(cleaned_response) < 50:
                # LLM sometimes returns just the field name without value or incomplete
                logger.warning(f"LLM returned incomplete JSON: {cleaned_response}")
                return self._synthesize_production_mode_fallback(query, contexts)
            elif cleaned_response.startswith('"') and cleaned_response.endswith('"') and '"' not in cleaned_response[1:-1]:
                # LLM sometimes returns just the field name in quotes
                logger.warning(f"LLM returned just field name: {cleaned_response}")
                return self._synthesize_production_mode_fallback(query, contexts)
            else:
                result = json.loads(cleaned_response)
                structured_info = result.get('structured_info', 'No information available')
                logger.info(f"Production mode synthesis completed with JSON output")
            
            # Ensure structured_info is always set
            if structured_info is None:
                logger.warning("structured_info was never assigned, using fallback")
                return self._synthesize_production_mode_fallback(query, contexts)
            
            return structured_info
            
        except json.JSONDecodeError as e:
            # Log the actual response for debugging
            logger.warning(f"JSON parsing failed: {e}")
            logger.warning(f"Raw response: {response.content.strip()}")
            # Fallback to original method if JSON parsing fails
            logger.warning("Using fallback method")
            return self._synthesize_production_mode_fallback(query, contexts)
                
        except Exception as e:
            logger.error(f"Error in production mode synthesis: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "Error processing information in production mode."
    
    def _synthesize_production_mode_fallback(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """
        Fallback production mode synthesis without JSON parsing.
        
        Args:
            query: Original user query
            contexts: List of context dictionaries
            
        Returns:
            Structured dictionary information as string
        """
        logger.info("Running production mode synthesis fallback")
        
        # Extract and organize information by company and patent IDs
        companies_info = {}
        patents_info = {}
        
        for context in contexts:
            tool = context.get('tool', '')
            result = context.get('result', {})
            
            if isinstance(result, dict):
                # Handle hybrid retrieval results
                if 'company_contexts' in result:
                    for company_ctx in result.get('company_contexts', []):
                        company_id = self._extract_company_id(company_ctx)
                        if company_id:
                            companies_info[company_id] = self._format_company_info(company_ctx)
                
                if 'patent_contexts' in result:
                    for patent_ctx in result.get('patent_contexts', []):
                        patent_id = self._extract_patent_id(patent_ctx)
                        if patent_id:
                            patents_info[patent_id] = self._format_patent_info(patent_ctx)
                
                # Handle regular chunks format
                elif 'chunks' in result:
                    chunks = result['chunks']
                    for chunk in chunks:
                        if isinstance(chunk, dict):
                            # Determine if this is company or patent data based on tool or content
                            if 'company' in tool.lower() or 'firm' in tool.lower():
                                company_id = self._extract_company_id(chunk)
                                if company_id:
                                    companies_info[company_id] = self._format_company_info(chunk)
                            elif 'patent' in tool.lower():
                                patent_id = self._extract_patent_id(chunk)
                                if patent_id:
                                    patents_info[patent_id] = self._format_patent_info(chunk)
        
        # Build structured output
        structured_output = []
        
        # Add company information
        for company_id, info in companies_info.items():
            structured_output.append(f"Company {company_id}: {info}")
        
        # Add patent information
        for patent_id, info in patents_info.items():
            structured_output.append(f"Patent {patent_id}: {info}")
        
        if not structured_output:
            logger.warning("No structured information extracted in production mode")
            return "No company or patent information found in the provided contexts."
        
        result = "; ".join(structured_output)
        logger.info(f"Production mode synthesis completed: {len(companies_info)} companies, {len(patents_info)} patents")
        return result
    
    def _extract_company_id(self, context: Dict[str, Any]) -> str:
        """
        Extract company ID (company_id or hojin_id) from context.
        
        Args:
            context: Company context dictionary
            
        Returns:
            Company ID string or None
        """
        # Try different possible ID fields
        for id_field in ['company_id', 'hojin_id', 'id', 'Company ID', 'Hojin ID']:
            if id_field in context and context[id_field]:
                return str(context[id_field])
        
        # Fallback: try to extract from company_name or source
        company_name = context.get('company_name', context.get('source', ''))
        if company_name:
            return company_name
        
        return "Unknown"
    
    def _extract_patent_id(self, context: Dict[str, Any]) -> str:
        """
        Extract patent ID (patent_id or appln_id) from context.
        
        Args:
            context: Patent context dictionary
            
        Returns:
            Patent ID string or None
        """
        # Try different possible ID fields
        for id_field in ['patent_id', 'appln_id', 'id', 'Patent ID', 'Application ID']:
            if id_field in context and context[id_field]:
                return str(context[id_field])
        
        # Fallback: try to extract from source or any other identifier
        source = context.get('source', context.get('patent_number', ''))
        if source:
            return source
        
        return "Unknown"
    
    def _format_company_info(self, context: Dict[str, Any]) -> str:
        """
        Format company information for structured output.
        
        Args:
            context: Company context dictionary
            
        Returns:
            Formatted company information string
        """
        info_parts = []
        
        # Add company name if available
        company_name = context.get('company_name', '')
        if company_name:
            info_parts.append(f"Name: {company_name}")
        
        # Add key information from chunk
        chunk = context.get('chunk', context.get('content', ''))
        if chunk:
            # Truncate if too long for structured output
            if len(chunk) > 200:
                chunk = chunk[:200] + "..."
            info_parts.append(f"Info: {chunk}")
        
        return "; ".join(info_parts) if info_parts else "No information available"
    
    def _format_patent_info(self, context: Dict[str, Any]) -> str:
        """
        Format patent information for structured output.
        
        Args:
            context: Patent context dictionary
            
        Returns:
            Formatted patent information string
        """
        info_parts = []
        
        # Add patent title if available
        title = context.get('title', context.get('patent_title', ''))
        if title:
            info_parts.append(f"Title: {title}")
        
        # Add key information from chunk
        chunk = context.get('chunk', context.get('content', ''))
        if chunk:
            # Truncate if too long for structured output
            if len(chunk) > 200:
                chunk = chunk[:200] + "..."
            info_parts.append(f"Info: {chunk}")
        
        return "; ".join(info_parts) if info_parts else "No information available"
    
    def _synthesize_development_mode(self, original_query: str, subquestions: List[str], contexts: List[Dict[str, Any]], accumulated_context: str = None) -> str:
        """
        Development mode synthesis: Full LLM-based synthesis (original behavior).
        
        Args:
            original_query: The original user query
            subquestions: List of subquestions
            contexts: List of context dictionaries
            accumulated_context: Optional accumulated context
            
        Returns:
            Comprehensive synthesized response
        """
        logger.info("Running development mode synthesis with LLM")
        
        # Format contexts for the prompt
        contexts_text = self._format_contexts(contexts)
        subquestions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(subquestions)])
        
        # Enhanced user prompt with accumulated context
        base_user_prompt = GENERALIZE_AGENT_USER_PROMPT.format(
            original_query=original_query,
            subquestions=subquestions_text,
            contexts=contexts_text
        )
        
        # Add accumulated context if available
        if accumulated_context and len(subquestions) > 1:
            enhanced_prompt = f"""
ACCUMULATED CONTEXT FROM PREVIOUS SUBQUESTIONS:
{accumulated_context}

CURRENT SYNTHESIS REQUEST:
{base_user_prompt}

NOTE: Use the accumulated context to ensure continuity and coherence across all subquestions. 
When referring to "these strategies", "the companies", or similar references, use the information 
from the accumulated context to provide specific details."""
            user_prompt = enhanced_prompt
            logger.info("Enhanced synthesis prompt with accumulated context")
        else:
            user_prompt = base_user_prompt
        
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
    
    def _format_contexts(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Format contexts for inclusion in the prompt.
        
        Args:
            contexts: List of context dictionaries
            
        Returns:
            Formatted context string
        """
        formatted_contexts = []
        total_chunks = 0
        
        for i, context in enumerate(contexts, 1):
            tool_name = context.get('tool', 'unknown')
            result = context.get('result', '')
            
            if isinstance(result, dict):
                # Handle optimized hybrid tool results
                if 'company_contexts' in result or 'patent_contexts' in result:
                    company_contexts = result.get('company_contexts', [])
                    patent_contexts = result.get('patent_contexts', [])
                    
                    # Process company contexts
                    for j, comp_ctx in enumerate(company_contexts):
                        total_chunks += 1
                        company_name = comp_ctx.get('company_name', 'Unknown Company')
                        company_id = comp_ctx.get('company_id', 'unknown')
                        chunk_content = comp_ctx.get('chunk', '')
                        score = comp_ctx.get('score', 0)
                        
                        formatted_context = f"Company Context {total_chunks} (ID: {company_id}, Name: {company_name}, Score: {score:.3f}):\n{chunk_content}"
                        formatted_contexts.append(formatted_context)
                    
                    # Process patent contexts
                    for j, pat_ctx in enumerate(patent_contexts):
                        total_chunks += 1
                        patent_id = pat_ctx.get('patent_number', pat_ctx.get('patent_id', 'Unknown Patent'))
                        chunk_content = pat_ctx.get('chunk', '')
                        score = pat_ctx.get('score', 0)
                        
                        formatted_context = f"Patent Context {total_chunks} (ID: {patent_id}, Score: {score:.3f}):\n{chunk_content}"
                        formatted_contexts.append(formatted_context)
                
                # Handle structured results with chunks
                elif 'chunks' in result:
                    chunks = result['chunks']
                    sources = []
                    
                    for chunk in chunks:
                        if isinstance(chunk, dict):
                            total_chunks += 1
                            content = chunk.get('chunk', '')
                            source = chunk.get('source', 'Unknown Source')
                            sources.append(source)
                            
                            formatted_context = f"Context {total_chunks} (Tool: {tool_name}, Source: {source}):\n{content}"
                            formatted_contexts.append(formatted_context)
                
                # Handle other dict results
                else:
                    total_chunks += 1
                    content = str(result)
                    formatted_context = f"Context {total_chunks} (Tool: {tool_name}):\n{content}"
                    formatted_contexts.append(formatted_context)
            else:
                total_chunks += 1
                content = str(result)
                formatted_context = f"Context {total_chunks} (Tool: {tool_name}):\n{content}"
                formatted_contexts.append(formatted_context)
        
        logger.info(f"Formatted {total_chunks} individual contexts for synthesis")
        return "\n\n".join(formatted_contexts)




