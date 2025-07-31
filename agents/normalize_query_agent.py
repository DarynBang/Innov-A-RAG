"""
NormalizeQueryAgent: Classifies and normalizes user queries using LLM.
Determines if query is about company, patent, or general technology/market analysis.
Enhanced with Langchain tool integration and comprehensive prompts.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from agents.base import BaseAgent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from config.prompts import (
    DEFAULT_MODELS,
    NORMALIZE_AGENT_USER_PROMPT
)
from utils.langchain_tool_registry import get_langchain_tool_registry
import json
import re
from typing import Dict, List, Any

logger = get_logger(__name__)

class NormalizeQueryAgent(BaseAgent):
    def __init__(self, name="NormalizeQueryAgent", qa_model="openai"):
        super().__init__(name, qa_model)
        logger.info(f"Initializing NormalizeQueryAgent with model: {qa_model}")
        
        self.llm_type = qa_model
        self.llm = self._initialize_llm()
        self.available_tools = []
        self.langchain_registry = get_langchain_tool_registry()
        
        logger.info("NormalizeQueryAgent initialized successfully")
    
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
        """Register available tools for the agent."""
        super().register_tools(tools)
        self.available_tools = list(tools.keys()) if tools else []
        
        logger.info(f"Registered {len(self.available_tools)} tools: {self.available_tools}")
    
    def run(self, input_data: dict) -> str:
        """
        Run the NormalizeQueryAgent with input data and return normalization results.
        
        Args:
            input_data: dict containing:
                - question: the query to normalize and process
                - product_suggestion_mode: optional boolean for product suggestion mode
                
        Returns:
            str: JSON string representation of normalization and retrieval results
        """
        logger.info("NormalizeQueryAgent.run() called")
        
        # Extract query from input data
        query = input_data.get("question", "")
        product_suggestion_mode = input_data.get("product_suggestion_mode", False)
        
        if not query:
            logger.warning("No query provided in input_data")
            return '{"error": "No query provided", "status": "failed"}'
        
        logger.info(f"Processing query: {query}")
        logger.info(f"Product suggestion mode: {product_suggestion_mode}")
        
        try:
            # Call the main normalization and retrieval functionality
            result = self.normalize_and_retrieve(query, product_suggestion_mode=product_suggestion_mode)
            
            # Convert result to JSON string for return
            import json
            json_result = json.dumps(result, ensure_ascii=False, indent=2)
            
            logger.info("NormalizeQueryAgent.run() completed successfully")
            return json_result
            
        except Exception as e:
            logger.error(f"Error in NormalizeQueryAgent.run(): {e}")
            error_result = {
                "error": str(e),
                "status": "failed",
                "query": query
            }
            import json
            return json.dumps(error_result, ensure_ascii=False, indent=2)
    
    def _generate_dynamic_system_prompt(self) -> str:
        """Generate dynamic system prompt with current tool information."""
        prompt_section = self.langchain_registry.generate_prompt_section()
        
        return f"""
            You are a Query Normalization Agent responsible for classifying queries and extracting relevant identifiers.

            Your role is to:
            1. Classify each query as 'company', 'patent', 'company_patents', 'comparison', or 'general'
            2. Extract specific identifiers when available (company names, patent IDs)
            3. Recommend the best tools to use for information retrieval
            4. Handle complex queries that involve both companies and patents

            Classification Guidelines:
            - 'company': Queries asking about specific companies, their business, financials, market position
            - 'patent': Queries asking about specific patents or patent portfolios
            - 'company_patents': Queries asking about patents of specific companies
            - 'comparison': Queries comparing multiple entities (companies, patents, technologies)
            - 'general': Broad industry trends, or queries that don't focus on specific entities

            Special handling for company-patent relationships:
            - "patents of company X" → use exact_company_lookup + patent_rag_retrieval
            - "company X's patent portfolio" → use exact_company_lookup + patent_rag_retrieval
            - "tell me about company X patents" → use exact_company_lookup + patent_rag_retrieval

            {prompt_section}

            Output format: JSON with the following structure:
            {{
                "query_type": "company|patent|company_patents|comparison|general",
                "identifiers": {{
                    "companies": ["company1", "company2", ...],
                    "patents": ["patent1", "patent2", ...]
                }},
                "recommended_tools": ["tool1", "tool2", ...],
                "reasoning": "Explanation of classification and tool selection"
            }}"""

    def normalize_query(self, query: str) -> Dict[str, Any]:
        """
        Normalize and categorize a user query with tool recommendations.
        
        Args:
            query: The user's input query
            
        Returns:
            Dictionary containing query type, extracted identifiers, recommended tools, and analysis
        """
        logger.info(f"Normalizing query: {query}")
        
        try:
            # Generate dynamic system prompt with tool descriptions
            dynamic_system_prompt = self._generate_dynamic_system_prompt()
            
            # Create the user prompt
            user_prompt = NORMALIZE_AGENT_USER_PROMPT.format(
                query=query
            )
            
            # Prepare messages
            messages = [
                SystemMessage(content=dynamic_system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            logger.debug(f"Sending query to {self.llm_type} LLM for normalization")
            
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Extract content based on LLM type
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            logger.debug(f"Raw LLM response: {content}")
            
            # Parse JSON response
            normalized_result = self._parse_normalization_response(content)
            
            # Apply exact matching heuristics to improve tool selection
            normalized_result = self._apply_exact_matching_heuristics(query, normalized_result)
            
            logger.info(f"Query normalized as '{normalized_result.get('query_type', 'unknown')}' type")
            
            return normalized_result
            
        except Exception as e:
            logger.error(f"Error during query normalization: {str(e)}")
            # Return fallback result
            return {
                "query_type": "general",
                "identifiers": {"companies": [], "patents": []},
                "recommended_tools": ["enhanced_hybrid_rag_retrieval"],
                "reasoning": f"Error during normalization: {str(e)}",
                "error": str(e)
            }
    
    def normalize_and_retrieve(self, query: str, product_suggestion_mode: bool = False) -> Dict[str, Any]:
        """
        Normalize a query and invoke recommended tools for information retrieval.
        
        Args:
            query: The user's input query
            
        Returns:
            Dictionary containing normalization results and retrieved information
        """
        mode_desc = "production" if product_suggestion_mode else "development"
        logger.info(f"Normalizing and retrieving information ({mode_desc} mode) for: {query}")
        
        try:
            if product_suggestion_mode:
                return self._normalize_and_retrieve_production(query)
            else:
                return self._normalize_and_retrieve_development(query)
            
        except Exception as e:
            logger.error(f"Error during normalize and retrieve: {e}")
            return {
                "normalization": {"error": str(e)},
                "retrieved_contexts": [],
                "total_contexts": 0,
                "error": str(e)
            }
    
    def _normalize_and_retrieve_production(self, query: str) -> Dict[str, Any]:
        """
        Production mode: Only normalize query and use single optimized tool.
        
        Args:
            query: The user's input query
            
        Returns:
            Dictionary containing normalization results and retrieved information
        """
        logger.info("Running production mode normalization and retrieval")
        
        # Step 1: Normalize query to shorter but meaningful version
        normalized_query = self._normalize_query_for_production(query)
        
        # Step 2: Use only optimized hybrid tool with fixed top-k
        retrieved_contexts = []
        if "optimized_hybrid_rag_retrieval" in self.tool_executor:
            try:
                logger.info(f"Using optimized hybrid retrieval with top-k=30 for query: {normalized_query}")
                tool_result = self.tool_executor["optimized_hybrid_rag_retrieval"](
                    query=normalized_query,
                    top_k=30,
                    search_type="both"  # Search both companies and patents
                )
                retrieved_contexts.append({
                    "tool": "optimized_hybrid_rag_retrieval",
                    "input": normalized_query,
                    "result": tool_result,
                    "execution_type": "production"
                })
                logger.info("Production mode retrieval completed successfully")
            except Exception as e:
                logger.error(f"Error in production mode retrieval: {e}")
                retrieved_contexts.append({
                    "tool": "optimized_hybrid_rag_retrieval",
                    "error": str(e),
                    "execution_type": "error"
                })
        else:
            logger.warning("Optimized hybrid retrieval tool not available in production mode")
        
        # Return simplified result structure for production mode
        total_chunks = 0
        for context in retrieved_contexts:
            if context.get('execution_type') == 'production':
                result = context.get('result', {})
                if isinstance(result, dict):
                    company_contexts = result.get('company_contexts', [])
                    patent_contexts = result.get('patent_contexts', [])
                    total_chunks += len(company_contexts) + len(patent_contexts)
        
        result = {
            "normalization": {
                "original_query": query,
                "normalized_query": normalized_query,
                "query_type": "production",
                "mode": "production"
            },
            "retrieved_contexts": retrieved_contexts,
            "total_contexts": total_chunks  # Now shows actual chunks count
        }
        
        logger.info(f"Production mode retrieved {total_chunks} total chunks from {len(retrieved_contexts)} tools")
        return result
    
    def _normalize_query_for_production(self, query: str) -> str:
        """
        Normalize query for production mode: make it shorter but still meaningful.
        
        Args:
            query: Original user query
            
        Returns:
            Shortened but meaningful query
        """
        from config.prompts import PRODUCTION_QUERY_NORMALIZATION_PROMPT
        logger.debug("Successfully imported PRODUCTION_QUERY_NORMALIZATION_PROMPT")
        
        try:
            
            formatted_prompt = PRODUCTION_QUERY_NORMALIZATION_PROMPT.format(query=query)
            logger.debug(f"Formatted prompt length: {len(formatted_prompt)}")
            
            messages = [
                SystemMessage(content="You are a query normalization specialist for production mode."),
                HumanMessage(content=formatted_prompt)
            ]
            logger.debug("Messages created successfully")
            logger.debug(f"Message types: {[type(msg).__name__ for msg in messages]}")
            
            try:
                response = self.llm.invoke(messages)
                logger.debug("LLM invoke completed successfully")
            except Exception as llm_error:
                logger.error(f"LLM invoke failed: {llm_error}")
                logger.error(f"LLM error type: {type(llm_error).__name__}")
                raise
            logger.debug(f"LLM response received: {type(response)}")
            logger.debug(f"LLM response content: {response.content if hasattr(response, 'content') else 'No content attribute'}")
            
            # Check if response.content is the problematic value
            if hasattr(response, 'content'):
                logger.debug(f"Raw response content repr: {repr(response.content)}")
            else:
                logger.error("Response has no content attribute")
            
            # Parse JSON response
            import json
            normalized_query = None  # Initialize to avoid NameError
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
                
                # Additional cleaning for common LLM issues
                logger.debug(f"Analyzing cleaned response: {repr(cleaned_response)}")
                
                if cleaned_response == '"normalized_query"' or cleaned_response.strip() == '"normalized_query"':
                    # LLM returned exactly just the field name in quotes
                    logger.warning(f"BRANCH 1: LLM returned just field name: {cleaned_response}")
                    normalized_query = query  # Use original query as fallback
                elif cleaned_response.startswith('"normalized_query"') and len(cleaned_response) < 50:
                    # LLM sometimes returns just the field name without value or incomplete
                    logger.warning(f"BRANCH 2: LLM returned incomplete JSON: {cleaned_response}")
                    normalized_query = query  # Use original query as fallback
                elif cleaned_response.startswith('"') and cleaned_response.endswith('"') and '"' not in cleaned_response[1:-1]:
                    # LLM sometimes returns just the field name in quotes
                    logger.warning(f"BRANCH 3: LLM returned just field name: {cleaned_response}")
                    normalized_query = query  # Use original query as fallback
                elif '"normalized_query"' in cleaned_response:
                    # Try to extract the value from partial JSON
                    try:
                        # Find the normalized_query field and extract its value
                        start_idx = cleaned_response.find('"normalized_query"')
                        if start_idx != -1:
                            # Find the colon after the field name
                            colon_idx = cleaned_response.find(':', start_idx)
                            if colon_idx != -1:
                                # Find the opening quote after the colon
                                quote_start = cleaned_response.find('"', colon_idx)
                                if quote_start != -1:
                                    # Find the closing quote
                                    quote_end = cleaned_response.find('"', quote_start + 1)
                                    if quote_end != -1:
                                        extracted_value = cleaned_response[quote_start + 1:quote_end]
                                        if extracted_value.strip():
                                            logger.info(f"Extracted normalized query from partial JSON: {extracted_value}")
                                            normalized_query = extracted_value
                                        else:
                                            normalized_query = query
                                    else:
                                        normalized_query = query
                                else:
                                    normalized_query = query
                            else:
                                normalized_query = query
                        else:
                            normalized_query = query
                    except Exception as e:
                        logger.warning(f"Failed to extract from partial JSON: {e}")
                        normalized_query = query
                else:
                    logger.debug(f"BRANCH 4: Attempting JSON parsing of: {repr(cleaned_response)}")
                    result = json.loads(cleaned_response)
                    normalized_query = result.get('normalized_query', query)
                    logger.debug(f"JSON parsing successful, normalized_query: {normalized_query}")
            except json.JSONDecodeError as e:
                # Log the actual response for debugging
                logger.warning(f"JSON parsing failed: {e}")
                logger.warning(f"Raw response: {response.content.strip()}")
                # Fallback if JSON parsing fails
                normalized_query = query  # Use original query instead of raw response
            
            # Ensure normalized_query is always set
            if normalized_query is None:
                logger.warning("normalized_query was never assigned, using original query")
                normalized_query = query
            
            # Fallback: if normalization failed or is too similar, use simple approach
            if not normalized_query or len(normalized_query) > len(query) * 0.8:
                # Simple fallback: remove common stop phrases
                stop_phrases = [
                    "tell me about", "i want to know", "please", "could you", "can you",
                    "what are", "what is", "how does", "how do", "explain", "describe"
                ]
                normalized_query = query.lower()
                for phrase in stop_phrases:
                    normalized_query = normalized_query.replace(phrase, "")
                normalized_query = " ".join(normalized_query.split())  # Clean up spaces
                
                if not normalized_query:
                    normalized_query = query  # Ultimate fallback
            
            logger.info(f"Query normalized: '{query}' -> '{normalized_query}'")
            return normalized_query
            
        except Exception as e:
            logger.warning(f"Error normalizing query, using original: {e}")
            logger.warning(f"Exception type: {type(e).__name__}")
            logger.warning(f"Exception details: {str(e)}")
            logger.warning(f"Exception repr: {repr(e)}")
            import traceback
            logger.warning(f"Full traceback: {traceback.format_exc()}")
            return query
    
    def _normalize_and_retrieve_development(self, query: str) -> Dict[str, Any]:
        """
        Development mode: Full normalization and tool selection (original behavior).
        
        Args:
            query: The user's input query
            
        Returns:
            Dictionary containing normalization results and retrieved information
        """
        logger.info("Running development mode normalization and retrieval")
        
        # First normalize the query
        normalized_result = self.normalize_query(query)
        
        # Get recommended tools
        recommended_tools = normalized_result.get('recommended_tools', [])
        identifiers = normalized_result.get('identifiers', {})
        query_type = normalized_result.get('query_type', 'general')
        
        # Enhanced tool execution with workflow awareness
        retrieved_contexts = []
        if self.tool_executor:
            retrieved_contexts = self._execute_tools_with_workflow_awareness(
                recommended_tools, query, identifiers, query_type
            )
        
        # Combine results
        result = {
            "normalization": normalized_result,
            "retrieved_contexts": retrieved_contexts,
            "total_contexts": len(retrieved_contexts)
        }
        
        logger.info(f"Development mode retrieved {len(retrieved_contexts)} contexts using {len(recommended_tools)} tools")
        return result

    def _execute_tools_with_workflow_awareness(self, recommended_tools: List[str], query: str, 
                                             identifiers: Dict[str, List[str]], query_type: str) -> List[Dict[str, Any]]:
        """
        Execute tools with enhanced workflow awareness for better results.
        
        Args:
            recommended_tools: List of recommended tool names
            query: Original query
            identifiers: Extracted identifiers
            query_type: Type of query
            
        Returns:
            List of context dictionaries from tool execution
        """
        retrieved_contexts = []
        executed_tools = set()
        
        # Handle special cases based on query type
        if query_type == "company_patents" and identifiers.get('companies'):
            # For company-patent queries, use exact company lookup + company patents lookup
            company_name = identifiers['companies'][0]
            
            # 1. Get exact company information
            if "exact_company_lookup" in self.tool_executor:
                try:
                    company_result = self.tool_executor["exact_company_lookup"](company_name)
                    retrieved_contexts.append({
                        "tool": "exact_company_lookup",
                        "input": company_name,
                        "result": company_result,
                        "execution_type": "primary"
                    })
                    executed_tools.add("exact_company_lookup")
                    logger.debug(f"Executed exact_company_lookup for {company_name}")
                except Exception as e:
                    logger.error(f"Error in exact_company_lookup: {e}")
            
            # 2. Get patents owned by that company using the new tool
            if "company_patents_lookup" in self.tool_executor:
                try:
                    patents_result = self.tool_executor["company_patents_lookup"](company_name)
                    retrieved_contexts.append({
                        "tool": "company_patents_lookup",
                        "input": company_name,
                        "result": patents_result,
                        "execution_type": "enhanced"
                    })
                    executed_tools.add("company_patents_lookup")
                    logger.debug(f"Executed company_patents_lookup for {company_name}")
                except Exception as e:
                    logger.error(f"Error in company_patents_lookup: {e}")
                    
                    # Fallback to patent_rag_retrieval if company_patents_lookup fails
                    if "patent_rag_retrieval" in self.tool_executor:
                        try:
                            enhanced_query = f"patents by {company_name} {query}"
                            patent_result = self.tool_executor["patent_rag_retrieval"](enhanced_query)
                            retrieved_contexts.append({
                                "tool": "patent_rag_retrieval",
                                "input": enhanced_query,
                                "result": patent_result,
                                "execution_type": "fallback"
                            })
                            executed_tools.add("patent_rag_retrieval")
                            logger.debug(f"Executed patent_rag_retrieval as fallback")
                        except Exception as fallback_e:
                            logger.error(f"Error in patent_rag_retrieval fallback: {fallback_e}")
        else:
            # Standard tool execution for other query types
            for tool_name in recommended_tools:
                if tool_name in self.tool_executor and tool_name not in executed_tools:
                    try:
                        logger.debug(f"Invoking tool: {tool_name}")
                        
                        # Prepare tool input based on tool type and identifiers
                        tool_input = self._prepare_tool_input(tool_name, query, identifiers)
                        
                        # Invoke the tool
                        tool_result = self.tool_executor[tool_name](tool_input)
                        
                        retrieved_contexts.append({
                            "tool": tool_name,
                            "input": tool_input,
                            "result": tool_result,
                            "execution_type": "standard"
                        })
                        
                        executed_tools.add(tool_name)
                        logger.debug(f"Tool {tool_name} returned {len(str(tool_result))} characters")
                        
                    except Exception as e:
                        logger.error(f"Error invoking tool {tool_name}: {e}")
                        retrieved_contexts.append({
                            "tool": tool_name,
                            "error": str(e),
                            "execution_type": "error"
                        })
                else:
                    if tool_name not in self.tool_executor:
                        logger.warning(f"Recommended tool {tool_name} not available")
        
        return retrieved_contexts

    def _prepare_tool_input(self, tool_name: str, query: str, identifiers: Dict[str, List[str]]) -> str:
        """
        Prepare appropriate input for different tools based on the tool type and query.
        
        Args:
            tool_name: Name of the tool to invoke
            query: Original user query (may contain "Previous context:" format)
            identifiers: Extracted identifiers (companies, patents)
            
        Returns:
            Formatted input string for the tool
        """
        companies = identifiers.get('companies', [])
        patents = identifiers.get('patents', [])
        
        # Extract actual question from enhanced context format
        actual_query = self._extract_actual_question(query)
        
        if tool_name == "exact_company_lookup" and companies:
            return companies[0]  # Use first company
        elif tool_name == "exact_patent_lookup" and patents:
            return patents[0]  # Use first patent
        elif tool_name == "company_patents_lookup" and companies:
            return companies[0]  # Use first company for company patents lookup
        elif tool_name in ["company_rag_retrieval", "patent_rag_retrieval", "hybrid_rag_retrieval", "enhanced_hybrid_rag_retrieval", "optimized_hybrid_rag_retrieval"]:
            return query  # Use full query (with context) for RAG tools as they can benefit from context
        else:
            return actual_query  # Use extracted question for other tools
    
    def _extract_actual_question(self, query: str) -> str:
        """
        Extract the actual question from enhanced context format.
        
        Args:
            query: Query that may contain "Previous context:" and "Current question:"
            
        Returns:
            The actual question without previous context
        """
        if "Current question:" in query:
            # Extract everything after "Current question:"
            parts = query.split("Current question:", 1)
            if len(parts) > 1:
                return parts[1].strip()
        
        return query  # Return original if no special format
    
    def _apply_exact_matching_heuristics(self, query: str, normalized_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply heuristics to improve exact matching for simple queries.
        
        Args:
            query: Original query
            normalized_result: Result from LLM normalization
            
        Returns:
            Enhanced normalized result with better tool selection
        """
        query_lower = query.lower().strip()
        identifiers = normalized_result.get('identifiers', {})
        companies = identifiers.get('companies', [])
        patents = identifiers.get('patents', [])
        
        # Heuristic 1: Single company name queries should use exact lookup
        if (len(companies) == 1 and len(patents) == 0 and 
            len(query.split()) <= 3 and 
            not any(word in query_lower for word in ['compare', 'find', 'search', 'analyze', 'trends', 'strategies'])):
            
            logger.info(f"Applying exact company lookup heuristic for: {query}")
            normalized_result['query_type'] = 'company'
            normalized_result['recommended_tools'] = ['exact_company_lookup']
            normalized_result['reasoning'] += ' [Enhanced with exact matching heuristic]'
            return normalized_result
        
        # Heuristic 2: Single patent ID queries should use exact lookup
        if (len(patents) == 1 and len(companies) == 0 and 
            len(query.split()) <= 4 and
            any(word in query_lower for word in ['patent', 'tell me about', 'information about', 'details about'])):
            
            logger.info(f"Applying exact patent lookup heuristic for: {query}")
            normalized_result['query_type'] = 'patent'
            normalized_result['recommended_tools'] = ['exact_patent_lookup']
            normalized_result['reasoning'] += ' [Enhanced with exact matching heuristic]'
            return normalized_result
        
        # Heuristic 3: "Company X patents" should use company_patents_lookup
        if (len(companies) == 1 and 
            any(word in query_lower for word in ['patents', 'patent portfolio', 'intellectual property', 'ip']) and
            not any(word in query_lower for word in ['compare', 'analyze', 'trends', 'market'])):
            
            logger.info(f"Applying company patents lookup heuristic for: {query}")
            normalized_result['query_type'] = 'company_patents'
            normalized_result['recommended_tools'] = ['exact_company_lookup', 'company_patents_lookup']
            normalized_result['reasoning'] += ' [Enhanced with company patents heuristic]'
            return normalized_result
        
        # Heuristic 4: Queries about companies with specific attributes should use company search tools
        if (any(phrase in query_lower for phrase in ['companies with', 'companies that', 'firms with', 'businesses with', 'find companies']) and
            any(word in query_lower for word in ['tech', 'technology', 'ai', 'machine learning', 'biotech', 'semiconductor', 'name', 'called'])):
            
            logger.info(f"Applying company substring search heuristic for: {query}")
            normalized_result['query_type'] = 'company'
            normalized_result['recommended_tools'] = ['company_rag_retrieval', 'optimized_hybrid_rag_retrieval']
            normalized_result['reasoning'] += ' [Enhanced with company substring search heuristic]'
            return normalized_result
        
        # Heuristic 5: Simple lookup queries should prefer exact tools over hybrid
        simple_lookup_patterns = ['what is', 'tell me about', 'information about', 'details about', 'show me']
        if (any(pattern in query_lower for pattern in simple_lookup_patterns) and
            len(companies) == 1 and len(patents) == 0):
            
            current_tools = normalized_result.get('recommended_tools', [])
            if 'optimized_hybrid_rag_retrieval' in current_tools and 'exact_company_lookup' not in current_tools:
                logger.info(f"Enhancing simple lookup with exact company lookup for: {query}")
                normalized_result['recommended_tools'] = ['exact_company_lookup'] + current_tools
                normalized_result['reasoning'] += ' [Enhanced with exact lookup priority]'
            return normalized_result
        
        # Return original result if no heuristics apply
        return normalized_result

    def _parse_normalization_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the JSON response from the normalization agent.
        
        Args:
            response: Raw response string from the LLM
            
        Returns:
            Parsed normalization result dictionary
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
            
            # Validate and set defaults
            required_fields = ['query_type', 'identifiers', 'recommended_tools', 'reasoning']
            for field in required_fields:
                if field not in result:
                    if field == 'query_type':
                        result[field] = 'general'
                    elif field == 'identifiers':
                        result[field] = {'companies': [], 'patents': []}
                    elif field == 'recommended_tools':
                        result[field] = ['enhanced_hybrid_rag_retrieval']
                    elif field == 'reasoning':
                        result[field] = 'Query successfully normalized'
            
            # Ensure identifiers structure
            if 'companies' not in result['identifiers']:
                result['identifiers']['companies'] = []
            if 'patents' not in result['identifiers']:
                result['identifiers']['patents'] = []
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Cleaned response: {response}")
            # Return fallback result
            return {
                "query_type": "general",
                "identifiers": {"companies": [], "patents": []},
                "recommended_tools": ["enhanced_hybrid_rag_retrieval"],
                "reasoning": f"JSON parsing error: {str(e)}"
            }
        
        except Exception as e:
            logger.error(f"Error parsing normalization response: {e}")
            # Return fallback result
            return {
                "query_type": "general",
                "identifiers": {"companies": [], "patents": []},
                "recommended_tools": ["enhanced_hybrid_rag_retrieval"],
                "reasoning": f"Parsing error: {str(e)}"
            }

 
        
        
