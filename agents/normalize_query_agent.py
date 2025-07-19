"""
NormalizeQueryAgent: Classifies and normalizes user queries using LLM.
Determines if query is about company, patent, or general technology/market analysis.
Enhanced with tool integration and comprehensive prompts.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from agents.base import BaseAgent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage, HumanMessage
from config.prompts import (
    NORMALIZE_AGENT_SYSTEM_PROMPT,
    NORMALIZE_AGENT_USER_PROMPT,
    DEFAULT_MODELS
)
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
        
        logger.info("NormalizeQueryAgent initialized successfully")
    
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
        """Register available tools for the agent."""
        super().register_tools(tools)
        self.available_tools = list(tools.keys()) if tools else []
        logger.info(f"Registered {len(self.available_tools)} tools: {self.available_tools}")
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean the response text to extract pure JSON."""
        # Remove markdown code blocks
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*$', '', response_text)
        
        # Find JSON object in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()
        
        return response_text.strip()
    
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
            # Create the prompt
            user_prompt = NORMALIZE_AGENT_USER_PROMPT.format(query=query)
            
            # Prepare messages
            messages = [
                SystemMessage(content=NORMALIZE_AGENT_SYSTEM_PROMPT),
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
            
            logger.info(f"Query normalized as '{normalized_result.get('query_type', 'unknown')}' type")
            
            return normalized_result
            
        except Exception as e:
            logger.error(f"Error during query normalization: {str(e)}")
            # Return fallback result
            return {
                "query_type": "general",
                "identifiers": {"companies": [], "patents": []},
                "recommended_tools": ["hybrid_rag_retrieval"],
                "reasoning": f"Error during normalization: {str(e)}",
                "error": str(e)
            }
    
    def normalize_and_retrieve(self, query: str) -> Dict[str, Any]:
        """
        Normalize a query and invoke recommended tools for information retrieval.
        
        Args:
            query: The user's input query
            
        Returns:
            Dictionary containing normalization results and retrieved information
        """
        logger.info(f"Normalizing and retrieving information for: {query}")
        
        try:
            # First normalize the query
            normalized_result = self.normalize_query(query)
            
            # Get recommended tools
            recommended_tools = normalized_result.get('recommended_tools', [])
            identifiers = normalized_result.get('identifiers', {})
            
            # Invoke recommended tools
            retrieved_contexts = []
            if self.tool_executor:
                for tool_name in recommended_tools:
                    if tool_name in self.tool_executor:
                        try:
                            logger.debug(f"Invoking tool: {tool_name}")
                            
                            # Prepare tool input based on tool type and identifiers
                            tool_input = self._prepare_tool_input(tool_name, query, identifiers)
                            
                            # Invoke the tool
                            tool_result = self.tool_executor[tool_name](tool_input)
                            
                            retrieved_contexts.append({
                                "tool": tool_name,
                                "input": tool_input,
                                "result": tool_result
                            })
                            
                            logger.debug(f"Tool {tool_name} returned {len(str(tool_result))} characters")
                            
                        except Exception as e:
                            logger.error(f"Error invoking tool {tool_name}: {e}")
                            retrieved_contexts.append({
                                "tool": tool_name,
                                "error": str(e)
                            })
                    else:
                        logger.warning(f"Recommended tool {tool_name} not available")
            
            # Combine results
            result = {
                "normalization": normalized_result,
                "retrieved_contexts": retrieved_contexts,
                "total_contexts": len(retrieved_contexts)
            }
            
            logger.info(f"Retrieved {len(retrieved_contexts)} contexts using {len(recommended_tools)} tools")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during normalize and retrieve: {e}")
            return {
                "normalization": {"error": str(e)},
                "retrieved_contexts": [],
                "total_contexts": 0,
                "error": str(e)
            }

    # This is defined but not used yet
    def run(self, input_data: dict) -> dict:
        """
        Normalize and classify the user query (maintaining backward compatibility).
        
        Args:
            input_data: dict containing "question" key
            
        Returns:
            dict: Classification results with category, identifiers, and normalized query
        """
        query = input_data.get("question", "")
        logger.info(f"Normalizing query: {query}")
        
        if not query:
            logger.warning("Empty query received")
            return {
                "category": "general",
                "company_names": [],
                "patent_ids": [],
                "keywords": [],
                "normalized_query": ""
            }
        
        try:
            # Use new normalization method
            normalized_result = self.normalize_query(query)
            
            # Convert to legacy format for backward compatibility
            legacy_result = {
                "category": normalized_result.get("query_type", "general"),
                "company_names": normalized_result.get("identifiers", {}).get("companies", []),
                "patent_ids": normalized_result.get("identifiers", {}).get("patents", []),
                "keywords": [],  # Can be extracted from reasoning if needed
                "normalized_query": query,
                "recommended_tools": normalized_result.get("recommended_tools", []),
                "reasoning": normalized_result.get("reasoning", "")
            }
            
            logger.info(f"Query classified as: {legacy_result.get('category', 'unknown')}")
            return legacy_result
                
        except Exception as e:
            logger.error(f"Error in query normalization: {e}")
            return self._simple_classify(query)
    
    def _prepare_tool_input(self, tool_name: str, query: str, identifiers: Dict[str, List[str]]) -> str:
        """
        Prepare appropriate input for different tools based on the tool type and query.
        
        Args:
            tool_name: Name of the tool to invoke
            query: Original user query
            identifiers: Extracted identifiers (companies, patents)
            
        Returns:
            Formatted input string for the tool
        """
        companies = identifiers.get('companies', [])
        patents = identifiers.get('patents', [])
        
        if tool_name == "exact_company_lookup" and companies:
            return companies[0]  # Use first company
        elif tool_name == "exact_patent_lookup" and patents:
            return patents[0]  # Use first patent
        elif tool_name in ["company_rag_retrieval", "patent_rag_retrieval", "hybrid_rag_retrieval"]:
            return query  # Use original query for RAG tools
        else:
            return query  # Default to original query
    
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
                        result[field] = ['hybrid_rag_retrieval']
                    elif field == 'reasoning':
                        result[field] = 'Query successfully normalized'
            
            # Ensure identifiers structure
            if 'companies' not in result['identifiers']:
                result['identifiers']['companies'] = []
            if 'patents' not in result['identifiers']:
                result['identifiers']['patents'] = []
            
            # Ensure recommended_tools is a list
            if not isinstance(result['recommended_tools'], list):
                result['recommended_tools'] = ['hybrid_rag_retrieval']
            
            logger.debug("Successfully parsed normalization response")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response}")
            # Return fallback result
            return {
                "query_type": "general",
                "identifiers": {"companies": [], "patents": []},
                "recommended_tools": ["hybrid_rag_retrieval"],
                "reasoning": f"JSON parsing error: {e}",
                "error": str(e)
            }
        
        except Exception as e:
            logger.error(f"Error parsing normalization response: {e}")
            return {
                "query_type": "general", 
                "identifiers": {"companies": [], "patents": []},
                "recommended_tools": ["hybrid_rag_retrieval"],
                "reasoning": f"Parsing error: {e}",
                "error": str(e)
            }

    def _simple_classify(self, query: str) -> dict:
        """Fallback classification method."""
        logger.info("Using fallback classification method")
        query_lower = query.lower()
        
        if "company" in query_lower or "firm" in query_lower:
            category = "company"
        elif "patent" in query_lower or "technology" in query_lower:
            category = "patent"
        else:
            category = "general"
            
        return {
            "category": category,
            "company_names": [],
            "patent_ids": [],
            "keywords": query.split(),
            "normalized_query": query
        } 
        
        
