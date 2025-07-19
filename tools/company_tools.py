"""
Company-related tools for exact lookup and RAG retrieval.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from firm_summary_rag import FirmSummaryRAG
from config.rag_config import firm_config
import pandas as pd
from typing import Dict, List, Any, Optional

logger = get_logger(__name__)

class CompanyTools:
    def __init__(self, firm_df: pd.DataFrame, index_dir: str):
        """Initialize company tools with data and RAG system."""
        self.firm_df = firm_df
        self.firm_rag = FirmSummaryRAG(df=firm_df, index_dir=index_dir, config=firm_config)
        logger.info("CompanyTools initialized")
    
    def get_exact_company_info(self, company_identifier: str) -> Dict[str, Any]:
        """
        Get exact company information by name or hojin_id.
        
        Args:
            company_identifier: Company name or hojin_id
            
        Returns:
            Dict containing company information or error
        """
        logger.info(f"Looking up exact company info for: {company_identifier}")
        
        try:
            # Search by company name
            name_match = self.firm_df[
                self.firm_df['company_name'].str.contains(company_identifier, case=False, na=False)
            ]
            
            # Search by hojin_id
            id_match = self.firm_df[
                self.firm_df['hojin_id'].astype(str).str.contains(company_identifier, case=False, na=False)
            ]
            
            # Combine results
            matches = pd.concat([name_match, id_match]).drop_duplicates()
            
            if matches.empty:
                logger.warning(f"No exact match found for company: {company_identifier}")
                return {
                    "success": False,
                    "message": f"No exact match found for company: {company_identifier}",
                    "data": None
                }
            
            # Return first match (most relevant)
            company_data = matches.iloc[0].to_dict()
            logger.info(f"Found exact company match: {company_data.get('company_name', 'N/A')}")
            
            return {
                "success": True,
                "message": "Company information retrieved successfully",
                "data": {
                    "company_name": company_data.get("company_name", ""),
                    "hojin_id": company_data.get("hojin_id", ""),
                    "company_keywords": company_data.get("company_keywords", ""),
                    "summary": company_data.get("summary", "")
                }
            }
            
        except Exception as e:
            logger.error(f"Error in exact company lookup: {e}")
            return {
                "success": False,
                "message": f"Error retrieving company information: {str(e)}",
                "data": None
            }
    
    def retrieve_company_contexts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve company contexts using RAG.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant company contexts
        """
        logger.info(f"Retrieving company contexts for query: {query} (top_k={top_k})")
        
        try:
            contexts = self.firm_rag.retrieve_firm_contexts(query, top_k=top_k)
            logger.info(f"Retrieved {len(contexts)} company contexts")
            return contexts
            
        except Exception as e:
            logger.error(f"Error retrieving company contexts: {e}")
            return []

# Global instance - will be initialized in main
company_tools_instance: Optional[CompanyTools] = None

def get_company_tools() -> Optional[CompanyTools]:
    """Get the global company tools instance."""
    return company_tools_instance

def init_company_tools(firm_df: pd.DataFrame, index_dir: str):
    """
    Initialize the global company tools instance and return a dictionary of tool functions.
    """
    global company_tools_instance
    company_tools_instance = CompanyTools(firm_df, index_dir)
    logger.info("Global company tools instance initialized")
    return {
        "exact_company_lookup": exact_company_lookup_tool,
        "company_rag_retrieval": company_rag_retrieval_tool,
    }

# Tool functions for LangChain integration
def exact_company_lookup_tool(company_identifier: str) -> str:
    """Tool function for exact company lookup."""
    tools = get_company_tools()
    if not tools:
        return "Company tools not initialized"
    
    result = tools.get_exact_company_info(company_identifier)
    if result["success"]:
        data = result["data"]
        return f"Company: {data['company_name']}\nHojin ID: {data['hojin_id']}\nKeywords: {data['company_keywords']}\nSummary: {data['summary']}"
    else:
        return result["message"]

def company_rag_retrieval_tool(query: str, top_k: int = 5) -> str:
    """Tool function for company RAG retrieval."""
    tools = get_company_tools()
    if not tools:
        return "Company tools not initialized"
    
    contexts = tools.retrieve_company_contexts(query, top_k)
    if not contexts:
        return "No relevant company information found"
    
    result = "Relevant Company Information:\n"
    for i, context in enumerate(contexts, 1):
        result += f"\n{i}. {context['company_name']}\n{context['chunk']}\n"
    
    return result 