"""
Patent-related tools for exact lookup and RAG retrieval.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from patent_rag import PatentRAG
from config.rag_config import patent_config
import pandas as pd
from typing import Dict, List, Any, Optional

logger = get_logger(__name__)

class PatentTools:
    def __init__(self, patent_df: pd.DataFrame, index_dir: str):
        """Initialize patent tools with data and RAG system."""
        self.patent_df = patent_df
        self.patent_rag = PatentRAG(df=patent_df, index_dir=index_dir, config=patent_config)
        logger.info("PatentTools initialized")
    
    def get_exact_patent_info(self, patent_id: str) -> Dict[str, Any]:
        """
        Get exact patent information by patent ID.
        
        Args:
            patent_id: Patent ID to search for
            
        Returns:
            Dict containing patent information or error
        """
        logger.info(f"Looking up exact patent info for: {patent_id}")
        
        try:
            # Search by patent_id
            matches = self.patent_df[
                self.patent_df['patent_id'].astype(str).str.contains(patent_id, case=False, na=False)
            ]
            
            if matches.empty:
                logger.warning(f"No exact match found for patent: {patent_id}")
                return {
                    "success": False,
                    "message": f"No exact match found for patent: {patent_id}",
                    "data": None
                }
            
            # Return first match (most relevant)
            patent_data = matches.iloc[0].to_dict()
            logger.info(f"Found exact patent match: {patent_data.get('patent_id', 'N/A')}")
            
            return {
                "success": True,
                "message": "Patent information retrieved successfully",
                "data": {
                    "patent_id": patent_data.get("patent_id", ""),
                    "company_name": patent_data.get("company_name", ""),
                    "company_id": patent_data.get("company_id", ""),
                    "full_text": patent_data.get("full_text", ""),
                    "abstract": patent_data.get("abstract", "")
                }
            }
            
        except Exception as e:
            logger.error(f"Error in exact patent lookup: {e}")
            return {
                "success": False,
                "message": f"Error retrieving patent information: {str(e)}",
                "data": None
            }
    
    def retrieve_patent_contexts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve patent contexts using RAG.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant patent contexts
        """
        logger.info(f"Retrieving patent contexts for query: {query} (top_k={top_k})")
        
        try:
            contexts = self.patent_rag.retrieve_patent_contexts(query, top_k=top_k)
            logger.info(f"Retrieved {len(contexts)} patent contexts")
            return contexts
            
        except Exception as e:
            logger.error(f"Error retrieving patent contexts: {e}")
            return []

# Global instance - will be initialized in main
patent_tools_instance: Optional[PatentTools] = None

def get_patent_tools() -> Optional[PatentTools]:
    """Get the global patent tools instance."""
    return patent_tools_instance

def init_patent_tools(patent_df: pd.DataFrame, index_dir: str):
    """
    Initialize the global patent tools instance and return a dictionary of tool functions.
    """
    global patent_tools_instance
    patent_tools_instance = PatentTools(patent_df, index_dir)
    logger.info("Global patent tools instance initialized")
    return {
        "exact_patent_lookup": exact_patent_lookup_tool,
        "patent_rag_retrieval": patent_rag_retrieval_tool,
    }

# Tool functions for LangChain integration
def exact_patent_lookup_tool(patent_id: str) -> str:
    """Tool function for exact patent lookup."""
    tools = get_patent_tools()
    if not tools:
        return "Patent tools not initialized"
    
    result = tools.get_exact_patent_info(patent_id)
    if result["success"]:
        data = result["data"]
        return f"Patent ID: {data['patent_id']}\nCompany: {data['company_name']}\nCompany ID: {data['company_id']}\nAbstract: {data['abstract']}\nFull Text: {data['full_text']}"
    else:
        return result["message"]

def patent_rag_retrieval_tool(query: str, top_k: int = 5) -> str:
    """Tool function for patent RAG retrieval."""
    tools = get_patent_tools()
    if not tools:
        return "Patent tools not initialized"
    
    contexts = tools.retrieve_patent_contexts(query, top_k)
    if not contexts:
        return "No relevant patent information found"
    
    result = "Relevant Patent Information:\n"
    for i, context in enumerate(contexts, 1):
        result += f"\n{i}. Patent {context['patent_id']} ({context['company_name']})\n{context['chunk']}\n"
    
    return result 

