"""
Hybrid RAG tool that combines company and patent retrieval.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from tools.company_tools import get_company_tools
from tools.patent_tools import get_patent_tools
from typing import Dict, List, Any

logger = get_logger(__name__)

def hybrid_rag_retrieval_tool(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Retrieve both company and patent contexts for a query.
    
    Args:
        query: Search query
        top_k: Number of top results from each source
        
    Returns:
        Dict containing both company and patent contexts
    """
    logger.info(f"Performing hybrid RAG retrieval for query: {query} (top_k={top_k})")
    
    company_tools = get_company_tools()
    patent_tools = get_patent_tools()
    
    result = {
        "company_contexts": [],
        "patent_contexts": [],
        "success": True,
        "message": "Hybrid retrieval completed"
    }
    
    # Retrieve company contexts
    if company_tools:
        try:
            company_contexts = company_tools.retrieve_company_contexts(query, top_k)
            result["company_contexts"] = company_contexts
            logger.info(f"Retrieved {len(company_contexts)} company contexts")
        except Exception as e:
            logger.error(f"Error retrieving company contexts: {e}")
            result["success"] = False
    else:
        logger.warning("Company tools not initialized")
    
    # Retrieve patent contexts
    if patent_tools:
        try:
            patent_contexts = patent_tools.retrieve_patent_contexts(query, top_k)
            result["patent_contexts"] = patent_contexts
            logger.info(f"Retrieved {len(patent_contexts)} patent contexts")
        except Exception as e:
            logger.error(f"Error retrieving patent contexts: {e}")
            result["success"] = False
    else:
        logger.warning("Patent tools not initialized")
    
    return result

def format_hybrid_retrieval_result(result: Dict[str, Any]) -> str:
    """Format hybrid retrieval result for LLM consumption."""
    if not result["success"]:
        return "Error in hybrid retrieval"
    
    formatted = "=== HYBRID RAG RETRIEVAL RESULTS ===\n\n"
    
    # Format company contexts
    if result["company_contexts"]:
        formatted += "--- COMPANY INFORMATION ---\n"
        for i, context in enumerate(result["company_contexts"], 1):
            formatted += f"{i}. {context['company_name']}\n{context['chunk']}\n\n"
    
    # Format patent contexts
    if result["patent_contexts"]:
        formatted += "--- PATENT INFORMATION ---\n"
        for i, context in enumerate(result["patent_contexts"], 1):
            formatted += f"{i}. Patent {context['patent_id']} ({context['company_name']})\n{context['chunk']}\n\n"
    
    return formatted 


