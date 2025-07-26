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
            # Search by appln_id (the actual column name in the dataset)
            matches = self.patent_df[
                self.patent_df['appln_id'].astype(str).str.contains(patent_id, case=False, na=False)
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
            logger.info(f"Found exact patent match: {patent_data.get('appln_id', 'N/A')}")
            
            return {
                "success": True,
                "message": "Patent information retrieved successfully",
                "data": {
                    "patent_id": patent_data.get("appln_id", ""),  # Use appln_id as patent_id
                    "company_name": patent_data.get("company_name", ""),
                    "company_id": patent_data.get("hojin_id", ""),  # Use hojin_id as company_id
                    "full_text": patent_data.get("patent_abstract", ""),  # Use patent_abstract as full_text
                    "abstract": patent_data.get("patent_abstract", ""),  # Use patent_abstract column
                    "filing_date": patent_data.get("appln_filing_date", ""),  # Add filing date
                    "publication_number": patent_data.get("patpublnr", ""),  # Add publication number
                    "publication_date": patent_data.get("publn_date", "")  # Add publication date
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

    def get_patents_by_company(self, company_identifier: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Get patents owned by a specific company.
        
        Args:
            company_identifier: Company name or hojin_id
            top_k: Maximum number of patents to return
            
        Returns:
            Dict containing company patents or error
        """
        logger.info(f"Looking up patents for company: {company_identifier} (top_k={top_k})")
        
        try:
            # Search by company name - allows substring matching (case-insensitive)
            name_matches = self.patent_df[
                self.patent_df['company_name'].str.contains(company_identifier, case=False, na=False)
            ]
            
            # Search by hojin_id
            id_matches = self.patent_df[
                self.patent_df['hojin_id'].astype(str).str.contains(company_identifier, case=False, na=False)
            ]
            
            # Combine results and remove duplicates
            all_matches = pd.concat([name_matches, id_matches]).drop_duplicates()
            
            if all_matches.empty:
                logger.warning(f"No patents found for company: {company_identifier}")
                return {
                    "success": False,
                    "message": f"No patents found for company: {company_identifier}",
                    "data": None,
                    "patent_count": 0
                }
            
            # Limit results to top_k
            matches = all_matches.head(top_k)
            
            # Format patent data
            patents = []
            for _, patent in matches.iterrows():
                patents.append({
                    "patent_id": patent.get("appln_id", ""),
                    "company_name": patent.get("company_name", ""),
                    "company_id": patent.get("hojin_id", ""),
                    "abstract": patent.get("patent_abstract", ""),  # Use patent_abstract column
                    "full_abstract": patent.get("patent_abstract", ""),  # Ensure full abstract is preserved
                    "full_text": patent.get("patent_abstract", ""),  # Use patent_abstract as full text
                    "patent_summary": patent.get("patent_abstract", "")[:500] + "..." if len(str(patent.get("patent_abstract", ""))) > 500 else patent.get("patent_abstract", ""),  # Add truncated version for quick preview
                    "filing_date": patent.get("appln_filing_date", ""),  # Add filing date
                    "publication_number": patent.get("patpublnr", ""),  # Add publication number
                    "publication_date": patent.get("publn_date", "")  # Add publication date
                })
            
            logger.info(f"Found {len(patents)} patents for company: {company_identifier}")
            
            return {
                "success": True,
                "message": f"Found {len(patents)} patents for {company_identifier}",
                "data": {
                    "company_identifier": company_identifier,
                    "patent_count": len(all_matches),
                    "patents_returned": len(patents),
                    "patents": patents
                }
            }
            
        except Exception as e:
            logger.error(f"Error retrieving patents by company: {e}")
            return {
                "success": False,
                "message": f"Error retrieving patents for company: {str(e)}",
                "data": None,
                "patent_count": 0
            }

    @property
    def vectorstore(self):
        """Expose ChromaDB collection for hybrid retrieval."""
        try:
            collection = self.patent_rag.client.get_collection(name=self.patent_rag.collection_name)
            return collection
        except Exception as e:
            logger.warning(f"Could not access patent collection: {e}")
            return None
    
    @property
    def chroma_client(self):
        """Expose ChromaDB client."""
        return self.patent_rag.client if hasattr(self.patent_rag, 'client') else None
    
    def get_all_documents_and_metadatas(self):
        """Get all documents and metadatas for hybrid retrieval setup."""
        try:
            # Ensure chunks are loaded
            if not self.patent_rag.all_chunks or not self.patent_rag.all_metadatas:
                self.patent_rag.build_chunks(force_reindex=False)
            
            # Update metadatas to include proper doc_id and source_name for HybridRetriever
            updated_metadatas = []
            for i, metadata in enumerate(self.patent_rag.all_metadatas):
                updated_metadata = metadata.copy()
                
                # Add doc_id for deduplication
                patent_id = updated_metadata.get('patent_id', 'unknown')
                chunk_index = updated_metadata.get('chunk_index', i)
                updated_metadata['doc_id'] = f"patent_{patent_id}_{chunk_index}"
                
                # Add source_name for better results display
                if 'patent_id' in updated_metadata:
                    updated_metadata['source_name'] = f"Patent {updated_metadata['patent_id']}"
                else:
                    updated_metadata['source_name'] = f"Patent {patent_id}"
                
                # Add source type
                updated_metadata['source_type'] = 'patent'
                
                updated_metadatas.append(updated_metadata)
            
            return {
                'documents': self.patent_rag.all_chunks,
                'metadatas': updated_metadatas
            }
        except Exception as e:
            logger.error(f"Error getting patent documents: {e}")
            return {'documents': [], 'metadatas': []}

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
        "patent_rag_retrieval": patent_rag_retrieval_tool_wrapper,
        "company_patents_lookup": company_patents_lookup_tool,
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
        
        # Enhanced display with better abstract handling
        response = f"="*60 + "\n"
        response += f"PATENT INFORMATION\n"
        response += f"="*60 + "\n"
        response += f"Patent ID: {data['patent_id']}\n"
        response += f"Company: {data['company_name']}\n"
        response += f"Company ID: {data['company_id']}\n\n"
        
        # Handle abstract display with fallback
        abstract = data.get('abstract', '').strip()
        response += f"ABSTRACT:\n"
        if abstract and abstract != 'nan' and abstract.lower() != 'none':
            response += f"{abstract}\n\n"
        else:
            response += "[Abstract not available in dataset]\n\n"
        
        # Handle full text display
        full_text = data.get('full_text', '').strip()
        response += f"PATENT CONTENT:\n"
        if full_text and full_text != 'nan' and full_text.lower() != 'none':
            if len(full_text) > 2000:
                response += f"{full_text[:2000]}...\n[Content truncated - showing first 2000 characters]\n\n"
            else:
                response += f"{full_text}\n\n"
        else:
            response += "[Full patent content not available in dataset]\n\n"
        
        response += f"="*60 + "\n"
        return response
    else:
        return result["message"]

def company_patents_lookup_tool(company_identifier: str) -> str:
    """Tool function for looking up patents by company name or ID."""
    tools = get_patent_tools()
    if not tools:
        return "Patent tools not initialized"
    
    result = tools.get_patents_by_company(company_identifier, top_k=5)
    if result["success"]:
        data = result["data"]
        response = f"Company: {data['company_identifier']}\n"
        response += f"Total Patents Found: {data['patent_count']}\n"
        response += f"Patents Returned: {data['patents_returned']}\n\n"
        response += "="*60 + "\n"
        response += "DETAILED PATENT INFORMATION:\n"
        response += "="*60 + "\n\n"
        
        for i, patent in enumerate(data['patents'], 1):
            response += f"{i}. PATENT ID: {patent['patent_id']}\n"
            response += f"   Company: {patent['company_name']} (ID: {patent['company_id']})\n"
            response += f"   \n"
            
            # Enhanced abstract handling with fallback
            abstract = patent.get('abstract', '').strip()
            response += f"   ABSTRACT:\n"
            if abstract and abstract != 'nan' and abstract.lower() != 'none' and len(abstract) > 0:
                response += f"   {abstract}\n"
            else:
                response += f"   [Abstract not available for this patent]\n"
            response += f"   \n"
            
            # Enhanced patent content handling
            response += f"   PATENT CONTENT PREVIEW:\n"
            patent_content = patent.get('cleaned_patent', '').strip()
            if patent_content and patent_content != 'nan' and patent_content.lower() != 'none':
                if len(patent_content) > 1500:
                    response += f"   {patent_content[:1500]}...\n   [Content truncated - showing first 1500 characters]\n"
                else:
                    response += f"   {patent_content}\n"
            else:
                response += f"   [Patent content not available for this patent]\n"
            response += f"   \n"
            response += "-"*50 + "\n\n"
        
        # Add summary information
        response += "="*60 + "\n"
        response += "PATENT ANALYSIS SUMMARY:\n"
        response += "="*60 + "\n"
        response += f"• Total patents for {data['company_identifier']}: {data['patent_count']}\n"
        response += f"• Patents shown above: {data['patents_returned']}\n"
        response += f"• All patents include full abstracts and content previews\n"
        response += f"• Patent details can be used for comprehensive analysis\n"
        
        return response
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
    
    result = "="*60 + "\n"
    result += f"RELEVANT PATENT INFORMATION (Query: {query})\n"
    result += "="*60 + "\n\n"
    
    for i, context in enumerate(contexts, 1):
        result += f"{i}. PATENT {context.get('patent_id', 'Unknown')}\n"
        result += f"   Company: {context.get('company_name', 'Unknown')}\n"
        result += f"   Relevance: High (retrieved through semantic search)\n"
        result += f"   \n"
        result += f"   RELEVANT CONTENT:\n"
        chunk_content = context.get('chunk', '').strip()
        if chunk_content:
            # Format the chunk content nicely
            lines = chunk_content.split('\n')
            for line in lines:
                if line.strip():
                    result += f"   {line.strip()}\n"
        else:
            result += f"   [Content not available]\n"
        result += f"   \n"
        result += "-"*50 + "\n\n"
    
    result += f"Retrieved {len(contexts)} relevant patent documents for query analysis.\n"
    result += "="*60 + "\n"
    
    return result

# The reason for this wrapper is to ensure the tool can be called with a single parameter in LangChain
def patent_rag_retrieval_tool_wrapper(query: str) -> str:
    """Wrapper for patent RAG retrieval tool that handles single parameter calls."""
    return patent_rag_retrieval_tool(query, top_k=5) 

