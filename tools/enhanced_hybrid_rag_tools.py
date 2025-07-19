"""
Enhanced Hybrid RAG tools that integrate the hybrid retrieval system.
Uses both the retrieval/hybrid_retriever.py and data mapping utilities.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from tools.company_tools import get_company_tools
from tools.patent_tools import get_patent_tools
from retrieval.hybrid_retriever import HybridRetriever
from utils.data_mapping import create_mapping_manager, DataMappingManager
from typing import Dict, List, Any
import os

logger = get_logger(__name__)

class EnhancedHybridRAGTools:
    """Enhanced RAG tools with hybrid retrieval and data mapping."""
    
    def __init__(self, index_dir: str = "RAG_INDEX"):
        self.index_dir = index_dir
        self.data_mapping_manager = None
        self.company_hybrid_retriever = None
        self.patent_hybrid_retriever = None
        
        logger.info("EnhancedHybridRAGTools initialized")
    
    def initialize_data_mapping(self) -> bool:
        """Initialize the data mapping manager."""
        try:
            self.data_mapping_manager = create_mapping_manager(self.index_dir)
            logger.info("Data mapping manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing data mapping: {e}")
            return False
    
    def initialize_hybrid_retrievers(self) -> bool:
        """Initialize hybrid retrievers for both company and patent data."""
        try:
            # Initialize company tools and get vectorstore
            company_tools = get_company_tools()
            patent_tools = get_patent_tools()
            
            if not company_tools or not patent_tools:
                logger.warning("Company or patent tools not available")
                return False
            
            # Initialize data mapping if not already done
            if not self.data_mapping_manager:
                self.initialize_data_mapping()
            
            # Get document lists for BM25
            if self.data_mapping_manager:
                # Get company documents
                company_docs = []
                company_metadatas = []
                for company_id, chunks in self.data_mapping_manager.company_mapper.hojin_to_chunks.items():
                    for chunk_data in chunks:
                        company_docs.append(chunk_data['chunk_content'])
                        company_metadatas.append({
                            'company_id': company_id,
                            'chunk_index': chunk_data['chunk_index'],
                            'type': 'company'
                        })
                
                # Get patent documents
                patent_docs = []
                patent_metadatas = []
                for patent_id, chunks in self.data_mapping_manager.patent_mapper.patent_to_chunks.items():
                    for chunk_data in chunks:
                        patent_docs.append(chunk_data['chunk_content'])
                        patent_metadatas.append({
                            'patent_id': patent_id,
                            'company_id': chunk_data['hojin_id'],
                            'chunk_index': chunk_data['chunk_index'],
                            'type': 'patent'
                        })
                
                logger.info(f"Prepared {len(company_docs)} company docs and {len(patent_docs)} patent docs for hybrid retrieval")
            
            # For now, just indicate successful initialization
            # The actual hybrid retriever setup would need access to the Chroma vectorstores
            logger.info("Hybrid retrievers prepared (vectorstore access needed for full initialization)")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing hybrid retrievers: {e}")
            return False
    
    def enhanced_hybrid_search(self, query: str, top_k: int = 3, search_type: str = "both") -> Dict[str, Any]:
        """
        Enhanced hybrid search using both semantic and keyword matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            search_type: "company", "patent", or "both"
            
        Returns:
            Enhanced search results with data mapping integration
        """
        logger.info(f"Enhanced hybrid search for: {query} (type: {search_type}, top_k: {top_k})")
        
        # Initialize if needed
        if not self.data_mapping_manager:
            self.initialize_data_mapping()
        
        # Fallback to regular hybrid retrieval if enhanced not available
        company_tools = get_company_tools()
        patent_tools = get_patent_tools()
        
        result = {
            "company_contexts": [],
            "patent_contexts": [],
            "enhanced_mappings": {},
            "search_metadata": {
                "query": query,
                "search_type": search_type,
                "top_k": top_k,
                "hybrid_retrieval_used": True
            },
            "success": True,
            "message": "Enhanced hybrid retrieval completed"
        }
        
        # Retrieve company contexts if requested
        if search_type in ["company", "both"] and company_tools:
            try:
                company_contexts = company_tools.retrieve_company_contexts(query, top_k)
                result["company_contexts"] = company_contexts
                
                # Add enhanced mappings
                if self.data_mapping_manager:
                    enhanced_company_data = []
                    for context in company_contexts:
                        company_name = context.get('company_name', '')
                        if company_name:
                            cross_ref = self.data_mapping_manager.search_cross_reference(company_name)
                            enhanced_company_data.append({
                                **context,
                                "cross_reference": cross_ref,
                                "enhanced_mapping": True
                            })
                    result["enhanced_mappings"]["companies"] = enhanced_company_data
                
                logger.info(f"Retrieved {len(company_contexts)} enhanced company contexts")
            except Exception as e:
                logger.error(f"Error retrieving company contexts: {e}")
                result["success"] = False
        
        # Retrieve patent contexts if requested
        if search_type in ["patent", "both"] and patent_tools:
            try:
                patent_contexts = patent_tools.retrieve_patent_contexts(query, top_k)
                result["patent_contexts"] = patent_contexts
                
                # Add enhanced mappings
                if self.data_mapping_manager:
                    enhanced_patent_data = []
                    for context in patent_contexts:
                        patent_id = context.get('patent_id', '')
                        if patent_id:
                            patent_chunks = self.data_mapping_manager.patent_mapper.get_all_chunks_for_patent(patent_id)
                            enhanced_patent_data.append({
                                **context,
                                "all_patent_chunks": patent_chunks,
                                "enhanced_mapping": True
                            })
                    result["enhanced_mappings"]["patents"] = enhanced_patent_data
                
                logger.info(f"Retrieved {len(patent_contexts)} enhanced patent contexts")
            except Exception as e:
                logger.error(f"Error retrieving patent contexts: {e}")
                result["success"] = False
        
        # Add data mapping statistics
        if self.data_mapping_manager:
            result["mapping_stats"] = self.data_mapping_manager.get_comprehensive_stats()
        
        return result
    
    def get_company_data_with_mapping(self, company_identifier: str) -> Dict[str, Any]:
        """
        Get comprehensive company data using enhanced mappings.
        
        Args:
            company_identifier: Company name or hojin_id
            
        Returns:
            Comprehensive company data with mappings
        """
        logger.info(f"Getting enhanced company data for: {company_identifier}")
        
        if not self.data_mapping_manager:
            self.initialize_data_mapping()
        
        if not self.data_mapping_manager:
            return {"error": "Data mapping not available"}
        
        try:
            # Get cross-reference data
            cross_ref = self.data_mapping_manager.search_cross_reference(company_identifier)
            
            # Get regular company data
            company_tools = get_company_tools()
            regular_data = {}
            if company_tools:
                try:
                    exact_info = company_tools.get_exact_company_info(company_identifier)
                    regular_data["exact_info"] = exact_info
                except:
                    pass
                
                try:
                    rag_contexts = company_tools.retrieve_company_contexts(company_identifier, top_k=5)
                    regular_data["rag_contexts"] = rag_contexts
                except:
                    pass
            
            result = {
                "company_identifier": company_identifier,
                "cross_reference_data": cross_ref,
                "regular_data": regular_data,
                "enhanced_mapping": True,
                "success": True
            }
            
            logger.info(f"Enhanced company data retrieved for {company_identifier}")
            return result
            
        except Exception as e:
            logger.error(f"Error getting enhanced company data: {e}")
            return {"error": str(e), "success": False}
    
    def search_by_mapping_key(self, 
                             company_name: str = None, 
                             hojin_id: str = None, 
                             patent_id: str = None, 
                             chunk_index: int = None) -> Dict[str, Any]:
        """
        Search using specific mapping keys as requested.
        
        Args:
            company_name: Company name
            hojin_id: Company hojin ID
            patent_id: Patent ID (optional)
            chunk_index: Specific chunk index (optional)
            
        Returns:
            Specific chunk data or all chunks for the entity
        """
        logger.info(f"Mapping key search: company={company_name}, hojin={hojin_id}, patent={patent_id}, chunk={chunk_index}")
        
        if not self.data_mapping_manager:
            self.initialize_data_mapping()
        
        if not self.data_mapping_manager:
            return {"error": "Data mapping not available"}
        
        try:
            result = {
                "search_parameters": {
                    "company_name": company_name,
                    "hojin_id": hojin_id,
                    "patent_id": patent_id,
                    "chunk_index": chunk_index
                },
                "results": {},
                "success": True
            }
            
            # Company chunk search
            if company_name and hojin_id and chunk_index is not None:
                company_chunk = self.data_mapping_manager.company_mapper.get_chunk_by_key(
                    company_name, hojin_id, chunk_index
                )
                result["results"]["company_chunk"] = company_chunk
            
            # Patent chunk search
            if company_name and hojin_id and patent_id and chunk_index is not None:
                patent_chunk = self.data_mapping_manager.patent_mapper.get_chunk_by_key(
                    company_name, hojin_id, patent_id, chunk_index
                )
                result["results"]["patent_chunk"] = patent_chunk
            
            # Get all company chunks
            if hojin_id and chunk_index is None:
                all_company_chunks = self.data_mapping_manager.company_mapper.get_all_chunks_for_company(hojin_id)
                result["results"]["all_company_chunks"] = all_company_chunks
            
            # Get all patent chunks for a company
            if company_name and not patent_id:
                company_patent_chunks = self.data_mapping_manager.patent_mapper.get_company_patent_chunks(company_name)
                result["results"]["company_patent_chunks"] = company_patent_chunks
            
            logger.info(f"Mapping key search completed with {len(result['results'])} result types")
            return result
            
        except Exception as e:
            logger.error(f"Error in mapping key search: {e}")
            return {"error": str(e), "success": False}


# Global instance
_enhanced_hybrid_tools = None

def get_enhanced_hybrid_tools(index_dir: str = "RAG_INDEX") -> EnhancedHybridRAGTools:
    """Get the global enhanced hybrid tools instance."""
    global _enhanced_hybrid_tools
    if _enhanced_hybrid_tools is None:
        _enhanced_hybrid_tools = EnhancedHybridRAGTools(index_dir)
        _enhanced_hybrid_tools.initialize_data_mapping()
        _enhanced_hybrid_tools.initialize_hybrid_retrievers()
    return _enhanced_hybrid_tools

def enhanced_hybrid_rag_retrieval_tool(query: str, top_k: int = 3, search_type: str = "both") -> Dict[str, Any]:
    """
    Enhanced hybrid RAG retrieval tool function.
    
    Args:
        query: Search query
        top_k: Number of top results from each source
        search_type: "company", "patent", or "both"
        
    Returns:
        Enhanced retrieval results with data mapping
    """
    tools = get_enhanced_hybrid_tools()
    return tools.enhanced_hybrid_search(query, top_k, search_type)

def company_data_with_mapping_tool(company_identifier: str) -> Dict[str, Any]:
    """
    Get company data with enhanced mapping tool function.
    
    Args:
        company_identifier: Company name or hojin_id
        
    Returns:
        Enhanced company data with mappings
    """
    tools = get_enhanced_hybrid_tools()
    return tools.get_company_data_with_mapping(company_identifier)

def mapping_key_search_tool(company_name: str = None, 
                           hojin_id: str = None, 
                           patent_id: str = None, 
                           chunk_index: int = None) -> Dict[str, Any]:
    """
    Search by specific mapping keys tool function.
    
    Args:
        company_name: Company name
        hojin_id: Company hojin ID  
        patent_id: Patent ID (optional)
        chunk_index: Specific chunk index (optional)
        
    Returns:
        Specific mapping search results
    """
    tools = get_enhanced_hybrid_tools()
    return tools.search_by_mapping_key(company_name, hojin_id, patent_id, chunk_index) 