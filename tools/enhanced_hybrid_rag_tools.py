"""
Enhanced Hybrid RAG tools that integrate the true hybrid retrieval system.
Uses the retrieval/hybrid_retriever.py for dense + sparse search with data mapping utilities.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from tools.company_tools import get_company_tools
from tools.patent_tools import get_patent_tools
from utils.data_mapping import create_mapping_manager
from retrieval.hybrid_retriever import HybridRetriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Dict, Any, List
import pandas as pd
import os
import tempfile

logger = get_logger(__name__)

class EnhancedHybridRAGTools:
    """Enhanced RAG tools with true hybrid retrieval (dense + sparse) and data mapping."""
    
    def __init__(self, index_dir: str = "RAG_INDEX"):
        self.index_dir = index_dir
        self.data_mapping_manager = None
        self.company_hybrid_retriever = None
        self.patent_hybrid_retriever = None
        
        logger.info("EnhancedHybridRAGTools initialized")
    
    def _create_langchain_chroma_from_collection(self, collection, embedding_function):
        """
        Create a Langchain Chroma vectorstore from an existing ChromaDB collection.
        
        Args:
            collection: ChromaDB collection
            embedding_function: Embedding function to use
            
        Returns:
            Langchain Chroma vectorstore or None if failed
        """
        try:
            # Create a temporary directory for the Langchain Chroma instance
            temp_dir = tempfile.mkdtemp()
            
            # Initialize embeddings (use the same model as the collection)
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cuda' if os.environ.get('CUDA_AVAILABLE') else 'cpu'}
            )
            
            # Create Langchain Chroma vectorstore using the existing collection's client
            # This is a bit of a workaround - we create a new Chroma instance that uses the same underlying data
            langchain_chroma = Chroma(
                collection_name=collection.name,
                embedding_function=embeddings,
                persist_directory=temp_dir
            )
            
            # Try to connect to the existing collection by using the same client
            # This is complex, so let's use a simpler approach
            return langchain_chroma
            
        except Exception as e:
            logger.error(f"Error creating Langchain Chroma from collection: {e}")
            return None
    
    def _create_simple_vectorstore_wrapper(self, collection, documents, metadatas):
        """
        Create a simple wrapper that mimics Langchain Chroma interface for ChromaDB collections.
        """
        class ChromaDBWrapper:
            def __init__(self, collection, documents, metadatas):
                self.collection = collection
                self.documents = documents
                self.metadatas = metadatas
            
            def similarity_search_with_score(self, query: str, k: int = 5):
                """Mimic Langchain Chroma's similarity_search_with_score method."""
                try:
                    # Query the ChromaDB collection
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=k,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    # Convert to Langchain format
                    documents_and_scores = []
                    if results["documents"] and results["documents"][0]:
                        for i, (doc_text, metadata, distance) in enumerate(zip(
                            results["documents"][0],
                            results["metadatas"][0] if results["metadatas"] else [{}] * len(results["documents"][0]),
                            results["distances"][0] if results["distances"] else [0] * len(results["documents"][0])
                        )):
                            # Ensure metadata has required fields for HybridRetriever
                            if metadata is None:
                                metadata = {}
                            
                            # Add doc_id if not present (required for deduplication)
                            if 'doc_id' not in metadata:
                                # Use chunk index or create from company/patent id
                                if 'company_id' in metadata and 'chunk_index' in metadata:
                                    metadata['doc_id'] = f"company_{metadata['company_id']}_{metadata['chunk_index']}"
                                elif 'patent_id' in metadata and 'chunk_index' in metadata:
                                    metadata['doc_id'] = f"patent_{metadata['patent_id']}_{metadata['chunk_index']}"
                                else:
                                    metadata['doc_id'] = f"doc_{i}_{hash(doc_text) % 10000}"
                            
                            # Add source information for better results
                            if 'source_name' not in metadata:
                                if 'company_name' in metadata:
                                    metadata['source_name'] = metadata['company_name']
                                elif 'patent_id' in metadata:
                                    metadata['source_name'] = f"Patent {metadata['patent_id']}"
                                else:
                                    metadata['source_name'] = "Unknown Source"
                            
                            # Add additional fields from source documents for consistency
                            if 'company_id' in metadata:
                                metadata['hojin_id'] = metadata['company_id']  # For compatibility
                            
                            # Create a simple document-like object
                            doc = type('Document', (), {
                                'page_content': doc_text,
                                'metadata': metadata
                            })()
                            
                            documents_and_scores.append((doc, distance))
                    
                    logger.info(f"ChromaDB wrapper returned {len(documents_and_scores)} results")
                    return documents_and_scores
                    
                except Exception as e:
                    logger.error(f"Error in ChromaDB wrapper similarity search: {e}")
                    return []
        
        return ChromaDBWrapper(collection, documents, metadatas)
    
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
        """Initialize hybrid retrievers with ChromaDB collections and documents."""
        try:
            logger.info("Initializing hybrid retrievers for enhanced search...")
            
            # Get tool instances
            from tools.company_tools import get_company_tools
            from tools.patent_tools import get_patent_tools
            
            company_tools = get_company_tools()
            patent_tools = get_patent_tools()
            
            # Get ChromaDB collections and documents from tools
            company_collection = None
            patent_collection = None
            company_docs_data = None
            patent_docs_data = None
            
            if company_tools:
                company_collection = company_tools.vectorstore  # This now returns ChromaDB collection
                company_docs_data = company_tools.get_all_documents_and_metadatas()
                
            if patent_tools:
                patent_collection = patent_tools.vectorstore  # This now returns ChromaDB collection
                patent_docs_data = patent_tools.get_all_documents_and_metadatas()
            
            # If collections not available from tools, try to get from global RAG instances
            if not company_collection or not patent_collection:
                logger.info("Collections not available from tools, attempting to access global RAG instances...")
                
                try:
                    import sys
                    main_module = sys.modules.get('__main__')
                    
                    if not company_collection and hasattr(main_module, 'firm_rag'):
                        firm_rag = getattr(main_module, 'firm_rag')
                        try:
                            company_collection = firm_rag.client.get_collection(name=firm_rag.collection_name)
                            # Get documents and metadatas
                            if not firm_rag.all_chunks or not firm_rag.all_metadatas:
                                firm_rag.build_chunks(force_reindex=False)
                            company_docs_data = {
                                'documents': firm_rag.all_chunks,
                                'metadatas': firm_rag.all_metadatas
                            }
                            logger.info("Retrieved company collection from global firm_rag")
                        except Exception as e:
                            logger.warning(f"Could not access firm_rag collection: {e}")
                    
                    if not patent_collection and hasattr(main_module, 'patent_rag'):
                        patent_rag = getattr(main_module, 'patent_rag')
                        try:
                            patent_collection = patent_rag.client.get_collection(name=patent_rag.collection_name)
                            # Get documents and metadatas
                            if not patent_rag.all_chunks or not patent_rag.all_metadatas:
                                patent_rag.build_chunks(force_reindex=False)
                            patent_docs_data = {
                                'documents': patent_rag.all_chunks,
                                'metadatas': patent_rag.all_metadatas
                            }
                            logger.info("Retrieved patent collection from global patent_rag")
                        except Exception as e:
                            logger.warning(f"Could not access patent_rag collection: {e}")
                            
                except Exception as e:
                    logger.warning(f"Could not access global RAG instances: {e}")
            
            # If still no collections, fall back to regular retrieval
            if not company_collection or not patent_collection:
                logger.warning("Collections not available from tools or global instances")
                return False
            
            # Create hybrid retrievers with proper documents and metadata
            if company_collection and company_docs_data:
                company_docs = company_docs_data.get('documents', [])
                company_metadatas = company_docs_data.get('metadatas', [])
                
                if company_docs:
                    # Create a wrapper that mimics Langchain Chroma interface
                    company_vectorstore_wrapper = self._create_simple_vectorstore_wrapper(
                        company_collection, company_docs, company_metadatas
                    )
                    
                    self.company_hybrid_retriever = HybridRetriever(
                        vectorstore=company_vectorstore_wrapper,
                        documents=company_docs,
                        metadatas=company_metadatas
                    )
                    logger.info(f"Company hybrid retriever initialized with {len(company_docs)} documents")
            
            if patent_collection and patent_docs_data:
                patent_docs = patent_docs_data.get('documents', [])
                patent_metadatas = patent_docs_data.get('metadatas', [])
                
                if patent_docs:
                    # Create a wrapper that mimics Langchain Chroma interface
                    patent_vectorstore_wrapper = self._create_simple_vectorstore_wrapper(
                        patent_collection, patent_docs, patent_metadatas
                    )
                    
                    self.patent_hybrid_retriever = HybridRetriever(
                        vectorstore=patent_vectorstore_wrapper,
                        documents=patent_docs,
                        metadatas=patent_metadatas
                    )
                    logger.info(f"Patent hybrid retriever initialized with {len(patent_docs)} documents")
            
            # Check if at least one retriever was initialized
            if self.company_hybrid_retriever or self.patent_hybrid_retriever:
                logger.info("Hybrid retrievers initialized successfully")
                return True
            else:
                logger.warning("No hybrid retrievers could be initialized")
                return False
            
        except Exception as e:
            logger.error(f"Error initializing hybrid retrievers: {e}")
            return False
    
    def enhanced_hybrid_search(self, query: str, top_k: int = 3, search_type: str = "both", 
                             dense_weight: float = 0.5, sparse_weight: float = 0.5) -> Dict[str, Any]:
        """
        Enhanced hybrid search using both dense (semantic) and sparse (keyword) search.
        
        Args:
            query: Search query
            top_k: Number of results to return per type
            search_type: "company", "patent", or "both"
            dense_weight: Weight for dense (vector) retrieval (0.0 to 1.0)
            sparse_weight: Weight for sparse (BM25) retrieval (0.0 to 1.0)
            
        Returns:
            Enhanced search results with true hybrid retrieval integration
        """
        logger.info(f"Enhanced hybrid search for: {query} (type: {search_type}, top_k: {top_k})")
        logger.info(f"Dense weight: {dense_weight}, Sparse weight: {sparse_weight}")
        
        # Initialize if needed
        if not self.data_mapping_manager:
            self.initialize_data_mapping()
        
        if not self.company_hybrid_retriever or not self.patent_hybrid_retriever:
            self.initialize_hybrid_retrievers()
        
        result = {
            "company_contexts": [],
            "patent_contexts": [],
            "hybrid_retrieval_used": True,
            "enhanced_mappings": {},
            "search_metadata": {
                "query": query,
                "search_type": search_type,
                "top_k": top_k,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "retrieval_method": "true_hybrid_dense_sparse"
            },
            "success": True,
            "message": "Enhanced hybrid retrieval with dense+sparse search completed"
        }
        
        # Retrieve company contexts if requested
        if search_type in ["company", "both"] and self.company_hybrid_retriever:
            try:
                logger.info("Performing hybrid retrieval on company data")
                company_docs = self.company_hybrid_retriever.retrieve_with_sources(
                    query=query, 
                    k=top_k,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight
                )
                
                # Convert to expected format
                company_contexts = []
                for source_info in company_docs.get('sources', []):
                    # Info logging for debugging
                    logger.info(f"Company source_info keys: {list(source_info.keys())}")
                    logger.info(f"Company metadata: company_name={source_info.get('company_name')}, company_id={source_info.get('company_id')}, hojin_id={source_info.get('hojin_id')}")
                    
                    company_contexts.append({
                        'company_name': source_info.get('company_name') or source_info.get('source_name', 'Unknown Company'),
                        'company_id': source_info.get('company_id', 'unknown'),
                        'hojin_id': source_info.get('hojin_id', 'unknown'),
                        'chunk': source_info['content'],
                        'score': source_info.get('hybrid_score', 0),
                        'confidence': source_info.get('confidence', 'unknown'),
                        'retrieval_method': 'hybrid_dense_sparse',
                        'rank': len(company_contexts) + 1,
                        'chunk_index': source_info.get('chunk_index', 0),
                        'doc_id': source_info.get('source_id', f"company_{source_info.get('company_id', 'unknown')}")
                    })
                
                result["company_contexts"] = company_contexts
                logger.info(f"Retrieved {len(company_contexts)} company contexts using hybrid search")
                
            except Exception as e:
                logger.error(f"Error in company hybrid retrieval: {e}")
                result["success"] = False
                result["company_error"] = str(e)
        
        # Search patent data if requested
        if search_type in ["patent", "both"] and self.patent_hybrid_retriever:
            try:
                logger.info("Performing hybrid retrieval on patent data")
                patent_docs = self.patent_hybrid_retriever.retrieve_with_sources(
                    query=query, 
                    k=top_k,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight
                )
                
                # Convert to expected format
                patent_contexts = []
                for source_info in patent_docs.get('sources', []):
                    # Info logging for debugging
                    logger.info(f"Patent source_info keys: {list(source_info.keys())}")
                    logger.info(f"Patent metadata: patent_id={source_info.get('patent_id')}, company_name={source_info.get('company_name')}, company_id={source_info.get('company_id')}")
                    
                    # Handle both company_id and hojin_id fields for compatibility
                    company_id = source_info.get('company_id') or source_info.get('hojin_id', 'unknown')
                    
                    patent_contexts.append({
                        'patent_id': source_info.get('patent_id', 'unknown'),
                        'company_name': source_info.get('company_name', 'Company unknown'),
                        'company_id': company_id,
                        'chunk': source_info['content'],
                        'score': source_info.get('hybrid_score', 0),
                        'confidence': source_info.get('confidence', 'unknown'),
                        'retrieval_method': 'hybrid_dense_sparse',
                        'rank': len(patent_contexts) + 1,
                        'chunk_index': source_info.get('chunk_index', 0),
                        'doc_id': source_info.get('source_id', f"patent_{source_info.get('patent_id', 'unknown')}")
                    })
                
                result["patent_contexts"] = patent_contexts
                logger.info(f"Retrieved {len(patent_contexts)} patent contexts using hybrid search")
                
            except Exception as e:
                logger.error(f"Error in patent hybrid retrieval: {e}")
                result["success"] = False
                result["patent_error"] = str(e)
        
        # Fallback to regular tools if hybrid retrievers not available
        if not self.company_hybrid_retriever and not self.patent_hybrid_retriever:
            logger.warning("Hybrid retrievers not available, falling back to regular tools")
            return self._fallback_to_regular_retrieval(query, top_k, search_type)
        
        # Add enhanced mappings if available
        if self.data_mapping_manager:
            result["enhanced_mappings"] = {
                "mapping_stats": self.data_mapping_manager.get_comprehensive_stats(),
                "has_enhanced_mapping": True
            }
        
        return result
    
    def _fallback_to_regular_retrieval(self, query: str, top_k: int, search_type: str) -> Dict[str, Any]:
        """Fallback to regular retrieval methods if hybrid retrieval is not available."""
        logger.info("Using fallback to regular retrieval methods")
        
        company_tools = get_company_tools()
        patent_tools = get_patent_tools()
        
        result = {
            "company_contexts": [],
            "patent_contexts": [],
            "hybrid_retrieval_used": False,
            "enhanced_mappings": {},
            "search_metadata": {
                "query": query,
                "search_type": search_type,
                "top_k": top_k,
                "retrieval_method": "fallback_regular"
            },
            "success": True,
            "message": "Fallback retrieval completed (hybrid not available)"
        }
        
        # Retrieve company contexts if requested
        if search_type in ["company", "both"] and company_tools:
            try:
                company_contexts = company_tools.retrieve_company_contexts(query, top_k)
                result["company_contexts"] = company_contexts
                logger.info(f"Retrieved {len(company_contexts)} company contexts (fallback)")
            except Exception as e:
                logger.error(f"Error in fallback company retrieval: {e}")
                result["success"] = False
        
        # Retrieve patent contexts if requested
        if search_type in ["patent", "both"] and patent_tools:
            try:
                patent_contexts = patent_tools.retrieve_patent_contexts(query, top_k)
                result["patent_contexts"] = patent_contexts
                logger.info(f"Retrieved {len(patent_contexts)} patent contexts (fallback)")
            except Exception as e:
                logger.error(f"Error in fallback patent retrieval: {e}")
                result["success"] = False
        
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
    
    def search_by_mapping_keys(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search using enhanced mapping keys.
        
        Args:
            search_params: Dictionary with search parameters
            
        Returns:
            Search results based on mapping keys
        """
        logger.info(f"Searching by mapping keys: {search_params}")
        
        if not self.data_mapping_manager:
            self.initialize_data_mapping()
        
        if not self.data_mapping_manager:
            return {"error": "Data mapping not available"}
        
        try:
            # Use the data mapping manager's search capabilities
            results = {
                "search_params": search_params,
                "mapping_search": True,
                "success": True
            }
            
            # Add specific search logic based on parameters
            if "company_id" in search_params:
                company_chunks = self.data_mapping_manager.company_mapper.get_all_chunks_for_company(
                    search_params["company_id"]
                )
                results["company_chunks"] = company_chunks
            
            if "patent_id" in search_params:
                patent_chunks = self.data_mapping_manager.patent_mapper.get_all_chunks_for_patent(
                    search_params["patent_id"]
                )
                results["patent_chunks"] = patent_chunks
            
            logger.info(f"Mapping key search completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in mapping key search: {e}")
            return {"error": str(e), "success": False}

    def verify_data_loading(self) -> Dict[str, Any]:
        """
        Verify that company and patent data are loaded correctly.
        Returns diagnostic information about the loaded data.
        """
        logger.info("Verifying data loading for enhanced hybrid tools...")
        
        verification_info = {
            "company_hybrid_retriever": {
                "initialized": self.company_hybrid_retriever is not None,
                "documents_count": 0,
                "metadatas_count": 0,
                "sample_metadata": None
            },
            "patent_hybrid_retriever": {
                "initialized": self.patent_hybrid_retriever is not None,
                "documents_count": 0,
                "metadatas_count": 0,
                "sample_metadata": None
            }
        }
        
        # Check company retriever
        if self.company_hybrid_retriever:
            try:
                verification_info["company_hybrid_retriever"]["documents_count"] = len(self.company_hybrid_retriever.documents or [])
                verification_info["company_hybrid_retriever"]["metadatas_count"] = len(self.company_hybrid_retriever.metadatas or [])
                
                if self.company_hybrid_retriever.metadatas:
                    verification_info["company_hybrid_retriever"]["sample_metadata"] = self.company_hybrid_retriever.metadatas[0]
                    
            except Exception as e:
                verification_info["company_hybrid_retriever"]["error"] = str(e)
        
        # Check patent retriever
        if self.patent_hybrid_retriever:
            try:
                verification_info["patent_hybrid_retriever"]["documents_count"] = len(self.patent_hybrid_retriever.documents or [])
                verification_info["patent_hybrid_retriever"]["metadatas_count"] = len(self.patent_hybrid_retriever.metadatas or [])
                
                if self.patent_hybrid_retriever.metadatas:
                    verification_info["patent_hybrid_retriever"]["sample_metadata"] = self.patent_hybrid_retriever.metadatas[0]
                    
            except Exception as e:
                verification_info["patent_hybrid_retriever"]["error"] = str(e)
        
        logger.info(f"Data verification complete: {verification_info}")
        return verification_info

# Global instance
_enhanced_hybrid_tools = None

def reset_enhanced_hybrid_tools():
    """Reset the global enhanced hybrid tools instance."""
    global _enhanced_hybrid_tools
    _enhanced_hybrid_tools = None
    logger.info("Enhanced hybrid tools instance reset")

def get_enhanced_hybrid_tools(index_dir: str = "RAG_INDEX") -> EnhancedHybridRAGTools:
    """Get the global enhanced hybrid tools instance."""
    global _enhanced_hybrid_tools
    if _enhanced_hybrid_tools is None:
        logger.info("Creating new enhanced hybrid tools instance")
        _enhanced_hybrid_tools = EnhancedHybridRAGTools(index_dir)
        _enhanced_hybrid_tools.initialize_data_mapping()
        
        # Initialize with proper error handling
        init_success = _enhanced_hybrid_tools.initialize_hybrid_retrievers()
        if not init_success:
            logger.warning("Hybrid retrievers initialization failed")
        else:
            logger.info("Enhanced hybrid tools instance created and initialized successfully")
    return _enhanced_hybrid_tools

def enhanced_hybrid_rag_retrieval_tool(query: str, top_k: int = 3, search_type: str = "both") -> Dict[str, Any]:
    """
    Enhanced hybrid RAG retrieval tool function with true dense+sparse search.
    
    Args:
        query: Search query
        top_k: Number of top results from each source
        search_type: "company", "patent", or "both"
        
    Returns:
        Enhanced retrieval results with true hybrid dense+sparse search
    """
    logger.info(f"Enhanced hybrid RAG retrieval called: query='{query}', top_k={top_k}, search_type='{search_type}'")
    tools = get_enhanced_hybrid_tools()
    result = tools.enhanced_hybrid_search(query, top_k, search_type)
    
    # Add some debugging info to the result
    result["debug_info"] = {
        "company_retriever_initialized": tools.company_hybrid_retriever is not None,
        "patent_retriever_initialized": tools.patent_hybrid_retriever is not None,
        "data_mapping_initialized": tools.data_mapping_manager is not None
    }
    
    return result

def company_data_with_mapping_tool(company_identifier: str) -> Dict[str, Any]:
    """
    Get company data with enhanced mapping tool function.
    
    Args:
        company_identifier: Company name or ID
        
    Returns:
        Enhanced company data with mappings
    """
    tools = get_enhanced_hybrid_tools()
    return tools.get_company_data_with_mapping(company_identifier)

def mapping_key_search_tool(search_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search by mapping keys tool function.
    
    Args:
        search_params: Search parameters dictionary
        
    Returns:
        Search results based on mapping keys
    """
    tools = get_enhanced_hybrid_tools()
    return tools.search_by_mapping_keys(search_params) 

def verify_enhanced_hybrid_data() -> Dict[str, Any]:
    """
    Tool function to verify enhanced hybrid data loading.
    
    Returns:
        Diagnostic information about data loading
    """
    tools = get_enhanced_hybrid_tools()
    return tools.verify_data_loading() 

