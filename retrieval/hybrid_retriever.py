"""
HybridRetriever: Combines dense (vector) and sparse (BM25) retrieval for hybrid search.
Implements proper score fusion and deduplication.
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

from langchain.retrievers import BM25Retriever
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List, Dict, Any
import numpy as np
from collections import defaultdict

logger = get_logger(__name__)

class HybridRetriever:
    """
    HybridRetriever combines dense (vector) and sparse (BM25) retrieval.
    It returns a merged, score-fused list of relevant documents.
    """
    def __init__(self, vectorstore: Chroma, documents: List[str], metadatas: List[Dict] = None):
        """
        Args:
            vectorstore: A Langchain Chroma vectorstore instance
            documents: List of raw text documents for BM25
            metadatas: Optional list of metadata dicts for each document
        """
        logger.info("Initializing HybridRetriever")
        
        self.vectorstore = vectorstore
        self.documents = documents
        self.metadatas = metadatas or [{} for _ in documents]
        
        # Initialize BM25 retriever
        try:
            # Create Document objects for BM25
            doc_objects = []
            for i, doc_text in enumerate(documents):
                metadata = self.metadatas[i] if i < len(self.metadatas) else {}
                metadata['doc_id'] = i  # Add unique document ID
                doc_objects.append(Document(page_content=doc_text, metadata=metadata))
            
            self.bm25_retriever = BM25Retriever.from_documents(doc_objects)
            logger.info(f"BM25 retriever initialized with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error initializing BM25 retriever: {e}")
            self.bm25_retriever = None

    def retrieve(self, query: str, k: int = 5, dense_weight: float = 0.5, sparse_weight: float = 0.5) -> List[Document]:
        """
        Retrieve top-k relevant documents using both dense and sparse methods with score fusion.
        
        Args:
            query: The user query string
            k: Number of top documents to retrieve
            dense_weight: Weight for dense (vector) scores
            sparse_weight: Weight for sparse (BM25) scores
            
        Returns:
            List of unique relevant documents with fused scores
        """
        logger.info(f"Performing hybrid retrieval for query: '{query}' (k={k})")
        
        # Get dense (vector) results
        dense_results = self._get_dense_results(query, k * 2)  # Get more to allow for fusion
        
        # Get sparse (BM25) results
        sparse_results = self._get_sparse_results(query, k * 2)  # Get more to allow for fusion
        
        # Fuse and rank results
        fused_results = self._fuse_results(dense_results, sparse_results, dense_weight, sparse_weight)
        
        # Return top-k results
        final_results = fused_results[:k]
        logger.info(f"Hybrid retrieval completed, returning {len(final_results)} results")
        
        return final_results
    
    def _get_dense_results(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Get dense retrieval results from vectorstore."""
        try:
            # Get similarity search with scores
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            dense_results = []
            for doc, score in results:
                # Convert distance to similarity score (Chroma returns distance, lower is better)
                similarity_score = 1.0 / (1.0 + score)  # Convert distance to similarity
                
                dense_results.append({
                    'document': doc,
                    'score': similarity_score,
                    'source': 'dense',
                    'doc_id': doc.metadata.get('doc_id', hash(doc.page_content))
                })
            
            logger.info(f"Retrieved {len(dense_results)} dense results")
            return dense_results
            
        except Exception as e:
            logger.error(f"Error in dense retrieval: {e}")
            return []
    
    def _get_sparse_results(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Get sparse retrieval results from BM25."""
        if not self.bm25_retriever:
            logger.warning("BM25 retriever not available")
            return []
        
        try:
            # BM25 retriever returns documents without scores, so we'll use rank-based scoring
            results = self.bm25_retriever.get_relevant_documents(query)[:k]
            
            sparse_results = []
            for i, doc in enumerate(results):
                # Use rank-based scoring (higher rank = lower score)
                rank_score = 1.0 / (i + 1)  # Score decreases with rank
                
                sparse_results.append({
                    'document': doc,
                    'score': rank_score,
                    'source': 'sparse',
                    'doc_id': doc.metadata.get('doc_id', hash(doc.page_content))
                })
            
            logger.info(f"Retrieved {len(sparse_results)} sparse results")
            return sparse_results
            
        except Exception as e:
            logger.error(f"Error in sparse retrieval: {e}")
            return []
    
    def _fuse_results(self, dense_results: List[Dict], sparse_results: List[Dict], 
                     dense_weight: float, sparse_weight: float) -> List[Document]:
        """
        Fuse dense and sparse results using weighted score combination.
        """
        logger.info("Fusing dense and sparse results")
        
        # Normalize scores within each method
        dense_scores = [r['score'] for r in dense_results]
        sparse_scores = [r['score'] for r in sparse_results]
        
        # Min-max normalization
        if dense_scores:
            min_dense, max_dense = min(dense_scores), max(dense_scores)
            if max_dense > min_dense:
                for result in dense_results:
                    result['normalized_score'] = (result['score'] - min_dense) / (max_dense - min_dense)
            else:
                for result in dense_results:
                    result['normalized_score'] = 1.0
        
        if sparse_scores:
            min_sparse, max_sparse = min(sparse_scores), max(sparse_scores)
            if max_sparse > min_sparse:
                for result in sparse_results:
                    result['normalized_score'] = (result['score'] - min_sparse) / (max_sparse - min_sparse)
            else:
                for result in sparse_results:
                    result['normalized_score'] = 1.0
        
        # Combine results by document ID
        combined_scores = defaultdict(lambda: {'dense': 0, 'sparse': 0, 'document': None})
        
        # Add dense scores
        for result in dense_results:
            doc_id = result['doc_id']
            combined_scores[doc_id]['dense'] = result['normalized_score']
            combined_scores[doc_id]['document'] = result['document']
        
        # Add sparse scores
        for result in sparse_results:
            doc_id = result['doc_id']
            combined_scores[doc_id]['sparse'] = result['normalized_score']
            if combined_scores[doc_id]['document'] is None:
                combined_scores[doc_id]['document'] = result['document']
        
        # Calculate final scores
        final_results = []
        for doc_id, scores in combined_scores.items():
            final_score = (dense_weight * scores['dense'] + sparse_weight * scores['sparse'])
            
            # Add comprehensive scoring information to document metadata
            doc = scores['document']
            if doc:
                doc.metadata['hybrid_score'] = final_score
                doc.metadata['dense_score'] = scores['dense']
                doc.metadata['sparse_score'] = scores['sparse']
                doc.metadata['retrieval_method'] = 'hybrid'
                doc.metadata['score_components'] = {
                    'dense_weight': dense_weight,
                    'sparse_weight': sparse_weight,
                    'dense_contribution': dense_weight * scores['dense'],
                    'sparse_contribution': sparse_weight * scores['sparse']
                }
                
                # Add confidence level based on score
                if final_score >= 0.8:
                    confidence = "high"
                elif final_score >= 0.6:
                    confidence = "medium"
                else:
                    confidence = "low"
                doc.metadata['confidence'] = confidence
                
                final_results.append((doc, final_score))
        
        # Sort by final score (descending)
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return documents only
        fused_documents = [doc for doc, score in final_results]
        
        logger.info(f"Fused {len(fused_documents)} unique documents")
        return fused_documents
    
    def retrieve_with_sources(self, query: str, k: int = 5, dense_weight: float = 0.5, 
                             sparse_weight: float = 0.5) -> Dict[str, Any]:
        """
        Retrieve documents with comprehensive source information and scoring details.
        
        Args:
            query: The user query string
            k: Number of top documents to retrieve
            dense_weight: Weight for dense (vector) scores  
            sparse_weight: Weight for sparse (BM25) scores
            
        Returns:
            Dictionary with documents, sources, scores, and metadata
        """
        logger.info(f"Performing hybrid retrieval with sources for query: '{query}' (k={k})")
        
        # Get documents using existing retrieve method
        documents = self.retrieve(query, k, dense_weight, sparse_weight)
        
        # Extract source information
        sources = []
        scored_results = []
        
        for doc in documents:
            # Extract source information
            source_info = {
                'content': doc.page_content,
                'source_id': doc.metadata.get('doc_id', 'unknown'),
                'source_type': doc.metadata.get('source_type', 'document'),
                'source_name': doc.metadata.get('source_name', 'Unknown Source'),
                'hybrid_score': doc.metadata.get('hybrid_score', 0),
                'confidence': doc.metadata.get('confidence', 'unknown'),
                'retrieval_method': 'hybrid'
            }
            
            sources.append(source_info)
            scored_results.append({
                'content': doc.page_content,
                'score': doc.metadata.get('hybrid_score', 0),
                'source': doc.metadata.get('source_name', 'Unknown Source'),
                'confidence': doc.metadata.get('confidence', 'unknown')
            })
        
        return {
            'query': query,
            'total_results': len(documents),
            'documents': documents,
            'sources': sources,
            'scored_results': scored_results,
            'retrieval_config': {
                'k': k,
                'dense_weight': dense_weight,
                'sparse_weight': sparse_weight
            }
        } 