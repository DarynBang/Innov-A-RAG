"""
Enhanced Data Chunking Mapping Utilities for InnovARAG

This module provides enhanced mapping capabilities for both company and patent data
that work alongside the existing RAG systems without modifying the original code.

Provides easy access to chunk mappings by:
- Company name + hojin_id + chunk_index → chunk_content
- Patent data with similar structure
- Various lookup utilities for better data access
"""
from utils.logging_utils import setup_logging, get_logger
setup_logging()

import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = get_logger(__name__)

class CompanyDataMapper:
    """Enhanced mapping for company data chunks."""
    
    def __init__(self, firm_rag_system=None, index_dir: str = "RAG_INDEX"):
        """
        Initialize company data mapper.
        
        Args:
            firm_rag_system: Optional FirmSummaryRAG instance
            index_dir: Index directory path
        """
        self.firm_rag = firm_rag_system
        self.index_dir = index_dir
        self.chunks_path = os.path.join(index_dir, "firm_summary_index", "chunks.json")
        self.mapping_path = os.path.join(index_dir, "firm_summary_index", "chunk_mapping.json")
        
        # Enhanced mapping structures
        self.company_chunk_map: Dict[str, Dict[str, Any]] = {}
        self.name_to_hojin_map: Dict[str, str] = {}
        self.hojin_to_chunks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        logger.info("CompanyDataMapper initialized")
    
    def load_mappings(self) -> bool:
        """Load and build enhanced mappings from existing data."""
        try:
            # Load existing chunks and metadata
            if not os.path.exists(self.chunks_path) or not os.path.exists(self.mapping_path):
                logger.warning("Company chunks or mapping files not found")
                return False
            
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                metadatas = json.load(f)
            
            if len(chunks) != len(metadatas):
                logger.error("Mismatch between chunks and metadata count")
                return False
            
            # Build enhanced mappings
            for i, (chunk, metadata) in enumerate(zip(chunks, metadatas)):
                company_id = metadata.get("company_id", "")
                company_name = metadata.get("company_name", "")
                company_keywords = metadata.get("company_keywords", "")
                chunk_index = metadata.get("chunk_index", 0)
                
                # Create unique key: company_name + hojin_id + chunk_index
                unique_key = f"{company_name}_{company_id}_{chunk_index}"
                
                self.company_chunk_map[unique_key] = {
                    "chunk_content": chunk,
                    "company_name": company_name,
                    "hojin_id": company_id,
                    "company_keywords": company_keywords,
                    "chunk_index": chunk_index,
                    "global_index": i
                }
                
                # Build additional lookup maps
                self.name_to_hojin_map[company_name] = company_id
                self.hojin_to_chunks[company_id].append({
                    "chunk_content": chunk,
                    "chunk_index": chunk_index,
                    "unique_key": unique_key
                })
            
            logger.info(f"Loaded {len(self.company_chunk_map)} company chunk mappings")
            return True
            
        except Exception as e:
            logger.error(f"Error loading company mappings: {e}")
            return False
    
    def get_chunk_by_key(self, company_name: str, hojin_id: str, chunk_index: int) -> Optional[Dict[str, Any]]:
        """
        Get chunk by company name + hojin_id + chunk_index.
        
        Args:
            company_name: Company name
            hojin_id: Company hojin ID
            chunk_index: Chunk index
            
        Returns:
            Chunk information or None if not found
        """
        unique_key = f"{company_name}_{hojin_id}_{chunk_index}"
        return self.company_chunk_map.get(unique_key)
    
    def get_all_chunks_for_company(self, hojin_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific company by hojin_id."""
        return self.hojin_to_chunks.get(hojin_id, [])
    
    def get_hojin_id_by_name(self, company_name: str) -> Optional[str]:
        """Get hojin_id by company name."""
        return self.name_to_hojin_map.get(company_name)
    
    def search_chunks_by_keywords(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search chunks by company keywords."""
        results = []
        keyword_set = set(keyword.lower() for keyword in keywords)
        
        for chunk_data in self.company_chunk_map.values():
            company_keywords = chunk_data.get("company_keywords", "").lower()
            if any(keyword in company_keywords for keyword in keyword_set):
                results.append(chunk_data)
        
        return results
    
    def get_mapping_stats(self) -> Dict[str, Any]:
        """Get statistics about the mappings."""
        return {
            "total_chunks": len(self.company_chunk_map),
            "unique_companies": len(self.name_to_hojin_map),
            "companies_with_chunks": len(self.hojin_to_chunks),
            "avg_chunks_per_company": sum(len(chunks) for chunks in self.hojin_to_chunks.values()) / max(len(self.hojin_to_chunks), 1)
        }


class PatentDataMapper:
    """Enhanced mapping for patent data chunks."""
    
    def __init__(self, patent_rag_system=None, index_dir: str = "RAG_INDEX"):
        """
        Initialize patent data mapper.
        
        Args:
            patent_rag_system: Optional PatentRAG instance
            index_dir: Index directory path
        """
        self.patent_rag = patent_rag_system
        self.index_dir = index_dir
        self.chunks_path = os.path.join(index_dir, "patent_chunks_index", "chunks.json")
        self.mapping_path = os.path.join(index_dir, "patent_chunks_index", "chunk_mapping.json")
        
        # Enhanced mapping structures
        self.patent_chunk_map: Dict[str, Dict[str, Any]] = {}
        self.company_to_patents: Dict[str, List[str]] = defaultdict(list)
        self.patent_to_chunks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.hojin_to_patents: Dict[str, List[str]] = defaultdict(list)
        
        logger.info("PatentDataMapper initialized")
    
    def load_mappings(self) -> bool:
        """Load and build enhanced mappings from existing data."""
        try:
            # Load existing chunks and metadata
            if not os.path.exists(self.chunks_path) or not os.path.exists(self.mapping_path):
                logger.warning("Patent chunks or mapping files not found")
                return False
            
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                metadatas = json.load(f)
            
            if len(chunks) != len(metadatas):
                logger.error("Mismatch between chunks and metadata count")
                return False
            
            # Build enhanced mappings
            for i, (chunk, metadata) in enumerate(zip(chunks, metadatas)):
                company_id = metadata.get("company_id", "")
                company_name = metadata.get("company_name", "")
                patent_id = metadata.get("patent_id", "")
                chunk_index = metadata.get("chunk_index", 0)
                
                # Create unique key: company_name + hojin_id + patent_id + chunk_index
                unique_key = f"{company_name}_{company_id}_{patent_id}_{chunk_index}"
                
                self.patent_chunk_map[unique_key] = {
                    "chunk_content": chunk,
                    "company_name": company_name,
                    "hojin_id": company_id,
                    "patent_id": patent_id,
                    "chunk_index": chunk_index,
                    "global_index": i
                }
                
                # Build additional lookup maps
                self.company_to_patents[company_name].append(patent_id)
                self.hojin_to_patents[company_id].append(patent_id)
                self.patent_to_chunks[patent_id].append({
                    "chunk_content": chunk,
                    "chunk_index": chunk_index,
                    "unique_key": unique_key,
                    "company_name": company_name,
                    "hojin_id": company_id
                })
            
            # Remove duplicates in company-to-patents mappings
            for company_name in self.company_to_patents:
                self.company_to_patents[company_name] = list(set(self.company_to_patents[company_name]))
            
            for hojin_id in self.hojin_to_patents:
                self.hojin_to_patents[hojin_id] = list(set(self.hojin_to_patents[hojin_id]))
            
            logger.info(f"Loaded {len(self.patent_chunk_map)} patent chunk mappings")
            return True
            
        except Exception as e:
            logger.error(f"Error loading patent mappings: {e}")
            return False
    
    def get_chunk_by_key(self, company_name: str, hojin_id: str, patent_id: str, chunk_index: int) -> Optional[Dict[str, Any]]:
        """
        Get chunk by company name + hojin_id + patent_id + chunk_index.
        
        Args:
            company_name: Company name
            hojin_id: Company hojin ID
            patent_id: Patent ID
            chunk_index: Chunk index
            
        Returns:
            Chunk information or None if not found
        """
        unique_key = f"{company_name}_{hojin_id}_{patent_id}_{chunk_index}"
        return self.patent_chunk_map.get(unique_key)
    
    def get_all_chunks_for_patent(self, patent_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific patent."""
        return self.patent_to_chunks.get(patent_id, [])
    
    def get_patents_by_company_name(self, company_name: str) -> List[str]:
        """Get all patent IDs for a company by name."""
        return self.company_to_patents.get(company_name, [])
    
    def get_patents_by_hojin_id(self, hojin_id: str) -> List[str]:
        """Get all patent IDs for a company by hojin_id."""
        return self.hojin_to_patents.get(hojin_id, [])
    
    def get_company_patent_chunks(self, company_identifier: str) -> List[Dict[str, Any]]:
        """
        Get all patent chunks for a company (by name or hojin_id).
        
        Args:
            company_identifier: Company name or hojin_id
            
        Returns:
            List of all patent chunks for the company
        """
        chunks = []
        
        # Try by company name first
        patents = self.get_patents_by_company_name(company_identifier)
        if not patents:
            # Try by hojin_id
            patents = self.get_patents_by_hojin_id(company_identifier)
        
        for patent_id in patents:
            chunks.extend(self.get_all_chunks_for_patent(patent_id))
        
        return chunks
    
    def get_mapping_stats(self) -> Dict[str, Any]:
        """Get statistics about the mappings."""
        return {
            "total_chunks": len(self.patent_chunk_map),
            "unique_patents": len(self.patent_to_chunks),
            "unique_companies": len(self.company_to_patents),
            "avg_chunks_per_patent": sum(len(chunks) for chunks in self.patent_to_chunks.values()) / max(len(self.patent_to_chunks), 1),
            "avg_patents_per_company": sum(len(patents) for patents in self.company_to_patents.values()) / max(len(self.company_to_patents), 1)
        }


class DataMappingManager:
    """Manager class for both company and patent data mappings."""
    
    def __init__(self, index_dir: str = "RAG_INDEX"):
        """
        Initialize the data mapping manager.
        
        Args:
            index_dir: Index directory path
        """
        self.index_dir = index_dir
        self.company_mapper = CompanyDataMapper(index_dir=index_dir)
        self.patent_mapper = PatentDataMapper(index_dir=index_dir)
        
        logger.info("DataMappingManager initialized")
    
    def initialize_all_mappings(self) -> bool:
        """Initialize all mappings."""
        try:
            company_success = self.company_mapper.load_mappings()
            patent_success = self.patent_mapper.load_mappings()
            
            if company_success and patent_success:
                logger.info("All data mappings initialized successfully")
                return True
            else:
                logger.warning("Some mappings failed to initialize")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing mappings: {e}")
            return False
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all mappings."""
        return {
            "company_stats": self.company_mapper.get_mapping_stats(),
            "patent_stats": self.patent_mapper.get_mapping_stats(),
            "initialization_time": "Available after load_mappings()"
        }
    
    def search_cross_reference(self, company_identifier: str) -> Dict[str, Any]:
        """
        Search for both company data and patents for a given company.
        
        Args:
            company_identifier: Company name or hojin_id
            
        Returns:
            Dict containing both company chunks and patent chunks
        """
        # Get hojin_id if we have company name
        hojin_id = company_identifier
        if company_identifier in self.company_mapper.name_to_hojin_map:
            hojin_id = self.company_mapper.name_to_hojin_map[company_identifier]
        
        return {
            "company_chunks": self.company_mapper.get_all_chunks_for_company(hojin_id),
            "patent_chunks": self.patent_mapper.get_company_patent_chunks(company_identifier),
            "hojin_id": hojin_id,
            "company_name": company_identifier
        }


# Utility functions for easy access
def create_mapping_manager(index_dir: str = "RAG_INDEX") -> DataMappingManager:
    """Create and initialize a data mapping manager."""
    manager = DataMappingManager(index_dir)
    manager.initialize_all_mappings()
    return manager

def get_company_chunk(company_name: str, hojin_id: str, chunk_index: int, index_dir: str = "RAG_INDEX") -> Optional[Dict[str, Any]]:
    """Quick utility to get a specific company chunk."""
    mapper = CompanyDataMapper(index_dir=index_dir)
    if mapper.load_mappings():
        return mapper.get_chunk_by_key(company_name, hojin_id, chunk_index)
    return None

def get_patent_chunk(company_name: str, hojin_id: str, patent_id: str, chunk_index: int, index_dir: str = "RAG_INDEX") -> Optional[Dict[str, Any]]:
    """Quick utility to get a specific patent chunk."""
    mapper = PatentDataMapper(index_dir=index_dir)
    if mapper.load_mappings():
        return mapper.get_chunk_by_key(company_name, hojin_id, patent_id, chunk_index)
    return None 