"""
BM25RetrieverWrapper: Handles sparse retrieval using BM25 algorithm.
"""
from langchain.retrievers import BM25Retriever
from typing import List

class BM25RetrieverWrapper:
    """
    Wrapper for BM25Retriever from Langchain for sparse keyword search.
    """
    def __init__(self, corpus: List[str]):
        self.retriever = BM25Retriever.from_texts(corpus)

    def search(self, query: str, k: int = 5):
        """Search for top-k relevant documents using BM25."""
        return self.retriever.get_relevant_documents(query, k=k) 