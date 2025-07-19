"""
PatentVectorstore: Handles vector storage and retrieval for patent data using HuggingFace embeddings.
"""
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class PatentVectorstore:
    """
    Wrapper for patent vectorstore using Chroma and HuggingFaceEmbeddings (all-MiniLM-L6-v2).
    """
    def __init__(self, persist_directory: str = "patent_chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def add_documents(self, docs):
        """Add documents to the vectorstore."""
        self.vectorstore.add_documents(docs)

    def search(self, query: str, k: int = 5):
        """Search for top-k relevant patent documents."""
        return self.vectorstore.similarity_search(query, k=k) 