# EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

patent_config = {
    "index_dir": "data",
    "embed_model": "minilm",          # or 'mpnet', 'bge'
    "top_k": 3,
    "chunk_size": 3000,
    "chunk_overlap": 300,
    "output_subdir": "patent_chunks_index",
    "chroma_subdir": r"patent_data/chroma_db",
    "collection_name": f"patent_text_index",
    "force_reindex": False
}

firm_config = {
    "data_dir": "data",
    "embed_model": "minilm",          # or 'mpnet', 'bge'
    "top_k": 3,
    "chunk_size": 3000,
    "chunk_overlap": 300,
    "output_subdir": r"firm_summary_index",
    "chroma_subdir": r"firm_data/chroma_db",
    "collection_name": f"firm_summary_index",
    "force_reindex": False
}
