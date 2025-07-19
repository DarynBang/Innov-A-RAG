# EMBED_MODEL     = "BAAI/bge-small-en-v1.5"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

patent_config = {
    "patent_csv": r'data/random100000_us_patents.csv',                      # <- Path to patent csv file
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",                # or 'mpnet', 'bge'
    "top_k": 3,
    "output_subdir": "patent_chunks_index",
    "chroma_subdir": r"patent_data/chroma_db",
    "collection_name": f"patent_text_index",
    "force_reindex": False
}

firm_config = {
    "firm_csv": r'data/firms_summary.csv',                                  # <- Path to firm csv file
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",                # or 'mpnet', 'bge'
    "top_k": 3,
    "output_subdir": r"firm_summary_index",
    "chroma_subdir": r"firm_data/chroma_db",
    "collection_name": f"firm_summary_index",
    "force_reindex": False
}
