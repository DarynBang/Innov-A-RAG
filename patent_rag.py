import os
import json
from typing import List, Dict, Any
from tqdm.auto import tqdm
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.rag_config import patent_config
import pandas as pd
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class PatentRAG:
    def __init__(
        self,
        df,
        index_dir: str,
        config: dict
    ):
        # Data + paths
        self.df = df
        # self.df = df.sample(frac=0.05, random_state=42)
        self.index_dir = index_dir
        self.embed_model = config.get("embed_model")
        self.device = config.get("device", torch.cuda.is_available() and "cuda" or "cpu")
        self.chunk_size = config.get("chunk_size", 2048)
        self.chunk_overlap = config.get("chunk_overlap", 256)
        self.batch_size = config.get("batch_size", 5000)
        self.top_k = config.get("top_k", 5)
        self.output_subdir = config.get("output_subdir", "patent_chunks_index")
        self.chroma_subdir = config.get("chroma_subdir", "patent_data/chroma_db")
        self.collection_name = config.get("collection_name", "patent_text_index")
        self.force_reindex = config.get("force_reindex", False)

        # build derived paths
        self.output_dir = os.path.join(self.index_dir, self.output_subdir)
        self.chroma_path = os.path.join(self.index_dir, self.chroma_subdir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.chroma_path, exist_ok=True)

        # JSON file paths
        self.chunks_path  = os.path.join(self.output_dir, "chunks.json")
        self.mapping_path = os.path.join(self.output_dir, "chunk_mapping.json")

        # Embedder + splitter
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embed_model,
            device=self.device
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        # Chroma client placeholder
        os.makedirs(self.chroma_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.batch_size = self.batch_size

        # In‐memory storage
        self.all_chunks: List[str] = []
        self.all_metadatas: List[Dict[str, Any]] = []

    def build_chunks(self, force_reindex: bool = False):
        """Load or (re)build chunks + metadata and write JSON."""
        os.makedirs(self.output_dir, exist_ok=True)

        if not force_reindex and os.path.exists(self.chunks_path) and os.path.exists(self.mapping_path):
            with open(self.chunks_path,  "r", encoding="utf-8") as f: self.all_chunks    = json.load(f)
            with open(self.mapping_path, "r", encoding="utf-8") as f: self.all_metadatas = json.load(f)
            logger.info(f"Loaded {len(self.all_chunks)} chunks & {len(self.all_metadatas)} metadata entries.")
            return

        self.all_chunks = []
        self.all_metadatas = []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Chunking Patents"):
            cid = row["hojin_id"]
            name = row["company_name"]
            patent_id   = row["appln_id"]
            text = row["cleaned_patent"] if "cleaned_patent" in row else row["patent_abstract"]

            chunks = self.splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                self.all_chunks.append(chunk)
                self.all_metadatas.append({
                    "company_id":       cid,
                    "company_name":     name,
                    "patent_id": patent_id,
                    "chunk_index":      i
                })

        # write JSON
        with open(self.chunks_path,  "w", encoding="utf-8") as f: json.dump(self.all_chunks,    f, indent=2, ensure_ascii=False)
        with open(self.mapping_path, "w", encoding="utf-8") as f: json.dump(self.all_metadatas, f, indent=2, ensure_ascii=False)
        logger.info(f"Built & stored {len(self.all_chunks)} chunks & {len(self.all_metadatas)} metadata entries for Patent data.")

    def ingest_all(self, force_reindex: bool = False):
        """(Re)create Chroma collection and ingest all chunks + metadata."""
        # delete existing if requested
        if force_reindex:
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info(f"Deleted existing collection '{self.collection_name}'.")
            except Exception:
                logger.warning(f"Failed to delete existing collection '{self.collection_name}'")  # didn’t exist

        # If not forcing, see if it already exists
        if not force_reindex:
            try:
                existing = self.client.get_collection(name=self.collection_name)
                logger.info(f"→ Collection '{self.collection_name}' already exists; skipping ingest.")
                return existing
            except Exception:
                # doesn’t exist yet → fall through to create
                pass

        # build or load chunks first
        self.build_chunks(force_reindex=force_reindex)

        # create collection
        collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"}
        )

        # batch ingest
        total = len(self.all_chunks)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            collection.add(
                documents=self.all_chunks[start:end],
                ids=[f"chunk_{i}" for i in range(start, end)],
                metadatas=self.all_metadatas[start:end]
            )
            logger.info(f" • ingested batches {start}-{end}/{total}")

        logger.info(f"Created collection '{self.collection_name}' with {total} chunks for Patent data.")
        return collection

    def add_one(
        self,
        patent_id: str,
        company_id: str,
        company_name: str,
        full_text: str,
    ):
        """
        Incrementally add one Patent:
        check existing, chunk, update JSON + Chroma.
        """
        # ensure chunks are loaded
        if not self.all_chunks:
            self.build_chunks(force_reindex=False)

        # open or create collection
        try:
            collection = self.client.get_collection(name=self.collection_name)
        except ValueError:
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embed_fn,
                metadata={"hnsw:space": "cosine"}
            )

        # check duplicate
        hits = collection.get(where={"patent_id": patent_id}, include=["metadatas"])
        if hits["metadatas"]:
            return {"error": f"Patent {patent_id} of Company {company_id} already indexed."}

        # chunk new patent
        chunks = self.splitter.split_text(full_text)
        offset = len(self.all_chunks)
        new_ids = [f"chunk_{offset + i}" for i in range(len(chunks))]
        metas   = [
            {"company_id": company_id,
             "company_name": company_name,
             "patent_id": patent_id,
             "chunk_index": i}
            for i in range(len(chunks))
        ]

        # update in-memory & JSON
        self.all_chunks.extend(chunks)
        self.all_metadatas.extend(metas)
        with open(self.chunks_path,  "w", encoding="utf-8") as f: json.dump(self.all_chunks,    f, indent=2, ensure_ascii=False)
        with open(self.mapping_path, "w", encoding="utf-8") as f: json.dump(self.all_metadatas, f, indent=2, ensure_ascii=False)

        # add to Chroma
        collection.add(documents=chunks, ids=new_ids, metadatas=metas)
        logger.info(f"Added {len(chunks)} chunks for Patent {patent_id} of Company {company_id}.")
        return collection

    def retrieve_patent_contexts(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Return the top-K most relevant summary chunks for `query`,
        each annotated with company_id, company_name, keywords, chunk_index, rank, and score.
        """
        # Make sure we have our JSON‑backed lists in memory
        if not self.all_chunks or not self.all_metadatas:
            self.build_chunks(force_reindex=False)

        # Get (or error if missing) our Chroma collection
        try:
            collection = self.client.get_collection(name=self.collection_name)
        except Exception:
            raise ValueError(f"Chroma collection '{self.collection_name}' for Patent not found; "
                             "run ingest_all() first.")

        # Query Chroma (returns cosine *distances*)
        result    = collection.query(query_texts=[query], n_results=top_k)
        ids       = result["ids"][0]
        distances = result["distances"][0]

        #  Build and return your contexts list
        contexts: List[Dict[str, Any]] = []
        for rank, (chunk_id, dist) in enumerate(zip(ids, distances), start=1):
            # extract original index from "chunk_{i}"
            idx  = int(chunk_id.split("_", 1)[1])
            meta = self.all_metadatas[idx]

            contexts.append({
                "company_id":       meta["company_id"],
                "chunk":            self.all_chunks[idx],
                "company_name":     meta["company_name"],
                "patent_id":          meta["patent_id"],
                "chunk_index":      meta["chunk_index"],
                "rank":             rank,
                "score":            float(1.0 - dist)
            })

        return contexts

def main():
    # CONFIGURATION
    INDEX_DIR = r"RAG_INDEX"
    # Use the proper config instead of hardcoded path
    patent_csv = patent_config.get("patent_csv")

    patent_df = pd.read_csv(patent_csv)
    patent_df_test = patent_df.head(10000)

    patent_rag = PatentRAG(
        df=patent_df_test,
        index_dir=INDEX_DIR,
        config=patent_config
    )

    # Full rebuild + ingest:
    patent_rag.ingest_all(force_reindex=False)

    # Full rebuild + ingest:
    patent_rag.ingest_all(force_reindex=True)

    # Add or skip a single company summary:
    res = patent_rag.add_one(
        patent_id="JP2024001234A",
        company_name="Random Company idk",
        company_id="1235678",
        full_text="Here is the full, cleaned patent text ..."
    )
    if isinstance(res, dict) and "error" in res:
        logger.error(res["error"])

    results = patent_rag.retrieve_patent_contexts("Machine Learning and Computer Vision", top_k=3)
    for hit in results:
        print(f"{hit['rank']}. [{hit['score']:.3f}] {hit['company_name']} {hit['patent_id']}→ “{hit['chunk'][:80]}…”")


if __name__ == '__main__':
    main()
