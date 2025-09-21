"""
Topic PDF → Chroma vector builder using Ollama embeddings.

Usage examples:
  python ollama-starter/build_topic_vector_db.py --topic-dir ollama-starter/topics/spring-boot
  python ollama-starter/build_topic_vector_db.py --topic-dir ./my_topic --topic-name my-topic --embedding-model mxbai-embed-large

This script scans a topic folder for PDFs, loads + splits them using
LangChain loaders and text splitter, and persists a Chroma DB per topic.

Requirements (install under ollama-starter):
  pip install -r ollama-starter/requireCan ments.txt

Ollama notes:
  - Ensure Ollama is running locally and the embedding model is available
    (e.g., `ollama pull mxbai-embed-large` to download once).
"""

from __future__ import annotations

import argparse
import os
import shutil
from glob import glob
from typing import List
import json
from datetime import datetime

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def find_pdfs(topic_dir: str) -> List[str]:
    patterns = [
        os.path.join(topic_dir, "**", "*.pdf"),
        os.path.join(topic_dir, "*.pdf"),
    ]
    pdfs: List[str] = []
    for pattern in patterns:
        pdfs.extend(glob(pattern, recursive=True))
    # Deduplicate preserving order
    seen = set()
    unique: List[str] = []
    for p in pdfs:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def build_topic_db(
    topic_dir: str,
    topic_name: str | None = None,
    persist_root: str = os.path.join("ollama-starter", "chroma_topics"),
    collection_name: str | None = None,
    embedding_model: str = "mxbai-embed-large",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    reset: bool = False,
) -> str:
    if not os.path.isdir(topic_dir):
        raise FileNotFoundError(f"Topic directory not found: {topic_dir}")

    topic = topic_name or os.path.basename(os.path.abspath(topic_dir))
    collection = collection_name or f"topic_{topic}"
    persist_dir = os.path.join(persist_root, topic)

    os.makedirs(persist_dir, exist_ok=True)

    if reset and os.path.exists(persist_dir):
        # Remove the directory to start fresh
        shutil.rmtree(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)

    pdf_paths = find_pdfs(topic_dir)
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found under: {topic_dir}")

    print(f"[builder] Topic: {topic}")
    print(f"[builder] PDFs found: {len(pdf_paths)}")
    for p in pdf_paths:
        print(f"  - {p}")

    # Load and split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    all_splits = []
    ids = []
    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        pages = loader.load()  # list[Document], page-wise
        splits = text_splitter.split_documents(pages)
        # attach topic/source metadata and create stable IDs
        for i, d in enumerate(splits):
            d.metadata = {
                **d.metadata,
                "topic": topic,
                "source": os.path.relpath(pdf, start=topic_dir),
            }
            # Construct id: <source>#<page>-<offset>-<i>
            page_num = d.metadata.get("page", 0)
            start = d.metadata.get("start_index", 0)
            ids.append(f"{d.metadata['source']}#p{page_num}-o{start}-{i}")
        all_splits.extend(splits)

    print(f"[builder] Total chunks: {len(all_splits)}")

    # Init embeddings + vector store
    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    # If the collection is fresh, add documents; otherwise, upsert via add with ids
    vector_store.add_documents(documents=all_splits, ids=ids)
    print(f"[builder] Persisted Chroma DB → {persist_dir}")
    print(f"[builder] Collection           → {collection}")

    # Persist metadata for auto-detection in the query client
    meta_path = os.path.join(persist_dir, "db_meta.json")
    metadata = {
        "topic": topic,
        "collection": collection,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "source_files": [os.path.relpath(p, start=topic_dir) for p in pdf_paths],
        "built_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"[builder] Wrote metadata       → {meta_path}")
    except Exception as e:
        print(f"[builder] Warning: could not write metadata file: {e}")

    return persist_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-topic Chroma DB from PDFs using Ollama embeddings.")
    parser.add_argument("--topic-dir", required=True, help="Path to topic directory containing PDFs")
    parser.add_argument("--topic-name", default=None, help="Optional explicit topic name; defaults to folder name")
    parser.add_argument("--persist-root", default=os.path.join("ollama-starter", "chroma_topics"), help="Root directory where topic DBs are stored")
    parser.add_argument("--collection-name", default=None, help="Optional Chroma collection name; defaults to topic_<topic>")
    parser.add_argument("--embedding-model", default="mxbai-embed-large", help="Ollama embedding model (e.g., mxbai-embed-large, nomic-embed-text)")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Text splitter chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Text splitter chunk overlap")
    parser.add_argument("--reset", action="store_true", help="If set, recreate the topic DB directory")

    args = parser.parse_args()

    build_topic_db(
        topic_dir=args.topic_dir,
        topic_name=args.topic_name,
        persist_root=args.persist_root,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset=bool(args.reset),
    )


if __name__ == "__main__":
    main()

