#!/usr/bin/env python3

"""Topic PDF → Chroma vector curator.

This utility scans a directory of PDFs, generates embeddings with Ollama,
and persists a Chroma vector store for the provided topic.

Example:
    python topic-curator.py \
        --pdf-dir ./topics/spring-boot \
        --topic-name spring-boot \
        --embedding-model mxbai-embed-large

Ensure Ollama is running locally and the embedding model is available
(e.g., `ollama pull mxbai-embed-large`).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from glob import glob
from typing import List

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def find_pdfs(pdf_dir: str) -> List[str]:
    patterns = [
        os.path.join(pdf_dir, "**", "*.pdf"),
        os.path.join(pdf_dir, "*.pdf"),
    ]
    pdfs: List[str] = []
    for pattern in patterns:
        pdfs.extend(glob(pattern, recursive=True))
    seen = set()
    ordered: List[str] = []
    for path in pdfs:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def build_topic_vectors(
    pdf_dir: str,
    topic_name: str | None = None,
    persist_root: str = ".chroma",
    embedding_model: str = "mxbai-embed-large",
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    reset: bool = False,
) -> str:
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    topic = topic_name or os.path.basename(os.path.abspath(pdf_dir))
    topic = topic.strip()
    if not topic:
        raise ValueError("Topic name could not be determined; provide --topic-name.")

    topic_dir = os.path.join(persist_root, topic)
    os.makedirs(persist_root, exist_ok=True)

    if reset and os.path.exists(topic_dir):
        shutil.rmtree(topic_dir)

    os.makedirs(topic_dir, exist_ok=True)

    pdf_paths = find_pdfs(pdf_dir)
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found under: {pdf_dir}")

    print(f"[curator] Topic: {topic}")
    print(f"[curator] Source directory: {os.path.abspath(pdf_dir)}")
    print(f"[curator] PDFs located: {len(pdf_paths)}")
    for path in pdf_paths:
        print(f"  - {path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    documents = []
    ids = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        splits = splitter.split_documents(pages)
        for index, doc in enumerate(splits):
            doc.metadata = {
                **doc.metadata,
                "topic": topic,
                "source": os.path.relpath(pdf_path, start=pdf_dir),
            }
            page_num = doc.metadata.get("page", 0)
            start_index = doc.metadata.get("start_index", 0)
            ids.append(f"{doc.metadata['source']}#p{page_num}-o{start_index}-{index}")
        documents.extend(splits)

    print(f"[curator] Total text chunks: {len(documents)}")

    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = Chroma(
        collection_name=topic,
        persist_directory=topic_dir,
        embedding_function=embeddings,
    )

    vector_store.add_documents(documents=documents, ids=ids)
    print(f"[curator] Persisted to: {topic_dir}")

    metadata_path = os.path.join(topic_dir, "db_meta.json")
    metadata = {
        "topic": topic,
        "collection": topic,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "source_dir": os.path.abspath(pdf_dir),
        "source_files": [os.path.relpath(path, start=pdf_dir) for path in pdf_paths],
        "built_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        print(f"[curator] Metadata saved: {metadata_path}")
    except OSError as error:
        print(f"[curator] Warning: metadata write failed ({error})")

    return topic_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a directory of PDFs into a topic-specific Chroma vector database.",
    )
    parser.add_argument(
        "--pdf-dir",
        required=True,
        help="Directory containing source PDFs",
    )
    parser.add_argument(
        "--topic-name",
        default=None,
        help="Name of the topic; defaults to the PDF directory name",
    )
    parser.add_argument(
        "--persist-root",
        default=".chroma",
        help="Root directory for persisted Chroma databases",
    )
    parser.add_argument(
        "--embedding-model",
        default="mxbai-embed-large",
        help="Ollama embedding model to use",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Character count per chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between consecutive chunks",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Recreate the topic directory before indexing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_topic_vectors(
        pdf_dir=args.pdf_dir,
        topic_name=args.topic_name,
        persist_root=args.persist_root,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
