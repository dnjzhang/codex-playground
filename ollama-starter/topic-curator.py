#!/usr/bin/env python3

"""Topic document → Chroma vector curator.

This utility scans a directory of PDFs (default) or plain text files, generates embeddings with Ollama or
OCI Generative AI, and persists a Chroma vector store for the provided topic.

Example:
    python topic-curator.py \
        --source-dir ./topics/spring-boot \
        --topic-name spring-boot \
        --embedding-provider ollama \
        --embedding-model mxbai-embed-large

    python topic-curator.py \
        --source-dir ./topics/sprint-planning \
        --topic-name sprint-planning \
        --embedding-provider oci \
        --oci-config ./oci-config.json

    python topic-curator.py \
        --source-dir ./topics/plain-notes \
        --topic-name planning-notes \
        --input-format text

Ensure Ollama is running locally and the embedding model is available
(e.g., `ollama pull mxbai-embed-large`). For OCI embeddings, configure
credentials in an `oci-config.json` (or provide `--oci-config`).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from glob import glob
from typing import Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings

try:
    from langchain_community.embeddings.oci_generative_ai import (
        OCIGenAIEmbeddings,
    )
except ImportError:  # noqa: WPS440 - optional dependency
    OCIGenAIEmbeddings = None  # type: ignore[assignment]


def find_source_files(source_dir: str, extension: str) -> List[str]:
    patterns = [
        os.path.join(source_dir, "**", f"*.{extension}"),
        os.path.join(source_dir, f"*.{extension}"),
    ]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob(pattern, recursive=True))
    seen = set()
    ordered: List[str] = []
    for path in files:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def load_oci_settings(
    *,
    oci_config: str | None,
    endpoint_override: str | None,
    compartment_override: str | None,
    auth_profile_override: str | None,
    model_override: str | None,
) -> Dict[str, str]:
    """Resolve OCI embedding settings from config + overrides."""

    config_path = oci_config or "oci-config.json"
    config_data: Dict[str, str] = {}

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                loaded = json.load(config_file)
            if isinstance(loaded, dict):
                config_data = {str(k): str(v) for k, v in loaded.items() if v is not None}
        except json.JSONDecodeError as exc:  # noqa: BLE001
            raise ValueError(f"Invalid OCI config JSON at {config_path}: {exc}") from exc
    elif oci_config:
        raise FileNotFoundError(f"OCI config file not found: {config_path}")

    resolved = {
        "service_endpoint": endpoint_override or config_data.get("endpoint"),
        "compartment_id": compartment_override or config_data.get("compartment_ocid"),
        "auth_profile": auth_profile_override or config_data.get("config_profile"),
        "model_id": model_override or config_data.get("embedding_model_name") or config_data.get("model_name"),
    }

    missing = [key for key, value in resolved.items() if not value]
    if missing:
        details = ", ".join(missing)
        hint = (
            "Provide the missing values via CLI options or ensure they are present "
            f"in {config_path}."
        )
        raise ValueError(f"Missing OCI configuration values: {details}. {hint}")

    return resolved


def init_embeddings(
    provider: str,
    embedding_model: str | None,
    *,
    oci_config: str | None,
    oci_endpoint: str | None,
    oci_compartment_id: str | None,
    oci_auth_profile: str | None,
) -> Tuple[object, str]:
    provider_key = provider.lower()

    if provider_key == "ollama":
        model_id = embedding_model or "mxbai-embed-large"
        embeddings = OllamaEmbeddings(model=model_id)
        return embeddings, model_id

    if provider_key == "oci":
        if OCIGenAIEmbeddings is None:
            raise ImportError(
                "langchain-community with OCI support is required. Install via "
                "`pip install langchain-community` or the project's requirements."
            )

        settings = load_oci_settings(
            oci_config=oci_config,
            endpoint_override=oci_endpoint,
            compartment_override=oci_compartment_id,
            auth_profile_override=oci_auth_profile,
            model_override=embedding_model,
        )
        embeddings = OCIGenAIEmbeddings(
            model_id=settings["model_id"],
            service_endpoint=settings["service_endpoint"],
            compartment_id=settings["compartment_id"],
            auth_profile=settings["auth_profile"],
            model_kwargs={"truncate": True},
        )
        return embeddings, settings["model_id"]

    raise ValueError(
        "Unsupported embedding provider. Choose from: ollama, oci."
    )


def build_topic_vectors(
    source_dir: str,
    topic_name: str | None = None,
    persist_root: str = ".chroma",
    embedding_provider: str = "ollama",
    embedding_model: str | None = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    reset: bool = False,
    oci_config: str | None = None,
    oci_endpoint: str | None = None,
    oci_compartment_id: str | None = None,
    oci_auth_profile: str | None = None,
    input_format: str = "pdf",
) -> str:
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if input_format not in {"pdf", "text"}:
        raise ValueError("input_format must be either 'pdf' or 'text'")

    topic = topic_name or os.path.basename(os.path.abspath(source_dir))
    topic = topic.strip()
    if not topic:
        raise ValueError("Topic name could not be determined; provide --topic-name.")

    topic_dir = os.path.join(persist_root, topic)
    os.makedirs(persist_root, exist_ok=True)

    if reset and os.path.exists(topic_dir):
        shutil.rmtree(topic_dir)

    os.makedirs(topic_dir, exist_ok=True)

    extension = "pdf" if input_format == "pdf" else "txt"
    source_paths = find_source_files(source_dir, extension)
    if not source_paths:
        raise FileNotFoundError(
            f"No {extension.upper()} files found under: {source_dir}"
        )

    print(f"[curator] Topic: {topic}")
    print(f"[curator] Source directory: {os.path.abspath(source_dir)}")
    print(f"[curator] {extension.upper()} files located: {len(source_paths)}")
    for path in source_paths:
        print(f"  - {path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    documents = []
    ids = []
    for doc_path in source_paths:
        if input_format == "pdf":
            loader = PyPDFLoader(doc_path)
        else:
            loader = TextLoader(doc_path, encoding="utf-8")
        pages = loader.load()
        splits = splitter.split_documents(pages)
        for index, doc in enumerate(splits):
            rel_source = os.path.relpath(doc_path, start=source_dir)
            doc.metadata = {
                **doc.metadata,
                "topic": topic,
                "source": rel_source,
            }
            page_num = doc.metadata.get("page")
            if page_num is None:
                page_num = 1
                doc.metadata["page"] = page_num
            start_index = doc.metadata.get("start_index", index * chunk_size)
            ids.append(f"{rel_source}#p{page_num}-o{start_index}-{index}")
        documents.extend(splits)

    print(f"[curator] Total text chunks: {len(documents)}")

    embeddings, resolved_model = init_embeddings(
        embedding_provider,
        embedding_model,
        oci_config=oci_config,
        oci_endpoint=oci_endpoint,
        oci_compartment_id=oci_compartment_id,
        oci_auth_profile=oci_auth_profile,
    )
    print(
        f"[curator] Embedding provider: {embedding_provider} (model: {resolved_model})"
    )
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
        "embedding_provider": embedding_provider,
        "embedding_model": resolved_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "input_format": input_format,
        "source_dir": os.path.abspath(source_dir),
        "source_files": [os.path.relpath(path, start=source_dir) for path in source_paths],
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
        description="Convert a directory of documents into a topic-specific Chroma vector database.",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing source documents",
    )
    parser.add_argument(
        "--topic-name",
        default=None,
        help="Name of the topic; defaults to the source directory name",
    )
    parser.add_argument(
        "--persist-root",
        default=".chroma",
        help="Root directory for persisted Chroma databases",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["ollama", "oci"],
        default="ollama",
        help="Embedding provider to use",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model identifier; defaults per provider",
    )
    parser.add_argument(
        "--input-format",
        choices=["pdf", "text"],
        default="pdf",
        help="Document format to ingest (default: pdf)",
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
    parser.add_argument(
        "--oci-config",
        default=None,
        help="Path to OCI config JSON (defaults to ./oci-config.json)",
    )
    parser.add_argument(
        "--oci-endpoint",
        default=None,
        help="Override OCI service endpoint",
    )
    parser.add_argument(
        "--oci-compartment-id",
        default=None,
        help="Override OCI compartment OCID",
    )
    parser.add_argument(
        "--oci-auth-profile",
        default=None,
        help="Override OCI config profile name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_topic_vectors(
        source_dir=args.source_dir,
        topic_name=args.topic_name,
        persist_root=args.persist_root,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset=args.reset,
        oci_config=args.oci_config,
        oci_endpoint=args.oci_endpoint,
        oci_compartment_id=args.oci_compartment_id,
        oci_auth_profile=args.oci_auth_profile,
        input_format=args.input_format,
    )


if __name__ == "__main__":
    main()
