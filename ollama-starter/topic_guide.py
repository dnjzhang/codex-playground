#!/usr/bin/env python3
"""
Chat/query client for a per-topic Chroma DB.

Examples:
  # One-off query
  python ollama-starter/topic_guide.py --topic-name spring-boot --query "What is Spring Boot auto-configuration?"

  # Interactive chatbot (q/exit to quit)
  python ollama-starter/topic_guide.py --topic-name spring-boot

Notes:
  - Auto-detects the embedding model from the topic DB metadata when available
  - Requires Ollama running locally with the chosen chat model available
  - DB should be built with build_topic_vector_db.py
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, TypedDict
import json

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langgraph.graph import StateGraph, END


class QAState(TypedDict):
    question: str
    context: List[Document]
    answer: str


def _load_db_metadata(persist_dir: str) -> Dict | None:
    meta_path = os.path.join(persist_dir, "db_meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_retriever(
    topic_name: str,
    persist_root: str,
    embedding_model: str | None,
    collection_name: str | None = None,
    k: int = 5,
    search_type: str = "similarity",
    fetch_k: int | None = None,
    lambda_mult: float | None = None,
):
    persist_dir = os.path.join(persist_root, topic_name)
    collection = collection_name or f"topic_{topic_name}"

    # Auto-detect embedding model from metadata when embedding_model is None or 'auto'
    if not embedding_model or embedding_model == "auto":
        meta = _load_db_metadata(persist_dir)
        detected = (meta or {}).get("embedding_model")
        if detected:
            embedding_model = detected
        else:
            # Fallback to a sensible default
            embedding_model = "mxbai-embed-large"

    embeddings = OllamaEmbeddings(model=embedding_model)
    vector_store = Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    kwargs = {"k": k}
    if fetch_k is not None:
        kwargs["fetch_k"] = fetch_k
    if lambda_mult is not None:
        kwargs["lambda_mult"] = lambda_mult
    retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=kwargs)
    info = {
        "persist_dir": persist_dir,
        "collection": collection,
        "embedding_model": embedding_model,
        "k": k,
        "search_type": search_type,
        "fetch_k": fetch_k,
        "lambda_mult": lambda_mult,
    }
    return retriever, info, vector_store


def build_qa_graph(chat_model: str, retriever, system_preamble: str | None = None):
    system_text = system_preamble or (
        "You are a helpful assistant. Use the provided context to answer "
        "the user's question concisely. If the answer is unknown, say so."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}\n\nProvide a clear, source-grounded answer.",
            ),
        ]
    )

    llm = ChatOllama(model=chat_model)
    chain = prompt | llm | StrOutputParser()

    def retrieve_node(state: QAState) -> Dict:
        docs = retriever.invoke(state["question"])
        return {"context": docs}

    def generate_node(state: QAState) -> Dict:
        # Join docs into text blocks for the prompt
        context_text = "\n\n".join(
            f"[source: {d.metadata.get('source','?')} p{d.metadata.get('page','?')}]\n{d.page_content}"
            for d in state.get("context", [])
        )
        answer = chain.invoke({"question": state["question"], "context": context_text})
        return {"answer": answer}

    graph = StateGraph(QAState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge("retrieve", "generate")
    graph.set_entry_point("retrieve")
    graph.add_edge("generate", END)
    return graph.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query/chat with a per-topic Chroma DB using a simple LangGraph QA pipeline.")
    parser.add_argument("--topic-name", required=True, help="Topic name used when building the DB")
    parser.add_argument("--persist-root", default=os.path.join("ollama-starter", "chroma_topics"), help="Root directory where topic DBs are stored")
    parser.add_argument("--collection-name", default=None, help="Optional Chroma collection name; defaults to topic_<topic>")
    parser.add_argument("--embedding-model", default="auto", help="Embedding model to use; 'auto' tries to read from DB metadata")
    parser.add_argument("--chat-model", default="llama3.2", help="Ollama chat model to generate answers")
    parser.add_argument("--k", type=int, default=5, help="Top-k documents to retrieve")
    parser.add_argument("--search-type", choices=["similarity", "mmr"], default="similarity", help="Retrieval search type")
    parser.add_argument("--fetch-k", type=int, default=None, help="Candidate pool size (used by MMR)")
    parser.add_argument("--lambda-mult", type=float, default=None, help="Diversity factor for MMR (0-1)")
    parser.add_argument("--query", default=None, help="Run a single query then exit; if omitted, starts interactive mode")
    parser.add_argument("--show-sources", action="store_true", help="Print brief source info after each answer")
    parser.add_argument("--debug", action="store_true", help="Print debug info about DB + retrieval")
    parser.add_argument("--inspect", default=None, help="Inspect retrieval only: print top docs and scores for a query, then exit")

    args = parser.parse_args()

    retriever, rinfo, vector_store = build_retriever(
        topic_name=args.topic_name,
        persist_root=args.persist_root,
        embedding_model=args.embedding_model,
        collection_name=args.collection_name,
        k=args.k,
        search_type=args.search_type,
        fetch_k=args.fetch_k,
        lambda_mult=args.lambda_mult,
    )

    app = build_qa_graph(chat_model=args.chat_model, retriever=retriever)

    if args.debug:
        print("[debug] topic:", args.topic_name)
        print("[debug] persist_dir:", rinfo["persist_dir"])
        print("[debug] collection:", rinfo["collection"])
        print("[debug] embedding_model:", rinfo["embedding_model"])
        print("[debug] search_type:", rinfo["search_type"])
        print("[debug] k:", rinfo["k"], "fetch_k:", rinfo["fetch_k"], "lambda_mult:", rinfo["lambda_mult"])

    if args.inspect:
        q = args.inspect
        docs_scores = vector_store.similarity_search_with_score(q, k=args.k)
        print("Top matches:")
        for d, score in docs_scores:
            src = d.metadata.get("source", "?")
            page = d.metadata.get("page", "?")
            print(f"- score={score:.4f}  {src} (p{page})")
        return

    def run_once(q: str) -> None:
        result = app.invoke({"question": q, "context": [], "answer": ""})
        print("\n=== Answer ===\n")
        print(result.get("answer", ""))
        if args.show_sources:
            print("\n--- Sources ---")
            try:
                docs_scores = vector_store.similarity_search_with_score(q, k=args.k)
                if not docs_scores:
                    print("(no sources found)")
                else:
                    for d, score in docs_scores:
                        src = d.metadata.get("source", "?")
                        page = d.metadata.get("page", "?")
                        print(f"- {src} (p{page})  score={score:.4f}")
            except Exception:
                # Fallback to retriever in case of API differences
                docs = retriever.invoke(q)
                if not docs:
                    print("(no sources found)")
                else:
                    for d in docs:
                        src = d.metadata.get("source", "?")
                        page = d.metadata.get("page", "?")
                        print(f"- {src} (p{page})")

    if args.query:
        run_once(args.query)
        return

    print("Interactive chatbot mode. Type 'q' or 'exit' to quit.\n")
    while True:
        try:
            q = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        run_once(q)


if __name__ == "__main__":
    main()
