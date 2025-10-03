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
import json
import math
import os
from typing import Dict, List, Sequence, Tuple, TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, StateGraph


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


def build_reranker(model_name: str):
    """Instantiate a HuggingFace cross-encoder reranker."""

    try:
        from langchain_community.cross_encoders.huggingface import (
            HuggingFaceCrossEncoder,
        )
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "Install sentence-transformers to use cross-encoder reranking: "
            "`pip install sentence-transformers`"
        ) from exc

    try:
        return HuggingFaceCrossEncoder(model_name=model_name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to initialize cross-encoder reranker. "
            "Ensure required ML dependencies (e.g., torch) are installed."
        ) from exc


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    reranker,
    top_n: int,
) -> List[Document]:
    """Apply cross-encoder reranking and return the top_n documents."""

    if not documents:
        return []

    bounded_top_n = max(1, min(top_n, len(documents)))
    pairs: List[Tuple[str, str]] = [
        (query, doc.page_content) for doc in documents
    ]
    scores = reranker.score(pairs)
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda item: item[1], reverse=True)

    for doc, score in scored_docs:
        doc.metadata["rerank_score"] = float(score)

    return [doc for doc, _ in scored_docs[:bounded_top_n]]


def build_retriever(
    topic_name: str,
    persist_root: str,
    embedding_model: str | None,
    collection_name: str | None = None,
    k: int = 5,
    search_type: str = "similarity",
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
    if lambda_mult is not None:
        kwargs["lambda_mult"] = lambda_mult
    retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=kwargs)
    info = {
        "persist_dir": persist_dir,
        "collection": collection,
        "embedding_model": embedding_model,
        "k": k,
        "search_type": search_type,
        "lambda_mult": lambda_mult,
    }
    return retriever, info, vector_store


def build_qa_graph(
    chat_model: str,
    retriever,
    *,
    reranker=None,
    rerank_top_n: int | None = None,
    system_preamble: str | None = None,
):
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
        docs: List[Document] = retriever.invoke(state["question"])
        desired_top_n = rerank_top_n or len(docs)
        if reranker and docs:
            context_docs = rerank_documents(
                state["question"], docs, reranker, desired_top_n
            )
        else:
            context_docs = docs[:desired_top_n]
        return {"context": context_docs}

    def generate_node(state: QAState) -> Dict:
        # Join docs into text blocks for the prompt
        context_text = "\n\n".join(
            f"[source: {d.metadata.get('source','?')} p{d.metadata.get('page','?')}]\n{d.page_content}"
            for d in state.get("context", [])
        )
        answer = chain.invoke({"question": state["question"], "context": context_text})
        return {"answer": answer, "context": state.get("context", [])}

    graph = StateGraph(QAState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge("retrieve", "generate")
    graph.set_entry_point("retrieve")
    graph.add_edge("generate", END)
    return graph.compile()


def export_graph_diagram(app, output_path: str, *, debug: bool = False) -> str:
    """Render the compiled LangGraph as a PNG diagram."""

    compiled_graph = app.get_graph()

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    png_path = output_path if output_path.lower().endswith(".png") else f"{output_path}.png"
    if debug and png_path != output_path:
        print(f"[debug] graph diagram path updated to {png_path} (PNG required)")

    try:
        compiled_graph.draw_mermaid_png(
            output_file_path=png_path,
            draw_method=MermaidDrawMethod.API,
        )
    except Exception as exc:  # noqa: BLE001
        fallback_path = f"{os.path.splitext(png_path)[0]}.mmd"
        fallback_error = None
        try:
            mermaid_diagram = compiled_graph.draw_mermaid()
            with open(fallback_path, "w", encoding="utf-8") as diagram_file:
                diagram_file.write(mermaid_diagram)
        except Exception as fallback_exc:  # noqa: BLE001
            fallback_error = fallback_exc

        if fallback_error:
            if debug:
                print(
                    "[debug] failed to save Mermaid fallback diagram: "
                    f"{fallback_error}"
                )
            raise RuntimeError(
                "Failed to render PNG diagram and unable to save Mermaid fallback"
            ) from exc

        if debug:
            print(f"[debug] Mermaid fallback saved to {fallback_path}")
        raise RuntimeError(
            (
                "Failed to render PNG diagram: "
                f"{exc}. Mermaid fallback saved to {fallback_path}"
            )
        ) from exc

    return png_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Query/chat with a per-topic Chroma DB using a simple LangGraph QA pipeline.")
    parser.add_argument("--topic-name", required=True, help="Topic name used when building the DB")
    parser.add_argument("--persist-root", default=os.path.join("ollama-starter", "chroma_topics"), help="Root directory where topic DBs are stored")
    parser.add_argument("--collection-name", default=None, help="Optional Chroma collection name; defaults to topic_<topic>")
    parser.add_argument("--embedding-model", default="auto", help="Embedding model to use; 'auto' tries to read from DB metadata")
    parser.add_argument("--chat-model", default="llama3.2", help="Ollama chat model to generate answers")
    parser.add_argument("--k", type=int, default=10, help="Top-k documents to retrieve")
    parser.add_argument("--search-type", choices=["similarity", "mmr"], default="similarity", help="Retrieval search type")
    parser.add_argument("--lambda-mult", type=float, default=None, help="Diversity factor for MMR (0-1)")
    parser.add_argument("--query", default=None, help="Run a single query then exit; if omitted, starts interactive mode")
    parser.add_argument("--show-sources", action="store_true", help="Print brief source info after each answer")
    parser.add_argument("--debug", action="store_true", help="Print debug info about DB + retrieval")
    parser.add_argument("--inspect", default=None, help="Inspect retrieval only: print top docs and scores for a query, then exit")
    parser.add_argument(
        "--rerank-model",
        default="BAAI/bge-reranker-base",
        help="Cross-encoder model to rerank retrieved documents",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=None,
        help="Number of reranked documents to send to the chat model (defaults to half of k)",
    )
    parser.add_argument(
        "--graph-diagram",
        default=None,
        help="Optional path to save the compiled LangGraph structure as a PNG image",
    )

    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be a positive integer")
    if args.rerank_top_n is not None and args.rerank_top_n <= 0:
        parser.error("--rerank-top-n must be a positive integer")

    if args.rerank_top_n is None:
        rerank_top_n = max(1, math.ceil(args.k / 2))
    else:
        rerank_top_n = max(1, min(args.rerank_top_n, args.k))

    retriever, rinfo, vector_store = build_retriever(
        topic_name=args.topic_name,
        persist_root=args.persist_root,
        embedding_model=args.embedding_model,
        collection_name=args.collection_name,
        k=args.k,
        search_type=args.search_type,
        lambda_mult=args.lambda_mult,
    )

    reranker = build_reranker(args.rerank_model)

    rinfo["rerank_model"] = args.rerank_model
    rinfo["rerank_top_n"] = rerank_top_n

    app = build_qa_graph(
        chat_model=args.chat_model,
        retriever=retriever,
        reranker=reranker,
        rerank_top_n=rerank_top_n,
    )

    if args.graph_diagram:
        try:
            png_path = export_graph_diagram(
                app,
                args.graph_diagram,
                debug=args.debug,
            )
            if args.debug:
                print(f"[debug] graph diagram saved to {png_path}")
        except Exception as exc:
            print(f"[warn] failed to save graph diagram: {exc}")

    if args.debug:
        print("[debug] topic:", args.topic_name)
        print("[debug] persist_dir:", rinfo["persist_dir"])
        print("[debug] collection:", rinfo["collection"])
        print("[debug] embedding_model:", rinfo["embedding_model"])
        print("[debug] search_type:", rinfo["search_type"])
        print("[debug] k:", rinfo["k"], "lambda_mult:", rinfo["lambda_mult"])
        print("[debug] rerank_model:", rinfo.get("rerank_model"))
        print("[debug] rerank_top_n:", rinfo.get("rerank_top_n"))

    if args.inspect:
        q = args.inspect
        docs_scores = vector_store.similarity_search_with_score(q, k=args.k)
        if not docs_scores:
            print("No documents found for the provided query.")
            return

        print("Top matches (vector similarity order):")
        inspected_docs: List[Document] = []
        for doc, score in docs_scores:
            doc.metadata["similarity_score"] = float(score)
            inspected_docs.append(doc)
            src = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            print(f"- score={score:.4f}  {src} (p{page})")

        if reranker:
            reranked = rerank_documents(q, inspected_docs, reranker, rerank_top_n)
            print("\nRe-ranked (cross-encoder) top results:")
            for doc in reranked:
                src = doc.metadata.get("source", "?")
                page = doc.metadata.get("page", "?")
                rscore = doc.metadata.get("rerank_score")
                sscore = doc.metadata.get("similarity_score")
                score_info = []
                if rscore is not None:
                    score_info.append(f"rerank={float(rscore):.4f}")
                if sscore is not None:
                    score_info.append(f"sim={float(sscore):.4f}")
                display_scores = " ".join(score_info)
                print(f"- {display_scores}  {src} (p{page})")
        else:
            print("\nCross-encoder reranker unavailable; skipping re-rank results.")
        return

    def run_once(q: str) -> None:
        result = app.invoke({"question": q, "context": [], "answer": ""})
        print("\n=== Answer ===\n")
        print(result.get("answer", ""))
        if args.show_sources:
            print("\n--- Sources ---")
            context_docs = result.get("context", [])
            if not context_docs:
                print("(no sources found)")
            else:
                for doc in context_docs:
                    src = doc.metadata.get("source", "?")
                    page = doc.metadata.get("page", "?")
                    rerank_score = doc.metadata.get("rerank_score")
                    score_info = (
                        f" rerank={float(rerank_score):.4f}"
                        if rerank_score is not None
                        else ""
                    )
                    print(f"- {src} (p{page}){score_info}")

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
