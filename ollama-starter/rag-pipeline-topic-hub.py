#!/usr/bin/env python3
"""Topic-centric RAG hub for querying per-topic Chroma databases.

Examples:
  # One-off query with Ollama models
  python ollama-starter/rag-pipeline-topic-hub.py \
      --topic-name spring-boot \
      --query "What is Spring Boot auto-configuration?"

  # Interactive chatbot using OCI chat + embeddings
  python ollama-starter/rag-pipeline-topic-hub.py \
      --topic-name spring-boot \
      --chat-provider oci \
      --embedding-provider oci \
      --user-id alice

Notes:
  - Auto-detects embedding provider/model from topic metadata when available
  - Supports Ollama (default) and OCI Generative AI for chat + embeddings
  - Build topic vectors with topic-curator.py to ensure metadata compatibility
  - Conversation memory persists per user ID when running multiple turns
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Sequence, Tuple, TypedDict

from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import END, StateGraph

from provider_lib import (
    build_reranker,
    init_chat_model,
    init_embedding_function,
    load_oci_config_data,
)


class QAState(TypedDict, total=False):
    question: str
    context: List[Document]
    answer: str
    history: List[BaseMessage]


def _load_db_metadata(persist_dir: str) -> Dict | None:
    meta_path = os.path.join(persist_dir, "db_meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


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


def _resolve_persist_dir(
    persist_root: str,
    topic_name: str,
    *,
    debug: bool = False,
) -> Tuple[str, List[str]]:
    """Resolve the vector store directory, tolerating different run locations."""

    candidates: List[str] = []
    root_is_absolute = os.path.isabs(persist_root)
    primary = os.path.join(persist_root, topic_name)
    candidates.append(primary)

    if not root_is_absolute:
        module_root = os.path.dirname(os.path.abspath(__file__))
        module_candidate = os.path.join(module_root, persist_root, topic_name)
        if module_candidate not in candidates:
            candidates.append(module_candidate)

    resolved = None
    for path in candidates:
        if os.path.isdir(path):
            resolved = path
            break

    if resolved is None:
        resolved = primary

    if debug:
        print(
            "[debug] retriever persist candidates:",
            " | ".join(candidates),
        )
        exists = "yes" if os.path.isdir(resolved) else "no"
        print(f"[debug] retriever persist_dir resolved: {resolved} (exists={exists})")

    return resolved, candidates


def build_retriever(
    topic_name: str,
    persist_root: str,
    embedding_provider: str,
    embedding_model: str | None,
    collection_name: str | None = None,
    k: int = 5,
    search_type: str = "similarity",
    lambda_mult: float | None = None,
    *,
    oci_config: str | None,
    oci_config_data: Dict[str, str] | None,
    oci_endpoint: str | None,
    oci_compartment_id: str | None,
    oci_auth_profile: str | None,
    debug: bool = False,
):
    persist_dir, persist_candidates = _resolve_persist_dir(
        persist_root,
        topic_name,
        debug=debug,
    )
    collection = collection_name or topic_name

    meta = _load_db_metadata(persist_dir) or {}

    if debug:
        meta_path = os.path.join(persist_dir, "db_meta.json")
        print(f"[debug] retriever meta path: {meta_path}")
        print(
            "[debug] retriever meta keys:",
            sorted(meta.keys()) if meta else "none",
        )

    provider_key = embedding_provider.lower()
    if provider_key in {"", "auto"}:
        provider_key = str(meta.get("embedding_provider", "ollama")).lower()
    if provider_key not in {"ollama", "oci"}:
        raise ValueError(
            "Unsupported embedding provider. Choose from: auto, ollama, oci."
        )

    candidate_model = embedding_model
    if not candidate_model or candidate_model == "auto":
        candidate_model = meta.get("embedding_model")

    embeddings, resolved_model = init_embedding_function(
        provider_key,
        candidate_model,
        oci_config=oci_config,
        oci_config_data=oci_config_data,
        oci_endpoint=oci_endpoint,
        oci_compartment_id=oci_compartment_id,
        oci_auth_profile=oci_auth_profile,
    )

    vector_store = Chroma(
        collection_name=collection,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    document_count = None
    try:
        document_count = vector_store._collection.count()
    except Exception:  # noqa: BLE001 - Chroma internals are optional
        if debug:
            print("[debug] unable to query Chroma document count")

    if debug:
        print(
            "[debug] retriever initialized:",
            f"provider={provider_key}",
            f"model={resolved_model}",
            f"collection={collection}",
        )
        print(
            "[debug] retriever Chroma stats:",
            f"documents={document_count if document_count is not None else 'unknown'}",
        )

    kwargs = {"k": k}
    if lambda_mult is not None:
        kwargs["lambda_mult"] = lambda_mult
    retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=kwargs)
    info = {
        "persist_dir": persist_dir,
        "persist_candidates": persist_candidates,
        "collection": collection,
        "embedding_provider": provider_key,
        "embedding_model": resolved_model,
        "k": k,
        "search_type": search_type,
        "lambda_mult": lambda_mult,
        "document_count": document_count,
    }
    return retriever, info, vector_store


def build_qa_graph(
    llm,
    retriever,
    *,
    reranker=None,
    rerank_top_n: int | None = None,
    system_preamble: str | None = None,
    debug: bool = False,
):
    system_text = system_preamble or (
        "You are a helpful assistant. Use the provided context to answer "
        "the user's question concisely. If the answer is unknown, say so."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}\n\nProvide a clear, source-grounded answer.",
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    def retrieve_node(state: QAState) -> Dict:
        question = state["question"]
        docs: List[Document] = retriever.invoke(question)
        if debug:
            print(
                "[debug] retrieve_node: question=",
                repr(question),
                "docs_found=",
                len(docs),
            )
        desired_top_n = rerank_top_n or len(docs)
        if reranker and docs:
            context_docs = rerank_documents(question, docs, reranker, desired_top_n)
            if debug:
                top_doc = context_docs[0] if context_docs else None
                if top_doc is not None:
                    print(
                        "[debug] retrieve_node reranked top source:",
                        top_doc.metadata.get("source"),
                        "page=",
                        top_doc.metadata.get("page"),
                        "score=",
                        top_doc.metadata.get("rerank_score"),
                    )
        else:
            context_docs = docs[:desired_top_n]
            if debug and context_docs:
                top_doc = context_docs[0]
                print(
                    "[debug] retrieve_node top source:",
                    top_doc.metadata.get("source"),
                    "page=",
                    top_doc.metadata.get("page"),
                )
        return {"context": context_docs, "history": state.get("history", [])}

    def generate_node(state: QAState) -> Dict:
        # Join docs into text blocks for the prompt
        context_text = "\n\n".join(
            f"[source: {d.metadata.get('source','?')} p{d.metadata.get('page','?')}]\n{d.page_content}"
            for d in state.get("context", [])
        )
        if debug:
            context_bytes = len(context_text.encode("utf-8"))
            print(
                "[debug] generate_node context docs:",
                len(state.get("context", [])),
                "context_bytes=",
                context_bytes,
            )
        answer = chain.invoke(
            {
                "question": state["question"],
                "context": context_text,
                "history": state.get("history", []),
            }
        )
        if debug:
            preview = answer[:120].replace("\n", " ") if isinstance(answer, str) else str(answer)
            print("[debug] generate_node answer preview:", preview)
        return {
            "answer": answer,
            "context": state.get("context", []),
            "history": state.get("history", []),
        }

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
    parser.add_argument(
        "--persist-root",
        default=".chroma",
        help="Root directory where topic DBs are stored",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Optional Chroma collection name; defaults to the topic name",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["auto", "ollama", "oci"],
        default="auto",
        help="Embedding provider to use",
    )
    parser.add_argument(
        "--embedding-model",
        default="auto",
        help="Embedding model identifier; 'auto' uses DB metadata",
    )
    parser.add_argument(
        "--chat-provider",
        choices=["ollama", "oci"],
        default="ollama",
        help="Chat model provider",
    )
    parser.add_argument(
        "--chat-model",
        default=None,
        help="Chat model identifier; defaults depend on provider",
    )
    parser.add_argument(
        "--chat-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the chat model",
    )
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
    parser.add_argument(
        "--oci-config",
        default="oci-config.json",
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
    parser.add_argument(
        "--user-id",
        default="default",
        help="Identifier used to scope conversation memory",
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

    memory_store: Dict[str, InMemoryChatMessageHistory] = {}

    def get_user_memory(user_key: str) -> InMemoryChatMessageHistory:
        key = user_key or "default"
        if key not in memory_store:
            memory_store[key] = InMemoryChatMessageHistory()
        return memory_store[key]

    user_id = args.user_id or "default"
    conversation_memory = get_user_memory(user_id)

    retriever, rinfo, vector_store = build_retriever(
        topic_name=args.topic_name,
        persist_root=args.persist_root,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        collection_name=args.collection_name,
        k=args.k,
        search_type=args.search_type,
        lambda_mult=args.lambda_mult,
        oci_config=args.oci_config,
        oci_config_data=None,
        oci_endpoint=args.oci_endpoint,
        oci_compartment_id=args.oci_compartment_id,
        oci_auth_profile=args.oci_auth_profile,
        debug=args.debug,
    )

    oci_config_data: Dict[str, str] | None = None
    if rinfo.get("embedding_provider") == "oci" or args.chat_provider == "oci":
        oci_config_data = load_oci_config_data(args.oci_config)

    chat_llm, resolved_chat_model = init_chat_model(
        args.chat_provider,
        args.chat_model,
        temperature=args.chat_temperature,
        oci_config=args.oci_config,
        oci_config_data=oci_config_data,
        oci_endpoint=args.oci_endpoint,
        oci_compartment_id=args.oci_compartment_id,
        oci_auth_profile=args.oci_auth_profile,
    )

    reranker = build_reranker(args.rerank_model)

    rinfo["rerank_model"] = args.rerank_model
    rinfo["rerank_top_n"] = rerank_top_n
    rinfo["chat_provider"] = args.chat_provider
    rinfo["chat_model"] = resolved_chat_model

    app = build_qa_graph(
        llm=chat_llm,
        retriever=retriever,
        reranker=reranker,
        rerank_top_n=rerank_top_n,
        debug=args.debug,
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
        print("[debug] persist_candidates:", " | ".join(rinfo.get("persist_candidates", [])))
        print("[debug] collection:", rinfo["collection"])
        print("[debug] embedding_provider:", rinfo.get("embedding_provider"))
        print("[debug] embedding_model:", rinfo["embedding_model"])
        print("[debug] search_type:", rinfo["search_type"])
        print("[debug] k:", rinfo["k"], "lambda_mult:", rinfo["lambda_mult"])
        if rinfo.get("document_count") is not None:
            print("[debug] retriever_document_count:", rinfo["document_count"])
        print("[debug] rerank_model:", rinfo.get("rerank_model"))
        print("[debug] rerank_top_n:", rinfo.get("rerank_top_n"))
        print("[debug] chat_provider:", rinfo.get("chat_provider"))
        print("[debug] chat_model:", rinfo.get("chat_model"))
        print("[debug] chat_temperature:", args.chat_temperature)
        print("[debug] user_id:", user_id)

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
        history_messages = list(conversation_memory.messages)
        state: QAState = {
            "question": q,
            "context": [],
            "answer": "",
            "history": history_messages,
        }
        if args.debug:
            print("[debug] run_once question:", q)
            print("[debug] run_once history message count:", len(history_messages))
        result = app.invoke(state)
        answer_raw = result.get("answer", "")
        answer = answer_raw if isinstance(answer_raw, str) else str(answer_raw)
        print("\n=== Answer ===\n")
        print(answer)
        if args.show_sources:
            print("\n--- Sources ---")
            context_docs = result.get("context", [])
            if not context_docs:
                print("(no sources found)")
            else:
                if args.debug:
                    print("[debug] run_once context doc count:", len(context_docs))
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

        conversation_memory.add_user_message(q)
        conversation_memory.add_ai_message(answer)

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
