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
      --embedding-provider oci

Notes:
  - Auto-detects embedding provider/model from topic metadata when available
  - Supports Ollama (default) and OCI Generative AI for chat + embeddings
  - Build topic vectors with topic-curator.py to ensure metadata compatibility
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

try:
    from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
except ImportError:  # noqa: WPS440 - optional dependency
    ChatOCIGenAI = None  # type: ignore[assignment]

try:
    from langchain_community.embeddings.oci_generative_ai import (
        OCIGenAIEmbeddings,
    )
except ImportError:  # noqa: WPS440 - optional dependency
    OCIGenAIEmbeddings = None  # type: ignore[assignment]


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


def load_oci_config_data(oci_config: str | None) -> Dict[str, str]:
    """Load OCI configuration data from JSON."""

    config_path = oci_config or "oci-config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"OCI config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            data = json.load(config_file)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise ValueError(
            f"Invalid OCI config JSON at {config_path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected OCI config to contain a JSON object at {config_path}"
        )

    return {str(key): str(value) for key, value in data.items() if value is not None}


def resolve_oci_base_settings(
    config_data: Dict[str, str],
    *,
    endpoint_override: str | None,
    compartment_override: str | None,
    auth_profile_override: str | None,
) -> Dict[str, str]:
    """Resolve core OCI settings with CLI overrides taking precedence."""

    settings = {
        "service_endpoint": endpoint_override or config_data.get("endpoint"),
        "compartment_id": compartment_override or config_data.get("compartment_ocid"),
        "auth_profile": auth_profile_override or config_data.get("config_profile"),
    }

    missing = [key for key, value in settings.items() if not value]
    if missing:
        details = ", ".join(missing)
        raise ValueError(
            "Missing OCI configuration values: "
            f"{details}. Provide CLI overrides or update the OCI config JSON."
        )

    return {key: str(value) for key, value in settings.items()}


def init_embedding_function(
    provider: str,
    embedding_model: str | None,
    *,
    oci_config: str | None,
    oci_config_data: Dict[str, str] | None,
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
                "langchain-community OCI embeddings are unavailable. Install the"
                " required extras (e.g., `pip install langchain-community`)."
            )

        config_data = oci_config_data or load_oci_config_data(oci_config)
        settings = resolve_oci_base_settings(
            config_data,
            endpoint_override=oci_endpoint,
            compartment_override=oci_compartment_id,
            auth_profile_override=oci_auth_profile,
        )
        model_id = (
            embedding_model
            or config_data.get("embedding_model_name")
            or config_data.get("model_name")
        )
        if not model_id:
            raise ValueError(
                "OCI embedding model id is missing. Provide --embedding-model or"
                " set embedding_model_name in the OCI config."
            )

        embeddings = OCIGenAIEmbeddings(
            model_id=model_id,
            service_endpoint=settings["service_endpoint"],
            compartment_id=settings["compartment_id"],
            auth_profile=settings["auth_profile"],
            model_kwargs={"truncate": True},
        )
        return embeddings, model_id

    raise ValueError("Unsupported embedding provider. Choose from: ollama, oci, auto.")


def init_chat_model(
    provider: str,
    chat_model: str | None,
    *,
    temperature: float,
    oci_config: str | None,
    oci_config_data: Dict[str, str] | None,
    oci_endpoint: str | None,
    oci_compartment_id: str | None,
    oci_auth_profile: str | None,
) -> Tuple[object, str]:
    provider_key = provider.lower()

    if provider_key == "ollama":
        model_id = chat_model or "llama3.2"
        llm = ChatOllama(model=model_id, temperature=temperature)
        return llm, model_id

    if provider_key == "oci":
        if ChatOCIGenAI is None:
            raise ImportError(
                "langchain-community OCI chat model support is unavailable. Install"
                " the required extras (e.g., `pip install langchain-community`)."
            )

        config_data = oci_config_data or load_oci_config_data(oci_config)
        settings = resolve_oci_base_settings(
            config_data,
            endpoint_override=oci_endpoint,
            compartment_override=oci_compartment_id,
            auth_profile_override=oci_auth_profile,
        )
        model_id = chat_model or config_data.get("model_name")
        if not model_id:
            raise ValueError(
                "OCI chat model id is missing. Provide --chat-model or set"
                " model_name in the OCI config."
            )

        llm = ChatOCIGenAI(
            model_id=model_id,
            service_endpoint=settings["service_endpoint"],
            compartment_id=settings["compartment_id"],
            auth_profile=settings["auth_profile"],
            model_kwargs={"max_tokens": 2048, "temperature": temperature},
        )
        return llm, model_id

    raise ValueError("Unsupported chat provider. Choose from: ollama, oci.")


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
):
    persist_dir = os.path.join(persist_root, topic_name)
    collection = collection_name or topic_name

    meta = _load_db_metadata(persist_dir) or {}

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
    kwargs = {"k": k}
    if lambda_mult is not None:
        kwargs["lambda_mult"] = lambda_mult
    retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=kwargs)
    info = {
        "persist_dir": persist_dir,
        "collection": collection,
        "embedding_provider": provider_key,
        "embedding_model": resolved_model,
        "k": k,
        "search_type": search_type,
        "lambda_mult": lambda_mult,
    }
    return retriever, info, vector_store


def build_qa_graph(
    llm,
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
        print("[debug] embedding_provider:", rinfo.get("embedding_provider"))
        print("[debug] embedding_model:", rinfo["embedding_model"])
        print("[debug] search_type:", rinfo["search_type"])
        print("[debug] k:", rinfo["k"], "lambda_mult:", rinfo["lambda_mult"])
        print("[debug] rerank_model:", rinfo.get("rerank_model"))
        print("[debug] rerank_top_n:", rinfo.get("rerank_top_n"))
        print("[debug] chat_provider:", rinfo.get("chat_provider"))
        print("[debug] chat_model:", rinfo.get("chat_model"))
        print("[debug] chat_temperature:", args.chat_temperature)

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
        result = app.invoke({"question"})
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
