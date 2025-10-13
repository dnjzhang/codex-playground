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
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, TypedDict

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
from pydantic import BaseModel, Field


class QAState(TypedDict, total=False):
    question: str
    context: List[Document]
    answer: str
    history: List[BaseMessage]


@dataclass
class PipelineConfig:
    topic_name: str
    persist_root: str = ".chroma"
    collection_name: str | None = None
    embedding_provider: str = "auto"
    embedding_model: str = "auto"
    chat_provider: str = "ollama"
    chat_model: str | None = None
    chat_temperature: float = 0.0
    k: int = 10
    search_type: str = "similarity"
    lambda_mult: float | None = None
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int | None = None
    oci_config: str = "oci-config.json"
    oci_endpoint: str | None = None
    oci_compartment_id: str | None = None
    oci_auth_profile: str | None = None
    user_id: str = "default"
    show_sources: bool = False
    debug: bool = False


@dataclass
class ChatResult:
    answer: str
    sources: List[Dict[str, Any]]
    context_docs: List[Document]


@dataclass
class SessionRuntime:
    config: PipelineConfig
    app: Any
    conversation_memory: InMemoryChatMessageHistory
    retriever_info: Dict[str, Any]
    vector_store: Any
    reranker: Any
    rerank_top_n: int
    resolved_chat_model: str | None

    def run_query(self, question: str) -> ChatResult:
        history_messages = list(self.conversation_memory.messages)
        state: QAState = {
            "question": question,
            "context": [],
            "answer": "",
            "history": history_messages,
        }
        if self.config.debug:
            print("[debug] run_query question:", question)
            print("[debug] run_query history message count:", len(history_messages))
        result = self.app.invoke(state)
        answer_raw = result.get("answer", "")
        answer = answer_raw if isinstance(answer_raw, str) else str(answer_raw)

        context_docs: List[Document] = result.get("context", []) or []
        sources: List[Dict[str, Any]] = []
        if self.config.show_sources:
            for doc in context_docs:
                src = doc.metadata.get("source", "?")
                page = doc.metadata.get("page", "?")
                rerank_score = doc.metadata.get("rerank_score")
                similarity_score = doc.metadata.get("similarity_score")
                source_info: Dict[str, Any] = {
                    "source": src,
                    "page": page,
                }
                if rerank_score is not None:
                    source_info["rerank_score"] = float(rerank_score)
                if similarity_score is not None:
                    source_info["similarity_score"] = float(similarity_score)
                sources.append(source_info)

        self.conversation_memory.add_user_message(question)
        self.conversation_memory.add_ai_message(answer)
        return ChatResult(answer=answer, sources=sources, context_docs=context_docs)

    def inspect(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        docs_scores = self.vector_store.similarity_search_with_score(query, k=self.config.k)
        top_matches: List[Dict[str, Any]] = []
        for doc, score in docs_scores or []:
            doc.metadata["similarity_score"] = float(score)
            top_matches.append(
                {
                    "source": doc.metadata.get("source", "?"),
                    "page": doc.metadata.get("page", "?"),
                    "similarity_score": float(score),
                    "document": doc.page_content,
                }
            )

        reranked_matches: List[Dict[str, Any]] = []
        if self.reranker and docs_scores:
            docs = [doc for doc, _ in docs_scores]
            reranked_docs = rerank_documents(query, docs, self.reranker, self.rerank_top_n)
            for doc in reranked_docs:
                reranked_matches.append(
                    {
                        "source": doc.metadata.get("source", "?"),
                        "page": doc.metadata.get("page", "?"),
                        "rerank_score": float(doc.metadata.get("rerank_score", 0.0)),
                        "similarity_score": float(doc.metadata.get("similarity_score", 0.0)),
                        "document": doc.page_content,
                    }
                )

        return {"top_matches": top_matches, "reranked_matches": reranked_matches}


@dataclass
class ChatSession:
    session_id: str
    runtime: SessionRuntime

    def ask(self, question: str) -> ChatResult:
        return self.runtime.run_query(question)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, ChatSession] = {}
        self._lock = threading.Lock()

    def create_session(self, configuration: PipelineConfig) -> ChatSession:
        runtime = initialize_session(configuration)
        session_id = uuid.uuid4().hex
        chat_session = ChatSession(session_id=session_id, runtime=runtime)
        with self._lock:
            self._sessions[session_id] = chat_session
        return chat_session

    def get_session(self, session_id: str) -> ChatSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session


session_manager = SessionManager()


class SessionCreatePayload(BaseModel):
    topic_name: str = Field(..., description="Topic name used when building the DB")
    persist_root: str = Field(".chroma", description="Root directory where topic DBs are stored")
    collection_name: str | None = Field(None, description="Optional Chroma collection name")
    embedding_provider: str = Field("auto", description="Embedding provider to use")
    embedding_model: str = Field("auto", description="Embedding model identifier")
    chat_provider: str = Field("ollama", description="Chat model provider")
    chat_model: str | None = Field(None, description="Chat model identifier")
    chat_temperature: float = Field(0.0, description="Sampling temperature for the chat model")
    k: int = Field(10, description="Top-k documents to retrieve", gt=0)
    search_type: str = Field("similarity", description="Retrieval search type")
    lambda_mult: float | None = Field(None, description="Diversity factor for MMR (0-1)")
    rerank_model: str = Field("BAAI/bge-reranker-base", description="Cross-encoder model to rerank retrieved documents")
    rerank_top_n: int | None = Field(None, description="Number of reranked documents to send to the chat model", gt=0)
    oci_config: str = Field("oci-config.json", description="Path to OCI config JSON")
    oci_endpoint: str | None = Field(None, description="Override OCI service endpoint")
    oci_compartment_id: str | None = Field(None, description="Override OCI compartment OCID")
    oci_auth_profile: str | None = Field(None, description="Override OCI config profile name")
    user_id: str = Field("default", description="Identifier used to scope conversation memory")
    show_sources: bool = Field(False, description="Include source metadata in responses")
    debug: bool = Field(False, description="Enable debug logging")
    graph_diagram: str | None = Field(None, description="Optional path to save the compiled LangGraph structure as a PNG image")

    def to_pipeline_config(self) -> PipelineConfig:
        return PipelineConfig(
            topic_name=self.topic_name,
            persist_root=self.persist_root,
            collection_name=self.collection_name,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            chat_provider=self.chat_provider,
            chat_model=self.chat_model,
            chat_temperature=self.chat_temperature,
            k=self.k,
            search_type=self.search_type,
            lambda_mult=self.lambda_mult,
            rerank_model=self.rerank_model,
            rerank_top_n=self.rerank_top_n,
            oci_config=self.oci_config,
            oci_endpoint=self.oci_endpoint,
            oci_compartment_id=self.oci_compartment_id,
            oci_auth_profile=self.oci_auth_profile,
            user_id=self.user_id,
            show_sources=self.show_sources,
            debug=self.debug,
        )


class SessionCreateResponseModel(BaseModel):
    session_id: str
    topic_name: str
    chat_provider: str | None = None
    chat_model: str | None = None
    rerank_top_n: int


class ChatRequestModel(BaseModel):
    question: str = Field(..., description="User question for the chatbot")


class ChatResponseModel(BaseModel):
    session_id: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)


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


def _compute_rerank_top_n(k: int, candidate: int | None) -> int:
    if candidate is None:
        return max(1, math.ceil(k / 2))
    return max(1, min(candidate, k))


def initialize_session(config: PipelineConfig) -> SessionRuntime:
    if config.k <= 0:
        raise ValueError("--k must be a positive integer")
    if config.rerank_top_n is not None and config.rerank_top_n <= 0:
        raise ValueError("--rerank-top-n must be a positive integer when provided")

    rerank_top_n = _compute_rerank_top_n(config.k, config.rerank_top_n)

    retriever, retriever_info, vector_store = build_retriever(
        topic_name=config.topic_name,
        persist_root=config.persist_root,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
        collection_name=config.collection_name,
        k=config.k,
        search_type=config.search_type,
        lambda_mult=config.lambda_mult,
        oci_config=config.oci_config,
        oci_config_data=None,
        oci_endpoint=config.oci_endpoint,
        oci_compartment_id=config.oci_compartment_id,
        oci_auth_profile=config.oci_auth_profile,
        debug=config.debug,
    )

    oci_config_data: Dict[str, str] | None = None
    if retriever_info.get("embedding_provider") == "oci" or config.chat_provider == "oci":
        oci_config_data = load_oci_config_data(config.oci_config)

    chat_llm, resolved_chat_model = init_chat_model(
        config.chat_provider,
        config.chat_model,
        temperature=config.chat_temperature,
        oci_config=config.oci_config,
        oci_config_data=oci_config_data,
        oci_endpoint=config.oci_endpoint,
        oci_compartment_id=config.oci_compartment_id,
        oci_auth_profile=config.oci_auth_profile,
    )

    reranker = build_reranker(config.rerank_model)

    app = build_qa_graph(
        chat_llm,
        retriever,
        reranker=reranker,
        rerank_top_n=rerank_top_n,
        debug=config.debug,
    )

    retriever_info["rerank_model"] = config.rerank_model
    retriever_info["rerank_top_n"] = rerank_top_n
    retriever_info["chat_provider"] = config.chat_provider
    retriever_info["chat_model"] = resolved_chat_model

    conversation_memory = InMemoryChatMessageHistory()

    return SessionRuntime(
        config=config,
        app=app,
        conversation_memory=conversation_memory,
        retriever_info=retriever_info,
        vector_store=vector_store,
        reranker=reranker,
        rerank_top_n=rerank_top_n,
        resolved_chat_model=resolved_chat_model,
    )


def create_api_app() -> Any:
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "FastAPI and Pydantic are required to run the API server. "
            "Install them with `pip install -r requirements.txt`."
        ) from exc

    api = FastAPI(title="RAG Topic Hub API", version="1.0.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.post("/sessions", response_model=SessionCreateResponseModel)
    def create_session_endpoint(
        payload: SessionCreatePayload = Body(...),
    ) -> SessionCreateResponseModel:
        config = payload.to_pipeline_config()
        try:
            session = session_manager.create_session(config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        runtime = session.runtime

        if payload.graph_diagram:
            try:
                export_graph_diagram(runtime.app, payload.graph_diagram, debug=config.debug)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save graph diagram: {exc}",
                ) from exc

        return SessionCreateResponseModel(
            session_id=session.session_id,
            topic_name=config.topic_name,
            chat_provider=runtime.retriever_info.get("chat_provider"),
            chat_model=runtime.retriever_info.get("chat_model"),
            rerank_top_n=runtime.rerank_top_n,
        )

    @api.post("/sessions/{session_id}/chat", response_model=ChatResponseModel)
    def chat_endpoint(
        session_id: str,
        payload: ChatRequestModel = Body(...),
    ) -> ChatResponseModel:
        try:
            session = session_manager.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc

        result = session.ask(payload.question)

        return ChatResponseModel(
            session_id=session_id,
            answer=result.answer,
            sources=result.sources,
        )

    return api


def main() -> None:
    parser = argparse.ArgumentParser(description="Query/chat with a per-topic Chroma DB using a simple LangGraph QA pipeline.")
    parser.add_argument(
        "--serve-api",
        action="store_true",
        help="Run the RESTful API server instead of the command-line interface",
    )
    parser.add_argument(
        "--api-host",
        default="127.0.0.1",
        help="Host interface to bind the RESTful API server",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port to bind the RESTful API server",
    )
    parser.add_argument("--topic-name", required=False, help="Topic name used when building the DB")
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

    if args.serve_api:
        if args.api_port <= 0:
            parser.error("--api-port must be a positive integer")
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit(
                "uvicorn is required to run the API server. "
                "Install it with `pip install -r requirements.txt`."
            ) from exc
        app = create_api_app()
        uvicorn.run(app, host=args.api_host, port=args.api_port)
        return

    if args.k <= 0:
        parser.error("--k must be a positive integer")
    if args.rerank_top_n is not None and args.rerank_top_n <= 0:
        parser.error("--rerank-top-n must be a positive integer")

    if not args.topic_name:
        parser.error("--topic-name is required unless --serve-api is specified")

    config = PipelineConfig(
        topic_name=args.topic_name,
        persist_root=args.persist_root,
        collection_name=args.collection_name,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        chat_provider=args.chat_provider,
        chat_model=args.chat_model,
        chat_temperature=args.chat_temperature,
        k=args.k,
        search_type=args.search_type,
        lambda_mult=args.lambda_mult,
        rerank_model=args.rerank_model,
        rerank_top_n=args.rerank_top_n,
        oci_config=args.oci_config,
        oci_endpoint=args.oci_endpoint,
        oci_compartment_id=args.oci_compartment_id,
        oci_auth_profile=args.oci_auth_profile,
        user_id=args.user_id or "default",
        show_sources=args.show_sources,
        debug=args.debug,
    )

    try:
        runtime = initialize_session(config)
    except ValueError as exc:
        parser.error(str(exc))
        return

    if args.graph_diagram:
        try:
            png_path = export_graph_diagram(
                runtime.app,
                args.graph_diagram,
                debug=args.debug,
            )
            if args.debug:
                print(f"[debug] graph diagram saved to {png_path}")
        except Exception as exc:
            print(f"[warn] failed to save graph diagram: {exc}")

    if args.debug:
        rinfo = runtime.retriever_info
        print("[debug] topic:", config.topic_name)
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
        print("[debug] chat_temperature:", config.chat_temperature)
        print("[debug] user_id:", config.user_id)

    if args.inspect:
        q = args.inspect
        inspection = runtime.inspect(q)
        top_matches = inspection["top_matches"]
        if not top_matches:
            print("No documents found for the provided query.")
            return

        print("Top matches (vector similarity order):")
        for doc_info in top_matches:
            score = doc_info.get("similarity_score")
            src = doc_info.get("source", "?")
            page = doc_info.get("page", "?")
            if score is not None:
                print(f"- score={score:.4f}  {src} (p{page})")
            else:
                print(f"- {src} (p{page})")

        if runtime.reranker:
            reranked_matches = inspection["reranked_matches"]
            print("\nRe-ranked (cross-encoder) top results:")
            for doc_info in reranked_matches:
                src = doc_info.get("source", "?")
                page = doc_info.get("page", "?")
                rscore = doc_info.get("rerank_score")
                sscore = doc_info.get("similarity_score")
                score_info = []
                if rscore is not None:
                    score_info.append(f"rerank={rscore:.4f}")
                if sscore is not None:
                    score_info.append(f"sim={sscore:.4f}")
                display_scores = " ".join(score_info)
                print(f"- {display_scores}  {src} (p{page})")
        else:
            print("\nCross-encoder reranker unavailable; skipping re-rank results.")
        return

    def run_once(question: str) -> None:
        result = runtime.run_query(question)

        print("\n=== Answer ===\n")
        print(result.answer)

        if config.show_sources:
            print("\n--- Sources ---")
            context_docs = result.context_docs
            if not context_docs:
                print("(no sources found)")
            else:
                if config.debug:
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
