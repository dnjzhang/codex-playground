"""Provider-specific helpers for the RAG topic hub."""

from __future__ import annotations

import json
import os
from typing import Dict, Tuple

from langchain_ollama import ChatOllama, OllamaEmbeddings

try:
    from langchain_oci import ChatOCIGenAI, OCIGenAIEmbeddings
except ImportError:  # noqa: WPS440 - optional dependency
    ChatOCIGenAI = None  # type: ignore[assignment]
    OCIGenAIEmbeddings = None  # type: ignore[assignment]

from mcp_registration import register_db_mcp_tools


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
    """Instantiate the embedding function for the requested provider."""

    provider_key = provider.lower()

    if provider_key == "ollama":
        model_id = embedding_model or "mxbai-embed-large"
        embeddings = OllamaEmbeddings(model=model_id)
        return embeddings, model_id

    if provider_key == "oci":
        if OCIGenAIEmbeddings is None:
            raise ImportError(
                "langchain-oci embeddings are unavailable. Install the required "
                "extras (e.g., `pip install langchain-oci`)."
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
            truncate="END",
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
    enable_mcp: bool = True,
) -> Tuple[object, str]:
    """Instantiate the chat model for the requested provider."""

    provider_key = provider.lower()

    if provider_key == "ollama":
        model_id = chat_model or "llama3.2"
        llm = ChatOllama(model=model_id, temperature=temperature)
        if enable_mcp:
            llm, _ = register_db_mcp_tools(llm)
        return llm, model_id

    if provider_key == "oci":
        if ChatOCIGenAI is None:
            raise ImportError(
                "langchain-oci chat model support is unavailable. Install the "
                "required extras (e.g., `pip install langchain-oci`)."
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
            model_kwargs={
                "max_tokens": int(config_data.get("max_tokens", 2048)),
                "temperature": temperature,
            },
        )
        if enable_mcp:
            llm, _ = register_db_mcp_tools(llm)
        return llm, model_id

    raise ValueError("Unsupported chat provider. Choose from: ollama, oci.")


def build_reranker(model_name: str):
    """Instantiate a HuggingFace cross-encoder reranker."""

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # noqa: BLE001
        raise ImportError(
            "Install sentence-transformers to use cross-encoder reranking: "
            "`pip install sentence-transformers`"
        ) from exc

    class _SentenceTransformersCrossEncoder:
        def __init__(self, model_id: str) -> None:
            self._model = CrossEncoder(model_id)

        def score(self, pairs):
            return self._model.predict(pairs).tolist()

    try:
        return _SentenceTransformersCrossEncoder(model_name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to initialize cross-encoder reranker. "
            "Ensure required ML dependencies (e.g., torch) are installed."
        ) from exc
