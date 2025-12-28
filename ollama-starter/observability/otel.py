"""OpenTelemetry + OpenInference instrumentation helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Mapping, Optional, Sequence

from langchain_core.callbacks import BaseCallbackHandler

LOGGER = logging.getLogger(__name__)

_OBSERVABILITY: "ObservabilityManager | None" = None
_INIT_LOCK = threading.Lock()
_QA_LOGGER: logging.Logger | None = None


def new_run_id() -> str:
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _str_to_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _extract_usage(response: Any) -> Dict[str, Optional[int]]:
    usage: Mapping[str, Any] | None = None

    def _from_mapping(candidate: Any) -> Mapping[str, Any] | None:
        if not isinstance(candidate, Mapping):
            return None
        token_usage = candidate.get("token_usage") or candidate.get("usage")
        token_usage = token_usage or candidate.get("usage_metadata")
        if isinstance(token_usage, Mapping):
            return token_usage
        return None

    llm_output = getattr(response, "llm_output", None)
    usage = _from_mapping(llm_output)

    if usage is None:
        generations = getattr(response, "generations", None) or []
        for generation_group in generations:
            for generation in generation_group:
                usage = _from_mapping(getattr(generation, "generation_info", None))
                if usage is not None:
                    break
                message = getattr(generation, "message", None)
                usage = _from_mapping(getattr(message, "usage_metadata", None))
                if usage is None:
                    metadata = getattr(message, "response_metadata", None)
                    if isinstance(metadata, Mapping):
                        usage = metadata
                if usage is not None:
                    break
            if usage is not None:
                break

    def _coerce(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    if usage is None:
        return {"prompt": None, "completion": None, "total": None}

    prompt = _coerce(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("prompt_token_count")
        or usage.get("prompt_eval_count")
    )
    completion = _coerce(
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("completion_token_count")
        or usage.get("eval_count")
    )
    total = _coerce(usage.get("total_tokens") or usage.get("total_token_count"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion

    return {"prompt": prompt, "completion": completion, "total": total}


def _extract_model_name(response: Any) -> Optional[str]:
    llm_output = getattr(response, "llm_output", None)
    if isinstance(llm_output, Mapping):
        for key in ("model", "model_name", "model_id"):
            value = llm_output.get(key)
            if value:
                return str(value)

    generations = getattr(response, "generations", None) or []
    for generation_group in generations:
        for generation in generation_group:
            message = getattr(generation, "message", None)
            metadata = getattr(message, "response_metadata", None)
            if isinstance(metadata, Mapping):
                for key in ("model", "model_name", "model_id"):
                    value = metadata.get(key)
                    if value:
                        return str(value)

    return None


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, socket.timeout):
        return True
    try:
        import asyncio

        if isinstance(exc, asyncio.TimeoutError):
            return True
    except Exception:
        pass
    try:
        import httpx

        if isinstance(exc, httpx.TimeoutException):
            return True
    except Exception:
        pass
    try:
        import requests

        if isinstance(exc, requests.Timeout):
            return True
    except Exception:
        pass
    return False


def _classify_error(exc: Exception) -> Dict[str, Any]:
    message = str(exc) or exc.__class__.__name__
    message_lower = message.lower()

    if _is_timeout_error(exc):
        stage = "llm" if any(token in message_lower for token in ("ollama", "oci", "model", "llm")) else "graph"
        return {
            "kind": "timeout",
            "stage": stage,
            "message": message,
            "retryable": True,
        }

    if any(token in message_lower for token in ("tool", "mcp")):
        return {
            "kind": "tool_error",
            "stage": "tool",
            "message": message,
            "retryable": False,
        }

    if any(token in message_lower for token in ("retriever", "retrieval", "chroma", "vector")):
        return {
            "kind": "retrieval_error",
            "stage": "retrieval",
            "message": message,
            "retryable": False,
        }

    if any(token in message_lower for token in ("ollama", "oci", "model", "llm")):
        return {
            "kind": "llm_error",
            "stage": "llm",
            "message": message,
            "retryable": False,
        }

    return {
        "kind": "graph_error",
        "stage": "graph",
        "message": message,
        "retryable": False,
    }


def emit_run_log(
    log_type: str,
    question: str,
    answer: str,
    run_id: str,
    *,
    start_timestamp: Optional[str] = None,
    elapsed_ms: Optional[float] = None,
    error_obj: Optional[Dict[str, Any]] = None,
    extra_attrs: Optional[Dict[str, Any]] = None,
    qa_log_content: bool = True,
) -> None:
    if _QA_LOGGER is None:
        return

    payload: Dict[str, Any] = {
        "type": log_type,
        "question": question,
        "answer": answer,
        "complete_timestamp": utc_now_iso(),
        "run_id": run_id,
    }
    if start_timestamp is not None:
        payload["start_timestamp"] = start_timestamp
    if elapsed_ms is not None:
        payload["elapsed_ms"] = elapsed_ms

    if not qa_log_content:
        payload["question"] = ""
        payload["answer"] = ""
        payload["question_hash"] = sha256_text(question)
        payload["answer_hash"] = sha256_text(answer)

    if error_obj is not None:
        payload["error"] = error_obj

    attributes: Dict[str, Any] = {
        "app.type": log_type,
        "app.run_id": run_id,
    }
    if extra_attrs:
        attributes.update(extra_attrs)

    _QA_LOGGER.info(json.dumps(payload, ensure_ascii=False), extra=attributes)


class OTelMetricsCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler emitting OTel metrics."""

    def __init__(self, meter: Any) -> None:
        self._meter = meter
        self._lock = threading.Lock()
        self._retriever_start: Dict[str, float] = {}

        self._prompt_tokens = meter.create_counter(
            "llm_prompt_tokens",
            unit="1",
            description="Prompt tokens used by LLM calls",
        )
        self._completion_tokens = meter.create_counter(
            "llm_completion_tokens",
            unit="1",
            description="Completion tokens used by LLM calls",
        )
        self._total_tokens = meter.create_counter(
            "llm_total_tokens",
            unit="1",
            description="Total tokens used by LLM calls",
        )
        self._tool_calls = meter.create_counter(
            "rag_tool_calls",
            unit="1",
            description="Number of tool calls executed",
        )
        self._retrieval_latency_ms = meter.create_histogram(
            "rag_retrieval_latency_ms",
            unit="ms",
            description="Retriever latency in milliseconds",
        )
        self._retrieval_docs = meter.create_histogram(
            "rag_retrieval_documents",
            unit="1",
            description="Number of documents returned by retrieval",
        )
        self._retrieval_context_chars = meter.create_histogram(
            "rag_retrieval_context_chars",
            unit="1",
            description="Total characters returned in retrieval context",
        )

    def on_llm_end(self, response: Any, *, run_id: str, **kwargs: Any) -> None:
        self._record_usage(response)

    def on_chat_model_end(self, response: Any, *, run_id: str, **kwargs: Any) -> None:
        self._record_usage(response)

    def _record_usage(self, response: Any) -> None:
        usage = _extract_usage(response)
        model_name = _extract_model_name(response)
        attrs = {"model": model_name} if model_name else {}

        if usage["prompt"] is not None:
            self._prompt_tokens.add(usage["prompt"], attrs)
        if usage["completion"] is not None:
            self._completion_tokens.add(usage["completion"], attrs)
        if usage["total"] is not None:
            self._total_tokens.add(usage["total"], attrs)

    def on_retriever_start(self, serialized: Any, query: str, *, run_id: str, **kwargs: Any) -> None:
        key = str(run_id)
        with self._lock:
            self._retriever_start[key] = time.perf_counter()

    def on_retriever_end(self, documents: Sequence[Any], *, run_id: str, **kwargs: Any) -> None:
        start_time = None
        with self._lock:
            start_time = self._retriever_start.pop(str(run_id), None)

        if start_time is not None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._retrieval_latency_ms.record(elapsed_ms)

        doc_count = len(documents or [])
        context_chars = 0
        for doc in documents or []:
            content = getattr(doc, "page_content", None)
            if content:
                context_chars += len(content)

        self._retrieval_docs.record(doc_count)
        self._retrieval_context_chars.record(context_chars)

    def record_tool_call(self, tool_name: str | None = None) -> None:
        attrs = {"tool.name": tool_name} if tool_name else {}
        self._tool_calls.add(1, attrs)


@dataclass
class ObservabilityManager:
    enabled: bool
    callbacks: Sequence[BaseCallbackHandler] = field(default_factory=tuple)
    _tool_callback: OTelMetricsCallbackHandler | None = None
    _trace: Any = None
    _qa_logging_enabled: bool = False
    _qa_log_content: bool = True
    _qa_lock: threading.Lock = field(default_factory=threading.Lock)
    _qa_logged: set[str] = field(default_factory=set)

    def callback_config(self) -> Optional[Dict[str, Any]]:
        if not self.enabled or not self.callbacks:
            return None
        return {"callbacks": list(self.callbacks)}

    def record_tool_call(self, tool_name: str | None = None) -> None:
        if not self.enabled:
            return
        if self._tool_callback is not None:
            self._tool_callback.record_tool_call(tool_name)

        if self._trace is None:
            return
        span = self._trace.get_current_span()
        if span and span.is_recording():
            attrs = {"tool.name": tool_name} if tool_name else {}
            span.add_event("tool.call", attrs)

    def attach_run_id(self, run_id: str) -> None:
        if not self.enabled or self._trace is None:
            return
        span = self._trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute("app.run_id", run_id)

    def span_for_run(self, run_id: str):
        if not self.enabled or self._trace is None:
            return nullcontext()

        tracer = self._trace.get_tracer("rag-pipeline-topic-hub")

        @contextmanager
        def _span():
            with tracer.start_as_current_span("agent.run") as span:
                span.set_attribute("app.run_id", run_id)
                yield span

        return _span()

    def log_response(
        self,
        run_id: str,
        question: str,
        answer: str,
        *,
        start_timestamp: str,
        elapsed_ms: float,
    ) -> None:
        if not self._qa_logging_enabled:
            return
        if not self._mark_logged(run_id):
            return
        emit_run_log(
            "response",
            question,
            answer,
            run_id,
            start_timestamp=start_timestamp,
            elapsed_ms=elapsed_ms,
            qa_log_content=self._qa_log_content,
        )

    def log_error(
        self,
        run_id: str,
        question: str,
        answer: str,
        exc: Exception,
        *,
        start_timestamp: str,
        elapsed_ms: float,
    ) -> None:
        if not self._qa_logging_enabled:
            return
        if not self._mark_logged(run_id):
            return
        error_obj = _classify_error(exc)
        emit_run_log(
            "error",
            question,
            answer,
            run_id,
            start_timestamp=start_timestamp,
            elapsed_ms=elapsed_ms,
            error_obj=error_obj,
            qa_log_content=self._qa_log_content,
        )

    def _mark_logged(self, run_id: str) -> bool:
        with self._qa_lock:
            if run_id in self._qa_logged:
                return False
            self._qa_logged.add(run_id)
            return True


def _instrument_openinference(tracer_provider: Any, capture_content: bool) -> None:
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("OpenInference LangChain instrumentation unavailable: %s", exc)
        return

    hide_inputs = not capture_content
    hide_outputs = not capture_content

    if not capture_content:
        os.environ.setdefault("OPENINFERENCE_HIDE_INPUTS", "true")
        os.environ.setdefault("OPENINFERENCE_HIDE_OUTPUTS", "true")

    kwargs = {"tracer_provider": tracer_provider}
    if hide_inputs is not None:
        kwargs["hide_inputs"] = hide_inputs
    if hide_outputs is not None:
        kwargs["hide_outputs"] = hide_outputs

    try:
        LangChainInstrumentor().instrument(**kwargs)
    except TypeError:
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    try:
        from openinference.instrumentation.langgraph import LangGraphInstrumentor
    except Exception:
        return

    try:
        LangGraphInstrumentor().instrument(tracer_provider=tracer_provider)
    except TypeError:
        LangGraphInstrumentor().instrument()


def init_observability() -> ObservabilityManager:
    """Initialize OpenTelemetry + OpenInference instrumentation."""

    global _OBSERVABILITY

    with _INIT_LOCK:
        if _OBSERVABILITY is not None:
            return _OBSERVABILITY

        enabled = _str_to_bool(os.getenv("OBSERVABILITY_ENABLED"), default=False)
        if not enabled:
            _OBSERVABILITY = ObservabilityManager(enabled=False)
            return _OBSERVABILITY

        try:
            from opentelemetry import metrics, trace
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource, SERVICE_NAME
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("OpenTelemetry SDK unavailable: %s", exc)
            _OBSERVABILITY = ObservabilityManager(enabled=False)
            return _OBSERVABILITY

        service_name = os.getenv("OTEL_SERVICE_NAME", "rag-pipeline-topic-hub")
        resource = Resource.create({SERVICE_NAME: service_name})

        endpoint = (
            os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            or "http://localhost:4317"
        )
        metrics_endpoint = (
            os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            or endpoint
        )
        insecure = _str_to_bool(
            os.getenv("OTEL_EXPORTER_OTLP_INSECURE"),
            default=endpoint.startswith("http://"),
        )

        tracer_provider = TracerProvider(resource=resource)
        trace_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=insecure)
        tracer_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
        trace.set_tracer_provider(tracer_provider)

        metric_exporter = OTLPMetricExporter(endpoint=metrics_endpoint, insecure=insecure)
        metric_reader = PeriodicExportingMetricReader(metric_exporter)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

        capture_content = _str_to_bool(
            os.getenv("OBSERVABILITY_CAPTURE_CONTENT"),
            default=False,
        )
        _instrument_openinference(tracer_provider, capture_content)

        meter = metrics.get_meter("rag-pipeline-topic-hub")
        metrics_handler = OTelMetricsCallbackHandler(meter)

        qa_logging_enabled = _str_to_bool(
            os.getenv("QA_LOGGING_ENABLED"),
            default=False,
        )
        qa_log_content = _str_to_bool(
            os.getenv("QA_LOG_CONTENT"),
            default=False,
        )
        global _QA_LOGGER
        if qa_logging_enabled:
            try:
                from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
                    OTLPLogExporter,
                )
                from opentelemetry._logs import set_logger_provider
                from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
                from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("OpenTelemetry log exporter unavailable: %s", exc)
                qa_logging_enabled = False
            else:
                logger_provider = LoggerProvider(resource=resource)
                log_exporter = OTLPLogExporter(endpoint=endpoint, insecure=insecure)
                logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
                set_logger_provider(logger_provider)
                handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
                qa_logger = logging.getLogger("qa_logging")
                if not qa_logger.handlers:
                    qa_logger.addHandler(handler)
                qa_logger.setLevel(logging.INFO)
                qa_logger.propagate = False
                _QA_LOGGER = qa_logger

        _OBSERVABILITY = ObservabilityManager(
            enabled=True,
            callbacks=(metrics_handler,),
            _tool_callback=metrics_handler,
            _trace=trace,
            _qa_logging_enabled=qa_logging_enabled,
            _qa_log_content=qa_log_content,
        )
        return _OBSERVABILITY
