"""Utilities for registering Model Context Protocol (MCP) tools with chat models.

This module centralises the logic to discover tools exposed by the local
``db-mcp-server`` instance and bind them to LangChain chat models so they can
issue tool calls when supported by the underlying provider.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union, get_args, get_origin

import concurrent.futures

from langchain_core.tools import StructuredTool
from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField, create_model

try:
    from mcp import types as mcp_types
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:  # pragma: no cover - optional dependency
    mcp_types = None  # type: ignore[assignment]
    ClientSession = None  # type: ignore[assignment]
    streamablehttp_client = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


def _str_to_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return _str_to_bool(value, default=default)
    return default


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_tool_whitelist(value: Any) -> Optional[Sequence[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(
            entry.strip()
            for entry in value.split(",")
            if entry.strip()
        ) or None
    if isinstance(value, Sequence):
        items = []
        for entry in value:
            if entry is None:
                continue
            text = str(entry).strip()
            if text:
                items.append(text)
        return tuple(items) if items else None
    return None


_JSON_TYPE_TO_PYTHON: Dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": List[Any],
    "object": Dict[str, Any],
    "null": type(None),
    "any": Any,
}


def _resolve_python_type(schema: Mapping[str, Any]) -> Any:
    type_value = schema.get("type")

    if isinstance(type_value, list):
        non_null = [item for item in type_value if str(item).lower() != "null"]
        if not non_null:
            return Optional[Any]
        primary = non_null[0]
        base = _JSON_TYPE_TO_PYTHON.get(str(primary).lower(), Any)
        return Optional[base]

    if isinstance(type_value, str):
        key = type_value.lower()
        if key == "array":
            items = schema.get("items")
            if isinstance(items, Mapping):
                inner = _resolve_python_type(items)
            else:
                inner = Any
            return List[inner]
        if key == "object":
            return Dict[str, Any]
        return _JSON_TYPE_TO_PYTHON.get(key, Any)

    return Any


def _ensure_optional(py_type: Any) -> Any:
    origin = get_origin(py_type)
    if origin is Union:
        if type(None) in get_args(py_type):
            return py_type
        return Optional[Any]
    return Optional[py_type]


def _create_args_model(spec: Mapping[str, Any]) -> Type[PydanticBaseModel]:
    parameters = spec.get("parameters") if isinstance(spec, Mapping) else {}
    properties: Mapping[str, Any] = {}
    if isinstance(parameters, Mapping):
        params_props = parameters.get("properties")
        if isinstance(params_props, Mapping):
            properties = params_props
    if not properties and isinstance(spec.get("properties"), Mapping):
        properties = spec["properties"]  # type: ignore[assignment]

    required: Iterable[str] = []
    if isinstance(parameters, Mapping) and isinstance(parameters.get("required"), Iterable):
        required = parameters.get("required")  # type: ignore[assignment]
    additional_required = spec.get("required")
    if isinstance(additional_required, Iterable):
        required = list(required) + list(additional_required)
    required_set = {str(item) for item in required}

    fields: Dict[str, Tuple[Any, Any]] = {}
    for name, prop in properties.items():
        if not isinstance(prop, Mapping):
            continue
        python_type = _resolve_python_type(prop)
        description = prop.get("description", "")
        default = prop.get("default", None)
        if name in required_set:
            fields[name] = (
                python_type,
                PydanticField(..., description=description),
            )
        else:
            optional_type = _ensure_optional(python_type)
            fields[name] = (
                optional_type,
                PydanticField(default, description=description),
            )

    model_name = f"MCP{spec.get('name', 'Tool').title()}Args"
    if not fields:
        return create_model(model_name, __base__=PydanticBaseModel)  # type: ignore[return-value]

    return create_model(model_name, __base__=PydanticBaseModel, **fields)  # type: ignore[return-value]


def _spec_to_structured_tool(spec: Mapping[str, Any]) -> StructuredTool:
    args_model = _create_args_model(spec)

    def _stub(**kwargs: Any) -> Dict[str, Any]:
        return {
            "tool": spec.get("name"),
            "args": kwargs,
            "message": "MCP tools are executed by the pipeline runtime.",
        }

    _stub.__name__ = f"mcp_{spec.get('name', 'tool')}_stub"

    return StructuredTool(
        name=str(spec.get("name")),
        description=str(spec.get("description", "")),
        args_schema=args_model,
        func=_stub,
    )


def build_structured_tools_from_specs(specs: Sequence[Mapping[str, Any]]) -> List[StructuredTool]:
    return [_spec_to_structured_tool(spec) for spec in specs]


def _ensure_logger_configured() -> None:
    """Attach a fallback handler so INFO-level messages surface by default."""

    if LOGGER.handlers:
        return

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[mcp] %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


@dataclass(slots=True)
class MCPRegistrationConfig:
    """Runtime options for discovering and binding MCP tools."""

    url: str = "http://localhost:8080/mcp"
    request_timeout: float = 10.0
    sse_read_timeout: float = 300.0
    tool_whitelist: Optional[Sequence[str]] = field(default=None)
    enabled: bool = True
    raise_on_error: bool = False

    @classmethod
    def from_environment(cls, *, server_prefix: str = "DB_MCP") -> "MCPRegistrationConfig":
        env = os.environ
        enabled = _str_to_bool(env.get(f"{server_prefix}_ENABLED"), default=True)

        url_value = env.get(f"{server_prefix}_URL")
        url = str(url_value) if url_value else "http://localhost:8080/mcp"

        tool_whitelist = _normalize_tool_whitelist(env.get(f"{server_prefix}_TOOLS"))

        request_timeout = _coerce_float(
            env.get(f"{server_prefix}_TIMEOUT") or env.get("MCP_TIMEOUT"),
            10.0,
        )
        sse_read_timeout = _coerce_float(
            env.get(f"{server_prefix}_READ_TIMEOUT") or env.get("MCP_READ_TIMEOUT"),
            300.0,
        )

        raise_on_error = _str_to_bool(env.get(f"{server_prefix}_STRICT"), default=False)

        return cls(
            url=url,
            request_timeout=request_timeout,
            sse_read_timeout=sse_read_timeout,
            tool_whitelist=tool_whitelist,
            enabled=enabled,
            raise_on_error=raise_on_error,
        )


def _run_coro_sync(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(coro))
            return future.result()
    return asyncio.run(coro)


@asynccontextmanager
async def _connect_transport(cfg: MCPRegistrationConfig):
    if ClientSession is None:
        raise ImportError(
            "The `mcp` Python package is required to register MCP tools. "
            "Install it with `pip install modelcontextprotocol`."
        )

    if streamablehttp_client is None:
        raise ImportError(
            "The installed `mcp` package does not support Streamable HTTP. "
            "Upgrade with `pip install --upgrade modelcontextprotocol`."
        )
    async with streamablehttp_client(
        cfg.url,
        timeout=cfg.request_timeout,
        sse_read_timeout=cfg.sse_read_timeout,
    ) as streams:
        read_stream, write_stream, _ = streams
        yield read_stream, write_stream


async def _list_mcp_tools(cfg: MCPRegistrationConfig):
    async with _connect_transport(cfg) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)
        async with session:
            await session.initialize()
            response = await session.list_tools()
            tools = list(response.tools or [])

            if cfg.tool_whitelist:
                allowed = {name for name in cfg.tool_whitelist}
                tools = [tool for tool in tools if tool.name in allowed]

            return tools


def _schema_to_json(schema: Any) -> Dict[str, Any]:
    if schema is None:
        return {"type": "object", "properties": {}}
    if hasattr(schema, "model_dump"):
        data = schema.model_dump()  # type: ignore[call-arg]
        if isinstance(data, dict):
            return _ensure_object_schema(data)
    if isinstance(schema, dict):
        return _ensure_object_schema(dict(schema))
    return {"type": "object", "properties": {}}


def _ensure_object_schema(data: Mapping[str, Any]) -> Dict[str, Any]:
    schema = dict(data)
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if "object" in schema_type:
            schema["type"] = "object"
        elif schema_type:
            schema["type"] = schema_type[0]
    if schema.get("type") is None:
        schema["type"] = "object"
    if schema.get("type") == "object" and "properties" not in schema:
        schema["properties"] = {}
    return schema


def _tool_to_spec(tool: mcp_types.Tool) -> Dict[str, Any]:
    description = tool.description or f"MCP tool '{tool.name}' exposed by db-mcp-server."
    schema = _schema_to_json(tool.inputSchema)
    properties = {}
    if isinstance(schema.get("properties"), Mapping):
        properties = dict(schema.get("properties", {}))
    required = schema.get("required") if isinstance(schema, Mapping) else None
    if isinstance(required, list):
        required_fields = [str(item) for item in required]
    else:
        required_fields = []
    return {
        "name": tool.name,
        "description": description,
        "parameters": schema,
        "title": tool.name,
        "properties": properties,
        **({"required": required_fields} if required_fields else {}),
    }


async def _collect_tool_specs(cfg: MCPRegistrationConfig) -> List[Dict[str, Any]]:
    tools = await _list_mcp_tools(cfg)
    if not tools:
        return []
    return [_tool_to_spec(tool) for tool in tools]


async def _call_tool_async(
    cfg: MCPRegistrationConfig,
    tool_name: str,
    arguments: Optional[Mapping[str, Any]] = None,
) -> str:
    async with _connect_transport(cfg) as (read_stream, write_stream):
        session = ClientSession(read_stream, write_stream)
        async with session:
            await session.initialize()
            response = await session.call_tool(tool_name, arguments=arguments or None)
            if response.isError:
                detail = response.error or "Unknown MCP tool error"
                raise RuntimeError(f"MCP tool '{tool_name}' returned an error: {detail}")

            if response.structuredContent:
                payload = [item.model_dump() for item in response.structuredContent]
                return json.dumps(payload, ensure_ascii=False, indent=2)

            texts: List[str] = []
            for item in response.content or []:
                if isinstance(item, mcp_types.TextContent):
                    texts.append(item.text)
                elif hasattr(item, "model_dump"):
                    texts.append(json.dumps(item.model_dump(), ensure_ascii=False, indent=2))
                else:
                    texts.append(str(item))
            return "\n".join(texts).strip()


def _call_tool_sync(
    cfg: MCPRegistrationConfig,
    tool_name: str,
    arguments: Optional[Mapping[str, Any]] = None,
) -> str:
    clean_args: Dict[str, Any] = {}
    if arguments:
        clean_args = {key: value for key, value in arguments.items() if value is not None}
    return _run_coro_sync(_call_tool_async(cfg, tool_name, clean_args or None))


@dataclass(slots=True)
class MCPToolContext:
    config: MCPRegistrationConfig
    specs: List[Dict[str, Any]]
    spec_map: Dict[str, Dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        self.spec_map = {spec["name"]: spec for spec in self.specs}

    def has_tool(self, name: str) -> bool:
        return name in self.spec_map

    def list_tool_names(self) -> List[str]:
        return sorted(self.spec_map)

    def call_tool(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> str:
        if not self.has_tool(name):
            raise KeyError(f"MCP tool '{name}' is not registered.")
        LOGGER.info("Invoking MCP tool %s with args %s", name, arguments or {})
        return _call_tool_sync(self.config, name, arguments)


def load_mcp_tool_specs(cfg: MCPRegistrationConfig | None = None) -> List[Dict[str, Any]]:
    """Return JSON-compatible LangChain tool specs exposed by the MCP server."""

    config = cfg or MCPRegistrationConfig.from_environment()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError(
            "load_mcp_tool_specs cannot be called while an event loop is running. "
            "Use `await async_load_mcp_tool_specs(...)` instead."
        )

    return asyncio.run(_collect_tool_specs(config))


async def async_load_mcp_tool_specs(cfg: MCPRegistrationConfig | None = None) -> List[Dict[str, Any]]:
    config = cfg or MCPRegistrationConfig.from_environment()
    return await _collect_tool_specs(config)


def register_db_mcp_tools(
    llm: Any,
    *,
    cfg: MCPRegistrationConfig | None = None,
    debug: bool = False,
) -> Tuple[Any, MCPToolContext | None]:
    """Bind db-mcp-server tools to the supplied chat model if available."""

    _ensure_logger_configured()

    config = cfg or MCPRegistrationConfig.from_environment()
    debug_enabled = debug or _str_to_bool(os.getenv("DB_MCP_DEBUG"), default=False)

    if not hasattr(llm, "bind_tools"):
        if debug_enabled:
            warnings.warn(
                f"Chat model {type(llm).__name__} does not support tool binding; skipping MCP registration.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            LOGGER.info(
                "Chat model %s does not support tool binding; skipping MCP registration.",
                type(llm).__name__,
            )
        return llm, None

    if not config.enabled:
        message = "DB_MCP_ENABLED resolved to false; skipping MCP tool registration."
        if debug_enabled:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        else:
            LOGGER.info(message)
        return llm, None

    LOGGER.info(
        "Attempting MCP registration via streamable_http (%s)",
        config.url,
    )

    try:
        specs = load_mcp_tool_specs(config)
    except Exception as exc:  # noqa: BLE001 - surface as configuration feedback
        if config.raise_on_error:
            raise
        message = f"Unable to register MCP tools: {exc}"
        if debug_enabled:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        else:
            LOGGER.warning(message)
        return llm, None

    if not specs:
        message = "db-mcp-server did not expose any tools; continuing without tool binding."
        if debug_enabled:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        else:
            LOGGER.warning(message)
        return llm, None

    tools_for_binding = [_spec_to_structured_tool(spec) for spec in specs]

    bound = llm.bind_tools(tools_for_binding)
    context = MCPToolContext(config=config, specs=specs)
    setattr(bound, "_mcp_context", context)

    tool_names = ", ".join(sorted(tool.get("name", "<unnamed>") for tool in specs))
    location_hint = f" (streamable HTTP {config.url})"

    LOGGER.info(
        "Registered %d MCP tool(s) from db-mcp-server via streamable_http%s: %s",
        len(specs),
        location_hint,
        tool_names,
    )

    if hasattr(bound, "with_config"):
        bound = bound.with_config(
            {
                "metadata": {
                    "mcp_server": "db-mcp-server",
                    "mcp_transport": "streamable_http",
                    "mcp_tool_specs": specs,
                }
            }
        )

    # Preserve MCP metadata for downstream inspection if supported.
    try:
        setattr(bound, "_mcp_tool_specs", tuple(specs))
    except Exception:
        pass

    try:
        setattr(bound, "_mcp_context", context)
    except Exception:
        pass

    return bound, context


__all__ = [
    "MCPRegistrationConfig",
    "async_load_mcp_tool_specs",
    "load_mcp_tool_specs",
    "register_db_mcp_tools",
    "MCPToolContext",
    "build_structured_tools_from_specs",
]
