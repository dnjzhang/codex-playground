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
import shlex
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
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:  # pragma: no cover - optional dependency
    mcp_types = None  # type: ignore[assignment]
    ClientSession = None  # type: ignore[assignment]
    sse_client = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
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


def _parse_headers(value: str) -> Mapping[str, str]:
    """Parse header overrides from either JSON or comma-separated ``key=value`` pairs."""

    if not value:
        return {}
    text = value.strip()
    if not text:
        return {}

    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return {str(key): str(val) for key, val in parsed.items()}
        except json.JSONDecodeError:
            pass

    headers: Dict[str, str] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" in part:
            key, raw_value = part.split("=", 1)
        elif ":" in part:
            key, raw_value = part.split(":", 1)
        else:
            continue
        headers[key.strip()] = raw_value.strip()
    return headers


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


def _normalize_command(value: Any) -> Optional[Sequence[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, Sequence):
        normalized = []
        for item in value:
            if item is None:
                continue
            normalized.append(str(item))
        return tuple(normalized)
    return None


def _normalize_headers(value: Any) -> Optional[Mapping[str, str]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): str(val) for key, val in value.items()}
    if isinstance(value, str):
        result = _parse_headers(value)
        return result or None
    return None


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


def _normalize_env(value: Any) -> Optional[Mapping[str, str]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(key): str(val) for key, val in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:  # pragma: no cover - configuration error
            raise ValueError(
                "Environment overrides must be provided as JSON when using string values."
            ) from exc
        if isinstance(parsed, Mapping):
            return {str(key): str(val) for key, val in parsed.items()}
    return None


def _load_config_file(path: str) -> Mapping[str, Any]:
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        raise FileNotFoundError(f"MCP config file not found: {path}")

    with open(expanded, "r", encoding="utf-8") as handle:
        text = handle.read()

    lower_path = expanded.lower()
    if lower_path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PyYAML is required to parse YAML MCP configs. Install it with `pip install pyyaml` "
                "or provide a JSON config file."
            ) from exc
        data = yaml.safe_load(text)  # type: ignore[arg-type]
    else:
        data = json.loads(text)

    if not isinstance(data, Mapping):
        raise ValueError(f"MCP config file {path} must contain a JSON/YAML object.")

    return data


def _select_config_section(data: Mapping[str, Any], server_prefix: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        return {}

    candidates = [
        server_prefix,
        server_prefix.lower(),
        server_prefix.upper(),
        server_prefix.lower().replace("_", ""),
        server_prefix.lower().replace("_", "-"),
    ]
    containers = [data]
    for container_key in ("servers", "mcp_servers", "mcp"):
        container = data.get(container_key)
        if isinstance(container, Mapping):
            containers.append(container)

    for container in containers:
        if not isinstance(container, Mapping):
            continue
        for key in candidates:
            value = container.get(key)
            if isinstance(value, Mapping):
                return value

    return data


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

    transport: str = "stdio"
    command: Optional[Sequence[str]] = field(default=None)
    env: Optional[Mapping[str, str]] = field(default=None)
    cwd: Optional[str] = field(default=None)
    sse_url: Optional[str] = field(default=None)
    http_url: Optional[str] = field(default=None)
    headers: Optional[Mapping[str, str]] = field(default=None)
    request_timeout: float = 10.0
    sse_read_timeout: float = 300.0
    tool_whitelist: Optional[Sequence[str]] = field(default=None)
    enabled: bool = True
    raise_on_error: bool = False

    @classmethod
    def from_environment(cls, *, server_prefix: str = "DB_MCP") -> "MCPRegistrationConfig":
        env = os.environ
        config_path = env.get(f"{server_prefix}_CONFIG") or env.get("MCP_CONFIG_FILE")
        config_section: Mapping[str, Any] = {}
        if config_path:
            config_data = _load_config_file(config_path)
            config_section = _select_config_section(config_data, server_prefix)

        enabled_default = _coerce_bool(config_section.get("enabled"), True)
        enabled = _str_to_bool(env.get(f"{server_prefix}_ENABLED"), default=enabled_default)

        command_env_value = env.get(f"{server_prefix}_COMMAND") or env.get("MCP_COMMAND")
        command_config_value = (
            config_section.get("command")
            or config_section.get("stdio_command")
        )
        command = _normalize_command(command_env_value or command_config_value)

        cwd_value = env.get(f"{server_prefix}_CWD") or config_section.get("cwd") or config_section.get("working_dir")
        cwd = os.path.expanduser(cwd_value) if isinstance(cwd_value, str) else None

        http_url_value = (
            env.get(f"{server_prefix}_HTTP_URL")
            or env.get(f"{server_prefix}_STREAMABLE_HTTP_URL")
            or env.get("MCP_HTTP_URL")
            or env.get("MCP_STREAMABLE_HTTP_URL")
            or config_section.get("http_url")
            or config_section.get("streamable_http_url")
            or config_section.get("streamable_url")
        )
        http_url = str(http_url_value) if http_url_value else None

        sse_url_value = (
            env.get(f"{server_prefix}_SSE_URL")
            or env.get("MCP_SSE_URL")
            or config_section.get("sse_url")
            or config_section.get("url")
        )
        sse_url = str(sse_url_value) if sse_url_value else None

        headers_env_value = env.get(f"{server_prefix}_SSE_HEADERS") or env.get("MCP_SSE_HEADERS")
        headers = _normalize_headers(headers_env_value) if headers_env_value else None
        if headers is None:
            headers = _normalize_headers(
                config_section.get("headers")
                or config_section.get("sse_headers")
            )

        whitelist_env_value = env.get(f"{server_prefix}_TOOLS")
        tool_whitelist = _normalize_tool_whitelist(whitelist_env_value)
        if tool_whitelist is None:
            tool_whitelist = _normalize_tool_whitelist(
                config_section.get("tools")
                or config_section.get("tool_whitelist")
            )

        transport_env_value = env.get(f"{server_prefix}_TRANSPORT") or env.get("MCP_TRANSPORT")
        transport_config_value = config_section.get("transport") or config_section.get("connection")
        if transport_env_value:
            transport = transport_env_value.strip().lower()
        elif transport_config_value:
            transport = str(transport_config_value).strip().lower()
        elif http_url:
            transport = "streamable_http"
        elif sse_url:
            transport = "sse"
        elif command:
            transport = "stdio"
        else:
            transport = "stdio"

        transport = transport.replace("-", "_")
        if transport in {"http", "streamablehttp"}:
            transport = "streamable_http"

        if transport not in {"stdio", "sse", "streamable_http"}:
            transport = "stdio"

        request_default = _coerce_float(
            config_section.get("timeout")
            or config_section.get("request_timeout"),
            10.0,
        )
        request_timeout = _coerce_float(
            env.get(f"{server_prefix}_TIMEOUT") or env.get("MCP_TIMEOUT"),
            request_default,
        )

        sse_read_default = _coerce_float(
            config_section.get("sse_read_timeout")
            or config_section.get("read_timeout"),
            300.0,
        )
        sse_read_timeout = _coerce_float(
            env.get(f"{server_prefix}_SSE_READ_TIMEOUT") or env.get("MCP_SSE_READ_TIMEOUT"),
            sse_read_default,
        )

        raise_default = _coerce_bool(
            config_section.get("raise_on_error")
            or config_section.get("strict"),
            False,
        )
        raise_on_error = _str_to_bool(env.get(f"{server_prefix}_STRICT"), default=raise_default)

        env_overrides = _normalize_env(config_section.get("env"))

        return cls(
            transport=transport,
            command=command,
            env=env_overrides,
            cwd=cwd,
            sse_url=sse_url,
            http_url=http_url,
            headers=headers,
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

    if cfg.transport == "streamable_http":
        if streamablehttp_client is None:
            raise ImportError(
                "The installed `mcp` package does not support Streamable HTTP. "
                "Upgrade with `pip install --upgrade modelcontextprotocol`."
            )
        url = cfg.http_url or cfg.sse_url
        if not url:
            raise ValueError(
                "MCP Streamable HTTP transport requires a URL. "
                "Set DB_MCP_HTTP_URL (or MCP_HTTP_URL)."
            )

        headers = dict(cfg.headers) if cfg.headers is not None else None
        async with streamablehttp_client(
            url,
            headers=headers,
            timeout=cfg.request_timeout,
            sse_read_timeout=cfg.sse_read_timeout,
        ) as streams:
            read_stream, write_stream, _ = streams
            yield read_stream, write_stream
            return

    if cfg.transport == "sse":
        if sse_client is None:
            raise ImportError(
                "The installed `mcp` package does not support SSE transport. "
                "Upgrade with `pip install --upgrade modelcontextprotocol`."
            )
        if not cfg.sse_url:
            raise ValueError("MCP SSE transport requires a URL. Set DB_MCP_SSE_URL or MCP_SSE_URL.")

        headers = dict(cfg.headers) if cfg.headers is not None else None
        async with sse_client(
            cfg.sse_url,
            headers=headers,
            timeout=cfg.request_timeout,
            sse_read_timeout=cfg.sse_read_timeout,
        ) as streams:
            yield streams
            return

    if stdio_client is None or StdioServerParameters is None:
        raise ImportError(
            "The installed `mcp` package does not support stdio transport. "
            "Upgrade with `pip install --upgrade modelcontextprotocol`."
        )
    if not cfg.command:
        raise ValueError(
            "MCP stdio transport requires a launch command. "
            "Set DB_MCP_COMMAND (or MCP_COMMAND) to the executable for your db-mcp-server."
        )
    command = list(cfg.command)

    server = StdioServerParameters(
        command=command[0],
        args=command[1:],
        env=dict(cfg.env) if cfg.env is not None else None,
        cwd=cfg.cwd,
    )
    async with stdio_client(server) as streams:
        yield streams


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
        "Attempting MCP registration via %s transport%s",
        config.transport,
        (
            f" ({config.http_url or config.sse_url})"
            if config.transport in {"sse", "streamable_http"} and (config.http_url or config.sse_url)
            else ""
        ),
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
    location_hint = ""
    if config.transport == "streamable_http" and (config.http_url or config.sse_url):
        location_hint = f" (streamable HTTP {config.http_url or config.sse_url})"
    elif config.transport == "sse" and config.sse_url:
        location_hint = f" (SSE {config.sse_url})"
    elif config.transport == "stdio" and config.command:
        location_hint = f" (stdio command: {' '.join(config.command)})"

    LOGGER.info(
        "Registered %d MCP tool(s) from db-mcp-server via %s transport%s: %s",
        len(specs),
        config.transport,
        location_hint,
        tool_names,
    )

    if hasattr(bound, "with_config"):
        bound = bound.with_config(
            {
                "metadata": {
                    "mcp_server": "db-mcp-server",
                    "mcp_transport": config.transport,
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
]
