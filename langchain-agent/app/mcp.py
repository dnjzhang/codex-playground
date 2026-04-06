"""MCP tool registration helpers."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools as load_tools_from_session
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import MCPServerConfig

LOGGER = logging.getLogger(__name__)


async def load_mcp_tools(
    server_configs: dict[str, MCPServerConfig],
    *,
    existing_session: Any | None = None,
    prefix_tool_names: bool = True,
) -> list[BaseTool]:
    """Load tools from enabled MCP servers."""
    if existing_session is not None:
        if len(server_configs) != 1:
            raise ValueError("existing_session mode expects exactly one server config")
        server_name, server_config = next(iter(server_configs.items()))
        try:
            tools = await load_tools_from_session(
                existing_session,
                connection=server_config.as_connection_dict(),
                server_name=server_name,
                tool_name_prefix=prefix_tool_names,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Persistent MCP tool loading failed: %s", exc)
            return []
        LOGGER.info("Loaded %s MCP tools from persistent server %s", len(tools), server_name)
        return tools

    connections: dict[str, dict[str, Any]] = {}
    for server_name, server_config in server_configs.items():
        if not server_config.enabled:
            continue
        connections[server_name] = server_config.as_connection_dict()

    if not connections:
        LOGGER.info("No enabled MCP servers configured")
        return []

    client = MultiServerMCPClient(
        connections=connections,
        tool_name_prefix=prefix_tool_names,
    )
    try:
        tools = await client.get_tools()
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("MCP tool loading failed: %s", exc)
        return []
    LOGGER.info("Loaded %s MCP tools from %s server(s)", len(tools), len(connections))
    return tools
