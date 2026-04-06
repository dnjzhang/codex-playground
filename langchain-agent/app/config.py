"""Application configuration for the langchain-agent project."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass(slots=True)
class MCPServerConfig:
    """Definition for one MCP server connection."""

    transport: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None
    enabled: bool = True
    headers: dict[str, Any] | None = None

    def as_connection_dict(self) -> dict[str, Any]:
        """Convert config to a dict accepted by langchain-mcp-adapters."""
        connection: dict[str, Any] = {"transport": self.transport}
        if self.command is not None:
            connection["command"] = self.command
        if self.args:
            connection["args"] = list(self.args)
        if self.url is not None:
            connection["url"] = self.url
        if self.env:
            connection["env"] = dict(self.env)
        if self.cwd is not None:
            connection["cwd"] = self.cwd
        if self.headers:
            connection["headers"] = dict(self.headers)
        return connection


@dataclass(slots=True)
class AppSettings:
    """Resolved runtime settings for the local chatbot."""

    project_root: Path
    ollama_base_url: str
    default_model: str
    mcp_config_path: Path
    skills_sources: list[str]
    memory_sources: list[str]

    @classmethod
    def load(cls, project_root: Path | None = None) -> "AppSettings":
        """Load settings from `.env` and environment variables."""
        root = project_root or Path(__file__).resolve().parent.parent
        load_dotenv(root / ".env")
        mcp_config = os.environ.get("LANGCHAIN_AGENT_MCP_CONFIG", "app/mcp_servers.json")
        return cls(
            project_root=root,
            ollama_base_url=os.environ.get(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            ),
            default_model=os.environ.get(
                "LANGCHAIN_AGENT_DEFAULT_MODEL",
                "gemma4:e4b",
            ),
            mcp_config_path=(root / mcp_config).resolve(),
            skills_sources=["/app/skills/"],
            memory_sources=["/AGENTS.md"],
        )


def load_mcp_servers(path: Path) -> dict[str, MCPServerConfig]:
    """Load MCP server definitions from a JSON config file."""
    raw = json.loads(path.read_text())
    servers = raw.get("servers", {})
    result: dict[str, MCPServerConfig] = {}
    for name, config in servers.items():
        result[name] = MCPServerConfig(
            transport=str(config["transport"]),
            command=config.get("command"),
            args=list(config.get("args", [])),
            url=config.get("url"),
            env=config.get("env"),
            cwd=config.get("cwd"),
            enabled=bool(config.get("enabled", True)),
            headers=config.get("headers"),
        )
    return result
