from pathlib import Path

from app.config import AppSettings, MCPServerConfig, load_mcp_servers
from app.runtime import AgentRuntime
from app.session import SessionStore


def test_settings_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("LANGCHAIN_AGENT_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("LANGCHAIN_AGENT_MCP_CONFIG", raising=False)

    settings = AppSettings.load(tmp_path)

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.default_model == "gemma4:e4b"
    assert settings.mcp_config_path == (tmp_path / "app/mcp_servers.json").resolve()


def test_load_mcp_servers(tmp_path: Path) -> None:
    config_path = tmp_path / "mcp_servers.json"
    config_path.write_text(
        """
{
  "servers": {
    "playwright": {
      "enabled": true,
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest", "--browser=chrome"]
    }
  }
}
        """.strip()
    )

    servers = load_mcp_servers(config_path)

    assert "playwright" in servers
    assert servers["playwright"].transport == "stdio"
    assert servers["playwright"].command == "npx"


def test_session_store_generates_server_ids() -> None:
    store = SessionStore()
    session = store.create("gemma4:e4b")

    assert session.session_id.startswith("session-")
    assert store.get(session.session_id).model_name == "gemma4:e4b"


def test_runtime_routes_simple_chat_without_deep_agent(tmp_path: Path) -> None:
    settings = AppSettings.load(tmp_path)
    runtime = AgentRuntime(settings)
    runtime._server_configs = {
        "playwright": MCPServerConfig(
            transport="stdio",
            command="npx",
            args=["-y", "@playwright/mcp@latest", "--browser=chrome"],
            enabled=True,
        )
    }

    assert not runtime.should_use_deep_agent("what is the tallest mountain in US?")
    assert not runtime.should_use_deep_agent(
        "How far away from Paris to London? Use straight line measurement."
    )
    assert runtime.should_use_deep_agent("use playwright to open example.com")
