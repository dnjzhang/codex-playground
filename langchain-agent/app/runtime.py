"""Reusable runtime for CLI and future API layers."""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.sessions import create_session

from app.config import AppSettings, MCPServerConfig, load_mcp_servers
from app.mcp import load_mcp_tools
from app.prompts import load_system_prompt
from app.session import ChatSession, SessionStore

LOGGER = logging.getLogger(__name__)

TOOL_TRIGGER_PATTERNS = (
    re.compile(r"\b(open|browse|navigate|search the web|website|web page|screenshot)\b", re.I),
    re.compile(r"\b(playwright|browser|tool|mcp)\b", re.I),
    re.compile(r"\b(click|type|fill|form|automation)\b", re.I),
    re.compile(r"\buse\s+(playwright|browser|tool|mcp)\b", re.I),
)

BROWSER_SKILL_PATTERNS = (
    re.compile(r"\b(playwright|browser|website|web page|page|navigate|screenshot)\b", re.I),
)


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts)
    return str(content)


def _tool_hint_block(tools: list[Any]) -> str:
    """Build a runtime hint listing the exact registered tool names."""
    if not tools:
        return (
            "\n\n## Runtime tool hint\n\n"
            "No external tools are currently registered in this session. "
            "Do not invent tools.\n"
        )

    tool_names = sorted(
        str(getattr(tool, "name", "")).strip()
        for tool in tools
        if str(getattr(tool, "name", "")).strip()
    )
    tool_lines = "\n".join(f"- `{name}`" for name in tool_names)
    return (
        "\n\n## Runtime tool hint\n\n"
        "The following are the exact registered tool names available in this session. "
        "Use only these names when calling tools. If a needed tool is not listed here, "
        "it is not available.\n\n"
        f"{tool_lines}\n"
    )


def _matched_skill_names(user_text: str) -> list[str]:
    """Infer which project-local skills are relevant to the request."""
    matches: list[str] = []
    if any(pattern.search(user_text) for pattern in BROWSER_SKILL_PATTERNS):
        matches.append("playwright-browser")
    return matches


@dataclass(slots=True)
class SessionResources:
    """Per-chat-session runtime resources."""

    tools: list[Any]
    agent: Any | None = None
    checkpointer: InMemorySaver | None = None
    exit_stack: AsyncExitStack | None = None


class AgentRuntime:
    """Async service that owns tool loading, agents, and session execution."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.sessions = SessionStore()
        self._chat_model_cache: dict[str, ChatOllama] = {}
        self._server_configs: dict[str, MCPServerConfig] = {}
        self._session_resources: dict[str, SessionResources] = {}

    async def initialize(self) -> None:
        """Load MCP tools once at startup."""
        self._server_configs = load_mcp_servers(self.settings.mcp_config_path)

    def _build_chat_model(self, model_name: str) -> ChatOllama:
        return ChatOllama(
            model=model_name,
            base_url=self.settings.ollama_base_url,
            temperature=0,
        )

    def _get_chat_model(self, model_name: str) -> ChatOllama:
        model = self._chat_model_cache.get(model_name)
        if model is None:
            model = self._build_chat_model(model_name)
            self._chat_model_cache[model_name] = model
        return model

    async def create_session(self, model_name: str | None = None) -> ChatSession:
        """Create a new chat session."""
        resolved_model = model_name or self.settings.default_model
        session = self.sessions.create(resolved_model)
        self._session_resources[session.session_id] = await self._create_session_resources()
        return session

    async def _create_session_resources(self) -> SessionResources:
        """Create persistent MCP sessions and tools for one chat session."""
        enabled_configs = {
            name: config
            for name, config in self._server_configs.items()
            if config.enabled
        }
        if not enabled_configs:
            return SessionResources(tools=[])

        exit_stack = AsyncExitStack()
        tools: list[Any] = []
        try:
            for server_name, server_config in enabled_configs.items():
                client_session = await exit_stack.enter_async_context(
                    create_session(server_config.as_connection_dict())
                )
                await client_session.initialize()
                tools.extend(
                    await load_mcp_tools(
                        {server_name: server_config},
                        existing_session=client_session,
                        prefix_tool_names=True,
                    )
                )
        except Exception:
            await exit_stack.aclose()
            raise
        return SessionResources(tools=tools, exit_stack=exit_stack)

    def _build_agent(self, session_id: str, model_name: str) -> Any:
        model = self._get_chat_model(model_name)
        backend = FilesystemBackend(
            root_dir=self.settings.project_root,
            virtual_mode=True,
        )
        resources = self._session_resources[session_id]
        system_prompt = load_system_prompt(self.settings.project_root) + _tool_hint_block(
            resources.tools
        )
        checkpointer = InMemorySaver()
        agent = create_deep_agent(
            model=model,
            tools=resources.tools,
            system_prompt=system_prompt,
            skills=self.settings.skills_sources,
            memory=self.settings.memory_sources,
            backend=backend,
            checkpointer=checkpointer,
            name=f"langchain-agent-{model_name}",
        )
        resources.agent = agent
        resources.checkpointer = checkpointer
        return agent

    def _get_agent(self, session_id: str, model_name: str) -> Any:
        resources = self._session_resources[session_id]
        if resources.agent is None:
            resources.agent = self._build_agent(session_id, model_name)
        return resources.agent

    @staticmethod
    def _config_for(session: ChatSession) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": session.session_id,
            }
        }

    def should_use_deep_agent(self, user_text: str) -> bool:
        """Route tool-oriented prompts through DeepAgent and keep simple chat direct."""
        if not any(config.enabled for config in self._server_configs.values()):
            return False
        return any(pattern.search(user_text) for pattern in TOOL_TRIGGER_PATTERNS)

    async def chat_once(self, session_id: str, user_text: str) -> str:
        """Run one non-streaming turn."""
        session = self.sessions.get(session_id)
        try:
            if not self.should_use_deep_agent(user_text):
                response = await self._get_chat_model(session.model_name).ainvoke(user_text)
                return _message_text(response)
            agent = self._get_agent(session_id, session.model_name)
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=user_text)]},
                config=self._config_for(session),
            )
            messages = result.get("messages", [])
            for message in reversed(messages):
                if isinstance(message, AIMessage):
                    return _message_text(message)
            return ""
        except Exception:
            LOGGER.exception("Chat failed for session %s", session_id)
            raise

    async def stream_chat(self, session_id: str, user_text: str) -> AsyncIterator[str]:
        """Stream assistant text for one turn."""
        session = self.sessions.get(session_id)
        try:
            if not self.should_use_deep_agent(user_text):
                async for chunk in self._get_chat_model(session.model_name).astream(user_text):
                    if isinstance(chunk, AIMessageChunk):
                        text = _message_text(chunk)
                        if text:
                            yield text
                return
            agent = self._get_agent(session_id, session.model_name)
            seen_chunks: set[str] = set()

            async for event in agent.astream(
                {"messages": [HumanMessage(content=user_text)]},
                config=self._config_for(session),
                stream_mode="messages",
            ):
                chunk = event[0] if isinstance(event, tuple) else event
                if not isinstance(chunk, AIMessageChunk):
                    continue
                text = _message_text(chunk)
                if not text:
                    continue
                chunk_id = f"{chunk.id}:{text}"
                if chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk_id)
                yield text
        except Exception:
            LOGGER.exception("Streaming chat failed for session %s", session_id)
            raise

    async def stream_deep_agent_updates(
        self,
        session_id: str,
        user_text: str,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yield DeepAgent status updates followed by the final response text."""
        session = self.sessions.get(session_id)
        agent = self._get_agent(session_id, session.model_name)
        emitted_statuses: set[str] = set()

        try:
            yield ("status", "using DeepAgent")
            for skill_name in _matched_skill_names(user_text):
                yield ("status", f"skill {skill_name}")

            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=user_text)]},
                config=self._config_for(session),
                version="v2",
            ):
                event_type = event.get("event")
                name = str(event.get("name", "")).strip()
                if event_type == "on_tool_start" and name:
                    status = f"tool {name}"
                    if status not in emitted_statuses:
                        emitted_statuses.add(status)
                        yield ("status", status)
                elif event_type == "on_tool_error" and name:
                    yield ("status", f"tool error {name}")

            final_text = await self.final_response_text(session_id)
            if final_text:
                yield ("text", final_text)
        except Exception:
            LOGGER.exception("DeepAgent execution failed for session %s", session_id)
            raise

    async def final_response_text(self, session_id: str) -> str:
        """Read the last assistant message persisted for a session."""
        session = self.sessions.get(session_id)
        agent = self._get_agent(session_id, session.model_name)
        state = await agent.aget_state(self._config_for(session))
        values = getattr(state, "values", {})
        messages = values.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return _message_text(message)
        return ""

    async def close_session(self, session_id: str) -> None:
        """Close persistent MCP resources for one chat session."""
        resources = self._session_resources.pop(session_id, None)
        if resources and resources.exit_stack is not None:
            await resources.exit_stack.aclose()

    async def aclose(self) -> None:
        """Close runtime resources."""
        for session_id in list(self._session_resources.keys()):
            await self.close_session(session_id)


def project_root() -> Path:
    """Return the current project root."""
    return Path(__file__).resolve().parent.parent
