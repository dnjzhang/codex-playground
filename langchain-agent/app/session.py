"""Session state for the local chatbot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4


@dataclass(slots=True)
class ChatSession:
    """Represents one chat session."""

    session_id: str
    model_name: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class SessionStore:
    """In-memory session registry with server-generated IDs."""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    def create(self, model_name: str) -> ChatSession:
        session = ChatSession(
            session_id=f"session-{uuid4().hex[:12]}",
            model_name=model_name,
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> ChatSession:
        return self._sessions[session_id]
