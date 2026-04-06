"""Prompt loading helpers for the local chatbot."""

from __future__ import annotations

from pathlib import Path


def load_system_prompt(project_root: Path) -> str:
    """Load the main system prompt from a project-local prompt file."""
    prompt_path = project_root / "app" / "prompts" / "system_prompt.md"
    return prompt_path.read_text()
