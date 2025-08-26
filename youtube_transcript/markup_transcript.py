#!/usr/bin/env python3
from __future__ import annotations
"""
Minimal: turn a raw transcript .txt into clean **Markdown** using LangChain + OpenAI 4o-mini.
Loads environment variables (including OPENAI_API_KEY) using `python-dotenv`.

Usage:
  pip install langchain-openai python-dotenv
  # .env file in the working dir (or pass --env path)
  #   OPENAI_API_KEY=sk-...
  python simple_transcript.py input.txt output.md --title "My Talk"
  # or specify a custom env file:
  python simple_transcript.py input.txt output.md --env ./secrets/.env
"""

import os
import argparse
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load variables from default .env at import time (non-destructive)
load_dotenv(override=False)

PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a concise editorial assistant. Convert the user's raw transcript into CLEAN, EASY-TO-READ Markdown."
        "Rules:"
        "- Start with a single H1 title: {title}."
        "- Add a few simple H2 section headers inferred from topic shifts or speaker changes."
        "- Break into paragraphs every 2-5 sentences; add line breaks where it improves readability."
        "- Normalize obvious speaker labels (e.g., **Interviewer:**, **Guest:**) if present."
        "- Remove filler words (uh, um) and repeated stutters; keep technical terms and numbers accurate."
        "- Do NOT invent content."
        "- Return Markdown only (no preface or explanations).",
    ),
    ("human", "Transcript; {transcript}")
])


def transcript_txt_to_markdown(
    input_txt_path: str,
    output_md_path: str,
    *,
    title: str = "Transcript",
    model: str = "gpt-4o-mini",
) -> str:
    """One-shot: read text -> ask LLM to format -> write Markdown. Returns MD path."""
    # Ensure the key is present in environment (dotenv should have loaded it)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env with OPENAI_API_KEY=... or export it, or pass --env."
        )

    with open(input_txt_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        raise ValueError("Input transcript is empty.")

    # ChatOpenAI will read OPENAI_API_KEY from environment; no need to pass it explicitly
    llm = ChatOpenAI(model=model, temperature=0.2)
    md_text: str = (PROMPT | llm | StrOutputParser()).invoke({"title": title, "transcript": raw})

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_text.strip() + "")

    return os.path.abspath(output_md_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Quick transcript -> Markdown")
    p.add_argument("input", help="Path to transcript .txt")
    p.add_argument("output_md", help="Path to output .md")
    p.add_argument("--title", default="Transcript", help="Document title (H1)")
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model")
    p.add_argument("--env", dest="env_file", help="Optional path to .env")
    args = p.parse_args()

    # If a custom .env path is provided, load it now (won't override pre-existing env vars)
    if args.env_file:
        load_dotenv(args.env_file, override=False)

    md_path = transcript_txt_to_markdown(
        args.input,
        args.output_md,
        title=args.title,
        model=args.model,
    )
    print("Markdown:", md_path)
