#!/usr/bin/env python3
from __future__ import annotations
"""
Enhanced transcript formatter: turn raw transcript .txt into clean **Markdown** using LangChain + OpenAI.
Loads environment variables (including OPENAI_API_KEY) using `python-dotenv`.

Features:
- Smart section header generation based on topic shifts
- Speaker label normalization  
- Filler word removal while preserving technical terms
- Comprehensive error handling and validation
- File statistics and processing metrics
- Overwrite protection with --force flag
- Configurable model and temperature settings

Usage:
  pip install langchain-openai python-dotenv
  # .env file in the working dir (or pass --env path)
  #   OPENAI_API_KEY=sk-...
  python markup_transcript.py input.txt output.md --title "My Talk"
  
  # With verbose output and custom settings:
  python markup_transcript.py input.txt output.md --title "Interview" --verbose --temperature 0.3
  
  # Specify custom env file:
  python markup_transcript.py input.txt output.md --env ./secrets/.env
"""

import os
import sys
import argparse
from typing import Tuple
import time

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
        "Rules:\n"
        "- Start with a single H1 title: {title}.\n"
        "- Add a few simple H2 section headers inferred from topic shifts or speaker changes.\n"
        "- Break into paragraphs every 2-5 sentences; add line breaks where it improves readability.\n"
        "- Normalize obvious speaker labels (e.g., **Interviewer:**, **Guest:**) if present.\n"
        "- Remove filler words (uh, um) and repeated stutters; keep technical terms and numbers accurate.\n"
        "- Do NOT invent content.\n"
        "- Return Markdown only (no preface or explanations).",
    ),
    ("human", "Transcript; {transcript}")
])


def get_file_stats(file_path: str) -> Tuple[int, int, int]:
    """Get basic stats about a text file: (chars, words, lines)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        chars = len(content)
        words = len(content.split())
        lines = len(content.splitlines())
    return chars, words, lines


def validate_model_name(model: str) -> bool:
    """Basic validation for OpenAI model names."""
    valid_models = [
        "gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-4-turbo", 
        "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
    ]
    return model in valid_models


def transcript_txt_to_markdown(
    input_txt_path: str,
    output_md_path: str,
    *,
    title: str = "Transcript",
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    """One-shot: read text -> ask LLM to format -> write Markdown. Returns MD path."""
    # Ensure the key is present in environment (dotenv should have loaded it)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Create a .env with OPENAI_API_KEY=... or export it, or pass --env."
        )

    if not os.path.exists(input_txt_path):
        raise FileNotFoundError(f"Input file not found: {input_txt_path}")
    
    with open(input_txt_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        raise ValueError("Input transcript is empty.")

    # ChatOpenAI will read OPENAI_API_KEY from environment; no need to pass it explicitly
    llm = ChatOpenAI(model=model, temperature=temperature)
    md_text: str = (PROMPT | llm | StrOutputParser()).invoke({"title": title, "transcript": raw})

    # Ensure output directory exists
    output_dir = os.path.dirname(output_md_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(md_text.strip() + "\n")

    return os.path.abspath(output_md_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Quick transcript -> Markdown")
    p.add_argument("input", help="Path to transcript .txt")
    p.add_argument("output_md", help="Path to output .md")
    p.add_argument("--title", default="Transcript", help="Document title (H1)")
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model")
    p.add_argument("--env", dest="env_file", help="Optional path to .env file")
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    p.add_argument("--force", "-f", action="store_true", help="Overwrite output file if it exists")
    p.add_argument("--temperature", type=float, default=0.2, help="LLM temperature (0.0-1.0)")
    args = p.parse_args()

    # If a custom .env path is provided, load it now (won't override pre-existing env vars)
    if args.env_file:
        if not os.path.exists(args.env_file):
            print(f"Warning: .env file not found: {args.env_file}", file=sys.stderr)
        else:
            load_dotenv(args.env_file, override=False)

    # Validate model name
    if not validate_model_name(args.model):
        print(f"Warning: Unknown model '{args.model}' - proceeding anyway", file=sys.stderr)

    # Check if output file exists
    if os.path.exists(args.output_md) and not args.force:
        print(f"Error: Output file '{args.output_md}' already exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    try:
        start_time = time.time()
        
        # Get input stats early so they're available later
        input_chars, input_words, input_lines = get_file_stats(args.input) if args.verbose else (0, 0, 0)
        
        if args.verbose:
            print(f"Processing: {args.input}")
            print(f"  Input stats: {input_chars:,} chars, {input_words:,} words, {input_lines:,} lines")
            print(f"Output: {args.output_md}")
            print(f"Title: {args.title}")
            print(f"Model: {args.model} (temp={args.temperature})")
            print("Processing with OpenAI...")
        
        md_path = transcript_txt_to_markdown(
            args.input,
            args.output_md,
            title=args.title,
            model=args.model,
            temperature=args.temperature,
        )
        
        elapsed = time.time() - start_time
        print(f"✓ Markdown created: {md_path}")
        
        if args.verbose:
            output_chars, output_words, output_lines = get_file_stats(md_path)
            print(f"  Output stats: {output_chars:,} chars, {output_words:,} words, {output_lines:,} lines")
            print(f"  Processing time: {elapsed:.1f}s")
            
            # Show compression ratio
            if input_chars > 0:
                ratio = (output_chars / input_chars) * 100
                print(f"  Size ratio: {ratio:.1f}% of original")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
