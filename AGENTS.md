# Repository Guidelines

## Project Structure & Module Organization
Treat each top-level folder as an independent project. `financial/` contains CSV-driven analytics and plotting utilities, `treasury/` holds Treasury yield scripts, `youtube_transcript/` manages transcript tooling, `mcp/` and `memgpt/` explore agent prototypes, `langmem_sample/` experiments with long-term memory patterns, and `ollama-starter/` ships LLM demos with its own `requirements.txt`. Keep modules, sample data, and README notes inside their respective directory; avoid cross-project imports unless absolutely required. Co-locate new tests beside the code they cover or inside a local `tests/` subfolder.

## Build, Test, and Development Commands
- `python -m venv .venv` — create a Python 3.11+ virtual environment for local work.
- `source .venv/bin/activate` — activate the environment before installing packages.
- `pip install -r <project>/requirements.txt` — install dependencies per project (e.g., `ollama-starter/requirements.txt`).
- `python financial/cd_maturity_plot.py` — generate current CD maturity plots.
- `python treasury/estimate_yields.py` — refresh Treasury yield estimates with the latest assumptions.
- `python youtube_transcript/youtube_transcript.py` — fetch and process YouTube transcripts.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and an 88–100 character soft limit. Use `snake_case` for files, modules, and functions, reserving `CamelCase` for classes. Prefer descriptive filenames like `letta_memory_inspector.py`. Run `black .` and `ruff .` before sharing code to enforce formatting and lint standards.

## Testing Guidelines
Write deterministic tests with `pytest` and name files `test_*.py`. Mock external services, APIs, and network calls to keep suites fast. Execute suites from the repo root (or the project directory) with `pytest -q`, and share relevant output when discussing changes. Target meaningful coverage, especially for financial calculations and agent logic.

## Commit & Pull Request Guidelines
Compose commit subjects in the imperative mood under 72 characters, optionally adding a scope tag, e.g., `[financial] Update yield curve`. In pull requests, summarize the change, list affected directories, describe setup steps, and attach logs or screenshots when they clarify results. Link issues or tasks so reviewers see context quickly.

## Security & Configuration Tips
Never commit secrets or large datasets; rely on local `.env` files that are already ignored. Document any required services (Ollama, Chroma, etc.) in the relevant README or script docstring, including ports and startup commands. When adding sample data, keep it lightweight and note its provenance alongside the code that consumes it.
