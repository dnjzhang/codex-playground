# Repository Guidelines

## Project Structure & Module Organization
- Top-level folders operate as independent projects: `financial/`, `treasury/`, `youtube_transcript/`, `mcp/`, `memgpt/`, `langmem_sample/`, and `ollama-starter/`.
- Keep code, sample data, and notes inside their respective folders; avoid cross-project imports unless absolutely necessary.
- Place new tests beside the code they cover or within a local `tests/` subfolder to keep context close to implementation.
- Store configuration samples as `.env.example` files; real secrets belong in untracked `.env` files.

## Build, Test, and Development Commands
- `python -m venv .venv` — create a Python 3.11+ virtual environment at the repo root.
- `source .venv/bin/activate` — activate the virtual environment before installing dependencies.
- `pip install -r <project>/requirements.txt` — install project-specific packages (e.g., `pip install -r ollama-starter/requirements.txt`).
- `python financial/cd_maturity_plot.py`, `python treasury/estimate_yields.py`, `python youtube_transcript/youtube_transcript.py` — run canonical entry points for analytics, yield estimation, and transcript processing.
- `./asset_updator.sh --in <input.csv> --out <output.csv>` or `./asset_updator.sh --in <input.csv> --inplace` — update portfolio CSVs via `asset_updator/cli.py`; `--inplace` creates `<input-name>-<YYYY-MM-DD_HHMMSS>.csv` backups.
- `pytest -q` — execute tests from the repo root or a project directory; keep runs deterministic by mocking external services.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation and an 88–100 character soft limit.
- Use `snake_case` for modules, functions, and files; reserve `CamelCase` for classes.
- Run `black .` and `ruff .` before sharing changes to maintain formatting and lint standards.
- Prefer descriptive filenames such as `letta_memory_inspector.py` that communicate intent.

## Testing Guidelines
- Write pytest-based suites with deterministic fixtures; mock APIs, network calls, and time-sensitive data.
- Name test files `test_*.py` and colocate them with the code under test or a sibling `tests/` directory.
- Favor coverage of financial calculations and agent logic; document notable gaps in PR descriptions.

## Commit & Pull Request Guidelines
- Compose commit subjects in the imperative mood under 72 characters, optionally prefixed with a scope (e.g., `[financial] Add CD workbook`).
- Summaries should list affected directories, setup steps, and any logs or screenshots that clarify results.
- Link related issues or tasks so reviewers can trace motivation quickly.
- Highlight assumptions, data sources, and follow-up work directly in the PR body; keep discussions anchored to specific files or functions.

## Security & Configuration Tips
- Never commit secrets, large datasets, or API keys; rely on local `.env` files already ignored by Git.
- Document required services (Ollama, Chroma, etc.) in module READMEs or script docstrings, including default ports and startup commands.
- When adding sample datasets, keep them lightweight and note provenance near the consuming code.
