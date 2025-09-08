# Repository Guidelines

## Project Structure & Module Organization
- Multi-project sandbox; each subfolder is self-contained.
- Key directories: `financial/` (CSV data, plots), `treasury/` (Treasury scripts), `youtube_transcript/` (transcript tools), `mcp/`, `memgpt/`, `langmem_sample/`, `ollama-starter/` (LLM demos; has `requirements.txt`).
- Python scripts live directly under each folder; PDFs and assets stay alongside their code.

## Build, Test, and Development Commands
- Python environment (recommended 3.11+):
  - Create venv: `python -m venv .venv`
  - Activate: `source .venv/bin/activate`
- Install deps per subproject as needed, e.g.: `pip install -r ollama-starter/requirements.txt`.
- Run scripts directly, e.g.:
  - `python financial/cd_maturity_plot.py`
  - `python treasury/estimate_yields.py`
  - `python youtube_transcript/youtube_transcript.py`

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indentation, 88–100 column soft limit.
- Names: `snake_case` for files/functions, `CamelCase` for classes. Prefer descriptive module names (e.g., `letta_memory_inspector.py`).
- Formatting/linting (optional but encouraged): `black .` and `ruff .` before commits.

## Testing Guidelines
- No global test harness yet. If adding tests, use `pytest` and name files `test_*.py` near the code or under `tests/` within the subproject.
- Keep tests fast and data-light; mock network and external tools.
- Run: `pytest -q` from the repo root or subproject.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise subject (<72 chars). Prefix with scope when useful, e.g., `[financial] Add CD maturity plot`.
- Body explains “why” and notable decisions; include sample command/output when changing CLIs.
- PRs include: clear description, affected folder(s), run instructions, data requirements, and screenshots/logs when applicable. Link related issues.

## Security & Configuration Tips
- Do not commit secrets or large datasets. Use local `.env` files and keep them in `.gitignore`.
- Clearly document any external service prerequisites (e.g., Ollama/Chroma) in the subproject README or script docstring, including setup and ports.

