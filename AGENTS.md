# Repository Guidelines

## Project Structure & Module Organization
Treat each top-level folder as an independent project: `financial/` handles CSV-backed analytics and plots, `treasury/` focuses on Treasury scripts, `youtube_transcript/` manages transcript tooling, `mcp/` and `memgpt/` host agent experiments, `langmem_sample/` explores memory techniques, and `ollama-starter/` ships LLM demos with its own `requirements.txt`. Place new Python modules directly inside the relevant directory, keep PDFs or sample data alongside the code they describe, and avoid cross-project imports unless absolutely necessary.

## Build, Test, and Development Commands
- `python -m venv .venv` — create a local virtual environment (target Python 3.11+).
- `source .venv/bin/activate` — activate that environment before installing packages.
- `pip install -r ollama-starter/requirements.txt` — install dependencies for the Ollama demo; repeat per project as needed.
- `python financial/cd_maturity_plot.py` — run the CD maturity plot pipeline.
- `python treasury/estimate_yields.py` — estimate Treasury yields with the latest assumptions.
- `python youtube_transcript/youtube_transcript.py` — pull and process YouTube transcripts.

## Coding Style & Naming Conventions
Adhere to PEP 8 with 4-space indentation and an 88–100 character soft limit. Use `snake_case` for files, modules, and functions, and reserve `CamelCase` for classes. Favor descriptive filenames like `letta_memory_inspector.py`. Run `black .` and `ruff .` before sharing code to catch formatting or lint issues early.

## Testing Guidelines
Use `pytest` for new tests and name files `test_*.py`, either adjacent to the code or within a local `tests/` folder. Keep tests deterministic, light on data, and mock any external services or network calls. Execute suites from the repo root (or project directory) with `pytest -q`, and mention relevant output when submitting changes.

## Commit & Pull Request Guidelines
Write commit subjects in the imperative mood under 72 characters, optionally prefixed with a scope tag such as `[financial] Add CD maturity plot`. Describe the reasoning, trade-offs, and CLI samples in the body when they help reviewers. Pull requests should summarize the change, list affected directories, document setup or data requirements, and include screenshots or logs when they clarify results. Link associated issues or tasks to give maintainers quick context.

## Security & Configuration Tips
Keep secrets and large datasets out of version control; rely on local `.env` files already ignored by Git. Document any required external services (for example, Ollama or Chroma) in the relevant README or script docstring, including ports, environment variables, and startup commands.
