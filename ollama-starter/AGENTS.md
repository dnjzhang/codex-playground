# Repository Guidelines

## Project Structure & Module Organization
Python utilities live at the repo root: `topic-curator.py` builds Chroma topic stores, `rag-pipeline-topic-hub.py` runs LangGraph retrieval workflows, and `provider_lib.py` bundles shared provider helpers. Observability helpers live in `observability/` and local collector setup lives in `agent-otel/`. Keep supporting modules beside these scripts or under lightweight subfolders. Store docs, diagrams, and walkthroughs under `docs/`, leave historical experiments in `archive/`, and host the React client inside `topic-hub-web/`. Place new datasets or fixtures next to the code that consumes them, and co-locate tests in the same folder or a local `tests/` subdirectory.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` bootstraps the Python toolchain.
- `pip install -r requirements.txt` installs backend dependencies for the CLI and API utilities.
- `./start-cli.sh --topic-name <topic>` runs the interactive CLI with observability defaults.
- `./start-api.sh` runs the FastAPI interface with observability defaults.
- `python topic-curator.py --help` and `python rag-pipeline-topic-hub.py --serve-api` run the ingestion tool and the FastAPI interface directly.
- `cd agent-otel && ./start-collect.sh` launches the local OpenTelemetry collector container.
- `pytest -q` executes backend tests from the repo root; add `-k <pattern>` to focus runs.
- `cd topic-hub-web && npm install` followed by `npm run dev` launches the chat UI; use `npm run build` for production assets and `npm run lint` for TypeScript diagnostics.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and an 88–100 character soft wrap. Use `snake_case` for functions/modules, `CamelCase` for classes, and descriptive filenames such as `letta_memory_inspector.py`. Run `black .` and `ruff .` before submitting Python changes. Keep docstrings focused on purpose/side effects, and prefer dependency injection over hard-coded paths. In `topic-hub-web/`, stick to TypeScript defaults, favor named exports, and group React components under clear folder names.

## Testing Guidelines
Write deterministic `pytest` suites, mock Ollama, OCI, and Chroma boundaries, and isolate file-system writes to temporary directories. Name Python tests `test_*.py` and keep them adjacent to the code under test. For the web client, add component or integration specs under `topic-hub-web/src/__tests__/` and run them via `npm test` (wire up Vitest if new suites are introduced). Share log excerpts or screenshots when behavior is hard to capture in assertions.

## Commit & Pull Request Guidelines
Compose commit subjects in the imperative mood, under 72 characters, and optionally prefix with a scope (e.g., `[backend] Add OCI client`). Each PR should summarize the change, list affected directories, document setup or migration steps, and link related issues. Include test results (`pytest`, `npm run lint`, or new harnesses) and attach media or sample output when it clarifies user-facing changes.

## Security & Configuration Tips
Never commit secrets, API keys, or large datasets. Store environment settings in ignored `.env` files and document required variables in README sections or docstrings. Keep `oci-config.json` templates scrubbed of live credentials, and rotate local models or embeddings outside version control. When adding sample data, note its provenance and ensure it is lightweight enough for quick indexing.
