# Topic-Centric RAG Toolkit

This directory provides two companion utilities for topic-focused Retrieval-Augmented Generation (RAG):

- `topic-curator.py` ingests PDFs or plain text files into a Chroma vector store, using either local Ollama embeddings or Oracle Cloud Infrastructure (OCI) Generative AI embeddings.
- `rag-pipeline-topic-hub.py` queries those topic stores through a LangGraph pipeline, supporting both a command line chat client and an optional FastAPI server.

Use `topic-curator.py` to prepare your knowledge base for each topic, then interact with it via `rag-pipeline-topic-hub.py`.

## Prerequisites
- Python 3.11+ with dependencies: `pip install -r requirements.txt`
- ChromaDB persistence directory (created automatically on first run)
- **For Ollama workflows**
  - Install Ollama, then start the service with `ollama serve`
  - Pull at least one chat model (e.g. `ollama pull llama3.2`) and one embedding model (e.g. `ollama pull mxbai-embed-large`)
- **For OCI workflows**
  - Provide credentials and defaults in `oci-config.json` (endpoint, compartment OCID, model ids, config profile)
  - Network access to OCI Generative AI endpoints
- Optional reranking requires `sentence-transformers` (install on demand when using `--rerank-model`)
- Optional providers may require extra Python packages (see below)

### Install Ollama
- macOS (Homebrew): `brew install ollama` or download the installer at https://ollama.com/download/mac
- Linux: run `curl -fsSL https://ollama.com/install.sh | sh` (see distro notes at https://ollama.com/download/linux)
- Windows: download the installer from https://ollama.com/download/windows and follow the setup wizard

After installation, start the service with `ollama serve` and keep it running while building or querying topics.

### Optional Python Extras
- OCI Generative AI support (for `--embedding-provider oci` or `--chat-provider oci`):
  ```bash
  pip install oci
  ```
- Cross-encoder reranking relies on Hugging Face cross encoders. If you skipped the shared `requirements.txt`, install:
  ```bash
  pip install sentence-transformers
  ```
  The package installs the appropriate PyTorch dependency for your OS and CPU/GPU automatically; refer to https://pytorch.org/get-started/locally/ if you prefer a specific build.

## Quick Start (Ollama)
1. Collect PDFs under a topic directory, for example `./topics/spring-boot`.
2. Build the topic store:
   ```bash
   python topic-curator.py \
     --source-dir ./topics/spring-boot \
     --topic-name spring-boot \
     --embedding-provider ollama \
     --embedding-model mxbai-embed-large
   ```
   This writes vectors and metadata to `.chroma/spring-boot/db_meta.json`.
3. Ask a question from the CLI:
   ```bash
   python rag-pipeline-topic-hub.py \
     --topic-name spring-boot \
     --query "What is Spring Boot auto-configuration?" \
     --show-sources
   ```
   Omit `--query` to enter interactive chat mode.

## Topic Curator (`topic-curator.py`)
- Accepts PDF (`--input-format pdf`) or UTF-8 text (`--input-format text`) sources.
- Embedding providers:
  - `--embedding-provider ollama` (default) uses `langchain_ollama.OllamaEmbeddings`.
  - `--embedding-provider oci` reads OCI settings from `oci-config.json` (override with CLI flags).
- Key options:
  - `--topic-name` overrides the topic folder name used in persistence paths.
  - `--persist-root` defaults to `.chroma` but can point anywhere writable.
  - `--chunk-size` and `--chunk-overlap` control the recursive text splitter.
  - `--reset` deletes and recreates the topic directory before indexing.
- Outputs:
  - Persisted Chroma collection under `<persist_root>/<topic_name>/`
  - Deterministic document ids combining source path, page, and offsets
  - `db_meta.json` capturing embedding provider/model, chunk settings, and source inventory (consumed by the query pipeline)

## Topic Hub (`rag-pipeline-topic-hub.py`)
- Resolves existing topic stores and auto-detects embedding provider/model when metadata is present.
- Retrieval settings:
  - `--k` sets top-k vector retrieval; `--search-type mmr` adds diversity with `--lambda-mult`.
  - `--rerank-model` enables cross-encoder reranking (defaults to `BAAI/bge-reranker-base`); configure `--rerank-top-n` to limit context passed to the LLM.
- Chat model providers:
  - `--chat-provider ollama` (default) with optional `--chat-model` override.
  - `--chat-provider oci` for OCI Generative AI chat (`oci-config.json` must include `model_name`).
- CLI modes:
- `--query "…" ` runs a single turn and exits.
- Interactive chat prompts until `q`, `quit`, or `exit`.
- `--show-sources` prints retrieved context metadata.
- `--inspect "…" ` inspects vector and reranker scores without generating an answer.
- `--disable-mcp` skips MCP tool registration if you want a pure vector-RAG run.
- Graph visualization: `--graph-diagram path/to/diagram.png` exports the LangGraph structure (requires a Mermaid rendering backend).

## FastAPI Server
Run `rag-pipeline-topic-hub.py --serve-api` to expose REST endpoints:
- `POST /sessions` with a JSON payload matching `SessionCreatePayload` starts a topic session and returns a session id.
- `POST /sessions/{session_id}/chat` accepts `{ "question": "..." }` and returns the model answer plus optional source metadata.
Configure host/port via `--api-host` and `--api-port`. Cross-origin requests are allowed for local prototyping.

## MCP Tool Integration (Optional)
`rag-pipeline-topic-hub.py` automatically binds chat providers to MCP tools published by your `db-mcp-server`. When registration succeeds, the agent emits an info-level log listing every tool it can call, and any tool call selected by the model (for example, `list_tables`) is executed against the MCP server with the results fed back into the conversation.

To opt out for a given run, start the script with `--disable-mcp` (the FastAPI payload exposes the same toggle via the `enable_mcp` flag).

### Configure via Environment Variables
Provide the Streamable HTTP endpoint for the server (defaults to `http://localhost:8080/mcp`):

```bash
export DB_MCP_URL=http://localhost:8080/mcp
# Optional: limit which tools are exposed
# export DB_MCP_TOOLS=searchLog,anotherTool
```

## Observability (OpenTelemetry + OpenInference)
The RAG pipeline can emit traces + metrics via OpenTelemetry with OpenInference auto-instrumentation.

Install dependencies:
```bash
pip install -r requirements.txt
```
Optional (if you want explicit LangGraph spans and the package is available):
```bash
pip install openinference-instrumentation-langgraph
```

Enable instrumentation (disabled by default) and configure OTLP export:
```bash
export OBSERVABILITY_ENABLED=true
export OTEL_SERVICE_NAME=rag-pipeline-topic-hub
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

By default, prompt/response content is hidden. To capture content:
```bash
export OBSERVABILITY_CAPTURE_CONTENT=true
```

`start-api.sh` and `start-cli.sh` set these values with sensible defaults for local collection; export overrides in your shell to change them.

Token usage metrics: Ollama reports token counts via `prompt_eval_count` and `eval_count`, while OCI (and other providers) typically use OpenAI-style `prompt_tokens`/`completion_tokens`.

QA logging to OTel (disabled by default):
```bash
export QA_LOGGING_ENABLED=true
# Set to false to store only hashes (question/answer fields remain empty strings)
export QA_LOG_CONTENT=true
```

Each QA log record includes `start_timestamp` (UTC), `complete_timestamp` (UTC), and `elapsed_ms` for end-to-end latency.

How to verify QA logs:
1) Run the agent once with `QA_LOGGING_ENABLED=true`.
2) Confirm the collector output contains a log record with `"type":"response"` or `"type":"error"`.
   - For the local file collector, check `agent-otel/otel-out/logs.json`.

For local macOS collection, see `agent-otel/README.md` for the collector container setup.

## Next Steps
- Capture additional topic folders and rebuild via `topic-curator.py --reset` to refresh embeddings.
- Tune retrieval/rerank parameters per topic to balance recall vs. precision.
- Deploy the FastAPI server behind authentication before exposing outside local development.
