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
- Graph visualization: `--graph-diagram path/to/diagram.png` exports the LangGraph structure (requires a Mermaid rendering backend).

## FastAPI Server
Run `rag-pipeline-topic-hub.py --serve-api` to expose REST endpoints:
- `POST /sessions` with a JSON payload matching `SessionCreatePayload` starts a topic session and returns a session id.
- `POST /sessions/{session_id}/chat` accepts `{ "question": "..." }` and returns the model answer plus optional source metadata.
Configure host/port via `--api-host` and `--api-port`. Cross-origin requests are allowed for local prototyping.

## MCP Tool Integration (Optional)
`rag-pipeline-topic-hub.py` automatically binds chat providers to MCP tools published by your `db-mcp-server`. When registration succeeds, the agent emits an info-level log listing every tool it can call, and any tool call selected by the model (for example, `list_tables`) is executed against the MCP server with the results fed back into the conversation.

### Configure via Environment Variables
Provide the SSE endpoint for the server (no auth headers are required by default):

```bash
export DB_MCP_TRANSPORT=sse
export DB_MCP_SSE_URL=http://localhost:8080/sse
# Optional: limit which tools are exposed
# export DB_MCP_TOOLS=searchLog,anotherTool
```

### Configure via YAML or JSON
You can capture the same settings in a config file and point the loader at it:

```yaml
# mcp-config.yaml
db_mcp:
  transport: sse
  sse_url: http://127.0.0.1:5173/mcp
  # headers:
  #   Authorization: Bearer <token-if-needed>
  tools:
    - searchLog
```

```bash
export DB_MCP_CONFIG=./mcp-config.yaml
```

JSON files follow the same shape. The loader expands `~` and relative paths, and environment variables still take precedence over file entries. Install `pyyaml` if you prefer YAML:

```bash
pip install pyyaml
```

## Next Steps
- Capture additional topic folders and rebuild via `topic-curator.py --reset` to refresh embeddings.
- Tune retrieval/rerank parameters per topic to balance recall vs. precision.
- Deploy the FastAPI server behind authentication before exposing outside local development.
