# Local AI Agent with RAG (Ollama + Chroma)

Interactive QA over local data using Ollama + Chroma. Includes:
- Legacy restaurant reviews demo
- Topic-based PDF vector builder + query client

## Prerequisites
- Ollama installed and running: `ollama serve`
- Pull models: `ollama pull llama3.2` and `ollama pull mxbai-embed-large`
- Python deps: `pip install -r ollama-starter/requirements.txt`

## First-time Setup (install deps)
- Install Python deps: `pip install -r ollama-starter/requirements.txt`
- Ensure Ollama models are present:
  - `ollama pull llama3.2` (or another chat model)
  - `ollama pull mxbai-embed-large` (or another embedding model)

## Restaurant Reviews Demo (legacy)
- Build/query flow remains unchanged:
  - `python ollama-starter/restaurant_review.py`
  - Follow the prompt: “Ask your question (q to quit):”

## Topic PDF Vector Builder
- Place PDFs for a topic under a folder, e.g. `ollama-starter/topics/spring-boot/`.
- Build a per-topic Chroma DB with Ollama embeddings:
  - `python ollama-starter/build_topic_vector_db.py --topic-dir ollama-starter/topics/spring-boot`
  - Optional flags:
    - `--topic-name <name>`: override topic name (defaults to folder name)
    - `--persist-root <dir>`: root directory for per-topic DBs (default `ollama-starter/chroma_topics`)
    - `--collection-name <name>`: override Chroma collection (default `topic_<topic>`)
    - `--embedding-model <model>`: Ollama embedding model (default `mxbai-embed-large`)
    - `--chunk-size 1200 --chunk-overlap 200`: splitting params
    - `--reset`: recreate topic DB directory

Output:
- DB under `ollama-starter/chroma_topics/<topic>/`
- Collection name `topic_<topic>` unless overridden

## Topic Query Client
- Query a built topic DB using a minimal LangGraph-based RAG pipeline:
  - One-off query:
    - `python ollama-starter/topic_guide.py --topic-name spring-boot --query "What is Spring Boot auto-configuration?"`
  - Interactive session:
    - `python ollama-starter/topic_guide.py --topic-name spring-boot`
  - Useful flags:
    - `--chat-model llama3.2` (default)
    - `--embedding-model mxbai-embed-large` (should match build-time model)
    - `--persist-root ollama-starter/chroma_topics`
    - `--k 5` top-k retrieval

Notes:
- Ensure the chat and embedding models are downloaded in Ollama before running.
- The builder uses LangChain loaders/splitters; the query app uses LangGraph to orchestrate retrieval → generation.
