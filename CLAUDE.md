# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a playground repository for experimenting with various AI/ML libraries and frameworks, particularly focused on:
- Memory systems and RAG (Retrieval Augmented Generation)
- Local AI models using Ollama
- Vector databases and embeddings
- Agent-based systems using Letta (formerly MemGPT)
- Financial data analysis and treasury operations

## Environment Setup

The repository uses Python with a virtual environment located at `./.venv/`. All scripts expect to be run from the project root directory.

### Key Dependencies
- Most projects use `python-dotenv` for environment variable management
- Ollama for local LLM models
- LangChain ecosystem (langchain-ollama, langchain-chroma) 
- Letta client for agent systems
- Standard data science stack (pandas, requests)

## Project Structure

The repository is organized into domain-specific directories:

### `memgpt/` - Letta Agent Systems
- **`letta_agent_chat.py`** - Interactive chat with Letta agents, supports agent reactivation from database
- **`letta_agent_inspector.py`** - Debug and inspect agent states and attached sources
- **`rag_agent_demo.py`** - RAG implementation using Letta's archival memory
- **`run-letta-server.sh`** - Docker script to run Letta server (requires OPENAI_API_KEY)

### `ollama-starter/` - Local LLM with RAG
- **`vector.py`** - ChromaDB vector store setup for restaurant reviews
- **`restaurant_review.py`** - RAG chatbot using Ollama and ChromaDB
- Uses mxbai-embed-large for embeddings and stores data in `./chrome_langchain_db/`

### `langmem_sample/` - Memory Management Systems  
- **`memory_store.py`** - InMemoryStore with similarity search capabilities
- **`episodic_memory_basic.py`** - Basic episodic memory implementation
- Uses OpenAI embeddings (text-embedding-3-small) for similarity search

### `treasury/` - Financial Data Analysis
- Treasury auction data querying and CD rate analysis
- Uses pandas-datareader for financial data

### `youtube_transcript/` - Content Processing
- YouTube transcript extraction and markup formatting

### `mcp/` - Model Context Protocol
- Research server and database query implementations

## Common Commands

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies for specific projects
pip install -r ollama-starter/requirements.txt
```

### Running Letta Server
```bash
# Set environment variable first
export OPENAI_API_KEY=sk-...

# Run Letta server in Docker
./memgpt/run-letta-server.sh
```

### Ollama Setup
Ensure Ollama is installed and running locally for the ollama-starter projects.

## Architecture Patterns

### Agent Systems
- Scripts use argument parsing with `argparse` for CLI interfaces
- Most agent scripts support both new agent creation and reactivation of existing agents
- Memory systems use namespace-based organization (e.g., `("user_123", "memories")`)

### Vector Stores  
- ChromaDB is used for persistent vector storage
- Scripts check for existing databases to avoid re-indexing
- Embedding models are specified per project (mxbai-embed-large for Ollama, OpenAI for others)

### Configuration Management
- Environment variables loaded via `python-dotenv`
- Docker configurations include volume mounts for persistent data
- Most scripts include error handling for missing required environment variables

## Key Files to Understand

- `memgpt/letta_agent_chat.py:39` - Shows message creation pattern for Letta SDK compatibility
- `ollama-starter/vector.py:26` - ChromaDB initialization and document loading pattern
- `langmem_sample/memory_store.py:34` - InMemoryStore usage for similarity search
- `memgpt/run-letta-server.sh:3` - Environment variable validation pattern