# MemGPT – RAG Agent Demo

Demo that uploads a PDF source to a running Letta server and chats with an agent using OpenAI models.

## Prerequisites
- Letta server running at `http://localhost:8283` (see `memgpt/run-letta-server.sh`).
- OpenAI API key configured in your environment (`OPENAI_API_KEY`).
- Dependencies: `pip install letta-client python-dotenv pydantic`
- Place `PLTR-FY2024-10-K.pdf` in `memgpt/` (or adjust the path in the script).

## Run
- `python memgpt/rag_agent_demo.py`

## Expected Output
- Prints job/source IDs, e.g., `Data source loaded with source ID: ...` and `Number passages loaded: N`.
- Then prints message stream with emojis (🧠 🤖 🔧) and the agent’s answer to the Palantir question.

