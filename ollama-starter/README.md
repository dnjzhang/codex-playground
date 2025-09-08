# Local AI Agent with RAG (Ollama + Chroma)

Interactive QA over restaurant reviews using local models via Ollama and a Chroma vector store.

## Prerequisites
- Ollama installed and running: `ollama serve`
- Pull models: `ollama pull llama3.2` and `ollama pull mxbai-embed-large`
- Python deps: `pip install -r ollama-starter/requirements.txt`

## First-time Setup (embeddings DB)
- Run once to build/persist the vector DB: `python ollama-starter/restaurant_review.py` (this imports `vector.py`, which creates `./chrome_langchain_db` and populates it from `realistic_restaurant_reviews.csv`).

## Run Chat
- `python ollama-starter/restaurant_review.py`
- You’ll see a prompt: “Ask your question (q to quit):”

## Expected Behavior
- For questions answerable from the reviews, the model responds with a grounded answer; otherwise it says it doesn’t know.
- The Chroma DB persists under `ollama-starter/chrome_langchain_db/` for faster subsequent runs.
