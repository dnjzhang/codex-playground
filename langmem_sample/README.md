# LangMem Samples

Examples that use LangGraph + LangMem for simple memory search and episodic logging.

## memory_store.py
- Purpose: store toy memories and run similarity search or key lookup.
- Run examples:
  - `python langmem_sample/memory_store.py --query "pizza"`
  - `python langmem_sample/memory_store.py --key 5`
- Expected: prints top similar memories with scores, or the specific memory value.

## record_episode.py
- Purpose: demonstrates recording an episode schema and leveraging similar past episodes.
- Requirements: `pip install langmem langgraph langchain python-dotenv pydantic` and `OPENAI_API_KEY`.
- Run: `python langmem_sample/record_episode.py`
- Expected: prints a model-generated answer and shows a search result list of stored “episodes”.

