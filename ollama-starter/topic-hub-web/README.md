## Topic Hub Web Client

This Vite/React + TypeScript single-page app provides an OpenAI-style chat UI for the RAG Topic Hub REST API.

### Quick start

1. Install dependencies:
   ```bash
   npm install
   ```
2. Run the dev server:
   ```bash
   npm run dev
   ```
3. In your browser, open the URL printed by Vite (default `http://localhost:5173`).
4. In the left sidebar, set the API Base URL for the Python service (default `http://localhost:8000`) and fill in the topic/options you want. Click **Start session** to POST `/sessions`, then chat in the main panel (messages are sent to `/sessions/{session_id}/chat`).

### Configuration notes

- The form mirrors the CLI flags and REST payload fields from `rag-pipeline-topic-hub.py`.
- Most fields default to the CLI defaults; leave optional inputs blank to rely on the server-side defaults (`null` values are omitted).
- Check “Show sources in responses” to request citation data from the API. Retrieved sources render under each assistant message when available.
- The API Base URL is saved to `localStorage` so you only have to set it once per browser.
