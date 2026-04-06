# langchain-agent

## Purpose
This project is a local-first chatbot built with LangChain, LangGraph, DeepAgent, and Ollama.

## Runtime assumptions
- Ollama runs locally by default on `http://localhost:11434`
- The default model is `gemma4:e4b`
- MCP tool registration is loaded from `app/mcp_servers.json`

## Scope
- Keep the first version focused on a terminal chatbot
- Design the runtime so a later REST API can reuse the same agent service layer
- Treat MCP configuration as project-local, not tied to a global personal config

## Safety
- Prefer project-local file operations only
- Do not assume external services beyond local Ollama and configured MCP servers
