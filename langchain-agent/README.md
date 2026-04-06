# langchain-agent

`langchain-agent` is a local-first chatbot project built with LangChain, LangGraph, DeepAgent, Ollama, and MCP tools.

The current version is a terminal chatbot. The runtime is intentionally separated from the CLI so the same agent service layer can later back an application-specific REST API and UI.

## Current capabilities

- Local Ollama chat model support
- Default model `gemma4:e4b`
- Model override from the CLI
- Project-local MCP server registration
- Playwright MCP configured as the first external tool
- DeepAgent-based tool and skill support
- Server-generated session IDs
- In-memory per-session conversation state
- Streamed CLI responses by default

## Project structure

```text
langchain-agent/
├── AGENTS.md
├── README.md
├── start-cli.sh
├── requirements.txt
├── .env.example
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── mcp.py
│   ├── mcp_servers.json
│   ├── prompts.py
│   ├── runtime.py
│   ├── session.py
│   └── skills/
│       └── playwright-browser/
│           └── SKILL.md
└── tests/
```

## Architecture

The runtime is centered around [runtime.py](/Users/jzhang/git-repo/codex-playground/langchain-agent/app/runtime.py).

- `AgentRuntime` loads MCP tools once at startup
- it creates DeepAgent graphs per model
- it creates server-generated session IDs
- it keeps LangGraph checkpoint state in memory
- it exposes both non-streaming and streaming chat methods

The CLI in [main.py](/Users/jzhang/git-repo/codex-playground/langchain-agent/app/main.py) is thin and only handles:

- argument parsing
- interactive terminal input
- session switching
- model switching
- printing streamed output

That split is what makes the later REST API straightforward.

## Prerequisites

You need:

- Python virtual environment already created at `.venv`
- Ollama running locally
- the target Ollama model installed, currently `gemma4:e4b`
- Node.js and `npx` available if you want Playwright MCP enabled

The runtime assumes Ollama is reachable at `http://localhost:11434` unless overridden by environment variable.

## Install dependencies

From the project directory:

```bash
cd /Users/jzhang/git-repo/codex-playground/langchain-agent
source .venv/bin/activate
pip install -r requirements.txt
```

The wrapper script [start-cli.sh](/Users/jzhang/git-repo/codex-playground/langchain-agent/start-cli.sh) assumes the local `.venv` exists and activates it automatically.

## Environment variables

You can keep the defaults, or copy `.env.example` to `.env` and change values.

Supported variables:

- `OLLAMA_BASE_URL`
  Default: `http://localhost:11434`
- `LANGCHAIN_AGENT_DEFAULT_MODEL`
  Default: `gemma4:e4b`
- `LANGCHAIN_AGENT_MCP_CONFIG`
  Default: `app/mcp_servers.json`

Example:

```bash
cp .env.example .env
```

Example `.env`:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
LANGCHAIN_AGENT_DEFAULT_MODEL=gemma4:e4b
LANGCHAIN_AGENT_MCP_CONFIG=app/mcp_servers.json
```

## Run the chatbot

The recommended way to start the chatbot is the wrapper script:

```bash
cd /Users/jzhang/git-repo/codex-playground/langchain-agent
./start-cli.sh
```

Start with a different model:

```bash
./start-cli.sh --model mistral-nemo:latest
```

Disable streaming and print only the final response each turn:

```bash
./start-cli.sh --no-stream
```

You can also combine flags:

```bash
./start-cli.sh --model gemma4:e4b --no-stream
```

### What `start-cli.sh` does

[start-cli.sh](/Users/jzhang/git-repo/codex-playground/langchain-agent/start-cli.sh):

- changes into the project directory
- activates `.venv`
- runs `python -m app.main`
- passes through any CLI arguments you provide

### Direct Python entrypoint

If you want to run the module directly instead of the wrapper:

```bash
cd /Users/jzhang/git-repo/codex-playground/langchain-agent
source .venv/bin/activate
python -m app.main
```

## CLI commands

Inside the chatbot:

- `/help`
  Show the built-in command list
- `/new`
  Start a fresh session using the current model
- `/model <name>`
  Start a fresh session using a different model
- `/session`
  Show the current session ID and model name
- `/quit`
  Exit the chatbot

Example session:

```text
Session: session-3f2c523d759e
Model:   gemma4:e4b
Type /help for commands.

You> /session
Session: session-3f2c523d759e
Model:   gemma4:e4b

You> /model mistral-nemo:latest
Session: session-9a4c1f66aa71
Model:   mistral-nemo:latest
```

## Model selection

Model selection is configurable in two ways:

1. Default model from environment:
   `LANGCHAIN_AGENT_DEFAULT_MODEL`
2. Per-session CLI override:
   `./start-cli.sh --model <model-name>`

The runtime is designed so the future REST API can also accept model name as a request field.

## MCP tools

MCP servers are defined in [mcp_servers.json](/Users/jzhang/git-repo/codex-playground/langchain-agent/app/mcp_servers.json).

The current default configuration registers Playwright MCP:

```json
{
  "servers": {
    "playwright": {
      "enabled": true,
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest", "--browser=chrome"],
      "env": {
        "NPM_CONFIG_CACHE": "/tmp/langchain-agent-npm-cache"
      }
    }
  }
}
```

### Why `NPM_CONFIG_CACHE` is set

On this machine, `npx` hit a permissions problem with the user npm cache under `~/.npm`. The default Playwright MCP config uses a writable temp cache at `/tmp/langchain-agent-npm-cache` to avoid startup failure.

### Add another MCP server

Add a new entry under `servers` in `app/mcp_servers.json`. Example:

```json
{
  "servers": {
    "playwright": {
      "enabled": true,
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest", "--browser=chrome"]
    },
    "example-server": {
      "enabled": true,
      "transport": "stdio",
      "command": "python",
      "args": ["-m", "my_mcp_server"]
    }
  }
}
```

If MCP tool loading fails, the chatbot still starts, but without MCP tools. That fallback is intentional so the local chat path is not blocked by one external tool process.

## DeepAgent skills

The project includes one initial DeepAgent skill:

- [playwright-browser/SKILL.md](/Users/jzhang/git-repo/codex-playground/langchain-agent/app/skills/playwright-browser/SKILL.md)

This skill tells the agent when to use Playwright browser automation tools and how to do it conservatively.

DeepAgent loads skills from `/app/skills/`, rooted at this project directory.

## Session behavior

Sessions are created by the server/runtime, not by the client.

Current behavior:

- each new chat session gets a generated ID like `session-3f2c523d759e`
- conversation history is preserved in memory for that session
- `/new` creates a new empty session
- `/model <name>` creates a new empty session with a different model

Current limitation:

- session state is not persisted across process restarts

## File access behavior

DeepAgent is configured with a filesystem backend rooted at the `langchain-agent` project directory and uses virtual path mode. That keeps DeepAgent file tools scoped to this project instead of the full machine filesystem.

## Testing

Run the unit tests:

```bash
cd /Users/jzhang/git-repo/codex-playground/langchain-agent
source .venv/bin/activate
pytest -q
```

## Verified behavior

The current scaffold was verified with:

- dependency install into `.venv`
- local Ollama API connectivity
- runtime initialization
- session creation
- one non-streaming chat turn
- one streaming chat turn
- `pytest -q`

## Known limitations

- no REST API yet
- no persistent session storage yet
- no structured API schema yet
- MCP tool loading depends on the configured external tool process being runnable
- DeepAgent tool quality still depends on the selected Ollama model's tool-calling behavior

## Next recommended steps

1. Exercise the CLI with `gemma4:e4b` and `mistral-nemo:latest` to compare tool use quality.
2. Add your next MCP or custom tools to `app/mcp_servers.json`.
3. Decide the first REST API surface.
4. Add persistent session storage if you want conversations to survive restarts.
