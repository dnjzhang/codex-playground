#!/bin/sh

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  echo "Export it first, e.g.: export OPENAI_API_KEY=sk-..." >&2
  exit 1
fi

docker run \
  --name letta \
  -v letta-pgdata:/var/lib/postgresql/data \
  -p 8283:8283 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  letta/letta:latest
