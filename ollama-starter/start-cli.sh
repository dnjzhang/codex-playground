#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OBSERVABILITY_ENABLED="${OBSERVABILITY_ENABLED:-true}"
export OTEL_SERVICE_NAME="${OTEL_SERVICE_NAME:-rag-pipeline-topic-hub}"
export OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:4317}"
export OBSERVABILITY_CAPTURE_CONTENT="${OBSERVABILITY_CAPTURE_CONTENT:-false}"
python "${script_dir}/rag-pipeline-topic-hub.py" "$@"
