#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/otel-out"
CONFIG_FILE="${SCRIPT_DIR}/otel-collector-config.yaml"
CONTAINER_NAME="otel-collector"
IMAGE_TAG="${OTEL_COLLECTOR_TAG:-0.109.0}"
IMAGE="ghcr.io/open-telemetry/opentelemetry-collector-releases/opentelemetry-collector-contrib:${IMAGE_TAG}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker was not found on PATH." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

docker pull "${IMAGE}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${CONTAINER_NAME}" \
  -p 4317:4317 \
  -p 4318:4318 \
  -v "${OUT_DIR}:/var/otel-out" \
  -v "${CONFIG_FILE}:/etc/otelcol/config.yaml:ro" \
  "${IMAGE}" \
  --config /etc/otelcol/config.yaml

docker ps --filter "name=${CONTAINER_NAME}"
