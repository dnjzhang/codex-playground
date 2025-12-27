# Local OpenTelemetry Collector (macOS)

This setup runs an OpenTelemetry Collector container locally (Docker required)
with OTLP over gRPC (4317) and HTTP (4318) and writes traces/metrics/logs to files.

## Start the collector

```bash
cd /Users/jzhang/git-repo/codex-playground/ollama-starter/agent-otel
./start-collect.sh
```

Override the collector image tag:

```bash
OTEL_COLLECTOR_TAG=0.110.0 ./start-collect.sh
```

## Verify output files

```bash
ls -l ./otel-out
# Optional: watch traces
# tail -f ./otel-out/traces.json
```

## Example env vars for a local Python app

HTTP/protobuf:
```bash
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_SERVICE_NAME=python-agent
```

gRPC:
```bash
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317
export OTEL_SERVICE_NAME=python-agent
```

## Troubleshooting (macOS)

- Ports busy: `lsof -i :4317` / `lsof -i :4318` and stop the conflicting process.
- App in another container: use `host.docker.internal:4318` (HTTP) or `host.docker.internal:4317` (gRPC).
- No output files: confirm the collector is running and that `./otel-out` is mounted to `/var/otel-out`.
