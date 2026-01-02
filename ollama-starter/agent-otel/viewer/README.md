# OTel Signal Viewer

Simple TypeScript-based viewer for OpenTelemetry file exports under `./otel-out`.

## Quick start

```bash
cd /Users/jzhang/git-repo/codex-playground/ollama-starter/agent-otel/viewer
npm install
npm run build
npm run serve
```

Then open `http://localhost:8000/viewer/index.html`.

## Notes

- The viewer reads `./otel-out/traces.json`, `./otel-out/metrics.json`, and `./otel-out/logs.json`.
- Each table has its own reload button and filter controls (all vs last N) plus per-column substring filters.
- Log answers are truncated by default; click to expand.
