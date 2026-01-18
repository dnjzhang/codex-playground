# OTel Signal Viewer

Simple TypeScript-based viewer for OpenTelemetry file exports under `./otel-out`.

## Quick start

```bash
cd /Users/jzhang/git-repo/codex-playground/ollama-starter/agent-otel/viewer
npm ci
npm run build
npm run serve
```

Then open `http://localhost:8000/viewer/index.html`.

## Clean rebuild

```bash
rm -rf node_modules dist
npm ci
npm run build
```

## Notes

- The viewer reads `./otel-out/traces.json`, `./otel-out/metrics.json`, and `./otel-out/logs.json`.
- `otel-out/` should live inside the `viewer/` directory (next to `app.ts`), so the files resolve
  to `viewer/otel-out/` when the page is served.
- Each table has its own reload button and filter controls (all vs last N) plus per-column substring filters.
- Log answers are truncated by default; click to expand.
