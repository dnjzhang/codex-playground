type JsonRecord = Record<string, unknown>;

type FilterMode = "all" | "last";

interface TableColumn<T> {
  key: string;
  header: string;
  className?: string;
  value: (row: T) => string;
  render?: (row: T) => string | HTMLElement;
}

interface TraceRow {
  startTime: string;
  startNs: string;
  name: string;
  runId: string;
  durationMs: string;
  status: string;
  traceId: string;
  spanId: string;
  parentSpanId: string;
  service: string;
  scope: string;
}

interface MetricRow {
  time: string;
  timeNs: string;
  name: string;
  type: string;
  value: string;
  runId: string;
  unit: string;
  description: string;
  attributes: string;
}

interface LogRow {
  runId: string;
  type: string;
  question: string;
  answer: string;
  startTimestamp: string;
  completeTimestamp: string;
  elapsedMs: string;
  errorKind: string;
  errorStage: string;
  errorMessage: string;
  errorRetryable: string;
  questionHash: string;
  answerHash: string;
  time: string;
  timeNs: string;
  severity: string;
  traceId: string;
  spanId: string;
  attributes: string;
}

const FILES = {
  traces: "./otel-out/traces.json",
  metrics: "./otel-out/metrics.json",
  logs: "./otel-out/logs.json",
};

const TABLE_CONFIG = {
  logs: {
    tableId: "logsTable",
    summaryId: "logsSummary",
    filterModeName: "logsFilterMode",
    limitInputId: "logsLimitInput",
  },
  metrics: {
    tableId: "metricsTable",
    summaryId: "metricsSummary",
    filterModeName: "metricsFilterMode",
    limitInputId: "metricsLimitInput",
  },
  traces: {
    tableId: "tracesTable",
    summaryId: "tracesSummary",
    filterModeName: "tracesFilterMode",
    limitInputId: "tracesLimitInput",
  },
};

type TableKey = keyof typeof TABLE_CONFIG;

const statusEl = document.getElementById("status") as HTMLDivElement | null;
const reloadBtn = document.getElementById("reloadBtn") as HTMLButtonElement | null;
const tracesSummary = document.getElementById(TABLE_CONFIG.traces.summaryId) as HTMLSpanElement | null;
const metricsSummary = document.getElementById(TABLE_CONFIG.metrics.summaryId) as HTMLSpanElement | null;
const logsSummary = document.getElementById(TABLE_CONFIG.logs.summaryId) as HTMLSpanElement | null;

let traceRows: TraceRow[] = [];
let metricRows: MetricRow[] = [];
let logRows: LogRow[] = [];

function setStatus(message: string) {
  if (statusEl) {
    statusEl.textContent = message;
  }
}

function getTableFilter(tableKey: TableKey): { mode: FilterMode; limit: number } {
  const config = TABLE_CONFIG[tableKey];
  const radios = Array.from(
    document.querySelectorAll<HTMLInputElement>(`input[name='${config.filterModeName}']`),
  );
  const mode = (radios.find((radio) => radio.checked)?.value || "all") as FilterMode;
  const limitInput = document.getElementById(config.limitInputId) as HTMLInputElement | null;
  const limitValue = limitInput?.value ? Number(limitInput.value) : 0;
  const limit = Number.isFinite(limitValue) && limitValue > 0 ? limitValue : 0;
  return { mode, limit };
}

function applyTableFilters<T>(
  rows: T[],
  columns: Array<TableColumn<T>>,
  tableKey: TableKey,
  getTimeNs: (row: T) => string,
  tableId: string,
): T[] {
  const columnFilters = getColumnFilters(tableId);
  const filteredByColumn = rows.filter((row) =>
    columns.every((column) => {
      const filterValue = columnFilters[column.key];
      if (!filterValue) {
        return true;
      }
      const value = column.value(row).toLowerCase();
      return value.includes(filterValue);
    }),
  );
  const sorted = [...filteredByColumn].sort((a, b) => getTimeNs(a).localeCompare(getTimeNs(b)));
  const { mode, limit } = getTableFilter(tableKey);
  if (mode === "last" && limit > 0) {
    return sorted.slice(-limit);
  }
  return sorted;
}

function parseJsonLines(text: string): JsonRecord[] {
  const trimmed = text.trim();
  if (!trimmed) {
    return [];
  }
  if (trimmed.startsWith("[")) {
    try {
      const parsed = JSON.parse(trimmed);
      return Array.isArray(parsed) ? (parsed as JsonRecord[]) : [parsed as JsonRecord];
    } catch (err) {
      console.warn("Failed to parse JSON array", err);
      return [];
    }
  }
  return trimmed
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .flatMap((line) => {
      try {
        return [JSON.parse(line) as JsonRecord];
      } catch (err) {
        console.warn("Failed to parse JSON line", err);
        return [];
      }
    });
}

async function loadFile(path: string): Promise<JsonRecord[]> {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}: ${response.status}`);
  }
  const text = await response.text();
  return parseJsonLines(text);
}

function anyValueToString(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value !== "object") {
    return String(value);
  }
  const record = value as JsonRecord;
  if (typeof record.stringValue === "string") {
    return record.stringValue;
  }
  if (typeof record.intValue === "string" || typeof record.intValue === "number") {
    return String(record.intValue);
  }
  if (typeof record.doubleValue === "number") {
    return String(record.doubleValue);
  }
  if (typeof record.boolValue === "boolean") {
    return String(record.boolValue);
  }
  if (record.arrayValue && typeof record.arrayValue === "object") {
    const arrayValues = (record.arrayValue as JsonRecord).values;
    if (Array.isArray(arrayValues)) {
      return `[${arrayValues.map(anyValueToString).join(", ")}]`;
    }
  }
  if (record.kvlistValue && typeof record.kvlistValue === "object") {
    const kvValues = (record.kvlistValue as JsonRecord).values;
    if (Array.isArray(kvValues)) {
      const entries = kvValues
        .map((entry) => {
          if (!entry || typeof entry !== "object") {
            return "";
          }
          const kv = entry as JsonRecord;
          const key = typeof kv.key === "string" ? kv.key : "";
          const val = anyValueToString(kv.value);
          return key ? `${key}=${val}` : "";
        })
        .filter(Boolean);
      return `{${entries.join(", ")}}`;
    }
  }
  return JSON.stringify(record);
}

function parseLogBody(body: string): JsonRecord | null {
  const trimmed = body.trim();
  if (!trimmed) {
    return null;
  }
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) {
    return null;
  }
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && typeof parsed === "object") {
      return parsed as JsonRecord;
    }
  } catch {
    return null;
  }
  return null;
}

function toAttrMap(attributes: unknown): Record<string, string> {
  if (!Array.isArray(attributes)) {
    return {};
  }
  const entries = attributes
    .map((attr) => {
      if (!attr || typeof attr !== "object") {
        return null;
      }
      const record = attr as JsonRecord;
      const key = typeof record.key === "string" ? record.key : "";
      const value = anyValueToString(record.value);
      return key ? [key, value] : null;
    })
    .filter(Boolean) as Array<[string, string]>;
  return Object.fromEntries(entries);
}

function getAttr(attributes: unknown, key: string): string {
  const map = toAttrMap(attributes);
  return map[key] || "";
}

function getColumnFilters(tableId: string): Record<string, string> {
  const table = document.getElementById(tableId);
  if (!table) {
    return {};
  }
  const inputs = Array.from(
    table.querySelectorAll<HTMLInputElement>("thead .filter-row input[data-column]"),
  );
  return inputs.reduce<Record<string, string>>((acc, input) => {
    const key = input.dataset.column;
    const value = input.value.trim().toLowerCase();
    if (key && value) {
      acc[key] = value;
    }
    return acc;
  }, {});
}

function nanoToIso(nano: unknown): string {
  if (nano === null || nano === undefined) {
    return "";
  }
  const numeric = Number(nano);
  if (!Number.isFinite(numeric)) {
    return "";
  }
  return new Date(numeric / 1_000_000).toISOString();
}

function durationMs(start: unknown, end: unknown): string {
  const startNum = Number(start);
  const endNum = Number(end);
  if (!Number.isFinite(startNum) || !Number.isFinite(endNum)) {
    return "";
  }
  const ms = (endNum - startNum) / 1_000_000;
  return ms >= 0 ? ms.toFixed(2) : "";
}

function createExpandableText(text: string, limit = 180): HTMLElement {
  if (text.length <= limit) {
    const span = document.createElement("span");
    span.className = "expandable-static";
    span.textContent = text;
    return span;
  }
  const button = document.createElement("button");
  button.type = "button";
  button.className = "expandable";
  const truncated = `${text.slice(0, limit - 1)}…`;
  button.textContent = truncated;
  button.dataset.full = text;
  button.dataset.truncated = truncated;
  button.dataset.state = "collapsed";
  button.classList.add("expandable-collapsed");
  button.addEventListener("click", () => {
    const isCollapsed = button.dataset.state === "collapsed";
    if (isCollapsed) {
      button.textContent = text;
      button.dataset.state = "expanded";
      button.classList.remove("expandable-collapsed");
    } else {
      button.textContent = truncated;
      button.dataset.state = "collapsed";
      button.classList.add("expandable-collapsed");
    }
  });
  return button;
}

function renderTable<T>(
  tableId: string,
  rows: T[],
  columns: Array<TableColumn<T>>,
  emptyMessage: string,
) {
  const table = document.getElementById(tableId) as HTMLTableElement | null;
  if (!table) {
    return;
  }
  const tbody = table.querySelector("tbody");
  if (!tbody) {
    return;
  }
  tbody.innerHTML = "";
  if (rows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = columns.length;
    td.className = "muted";
    td.textContent = emptyMessage;
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((column) => {
      const td = document.createElement("td");
      if (column.className) {
        td.className = column.className;
      }
      const renderFn = column.render ?? ((entry: T) => column.value(entry));
      const rendered = renderFn(row);
      if (rendered instanceof HTMLElement) {
        td.appendChild(rendered);
      } else {
        td.textContent = rendered;
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

const TRACE_COLUMNS: Array<TableColumn<TraceRow>> = [
  { key: "runId", header: "Run ID", value: (row) => row.runId },
  { key: "startTime", header: "Start", value: (row) => row.startTime },
  { key: "name", header: "Name", value: (row) => row.name },
  { key: "durationMs", header: "Duration (ms)", value: (row) => row.durationMs },
  { key: "status", header: "Status", value: (row) => row.status },
  { key: "traceId", header: "Trace ID", value: (row) => row.traceId },
  { key: "spanId", header: "Span ID", value: (row) => row.spanId },
  { key: "parentSpanId", header: "Parent", value: (row) => row.parentSpanId },
  { key: "service", header: "Service", value: (row) => row.service },
  { key: "scope", header: "Scope", value: (row) => row.scope },
];

const METRIC_COLUMNS: Array<TableColumn<MetricRow>> = [
  { key: "runId", header: "Run ID", value: (row) => row.runId },
  { key: "time", header: "Time", value: (row) => row.time },
  { key: "name", header: "Name", value: (row) => row.name },
  { key: "type", header: "Type", value: (row) => row.type },
  { key: "value", header: "Value", value: (row) => row.value },
  { key: "unit", header: "Unit", value: (row) => row.unit },
  { key: "description", header: "Description", value: (row) => row.description },
  { key: "attributes", header: "Attributes", value: (row) => row.attributes },
];

const LOG_COLUMNS: Array<TableColumn<LogRow>> = [
  { key: "runId", header: "Run ID", value: (row) => row.runId },
  { key: "type", header: "Type", value: (row) => row.type },
  { key: "question", header: "Question", value: (row) => row.question },
  {
    key: "answer",
    header: "Answer",
    value: (row) => row.answer,
    className: "cell-pre",
    render: (row) => createExpandableText(row.answer),
  },
  { key: "startTimestamp", header: "Start", value: (row) => row.startTimestamp },
  { key: "completeTimestamp", header: "Complete", value: (row) => row.completeTimestamp },
  { key: "elapsedMs", header: "Elapsed (ms)", value: (row) => row.elapsedMs },
  { key: "errorKind", header: "Error Kind", value: (row) => row.errorKind },
  { key: "errorStage", header: "Error Stage", value: (row) => row.errorStage },
  { key: "errorMessage", header: "Error Message", value: (row) => row.errorMessage },
  { key: "errorRetryable", header: "Error Retryable", value: (row) => row.errorRetryable },
  { key: "questionHash", header: "Question Hash", value: (row) => row.questionHash },
  { key: "answerHash", header: "Answer Hash", value: (row) => row.answerHash },
  { key: "time", header: "Time", value: (row) => row.time },
  { key: "severity", header: "Severity", value: (row) => row.severity },
  { key: "traceId", header: "Trace ID", value: (row) => row.traceId },
  { key: "spanId", header: "Span ID", value: (row) => row.spanId },
  { key: "attributes", header: "Attributes", value: (row) => row.attributes },
];

function renderTraces(rows: TraceRow[]) {
  const filtered = applyTableFilters(
    rows,
    TRACE_COLUMNS,
    "traces",
    (row) => row.startNs,
    TABLE_CONFIG.traces.tableId,
  );
  if (tracesSummary) {
    tracesSummary.textContent = `${filtered.length} / ${rows.length} spans`;
  }
  renderTable(TABLE_CONFIG.traces.tableId, filtered, TRACE_COLUMNS, "No spans found.");
}

function renderMetrics(rows: MetricRow[]) {
  const filtered = applyTableFilters(
    rows,
    METRIC_COLUMNS,
    "metrics",
    (row) => row.timeNs,
    TABLE_CONFIG.metrics.tableId,
  );
  if (metricsSummary) {
    metricsSummary.textContent = `${filtered.length} / ${rows.length} datapoints`;
  }
  renderTable(TABLE_CONFIG.metrics.tableId, filtered, METRIC_COLUMNS, "No metrics found.");
}

function renderLogs(rows: LogRow[]) {
  const filtered = applyTableFilters(
    rows,
    LOG_COLUMNS,
    "logs",
    (row) => row.timeNs,
    TABLE_CONFIG.logs.tableId,
  );
  if (logsSummary) {
    logsSummary.textContent = `${filtered.length} / ${rows.length} logs`;
  }
  renderTable(TABLE_CONFIG.logs.tableId, filtered, LOG_COLUMNS, "No logs found.");
}

function setupToggles() {
  const buttons = Array.from(document.querySelectorAll<HTMLButtonElement>(".toggle-btn"));
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const targetId = button.dataset.target;
      if (!targetId) {
        return;
      }
      const container = document.getElementById(targetId);
      if (!container) {
        return;
      }
      const isHidden = container.classList.toggle("is-hidden");
      button.textContent = isHidden ? "Show" : "Hide";
      button.setAttribute("aria-expanded", String(!isHidden));
    });
  });
}

function flattenTraces(entries: JsonRecord[]): TraceRow[] {
  const rows: TraceRow[] = [];
  entries.forEach((entry) => {
    const resourceSpans = (entry.resourceSpans as JsonRecord[]) || [];
    resourceSpans.forEach((resourceSpan) => {
      const resource = resourceSpan.resource as JsonRecord | undefined;
      const resourceAttrs = resource ? toAttrMap(resource.attributes) : {};
      const serviceName = resourceAttrs["service.name"] || "";
      const scopeSpans = (resourceSpan.scopeSpans as JsonRecord[]) || [];
      scopeSpans.forEach((scopeSpan) => {
        const scope = scopeSpan.scope as JsonRecord | undefined;
        const scopeName = typeof scope?.name === "string" ? scope.name : "";
        const spans = (scopeSpan.spans as JsonRecord[]) || [];
        spans.forEach((span) => {
          const attrs = toAttrMap(span.attributes);
          const startNs = span.startTimeUnixNano ? String(span.startTimeUnixNano) : "";
          rows.push({
            startTime: nanoToIso(span.startTimeUnixNano),
            startNs,
            name: typeof span.name === "string" ? span.name : "",
            runId: attrs["app.run_id"] || attrs["run_id"] || "",
            durationMs: durationMs(span.startTimeUnixNano, span.endTimeUnixNano),
            status: span.status && typeof span.status === "object" ? String((span.status as JsonRecord).code ?? "") : "",
            traceId: typeof span.traceId === "string" ? span.traceId : "",
            spanId: typeof span.spanId === "string" ? span.spanId : "",
            parentSpanId: typeof span.parentSpanId === "string" ? span.parentSpanId : "",
            service: serviceName,
            scope: scopeName,
          });
        });
      });
    });
  });
  return rows;
}

function metricPointValue(metric: JsonRecord): { type: string; points: JsonRecord[] } {
  if (metric.sum && typeof metric.sum === "object") {
    return { type: "sum", points: (metric.sum as JsonRecord).dataPoints as JsonRecord[] };
  }
  if (metric.gauge && typeof metric.gauge === "object") {
    return { type: "gauge", points: (metric.gauge as JsonRecord).dataPoints as JsonRecord[] };
  }
  if (metric.histogram && typeof metric.histogram === "object") {
    return { type: "histogram", points: (metric.histogram as JsonRecord).dataPoints as JsonRecord[] };
  }
  return { type: "unknown", points: [] };
}

function formatMetricValue(type: string, point: JsonRecord): string {
  if (type === "sum" || type === "gauge") {
    if (point.asInt !== undefined) {
      return String(point.asInt);
    }
    if (point.asDouble !== undefined) {
      return String(point.asDouble);
    }
  }
  if (type === "histogram") {
    const count = point.count !== undefined ? `count=${point.count}` : "";
    const sum = point.sum !== undefined ? `sum=${point.sum}` : "";
    const min = point.min !== undefined ? `min=${point.min}` : "";
    const max = point.max !== undefined ? `max=${point.max}` : "";
    return [count, sum, min, max].filter(Boolean).join(" ");
  }
  return "";
}

function flattenMetrics(entries: JsonRecord[]): MetricRow[] {
  const rows: MetricRow[] = [];
  entries.forEach((entry) => {
    const resourceMetrics = (entry.resourceMetrics as JsonRecord[]) || [];
    resourceMetrics.forEach((resourceMetric) => {
      const scopeMetrics = (resourceMetric.scopeMetrics as JsonRecord[]) || [];
      scopeMetrics.forEach((scopeMetric) => {
        const metrics = (scopeMetric.metrics as JsonRecord[]) || [];
        metrics.forEach((metric) => {
          const name = typeof metric.name === "string" ? metric.name : "";
          const description = typeof metric.description === "string" ? metric.description : "";
          const unit = typeof metric.unit === "string" ? metric.unit : "";
          const { type, points } = metricPointValue(metric);
          points?.forEach((point) => {
            const attrsMap = toAttrMap(point.attributes);
            const timeNs = point.timeUnixNano ? String(point.timeUnixNano) : "";
            rows.push({
              time: nanoToIso(point.timeUnixNano),
              timeNs,
              name,
              type,
              value: formatMetricValue(type, point),
              runId: attrsMap["app.run_id"] || attrsMap["run_id"] || "",
              unit,
              description,
              attributes: Object.entries(attrsMap)
                .map(([key, value]) => `${key}=${value}`)
                .join(", "),
            });
          });
        });
      });
    });
  });
  return rows;
}

function flattenLogs(entries: JsonRecord[]): LogRow[] {
  const rows: LogRow[] = [];
  entries.forEach((entry) => {
    const resourceLogs = (entry.resourceLogs as JsonRecord[]) || [];
    resourceLogs.forEach((resourceLog) => {
      const scopeLogs = (resourceLog.scopeLogs as JsonRecord[]) || [];
      scopeLogs.forEach((scopeLog) => {
        const logRecords = (scopeLog.logRecords as JsonRecord[]) || [];
        logRecords.forEach((record) => {
          const attrsMap = toAttrMap(record.attributes);
          const timeNs = record.timeUnixNano ? String(record.timeUnixNano) : "";
          const bodyText = anyValueToString(record.body);
          const payload = parseLogBody(bodyText);
          const errorObj =
            payload && typeof payload.error === "object" ? (payload.error as JsonRecord) : null;
          const getPayloadValue = (key: string): string => {
            if (!payload || !(key in payload)) {
              return "";
            }
            const value = payload[key];
            return value === null || value === undefined ? "" : String(value);
          };
          const runId =
            getPayloadValue("run_id") || attrsMap["app.run_id"] || attrsMap["run_id"] || "";
          rows.push({
            runId,
            type: getPayloadValue("type"),
            question: getPayloadValue("question"),
            answer: getPayloadValue("answer"),
            startTimestamp: getPayloadValue("start_timestamp"),
            completeTimestamp: getPayloadValue("complete_timestamp"),
            elapsedMs: getPayloadValue("elapsed_ms"),
            errorKind: errorObj && errorObj.kind ? String(errorObj.kind) : "",
            errorStage: errorObj && errorObj.stage ? String(errorObj.stage) : "",
            errorMessage: errorObj && errorObj.message ? String(errorObj.message) : "",
            errorRetryable:
              errorObj && errorObj.retryable !== undefined ? String(errorObj.retryable) : "",
            questionHash: getPayloadValue("question_hash"),
            answerHash: getPayloadValue("answer_hash"),
            time: nanoToIso(record.timeUnixNano),
            timeNs,
            severity: typeof record.severityText === "string" ? record.severityText : "",
            traceId: typeof record.traceId === "string" ? record.traceId : "",
            spanId: typeof record.spanId === "string" ? record.spanId : "",
            attributes: Object.entries(attrsMap)
              .map(([key, value]) => `${key}=${value}`)
              .join(", "),
          });
        });
      });
    });
  });
  return rows;
}

async function loadAll() {
  setStatus("Loading...");
  try {
    const [traceEntries, metricEntries, logEntries] = await Promise.all([
      loadFile(FILES.traces),
      loadFile(FILES.metrics),
      loadFile(FILES.logs),
    ]);

    traceRows = flattenTraces(traceEntries);
    metricRows = flattenMetrics(metricEntries);
    logRows = flattenLogs(logEntries);

    renderTraces(traceRows);
    renderMetrics(metricRows);
    renderLogs(logRows);
    setStatus(`Loaded ${traceRows.length} spans, ${metricRows.length} metrics, ${logRows.length} logs`);
  } catch (err) {
    console.error(err);
    setStatus("Failed to load signals (check console)");
  }
}

async function loadTraces() {
  setStatus("Loading traces...");
  try {
    const entries = await loadFile(FILES.traces);
    traceRows = flattenTraces(entries);
    renderTraces(traceRows);
    setStatus(`Loaded ${traceRows.length} spans`);
  } catch (err) {
    console.error(err);
    setStatus("Failed to load traces (check console)");
  }
}

async function loadMetrics() {
  setStatus("Loading metrics...");
  try {
    const entries = await loadFile(FILES.metrics);
    metricRows = flattenMetrics(entries);
    renderMetrics(metricRows);
    setStatus(`Loaded ${metricRows.length} metrics`);
  } catch (err) {
    console.error(err);
    setStatus("Failed to load metrics (check console)");
  }
}

async function loadLogs() {
  setStatus("Loading logs...");
  try {
    const entries = await loadFile(FILES.logs);
    logRows = flattenLogs(entries);
    renderLogs(logRows);
    setStatus(`Loaded ${logRows.length} logs`);
  } catch (err) {
    console.error(err);
    setStatus("Failed to load logs (check console)");
  }
}

function setupLoadButtons() {
  const buttons = Array.from(document.querySelectorAll<HTMLButtonElement>("button[data-load]"));
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.load as TableKey | undefined;
      if (!target) {
        return;
      }
      if (target === "logs") {
        void loadLogs();
      } else if (target === "metrics") {
        void loadMetrics();
      } else if (target === "traces") {
        void loadTraces();
      }
    });
  });
}

function updateLimitDisabled(tableKey: TableKey) {
  const config = TABLE_CONFIG[tableKey];
  const { mode } = getTableFilter(tableKey);
  const limitInput = document.getElementById(config.limitInputId) as HTMLInputElement | null;
  if (limitInput) {
    limitInput.disabled = mode !== "last";
  }
}

function setupTableFilters(tableKey: TableKey, rerender: () => void) {
  const config = TABLE_CONFIG[tableKey];
  const radios = Array.from(
    document.querySelectorAll<HTMLInputElement>(`input[name='${config.filterModeName}']`),
  );
  const limitInput = document.getElementById(config.limitInputId) as HTMLInputElement | null;
  const table = document.getElementById(config.tableId);
  const filterInputs = Array.from(
    table?.querySelectorAll<HTMLInputElement>("thead .filter-row input[data-column]") || [],
  );

  const update = () => {
    updateLimitDisabled(tableKey);
    rerender();
  };

  radios.forEach((radio) => radio.addEventListener("change", update));
  limitInput?.addEventListener("change", update);
  filterInputs.forEach((input) => input.addEventListener("input", rerender));

  updateLimitDisabled(tableKey);
}

function setupFilters() {
  setupTableFilters("logs", () => renderLogs(logRows));
  setupTableFilters("metrics", () => renderMetrics(metricRows));
  setupTableFilters("traces", () => renderTraces(traceRows));
  reloadBtn?.addEventListener("click", loadAll);
  setupLoadButtons();
  loadAll();
}

setupFilters();
setupToggles();
