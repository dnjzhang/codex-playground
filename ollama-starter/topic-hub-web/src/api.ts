import { ChatResponse, SessionConfigForm, SessionCreateResponse } from "./types";

export interface ApiClientOptions {
  baseUrl?: string;
}

const DEFAULT_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, "") || "http://localhost:8000";

const toFloat = (value: string, fallback: number) => {
  const trimmed = value.trim();
  if (!trimmed) {
    return fallback;
  }
  const parsed = Number.parseFloat(trimmed);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const toInt = (value: string, fallback: number) => {
  const trimmed = value.trim();
  if (!trimmed) {
    return fallback;
  }
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
};

function normalizeConfig(form: SessionConfigForm) {
  const payload: Record<string, unknown> = {
    topic_name: form.topic_name.trim(),
    persist_root: form.persist_root.trim() || ".chroma",
    embedding_provider: form.embedding_provider,
    embedding_model: form.embedding_model.trim() || "auto",
    chat_provider: form.chat_provider,
    chat_temperature: toFloat(form.chat_temperature, 0),
    k: toInt(form.k, 10),
    search_type: form.search_type,
    rerank_model: form.rerank_model.trim() || "BAAI/bge-reranker-base",
    oci_config: form.oci_config.trim() || "oci-config.json",
    user_id: form.user_id.trim() || "default",
    show_sources: form.show_sources,
    debug: form.debug,
  };

  const collectionName = form.collection_name.trim();
  if (collectionName) {
    payload.collection_name = collectionName;
  }

  const chatModel = form.chat_model.trim();
  if (chatModel) {
    payload.chat_model = chatModel;
  }

  const lambdaMult = form.lambda_mult.trim();
  if (lambdaMult) {
    const parsed = Number.parseFloat(lambdaMult);
    if (Number.isFinite(parsed)) {
      payload.lambda_mult = parsed;
    }
  }

  const rerankTopN = form.rerank_top_n.trim();
  if (rerankTopN) {
    const parsed = Number.parseInt(rerankTopN, 10);
    if (Number.isFinite(parsed)) {
      payload.rerank_top_n = parsed;
    }
  }

  const ociEndpoint = form.oci_endpoint.trim();
  if (ociEndpoint) {
    payload.oci_endpoint = ociEndpoint;
  }

  const ociCompartment = form.oci_compartment_id.trim();
  if (ociCompartment) {
    payload.oci_compartment_id = ociCompartment;
  }

  const ociProfile = form.oci_auth_profile.trim();
  if (ociProfile) {
    payload.oci_auth_profile = ociProfile;
  }

  const graphDiagram = form.graph_diagram.trim();
  if (graphDiagram) {
    payload.graph_diagram = graphDiagram;
  }

  return payload;
}

async function request<TResponse>(
  path: string,
  init: RequestInit,
  options?: ApiClientOptions,
): Promise<TResponse> {
  const baseUrl = (options?.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
  const response = await fetch(`${baseUrl}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init.headers || {}),
    },
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    let message = response.statusText;
    if (detail?.detail) {
      if (Array.isArray(detail.detail)) {
        message = detail.detail
          .map((item) => {
            const path = Array.isArray(item.loc) ? item.loc.join(".") : "";
            return path ? `${path}: ${item.msg}` : item.msg;
          })
          .join("; ");
      } else if (typeof detail.detail === "string") {
        message = detail.detail;
      }
    }
    throw new Error(`API error ${response.status}: ${message}`);
  }
  return response.json() as Promise<TResponse>;
}

export async function createSession(
  form: SessionConfigForm,
  options?: ApiClientOptions,
): Promise<SessionCreateResponse> {
  const payload = normalizeConfig(form);
  return request<SessionCreateResponse>(
    "/sessions",
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
    options,
  );
}

export async function sendChatMessage(
  sessionId: string,
  question: string,
  options?: ApiClientOptions,
): Promise<ChatResponse> {
  return request<ChatResponse>(
    `/sessions/${sessionId}/chat`,
    {
      method: "POST",
      body: JSON.stringify({ question }),
    },
    options,
  );
}
