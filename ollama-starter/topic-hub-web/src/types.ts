export type ChatRole = "user" | "assistant";

export interface Message {
  id: string;
  role: ChatRole;
  content: string;
  sources?: ChatSource[];
}

export interface ChatSource {
  source: string;
  page?: number | string;
  rerank_score?: number;
  similarity_score?: number;
}

export interface SessionConfigForm {
  topic_name: string;
  persist_root: string;
  collection_name: string;
  embedding_provider: "auto" | "ollama" | "oci";
  embedding_model: string;
  chat_provider: "ollama" | "oci";
  chat_model: string;
  chat_temperature: string;
  k: string;
  search_type: "similarity" | "mmr";
  lambda_mult: string;
  rerank_model: string;
  rerank_top_n: string;
  oci_config: string;
  oci_endpoint: string;
  oci_compartment_id: string;
  oci_auth_profile: string;
  user_id: string;
  show_sources: boolean;
  debug: boolean;
  graph_diagram: string;
}

export interface SessionCreateResponse {
  session_id: string;
  topic_name: string;
  chat_provider?: string | null;
  chat_model?: string | null;
  rerank_top_n: number;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  sources: ChatSource[];
}
