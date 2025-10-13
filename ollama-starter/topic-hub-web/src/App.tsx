import { FormEvent, useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { useChatSession } from "./hooks/useChatSession";
import { Message, SessionConfigForm } from "./types";

const defaultForm: SessionConfigForm = {
  topic_name: "",
  persist_root: ".chroma",
  collection_name: "",
  embedding_provider: "auto",
  embedding_model: "auto",
  chat_provider: "ollama",
  chat_model: "",
  chat_temperature: "0",
  k: "10",
  search_type: "similarity",
  lambda_mult: "",
  rerank_model: "BAAI/bge-reranker-base",
  rerank_top_n: "",
  oci_config: "oci-config.json",
  oci_endpoint: "",
  oci_compartment_id: "",
  oci_auth_profile: "",
  user_id: "default",
  show_sources: false,
  debug: false,
  graph_diagram: "",
};

const apiStorageKey = "topicHub.apiBaseUrl";

function MessageBubble({ message }: { message: Message }) {
  return (
    <div className={clsx("message", message.role)}>
      <div className="message-avatar">
        {message.role === "user" ? "U" : "AI"}
      </div>
      <div className="message-content">
        <div className="message-bubble">{message.content}</div>
        {message.role === "assistant" && message.sources && message.sources.length > 0 && (
          <div className="message-sources">
            <strong>Sources</strong>
            <ul>
              {message.sources.map((source, index) => (
                <li key={`${source.source}-${index}`}>
                  {source.source}
                  {source.page !== undefined && source.page !== null
                    ? ` (p${source.page})`
                    : ""}
                  {source.rerank_score !== undefined
                    ? ` — rerank ${source.rerank_score.toFixed(4)}`
                    : ""}
                  {source.similarity_score !== undefined
                    ? ` — sim ${source.similarity_score.toFixed(4)}`
                    : ""}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [form, setForm] = useState<SessionConfigForm>(defaultForm);
  const [apiBaseUrl, setApiBaseUrl] = useState(() => {
    return (
      localStorage.getItem(apiStorageKey) ||
      import.meta.env.VITE_API_BASE_URL ||
      "http://localhost:8000"
    );
  });
  const [question, setQuestion] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const chat = useChatSession(apiBaseUrl);

  useEffect(() => {
    localStorage.setItem(apiStorageKey, apiBaseUrl);
  }, [apiBaseUrl]);

  const allowSend = chat.session && question.trim().length > 0 && !chat.sending;

  const sessionMeta = useMemo(() => {
    if (!chat.session) {
      return null;
    }
    return {
      id: chat.session.session_id,
      topic: chat.session.topic_name,
      chatModel: chat.session.chat_model ?? "default",
      provider: chat.session.chat_provider ?? form.chat_provider,
      rerankTopN: chat.session.rerank_top_n,
    };
  }, [chat.session, form.chat_provider]);

  const handleFieldChange =
    (field: Exclude<keyof SessionConfigForm, "show_sources" | "debug">) =>
    (event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
      setForm((prev) => ({
        ...prev,
        [field]: event.target.value,
      }));
    };

  const handleCheckboxChange = (field: "show_sources" | "debug") => {
    return (event: React.ChangeEvent<HTMLInputElement>) => {
      setForm((prev) => ({
        ...prev,
        [field]: event.target.checked,
      }));
    };
  };

  const handleFormSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!form.topic_name.trim()) {
      alert("Topic name is required");
      return;
    }
    try {
      await chat.create(form);
      setQuestion("");
    } catch (error) {
      console.error(error);
    }
  };

  const handleSendMessage = async () => {
    const trimmed = question.trim();
    if (!trimmed || !chat.session) {
      return;
    }
    setQuestion("");
    await chat.send(trimmed);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (allowSend) {
        void handleSendMessage();
      }
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <header>
          <h1>Topic Hub Control</h1>
          <p className="status-text">
            Configure the session then start chatting with your topic-specific RAG
            assistant.
          </p>
        </header>
        <form className="session-form" onSubmit={handleFormSubmit}>
          <div className="form-field">
            <label htmlFor="api-base">API Base URL</label>
            <input
              id="api-base"
              value={apiBaseUrl}
              onChange={(event) => setApiBaseUrl(event.target.value)}
              placeholder="http://localhost:8000"
            />
          </div>
          <div className="form-field">
            <label htmlFor="topic">Topic Name *</label>
            <input
              id="topic"
              value={form.topic_name}
              onChange={handleFieldChange("topic_name")}
              placeholder="spring-boot"
              required
            />
          </div>
          <div className="form-row">
            <div className="form-field">
              <label htmlFor="embedding-provider">Embedding Provider</label>
              <select
                id="embedding-provider"
                value={form.embedding_provider}
                onChange={handleFieldChange("embedding_provider")}
              >
                <option value="auto">auto</option>
                <option value="ollama">ollama</option>
                <option value="oci">oci</option>
              </select>
            </div>
            <div className="form-field">
              <label htmlFor="chat-provider">Chat Provider</label>
              <select
                id="chat-provider"
                value={form.chat_provider}
                onChange={handleFieldChange("chat_provider")}
              >
                <option value="ollama">ollama</option>
                <option value="oci">oci</option>
              </select>
            </div>
          </div>
          <details open={advancedOpen} onToggle={(e) => setAdvancedOpen(e.currentTarget.open)}>
            <summary className="status-text">Advanced Options</summary>
            <div className="form-field">
              <label htmlFor="persist">Persist Root</label>
              <input
                id="persist"
                value={form.persist_root}
                onChange={handleFieldChange("persist_root")}
              />
            </div>
            <div className="form-field">
              <label htmlFor="collection">Collection Name</label>
              <input
                id="collection"
                value={form.collection_name}
                onChange={handleFieldChange("collection_name")}
              />
            </div>
            <div className="form-field">
              <label htmlFor="embedding-model">Embedding Model</label>
              <input
                id="embedding-model"
                value={form.embedding_model}
                onChange={handleFieldChange("embedding_model")}
              />
            </div>
            <div className="form-field">
              <label htmlFor="chat-model">Chat Model</label>
              <input
                id="chat-model"
                value={form.chat_model}
                onChange={handleFieldChange("chat_model")}
                placeholder="leave blank for provider default"
              />
            </div>
            <div className="form-row">
              <div className="form-field">
                <label htmlFor="temperature">Temperature</label>
                <input
                  id="temperature"
                  type="number"
                  step="0.1"
                  value={form.chat_temperature}
                  onChange={handleFieldChange("chat_temperature")}
                />
              </div>
              <div className="form-field">
                <label htmlFor="top-k">Top K</label>
                <input
                  id="top-k"
                  type="number"
                  min={1}
                  value={form.k}
                  onChange={handleFieldChange("k")}
                />
              </div>
            </div>
            <div className="form-row">
              <div className="form-field">
                <label htmlFor="search-type">Search Type</label>
                <select
                  id="search-type"
                  value={form.search_type}
                  onChange={handleFieldChange("search_type")}
                >
                  <option value="similarity">similarity</option>
                  <option value="mmr">mmr</option>
                </select>
              </div>
              <div className="form-field">
                <label htmlFor="lambda">Lambda Mult</label>
                <input
                  id="lambda"
                  value={form.lambda_mult}
                  onChange={handleFieldChange("lambda_mult")}
                  placeholder="0.5"
                />
              </div>
            </div>
            <div className="form-field">
              <label htmlFor="rerank-model">Rerank Model</label>
              <input
                id="rerank-model"
                value={form.rerank_model}
                onChange={handleFieldChange("rerank_model")}
              />
            </div>
            <div className="form-field">
              <label htmlFor="rerank-top-n">Rerank Top N</label>
              <input
                id="rerank-top-n"
                value={form.rerank_top_n}
                onChange={handleFieldChange("rerank_top_n")}
                placeholder="auto"
              />
            </div>
            <div className="form-field">
              <label htmlFor="oci-config">OCI Config Path</label>
              <input
                id="oci-config"
                value={form.oci_config}
                onChange={handleFieldChange("oci_config")}
              />
            </div>
            <div className="form-field">
              <label htmlFor="oci-endpoint">OCI Endpoint</label>
              <input
                id="oci-endpoint"
                value={form.oci_endpoint}
                onChange={handleFieldChange("oci_endpoint")}
              />
            </div>
            <div className="form-field">
              <label htmlFor="oci-compartment">OCI Compartment ID</label>
              <input
                id="oci-compartment"
                value={form.oci_compartment_id}
                onChange={handleFieldChange("oci_compartment_id")}
              />
            </div>
            <div className="form-field">
              <label htmlFor="oci-profile">OCI Auth Profile</label>
              <input
                id="oci-profile"
                value={form.oci_auth_profile}
                onChange={handleFieldChange("oci_auth_profile")}
              />
            </div>
            <div className="form-field">
              <label htmlFor="user-id">User ID</label>
              <input
                id="user-id"
                value={form.user_id}
                onChange={handleFieldChange("user_id")}
              />
            </div>
            <div className="form-field">
              <label htmlFor="graph-diagram">Graph Diagram Output</label>
              <input
                id="graph-diagram"
                value={form.graph_diagram}
                onChange={handleFieldChange("graph_diagram")}
                placeholder="graph.png"
              />
            </div>
            <label className="checkbox-field">
              <input
                type="checkbox"
                checked={form.show_sources}
                onChange={handleCheckboxChange("show_sources")}
              />
              Show sources in responses
            </label>
            <label className="checkbox-field">
              <input
                type="checkbox"
                checked={form.debug}
                onChange={handleCheckboxChange("debug")}
              />
              Enable debug logging
            </label>
          </details>
          <button
            className="primary-button"
            type="submit"
            disabled={chat.creating}
          >
            {chat.creating ? "Creating session..." : "Start session"}
          </button>
        </form>
        {sessionMeta && (
          <section className="status-text">
            <p>
              <strong>Session</strong> {sessionMeta.id.slice(0, 8)}…
            </p>
            <p>
              Topic <strong>{sessionMeta.topic}</strong> using{" "}
              {sessionMeta.provider}:{sessionMeta.chatModel}
            </p>
            <p>Rerank top N: {sessionMeta.rerankTopN}</p>
            <button className="primary-button" onClick={chat.reset}>
              Reset session
            </button>
          </section>
        )}
        {chat.error && <div className="error-banner">{chat.error}</div>}
      </aside>
      <main className="chat-pane">
        <div className="chat-scroll">
          {chat.messages.length === 0 ? (
            <div className="empty-state">
              <h2>Start a new conversation</h2>
              <p>
                Choose your topic and create a session on the left. Messages will appear
                here once you begin chatting.
              </p>
            </div>
          ) : (
            chat.messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))
          )}
        </div>
        <div className="composer">
          {!chat.session ? (
            <div className="status-text">
              Create a session to enable the chat composer.
            </div>
          ) : (
            <>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask something about the selected topic..."
                disabled={chat.sending}
              />
              <div className="composer-actions">
                <div className="status-text">
                  {chat.sending
                    ? "Waiting for the assistant..."
                    : "Shift + Enter for a new line"}
                </div>
                <button
                  className="primary-button"
                  onClick={handleSendMessage}
                  disabled={!allowSend}
                >
                  Send
                </button>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}
