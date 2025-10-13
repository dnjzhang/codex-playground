import { useCallback, useMemo, useReducer } from "react";
import { createSession, sendChatMessage } from "../api";
import { Message, SessionConfigForm, SessionCreateResponse } from "../types";

interface ChatState {
  session: SessionCreateResponse | null;
  messages: Message[];
  creating: boolean;
  sending: boolean;
  error: string | null;
}

type Action =
  | { type: "RESET" }
  | { type: "SET_CREATING"; creating: boolean }
  | { type: "SET_SESSION"; session: SessionCreateResponse }
  | { type: "SET_ERROR"; error: string | null }
  | { type: "ADD_MESSAGE"; message: Message }
  | { type: "SET_SENDING"; sending: boolean };

const initialState: ChatState = {
  session: null,
  messages: [],
  creating: false,
  sending: false,
  error: null,
};

function reducer(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case "RESET":
      return { ...initialState };
    case "SET_CREATING":
      return { ...state, creating: action.creating, error: null };
    case "SET_SESSION":
      return { ...state, session: action.session, messages: [], error: null };
    case "SET_ERROR":
      return { ...state, error: action.error };
    case "ADD_MESSAGE":
      return { ...state, messages: [...state.messages, action.message] };
    case "SET_SENDING":
      return { ...state, sending: action.sending, error: null };
    default:
      return state;
  }
}

export function useChatSession(apiBaseUrl: string) {
  const [state, dispatch] = useReducer(reducer, initialState);

  const create = useCallback(
    async (config: SessionConfigForm) => {
      dispatch({ type: "SET_CREATING", creating: true });
      try {
        const session = await createSession(config, { baseUrl: apiBaseUrl });
        dispatch({ type: "SET_SESSION", session });
        return session;
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Failed to create session";
        dispatch({ type: "SET_ERROR", error: message });
        throw error;
      } finally {
        dispatch({ type: "SET_CREATING", creating: false });
      }
    },
    [apiBaseUrl],
  );

  const send = useCallback(
    async (question: string) => {
      if (!state.session) {
        throw new Error("Session is not established");
      }
      dispatch({
        type: "ADD_MESSAGE",
        message: {
          id: crypto.randomUUID(),
          role: "user",
          content: question,
        },
      });
      dispatch({ type: "SET_SENDING", sending: true });
      try {
        const response = await sendChatMessage(
          state.session.session_id,
          question,
          { baseUrl: apiBaseUrl },
        );
        dispatch({
          type: "ADD_MESSAGE",
          message: {
            id: crypto.randomUUID(),
            role: "assistant",
            content: response.answer,
            sources: response.sources,
          },
        });
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Failed to send message";
        dispatch({ type: "SET_ERROR", error: message });
      } finally {
        dispatch({ type: "SET_SENDING", sending: false });
      }
    },
    [apiBaseUrl, state.session],
  );

  const reset = useCallback(() => {
    dispatch({ type: "RESET" });
  }, []);

  return useMemo(
    () => ({
      ...state,
      create,
      send,
      reset,
    }),
    [state, create, send, reset],
  );
}
