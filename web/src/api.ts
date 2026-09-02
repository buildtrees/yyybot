export type Role = "system" | "user" | "assistant" | "tool";

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface Message {
  role: Role;
  content: string;
  tool_calls: ToolCall[];
  tool_call_id: string | null;
  reasoning_content: string;
  reasoning_signature: string | null;
}

export interface Workspace {
  workspace_id: string;
  name: string;
  workspace_type: "personal" | "team";
  created_at: string;
  owner_account_id: string | null;
}

export interface SessionSummary {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  turn_count: number;
}

export interface SessionDetail extends SessionSummary {
  system_prompt: string | null;
  messages: Message[];
}

export interface AgentResult {
  final_message: Message;
  messages: Message[];
  model_turns: number;
  usage_by_turn: Record<string, number>[];
  usage: Record<string, number>;
}

export interface RunEvent {
  type: "model_start" | "model_delta" | "model_end" | "tool_start" | "tool_end" | "final" | "error";
  data: Record<string, unknown>;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      // Keep the status text when the server did not return JSON.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  listWorkspaces: () => request<Workspace[]>("/api/workspaces"),
  createWorkspace: (name: string) =>
    request<Workspace>("/api/workspaces", {
      method: "POST",
      body: JSON.stringify({ name }),
    }),
  listSessions: (workspaceId: string) =>
    request<SessionSummary[]>(`/api/workspaces/${workspaceId}/sessions`),
  createSession: (workspaceId: string, title: string) =>
    request<SessionDetail>(`/api/workspaces/${workspaceId}/sessions`, {
      method: "POST",
      body: JSON.stringify({ title }),
    }),
  getSession: (workspaceId: string, sessionId: string) =>
    request<SessionDetail>(
      `/api/workspaces/${workspaceId}/sessions/${sessionId}`,
    ),
  startRun: (workspaceId: string, sessionId: string, prompt: string) =>
    request<{ run_id: string; events_url: string }>(
      `/api/workspaces/${workspaceId}/sessions/${sessionId}/runs`,
      { method: "POST", body: JSON.stringify({ prompt }) },
    ),
};

export function streamRun(
  url: string,
  onEvent: (event: RunEvent) => void,
  onConnectionError: () => void,
): () => void {
  const source = new EventSource(url);
  source.onmessage = (message) => {
    onEvent(JSON.parse(message.data) as RunEvent);
  };
  source.onerror = () => {
    source.close();
    onConnectionError();
  };
  return () => source.close();
}
