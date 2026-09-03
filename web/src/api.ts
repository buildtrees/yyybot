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
  type: "model_start" | "model_delta" | "model_end" | "tool_start" | "tool_end" | "final" | "cancelled" | "error";
  data: Record<string, unknown>;
}

export interface VoiceOption {
  voice_id: string;
  name: string;
  custom: boolean;
}

export interface AudioConfig {
  transcription_enabled: boolean;
  synthesis_enabled: boolean;
  streaming_enabled: boolean;
  sample_rate: number | null;
  voice_upload_enabled: boolean;
  default_voice: string | null;
  voices: VoiceOption[];
}

export interface SpeechStream {
  sendText: (text: string) => void;
  finish: () => void;
  close: () => void;
}

async function responseError(response: Response): Promise<Error> {
  let detail = `${response.status} ${response.statusText}`;
  try {
    const body = (await response.json()) as { detail?: string };
    detail = body.detail ?? detail;
  } catch {
    // Keep the status text when the server did not return JSON.
  }
  return new Error(detail);
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
    throw await responseError(response);
  }
  return response.json() as Promise<T>;
}

export const api = {
  audioConfig: () => request<AudioConfig>("/api/audio/config"),
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
  cancelRun: (runId: string) =>
    request<{ run_id: string; status: "cancelled" | "completed" }>(
      `/api/runs/${encodeURIComponent(runId)}`,
      { method: "DELETE" },
    ),
  transcribeAudio: async (
    audio: Blob,
    filename: string,
    language = "zh",
  ) => {
    const query = new URLSearchParams({ filename, language });
    const response = await fetch(`/api/audio/transcriptions?${query}`, {
      method: "POST",
      headers: { "Content-Type": audio.type || "audio/webm" },
      body: audio,
    });
    if (!response.ok) throw await responseError(response);
    return response.json() as Promise<{ text: string }>;
  },
  synthesizeSpeech: async (text: string, voice: string) => {
    const response = await fetch("/api/audio/speech", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, voice }),
    });
    if (!response.ok) throw await responseError(response);
    return response.blob();
  },
  uploadVoice: async (
    audio: File,
    name: string,
    referenceText: string,
  ) => {
    const query = new URLSearchParams({
      name,
      filename: audio.name,
      consent_confirmed: "true",
    });
    if (referenceText.trim()) query.set("reference_text", referenceText.trim());
    const response = await fetch(`/api/audio/voices?${query}`, {
      method: "POST",
      headers: { "Content-Type": audio.type || "audio/wav" },
      body: audio,
    });
    if (!response.ok) throw await responseError(response);
    return response.json() as Promise<VoiceOption>;
  },
  deleteVoice: async (voiceId: string) => {
    const response = await fetch(`/api/audio/voices/${encodeURIComponent(voiceId)}`, {
      method: "DELETE",
    });
    if (!response.ok) throw await responseError(response);
  },
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

export function openSpeechStream(
  voice: string,
  onChunk: (pcm: ArrayBuffer, sampleRate: number) => void,
  onDone: () => void,
  onError: (message: string) => void,
): SpeechStream {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(
    `${protocol}//${window.location.host}/api/audio/speech/stream`,
  );
  socket.binaryType = "arraybuffer";
  const pending: string[] = [];
  let sampleRate = 0;
  let terminal = false;
  let manuallyClosed = false;

  function send(message: string) {
    if (socket.readyState === WebSocket.OPEN) socket.send(message);
    else if (socket.readyState === WebSocket.CONNECTING) pending.push(message);
  }

  socket.onopen = () => {
    socket.send(JSON.stringify({ type: "start", voice }));
    pending.splice(0).forEach((message) => socket.send(message));
  };
  socket.onmessage = (event) => {
    if (event.data instanceof ArrayBuffer) {
      if (sampleRate) onChunk(event.data, sampleRate);
      return;
    }
    const message = JSON.parse(String(event.data)) as {
      type?: string;
      sample_rate?: number;
      message?: string;
    };
    if (message.type === "ready") {
      sampleRate = Number(message.sample_rate ?? 0);
    } else if (message.type === "done") {
      terminal = true;
      onDone();
      socket.close();
    } else if (message.type === "error") {
      terminal = true;
      onError(message.message || "流式语音合成失败");
      socket.close();
    }
  };
  socket.onerror = () => {
    if (!terminal && !manuallyClosed) {
      terminal = true;
      onError("流式语音连接失败");
    }
  };
  socket.onclose = () => {
    if (!terminal && !manuallyClosed) {
      terminal = true;
      onError("流式语音连接意外关闭");
    }
  };

  return {
    sendText: (text: string) => {
      if (text) send(JSON.stringify({ type: "text", text }));
    },
    finish: () => send(JSON.stringify({ type: "end" })),
    close: () => {
      manuallyClosed = true;
      pending.length = 0;
      socket.close();
    },
  };
}
