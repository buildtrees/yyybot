import {
  FormEvent,
  KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import Markdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import {
  AgentResult,
  AudioConfig,
  Message,
  RunEvent,
  SessionDetail,
  SessionSummary,
  SpeechStream,
  Workspace,
  api,
  openSpeechStream,
  streamRun,
} from "./api";

type Activity = { key: string; label: string; state: "active" | "done" };
type RecordingSegment = {
  recorder: MediaRecorder;
  chunks: Blob[];
  id: number;
  send: boolean;
};

const emptyAudioConfig: AudioConfig = {
  transcription_enabled: false,
  synthesis_enabled: false,
  streaming_enabled: false,
  sample_rate: null,
  voice_upload_enabled: false,
  default_voice: null,
  voices: [],
};

const SILENCE_TIMEOUT_MS = 1_500;
const MIN_SPEECH_MS = 300;
const VOICE_START_MS = 120;
const VOICE_RMS_THRESHOLD = 0.018;
const IDLE_AUDIO_FLUSH_MS = 10_000;

function recordingMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
  return candidates.find((item) => MediaRecorder.isTypeSupported(item)) ?? "";
}

function recordingFilename(contentType: string) {
  if (contentType.includes("mp4")) return "recording.mp4";
  if (contentType.includes("ogg")) return "recording.ogg";
  return "recording.webm";
}

function appendVoiceDraft(current: string, addition: string) {
  const before = current.trim();
  const after = addition.trim();
  if (!before) return after;
  if (!after) return before;
  const separator = /[，。！？!?；;：:\s]$/.test(before) ? "" : "，";
  return `${before}${separator}${after}`;
}

function splitSpeechBuffer(text: string, force: boolean) {
  const chunks: string[] = [];
  let remainder = text;
  while (remainder) {
    let boundary = -1;
    for (let index = 5; index < remainder.length; index += 1) {
      if (/[。！？!?；;\n]/.test(remainder[index])) {
        boundary = index + 1;
        break;
      }
    }
    if (boundary < 0 && remainder.length >= 18) {
      for (let index = 17; index >= 9; index -= 1) {
        if (/[，,：:\s]/.test(remainder[index])) {
          boundary = index + 1;
          break;
        }
      }
      if (boundary < 0) boundary = 18;
    }
    if (boundary < 0) break;
    const chunk = remainder.slice(0, boundary).trim();
    if (chunk) chunks.push(chunk);
    remainder = remainder.slice(boundary);
  }
  if (force && remainder.trim()) {
    chunks.push(remainder.trim());
    remainder = "";
  }
  return { chunks, remainder };
}

function MarkdownContent({ content }: { content: string }) {
  return (
    <div className="message-content markdown-body">
      <Markdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          a: ({ node: _node, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer noopener" />
          ),
        }}
      >
        {content}
      </Markdown>
    </div>
  );
}

function Icon({ name }: { name: "plus" | "send" | "menu" | "spark" | "mic" | "stop" | "volume" | "trash" }) {
  const paths = {
    plus: <path d="M12 5v14M5 12h14" />,
    send: <path d="m4 4 16 8-16 8 3-8-3-8Zm3 8h13" />,
    menu: <path d="M4 7h16M4 12h16M4 17h16" />,
    spark: <path d="m12 3 1.3 4.2L17 9l-3.7 1.8L12 15l-1.3-4.2L7 9l3.7-1.8L12 3Zm6 11 .7 2.3L21 17l-2.3.7L18 20l-.7-2.3L15 17l2.3-.7L18 14Z" />,
    mic: <path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3Zm-6 9a6 6 0 0 0 12 0M12 18v3M9 21h6" />,
    stop: <path d="M7 7h10v10H7z" />,
    volume: <path d="M11 6 7 10H4v4h3l4 4V6Zm4 4a3 3 0 0 1 0 4m2-6a6 6 0 0 1 0 8" />,
    trash: <path d="M4 7h16M9 7V4h6v3m-9 0 1 14h10l1-14M10 11v6m4-6v6" />,
  };
  return <svg viewBox="0 0 24 24" aria-hidden="true">{paths[name]}</svg>;
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

function MessageView({ message }: { message: Message }) {
  if (message.role === "tool") {
    let preview = message.content;
    try {
      const parsed = JSON.parse(message.content) as { result?: unknown; error?: unknown };
      preview = JSON.stringify(parsed.result ?? parsed.error ?? parsed, null, 2);
    } catch {
      // Tool output is allowed to be plain text.
    }
    return (
      <details className="tool-result">
        <summary>工具结果 · {message.tool_call_id?.slice(0, 8)}</summary>
        <pre>{preview}</pre>
      </details>
    );
  }
  if (!message.content && message.tool_calls.length) {
    return (
      <div className="tool-call-note">
        调用工具：{message.tool_calls.map((call) => call.name).join("、")}
      </div>
    );
  }
  if ((!message.content && !message.reasoning_content) || message.role === "system") return null;
  return (
    <article className={`message ${message.role}`}>
      <div className="message-role">{message.role === "user" ? "你" : "yyybot"}</div>
      {message.reasoning_content && <details className="reasoning-block">
        <summary>思考过程</summary>
        <div>{message.reasoning_content}</div>
      </details>}
      <MarkdownContent content={message.content} />
    </article>
  );
}

function EmptyChat({ onCreate }: { onCreate: () => void }) {
  return (
    <div className="empty-state">
      <div className="empty-mark"><Icon name="spark" /></div>
      <p className="eyebrow">YOUR PERSONAL AGENT</p>
      <h1>从一个清晰的问题开始。</h1>
      <p>会话会完整记录模型轮次、工具轨迹和 token 使用，并保存在当前 Workspace。</p>
      <button className="primary compact" onClick={onCreate}>
        <Icon name="plus" /> 创建第一个会话
      </button>
    </div>
  );
}

export default function App() {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [workspaceId, setWorkspaceId] = useState("");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [session, setSession] = useState<SessionDetail | null>(null);
  const [prompt, setPrompt] = useState("");
  const [running, setRunning] = useState(false);
  const [activities, setActivities] = useState<Activity[]>([]);
  const [liveAnswer, setLiveAnswer] = useState("");
  const [liveReasoning, setLiveReasoning] = useState("");
  const [lastResult, setLastResult] = useState<AgentResult | null>(null);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [workspaceForm, setWorkspaceForm] = useState(false);
  const [sessionForm, setSessionForm] = useState(false);
  const [newName, setNewName] = useState("");
  const [newTitle, setNewTitle] = useState("");
  const [audioConfig, setAudioConfig] = useState<AudioConfig>(emptyAudioConfig);
  const [selectedVoice, setSelectedVoice] = useState("");
  const [autoSpeak, setAutoSpeak] = useState(
    () => window.localStorage.getItem("yyybot.autoSpeak") !== "false",
  );
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [voiceForm, setVoiceForm] = useState(false);
  const [voiceName, setVoiceName] = useState("");
  const [voiceReferenceText, setVoiceReferenceText] = useState("");
  const [voiceFile, setVoiceFile] = useState<File | null>(null);
  const [voiceConsent, setVoiceConsent] = useState(false);
  const [uploadingVoice, setUploadingVoice] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const streamCloser = useRef<null | (() => void)>(null);
  const activeRunIdRef = useRef<string | null>(null);
  const activePromptRef = useRef("");
  const runGenerationRef = useRef(0);
  const runningRef = useRef(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const microphoneStreamRef = useRef<MediaStream | null>(null);
  const recordingRef = useRef(false);
  const vadAudioContextRef = useRef<AudioContext | null>(null);
  const vadSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const vadFrameRef = useRef<number | null>(null);
  const segmentHasSpeechRef = useRef(false);
  const voiceCandidateAtRef = useRef(0);
  const segmentStartedAtRef = useRef(0);
  const lastVoiceAtRef = useRef(0);
  const lastAudioFlushAtRef = useRef(0);
  const latestSpeechSegmentRef = useRef(0);
  const activeRecordingSegmentRef = useRef<RecordingSegment | null>(null);
  const transcriptionsInFlightRef = useRef(0);
  const transcriptionQueueRef = useRef<Promise<void>>(Promise.resolve());
  const cancellationPromiseRef = useRef<Promise<void> | null>(null);
  const voiceDraftRef = useRef("");
  const voiceTurnOpenRef = useRef(false);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  const playbackRequestRef = useRef(0);
  const speechTextQueueRef = useRef<string[]>([]);
  const speechAudioQueueRef = useRef<string[]>([]);
  const speechSynthesisActiveRef = useRef(false);
  const speechBufferRef = useRef("");
  const speechReceivedRef = useRef(false);
  const speechVoiceRef = useRef("");
  const speechStreamRef = useRef<SpeechStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const speechSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const nextSpeechTimeRef = useRef(0);
  const speechStreamEndedRef = useRef(false);

  const selectedWorkspace = useMemo(
    () => workspaces.find((item) => item.workspace_id === workspaceId),
    [workspaces, workspaceId],
  );
  const selectedVoiceOption = audioConfig.voices.find(
    (voice) => voice.voice_id === selectedVoice,
  );

  function updateRunning(value: boolean) {
    runningRef.current = value;
    setRunning(value);
  }

  function stopVadMonitor() {
    if (vadFrameRef.current !== null) {
      cancelAnimationFrame(vadFrameRef.current);
      vadFrameRef.current = null;
    }
    vadSourceRef.current?.disconnect();
    vadSourceRef.current = null;
    const context = vadAudioContextRef.current;
    vadAudioContextRef.current = null;
    if (context) void context.close();
  }

  const loadWorkspaces = useCallback(async () => {
    const items = await api.listWorkspaces();
    setWorkspaces(items);
    setWorkspaceId((current) => current || items[0]?.workspace_id || "");
  }, []);

  const loadSessions = useCallback(async (selected: string) => {
    const items = await api.listSessions(selected);
    setSessions(items);
    return items;
  }, []);

  const loadSession = useCallback(async (selectedWorkspaceId: string, sessionId: string, clearResult = true) => {
    setError("");
    setSession(await api.getSession(selectedWorkspaceId, sessionId));
    if (clearResult) setLastResult(null);
    setSidebarOpen(false);
  }, []);

  useEffect(() => {
    loadWorkspaces().catch((reason: Error) => setError(reason.message));
    api.audioConfig()
      .then((config) => {
        setAudioConfig(config);
        const storedVoice = window.localStorage.getItem("yyybot.voice");
        const voice = config.voices.some((item) => item.voice_id === storedVoice)
          ? storedVoice
          : config.default_voice;
        setSelectedVoice(voice ?? "");
        setAutoSpeak(
          config.synthesis_enabled
          && window.localStorage.getItem("yyybot.autoSpeak") !== "false"
        );
      })
      .catch(() => setAudioConfig(emptyAudioConfig));
    return () => {
      streamCloser.current?.();
      recordingRef.current = false;
      stopVadMonitor();
      const recorder = mediaRecorderRef.current;
      if (recorder) {
        recorder.ondataavailable = null;
        recorder.onstop = null;
        if (recorder.state !== "inactive") recorder.stop();
      }
      microphoneStreamRef.current?.getTracks().forEach((track) => track.stop());
      audioPlayerRef.current?.pause();
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
      speechStreamRef.current?.close();
      speechSourcesRef.current.forEach((source) => source.stop());
      void audioContextRef.current?.close();
    };
  }, [loadWorkspaces]);

  useEffect(() => {
    if (selectedVoice) window.localStorage.setItem("yyybot.voice", selectedVoice);
  }, [selectedVoice]);

  useEffect(() => {
    window.localStorage.setItem("yyybot.autoSpeak", String(autoSpeak));
  }, [autoSpeak]);

  useEffect(() => {
    if (!workspaceId) {
      setSessions([]);
      setSession(null);
      return;
    }
    setSession(null);
    setLastResult(null);
    loadSessions(workspaceId)
      .then((items) => {
        if (items[0]) return loadSession(workspaceId, items[0].session_id);
      })
      .catch((reason: Error) => setError(reason.message));
  }, [workspaceId, loadSessions, loadSession]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [session?.messages, activities]);

  async function createWorkspace(event: FormEvent) {
    event.preventDefault();
    if (!newName.trim()) return;
    try {
      const created = await api.createWorkspace(newName.trim());
      setWorkspaces((items) => [created, ...items]);
      setWorkspaceId(created.workspace_id);
      setNewName("");
      setWorkspaceForm(false);
    } catch (reason) {
      setError((reason as Error).message);
    }
  }

  async function createSession(event?: FormEvent) {
    event?.preventDefault();
    if (!workspaceId) return;
    try {
      const created = await api.createSession(workspaceId, newTitle.trim() || "新会话");
      setSession(created);
      setSessions((items) => [created, ...items]);
      setNewTitle("");
      setSessionForm(false);
      setSidebarOpen(false);
    } catch (reason) {
      setError((reason as Error).message);
    }
  }

  function stopSpeaking() {
    playbackRequestRef.current += 1;
    speechStreamRef.current?.close();
    speechStreamRef.current = null;
    speechSourcesRef.current.forEach((source) => {
      try {
        source.stop();
      } catch {
        // A source that already ended does not need to be stopped again.
      }
    });
    speechSourcesRef.current.clear();
    nextSpeechTimeRef.current = 0;
    speechStreamEndedRef.current = false;
    audioPlayerRef.current?.pause();
    audioPlayerRef.current = null;
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
    speechAudioQueueRef.current.forEach((url) => URL.revokeObjectURL(url));
    speechAudioQueueRef.current = [];
    speechTextQueueRef.current = [];
    speechBufferRef.current = "";
    speechReceivedRef.current = false;
    setSpeaking(false);
  }

  function scheduleStreamingAudio(
    pcm: ArrayBuffer,
    sampleRate: number,
    requestId: number,
  ) {
    if (requestId !== playbackRequestRef.current || !pcm.byteLength) return;
    const context = audioContextRef.current ?? new AudioContext();
    audioContextRef.current = context;
    void context.resume();
    const view = new DataView(pcm);
    const frameCount = Math.floor(pcm.byteLength / 2);
    const buffer = context.createBuffer(1, frameCount, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let index = 0; index < frameCount; index += 1) {
      channel[index] = view.getInt16(index * 2, true) / 32768;
    }
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    const startAt = Math.max(nextSpeechTimeRef.current, context.currentTime + 0.04);
    nextSpeechTimeRef.current = startAt + buffer.duration;
    speechSourcesRef.current.add(source);
    source.onended = () => {
      speechSourcesRef.current.delete(source);
      if (
        requestId === playbackRequestRef.current
        && speechStreamEndedRef.current
        && speechSourcesRef.current.size === 0
      ) {
        setSpeaking(false);
      }
    };
    source.start(startAt);
    setSpeaking(true);
  }

  function beginStreamingSpeech(voice: string) {
    if (!voice || speechStreamRef.current) return;
    const requestId = playbackRequestRef.current;
    const context = audioContextRef.current ?? new AudioContext();
    audioContextRef.current = context;
    void context.resume();
    nextSpeechTimeRef.current = context.currentTime;
    speechStreamEndedRef.current = false;
    setSpeaking(true);
    speechStreamRef.current = openSpeechStream(
      voice,
      (pcm, sampleRate) => scheduleStreamingAudio(pcm, sampleRate, requestId),
      () => {
        if (requestId !== playbackRequestRef.current) return;
        speechStreamRef.current = null;
        speechStreamEndedRef.current = true;
        if (speechSourcesRef.current.size === 0) setSpeaking(false);
      },
      (message) => {
        if (requestId !== playbackRequestRef.current) return;
        stopSpeaking();
        setError(message);
      },
    );
  }

  function playNextSpeechChunk(requestId: number) {
    if (
      requestId !== playbackRequestRef.current
      || audioPlayerRef.current
    ) return;
    const url = speechAudioQueueRef.current.shift();
    if (!url) {
      if (!speechSynthesisActiveRef.current && !speechTextQueueRef.current.length) {
        setSpeaking(false);
      }
      return;
    }
    const player = new Audio(url);
    audioUrlRef.current = url;
    audioPlayerRef.current = player;
    const finish = () => {
      if (requestId !== playbackRequestRef.current) return;
      URL.revokeObjectURL(url);
      audioUrlRef.current = null;
      audioPlayerRef.current = null;
      playNextSpeechChunk(requestId);
    };
    player.onended = finish;
    player.onerror = () => {
      if (requestId === playbackRequestRef.current) {
        stopSpeaking();
        setError("语音播放失败");
      }
    };
    void player.play().catch((reason: Error) => {
      if (requestId === playbackRequestRef.current) {
        stopSpeaking();
        setError(reason.message || "语音播放失败");
      }
    });
  }

  async function pumpSpeechSynthesis(requestId: number) {
    if (speechSynthesisActiveRef.current) return;
    speechSynthesisActiveRef.current = true;
    try {
      while (
        requestId === playbackRequestRef.current
        && speechTextQueueRef.current.length
      ) {
        const text = speechTextQueueRef.current.shift();
        if (!text) continue;
        const blob = await api.synthesizeSpeech(text, speechVoiceRef.current);
        if (requestId !== playbackRequestRef.current) return;
        speechAudioQueueRef.current.push(URL.createObjectURL(blob));
        playNextSpeechChunk(requestId);
      }
    } catch (reason) {
      if (requestId === playbackRequestRef.current) {
        stopSpeaking();
        setError((reason as Error).message || "语音合成失败");
      }
    } finally {
      speechSynthesisActiveRef.current = false;
      if (requestId === playbackRequestRef.current) {
        playNextSpeechChunk(requestId);
      }
    }
  }

  function queueSpeechChunks(chunks: string[]) {
    if (!chunks.length || !speechVoiceRef.current) return;
    speechTextQueueRef.current.push(...chunks);
    setSpeaking(true);
    void pumpSpeechSynthesis(playbackRequestRef.current);
  }

  function flushSpeechBuffer(force: boolean) {
    const split = splitSpeechBuffer(speechBufferRef.current, force);
    speechBufferRef.current = split.remainder;
    queueSpeechChunks(split.chunks);
  }

  function streamSpeechDelta(content: string) {
    speechReceivedRef.current = true;
    if (audioConfig.streaming_enabled) {
      beginStreamingSpeech(speechVoiceRef.current);
      speechStreamRef.current?.sendText(content);
      return;
    }
    speechBufferRef.current += content;
    flushSpeechBuffer(false);
  }

  async function uploadPersonalVoice() {
    const name = voiceName.trim();
    if (!name || !voiceFile || !voiceConsent || uploadingVoice) return;
    if (voiceFile.size > 25 * 1024 * 1024) {
      setError("参考录音不能超过 25 MB");
      return;
    }
    try {
      setError("");
      setUploadingVoice(true);
      const voice = await api.uploadVoice(voiceFile, name, voiceReferenceText);
      const config = await api.audioConfig();
      setAudioConfig(config);
      setSelectedVoice(voice.voice_id);
      setVoiceName("");
      setVoiceReferenceText("");
      setVoiceFile(null);
      setVoiceConsent(false);
      setVoiceForm(false);
    } catch (reason) {
      setError((reason as Error).message || "个人音色创建失败");
    } finally {
      setUploadingVoice(false);
    }
  }

  async function deletePersonalVoice() {
    if (!selectedVoiceOption?.custom || uploadingVoice) return;
    if (!window.confirm(`确定删除“${selectedVoiceOption.name}”吗？`)) return;
    try {
      setError("");
      stopSpeaking();
      await api.deleteVoice(selectedVoiceOption.voice_id);
      const config = await api.audioConfig();
      setAudioConfig(config);
      setSelectedVoice(config.default_voice ?? config.voices[0]?.voice_id ?? "");
    } catch (reason) {
      setError((reason as Error).message || "个人音色删除失败");
    }
  }

  function playSpeech(text: string) {
    const spokenText = text.trim();
    if (!audioConfig.synthesis_enabled || !selectedVoice || !spokenText) return;
    if (spokenText.length > 4_096) {
      setError("回答超过当前单次朗读长度限制，请缩短回答后重试");
      return;
    }
    stopSpeaking();
    speechVoiceRef.current = selectedVoice;
    if (audioConfig.streaming_enabled) {
      beginStreamingSpeech(selectedVoice);
      speechStreamRef.current?.sendText(spokenText);
      speechStreamRef.current?.finish();
      return;
    }
    speechBufferRef.current = spokenText;
    flushSpeechBuffer(true);
  }

  async function transcribeSegment(audio: Blob, segmentId: number) {
    const type = audio.type || "audio/webm";
    try {
      transcriptionsInFlightRef.current += 1;
      setTranscribing(true);
      const result = await api.transcribeAudio(
        audio,
        recordingFilename(type),
      );
      const text = result.text.trim();
      if (!text) return;
      voiceDraftRef.current = appendVoiceDraft(voiceDraftRef.current, text);
      voiceTurnOpenRef.current = true;
      setPrompt(voiceDraftRef.current);
      if (segmentId !== latestSpeechSegmentRef.current) return;
      await cancellationPromiseRef.current;
      if (segmentId !== latestSpeechSegmentRef.current) return;
      if (runningRef.current || activeRunIdRef.current) {
        await cancelActiveRun(true, true);
      }
      await submitPromptText(voiceDraftRef.current, true);
    } catch (reason) {
      if (segmentId === latestSpeechSegmentRef.current) {
        setError((reason as Error).message || "语音转写失败");
      }
    } finally {
      transcriptionsInFlightRef.current -= 1;
      if (transcriptionsInFlightRef.current === 0) setTranscribing(false);
    }
  }

  function enqueueTranscription(audio: Blob, segmentId: number) {
    transcriptionQueueRef.current = transcriptionQueueRef.current
      .catch(() => undefined)
      .then(() => transcribeSegment(audio, segmentId));
  }

  function startSegmentRecorder(stream: MediaStream, mimeType: string) {
    if (!recordingRef.current) return;
    const recorder = new MediaRecorder(
      stream,
      mimeType ? { mimeType } : undefined,
    );
    const segment: RecordingSegment = {
      recorder,
      chunks: [],
      id: latestSpeechSegmentRef.current,
      send: false,
    };
    activeRecordingSegmentRef.current = segment;
    mediaRecorderRef.current = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data.size) segment.chunks.push(event.data);
    };
    recorder.onstop = () => {
      if (segment.send && segment.chunks.length) {
        const type = recorder.mimeType || segment.chunks[0].type || "audio/webm";
        enqueueTranscription(new Blob(segment.chunks, { type }), segment.id);
      }
      if (recordingRef.current) {
        try {
          startSegmentRecorder(stream, mimeType);
        } catch (reason) {
          recordingRef.current = false;
          setRecording(false);
          stopVadMonitor();
          stream.getTracks().forEach((track) => track.stop());
          microphoneStreamRef.current = null;
          mediaRecorderRef.current = null;
          activeRecordingSegmentRef.current = null;
          setError((reason as Error).message || "无法继续录音");
        }
        return;
      }
      if (activeRecordingSegmentRef.current === segment) {
        activeRecordingSegmentRef.current = null;
        mediaRecorderRef.current = null;
        microphoneStreamRef.current?.getTracks().forEach((track) => track.stop());
        microphoneStreamRef.current = null;
      }
    };
    recorder.start();
  }

  function flushRecordingSegment(send: boolean) {
    const segment = activeRecordingSegmentRef.current;
    if (!segment || segment.recorder.state !== "recording") return;
    segment.id = latestSpeechSegmentRef.current;
    segment.send = send;
    lastAudioFlushAtRef.current = performance.now();
    segment.recorder.stop();
  }

  function startVadMonitor(stream: MediaStream) {
    const context = new AudioContext();
    const source = context.createMediaStreamSource(stream);
    const analyser = context.createAnalyser();
    analyser.fftSize = 2_048;
    analyser.smoothingTimeConstant = 0.2;
    source.connect(analyser);
    vadAudioContextRef.current = context;
    vadSourceRef.current = source;
    const samples = new Float32Array(analyser.fftSize);

    const monitor = () => {
      if (!recordingRef.current) return;
      analyser.getFloatTimeDomainData(samples);
      let energy = 0;
      for (const sample of samples) energy += sample * sample;
      const rms = Math.sqrt(energy / samples.length);
      const now = performance.now();

      if (rms >= VOICE_RMS_THRESHOLD) {
        if (!segmentHasSpeechRef.current) {
          if (!voiceCandidateAtRef.current) voiceCandidateAtRef.current = now;
          if (now - voiceCandidateAtRef.current >= VOICE_START_MS) {
            segmentHasSpeechRef.current = true;
            segmentStartedAtRef.current = voiceCandidateAtRef.current;
            latestSpeechSegmentRef.current += 1;
            if (!voiceTurnOpenRef.current) {
              voiceDraftRef.current = activePromptRef.current;
              voiceTurnOpenRef.current = true;
            }
            stopSpeaking();
            if (runningRef.current || activeRunIdRef.current) {
              cancellationPromiseRef.current = cancelActiveRun(true, true);
            }
          }
        }
        if (segmentHasSpeechRef.current) lastVoiceAtRef.current = now;
      } else {
        voiceCandidateAtRef.current = 0;
        if (
          segmentHasSpeechRef.current
          && now - lastVoiceAtRef.current >= SILENCE_TIMEOUT_MS
        ) {
          const shouldSend = lastVoiceAtRef.current - segmentStartedAtRef.current >= MIN_SPEECH_MS;
          segmentHasSpeechRef.current = false;
          flushRecordingSegment(shouldSend);
        } else if (
          !segmentHasSpeechRef.current
          && now - lastAudioFlushAtRef.current >= IDLE_AUDIO_FLUSH_MS
        ) {
          flushRecordingSegment(false);
        }
      }
      vadFrameRef.current = requestAnimationFrame(monitor);
    };

    void context.resume();
    vadFrameRef.current = requestAnimationFrame(monitor);
  }

  async function startRecording() {
    if (recordingRef.current) return;
    if (!audioConfig.transcription_enabled) {
      setError("服务端尚未配置语音转写");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setError("当前浏览器不支持麦克风录音");
      return;
    }
    try {
      setError("");
      stopSpeaking();
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      const mimeType = recordingMimeType();
      microphoneStreamRef.current = stream;
      activeRecordingSegmentRef.current = null;
      segmentHasSpeechRef.current = false;
      voiceCandidateAtRef.current = 0;
      lastVoiceAtRef.current = 0;
      lastAudioFlushAtRef.current = performance.now();
      recordingRef.current = true;
      setRecording(true);
      startSegmentRecorder(stream, mimeType);
      startVadMonitor(stream);
    } catch (reason) {
      recordingRef.current = false;
      stopVadMonitor();
      microphoneStreamRef.current?.getTracks().forEach((track) => track.stop());
      microphoneStreamRef.current = null;
      mediaRecorderRef.current = null;
      setRecording(false);
      setError((reason as Error).message || "无法访问麦克风");
    }
  }

  function stopRecording() {
    const segment = activeRecordingSegmentRef.current;
    if (!segment) return;
    recordingRef.current = false;
    setRecording(false);
    stopVadMonitor();
    if (segment.recorder.state === "recording") {
      segment.id = latestSpeechSegmentRef.current;
      segment.send = segmentHasSpeechRef.current;
      segment.recorder.stop();
    }
    segmentHasSpeechRef.current = false;
  }

  function applyEvent(
    event: RunEvent,
    currentWorkspace: string,
    currentSession: string,
    fromVoice: boolean,
  ) {
    if (event.type === "model_start") {
      const turn = Number(event.data.turn ?? 0);
      if (turn > 1) {
        setLiveAnswer("");
        setLiveReasoning("");
      }
      setActivities((items) => [
        ...items.map((item) => ({ ...item, state: "done" as const })),
        { key: `model-${turn}`, label: `模型正在思考 · 第 ${turn} 轮`, state: "active" },
      ]);
    } else if (event.type === "model_delta") {
      const content = String(event.data.content ?? "");
      const reasoning = String(event.data.reasoning_content ?? "");
      if (content) setLiveAnswer((current) => current + content);
      if (content && autoSpeak) streamSpeechDelta(content);
      if (reasoning) setLiveReasoning((current) => current + reasoning);
    } else if (event.type === "model_end") {
      setActivities((items) => items.map((item) => ({ ...item, state: "done" })));
    } else if (event.type === "tool_start") {
      const name = String(event.data.name ?? "tool");
      setActivities((items) => [
        ...items.map((item) => ({ ...item, state: "done" as const })),
        { key: `tool-${String(event.data.id)}`, label: `正在使用 ${name}`, state: "active" },
      ]);
    } else if (event.type === "tool_end") {
      setActivities((items) => items.map((item) => ({ ...item, state: "done" })));
    } else if (event.type === "final") {
      streamCloser.current?.();
      const result = event.data as unknown as AgentResult;
      activePromptRef.current = "";
      if (fromVoice) {
        voiceDraftRef.current = "";
        voiceTurnOpenRef.current = false;
      }
      setLastResult(result);
      setActivities([]);
      updateRunning(false);
      if (autoSpeak) {
        if (audioConfig.streaming_enabled) {
          beginStreamingSpeech(speechVoiceRef.current);
          if (!speechReceivedRef.current) {
            speechStreamRef.current?.sendText(result.final_message.content);
          }
          speechStreamRef.current?.finish();
        } else {
          if (!speechReceivedRef.current) {
            speechBufferRef.current += result.final_message.content;
          }
          flushSpeechBuffer(true);
        }
      }
      Promise.all([
        loadSession(currentWorkspace, currentSession, false),
        loadSessions(currentWorkspace),
      ])
        .catch((reason: Error) => setError(reason.message))
        .finally(() => {
          setLiveAnswer("");
          setLiveReasoning("");
        });
    } else if (event.type === "cancelled") {
      streamCloser.current?.();
      streamCloser.current = null;
      setActivities([]);
      updateRunning(false);
      stopSpeaking();
    } else if (event.type === "error") {
      streamCloser.current?.();
      activePromptRef.current = "";
      setError(String(event.data.message ?? "运行失败"));
      setActivities((items) => items.map((item) => ({ ...item, state: "done" })));
      updateRunning(false);
      stopSpeaking();
    }
  }

  async function submitPromptText(rawText: string, fromVoice = false) {
    const text = rawText.trim();
    if (
      !text
      || !session
      || runningRef.current
      || (!fromVoice && (recordingRef.current || transcriptionsInFlightRef.current > 0))
    ) return;
    const currentWorkspace = workspaceId;
    const currentSession = session.session_id;
    const generation = ++runGenerationRef.current;
    activePromptRef.current = text;
    if (!fromVoice) {
      voiceDraftRef.current = "";
      voiceTurnOpenRef.current = false;
    }
    setPrompt("");
    setError("");
    updateRunning(true);
    setActivities([]);
    setLiveAnswer("");
    setLiveReasoning("");
    setLastResult(null);
    stopSpeaking();
    speechVoiceRef.current = selectedVoice;
    if (autoSpeak && audioConfig.streaming_enabled && selectedVoice) {
      beginStreamingSpeech(selectedVoice);
    }
    setSession((current) => current ? {
      ...current,
      messages: [...current.messages, {
        role: "user",
        content: text,
        tool_calls: [],
        tool_call_id: null,
        reasoning_content: "",
        reasoning_signature: null,
      }],
    } : current);
    try {
      const run = await api.startRun(currentWorkspace, currentSession, text);
      if (generation !== runGenerationRef.current) {
        await api.cancelRun(run.run_id).catch(() => undefined);
        return;
      }
      activeRunIdRef.current = run.run_id;
      let completed = false;
      streamCloser.current = streamRun(
        run.events_url,
        (event) => {
          if (generation !== runGenerationRef.current) return;
          if (
            event.type === "final"
            || event.type === "cancelled"
            || event.type === "error"
          ) completed = true;
          if (event.type === "final" || event.type === "cancelled" || event.type === "error") {
            activeRunIdRef.current = null;
          }
          applyEvent(event, currentWorkspace, currentSession, fromVoice);
        },
        () => {
          if (generation !== runGenerationRef.current) return;
          if (!completed) {
            setError("运行事件连接中断，请检查服务端状态");
            updateRunning(false);
          }
        },
      );
    } catch (reason) {
      if (generation !== runGenerationRef.current) return;
      setError((reason as Error).message);
      updateRunning(false);
      await loadSession(currentWorkspace, currentSession).catch(() => undefined);
    }
  }

  async function sendPrompt() {
    await submitPromptText(prompt);
  }

  async function cancelActiveRun(
    reloadSession: boolean,
    preservePrompt = false,
  ) {
    if (!runningRef.current && !activeRunIdRef.current) return;
    const cancellationGeneration = ++runGenerationRef.current;
    const runId = activeRunIdRef.current;
    activeRunIdRef.current = null;
    streamCloser.current?.();
    streamCloser.current = null;
    stopSpeaking();
    updateRunning(false);
    setActivities([]);
    setLiveAnswer("");
    setLiveReasoning("");
    if (!preservePrompt) {
      activePromptRef.current = "";
      voiceDraftRef.current = "";
      voiceTurnOpenRef.current = false;
    }
    if (runId) {
      try {
        await api.cancelRun(runId);
      } catch (reason) {
        setError((reason as Error).message || "停止生成失败");
      }
    }
    if (
      reloadSession
      && session
      && cancellationGeneration === runGenerationRef.current
    ) {
      await loadSession(workspaceId, session.session_id).catch(() => undefined);
    }
  }

  async function stopGeneration() {
    await cancelActiveRun(true);
  }

  function composerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendPrompt();
    }
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="brand"><span>y</span> yyybot</div>
        <div className="sidebar-section">
          <div className="section-title"><span>WORKSPACE</span><button disabled={running} onClick={() => setWorkspaceForm(!workspaceForm)} aria-label="新建 Workspace"><Icon name="plus" /></button></div>
          {workspaceForm && <form className="inline-form" onSubmit={createWorkspace}>
            <input autoFocus value={newName} onChange={(event) => setNewName(event.target.value)} placeholder="Workspace 名称" />
            <button>创建</button>
          </form>}
          <select value={workspaceId} onChange={(event) => setWorkspaceId(event.target.value)} disabled={!workspaces.length || running}>
            {!workspaces.length && <option value="">还没有 Workspace</option>}
            {workspaces.map((workspace) => <option key={workspace.workspace_id} value={workspace.workspace_id}>{workspace.name}</option>)}
          </select>
        </div>

        <div className="sidebar-section sessions-section">
          <div className="section-title"><span>SESSIONS</span><button disabled={!workspaceId || running} onClick={() => setSessionForm(!sessionForm)} aria-label="新建 Session"><Icon name="plus" /></button></div>
          {sessionForm && <form className="inline-form" onSubmit={createSession}>
            <input autoFocus value={newTitle} onChange={(event) => setNewTitle(event.target.value)} placeholder="会话标题" />
            <button>创建</button>
          </form>}
          <nav className="session-list">
            {sessions.map((item) => (
              <button disabled={running} key={item.session_id} className={session?.session_id === item.session_id ? "selected" : ""} onClick={() => void loadSession(workspaceId, item.session_id)}>
                <strong>{item.title}</strong>
                <span>{item.turn_count} 轮 · {formatDate(item.updated_at)}</span>
              </button>
            ))}
          </nav>
        </div>
        <div className="sidebar-foot"><span className="status-dot" /> 本地运行 · JSONL 持久化</div>
      </aside>
      {sidebarOpen && <button className="scrim" aria-label="关闭侧栏" onClick={() => setSidebarOpen(false)} />}

      <main className="main-panel">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setSidebarOpen(true)}><Icon name="menu" /></button>
          <div>
            <p>{selectedWorkspace?.name ?? "yyybot"}</p>
            <h2>{session?.title ?? "准备开始"}</h2>
          </div>
          {lastResult && <div className="run-summary"><span>{lastResult.model_turns} 模型轮次</span><span>{lastResult.usage.total_tokens ?? 0} tokens</span></div>}
        </header>

        <section className="conversation">
          {!workspaces.length ? (
            <div className="empty-state">
              <div className="empty-mark"><Icon name="spark" /></div>
              <p className="eyebrow">FIRST THINGS FIRST</p>
              <h1>创建你的第一个 Workspace。</h1>
              <p>Workspace 隔离会话和运行数据，未来也可以挂接不同账号。</p>
              <button className="primary compact" onClick={() => { setWorkspaceForm(true); setSidebarOpen(true); }}><Icon name="plus" /> 新建 Workspace</button>
            </div>
          ) : !session ? <EmptyChat onCreate={() => void createSession()} /> : (
            <div className="message-column">
              {session.messages.map((message, index) => <MessageView key={`${index}-${message.tool_call_id ?? message.role}`} message={message} />)}
              {(running || liveAnswer || liveReasoning) && <article className="message assistant live-message">
                <div className="message-role">yyybot · 实时</div>
                {liveReasoning && <details className="reasoning-block live" open>
                  <summary>正在思考</summary>
                  <div>{liveReasoning}<span className="stream-cursor" /></div>
                </details>}
                {liveAnswer && <div className="live-answer"><MarkdownContent content={liveAnswer} /><span className="stream-cursor" /></div>}
              </article>}
              {activities.length > 0 && <div className="activity-stack">
                {activities.map((activity) => <div className={activity.state} key={activity.key}><i />{activity.label}</div>)}
              </div>}
              <div ref={endRef} />
            </div>
          )}
        </section>

        {session && <footer className="composer-wrap">
          {error && <div className="error-banner">{error}<button onClick={() => setError("")}>×</button></div>}
          {voiceForm && audioConfig.voice_upload_enabled && <div className="voice-upload-panel">
            <div className="voice-upload-heading">
              <strong>添加个人音色</strong>
              <span>推荐 5–15 秒、环境安静、单人清晰录音</span>
            </div>
            <div className="voice-upload-fields">
              <input
                value={voiceName}
                onChange={(event) => setVoiceName(event.target.value)}
                placeholder="音色名称"
                maxLength={80}
                disabled={uploadingVoice}
              />
              <input
                value={voiceReferenceText}
                onChange={(event) => setVoiceReferenceText(event.target.value)}
                placeholder="录音中的准确文字（推荐填写）"
                maxLength={1000}
                disabled={uploadingVoice}
              />
              <label className="voice-file">
                <span>{voiceFile?.name ?? "选择参考录音"}</span>
                <input
                  type="file"
                  accept="audio/*,video/webm,video/mp4"
                  onChange={(event) => setVoiceFile(event.target.files?.[0] ?? null)}
                  disabled={uploadingVoice}
                />
              </label>
            </div>
            <div className="voice-upload-actions">
              <label>
                <input
                  type="checkbox"
                  checked={voiceConsent}
                  onChange={(event) => setVoiceConsent(event.target.checked)}
                  disabled={uploadingVoice}
                />
                我确认已获得该声音所有者授权
              </label>
              <button onClick={() => setVoiceForm(false)} disabled={uploadingVoice}>取消</button>
              <button
                className="primary"
                onClick={() => void uploadPersonalVoice()}
                disabled={!voiceName.trim() || !voiceFile || !voiceConsent || uploadingVoice}
              >{uploadingVoice ? "正在创建…" : "创建音色"}</button>
            </div>
          </div>}
          {(audioConfig.transcription_enabled || audioConfig.synthesis_enabled) && <div className="voice-toolbar">
            {audioConfig.synthesis_enabled && <>
              <label className="voice-field">
                <span>AI 音色</span>
                <select value={selectedVoice} onChange={(event) => setSelectedVoice(event.target.value)} disabled={speaking}>
                  {audioConfig.voices.map((voice) => <option value={voice.voice_id} key={voice.voice_id}>{voice.name}</option>)}
                </select>
              </label>
              {audioConfig.voice_upload_enabled && <button
                className="speech-action"
                onClick={() => setVoiceForm((current) => !current)}
                disabled={uploadingVoice}
              ><Icon name="plus" />个人音色</button>}
              {selectedVoiceOption?.custom && <button
                className="speech-action danger"
                onClick={() => void deletePersonalVoice()}
                disabled={uploadingVoice}
              ><Icon name="trash" />删除</button>}
              <label className="auto-speak">
                <input
                  type="checkbox"
                  checked={autoSpeak}
                  onChange={(event) => {
                    setAutoSpeak(event.target.checked);
                    if (!event.target.checked) stopSpeaking();
                  }}
                />
                自动朗读
              </label>
              {(speaking || lastResult) && <button
                className={`speech-action ${speaking ? "active" : ""}`}
                onClick={() => speaking ? stopSpeaking() : void playSpeech(lastResult?.final_message.content ?? "")}
              >
                <Icon name={speaking ? "stop" : "volume"} />
                {speaking ? "停止朗读" : "朗读回答"}
              </button>}
            </>}
            {(recording || transcribing) && <span className={`speech-status ${recording ? "recording" : ""}`}>
              {recording
                ? transcribing
                  ? "持续监听中 · 正在识别上一段…"
                  : "持续监听中 · 静音 1.5 秒自动发送"
                : "正在识别语音…"}
            </span>}
          </div>}
          <div className="composer">
            <button
              className={`mic-button ${recording ? "recording" : ""} ${audioConfig.transcription_enabled ? "" : "unavailable"}`}
              onClick={() => recording ? stopRecording() : void startRecording()}
              disabled={!recording && transcribing}
              aria-label={
                recording
                  ? "停止录音"
                  : audioConfig.transcription_enabled
                    ? "开始录音"
                    : "语音转写未启用"
              }
              title={
                recording
                  ? "停止录音"
                  : audioConfig.transcription_enabled
                    ? "语音输入"
                    : "服务端尚未启用语音转写"
              }
            ><Icon name={recording ? "stop" : "mic"} /></button>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={composerKeyDown}
              placeholder={transcribing ? "正在把声音转换成文字…" : "给 yyybot 发消息…"}
              rows={1}
              disabled={running || transcribing}
            />
            <button
              className={`send-button ${running ? "stop-generation" : ""}`}
              onClick={() => running ? void stopGeneration() : void sendPrompt()}
              disabled={!running && (recording || transcribing || !prompt.trim())}
              aria-label={running ? "停止生成" : "发送"}
              title={running ? "停止生成" : "发送"}
            ><Icon name={running ? "stop" : "send"} /></button>
          </div>
          <p>{audioConfig.transcription_enabled ? "语音静音 1.5 秒自动发送，可继续说话插话 · " : ""}{audioConfig.synthesis_enabled ? "朗读声音由 AI 合成 · " : ""}Enter 发送 · Shift + Enter 换行 · Bash 工具已停用</p>
        </footer>}
        {!session && error && <div className="floating-error">{error}</div>}
      </main>
    </div>
  );
}
