import {
  FormEvent,
  KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  AgentResult,
  Message,
  RunEvent,
  SessionDetail,
  SessionSummary,
  Workspace,
  api,
  streamRun,
} from "./api";

type Activity = { key: string; label: string; state: "active" | "done" };

function Icon({ name }: { name: "plus" | "send" | "menu" | "spark" }) {
  const paths = {
    plus: <path d="M12 5v14M5 12h14" />,
    send: <path d="m4 4 16 8-16 8 3-8-3-8Zm3 8h13" />,
    menu: <path d="M4 7h16M4 12h16M4 17h16" />,
    spark: <path d="m12 3 1.3 4.2L17 9l-3.7 1.8L12 15l-1.3-4.2L7 9l3.7-1.8L12 3Zm6 11 .7 2.3L21 17l-2.3.7L18 20l-.7-2.3L15 17l2.3-.7L18 14Z" />,
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
      <div className="message-content">{message.content}</div>
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
  const endRef = useRef<HTMLDivElement>(null);
  const streamCloser = useRef<null | (() => void)>(null);

  const selectedWorkspace = useMemo(
    () => workspaces.find((item) => item.workspace_id === workspaceId),
    [workspaces, workspaceId],
  );

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
    return () => streamCloser.current?.();
  }, [loadWorkspaces]);

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

  function applyEvent(event: RunEvent, currentWorkspace: string, currentSession: string) {
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
      setLastResult(result);
      setActivities([]);
      setRunning(false);
      Promise.all([
        loadSession(currentWorkspace, currentSession, false),
        loadSessions(currentWorkspace),
      ])
        .catch((reason: Error) => setError(reason.message))
        .finally(() => {
          setLiveAnswer("");
          setLiveReasoning("");
        });
    } else if (event.type === "error") {
      streamCloser.current?.();
      setError(String(event.data.message ?? "运行失败"));
      setActivities((items) => items.map((item) => ({ ...item, state: "done" })));
      setRunning(false);
    }
  }

  async function sendPrompt() {
    const text = prompt.trim();
    if (!text || !session || running) return;
    const currentWorkspace = workspaceId;
    const currentSession = session.session_id;
    setPrompt("");
    setError("");
    setRunning(true);
    setActivities([]);
    setLiveAnswer("");
    setLiveReasoning("");
    setLastResult(null);
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
      let completed = false;
      streamCloser.current = streamRun(
        run.events_url,
        (event) => {
          if (event.type === "final" || event.type === "error") completed = true;
          applyEvent(event, currentWorkspace, currentSession);
        },
        () => {
          if (!completed) {
            setError("运行事件连接中断，请检查服务端状态");
            setRunning(false);
          }
        },
      );
    } catch (reason) {
      setError((reason as Error).message);
      setRunning(false);
      await loadSession(currentWorkspace, currentSession).catch(() => undefined);
    }
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
                {liveAnswer && <div className="message-content">{liveAnswer}<span className="stream-cursor" /></div>}
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
          <div className="composer">
            <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={composerKeyDown} placeholder="给 yyybot 发消息…" rows={1} disabled={running} />
            <button className="send-button" onClick={() => void sendPrompt()} disabled={!prompt.trim() || running} aria-label="发送"><Icon name="send" /></button>
          </div>
          <p>Enter 发送 · Shift + Enter 换行 · Web 默认禁用 Bash</p>
        </footer>}
        {!session && error && <div className="floating-error">{error}</div>}
      </main>
    </div>
  );
}
