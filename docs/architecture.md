# Agent Runtime 架构

## 核心关系

```text
React UI ──HTTP/SSE──► FastAPI ──┐
                                 ├──► ChatService
CLI ─────────────────────────────┘         │
                                          │ workspace_id / session_id
                                          ▼
                                  WorkspaceManager ──────► Workspace
                                     │ sessions/
                                     ▼
                              SessionManager ─────► Per-session JSONL
                                     │
                                     ▼
                           ConversationContext
                                     │ messages
                                     ▼
                                   Agent ─────────► Model ─────────► Platform Provider
                                     │                │                    │
                                     ▼                │                    ▼
                                ToolRegistry          │              Platform SDK
                                     │                │                    │
                                     ▼                └── model_id ────────┘
                               User Tool Functions                         │
                                                                           ▼
                                                                      Actual Model
```

语音是交互层适配能力，不进入 Agent 的消息协议：

```text
浏览器麦克风 ──原始音频──► SpeechService ──STT──► prompt
                                                     │
                                                     ▼
                                              原有 ChatService/Agent
                                                     │ final text
                                                     ▼
浏览器播放器 ◄────音频──── SpeechService ◄──TTS──────┘
```

职责边界：

- `WorkspaceManager` 负责 `~/.yyybot/workspaces` 下 Workspace 的创建、列出和加载；
- `ChatService` 是 CLI/API 共用的应用服务，协调 Workspace、Session、Context 和 Agent，并串行化同一 Session 的运行；
- `FastAPI` 暴露 Workspace/Session REST API，把 `AgentEvent` 和最终 `AgentResult` 通过 SSE 推送给 React UI；
- `SpeechService` 在 Agent 外部协调可独立替换的语音转写和语音合成 Provider；
- `Workspace` 是会话、文件、知识库和未来工具配置的数据隔离边界；
- `SessionManager` 负责创建、列出和加载会话，并把成功的运行按 turn 追加到独立 JSONL 文件；
- `ConversationContext` 负责 system prompt、已加载历史和当前输入的内存组装；
- `Agent` 负责消息循环、工具执行和终止条件；
- `Model` 负责绑定 `model_id`、生成参数和一个平台 Provider；
- 平台 Provider 负责对应 SDK、凭据、地址、请求映射、响应映射和错误归一化；
- SDK 的协议对象不泄漏到 Model、Agent 或 Tool 层。

没有独立的 Protocol 或 Transport 领域层。HTTP、SDK 类型和各平台协议都是 Provider 内部实现细节。

## Web 请求与事件流

```text
React UI
  │ POST /runs {prompt}
  ▼
FastAPI ──► RunRegistry ──► 后台 ChatService.run()
  ▲                              │
  │ SSE model_start/model_delta  │ AgentEvent
  │     model_end/tool events ◄──┘
  │     final / error
  └──────────────────────────────
```

`RunRegistry` 在进程内保存最近的运行事件，SSE 客户端即使稍晚连接也会先收到
已发生事件。运行完成后，`ChatService` 才将整轮结果追加到 JSONL；失败的运行不会
写入半个 turn。`model_delta` 分别携带 `content` 和 `reasoning_content`，页面可以
边生成边显示；`final` 仍返回聚合后的完整 `AgentResult`，用于校验与持久化。

Web 服务默认注册网络工具和 Bash；可通过 `YYYBOT_ENABLE_BASH=0` 显式关闭
Web 入口的 Bash。CLI 保持本地可信入口的现有行为。

## Provider 布局

```text
providers/
├── base.py          # Agent/Model 使用的最小 Provider 接口
├── openai.py        # OpenAIProvider → openai.AsyncOpenAI
├── anthropic.py     # AnthropicProvider → anthropic.AsyncAnthropic
├── ollama.py        # OllamaProvider → OpenAI-compatible SDK 调用
├── vllm.py          # VLLMProvider → OpenAI-compatible SDK 调用
├── _openai.py       # OpenAI SDK 族平台共享的内部映射函数
└── _anthropic.py    # Anthropic SDK 内部映射函数
```

以下类才是对外的平台概念：

- `OpenAIProvider`
- `AnthropicProvider`
- `OllamaProvider`
- `VLLMProvider`

以下内容不是公共架构扩展点：SDK 消息对象、HTTP 客户端、URL 路径和供应商原始响应。

## 一次请求的时序

```text
Agent
  │ model.complete(messages, tools)
  ▼
Model
  │ provider.complete(model_id, messages, tools, options)
  ▼
Platform Provider
  │ 转为 SDK 参数并调用对应异步客户端
  ▼
Platform SDK / API
  │ SDK response
  ▼
Platform Provider
  │ 转为统一 ModelResponse
  ▼
Model → Agent
```

Provider 输出统一的 `ModelResponse`，因此 Agent 不需要针对 OpenAI、Anthropic、Ollama 或 vLLM 编写分支。

## 语音适配层

`SpeechService` 与 `ChatService` 平行。`POST /api/audio/transcriptions` 接收浏览器录制
的原始音频并只返回转写文本；前端将文本放入输入框，用户确认后仍调用原来的
`POST /runs`。`POST /api/audio/speech` 提供整段音频兼容接口，音色列表与流式能力由
`GET /api/audio/config` 提供。支持原生流式的 Provider 还通过
`/api/audio/speech/stream` 建立 WebSocket：客户端连续发送 `text` 消息并以 `end`
结束输入，服务端返回 PCM16 二进制音频块。这样 LLM 文本增量和 TTS 音频增量可以
在同一合成会话内流水执行。

语音 Provider 不实现模型 `Provider`，也不扩展 `Message` 的音频字段。Session 继续
只持久化用户确认后的文字和 Agent 输出文字，因此切换 STT/TTS 平台不会改变 Agent
上下文、工具调用或 JSONL 格式。

## Workspace、会话与上下文

默认目录结构：

```text
~/.yyybot/
└── workspaces/
    ├── default/
    │   ├── workspace.json
    │   └── sessions/
    │       └── <session_id>.jsonl
    └── <workspace_id>/
        ├── workspace.json
        └── sessions/
```

`WorkspaceManager` 不保存全局“当前 Workspace”。CLI 或 UI 持有选中的
workspace ID。未来账号系统通过 membership 决定账号可以访问哪些 Workspace，
无需改变 Session、Context 或 Agent 边界。

每个 session 使用一个 `<session_id>.jsonl` 文件。第一行是带
`schema_version` 的会话元数据，之后每一行是一次成功完成的 turn。turn
只保存当前输入、新生成的 assistant/tool 消息以及逐轮 `ModelResponse`，不会
重复写入之前的上下文。

`SessionManager` 不保存全局“当前会话”。CLI 或 UI 持有选中的 session ID，
选择时调用 `load()` 或 `load_context()`。这样同一个 Manager 可以安全服务多个
独立的会话选择而不会互相污染状态，也方便未来将 JSONL 替换为
SQLite/PostgreSQL 存储。

`ChatService` 为 `(workspace_id, session_id)` 维护运行锁。两个浏览器请求同时写入
同一 Session 时，后一个请求会在锁内重新读取最新历史，避免从同一旧快照分别生成
并追加；不同 Session 仍可并行运行。

持久 Session 调用 Agent 时，`ChatService` 还会把工具执行目录绑定为对应的
Workspace 根目录。该绑定基于 `ContextVar`，因此不会调用全局 `os.chdir()`，不同
Workspace 的并发 Bash 调用也不会互相串目录。

## 扩展规则

### 新增模型

同一平台下只创建新的 Model，不增加类：

```python
provider = OllamaProvider()
fast = Model(model_id="small-model", provider=provider)
smart = Model(model_id="large-model", provider=provider)
```

### 新增平台

新增一个以平台命名的 Provider，在内部使用该平台 SDK，并实现统一的 `complete()` 方法。不要把供应商 SDK 类型带入共享契约。

### 新增工具

使用 `ToolRegistry.add()` 注册同步或异步函数。工具与 Provider 完全独立，只通过 `ToolSpec` 和 `ToolCall` 与模型调用链连接。

## 依赖策略

核心 Runtime 不强制安装所有模型 SDK：

```bash
pip install -e ".[openai]"    # OpenAI、Ollama、vLLM
pip install -e ".[anthropic]" # Anthropic
pip install -e ".[server]"    # FastAPI + Uvicorn
pip install -e ".[all]"       # 全部平台
```

Provider 使用延迟导入；未安装对应 SDK 时会给出明确安装提示。测试通过注入假 SDK Client 运行，不访问外部网络。
