# yyybot

一个小而完整、可扩展的个人助手 Agent Runtime。当前版本专注四个稳定边界：

```text
React UI / CLI
   │  workspace_id / session_id / AgentEvent
   ▼
FastAPI / ChatService
   │
   ├──► WorkspaceManager ──► Workspace ──► SessionManager ──► JSONL
                                         │
                                         ▼
                               ConversationContext
                                         │
                                         ▼
Agent ─────────► Model ───────► 平台 Provider ───────► 官方/兼容 SDK
  │
  └────────────► ToolRegistry ─────────► 本地函数 / API / MCP（后续）
```

## 设计原则

- **职责逐层收敛**：Agent 只面对 Model；Model 绑定模型 ID 和生成参数；Provider 封装对应平台 SDK。
- **模型配置与供应商连接分离**：同一个 Provider 可以供多个 Model 复用。
- **消息结构统一**：模型、工具和未来会话存储共享 `Message`、`ToolCall` 等契约。
- **会话与上下文分离**：`SessionManager` 管理每会话一个 JSONL 文件，`ConversationContext` 只负责内存中的模型上下文。
- **Workspace 是数据边界**：`WorkspaceManager` 管理 `~/.yyybot/workspaces`，每个 Workspace 拥有独立的 `sessions/`。
- **工具即函数**：同步和异步函数都能注册，可自动生成基础 JSON Schema，也允许显式传入完整 Schema。
- **UI 不侵入核心**：`AgentEvent` 可直接桥接日志、SSE 或 WebSocket。
- **入口共享服务层**：CLI 和 FastAPI 都通过 `ChatService` 运行 Agent，同一 Session 的并发请求会被串行化。
- **依赖按平台安装**：核心零依赖；OpenAI-compatible 和 Anthropic SDK 作为可选依赖。
- **内置网络工具**：CLI 注册 Google 优先、DuckDuckGo 自动回退的 `web_search`，并通过 `web_fetch` 安全读取搜索结果正文。
- **内置工具边界**：默认注册网络搜索和正文抓取；Bash 工具实现暂时保留，但不在 CLI 或 Web Agent 中注册。

## 快速开始

```bash
python -m pip install -e ".[all]"
export YYYBOT_MODEL=gpt-4.1-mini
export YYYBOT_API_KEY=your-key
yyybot "请计算 12.5 + 7.5"
```

创建持久会话后，CLI 会在标准错误中打印 session ID；后续通过该 ID
重新读取历史并追加新的一轮：

```bash
yyybot --new-session --session-title "旅行计划" "帮我规划上海行程"
yyybot --session <session-id> "把行程缩短到两天"
```

默认使用 `default` Workspace。创建并使用其他 Workspace：

```bash
yyybot --workspace research --create-workspace --workspace-name "研究" \
  --new-session "建立研究计划"
yyybot --workspace research --session <session-id> "继续完善计划"
```

数据默认保存在 `~/.yyybot/workspaces/<workspace-id>/sessions/`。可以通过
`YYYBOT_HOME` 或 `--yyybot-home` 覆盖 `.yyybot` 数据根目录。

Bash 工具当前没有注册。若后续恢复，持久 Session 会以对应的
`~/.yyybot/workspaces/<workspace-id>/` 作为工作目录；执行目录通过任务局部上下文
绑定，不会修改服务进程的全局 cwd。

网络搜索由 `.[all]` 一并安装；只需搜索功能时可安装 `.[web]`。如需为搜索单独设置代理，使用 `YYYBOT_WEB_PROXY`。

兼容服务可额外设置：

```bash
export YYYBOT_BASE_URL=http://localhost:11434/v1
export YYYBOT_PROVIDER=ollama
```

本地服务不要求 `YYYBOT_API_KEY`。更完整的边界与时序见
[`docs/architecture.md`](docs/architecture.md)。

`OllamaProvider` 和 `VLLMProvider` 默认不读取系统代理，避免本机请求被
`HTTP_PROXY` / `ALL_PROXY` 转发；连接远程部署时可显式设置 `trust_env=True`。

## Web UI

Web UI 已支持创建和选择 Workspace、创建和读取 Session、发送消息、实时查看
模型思考与回答正文、跟踪工具事件，以及查看模型轮数和 token 汇总。Provider
没有单独提供思考字段时，页面只流式展示回答正文。先构建前端：

```bash
cd web
npm install
npm run build
cd ..
```

然后配置模型并启动 API。以本机 Ollama 为例：

```bash
python -m pip install -e ".[all]"
export YYYBOT_PROVIDER=ollama
export YYYBOT_MODEL=qwen3.8
yyybot-server
```

服务默认监听所有网络接口。本机浏览器访问 `http://127.0.0.1:8000`，同一局域网
内的设备访问 `http://<服务器局域网 IP>:8000`。可用 `hostname -I` 查看服务器
地址；如果系统启用了防火墙，还需允许局域网访问 TCP 8000 端口。构建后的 React
页面由同一个 FastAPI 进程提供，因此生产使用不需要再运行 Vite。

Web API 当前没有登录认证，只应暴露在可信局域网中；Bash 工具已暂时停止注册。
要恢复为仅本机访问，可设置 `YYYBOT_HOST=127.0.0.1`，监听端口可通过
`YYYBOT_PORT` 修改。

开发前端时可以在第二个终端运行热更新服务：

```bash
cd web
npm run dev
```

Vite 同样默认监听所有网络接口。此时本机访问 `http://127.0.0.1:5173`，局域网
设备访问 `http://<服务器局域网 IP>:5173`；Vite 会把 `/api` 转发到 8000 端口。
OpenAI 模式需要设置 `YYYBOT_API_KEY`（或 SDK 支持的 `OPENAI_API_KEY`）。Web
入口当前只注册网络工具，不会向 Web Agent 提供 Bash 工具。

### 语音输入与朗读

Web UI 支持点击麦克风进入持续监听。检测到有效语音后，连续静音 1.5 秒会切出一个
独立录音片段，转写完成后自动进入原有 Agent 流程，而麦克风保持监听。回答生成期间
再次说话会取消旧 Run，把新转写拼接到当前语音草稿，并用完整内容重新推理。模型回答
仍以文字作为会话事实来源，并可使用所选 AI 音色自动或手动朗读；音频不会写入
Session JSONL。

使用官方 OpenAI Provider 且没有配置兼容服务 `YYYBOT_BASE_URL` 时，语音默认与
对话共用 API Key。也可以显式配置语音服务，尤其适合 Ollama/vLLM 对话加 OpenAI
语音的组合：

```bash
export YYYBOT_SPEECH_PROVIDER=openai
export YYYBOT_SPEECH_API_KEY=your-openai-key
export YYYBOT_STT_MODEL=gpt-transcribe
export YYYBOT_TTS_MODEL=gpt-4o-mini-tts
export YYYBOT_TTS_VOICE=marin
```

可通过 `YYYBOT_SPEECH_BASE_URL` 为语音设置独立服务地址，通过
`YYYBOT_SPEECH_PROVIDER=off` 关闭语音。内置音色可直接在页面选择，浏览器会记住
音色和自动朗读偏好。麦克风权限要求 HTTPS 安全上下文或 localhost。

中文语音转写也可以使用本地 SenseVoiceSmall INT8 模型：

```bash
pip install -e '.[all]'
mkdir -p ~/.yyybot/models/sensevoice-small-int8
curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2 \
  | tar -xj --strip-components=1 -C ~/.yyybot/models/sensevoice-small-int8
export YYYBOT_STT_PROVIDER=sherpa
# 默认从 ~/.yyybot/models/sensevoice-small-int8/ 读取：
#   model.int8.onnx
#   tokens.txt
```

可通过 `YYYBOT_STT_MODEL_DIR` 修改模型目录。`YYYBOT_STT_LANGUAGE` 默认为 `zh`，
`YYYBOT_STT_THREADS` 默认为 `4`。如果只使用本地转写、不需要云端朗读，设置
`YYYBOT_TTS_PROVIDER=off`。本地模型在服务启动时加载一次，浏览器录音会先解码为
16 kHz 单声道音频，再在线程中执行识别，避免阻塞 API 事件循环。

免费、本地的语音朗读可以使用 Qwen3-TTS 0.6B CustomVoice。先安装与显卡环境匹配
的 PyTorch，再安装本项目的 TTS 可选依赖；当前部署已验证 Python 3.13、
PyTorch 2.6 和 CUDA 12.4 组合：

```bash
pip install -e '.[local-tts]'
export YYYBOT_TTS_PROVIDER=qwen3
export YYYBOT_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
export YYYBOT_TTS_CLONE_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base
export YYYBOT_TTS_DEVICE=cuda:1
export YYYBOT_TTS_ATTENTION=flash_attention_2
export YYYBOT_TTS_VOICE=Serena
```

首次启动会下载模型，此后从本机缓存加载。也可以把模型提前下载到本地目录，并用
`YYYBOT_TTS_MODEL_DIR` 指向内置音色模型目录、用 `YYYBOT_TTS_CLONE_MODEL_DIR` 指向
Base 模型目录。默认使用 `sdpa`；安装 flash-attn 后可选择
`flash_attention_2`。当前机器在 Python 3.11、PyTorch 2.6、CUDA 12.4 和
FlashAttention 2.7.4 下验证通过；FlashAttention 2.8.3 暂无 CPython 3.13
预编译轮子。可通过
`YYYBOT_TTS_ATTENTION`、`YYYBOT_TTS_DTYPE` 和 `YYYBOT_TTS_LANGUAGE` 调整推理配置。
页面可选 Vivian、Serena、Uncle Fu、Dylan、Eric、Ryan、Aiden、Ono Anna 和
Sohee，默认使用适合中文的 Serena。点击“个人音色”可以上传参考录音；推荐使用
5–15 秒、环境安静、单人清晰的音频，并填写录音中的准确文字。个人音色默认持久化
到 `~/.yyybot/voices/`，也可通过 `YYYBOT_TTS_VOICES_DIR` 修改。仅应上传本人声音
或已获得声音所有者明确授权的录音。

开启自动朗读时，Web UI 会接收模型的流式文字 token，并在遇到句末标点或累计约
18 个字符后立即提交一个 TTS 短语。前一段音频播放时会继续生成后一段，避免等待
整篇回答完成。Qwen3-TTS 当前公开接口仍以整段 WAV 返回，不提供逐帧音频生成器，
因此这里采用短语级流式队列。单段声学 token 上限会按文本长度计算，总上限可通过
`YYYBOT_TTS_MAX_NEW_TOKENS` 设置，默认 `1024`，用于避免模型未及时生成结束标记时
长时间空转。
已有个人音色会在服务启动时预加载，以换取首次朗读的低延迟；可设置
`YYYBOT_TTS_PRELOAD_CUSTOM_VOICES=0` 恢复按需加载。

需要模型原生的文本、音频双向流式处理时，可以改用 Fun-CosyVoice3。yyybot 通过
`/api/audio/speech/stream` 建立 WebSocket：浏览器把模型回答的增量文本直接发送给
CosyVoice，服务端持续返回 24 kHz、单声道 PCM16 音频块，前端使用 Web Audio 无缝
排播，不再按短句反复生成完整 WAV。个人音色目录与 Qwen3-TTS 兼容，已有参考音频
和文字可以直接复用；新上传的 MP4/WebM 会先提取音轨并转换为 24 kHz 单声道 WAV。

CosyVoice 官方源码及其 `third_party/Matcha-TTS` 子模块需要放在本机目录，并提前下载
`FunAudioLLM/Fun-CosyVoice3-0.5B-2512` 权重。当前部署使用：

```bash
export YYYBOT_TTS_PROVIDER=cosyvoice3
export YYYBOT_TTS_MODEL_DIR=$HOME/.yyybot/models/Fun-CosyVoice3-0.5B
export YYYBOT_COSYVOICE_REPO=/path/to/CosyVoice
export YYYBOT_TTS_VOICES_DIR=$HOME/.yyybot/voices
export YYYBOT_TTS_DEVICE=cuda:1
export YYYBOT_TTS_FP16=1
export YYYBOT_TTS_VOICE=personal-xxxxxxxx
```

CosyVoice 与 sherpa-onnx 同时启用时，运行时会先加载 CosyVoice 的 ONNX 会话，再加载
SenseVoice，避免两个原生运行库按相反顺序初始化时发生冲突。WebSocket 支持需要安装
`.[server]`（其中包含 `websockets`）。

## 作为库使用

```python
import asyncio

from yyybot import Agent, Message, Model, ToolRegistry, WorkspaceManager
from yyybot.providers import OpenAIProvider

tools = ToolRegistry()

async def search_notes(query: str) -> list[str]:
    """Search the user's notes."""
    return [f"result for {query}"]

tools.add(search_notes)
provider = OpenAIProvider(
    api_key="your-key",
    base_url="https://provider.example/v1",
)
model = Model(model_id="your-model", provider=provider)

workspaces = WorkspaceManager()  # ~/.yyybot/workspaces
workspace = workspaces.ensure_default()
sessions = workspaces.sessions(workspace.workspace_id)
session = sessions.create(title="旅行计划")
context = sessions.load_context(session.session_id)
user_message = Message(role="user", content="查找我的旅行计划")
result = asyncio.run(Agent(model, tools=tools).run(context.build(user_message)))
sessions.append_turn(
    session.session_id,
    incoming=(user_message,),
    result=result,
)
print(result.output)

# 完整轨迹、模型轮数、逐轮及汇总 token usage 均保留在结果中
print(result.messages)
print(result.model_turns)
print(result.usage_by_turn)
print(result.usage)
```

## 扩展一个新模型供应商

每个平台拥有独立 Provider，直接使用相应 SDK：

```python
from yyybot.providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    VLLMProvider,
)
```

扩展新平台时实现 `Provider` 的统一调用边界即可，无需修改 Model 和 Agent：

```python
from collections.abc import Sequence
from yyybot import GenerationOptions, Message, ModelResponse, ToolSpec

class MyProvider:
    async def complete(
        self,
        *,
        model_id: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        options: GenerationOptions | None = None,
    ) -> ModelResponse:
        # 把统一结构映射为供应商请求，再映射回来
        ...
```

同一供应商增加模型只需创建新配置：

```python
fast = Model(model_id="fast-model", provider=provider)
smart = Model(model_id="smart-model", provider=provider)
```

## 当前范围与下一步

当前已具备单 Agent、连续工具调用、错误回传、轮次保护、完整运行结果、
Workspace 隔离、JSONL 会话存储，以及支持模型正文/思考增量的 SSE Web UI。
建议按以下顺序演进：

1. 运行取消、断线续传和更细的流控；
2. 账号、Workspace Membership 与角色授权；
3. 权限策略、工具审批与更强的执行沙箱；
4. SQLite/PostgreSQL `SessionStore` 适配器；
5. MCP 与多 Agent 编排。

运行测试：`pytest -q`。
