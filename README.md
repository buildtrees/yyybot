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
- **内置 Bash 工具**：CLI 注册非交互式 `bash`，提供超时、进程组终止和输出截断；仅应在可信环境中启用。

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

浏览器访问 `http://127.0.0.1:8000`。构建后的 React 页面由同一个 FastAPI
进程提供，因此生产使用不需要再运行 Vite。

开发前端时可以在第二个终端运行热更新服务：

```bash
cd web
npm run dev
```

此时访问 `http://127.0.0.1:5173`；Vite 会把 `/api` 转发到 8000 端口。
OpenAI 模式需要设置 `YYYBOT_API_KEY`（或 SDK 支持的 `OPENAI_API_KEY`）。Web
入口出于安全考虑默认不注册 Bash 工具；仅在完全可信的本地环境中可显式设置
`YYYBOT_ENABLE_BASH=1`。

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
