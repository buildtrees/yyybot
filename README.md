# yyybot

一个小而完整、可扩展的个人助手 Agent Runtime。当前版本专注四个稳定边界：

```text
UI / CLI
   │  AgentEvent
   ▼
Agent ───────► Model ───────► 平台 Provider ───────► 官方/兼容 SDK
  │
  └──────────► ToolRegistry ─────────► 本地函数 / API / MCP（后续）
```

## 设计原则

- **职责逐层收敛**：Agent 只面对 Model；Model 绑定模型 ID 和生成参数；Provider 封装对应平台 SDK。
- **模型配置与供应商连接分离**：同一个 Provider 可以供多个 Model 复用。
- **消息结构统一**：模型、工具和未来会话存储共享 `Message`、`ToolCall` 等契约。
- **工具即函数**：同步和异步函数都能注册，可自动生成基础 JSON Schema，也允许显式传入完整 Schema。
- **UI 不侵入核心**：`AgentEvent` 可直接桥接日志、SSE 或 WebSocket。
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

## 作为库使用

```python
import asyncio

from yyybot import Agent, Model, ToolRegistry
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

result = asyncio.run(Agent(model, tools=tools).run("查找我的旅行计划"))
print(result.output)
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

当前已具备单 Agent、连续工具调用、错误回传、轮次保护和事件钩子。建议按以下顺序演进：

1. 会话与检查点存储（SQLite/PostgreSQL）；
2. 流式模型接口及 SSE/WebSocket 网关；
3. 权限策略、工具超时与沙箱；
4. MCP 工具适配器；
5. Web UI 与多 Agent 编排。

运行测试：`pytest -q`。
