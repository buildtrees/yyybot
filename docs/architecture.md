# Agent Runtime 架构

## 核心关系

```text
CLI / Future API / Future UI
             │
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

职责边界：

- `Agent` 负责消息循环、工具执行和终止条件；
- `Model` 负责绑定 `model_id`、生成参数和一个平台 Provider；
- 平台 Provider 负责对应 SDK、凭据、地址、请求映射、响应映射和错误归一化；
- SDK 的协议对象不泄漏到 Model、Agent 或 Tool 层。

没有独立的 Protocol 或 Transport 领域层。HTTP、SDK 类型和各平台协议都是 Provider 内部实现细节。

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
pip install -e ".[all]"       # 全部平台
```

Provider 使用延迟导入；未安装对应 SDK 时会给出明确安装提示。测试通过注入假 SDK Client 运行，不访问外部网络。
