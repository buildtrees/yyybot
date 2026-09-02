import asyncio
from types import SimpleNamespace

from yyybot import GenerationOptions, Message, Model, ToolCall, ToolSpec
from yyybot.providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    VLLMProvider,
)


class FakeCreateEndpoint:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def create(self, **request):
        self.requests.append(request)
        return self.response


class FakeAsyncStream:
    def __init__(self, items, final_message=None):
        self.items = list(items)
        self.final_message = final_message

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def get_final_message(self):
        return self.final_message


class FakeAnthropicEndpoint(FakeCreateEndpoint):
    def __init__(self, response, events):
        super().__init__(response)
        self.events = events

    def stream(self, **request):
        self.requests.append(request)
        return FakeAsyncStream(self.events, self.response)


def openai_client(response):
    endpoint = FakeCreateEndpoint(response)
    return SimpleNamespace(chat=SimpleNamespace(completions=endpoint)), endpoint


def anthropic_client(response):
    endpoint = FakeCreateEndpoint(response)
    return SimpleNamespace(messages=endpoint), endpoint


def test_openai_provider_uses_sdk_client_and_maps_tool_calls():
    client, endpoint = openai_client(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "weather",
                                    "arguments": '{"city": "Shanghai"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"total_tokens": 12},
        }
    )
    provider = OpenAIProvider(client=client)
    model = Model(
        model_id="test-model",
        provider=provider,
        options=GenerationOptions(temperature=0.2, max_tokens=100),
    )

    response = asyncio.run(
        model.complete(
            [Message(role="user", content="Weather?")],
            [
                ToolSpec(
                    "weather",
                    "Get weather",
                    {"type": "object", "properties": {"city": {"type": "string"}}},
                )
            ],
        )
    )

    request = endpoint.requests[0]
    assert request["model"] == "test-model"
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 100
    assert request["tools"][0]["function"]["name"] == "weather"
    assert response.message.tool_calls[0].arguments == {"city": "Shanghai"}
    assert response.usage["total_tokens"] == 12


def test_ollama_and_vllm_are_platforms_using_openai_compatible_clients():
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ]
    }
    ollama_client, ollama_endpoint = openai_client(response)
    vllm_client, vllm_endpoint = openai_client(response)
    ollama = Model("local-a", OllamaProvider(client=ollama_client))
    vllm = Model("local-b", VLLMProvider(client=vllm_client))

    assert asyncio.run(ollama.complete([])).message.content == "ok"
    assert asyncio.run(vllm.complete([])).message.content == "ok"
    assert ollama_endpoint.requests[0]["model"] == "local-a"
    assert vllm_endpoint.requests[0]["model"] == "local-b"
    assert ollama.provider.platform == "ollama"
    assert vllm.provider.platform == "vllm"


def test_local_providers_ignore_system_proxy_by_default(monkeypatch):
    calls = []
    fake_client, _ = openai_client({})

    def fake_create_client(**options):
        calls.append(options)
        return fake_client

    monkeypatch.setattr("yyybot.providers.ollama.create_client", fake_create_client)
    monkeypatch.setattr("yyybot.providers.vllm.create_client", fake_create_client)

    OllamaProvider()
    VLLMProvider()

    assert [call["trust_env"] for call in calls] == [False, False]


def test_anthropic_provider_uses_messages_sdk_shape():
    client, endpoint = anthropic_client(
        {
            "content": [
                {"type": "text", "text": "Checking."},
                {
                    "type": "tool_use",
                    "id": "toolu-1",
                    "name": "weather",
                    "input": {"city": "Shanghai"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    provider = AnthropicProvider(client=client, default_max_tokens=512)
    model = Model("claude-test", provider)
    messages = [
        Message(role="system", content="Be concise."),
        Message(role="user", content="Weather?"),
        Message(
            role="assistant",
            tool_calls=(ToolCall("old-call", "weather", {"city": "Beijing"}),),
        ),
        Message(role="tool", content="Sunny", tool_call_id="old-call"),
    ]
    tools = [
        ToolSpec(
            "weather",
            "Get weather",
            {"type": "object", "properties": {"city": {"type": "string"}}},
        )
    ]

    response = asyncio.run(model.complete(messages, tools))

    request = endpoint.requests[0]
    assert request["model"] == "claude-test"
    assert request["system"] == "Be concise."
    assert request["max_tokens"] == 512
    assert request["tools"][0]["input_schema"] == tools[0].parameters
    assert request["messages"][-1]["content"][0]["type"] == "tool_result"
    assert response.message.content == "Checking."
    assert response.message.tool_calls[0].name == "weather"
    assert response.usage["total_tokens"] == 15


def test_openai_compatible_stream_maps_reasoning_text_tools_and_usage():
    chunks = [
        {
            "choices": [
                {"delta": {"reasoning_content": "Think "}, "finish_reason": None}
            ]
        },
        {"choices": [{"delta": {"content": "Answer."}, "finish_reason": None}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "function": {
                                    "name": "wea",
                                    "arguments": '{"city":',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "name": "ther",
                                    "arguments": '"Shanghai"}',
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"choices": [], "usage": {"total_tokens": 9}},
    ]
    client, endpoint = openai_client(FakeAsyncStream(chunks))
    model = Model("stream-model", OllamaProvider(client=client))

    async def collect():
        return [event async for event in model.stream([])]

    events = asyncio.run(collect())

    assert endpoint.requests[0]["stream"] is True
    assert endpoint.requests[0]["stream_options"] == {"include_usage": True}
    assert events[0].delta.reasoning_content == "Think "
    assert events[1].delta.content == "Answer."
    response = events[-1].response
    assert response.message.reasoning_content == "Think "
    assert response.message.content == "Answer."
    assert response.message.tool_calls[0].name == "weather"
    assert response.message.tool_calls[0].arguments == {"city": "Shanghai"}
    assert response.usage == {"total_tokens": 9}


def test_anthropic_stream_maps_thinking_and_text_deltas():
    final = {
        "content": [
            {"type": "thinking", "thinking": "Consider.", "signature": "sig"},
            {"type": "text", "text": "Done."},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 2, "output_tokens": 3},
    }
    events = [
        {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "Consider."},
        },
        {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Done."},
        },
    ]
    endpoint = FakeAnthropicEndpoint(final, events)
    client = SimpleNamespace(messages=endpoint)
    model = Model("claude-stream", AnthropicProvider(client=client))

    async def collect():
        return [event async for event in model.stream([])]

    streamed = asyncio.run(collect())

    assert streamed[0].delta.reasoning_content == "Consider."
    assert streamed[1].delta.content == "Done."
    response = streamed[-1].response
    assert response.message.reasoning_content == "Consider."
    assert response.message.reasoning_signature == "sig"
    assert response.message.content == "Done."
    assert response.usage["total_tokens"] == 5
