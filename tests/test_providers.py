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
