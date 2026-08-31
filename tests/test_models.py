import asyncio

from yyybot import GenerationOptions, Message, Model, ModelResponse, Provider, ToolSpec


class RecordingProvider(Provider):
    def __init__(self):
        self.requests = []

    async def complete(self, *, model_id, messages, tools=(), options=None):
        self.requests.append(
            {
                "model_id": model_id,
                "messages": tuple(messages),
                "tools": tuple(tools),
                "options": options,
            }
        )
        return ModelResponse(Message(role="assistant", content=model_id))


def test_model_binds_identity_and_options_to_provider():
    provider = RecordingProvider()
    options = GenerationOptions(temperature=0.4, extra={"seed": 7})
    model = Model(model_id="model-a", provider=provider, options=options)
    messages = [Message(role="user", content="Hello")]
    tools = [ToolSpec("ping", "Ping", {"type": "object", "properties": {}})]

    response = asyncio.run(model.complete(messages, tools))

    assert response.message.content == "model-a"
    assert provider.requests[0] == {
        "model_id": "model-a",
        "messages": tuple(messages),
        "tools": tuple(tools),
        "options": options,
    }


def test_one_provider_can_serve_multiple_models():
    provider = RecordingProvider()
    first = Model(model_id="model-a", provider=provider)
    second = Model(model_id="model-b", provider=provider)

    asyncio.run(first.complete([]))
    asyncio.run(second.complete([]))

    assert [request["model_id"] for request in provider.requests] == [
        "model-a",
        "model-b",
    ]
