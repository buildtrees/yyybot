from __future__ import annotations

import asyncio

from yyybot import (
    Agent,
    AgentEvent,
    Message,
    Model,
    ModelDelta,
    ModelResponse,
    ModelStreamEvent,
    Provider,
    ToolCall,
    ToolRegistry,
)


class ScriptedProvider(Provider):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[dict] = []

    async def complete(self, *, model_id, messages, tools=(), options=None):
        self.requests.append(
            {
                "model_id": model_id,
                "messages": tuple(messages),
                "tools": tuple(tools),
                "options": options,
            }
        )
        return self.responses.pop(0)


class StreamingProvider(Provider):
    async def complete(self, *, model_id, messages, tools=(), options=None):
        raise AssertionError("Agent should use stream when it has an event handler")

    async def stream(self, *, model_id, messages, tools=(), options=None):
        yield ModelStreamEvent(delta=ModelDelta(reasoning_content="Working "))
        yield ModelStreamEvent(delta=ModelDelta(content="Done"))
        yield ModelStreamEvent(
            response=ModelResponse(
                Message(
                    role="assistant",
                    content="Done",
                    reasoning_content="Working ",
                ),
                "stop",
                {"total_tokens": 4},
            )
        )


def test_agent_executes_tool_then_returns_answer():
    provider = ScriptedProvider(
        [
            ModelResponse(
                Message(
                    role="assistant",
                    tool_calls=(ToolCall("call-1", "add", {"a": 2, "b": 3}),),
                ),
                "tool_calls",
                {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            ),
            ModelResponse(
                Message(role="assistant", content="The answer is 5."),
                "stop",
                {"input_tokens": 15, "output_tokens": 5, "total_tokens": 20},
            ),
        ]
    )
    model = Model(model_id="test-model", provider=provider)
    tools = ToolRegistry()
    tools.add(lambda a, b: a + b, name="add", description="Add numbers")

    messages = (Message(role="user", content="What is 2 + 3?"),)
    result = asyncio.run(Agent(model, tools=tools).run(messages))

    assert result.final_message == Message(role="assistant", content="The answer is 5.")
    assert result.output == "The answer is 5."
    assert result.model_turns == 2
    assert result.usage_by_turn == (
        {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        {"input_tokens": 15, "output_tokens": 5, "total_tokens": 20},
    )
    assert result.usage == {
        "input_tokens": 25,
        "output_tokens": 7,
        "total_tokens": 32,
    }
    assert messages == (Message(role="user", content="What is 2 + 3?"),)
    assert result.messages[:2] == (
        messages[0],
        result.responses[0].message,
    )
    tool_message = result.messages[2]
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id == "call-1"
    assert '"result": 5' in tool_message.content
    assert result.messages[3] == result.final_message
    assert provider.requests[1]["messages"] == result.messages[:-1]


def test_unknown_tool_is_reported_to_model():
    provider = ScriptedProvider(
        [
            ModelResponse(
                Message(
                    role="assistant",
                    tool_calls=(ToolCall("bad-1", "missing", {}),),
                )
            ),
            ModelResponse(Message(role="assistant", content="I cannot do that.")),
        ]
    )
    model = Model(model_id="test-model", provider=provider)

    result = asyncio.run(Agent(model).run((Message(role="user", content="Do it"),)))

    assert result.output == "I cannot do that."
    assert '"ok": false' in provider.requests[1]["messages"][-1].content
    assert "Unknown tool" in provider.requests[1]["messages"][-1].content


def test_events_form_ui_friendly_boundary():
    events: list[AgentEvent] = []
    provider = ScriptedProvider(
        [
            ModelResponse(
                Message(role="assistant", content="Done"),
                "stop",
                {"total_tokens": 7},
            )
        ]
    )
    model = Model(model_id="test-model", provider=provider)

    asyncio.run(
        Agent(model, on_event=events.append).run(
            (Message(role="user", content="Hello"),)
        )
    )

    assert [event.type for event in events] == ["model_start", "model_end"]
    assert events[-1].data["usage"] == {"total_tokens": 7}


def test_agent_forwards_model_deltas_and_returns_assembled_message():
    events: list[AgentEvent] = []
    model = Model("stream-model", StreamingProvider())

    result = asyncio.run(
        Agent(model, on_event=events.append).run(
            (Message(role="user", content="Hello"),)
        )
    )

    assert [event.type for event in events] == [
        "model_start",
        "model_delta",
        "model_delta",
        "model_end",
    ]
    assert events[1].data["reasoning_content"] == "Working "
    assert events[2].data["content"] == "Done"
    assert result.output == "Done"
    assert result.final_message.reasoning_content == "Working "
