from __future__ import annotations

import asyncio

from yyybot import (
    Agent,
    AgentEvent,
    Message,
    Model,
    ModelResponse,
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


def test_agent_executes_tool_then_returns_answer():
    provider = ScriptedProvider(
        [
            ModelResponse(
                Message(
                    role="assistant",
                    tool_calls=(ToolCall("call-1", "add", {"a": 2, "b": 3}),),
                ),
                "tool_calls",
            ),
            ModelResponse(Message(role="assistant", content="The answer is 5."), "stop"),
        ]
    )
    model = Model(model_id="test-model", provider=provider)
    tools = ToolRegistry()
    tools.add(lambda a, b: a + b, name="add", description="Add numbers")

    result = asyncio.run(Agent(model, tools=tools).run("What is 2 + 3?"))

    assert result.output == "The answer is 5."
    assert result.model_turns == 2
    tool_message = provider.requests[1]["messages"][-1]
    assert tool_message.role == "tool"
    assert tool_message.tool_call_id == "call-1"
    assert '"result": 5' in tool_message.content


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

    result = asyncio.run(Agent(model).run("Do it"))

    assert result.output == "I cannot do that."
    assert '"ok": false' in provider.requests[1]["messages"][-1].content
    assert "Unknown tool" in provider.requests[1]["messages"][-1].content


def test_events_form_ui_friendly_boundary():
    events: list[AgentEvent] = []
    provider = ScriptedProvider(
        [ModelResponse(Message(role="assistant", content="Done"), "stop")]
    )
    model = Model(model_id="test-model", provider=provider)

    asyncio.run(Agent(model, on_event=events.append).run("Hello"))

    assert [event.type for event in events] == ["model_start", "model_end"]
