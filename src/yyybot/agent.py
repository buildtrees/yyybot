"""The provider-neutral agent execution loop."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from .models import Model
from .tools import ToolError, ToolRegistry
from .contracts import AgentEvent, AgentResult, Message, ModelResponse

EventHandler = Callable[[AgentEvent], None | Awaitable[None]]


class AgentLimitError(RuntimeError):
    pass


class Agent:
    def __init__(
        self,
        model: Model,
        *,
        tools: ToolRegistry | None = None,
        max_model_turns: int = 16,
        on_event: EventHandler | None = None,
    ) -> None:
        if max_model_turns < 1:
            raise ValueError("max_model_turns must be positive")
        self.model = model
        self.tools = tools or ToolRegistry()
        self.max_model_turns = max_model_turns
        self.on_event = on_event

    async def _emit(self, event: AgentEvent) -> None:
        if self.on_event is None:
            return
        result = self.on_event(event)
        if inspect.isawaitable(result):
            await result

    @staticmethod
    def _tool_content(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, default=str)

    async def run(
        self,
        messages: Sequence[Message],
    ) -> AgentResult:
        working_messages = list(messages)
        if not all(isinstance(message, Message) for message in working_messages):
            raise TypeError("messages must contain only Message instances")
        responses: list[ModelResponse] = []

        for turn in range(1, self.max_model_turns + 1):
            await self._emit(AgentEvent("model_start", {"turn": turn}))
            if self.on_event is None:
                response = await self.model.complete(working_messages, self.tools.specs)
            else:
                response = None
                async for event in self.model.stream(
                    working_messages, self.tools.specs
                ):
                    if event.delta is not None:
                        await self._emit(
                            AgentEvent(
                                "model_delta",
                                {
                                    "turn": turn,
                                    "content": event.delta.content,
                                    "reasoning_content": (
                                        event.delta.reasoning_content
                                    ),
                                },
                            )
                        )
                    else:
                        response = event.response
                if response is None:
                    raise RuntimeError(
                        "Provider stream ended without a completed response"
                    )
            assistant = response.message
            if assistant.role != "assistant":
                raise TypeError("A model response must contain an assistant message")
            responses.append(response)
            working_messages.append(assistant)
            await self._emit(
                AgentEvent(
                    "model_end",
                    {
                        "turn": turn,
                        "finish_reason": response.finish_reason,
                        "usage": dict(response.usage),
                    },
                )
            )

            if not assistant.tool_calls:
                return AgentResult(
                    final_message=assistant,
                    messages=tuple(working_messages),
                    responses=tuple(responses),
                )

            for call in assistant.tool_calls:
                await self._emit(
                    AgentEvent("tool_start", {"id": call.id, "name": call.name})
                )
                try:
                    value = await self.tools.execute(call)
                    content = self._tool_content({"ok": True, "result": value})
                except ToolError as exc:
                    content = self._tool_content({"ok": False, "error": str(exc)})
                working_messages.append(
                    Message(role="tool", content=content, tool_call_id=call.id)
                )
                await self._emit(
                    AgentEvent(
                        "tool_end",
                        {"id": call.id, "name": call.name, "content": content},
                    )
                )

        raise AgentLimitError(
            f"Agent exceeded {self.max_model_turns} model turns without a final answer"
        )
