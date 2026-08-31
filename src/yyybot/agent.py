"""The provider-neutral agent execution loop."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from .models import Model
from .tools import ToolError, ToolRegistry
from .contracts import AgentEvent, AgentResult, Message

EventHandler = Callable[[AgentEvent], None | Awaitable[None]]


class AgentLimitError(RuntimeError):
    pass


class Agent:
    def __init__(
        self,
        model: Model,
        *,
        tools: ToolRegistry | None = None,
        system_prompt: str = "You are a helpful personal assistant.",
        max_model_turns: int = 8,
        on_event: EventHandler | None = None,
    ) -> None:
        if max_model_turns < 1:
            raise ValueError("max_model_turns must be positive")
        self.model = model
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
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
        prompt: str,
        *,
        history: Sequence[Message] = (),
    ) -> AgentResult:
        messages: list[Message] = list(history)
        if self.system_prompt and not any(m.role == "system" for m in messages):
            messages.insert(0, Message(role="system", content=self.system_prompt))
        messages.append(Message(role="user", content=prompt))

        for turn in range(1, self.max_model_turns + 1):
            await self._emit(AgentEvent("model_start", {"turn": turn}))
            response = await self.model.complete(messages, self.tools.specs)
            assistant = response.message
            if assistant.role != "assistant":
                raise TypeError("A model response must contain an assistant message")
            messages.append(assistant)
            await self._emit(
                AgentEvent(
                    "model_end",
                    {"turn": turn, "finish_reason": response.finish_reason},
                )
            )

            if not assistant.tool_calls:
                return AgentResult(assistant.content, tuple(messages), turn)

            for call in assistant.tool_calls:
                await self._emit(
                    AgentEvent("tool_start", {"id": call.id, "name": call.name})
                )
                try:
                    value = await self.tools.execute(call)
                    content = self._tool_content({"ok": True, "result": value})
                except ToolError as exc:
                    content = self._tool_content({"ok": False, "error": str(exc)})
                messages.append(
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
