"""Provider-neutral data contracts used by the runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelResponse:
    message: Message
    finish_reason: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Provider-neutral generation settings owned by a configured model."""

    temperature: float | None = None
    max_tokens: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """A stable event boundary for a future UI, logger, or websocket layer."""

    type: Literal["model_start", "model_end", "tool_start", "tool_end"]
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentResult:
    output: str
    messages: tuple[Message, ...]
    model_turns: int
