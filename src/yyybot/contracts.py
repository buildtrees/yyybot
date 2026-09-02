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
    reasoning_content: str = ""
    reasoning_signature: str | None = None


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
class ModelDelta:
    """One provider-neutral increment of model-visible output."""

    content: str = ""
    reasoning_content: str = ""


@dataclass(frozen=True, slots=True)
class ModelStreamEvent:
    """A stream item containing either a delta or the completed response."""

    delta: ModelDelta | None = None
    response: ModelResponse | None = None

    def __post_init__(self) -> None:
        if (self.delta is None) == (self.response is None):
            raise ValueError("A model stream event must contain one payload")


@dataclass(frozen=True, slots=True)
class AgentResult:
    """The complete, provider-neutral record of one agent run.

    ``messages`` contains both the input context and every assistant/tool message
    produced during the run. ``responses`` preserves model metadata for each
    turn, including its finish reason and token usage.
    """

    final_message: Message
    messages: tuple[Message, ...]
    responses: tuple[ModelResponse, ...]

    @property
    def output(self) -> str:
        """Return the final assistant text for compatibility and convenience."""

        return self.final_message.content

    @property
    def model_turns(self) -> int:
        """Return the number of model calls made during the run."""

        return len(self.responses)

    @property
    def usage_by_turn(self) -> tuple[Mapping[str, int], ...]:
        """Return provider-reported token usage in model-turn order."""

        return tuple(response.usage for response in self.responses)

    @property
    def usage(self) -> Mapping[str, int]:
        """Return provider-reported usage summed across all model turns."""

        totals: dict[str, int] = {}
        for response in self.responses:
            for key, value in response.usage.items():
                totals[key] = totals.get(key, 0) + value
        return totals


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Provider-neutral generation settings owned by a configured model."""

    temperature: float | None = None
    max_tokens: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """A stable event boundary for a future UI, logger, or websocket layer."""

    type: Literal[
        "model_start",
        "model_delta",
        "model_end",
        "tool_start",
        "tool_end",
    ]
    data: Mapping[str, Any] = field(default_factory=dict)
