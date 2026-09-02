"""Agent-facing model objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import AsyncIterator, Sequence

from .contracts import (
    GenerationOptions,
    Message,
    ModelResponse,
    ModelStreamEvent,
    ToolSpec,
)
from .providers.base import Provider


@dataclass(frozen=True, slots=True)
class Model:
    """A concrete model id bound to a provider and generation settings.

    The Agent only sees this class. SDK usage, credentials and API mapping remain
    inside the provider.
    """

    model_id: str
    provider: Provider
    options: GenerationOptions = field(default_factory=GenerationOptions)

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty")

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> ModelResponse:
        return await self.provider.complete(
            model_id=self.model_id,
            messages=messages,
            tools=tools,
            options=self.options,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
    ) -> AsyncIterator[ModelStreamEvent]:
        async for event in self.provider.stream(
            model_id=self.model_id,
            messages=messages,
            tools=tools,
            options=self.options,
        ):
            yield event
