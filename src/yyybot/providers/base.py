"""The common interface implemented by platform providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence

from ..contracts import (
    GenerationOptions,
    Message,
    ModelResponse,
    ModelStreamEvent,
    ToolSpec,
)


class ProviderError(RuntimeError):
    pass


class Provider(ABC):
    """Base class for platform-named providers consumed by Model."""

    @abstractmethod
    async def complete(
        self,
        *,
        model_id: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        options: GenerationOptions | None = None,
    ) -> ModelResponse:
        raise NotImplementedError

    async def stream(
        self,
        *,
        model_id: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        options: GenerationOptions | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        """Stream when supported, otherwise adapt the regular completion."""

        response = await self.complete(
            model_id=model_id,
            messages=messages,
            tools=tools,
            options=options,
        )
        yield ModelStreamEvent(response=response)
