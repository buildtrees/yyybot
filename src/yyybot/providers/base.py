"""The common interface implemented by platform providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..contracts import GenerationOptions, Message, ModelResponse, ToolSpec


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
