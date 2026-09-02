"""Anthropic platform provider."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from ..contracts import (
    GenerationOptions,
    Message,
    ModelResponse,
    ModelStreamEvent,
    ToolSpec,
)
from ._anthropic import (
    create_client,
    decode_response,
    request_payload,
    stream_with_client,
)
from .base import Provider, ProviderError


class AnthropicProvider(Provider):
    platform = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        client: Any | None = None,
        timeout: float = 60.0,
        default_max_tokens: int = 1024,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.client = client or create_client(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        self.base_url = base_url
        self.default_max_tokens = default_max_tokens

    async def complete(
        self,
        *,
        model_id: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        options: GenerationOptions | None = None,
    ) -> ModelResponse:
        request = request_payload(
            model_id=model_id,
            messages=messages,
            tools=tools,
            options=options or GenerationOptions(),
            default_max_tokens=self.default_max_tokens,
        )
        try:
            response = await self.client.messages.create(**request)
            return decode_response(response)
        except Exception as exc:
            raise ProviderError(f"anthropic request failed: {exc}") from exc

    async def stream(
        self,
        *,
        model_id: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        options: GenerationOptions | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        request = request_payload(
            model_id=model_id,
            messages=messages,
            tools=tools,
            options=options or GenerationOptions(),
            default_max_tokens=self.default_max_tokens,
        )
        try:
            async for event in stream_with_client(
                client=self.client,
                request=request,
            ):
                yield event
        except Exception as exc:
            raise ProviderError(f"anthropic streaming request failed: {exc}") from exc
