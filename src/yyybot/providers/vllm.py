"""vLLM platform provider using its OpenAI-compatible server."""

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
from ._openai import complete_with_client, create_client, stream_with_client
from .base import Provider


class VLLMProvider(Provider):
    platform = "vllm"

    def __init__(
        self,
        *,
        api_key: str = "EMPTY",
        base_url: str = "http://localhost:8000/v1",
        client: Any | None = None,
        timeout: float = 60.0,
        extra_headers: Mapping[str, str] | None = None,
        trust_env: bool = False,
    ) -> None:
        self.client = client or create_client(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            extra_headers=extra_headers,
            trust_env=trust_env,
        )
        self.base_url = base_url

    async def complete(
        self,
        *,
        model_id: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        options: GenerationOptions | None = None,
    ) -> ModelResponse:
        return await complete_with_client(
            client=self.client,
            platform=self.platform,
            model_id=model_id,
            messages=messages,
            tools=tools,
            options=options,
        )

    async def stream(
        self,
        *,
        model_id: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        options: GenerationOptions | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        async for event in stream_with_client(
            client=self.client,
            platform=self.platform,
            model_id=model_id,
            messages=messages,
            tools=tools,
            options=options,
        ):
            yield event
