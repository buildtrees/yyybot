"""OpenAI platform provider."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..contracts import GenerationOptions, Message, ModelResponse, ToolSpec
from ._openai import complete_with_client, create_client
from .base import Provider


class OpenAIProvider(Provider):
    platform = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        client: Any | None = None,
        timeout: float = 60.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.client = client or create_client(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            extra_headers=extra_headers,
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
