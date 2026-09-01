"""Internal helpers shared by platforms that expose an OpenAI-compatible API."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ..contracts import GenerationOptions, Message, ModelResponse, ToolCall, ToolSpec
from .base import ProviderError


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        result = model_dump(mode="json")
        if isinstance(result, Mapping):
            return result
    raise TypeError("SDK response must be a mapping or support model_dump()")


def _message_payload(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in message.tool_calls
        ]
    return payload


def _tool_payload(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _decode_response(response: Any) -> ModelResponse:
    payload = _as_mapping(response)
    choice = payload["choices"][0]
    raw_message = choice["message"]
    calls: list[ToolCall] = []
    for item in raw_message.get("tool_calls") or ():
        arguments = json.loads(item["function"].get("arguments") or "{}")
        if not isinstance(arguments, dict):
            raise TypeError("tool arguments must decode to an object")
        calls.append(
            ToolCall(
                id=item["id"],
                name=item["function"]["name"],
                arguments=arguments,
            )
        )
    usage = {
        key: int(value)
        for key, value in (payload.get("usage") or {}).items()
        if isinstance(value, int)
    }
    return ModelResponse(
        Message(
            role="assistant",
            content=raw_message.get("content") or "",
            tool_calls=tuple(calls),
        ),
        choice.get("finish_reason"),
        usage,
    )


def create_client(
    *,
    api_key: str | None,
    base_url: str,
    timeout: float,
    extra_headers: Mapping[str, str] | None,
    trust_env: bool = True,
) -> Any:
    try:
        from openai import AsyncOpenAI, DefaultAsyncHttpxClient
    except ImportError as exc:
        raise ProviderError(
            "OpenAI SDK is not installed; run `pip install -e '.[openai]'`"
        ) from exc
    client_options: dict[str, Any] = {}
    if not trust_env:
        client_options["http_client"] = DefaultAsyncHttpxClient(trust_env=False)
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        default_headers=extra_headers,
        **client_options,
    )


async def complete_with_client(
    *,
    client: Any,
    platform: str,
    model_id: str,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
    options: GenerationOptions | None,
) -> ModelResponse:
    generation = options or GenerationOptions()
    request: dict[str, Any] = {
        **generation.extra,
        "model": model_id,
        "messages": [_message_payload(message) for message in messages],
    }
    if tools:
        request["tools"] = [_tool_payload(tool) for tool in tools]
    if generation.temperature is not None:
        request["temperature"] = generation.temperature
    if generation.max_tokens is not None:
        request["max_tokens"] = generation.max_tokens
    try:
        response = await client.chat.completions.create(**request)
        return _decode_response(response)
    except Exception as exc:
        raise ProviderError(f"{platform} request failed: {exc}") from exc
