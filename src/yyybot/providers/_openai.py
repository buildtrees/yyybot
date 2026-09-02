"""Internal helpers shared by platforms that expose an OpenAI-compatible API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from ..contracts import (
    GenerationOptions,
    Message,
    ModelDelta,
    ModelResponse,
    ModelStreamEvent,
    ToolCall,
    ToolSpec,
)
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
            reasoning_content=_reasoning_text(raw_message),
        ),
        choice.get("finish_reason"),
        usage,
    )


def _reasoning_text(payload: Mapping[str, Any]) -> str:
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def _request_payload(
    *,
    model_id: str,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
    options: GenerationOptions | None,
) -> dict[str, Any]:
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
    return request


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
    request = _request_payload(
        model_id=model_id,
        messages=messages,
        tools=tools,
        options=options,
    )
    try:
        response = await client.chat.completions.create(**request)
        return _decode_response(response)
    except Exception as exc:
        raise ProviderError(f"{platform} request failed: {exc}") from exc


async def stream_with_client(
    *,
    client: Any,
    platform: str,
    model_id: str,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
    options: GenerationOptions | None,
) -> AsyncIterator[ModelStreamEvent]:
    request = _request_payload(
        model_id=model_id,
        messages=messages,
        tools=tools,
        options=options,
    )
    request["stream"] = True
    request.setdefault("stream_options", {"include_usage": True})
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_parts: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None
    usage: dict[str, int] = {}
    try:
        stream = await client.chat.completions.create(**request)
        async with stream:
            async for chunk in stream:
                payload = _as_mapping(chunk)
                usage.update(
                    {
                        key: int(value)
                        for key, value in (payload.get("usage") or {}).items()
                        if isinstance(value, int)
                    }
                )
                choices = payload.get("choices") or ()
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                reasoning = _reasoning_text(delta)
                if content:
                    content_parts.append(content)
                if reasoning:
                    reasoning_parts.append(reasoning)
                if content or reasoning:
                    yield ModelStreamEvent(
                        delta=ModelDelta(
                            content=content,
                            reasoning_content=reasoning,
                        )
                    )
                for raw_call in delta.get("tool_calls") or ():
                    index = int(raw_call.get("index", 0))
                    part = tool_parts.setdefault(
                        index,
                        {"id": "", "name": "", "arguments": ""},
                    )
                    if raw_call.get("id"):
                        part["id"] = raw_call["id"]
                    function = raw_call.get("function") or {}
                    part["name"] += function.get("name") or ""
                    part["arguments"] += function.get("arguments") or ""

        calls: list[ToolCall] = []
        for index, part in sorted(tool_parts.items()):
            arguments = json.loads(part["arguments"] or "{}")
            if not isinstance(arguments, dict):
                raise TypeError("tool arguments must decode to an object")
            calls.append(
                ToolCall(
                    id=part["id"] or f"call_{index}",
                    name=part["name"],
                    arguments=arguments,
                )
            )
        yield ModelStreamEvent(
            response=ModelResponse(
                Message(
                    role="assistant",
                    content="".join(content_parts),
                    tool_calls=tuple(calls),
                    reasoning_content="".join(reasoning_parts),
                ),
                finish_reason,
                usage,
            )
        )
    except Exception as exc:
        raise ProviderError(f"{platform} streaming request failed: {exc}") from exc
