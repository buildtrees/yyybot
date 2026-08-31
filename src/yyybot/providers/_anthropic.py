"""Internal Anthropic SDK mapping helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..contracts import GenerationOptions, Message, ModelResponse, ToolCall, ToolSpec
from .base import ProviderError


def create_client(
    *,
    api_key: str | None,
    base_url: str,
    timeout: float,
    extra_headers: Mapping[str, str] | None,
) -> Any:
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise ProviderError(
            "Anthropic SDK is not installed; run `pip install -e '.[anthropic]'`"
        ) from exc
    return AsyncAnthropic(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        default_headers=extra_headers,
    )


def _blocks(message: Message) -> list[dict[str, Any]]:
    if message.role == "tool":
        return [
            {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
            }
        ]
    blocks: list[dict[str, Any]] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    blocks.extend(
        {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": dict(call.arguments),
        }
        for call in message.tool_calls
    )
    return blocks


def _messages(messages: Sequence[Message]) -> tuple[str, list[dict[str, Any]]]:
    system = "\n\n".join(
        message.content for message in messages if message.role == "system"
    )
    turns: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        role = "user" if message.role in ("user", "tool") else "assistant"
        blocks = _blocks(message)
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": blocks})
    return system, turns


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        result = model_dump(mode="json")
        if isinstance(result, Mapping):
            return result
    raise TypeError("SDK response must be a mapping or support model_dump()")


def decode_response(response: Any) -> ModelResponse:
    payload = _as_mapping(response)
    text: list[str] = []
    calls: list[ToolCall] = []
    for block in payload["content"]:
        if block.get("type") == "text":
            text.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            arguments = block.get("input") or {}
            if not isinstance(arguments, dict):
                raise TypeError("tool input must be an object")
            calls.append(
                ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=arguments,
                )
            )
    usage = {
        key: int(value)
        for key, value in (payload.get("usage") or {}).items()
        if isinstance(value, int)
    }
    if "input_tokens" in usage or "output_tokens" in usage:
        usage["total_tokens"] = usage.get("input_tokens", 0) + usage.get(
            "output_tokens", 0
        )
    return ModelResponse(
        Message(role="assistant", content="".join(text), tool_calls=tuple(calls)),
        payload.get("stop_reason"),
        usage,
    )


def request_payload(
    *,
    model_id: str,
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
    options: GenerationOptions,
    default_max_tokens: int,
) -> dict[str, Any]:
    system, turns = _messages(messages)
    request: dict[str, Any] = {
        **options.extra,
        "model": model_id,
        "max_tokens": (
            options.max_tokens
            if options.max_tokens is not None
            else default_max_tokens
        ),
        "messages": turns,
    }
    if system:
        request["system"] = system
    if tools:
        request["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]
    if options.temperature is not None:
        request["temperature"] = options.temperature
    return request
