"""Factories for assembling the default yyybot runtime."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from .agent import Agent, EventHandler
from .models import Model
from .providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    Provider,
    VLLMProvider,
)
from .service import ChatService
from .terminal import bash
from .tools import ToolRegistry
from .web import web_fetch, web_search
from .workspace import WorkspaceManager


def create_provider(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Provider:
    options: dict[str, str] = {}
    if api_key:
        options["api_key"] = api_key
    if base_url:
        options["base_url"] = base_url
    if provider == "openai":
        return OpenAIProvider(**options)
    if provider == "anthropic":
        return AnthropicProvider(**options)
    if provider == "vllm":
        return VLLMProvider(**options)
    if provider == "ollama":
        options.pop("api_key", None)
        return OllamaProvider(**options)
    raise ValueError(f"Unsupported provider: {provider}")


def create_tools(*, enable_bash: bool) -> ToolRegistry:
    tools = ToolRegistry()
    tools.add(web_search)
    tools.add(web_fetch)
    if enable_bash:
        tools.add(bash)
    return tools


def create_chat_service(
    *,
    model_id: str,
    provider: str = "openai",
    api_key: str | None = None,
    base_url: str | None = None,
    yyybot_home: str | Path | None = None,
    enable_bash: bool = False,
) -> ChatService:
    model = Model(
        model_id=model_id,
        provider=create_provider(provider, api_key=api_key, base_url=base_url),
    )
    tools = create_tools(enable_bash=enable_bash)

    def agent_factory(on_event: EventHandler | None = None) -> Agent:
        return Agent(model, tools=tools, on_event=on_event)

    return ChatService(WorkspaceManager(yyybot_home), agent_factory)


def create_chat_service_from_env(
    environment: Mapping[str, str] | None = None,
) -> ChatService:
    env = os.environ if environment is None else environment
    model_id = env.get("YYYBOT_MODEL")
    if not model_id:
        raise RuntimeError("Set YYYBOT_MODEL before starting the server.")
    enable_bash = env.get("YYYBOT_ENABLE_BASH", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return create_chat_service(
        model_id=model_id,
        provider=env.get("YYYBOT_PROVIDER", "openai"),
        api_key=env.get("YYYBOT_API_KEY"),
        base_url=env.get("YYYBOT_BASE_URL"),
        yyybot_home=env.get("YYYBOT_HOME"),
        enable_bash=enable_bash,
    )
