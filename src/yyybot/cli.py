"""A deliberately small CLI proving the runtime is usable without a UI."""

from __future__ import annotations

import argparse
import asyncio
import os

if __package__:
    from .agent import Agent
    from .models import Model
    from .providers import (
        AnthropicProvider,
        OllamaProvider,
        OpenAIProvider,
        Provider,
        VLLMProvider,
    )
    from .terminal import bash
    from .tools import ToolRegistry
    from .web import web_fetch, web_search
else:
    # Support `python src/yyybot/cli.py ...` in addition to `python -m yyybot.cli`.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from yyybot.agent import Agent
    from yyybot.models import Model
    from yyybot.providers import (
        AnthropicProvider,
        OllamaProvider,
        OpenAIProvider,
        Provider,
        VLLMProvider,
    )
    from yyybot.terminal import bash
    from yyybot.tools import ToolRegistry
    from yyybot.web import web_fetch, web_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a yyybot agent")
    parser.add_argument("prompt")
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic", "ollama", "vllm"),
        default=os.getenv("YYYBOT_PROVIDER", "openai"),
    )
    parser.add_argument("--model", default=os.getenv("YYYBOT_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("YYYBOT_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("YYYBOT_API_KEY"))
    return parser


def build_provider(args: argparse.Namespace) -> Provider:
    common = {"api_key": args.api_key} if args.api_key else {}
    if args.base_url:
        common["base_url"] = args.base_url
    if args.provider == "openai":
        return OpenAIProvider(**common)
    if args.provider == "anthropic":
        return AnthropicProvider(**common)
    if args.provider == "vllm":
        return VLLMProvider(**common)
    common.pop("api_key", None)
    return OllamaProvider(**common)


async def _run(args: argparse.Namespace) -> None:
    if not args.model:
        raise SystemExit("Set YYYBOT_MODEL (or pass --model).")

    tools = ToolRegistry()

    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    tools.add(add)
    tools.add(web_search)
    tools.add(web_fetch)
    tools.add(bash)
    provider = build_provider(args)
    model = Model(model_id=args.model, provider=provider)
    result = await Agent(model, tools=tools).run(args.prompt)
    print(result.output)


def main() -> None:
    asyncio.run(_run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
