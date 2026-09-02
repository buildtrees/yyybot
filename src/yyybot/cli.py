"""A deliberately small CLI proving the runtime is usable without a UI."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

if __package__:
    from .runtime import create_chat_service
    from .session import SessionError
    from .workspace import WorkspaceError
else:
    # Support `python src/yyybot/cli.py ...` in addition to `python -m yyybot.cli`.
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from yyybot.runtime import create_chat_service
    from yyybot.session import SessionError
    from yyybot.workspace import WorkspaceError


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
    sessions = parser.add_mutually_exclusive_group()
    sessions.add_argument("--session", help="Load and append to an existing session")
    sessions.add_argument(
        "--new-session",
        action="store_true",
        help="Create a new persisted session",
    )
    parser.add_argument("--session-title")
    parser.add_argument(
        "--workspace",
        default=os.getenv("YYYBOT_WORKSPACE", "default"),
        help="Workspace containing the selected session (default: default)",
    )
    parser.add_argument(
        "--create-workspace",
        action="store_true",
        help="Create --workspace before creating a new session",
    )
    parser.add_argument("--workspace-name")
    parser.add_argument(
        "--workspace-type",
        choices=("personal", "team"),
        default="personal",
    )
    parser.add_argument(
        "--yyybot-home",
        default=os.getenv("YYYBOT_HOME"),
        help="yyybot data root (default: ~/.yyybot)",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    if not args.model:
        raise SystemExit("Set YYYBOT_MODEL (or pass --model).")
    if args.session_title and not args.new_session:
        raise SystemExit("--session-title requires --new-session.")
    if args.create_workspace and not args.new_session:
        raise SystemExit("--create-workspace requires --new-session.")
    if args.workspace_name and not args.create_workspace:
        raise SystemExit("--workspace-name requires --create-workspace.")

    service = create_chat_service(
        model_id=args.model,
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        yyybot_home=args.yyybot_home,
        enable_bash=True,
    )
    session = None
    if args.new_session or args.session:
        try:
            if args.create_workspace:
                workspace = service.create_workspace(
                    workspace_id=args.workspace,
                    name=args.workspace_name or args.workspace,
                    workspace_type=args.workspace_type,
                )
                print(f"Created workspace: {workspace.workspace_id}", file=sys.stderr)
            elif args.workspace == "default" and args.new_session:
                workspace = service.ensure_default_workspace()
            else:
                workspace = service.get_workspace(args.workspace)
            if args.new_session:
                session = service.create_session(
                    workspace.workspace_id,
                    title=args.session_title or args.prompt[:60],
                )
                print(f"Created session: {session.session_id}", file=sys.stderr)
            else:
                session = service.get_session(workspace.workspace_id, args.session)
        except (WorkspaceError, SessionError) as exc:
            raise SystemExit(str(exc)) from exc

    if session:
        try:
            result = await service.run(
                workspace.workspace_id,
                session.session_id,
                args.prompt,
            )
        except SessionError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        result = await service.run_once(args.prompt)
    print(result.output)


def main() -> None:
    asyncio.run(_run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
