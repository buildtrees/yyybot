"""FastAPI gateway for the yyybot web interface."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Coroutine
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .contracts import AgentEvent, AgentResult, Message, ModelResponse
from .runtime import create_chat_service_from_env
from .service import ChatService
from .session import Session, SessionError, SessionNotFoundError
from .workspace import (
    Workspace,
    WorkspaceError,
    WorkspaceNotFoundError,
    WorkspaceType,
)


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    workspace_id: str | None = None
    workspace_type: WorkspaceType = "personal"


class SessionCreate(BaseModel):
    title: str = Field(default="New session", max_length=200)
    session_id: str | None = None
    system_prompt: str | None = "You are a helpful personal assistant."


class RunCreate(BaseModel):
    prompt: str = Field(min_length=1, max_length=100_000)


def _message_data(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": [
            {
                "id": call.id,
                "name": call.name,
                "arguments": dict(call.arguments),
            }
            for call in message.tool_calls
        ],
        "tool_call_id": message.tool_call_id,
        "reasoning_content": message.reasoning_content,
        "reasoning_signature": message.reasoning_signature,
    }


def _response_data(response: ModelResponse) -> dict[str, Any]:
    return {
        "message": _message_data(response.message),
        "finish_reason": response.finish_reason,
        "usage": dict(response.usage),
    }


def _workspace_data(workspace: Workspace) -> dict[str, Any]:
    return {
        "workspace_id": workspace.workspace_id,
        "name": workspace.name,
        "workspace_type": workspace.workspace_type,
        "created_at": workspace.created_at,
        "owner_account_id": workspace.owner_account_id,
    }


def _session_summary(session: Session) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "turn_count": len(session.turns),
    }


def _session_data(session: Session) -> dict[str, Any]:
    return {
        **_session_summary(session),
        "system_prompt": session.system_prompt,
        "messages": [_message_data(message) for message in session.messages],
        "turns": [
            {
                "turn_id": turn.turn_id,
                "created_at": turn.created_at,
                "incoming": [_message_data(message) for message in turn.incoming],
                "generated": [_message_data(message) for message in turn.generated],
                "responses": [_response_data(response) for response in turn.responses],
            }
            for turn in session.turns
        ],
    }


def _result_data(result: AgentResult) -> dict[str, Any]:
    return {
        "final_message": _message_data(result.final_message),
        "messages": [_message_data(message) for message in result.messages],
        "responses": [_response_data(response) for response in result.responses],
        "model_turns": result.model_turns,
        "usage_by_turn": [dict(usage) for usage in result.usage_by_turn],
        "usage": dict(result.usage),
    }


@dataclass(frozen=True, slots=True)
class RunEvent:
    event_id: int
    event_type: str
    data: dict[str, Any]


@dataclass(slots=True)
class RunState:
    run_id: str
    events: list[RunEvent] = field(default_factory=list)
    done: bool = False
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def publish(self, event_type: str, data: dict[str, Any]) -> None:
        async with self.condition:
            self.events.append(RunEvent(len(self.events) + 1, event_type, data))
            self.condition.notify_all()

    async def finish(self) -> None:
        async with self.condition:
            self.done = True
            self.condition.notify_all()

    async def stream(self, after: int = 0):
        index = max(after, 0)
        while True:
            async with self.condition:
                await self.condition.wait_for(
                    lambda: index < len(self.events) or self.done
                )
                batch = self.events[index:]
                finished = self.done and index + len(batch) >= len(self.events)
            for event in batch:
                index = event.event_id
                yield event
            if finished:
                return


class RunRegistry:
    def __init__(self, max_runs: int = 100) -> None:
        self.max_runs = max_runs
        self.states: dict[str, RunState] = {}
        self.tasks: set[asyncio.Task[None]] = set()

    def create(self) -> RunState:
        if len(self.states) >= self.max_runs:
            completed = [key for key, state in self.states.items() if state.done]
            for key in completed[: max(1, len(self.states) - self.max_runs + 1)]:
                self.states.pop(key, None)
        state = RunState(uuid4().hex)
        self.states[state.run_id] = state
        return state

    def get(self, run_id: str) -> RunState:
        try:
            return self.states[run_id]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    def start(self, coroutine: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)


def create_app(service: ChatService | None = None) -> FastAPI:
    service = service or create_chat_service_from_env()
    runs = RunRegistry()
    app = FastAPI(title="yyybot API", version="0.1.0")
    app.state.chat_service = service
    app.state.runs = runs
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(WorkspaceError)
    async def workspace_error_handler(
        request: Request, exc: WorkspaceError
    ) -> JSONResponse:
        status = 404 if isinstance(exc, WorkspaceNotFoundError) else 400
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    @app.exception_handler(SessionError)
    async def session_error_handler(
        request: Request, exc: SessionError
    ) -> JSONResponse:
        status = 404 if isinstance(exc, SessionNotFoundError) else 400
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/workspaces")
    async def list_workspaces() -> list[dict[str, Any]]:
        return [_workspace_data(item) for item in service.list_workspaces()]

    @app.post("/api/workspaces", status_code=201)
    async def create_workspace(payload: WorkspaceCreate) -> dict[str, Any]:
        return _workspace_data(
            service.create_workspace(
                name=payload.name,
                workspace_id=payload.workspace_id,
                workspace_type=payload.workspace_type,
            )
        )

    @app.get("/api/workspaces/{workspace_id}/sessions")
    async def list_sessions(workspace_id: str) -> list[dict[str, Any]]:
        return [_session_summary(item) for item in service.list_sessions(workspace_id)]

    @app.post("/api/workspaces/{workspace_id}/sessions", status_code=201)
    async def create_session(
        workspace_id: str, payload: SessionCreate
    ) -> dict[str, Any]:
        return _session_data(
            service.create_session(
                workspace_id,
                title=payload.title,
                system_prompt=payload.system_prompt,
                session_id=payload.session_id,
            )
        )

    @app.get("/api/workspaces/{workspace_id}/sessions/{session_id}")
    async def get_session(workspace_id: str, session_id: str) -> dict[str, Any]:
        return _session_data(service.get_session(workspace_id, session_id))

    @app.post(
        "/api/workspaces/{workspace_id}/sessions/{session_id}/runs",
        status_code=202,
    )
    async def start_run(
        workspace_id: str,
        session_id: str,
        payload: RunCreate,
    ) -> dict[str, str]:
        service.get_session(workspace_id, session_id)
        state = runs.create()

        async def execute() -> None:
            async def on_event(event: AgentEvent) -> None:
                await state.publish(event.type, dict(event.data))

            try:
                result = await service.run(
                    workspace_id,
                    session_id,
                    payload.prompt,
                    on_event=on_event,
                )
                await state.publish("final", _result_data(result))
            except asyncio.CancelledError:
                await state.publish("error", {"message": "Run was cancelled"})
                raise
            except Exception as exc:
                await state.publish(
                    "error",
                    {"message": str(exc), "error_type": type(exc).__name__},
                )
            finally:
                await state.finish()

        runs.start(execute())
        return {
            "run_id": state.run_id,
            "events_url": f"/api/runs/{state.run_id}/events",
        }

    @app.get("/api/runs/{run_id}/events")
    async def run_events(
        run_id: str,
        request: Request,
        after: int = 0,
    ) -> StreamingResponse:
        state = runs.get(run_id)

        async def event_stream():
            async for event in state.stream(after):
                if await request.is_disconnected():
                    return
                payload = json.dumps(
                    {"type": event.event_type, "data": event.data},
                    ensure_ascii=False,
                )
                yield f"id: {event.event_id}\ndata: {payload}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    web_dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if web_dist.is_dir():
        assets = web_dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def frontend(path: str) -> FileResponse:
            return FileResponse(web_dist / "index.html")

    return app


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "Install server support with `pip install -e '.[server]'`"
        ) from exc
    uvicorn.run(
        "yyybot.api:create_app",
        factory=True,
        host=os.getenv("YYYBOT_HOST", "127.0.0.1"),
        port=int(os.getenv("YYYBOT_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
