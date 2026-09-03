"""FastAPI gateway for the yyybot web interface."""

from __future__ import annotations

import asyncio
import json
import os
import queue
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Coroutine
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .contracts import AgentEvent, AgentResult, Message, ModelResponse
from .runtime import create_chat_service_from_env, create_speech_service_from_env
from .service import ChatService
from .session import Session, SessionError, SessionNotFoundError
from .workspace import (
    Workspace,
    WorkspaceError,
    WorkspaceNotFoundError,
    WorkspaceType,
)
from .speech import SpeechError, SpeechProviderError, SpeechService, Voice

_MAX_AUDIO_BYTES = 25 * 1024 * 1024
_SUPPORTED_AUDIO_TYPES = {
    "audio/aac",
    "audio/flac",
    "audio/m4a",
    "audio/mp3",
    "audio/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/wav",
    "audio/webm",
    "audio/x-m4a",
    "audio/x-wav",
    "video/mp4",
    "video/webm",
}


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


class SpeechCreate(BaseModel):
    text: str = Field(min_length=1, max_length=4_096)
    voice: str | None = Field(default=None, min_length=1, max_length=64)


def _voice_data(voice: Voice) -> dict[str, Any]:
    return {
        "voice_id": voice.voice_id,
        "name": voice.name,
        "custom": voice.custom,
    }


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
    cancelled: bool = False
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)

    async def publish(self, event_type: str, data: dict[str, Any]) -> None:
        async with self.condition:
            self.events.append(RunEvent(len(self.events) + 1, event_type, data))
            self.condition.notify_all()

    async def finish(self) -> None:
        async with self.condition:
            self.done = True
            self.condition.notify_all()

    async def mark_cancelled(self) -> None:
        async with self.condition:
            if self.cancelled:
                return
            self.cancelled = True
            self.events.append(
                RunEvent(len(self.events) + 1, "cancelled", {})
            )
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
        self.tasks: dict[str, asyncio.Task[None]] = {}

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

    def start(
        self,
        state: RunState,
        coroutine: Coroutine[Any, Any, None],
    ) -> None:
        task = asyncio.create_task(coroutine)
        self.tasks[state.run_id] = task
        task.add_done_callback(
            lambda _task: self.tasks.pop(state.run_id, None)
        )

    async def cancel(self, run_id: str) -> str:
        state = self.get(run_id)
        if state.cancelled:
            return "cancelled"
        task = self.tasks.get(run_id)
        if state.done or task is None:
            return "completed"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        if not state.done:
            await state.mark_cancelled()
            await state.finish()
        return "cancelled" if state.cancelled else "completed"


def create_app(
    service: ChatService | None = None,
    speech_service: SpeechService | None = None,
) -> FastAPI:
    if service is None:
        service = create_chat_service_from_env()
        if speech_service is None:
            speech_service = create_speech_service_from_env()
    runs = RunRegistry()
    app = FastAPI(title="yyybot API", version="0.1.0")
    app.state.chat_service = service
    app.state.speech_service = speech_service
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

    @app.exception_handler(SpeechError)
    async def speech_error_handler(
        request: Request, exc: SpeechError
    ) -> JSONResponse:
        status = 502 if isinstance(exc, SpeechProviderError) else 503
        return JSONResponse(status_code=status, content={"detail": str(exc)})

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/audio/config")
    async def audio_config() -> dict[str, Any]:
        return {
            "transcription_enabled": bool(
                speech_service and speech_service.transcription_enabled
            ),
            "synthesis_enabled": bool(
                speech_service and speech_service.synthesis_enabled
            ),
            "streaming_enabled": bool(
                speech_service and speech_service.streaming_enabled
            ),
            "sample_rate": (
                speech_service.synthesis_sample_rate if speech_service else None
            ),
            "voice_upload_enabled": bool(
                speech_service and speech_service.voice_upload_enabled
            ),
            "default_voice": (
                speech_service.default_voice if speech_service else None
            ),
            "voices": [
                _voice_data(voice)
                for voice in (speech_service.voices if speech_service else ())
            ],
        }

    @app.post("/api/audio/voices", status_code=201)
    async def upload_voice(
        request: Request,
        name: str,
        filename: str = "reference.wav",
        reference_text: str | None = None,
        consent_confirmed: bool = False,
    ) -> dict[str, Any]:
        if speech_service is None or not speech_service.voice_upload_enabled:
            raise HTTPException(
                status_code=503,
                detail="Custom voice upload is not configured",
            )
        clean_name = name.strip()
        if not clean_name or len(clean_name) > 80:
            raise HTTPException(status_code=400, detail="Invalid voice name")
        if reference_text is not None and len(reference_text) > 1_000:
            raise HTTPException(status_code=400, detail="Reference text is too long")
        if not consent_confirmed:
            raise HTTPException(
                status_code=400,
                detail="Voice owner consent must be confirmed",
            )
        content_type = request.headers.get("content-type", "").split(";", 1)[0]
        if content_type not in _SUPPORTED_AUDIO_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported audio format")
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Reference audio is too large")
        audio = await request.body()
        if not audio:
            raise HTTPException(status_code=400, detail="Reference audio is empty")
        if len(audio) > _MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Reference audio is too large")
        voice = await speech_service.add_voice(
            name=clean_name,
            audio=audio,
            filename=Path(filename).name or "reference.wav",
            content_type=content_type,
            reference_text=reference_text,
        )
        return _voice_data(voice)

    @app.delete("/api/audio/voices/{voice_id}", status_code=204)
    async def delete_voice(voice_id: str) -> Response:
        if speech_service is None or not speech_service.voice_upload_enabled:
            raise HTTPException(
                status_code=503,
                detail="Custom voice upload is not configured",
            )
        await speech_service.delete_voice(voice_id)
        return Response(status_code=204)

    @app.post("/api/audio/transcriptions")
    async def transcribe_audio(
        request: Request,
        filename: str = "recording.webm",
        language: str | None = None,
    ) -> dict[str, str]:
        if speech_service is None or not speech_service.transcription_enabled:
            raise HTTPException(
                status_code=503,
                detail="Speech transcription is not configured",
            )
        content_type = request.headers.get("content-type", "").split(";", 1)[0]
        if content_type not in _SUPPORTED_AUDIO_TYPES:
            raise HTTPException(status_code=415, detail="Unsupported audio format")
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio recording is too large")
        audio = await request.body()
        if not audio:
            raise HTTPException(status_code=400, detail="Audio recording is empty")
        if len(audio) > _MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio recording is too large")
        safe_filename = Path(filename).name or "recording.webm"
        text = await speech_service.transcribe(
            audio,
            filename=safe_filename,
            content_type=content_type,
            language=language,
        )
        return {"text": text}

    @app.post("/api/audio/speech")
    async def synthesize_speech(payload: SpeechCreate) -> Response:
        if speech_service is None or not speech_service.synthesis_enabled:
            raise HTTPException(
                status_code=503,
                detail="Speech synthesis is not configured",
            )
        text = payload.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Speech text is empty")
        audio = await speech_service.synthesize(
            text,
            voice=payload.voice,
        )
        return Response(
            content=audio.data,
            media_type=audio.media_type,
            headers={"Cache-Control": "no-store"},
        )

    @app.websocket("/api/audio/speech/stream")
    async def stream_speech(websocket: WebSocket) -> None:
        await websocket.accept()
        if speech_service is None or not speech_service.streaming_enabled:
            await websocket.send_json(
                {"type": "error", "message": "Streaming speech is not configured"}
            )
            await websocket.close(code=1011)
            return

        stop = object()
        text_queue: queue.Queue[str | object] = queue.Queue()
        output_queue: asyncio.Queue[bytes | Exception | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        producer: asyncio.Task[None] | None = None
        receiver: asyncio.Task[None] | None = None

        def text_generator():
            while True:
                item = text_queue.get()
                if item is stop:
                    return
                yield str(item)

        try:
            start = await websocket.receive_json()
            if start.get("type") != "start":
                raise ValueError("First speech stream message must be 'start'")
            voice = start.get("voice") or speech_service.default_voice
            available = {item.voice_id for item in speech_service.voices}
            if voice not in available:
                raise ValueError(f"Unsupported voice: {voice}")

            def produce() -> None:
                try:
                    for chunk in speech_service.stream_synthesize(
                        text_generator(),
                        voice=str(voice),
                    ):
                        loop.call_soon_threadsafe(
                            output_queue.put_nowait,
                            chunk.data,
                        )
                except Exception as exc:
                    loop.call_soon_threadsafe(output_queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(output_queue.put_nowait, None)

            async def receive_text() -> None:
                total_length = 0
                try:
                    while True:
                        message = await websocket.receive_json()
                        message_type = message.get("type")
                        if message_type == "end":
                            return
                        if message_type != "text":
                            raise ValueError("Unsupported speech stream message")
                        text = str(message.get("text") or "")
                        if not text:
                            continue
                        total_length += len(text)
                        if total_length > 4_096:
                            raise ValueError("Speech text is too long")
                        text_queue.put(text)
                except WebSocketDisconnect:
                    pass
                except Exception as exc:
                    output_queue.put_nowait(exc)
                finally:
                    text_queue.put(stop)

            await websocket.send_json(
                {
                    "type": "ready",
                    "sample_rate": speech_service.synthesis_sample_rate,
                    "format": "pcm_s16le",
                    "channels": 1,
                }
            )
            producer = asyncio.create_task(asyncio.to_thread(produce))
            receiver = asyncio.create_task(receive_text())
            while True:
                output = await output_queue.get()
                if output is None:
                    break
                if isinstance(output, Exception):
                    raise output
                await websocket.send_bytes(output)
            await websocket.send_json({"type": "done"})
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            with suppress(Exception):
                await websocket.send_json({"type": "error", "message": str(exc)})
        finally:
            text_queue.put(stop)
            if receiver is not None and not receiver.done():
                receiver.cancel()
            if receiver is not None:
                with suppress(asyncio.CancelledError, WebSocketDisconnect):
                    await receiver
            if producer is not None:
                with suppress(asyncio.CancelledError):
                    await producer
            with suppress(Exception):
                await websocket.close()

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
                await state.mark_cancelled()
                raise
            except Exception as exc:
                await state.publish(
                    "error",
                    {"message": str(exc), "error_type": type(exc).__name__},
                )
            finally:
                await state.finish()

        runs.start(state, execute())
        return {
            "run_id": state.run_id,
            "events_url": f"/api/runs/{state.run_id}/events",
        }

    @app.delete("/api/runs/{run_id}")
    async def cancel_run(run_id: str) -> dict[str, str]:
        status = await runs.cancel(run_id)
        return {"run_id": run_id, "status": status}

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
        host=os.getenv("YYYBOT_HOST", "0.0.0.0"),
        port=int(os.getenv("YYYBOT_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
