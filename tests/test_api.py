from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from httpx import ASGITransport, AsyncClient

from yyybot import (
    Agent,
    Message,
    Model,
    ModelDelta,
    ModelResponse,
    ModelStreamEvent,
    Provider,
)
from yyybot.api import create_app, main
from yyybot.service import ChatService
from yyybot.speech import SpeechAudio, SpeechService, Voice
from yyybot.workspace import WorkspaceManager


def test_server_listens_on_all_interfaces_by_default(monkeypatch):
    observed = {}

    monkeypatch.delenv("YYYBOT_HOST", raising=False)
    monkeypatch.delenv("YYYBOT_PORT", raising=False)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: observed.update(kwargs))

    main()

    assert observed["host"] == "0.0.0.0"
    assert observed["port"] == 8000


def test_server_listen_address_can_be_overridden(monkeypatch):
    observed = {}

    monkeypatch.setenv("YYYBOT_HOST", "127.0.0.1")
    monkeypatch.setenv("YYYBOT_PORT", "8080")
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: observed.update(kwargs))

    main()

    assert observed["host"] == "127.0.0.1"
    assert observed["port"] == 8080


class ApiProvider(Provider):
    async def complete(self, *, model_id, messages, tools=(), options=None):
        return ModelResponse(
            Message(
                role="assistant",
                content="API answer",
                reasoning_content="API thought",
            ),
            "stop",
            {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        )

    async def stream(self, *, model_id, messages, tools=(), options=None):
        yield ModelStreamEvent(delta=ModelDelta(reasoning_content="API thought"))
        yield ModelStreamEvent(delta=ModelDelta(content="API answer"))
        yield ModelStreamEvent(
            response=await self.complete(
                model_id=model_id,
                messages=messages,
                tools=tools,
                options=options,
            )
        )


class BlockingApiProvider(Provider):
    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def complete(self, *, model_id, messages, tools=(), options=None):
        raise AssertionError("Cancellation test must use streaming")

    async def stream(self, *, model_id, messages, tools=(), options=None):
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if False:  # pragma: no cover - make this an async generator
            yield ModelStreamEvent(delta=ModelDelta(content="unreachable"))

def test_api_creates_storage_runs_agent_and_replays_sse(tmp_path):
    model = Model("api-model", ApiProvider())
    service = ChatService(
        WorkspaceManager(tmp_path),
        lambda on_event: Agent(model, on_event=on_event),
    )

    async def exercise_api():
        transport = ASGITransport(app=create_app(service))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            workspace_response = await client.post(
                "/api/workspaces",
                json={"name": "API workspace", "workspace_id": "api"},
            )
            assert workspace_response.status_code == 201

            session_response = await client.post(
                "/api/workspaces/api/sessions",
                json={"title": "API session", "session_id": "chat"},
            )
            assert session_response.status_code == 201

            run_response = await client.post(
                "/api/workspaces/api/sessions/chat/runs",
                json={"prompt": "hello"},
            )
            assert run_response.status_code == 202
            events_response = await client.get(run_response.json()["events_url"])

            assert events_response.status_code == 200
            payloads = [
                json.loads(line.removeprefix("data: "))
                for line in events_response.text.splitlines()
                if line.startswith("data: ")
            ]
            assert [payload["type"] for payload in payloads] == [
                "model_start",
                "model_delta",
                "model_delta",
                "model_end",
                "final",
            ]
            assert payloads[1]["data"]["reasoning_content"] == "API thought"
            assert payloads[2]["data"]["content"] == "API answer"
            final = payloads[-1]["data"]
            assert final["final_message"]["content"] == "API answer"
            assert final["final_message"]["reasoning_content"] == "API thought"
            assert final["model_turns"] == 1
            assert final["usage"] == {
                "input_tokens": 2,
                "output_tokens": 3,
                "total_tokens": 5,
            }
            assert len(final["messages"]) == 3
            assert len(final["responses"]) == 1

            persisted = (await client.get("/api/workspaces/api/sessions/chat")).json()
            assert persisted["turn_count"] == 1
            assert persisted["messages"][-1]["content"] == "API answer"

    asyncio.run(exercise_api())


def test_api_cancels_active_run_without_persisting_partial_turn(tmp_path):
    provider = BlockingApiProvider()
    model = Model("api-model", provider)
    service = ChatService(
        WorkspaceManager(tmp_path),
        lambda on_event: Agent(model, on_event=on_event),
    )

    async def exercise_api():
        transport = ASGITransport(app=create_app(service))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/workspaces",
                json={"name": "API workspace", "workspace_id": "api"},
            )
            await client.post(
                "/api/workspaces/api/sessions",
                json={"title": "API session", "session_id": "chat"},
            )
            run = (
                await client.post(
                    "/api/workspaces/api/sessions/chat/runs",
                    json={"prompt": "please stop"},
                )
            ).json()
            await asyncio.wait_for(provider.started.wait(), timeout=1)

            cancelled = await client.delete(f"/api/runs/{run['run_id']}")
            repeated = await client.delete(f"/api/runs/{run['run_id']}")
            events = await client.get(run["events_url"])
            session = await client.get("/api/workspaces/api/sessions/chat")

        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        assert repeated.json()["status"] == "cancelled"
        assert provider.cancelled.is_set()
        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in events.text.splitlines()
            if line.startswith("data: ")
        ]
        assert [payload["type"] for payload in payloads] == [
            "model_start",
            "cancelled",
        ]
        assert session.json()["turn_count"] == 0
        assert session.json()["messages"] == []

    asyncio.run(exercise_api())


def test_api_reports_missing_workspace_as_404(tmp_path):
    model = Model("api-model", ApiProvider())
    service = ChatService(
        WorkspaceManager(tmp_path),
        lambda on_event: Agent(model, on_event=on_event),
    )

    async def request_missing_workspace():
        transport = ASGITransport(app=create_app(service))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/api/workspaces/missing/sessions")

    response = asyncio.run(request_missing_workspace())
    assert response.status_code == 404
    assert "missing" in response.json()["detail"]


class FakeSpeechProvider:
    voices = (Voice("calm", "Calm"), Voice("bright", "Bright"))
    default_voice = "calm"
    voice_upload_enabled = False

    def __init__(self):
        self.transcriptions = []
        self.syntheses = []

    async def transcribe(
        self, audio, *, filename, content_type, language=None
    ):
        self.transcriptions.append((audio, filename, content_type, language))
        return "转写后的文字"

    async def synthesize(self, text, *, voice):
        self.syntheses.append((text, voice))
        return SpeechAudio(b"audio-data", "audio/mpeg")


def test_api_exposes_speech_adapter_without_changing_agent_flow(tmp_path):
    model = Model("api-model", ApiProvider())
    service = ChatService(
        WorkspaceManager(tmp_path),
        lambda on_event: Agent(model, on_event=on_event),
    )
    provider = FakeSpeechProvider()
    speech_service = SpeechService(transcriber=provider, synthesizer=provider)

    async def exercise_audio_api():
        transport = ASGITransport(app=create_app(service, speech_service))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            config = await client.get("/api/audio/config")
            transcription = await client.post(
                "/api/audio/transcriptions?filename=voice.webm&language=zh",
                content=b"recorded-audio",
                headers={"Content-Type": "audio/webm;codecs=opus"},
            )
            synthesis = await client.post(
                "/api/audio/speech",
                json={"text": "模型回答", "voice": "bright"},
            )

        assert config.json() == {
            "transcription_enabled": True,
            "synthesis_enabled": True,
            "streaming_enabled": False,
            "sample_rate": None,
            "voice_upload_enabled": False,
            "default_voice": "calm",
            "voices": [
                {"voice_id": "calm", "name": "Calm", "custom": False},
                {"voice_id": "bright", "name": "Bright", "custom": False},
            ],
        }
        assert transcription.json() == {"text": "转写后的文字"}
        assert synthesis.content == b"audio-data"
        assert synthesis.headers["content-type"] == "audio/mpeg"

    asyncio.run(exercise_audio_api())

    assert provider.transcriptions == [
        (b"recorded-audio", "voice.webm", "audio/webm", "zh")
    ]
    assert provider.syntheses == [("模型回答", "bright")]


def test_api_reports_speech_disabled_when_no_adapter_is_injected(tmp_path):
    model = Model("api-model", ApiProvider())
    service = ChatService(
        WorkspaceManager(tmp_path),
        lambda on_event: Agent(model, on_event=on_event),
    )

    async def read_config():
        transport = ASGITransport(app=create_app(service))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/api/audio/config")

    response = asyncio.run(read_config())
    assert response.json() == {
        "transcription_enabled": False,
        "synthesis_enabled": False,
        "streaming_enabled": False,
        "sample_rate": None,
        "voice_upload_enabled": False,
        "default_voice": None,
        "voices": [],
    }


def test_api_uploads_and_deletes_authorized_personal_voice(tmp_path):
    model = Model("api-model", ApiProvider())
    service = ChatService(
        WorkspaceManager(tmp_path),
        lambda on_event: Agent(model, on_event=on_event),
    )

    class FakeManagedProvider(FakeSpeechProvider):
        voice_upload_enabled = True

        def __init__(self):
            super().__init__()
            self.uploads = []
            self.deleted = []

        async def add_voice(self, **options):
            self.uploads.append(options)
            return Voice("personal-test", "我的声音 · 个人音色", custom=True)

        async def delete_voice(self, voice_id):
            self.deleted.append(voice_id)

    provider = FakeManagedProvider()
    speech_service = SpeechService(synthesizer=provider)

    async def exercise_voice_api():
        transport = ASGITransport(app=create_app(service, speech_service))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            rejected = await client.post(
                "/api/audio/voices?name=我的声音&filename=sample.wav",
                content=b"audio",
                headers={"Content-Type": "audio/wav"},
            )
            uploaded = await client.post(
                "/api/audio/voices?name=我的声音&filename=sample.wav"
                "&reference_text=参考文字&consent_confirmed=true",
                content=b"audio",
                headers={"Content-Type": "audio/wav"},
            )
            deleted = await client.delete("/api/audio/voices/personal-test")
        return rejected, uploaded, deleted

    rejected, uploaded, deleted = asyncio.run(exercise_voice_api())
    assert rejected.status_code == 400
    assert uploaded.status_code == 201
    assert uploaded.json() == {
        "voice_id": "personal-test",
        "name": "我的声音 · 个人音色",
        "custom": True,
    }
    assert deleted.status_code == 204
    assert provider.uploads == [
        {
            "name": "我的声音",
            "audio": b"audio",
            "filename": "sample.wav",
            "content_type": "audio/wav",
            "reference_text": "参考文字",
        }
    ]
    assert provider.deleted == ["personal-test"]
