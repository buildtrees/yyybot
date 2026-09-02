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
from yyybot.api import create_app
from yyybot.service import ChatService
from yyybot.workspace import WorkspaceManager


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
