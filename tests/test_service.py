from __future__ import annotations

import asyncio

from yyybot import Agent, Message, Model, ModelResponse, Provider
from yyybot.service import ChatService
from yyybot.workspace import WorkspaceManager


class EchoProvider(Provider):
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.requests: list[tuple[Message, ...]] = []

    async def complete(self, *, model_id, messages, tools=(), options=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.requests.append(tuple(messages))
        await asyncio.sleep(0.01)
        self.active -= 1
        user = next(message for message in reversed(messages) if message.role == "user")
        return ModelResponse(
            Message(role="assistant", content=f"answer:{user.content}"),
            "stop",
            {"total_tokens": 3},
        )


def test_service_serializes_runs_and_persists_fresh_session_context(tmp_path):
    provider = EchoProvider()
    model = Model("test-model", provider)
    service = ChatService(
        WorkspaceManager(tmp_path),
        lambda on_event: Agent(model, on_event=on_event),
    )
    workspace = service.create_workspace(name="Test", workspace_id="test")
    service.create_session(workspace.workspace_id, session_id="chat")

    async def run_both():
        return await asyncio.gather(
            service.run("test", "chat", "first"),
            service.run("test", "chat", "second"),
        )

    first, second = asyncio.run(run_both())

    assert first.output == "answer:first"
    assert second.output == "answer:second"
    assert provider.max_active == 1
    assert Message(role="assistant", content="answer:first") in provider.requests[1]
    persisted = service.get_session("test", "chat")
    assert len(persisted.turns) == 2
    assert persisted.messages[-1].content == "answer:second"
