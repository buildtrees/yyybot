"""Application service shared by CLI and HTTP frontends."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

from .agent import Agent, EventHandler
from .context import ConversationContext
from .contracts import AgentResult, Message
from .session import Session, SessionManager
from .workspace import Workspace, WorkspaceManager, WorkspaceType

AgentFactory = Callable[[EventHandler | None], Agent]


class ChatService:
    """Coordinate workspaces, sessions, context assembly, and agent runs.

    A lock serializes runs targeting the same session so two browser requests
    cannot both append from the same stale history snapshot.
    """

    def __init__(
        self,
        workspaces: WorkspaceManager,
        agent_factory: AgentFactory,
    ) -> None:
        self.workspaces = workspaces
        self._agent_factory = agent_factory
        self._session_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _session_lock(self, workspace_id: str, session_id: str) -> asyncio.Lock:
        key = (workspace_id, session_id)
        async with self._locks_guard:
            return self._session_locks.setdefault(key, asyncio.Lock())

    def list_workspaces(self) -> tuple[Workspace, ...]:
        return self.workspaces.list()

    def get_workspace(self, workspace_id: str) -> Workspace:
        return self.workspaces.load(workspace_id)

    def create_workspace(
        self,
        *,
        name: str,
        workspace_id: str | None = None,
        workspace_type: WorkspaceType = "personal",
        owner_account_id: str | None = None,
    ) -> Workspace:
        return self.workspaces.create(
            name=name,
            workspace_id=workspace_id,
            workspace_type=workspace_type,
            owner_account_id=owner_account_id,
        )

    def ensure_default_workspace(self) -> Workspace:
        return self.workspaces.ensure_default()

    def session_manager(self, workspace_id: str) -> SessionManager:
        return self.workspaces.sessions(workspace_id)

    def list_sessions(self, workspace_id: str) -> tuple[Session, ...]:
        return self.session_manager(workspace_id).list()

    def get_session(self, workspace_id: str, session_id: str) -> Session:
        return self.session_manager(workspace_id).load(session_id)

    def create_session(
        self,
        workspace_id: str,
        *,
        title: str = "New session",
        system_prompt: str | None = "You are a helpful personal assistant.",
        session_id: str | None = None,
    ) -> Session:
        return self.session_manager(workspace_id).create(
            title=title,
            system_prompt=system_prompt,
            session_id=session_id,
        )

    async def run_once(
        self,
        prompt: str,
        *,
        on_event: EventHandler | None = None,
    ) -> AgentResult:
        user_message = Message(role="user", content=prompt)
        messages = ConversationContext().build(user_message)
        agent: Agent = self._agent_factory(on_event)
        return await agent.run(messages)

    async def run(
        self,
        workspace_id: str,
        session_id: str,
        prompt: str,
        *,
        on_event: EventHandler | None = None,
        incoming: Sequence[Message] | None = None,
    ) -> AgentResult:
        lock = await self._session_lock(workspace_id, session_id)
        async with lock:
            sessions = self.session_manager(workspace_id)
            session = sessions.load(session_id)
            current = tuple(
                (Message(role="user", content=prompt),)
                if incoming is None
                else incoming
            )
            context = ConversationContext(
                system_prompt=session.system_prompt,
                history=session.messages,
            )
            messages = context.build(*current)
            agent: Agent = self._agent_factory(on_event)
            result = await agent.run(messages)
            sessions.append_turn(session_id, incoming=current, result=result)
            return result
