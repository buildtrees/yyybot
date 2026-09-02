from __future__ import annotations

import json

import pytest

from yyybot import (
    AgentResult,
    Message,
    ModelResponse,
    WorkspaceError,
    WorkspaceManager,
    WorkspaceNotFoundError,
)


def test_workspace_owns_its_sessions_directory(tmp_path):
    home = tmp_path / ".yyybot"
    manager = WorkspaceManager(home)

    workspace = manager.create(
        workspace_id="research",
        name="Research",
        workspace_type="team",
        owner_account_id="account-1",
    )

    assert workspace.directory == home / "workspaces" / "research"
    assert workspace.sessions_directory == workspace.directory / "sessions"
    assert workspace.sessions_directory.is_dir()
    assert manager.load("research") == workspace
    assert manager.list() == (workspace,)

    metadata = json.loads((workspace.directory / "workspace.json").read_text())
    assert metadata["workspace_id"] == "research"
    assert metadata["workspace_type"] == "team"
    assert metadata["owner_account_id"] == "account-1"


def test_workspace_session_manager_persists_inside_selected_workspace(tmp_path):
    workspaces = WorkspaceManager(tmp_path / ".yyybot")
    workspace = workspaces.create(workspace_id="personal", name="Personal")
    sessions = workspaces.sessions(workspace.workspace_id)
    session = sessions.create(session_id="conversation", title="Conversation")
    user = Message(role="user", content="Hello")
    model_input = sessions.load_context(session.session_id).build(user)
    final = Message(role="assistant", content="Hi")
    result = AgentResult(
        final_message=final,
        messages=(*model_input, final),
        responses=(ModelResponse(final, "stop", {"total_tokens": 3}),),
    )

    sessions.append_turn(session.session_id, incoming=(user,), result=result)

    session_path = workspace.sessions_directory / "conversation.jsonl"
    assert session_path.is_file()
    assert sessions.load("conversation").messages == (user, final)


def test_default_workspace_is_created_once(tmp_path):
    manager = WorkspaceManager(tmp_path / ".yyybot")

    first = manager.ensure_default()
    second = manager.ensure_default()

    assert first == second
    assert first.workspace_id == "default"
    assert first.name == "Default Workspace"


def test_workspace_ids_cannot_escape_yyybot_home(tmp_path):
    manager = WorkspaceManager(tmp_path / ".yyybot")

    with pytest.raises(ValueError, match="workspace_id"):
        manager.create(workspace_id="../outside", name="Outside")


def test_missing_and_duplicate_workspaces_are_reported(tmp_path):
    manager = WorkspaceManager(tmp_path / ".yyybot")

    with pytest.raises(WorkspaceNotFoundError):
        manager.load("missing")

    manager.create(workspace_id="existing", name="Existing")
    with pytest.raises(WorkspaceError, match="already exists"):
        manager.create(workspace_id="existing", name="Existing")
