"""Filesystem workspaces that isolate sessions and future runtime data."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from .session import SessionManager

WorkspaceType = Literal["personal", "team"]

_SCHEMA_VERSION = 1
_WORKSPACE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


class WorkspaceError(RuntimeError):
    """Base error for workspace operations."""


class WorkspaceNotFoundError(WorkspaceError):
    """Raised when a requested workspace does not exist."""


class WorkspaceFormatError(WorkspaceError):
    """Raised when workspace metadata is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class Workspace:
    """An immutable snapshot of a workspace and its storage locations."""

    workspace_id: str
    name: str
    workspace_type: WorkspaceType
    created_at: str
    directory: Path
    owner_account_id: str | None = None

    @property
    def sessions_directory(self) -> Path:
        return self.directory / "sessions"


class WorkspaceManager:
    """Create, list, and load workspaces below the yyybot home directory.

    The manager has no globally selected workspace. The CLI or UI keeps the
    selected workspace id, leaving account membership and authorization to a
    future identity layer.
    """

    def __init__(self, home: str | Path | None = None) -> None:
        self.home = Path(home).expanduser() if home is not None else self.default_home()
        self.directory = self.home / "workspaces"

    @staticmethod
    def default_home() -> Path:
        return Path.home() / ".yyybot"

    @staticmethod
    def _validate_workspace_id(workspace_id: str) -> str:
        if not _WORKSPACE_ID_PATTERN.fullmatch(workspace_id):
            raise ValueError(
                "workspace_id must contain only letters, numbers, '-' or '_'"
            )
        return workspace_id

    def _path(self, workspace_id: str) -> Path:
        return self.directory / self._validate_workspace_id(workspace_id)

    @staticmethod
    def _decode_metadata(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise WorkspaceNotFoundError(
                f"Workspace metadata not found: {path.parent.name}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise WorkspaceFormatError(
                f"Invalid workspace metadata in {path}: {exc.msg}"
            ) from exc
        except OSError as exc:
            raise WorkspaceError(f"Could not read workspace metadata: {exc}") from exc
        if not isinstance(data, dict):
            raise WorkspaceFormatError("Workspace metadata must be a JSON object")
        return data

    def create(
        self,
        *,
        name: str,
        workspace_id: str | None = None,
        workspace_type: WorkspaceType = "personal",
        owner_account_id: str | None = None,
    ) -> Workspace:
        workspace_id = self._validate_workspace_id(
            uuid4().hex if workspace_id is None else workspace_id
        )
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        name = name.strip()
        if not name:
            raise ValueError("name cannot be empty")
        if workspace_type not in {"personal", "team"}:
            raise ValueError("workspace_type must be 'personal' or 'team'")
        if owner_account_id is not None and not isinstance(owner_account_id, str):
            raise TypeError("owner_account_id must be a string or None")

        workspace_directory = self._path(workspace_id)
        sessions_directory = workspace_directory / "sessions"
        created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        metadata = {
            "schema_version": _SCHEMA_VERSION,
            "workspace_id": workspace_id,
            "name": name,
            "workspace_type": workspace_type,
            "owner_account_id": owner_account_id,
            "created_at": created_at,
        }

        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            workspace_directory.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise WorkspaceError(f"Workspace already exists: {workspace_id}") from exc
        try:
            sessions_directory.mkdir(mode=0o700)
            metadata_path = workspace_directory / "workspace.json"
            descriptor = os.open(
                metadata_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(metadata, file, ensure_ascii=False, separators=(",", ":"))
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
        except OSError as exc:
            metadata_path = workspace_directory / "workspace.json"
            try:
                metadata_path.unlink(missing_ok=True)
                sessions_directory.rmdir()
                workspace_directory.rmdir()
            except OSError:
                pass
            raise WorkspaceError(
                f"Could not create workspace {workspace_id}: {exc}"
            ) from exc
        return Workspace(
            workspace_id=workspace_id,
            name=name,
            workspace_type=workspace_type,
            created_at=created_at,
            directory=workspace_directory,
            owner_account_id=owner_account_id,
        )

    def load(self, workspace_id: str) -> Workspace:
        workspace_directory = self._path(workspace_id)
        data = self._decode_metadata(workspace_directory / "workspace.json")
        if data.get("schema_version") != _SCHEMA_VERSION:
            raise WorkspaceFormatError(
                f"Unsupported workspace schema version: {workspace_id}"
            )
        if data.get("workspace_id") != workspace_id:
            raise WorkspaceFormatError("Workspace id does not match its directory")
        name = data.get("name")
        workspace_type = data.get("workspace_type")
        created_at = data.get("created_at")
        owner_account_id = data.get("owner_account_id")
        if not isinstance(name, str) or not isinstance(created_at, str):
            raise WorkspaceFormatError("Workspace name and created_at must be strings")
        if workspace_type not in {"personal", "team"}:
            raise WorkspaceFormatError("Invalid persisted workspace_type")
        if owner_account_id is not None and not isinstance(owner_account_id, str):
            raise WorkspaceFormatError("Workspace owner_account_id must be a string")
        if not (workspace_directory / "sessions").is_dir():
            raise WorkspaceFormatError("Workspace sessions directory is missing")
        return Workspace(
            workspace_id=workspace_id,
            name=name,
            workspace_type=workspace_type,
            created_at=created_at,
            directory=workspace_directory,
            owner_account_id=owner_account_id,
        )

    def list(self) -> tuple[Workspace, ...]:
        if not self.directory.exists():
            return ()
        workspaces = [
            self.load(path.name) for path in self.directory.iterdir() if path.is_dir()
        ]
        return tuple(
            sorted(workspaces, key=lambda workspace: workspace.created_at, reverse=True)
        )

    def ensure_default(self) -> Workspace:
        try:
            return self.load("default")
        except WorkspaceNotFoundError:
            return self.create(name="Default Workspace", workspace_id="default")

    def sessions(self, workspace_id: str) -> SessionManager:
        workspace = self.load(workspace_id)
        return SessionManager(workspace.sessions_directory)
