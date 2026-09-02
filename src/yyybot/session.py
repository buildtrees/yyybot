"""Append-only JSONL conversation session persistence."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .context import ConversationContext
from .contracts import AgentResult, Message, ModelResponse, ToolCall

_SCHEMA_VERSION = 1
_DEFAULT_SYSTEM_PROMPT = "You are a helpful personal assistant."
_SESSION_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


class SessionError(RuntimeError):
    """Base error for session storage operations."""


class SessionNotFoundError(SessionError):
    """Raised when a requested session does not exist."""


class SessionFormatError(SessionError):
    """Raised when a session JSONL file is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class SessionTurn:
    """One atomically persisted user-to-agent interaction."""

    turn_id: str
    created_at: str
    incoming: tuple[Message, ...]
    generated: tuple[Message, ...]
    responses: tuple[ModelResponse, ...]

    @property
    def messages(self) -> tuple[Message, ...]:
        return (*self.incoming, *self.generated)


@dataclass(frozen=True, slots=True)
class Session:
    """An immutable snapshot reconstructed from one session JSONL file."""

    session_id: str
    title: str
    system_prompt: str | None
    created_at: str
    turns: tuple[SessionTurn, ...] = ()

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(message for turn in self.turns for message in turn.messages)

    @property
    def updated_at(self) -> str:
        return self.turns[-1].created_at if self.turns else self.created_at


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _message_to_data(message: Message) -> dict[str, Any]:
    data: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        data["tool_calls"] = [
            {
                "id": call.id,
                "name": call.name,
                "arguments": dict(call.arguments),
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        data["tool_call_id"] = message.tool_call_id
    if message.reasoning_content:
        data["reasoning_content"] = message.reasoning_content
    if message.reasoning_signature is not None:
        data["reasoning_signature"] = message.reasoning_signature
    return data


def _message_from_data(data: Any) -> Message:
    if not isinstance(data, Mapping):
        raise SessionFormatError("A persisted message must be a JSON object")
    role = data.get("role")
    if role not in {"system", "user", "assistant", "tool"}:
        raise SessionFormatError(f"Invalid persisted message role: {role!r}")
    content = data.get("content", "")
    if not isinstance(content, str):
        raise SessionFormatError("Persisted message content must be a string")
    raw_calls = data.get("tool_calls") or ()
    if not isinstance(raw_calls, list | tuple):
        raise SessionFormatError("Persisted tool_calls must be an array")
    calls: list[ToolCall] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            raise SessionFormatError("A persisted tool call must be a JSON object")
        arguments = raw_call.get("arguments") or {}
        if not isinstance(arguments, Mapping):
            raise SessionFormatError("Persisted tool arguments must be an object")
        try:
            calls.append(
                ToolCall(
                    id=str(raw_call["id"]),
                    name=str(raw_call["name"]),
                    arguments=dict(arguments),
                )
            )
        except KeyError as exc:
            raise SessionFormatError(
                f"Persisted tool call is missing {exc.args[0]!r}"
            ) from exc
    tool_call_id = data.get("tool_call_id")
    if tool_call_id is not None and not isinstance(tool_call_id, str):
        raise SessionFormatError("Persisted tool_call_id must be a string")
    reasoning_content = data.get("reasoning_content", "")
    if not isinstance(reasoning_content, str):
        raise SessionFormatError("Persisted reasoning_content must be a string")
    reasoning_signature = data.get("reasoning_signature")
    if reasoning_signature is not None and not isinstance(reasoning_signature, str):
        raise SessionFormatError("Persisted reasoning_signature must be a string")
    return Message(
        role=role,
        content=content,
        tool_calls=tuple(calls),
        tool_call_id=tool_call_id,
        reasoning_content=reasoning_content,
        reasoning_signature=reasoning_signature,
    )


def _response_to_data(response: ModelResponse) -> dict[str, Any]:
    return {
        "message": _message_to_data(response.message),
        "finish_reason": response.finish_reason,
        "usage": dict(response.usage),
    }


def _response_from_data(data: Any) -> ModelResponse:
    if not isinstance(data, Mapping):
        raise SessionFormatError("A persisted model response must be a JSON object")
    usage = data.get("usage") or {}
    if not isinstance(usage, Mapping) or not all(
        isinstance(key, str) and isinstance(value, int) for key, value in usage.items()
    ):
        raise SessionFormatError("Persisted response usage must contain integers")
    finish_reason = data.get("finish_reason")
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise SessionFormatError("Persisted finish_reason must be a string")
    if "message" not in data:
        raise SessionFormatError("Persisted model response is missing 'message'")
    return ModelResponse(
        message=_message_from_data(data["message"]),
        finish_reason=finish_reason,
        usage=dict(usage),
    )


def _messages_from_data(data: Any, field: str) -> tuple[Message, ...]:
    if not isinstance(data, list):
        raise SessionFormatError(f"Persisted {field} must be an array")
    return tuple(_message_from_data(item) for item in data)


class SessionManager:
    """Create, list, load, and append to per-session JSONL files.

    The manager deliberately has no globally selected session. A CLI or UI owns
    the current session id and calls ``load`` whenever the user selects it.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser()

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        if not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise ValueError(
                "session_id must contain only letters, numbers, '-' or '_'"
            )
        return session_id

    def _path(self, session_id: str) -> Path:
        return self.directory / f"{self._validate_session_id(session_id)}.jsonl"

    @staticmethod
    def _encode(record: Mapping[str, Any]) -> str:
        try:
            return json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise SessionError(f"Session data is not JSON serializable: {exc}") from exc

    def create(
        self,
        *,
        title: str = "New session",
        system_prompt: str | None = _DEFAULT_SYSTEM_PROMPT,
        session_id: str | None = None,
    ) -> Session:
        session_id = self._validate_session_id(
            uuid4().hex if session_id is None else session_id
        )
        created_at = _now()
        if not isinstance(title, str):
            raise TypeError("title must be a string")
        title = title.strip() or "New session"
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string or None")
        record = {
            "type": "session",
            "schema_version": _SCHEMA_VERSION,
            "session_id": session_id,
            "title": title,
            "system_prompt": system_prompt,
            "created_at": created_at,
        }
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self._path(session_id)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(self._encode(record) + "\n")
                file.flush()
                os.fsync(file.fileno())
        except FileExistsError as exc:
            raise SessionError(f"Session already exists: {session_id}") from exc
        return Session(session_id, title, system_prompt, created_at)

    def load(self, session_id: str) -> Session:
        path = self._path(session_id)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as exc:
            raise SessionNotFoundError(f"Session not found: {session_id}") from exc
        except OSError as exc:
            raise SessionError(f"Could not read session {session_id}: {exc}") from exc
        if not lines:
            raise SessionFormatError(f"Session file is empty: {path}")

        records: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(lines, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SessionFormatError(
                    f"Invalid JSON in {path.name} at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, Mapping):
                raise SessionFormatError(
                    f"Record in {path.name} at line {line_number} must be an object"
                )
            if record.get("schema_version") != _SCHEMA_VERSION:
                raise SessionFormatError(
                    f"Unsupported schema version in {path.name} at line {line_number}"
                )
            records.append(record)

        metadata = records[0]
        if metadata.get("type") != "session":
            raise SessionFormatError("The first session record must contain metadata")
        if metadata.get("session_id") != session_id:
            raise SessionFormatError("Session id does not match its filename")
        title = metadata.get("title")
        created_at = metadata.get("created_at")
        system_prompt = metadata.get("system_prompt")
        if not isinstance(title, str) or not isinstance(created_at, str):
            raise SessionFormatError("Session metadata contains invalid title or time")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise SessionFormatError("Session system_prompt must be a string or null")

        turns: list[SessionTurn] = []
        for record in records[1:]:
            if record.get("type") != "turn":
                raise SessionFormatError(
                    "Only turn records may follow session metadata"
                )
            turn_id = record.get("turn_id")
            turn_created_at = record.get("created_at")
            raw_responses = record.get("responses")
            if not isinstance(turn_id, str) or not isinstance(turn_created_at, str):
                raise SessionFormatError("Persisted turn id and time must be strings")
            if not isinstance(raw_responses, list):
                raise SessionFormatError("Persisted responses must be an array")
            turns.append(
                SessionTurn(
                    turn_id=turn_id,
                    created_at=turn_created_at,
                    incoming=_messages_from_data(record.get("incoming"), "incoming"),
                    generated=_messages_from_data(record.get("generated"), "generated"),
                    responses=tuple(
                        _response_from_data(response) for response in raw_responses
                    ),
                )
            )
        return Session(session_id, title, system_prompt, created_at, tuple(turns))

    def list(self) -> tuple[Session, ...]:
        if not self.directory.exists():
            return ()
        sessions = [self.load(path.stem) for path in self.directory.glob("*.jsonl")]
        return tuple(
            sorted(sessions, key=lambda session: session.updated_at, reverse=True)
        )

    def load_context(self, session_id: str) -> ConversationContext:
        session = self.load(session_id)
        return ConversationContext(
            system_prompt=session.system_prompt,
            history=session.messages,
        )

    def append_turn(
        self,
        session_id: str,
        *,
        incoming: Sequence[Message],
        result: AgentResult,
    ) -> SessionTurn:
        incoming = tuple(incoming)
        if not incoming or not all(
            isinstance(message, Message) for message in incoming
        ):
            raise TypeError("incoming must contain at least one Message")
        session = self.load(session_id)
        expected_input = ConversationContext(
            system_prompt=session.system_prompt,
            history=session.messages,
        ).build(*incoming)
        if result.messages[: len(expected_input)] != expected_input:
            raise SessionError(
                "Agent result does not match the selected session context"
            )
        generated = result.messages[len(expected_input) :]
        if not generated or generated[-1] != result.final_message:
            raise SessionError(
                "Agent result does not contain a generated final message"
            )

        turn = SessionTurn(
            turn_id=uuid4().hex,
            created_at=_now(),
            incoming=incoming,
            generated=generated,
            responses=result.responses,
        )
        record = {
            "type": "turn",
            "schema_version": _SCHEMA_VERSION,
            "turn_id": turn.turn_id,
            "created_at": turn.created_at,
            "incoming": [_message_to_data(message) for message in turn.incoming],
            "generated": [_message_to_data(message) for message in turn.generated],
            "responses": [_response_to_data(response) for response in turn.responses],
        }
        path = self._path(session_id)
        try:
            with path.open("a", encoding="utf-8") as file:
                file.write(self._encode(record) + "\n")
                file.flush()
                os.fsync(file.fileno())
        except OSError as exc:
            raise SessionError(
                f"Could not append to session {session_id}: {exc}"
            ) from exc
        return turn

    def delete(self, session_id: str) -> None:
        try:
            self._path(session_id).unlink()
        except FileNotFoundError as exc:
            raise SessionNotFoundError(f"Session not found: {session_id}") from exc
        except OSError as exc:
            raise SessionError(f"Could not delete session {session_id}: {exc}") from exc
