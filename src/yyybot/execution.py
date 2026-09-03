"""Task-local execution settings shared by runtime tools."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_execution_directory: ContextVar[Path | None] = ContextVar(
    "yyybot_execution_directory",
    default=None,
)


def current_execution_directory() -> Path:
    """Return the directory assigned to this run, or the process cwd."""

    return _execution_directory.get() or Path.cwd()


@contextmanager
def use_execution_directory(directory: str | Path) -> Iterator[Path]:
    """Bind tool execution to a directory without changing global process state."""

    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Execution directory does not exist: {path}")
    token = _execution_directory.set(path)
    try:
        yield path
    finally:
        _execution_directory.reset(token)
