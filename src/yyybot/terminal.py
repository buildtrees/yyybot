"""Built-in non-interactive Bash tool."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import BinaryIO

from .tools import ToolError

_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 300
_MAX_OUTPUT_BYTES = 100_000


def _capture_stream(
    stream: BinaryIO | None,
    output: bytearray,
    truncated: list[bool],
) -> None:
    if stream is None:
        return
    while chunk := stream.read(4096):
        remaining = _MAX_OUTPUT_BYTES - len(output)
        if remaining > 0:
            output.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated[0] = True


def _run_bash(command: str, timeout: int, cwd: Path) -> dict[str, object]:
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    try:
        process = subprocess.Popen(
            ["/bin/bash", "-c", command],
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except Exception as exc:
        raise ToolError(f"Could not start Bash: {exc}") from exc

    stdout = bytearray()
    stderr = bytearray()
    stdout_truncated = [False]
    stderr_truncated = [False]
    stdout_thread = threading.Thread(
        target=_capture_stream,
        args=(process.stdout, stdout, stdout_truncated),
        name="yyybot-bash-stdout",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_capture_stream,
        args=(process.stderr, stderr, stderr_truncated),
        name="yyybot-bash-stderr",
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)
    return {
        "command": command,
        "cwd": str(cwd),
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "stdout": bytes(stdout).decode(errors="replace"),
        "stderr": bytes(stderr).decode(errors="replace"),
        "stdout_truncated": stdout_truncated[0],
        "stderr_truncated": stderr_truncated[0],
    }


async def bash(command: str, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Execute a Bash command and return its exit code, stdout, and stderr."""

    if not command.strip():
        raise ToolError("Bash command cannot be empty")
    timeout = min(max(timeout, 1), _MAX_TIMEOUT)
    cwd = Path.cwd()
    result = _run_bash(command, timeout, cwd)
    return json.dumps(result, ensure_ascii=False)
