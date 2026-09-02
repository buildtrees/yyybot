"""Built-in non-interactive Bash tool."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import BinaryIO

from .tools import ToolError

_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 300
_MAX_OUTPUT_BYTES = 100_000


def _drain_stream(
    stream: BinaryIO | None,
    output: bytearray,
    truncated: list[bool],
) -> bool:
    """Drain currently available bytes and report whether EOF was reached."""

    if stream is None:
        return True
    while True:
        try:
            chunk = os.read(stream.fileno(), 4096)
        except BlockingIOError:
            return False
        if not chunk:
            return True
        remaining = _MAX_OUTPUT_BYTES - len(output)
        if remaining > 0:
            output.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated[0] = True


async def _run_bash(command: str, timeout: int, cwd: Path) -> dict[str, object]:
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

    for stream in (process.stdout, process.stderr):
        if stream is not None:
            os.set_blocking(stream.fileno(), False)

    stdout = bytearray()
    stderr = bytearray()
    stdout_truncated = [False]
    stderr_truncated = [False]
    stdout_eof = False
    stderr_eof = False
    timed_out = False
    deadline = asyncio.get_running_loop().time() + timeout

    while True:
        stdout_eof = stdout_eof or _drain_stream(
            process.stdout, stdout, stdout_truncated
        )
        stderr_eof = stderr_eof or _drain_stream(
            process.stderr, stderr, stderr_truncated
        )
        returncode = process.poll()
        if returncode is not None and stdout_eof and stderr_eof:
            break
        if asyncio.get_running_loop().time() >= deadline:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            _drain_stream(process.stdout, stdout, stdout_truncated)
            _drain_stream(process.stderr, stderr, stderr_truncated)
            break
        await asyncio.sleep(0.01)

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
    result = await _run_bash(command, timeout, Path.cwd())
    return json.dumps(result, ensure_ascii=False)
