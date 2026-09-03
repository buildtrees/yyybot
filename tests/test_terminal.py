from __future__ import annotations

import asyncio
import json

from yyybot import ToolRegistry, bash
from yyybot.execution import use_execution_directory


def test_bash_returns_stdout_stderr_and_exit_code():
    result = json.loads(
        asyncio.run(
            bash(
                "printf 'hello'; printf 'warning' >&2; exit 7",
            )
        )
    )

    assert result["exit_code"] == 7
    assert result["timed_out"] is False
    assert result["stdout"] == "hello"
    assert result["stderr"] == "warning"


def test_bash_times_out_and_kills_process_group():
    result = json.loads(asyncio.run(bash("sleep 5", timeout=1)))

    assert result["exit_code"] == -9
    assert result["timed_out"] is True


def test_bash_registers_with_expected_schema():
    registry = ToolRegistry()

    tool = registry.add(bash)

    assert tool.spec.name == "bash"
    assert tool.spec.parameters == {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer"},
        },
        "required": ["command"],
    }


def test_bash_uses_task_local_execution_directory(tmp_path):
    with use_execution_directory(tmp_path):
        result = json.loads(asyncio.run(bash("pwd")))

    assert result["cwd"] == str(tmp_path)
    assert result["stdout"].strip() == str(tmp_path)
