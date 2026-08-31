from __future__ import annotations

import asyncio

from yyybot import ToolCall, ToolRegistry


def test_sync_and_async_tools_share_one_interface():
    registry = ToolRegistry()

    def sync_tool(value: int) -> int:
        """Double a value."""
        return value * 2

    async def async_tool(value: int) -> int:
        """Triple a value."""
        return value * 3

    sync = registry.add(sync_tool)
    registry.add(async_tool)

    assert sync.spec.parameters["properties"]["value"]["type"] == "integer"
    assert asyncio.run(registry.execute(ToolCall("1", "sync_tool", {"value": 2}))) == 4
    assert asyncio.run(registry.execute(ToolCall("2", "async_tool", {"value": 2}))) == 6
