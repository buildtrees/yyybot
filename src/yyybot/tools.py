"""Tool definitions and execution registry."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Mapping, get_args, get_origin, get_type_hints

from ._async import run_sync
from .contracts import ToolCall, ToolSpec

ToolHandler = Callable[..., Any]


class ToolError(Exception):
    """A safe, user-facing tool execution error."""


def _json_type(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is not None:
        if origin in (list, tuple, set):
            return "array"
        if origin is dict:
            return "object"
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _json_type(args[0])
    return {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }.get(annotation, "string")


def schema_from_function(handler: ToolHandler) -> Mapping[str, Any]:
    signature = inspect.signature(handler)
    type_hints = get_type_hints(handler)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise TypeError("Tool handlers cannot use *args or **kwargs")
        properties[name] = {
            "type": _json_type(type_hints.get(name, parameter.annotation))
        }
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


@dataclass(frozen=True, slots=True)
class Tool:
    spec: ToolSpec
    handler: ToolHandler

    @classmethod
    def from_function(
        cls,
        handler: ToolHandler,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> "Tool":
        tool_name = name or handler.__name__
        tool_description = description or inspect.getdoc(handler) or tool_name
        return cls(
            spec=ToolSpec(
                name=tool_name,
                description=tool_description,
                parameters=parameters or schema_from_function(handler),
            ),
            handler=handler,
        )

    async def invoke(self, arguments: Mapping[str, Any]) -> Any:
        try:
            if inspect.iscoroutinefunction(self.handler):
                return await self.handler(**arguments)
            result = await run_sync(self.handler, **arguments)
            return await result if inspect.isawaitable(result) else result
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool, *, replace: bool = False) -> None:
        name = tool.spec.name
        if name in self._tools and not replace:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def add(
        self,
        handler: ToolHandler,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> Tool:
        tool = Tool.from_function(
            handler,
            name=name,
            description=description,
            parameters=parameters,
        )
        self.register(tool)
        return tool

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    async def execute(self, call: ToolCall) -> Any:
        tool = self._tools.get(call.name)
        if tool is None:
            raise ToolError(f"Unknown tool: {call.name}")
        return await tool.invoke(call.arguments)
