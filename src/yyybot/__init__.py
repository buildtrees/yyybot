"""Public API for the yyybot agent runtime."""

from .agent import Agent, AgentLimitError
from .contracts import (
    AgentEvent,
    AgentResult,
    GenerationOptions,
    Message,
    ModelResponse,
    ToolCall,
    ToolSpec,
)
from .models import Model
from .providers import Provider
from .terminal import bash
from .tools import Tool, ToolRegistry
from .web import web_fetch, web_search

__all__ = [
    "Agent",
    "AgentEvent",
    "AgentLimitError",
    "AgentResult",
    "bash",
    "GenerationOptions",
    "Message",
    "Model",
    "ModelResponse",
    "Provider",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolSpec",
    "web_fetch",
    "web_search",
]
