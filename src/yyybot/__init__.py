"""Public API for the yyybot agent runtime."""

from .agent import Agent, AgentLimitError
from .context import ConversationContext
from .contracts import (
    AgentEvent,
    AgentResult,
    GenerationOptions,
    Message,
    ModelDelta,
    ModelResponse,
    ModelStreamEvent,
    ToolCall,
    ToolSpec,
)
from .models import Model
from .providers import Provider
from .session import (
    Session,
    SessionError,
    SessionFormatError,
    SessionManager,
    SessionNotFoundError,
    SessionTurn,
)
from .service import ChatService
from .speech import (
    CosyVoiceSpeechSynthesizer,
    OpenAISpeechProvider,
    Qwen3SpeechSynthesizer,
    SherpaSpeechTranscriber,
    SpeechAudio,
    SpeechChunk,
    SpeechError,
    SpeechProviderError,
    SpeechService,
    SpeechUnavailableError,
    TranscriptionProvider,
    Voice,
    VoiceSynthesisProvider,
)
from .terminal import bash
from .tools import Tool, ToolRegistry
from .web import web_fetch, web_search
from .workspace import (
    Workspace,
    WorkspaceError,
    WorkspaceFormatError,
    WorkspaceManager,
    WorkspaceNotFoundError,
    WorkspaceType,
)

__all__ = [
    "Agent",
    "AgentEvent",
    "AgentLimitError",
    "AgentResult",
    "bash",
    "ChatService",
    "ConversationContext",
    "CosyVoiceSpeechSynthesizer",
    "GenerationOptions",
    "Message",
    "Model",
    "ModelDelta",
    "ModelResponse",
    "ModelStreamEvent",
    "Provider",
    "Session",
    "SessionError",
    "SessionFormatError",
    "SessionManager",
    "SessionNotFoundError",
    "SessionTurn",
    "OpenAISpeechProvider",
    "Qwen3SpeechSynthesizer",
    "SherpaSpeechTranscriber",
    "SpeechAudio",
    "SpeechChunk",
    "SpeechError",
    "SpeechProviderError",
    "SpeechService",
    "SpeechUnavailableError",
    "Tool",
    "ToolCall",
    "ToolRegistry",
    "ToolSpec",
    "TranscriptionProvider",
    "Voice",
    "VoiceSynthesisProvider",
    "web_fetch",
    "web_search",
    "Workspace",
    "WorkspaceError",
    "WorkspaceFormatError",
    "WorkspaceManager",
    "WorkspaceNotFoundError",
    "WorkspaceType",
]
