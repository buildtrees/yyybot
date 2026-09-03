"""Provider-neutral speech input and output services."""

from .base import (
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
from .cosyvoice import CosyVoiceSpeechSynthesizer
from .openai import OpenAISpeechProvider
from .qwen import Qwen3SpeechSynthesizer
from .sherpa import SherpaSpeechTranscriber

__all__ = [
    "CosyVoiceSpeechSynthesizer",
    "OpenAISpeechProvider",
    "Qwen3SpeechSynthesizer",
    "SherpaSpeechTranscriber",
    "SpeechAudio",
    "SpeechChunk",
    "SpeechError",
    "SpeechProviderError",
    "SpeechService",
    "SpeechUnavailableError",
    "TranscriptionProvider",
    "Voice",
    "VoiceSynthesisProvider",
]
