"""OpenAI speech-to-text and text-to-speech adapter."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from ..providers._openai import create_client
from .base import SpeechAudio, SpeechProviderError, Voice

_OPENAI_VOICES = (
    Voice("alloy", "Alloy"),
    Voice("ash", "Ash"),
    Voice("ballad", "Ballad"),
    Voice("coral", "Coral"),
    Voice("echo", "Echo"),
    Voice("fable", "Fable"),
    Voice("nova", "Nova"),
    Voice("onyx", "Onyx"),
    Voice("sage", "Sage"),
    Voice("shimmer", "Shimmer"),
    Voice("verse", "Verse"),
    Voice("marin", "Marin"),
    Voice("cedar", "Cedar"),
)


def _response_text(response: Any) -> str:
    if isinstance(response, Mapping):
        value = response.get("text")
    else:
        value = getattr(response, "text", None)
    if not isinstance(value, str):
        raise SpeechProviderError("OpenAI transcription response did not contain text")
    return value


async def _response_bytes(response: Any) -> bytes:
    if isinstance(response, bytes):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, bytes):
        return content
    read = getattr(response, "read", None)
    if callable(read):
        value = read()
        if inspect.isawaitable(value):
            value = await value
        if isinstance(value, bytes):
            return value
    raise SpeechProviderError("OpenAI speech response did not contain audio bytes")


class OpenAISpeechProvider:
    """Use one OpenAI SDK client for bounded transcription and speech output."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        client: Any | None = None,
        transcription_model: str = "gpt-transcribe",
        synthesis_model: str = "gpt-4o-mini-tts",
        default_voice: str = "marin",
        voices: Sequence[Voice] = _OPENAI_VOICES,
        timeout: float = 90.0,
    ) -> None:
        self.client = client or create_client(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            extra_headers=None,
        )
        self.transcription_model = transcription_model
        self.synthesis_model = synthesis_model
        self._voices = tuple(voices)
        if default_voice not in {voice.voice_id for voice in self._voices}:
            raise ValueError(f"Unsupported default voice: {default_voice}")
        self._default_voice = default_voice

    @property
    def voices(self) -> tuple[Voice, ...]:
        return self._voices

    @property
    def default_voice(self) -> str:
        return self._default_voice

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str,
        content_type: str,
        language: str | None = None,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.transcription_model,
            "file": (filename, audio, content_type),
        }
        if language:
            request["language"] = language
        try:
            response = await self.client.audio.transcriptions.create(**request)
            return _response_text(response)
        except SpeechProviderError:
            raise
        except Exception as exc:
            raise SpeechProviderError(
                f"OpenAI transcription request failed: {exc}"
            ) from exc

    async def synthesize(self, text: str, *, voice: str) -> SpeechAudio:
        try:
            response = await self.client.audio.speech.create(
                model=self.synthesis_model,
                voice=voice,
                input=text,
                response_format="mp3",
            )
            data = await _response_bytes(response)
        except SpeechProviderError:
            raise
        except Exception as exc:
            raise SpeechProviderError(
                f"OpenAI speech synthesis request failed: {exc}"
            ) from exc
        if not data:
            raise SpeechProviderError("OpenAI speech synthesis returned empty audio")
        return SpeechAudio(data=data, media_type="audio/mpeg")
