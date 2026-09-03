"""Provider-neutral speech contracts and application service."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol


class SpeechError(RuntimeError):
    """Base error for speech input and output."""


class SpeechUnavailableError(SpeechError):
    """Raised when the requested speech capability is not configured."""


class SpeechProviderError(SpeechError):
    """Raised when a speech provider request fails or returns invalid data."""


@dataclass(frozen=True, slots=True)
class Voice:
    voice_id: str
    name: str
    custom: bool = False


@dataclass(frozen=True, slots=True)
class SpeechAudio:
    data: bytes
    media_type: str


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    data: bytes
    sample_rate: int


class TranscriptionProvider(Protocol):
    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str,
        content_type: str,
        language: str | None = None,
    ) -> str: ...


class VoiceSynthesisProvider(Protocol):
    @property
    def voices(self) -> Sequence[Voice]: ...

    @property
    def default_voice(self) -> str: ...

    async def synthesize(self, text: str, *, voice: str) -> SpeechAudio: ...


class StreamingVoiceSynthesisProvider(VoiceSynthesisProvider, Protocol):
    @property
    def sample_rate(self) -> int: ...

    def stream_synthesize(
        self,
        texts: Iterable[str],
        *,
        voice: str,
    ) -> Iterator[SpeechChunk]: ...


class SpeechService:
    """Keep audio conversion outside the text-only Agent runtime."""

    def __init__(
        self,
        *,
        transcriber: TranscriptionProvider | None = None,
        synthesizer: VoiceSynthesisProvider | None = None,
    ) -> None:
        self.transcriber = transcriber
        self.synthesizer = synthesizer

    @property
    def transcription_enabled(self) -> bool:
        return self.transcriber is not None

    @property
    def synthesis_enabled(self) -> bool:
        return self.synthesizer is not None

    @property
    def voices(self) -> tuple[Voice, ...]:
        if self.synthesizer is None:
            return ()
        return tuple(self.synthesizer.voices)

    @property
    def default_voice(self) -> str | None:
        if self.synthesizer is None:
            return None
        return self.synthesizer.default_voice

    @property
    def voice_upload_enabled(self) -> bool:
        return bool(
            self.synthesizer
            and getattr(self.synthesizer, "voice_upload_enabled", False)
        )

    @property
    def streaming_enabled(self) -> bool:
        return bool(
            self.synthesizer
            and callable(getattr(self.synthesizer, "stream_synthesize", None))
            and isinstance(getattr(self.synthesizer, "sample_rate", None), int)
        )

    @property
    def synthesis_sample_rate(self) -> int | None:
        if not self.streaming_enabled or self.synthesizer is None:
            return None
        return int(getattr(self.synthesizer, "sample_rate"))

    def _select_voice(self, voice: str | None) -> str:
        if self.synthesizer is None:
            raise SpeechUnavailableError("Speech synthesis is not configured")
        selected_voice = voice or self.synthesizer.default_voice
        available = {item.voice_id for item in self.synthesizer.voices}
        if selected_voice not in available:
            raise ValueError(f"Unsupported voice: {selected_voice}")
        return selected_voice

    async def add_voice(
        self,
        *,
        name: str,
        audio: bytes,
        filename: str,
        content_type: str,
        reference_text: str | None = None,
    ) -> Voice:
        if not self.voice_upload_enabled or self.synthesizer is None:
            raise SpeechUnavailableError("Custom voice upload is not configured")
        add_voice = getattr(self.synthesizer, "add_voice")
        return await add_voice(
            name=name,
            audio=audio,
            filename=filename,
            content_type=content_type,
            reference_text=reference_text,
        )

    async def delete_voice(self, voice_id: str) -> None:
        if not self.voice_upload_enabled or self.synthesizer is None:
            raise SpeechUnavailableError("Custom voice upload is not configured")
        delete_voice = getattr(self.synthesizer, "delete_voice")
        await delete_voice(voice_id)

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str,
        content_type: str,
        language: str | None = None,
    ) -> str:
        if self.transcriber is None:
            raise SpeechUnavailableError("Speech transcription is not configured")
        text = await self.transcriber.transcribe(
            audio,
            filename=filename,
            content_type=content_type,
            language=language,
        )
        text = text.strip()
        if not text:
            raise SpeechProviderError("No speech was recognized in the recording")
        return text

    async def synthesize(self, text: str, *, voice: str | None = None) -> SpeechAudio:
        selected_voice = self._select_voice(voice)
        assert self.synthesizer is not None
        return await self.synthesizer.synthesize(text, voice=selected_voice)

    def stream_synthesize(
        self,
        texts: Iterable[str],
        *,
        voice: str | None = None,
    ) -> Iterator[SpeechChunk]:
        if not self.streaming_enabled or self.synthesizer is None:
            raise SpeechUnavailableError("Streaming speech synthesis is not configured")
        selected_voice = self._select_voice(voice)
        stream_synthesize = getattr(self.synthesizer, "stream_synthesize")
        return stream_synthesize(texts, voice=selected_voice)
