"""Local text-to-speech adapter backed by Qwen3-TTS."""

from __future__ import annotations

import io
import json
import shutil
import sys
import threading
import wave
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .._async import run_sync
from .base import SpeechAudio, SpeechProviderError, Voice

_QWEN3_VOICES = (
    Voice("Vivian", "Vivian · 明亮女声"),
    Voice("Serena", "Serena · 温柔女声"),
    Voice("Uncle_Fu", "Uncle Fu · 低沉男声"),
    Voice("Dylan", "Dylan · 北京男声"),
    Voice("Eric", "Eric · 四川男声"),
    Voice("Ryan", "Ryan · 英语男声"),
    Voice("Aiden", "Aiden · 美式男声"),
    Voice("Ono_Anna", "Ono Anna · 日语女声"),
    Voice("Sohee", "Sohee · 韩语女声"),
)

_AUDIO_SUFFIXES = {
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/m4a": ".m4a",
    "audio/mp3": ".mp3",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "audio/x-m4a": ".m4a",
    "audio/x-wav": ".wav",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}


@dataclass(frozen=True, slots=True)
class _StoredVoice:
    voice: Voice
    directory: Path
    audio_path: Path
    reference_text: str | None


def _wav_bytes(samples: Any, sample_rate: int) -> bytes:
    if sample_rate < 1:
        raise SpeechProviderError("Qwen3-TTS returned an invalid sample rate")
    pcm = array(
        "h",
        (
            round(max(-1.0, min(1.0, float(sample))) * 32767)
            for sample in samples
        ),
    )
    if not pcm:
        raise SpeechProviderError("Qwen3-TTS returned empty audio")
    if sys.byteorder != "little":  # pragma: no cover - CI is little-endian
        pcm.byteswap()
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm.tobytes())
    return output.getvalue()


class Qwen3SpeechSynthesizer:
    """Generate built-in and persisted custom voices with local Qwen3-TTS."""

    def __init__(
        self,
        *,
        model_id: str | Path = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device: str = "cuda:1",
        dtype: str = "bfloat16",
        attention: str = "sdpa",
        language: str = "Chinese",
        max_new_tokens: int = 1_024,
        default_voice: str = "Serena",
        voices: Sequence[Voice] = _QWEN3_VOICES,
        clone_model_id: str | Path | None = None,
        voices_dir: str | Path | None = None,
        preload_custom_voices: bool = True,
        model: Any | None = None,
        clone_model: Any | None = None,
    ) -> None:
        self.model_id = str(model_id)
        self.device = device
        self.dtype = dtype
        self.attention = attention
        self.language = language
        if max_new_tokens < 96:
            raise ValueError("Qwen3-TTS max_new_tokens must be at least 96")
        self.max_new_tokens = max_new_tokens
        self.clone_model_id = str(clone_model_id) if clone_model_id else None
        self.voices_dir = Path(voices_dir).expanduser() if voices_dir else None
        self._builtin_voices = tuple(voices)
        self._custom_voices: dict[str, _StoredVoice] = {}
        self._voice_prompts: dict[str, Any] = {}
        if default_voice not in {voice.voice_id for voice in self._builtin_voices}:
            raise ValueError(f"Unsupported default voice: {default_voice}")
        self._default_voice = default_voice
        self._lock = threading.RLock()
        self._model = model or self._create_model()
        self._clone_model = clone_model
        if self.voice_upload_enabled:
            self.voices_dir.mkdir(parents=True, exist_ok=True)
            self._load_stored_voices()
            if preload_custom_voices and self._custom_voices:
                with self._lock:
                    for stored in self._custom_voices.values():
                        self._create_clone_prompt(stored)

    def _create_model(self) -> Any:
        return self._load_model(self.model_id)

    def _load_model(self, model_id: str) -> Any:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "Local Qwen3 speech synthesis requires the 'local-tts' extra"
            ) from exc
        torch_dtype = getattr(torch, self.dtype, None)
        if torch_dtype is None:
            raise ValueError(f"Unsupported Qwen3-TTS dtype: {self.dtype}")
        try:
            return Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=self.device,
                dtype=torch_dtype,
                attn_implementation=self.attention,
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load Qwen3-TTS model: {exc}") from exc

    @property
    def voices(self) -> tuple[Voice, ...]:
        with self._lock:
            custom = tuple(item.voice for item in self._custom_voices.values())
        return (*self._builtin_voices, *custom)

    @property
    def default_voice(self) -> str:
        return self._default_voice

    @property
    def voice_upload_enabled(self) -> bool:
        return self.clone_model_id is not None and self.voices_dir is not None

    def _load_stored_voices(self) -> None:
        assert self.voices_dir is not None
        for metadata_path in sorted(self.voices_dir.glob("*/voice.json")):
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                voice_id = str(payload["voice_id"])
                name = str(payload["name"])
                audio_file = Path(str(payload["audio_file"]))
                if (
                    not voice_id.startswith("personal-")
                    or metadata_path.parent.name != voice_id
                    or audio_file.name != str(audio_file)
                ):
                    continue
                audio_path = metadata_path.parent / audio_file
                if not audio_path.is_file():
                    continue
                reference_text = payload.get("reference_text") or None
                self._custom_voices[voice_id] = _StoredVoice(
                    voice=Voice(voice_id, f"{name} · 个人音色", custom=True),
                    directory=metadata_path.parent,
                    audio_path=audio_path,
                    reference_text=reference_text,
                )
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue

    def _get_clone_model(self) -> Any:
        if not self.voice_upload_enabled or self.clone_model_id is None:
            raise SpeechProviderError("Qwen3-TTS voice cloning is not configured")
        if self._clone_model is None:
            self._clone_model = self._load_model(self.clone_model_id)
        return self._clone_model

    def _create_clone_prompt(self, stored: _StoredVoice) -> Any:
        prompt = self._voice_prompts.get(stored.voice.voice_id)
        if prompt is None:
            prompt = self._get_clone_model().create_voice_clone_prompt(
                ref_audio=str(stored.audio_path),
                ref_text=stored.reference_text,
                x_vector_only_mode=not bool(stored.reference_text),
            )
            self._voice_prompts[stored.voice.voice_id] = prompt
        return prompt

    async def add_voice(
        self,
        *,
        name: str,
        audio: bytes,
        filename: str,
        content_type: str,
        reference_text: str | None = None,
    ) -> Voice:
        return await run_sync(
            self._add_voice_sync,
            name,
            audio,
            filename,
            content_type,
            reference_text,
        )

    def _add_voice_sync(
        self,
        name: str,
        audio: bytes,
        filename: str,
        content_type: str,
        reference_text: str | None,
    ) -> Voice:
        if not self.voice_upload_enabled or self.voices_dir is None:
            raise SpeechProviderError("Qwen3-TTS voice cloning is not configured")
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Voice name is empty")
        if len(clean_name) > 80:
            raise ValueError("Voice name is too long")
        if not audio:
            raise ValueError("Reference audio is empty")
        reference_text = reference_text.strip() if reference_text else None
        suffix = Path(filename).suffix.lower()
        if suffix not in set(_AUDIO_SUFFIXES.values()):
            suffix = _AUDIO_SUFFIXES.get(content_type, ".wav")
        voice_id = f"personal-{uuid4().hex}"
        pending_dir = self.voices_dir / f".{voice_id}.pending"
        final_dir = self.voices_dir / voice_id
        audio_path = pending_dir / f"reference{suffix}"
        stored = _StoredVoice(
            voice=Voice(voice_id, f"{clean_name} · 个人音色", custom=True),
            directory=final_dir,
            audio_path=final_dir / audio_path.name,
            reference_text=reference_text,
        )
        try:
            pending_dir.mkdir(parents=False)
            audio_path.write_bytes(audio)
            pending_stored = _StoredVoice(
                voice=stored.voice,
                directory=pending_dir,
                audio_path=audio_path,
                reference_text=reference_text,
            )
            with self._lock:
                prompt = self._create_clone_prompt(pending_stored)
                metadata = {
                    "voice_id": voice_id,
                    "name": clean_name,
                    "audio_file": audio_path.name,
                    "reference_text": reference_text,
                }
                (pending_dir / "voice.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                pending_dir.replace(final_dir)
                self._voice_prompts.pop(voice_id, None)
                self._voice_prompts[voice_id] = prompt
                self._custom_voices[voice_id] = stored
            return stored.voice
        except Exception as exc:
            shutil.rmtree(pending_dir, ignore_errors=True)
            if isinstance(exc, (SpeechProviderError, ValueError)):
                raise
            raise SpeechProviderError(
                f"Could not create Qwen3-TTS custom voice: {exc}"
            ) from exc

    async def delete_voice(self, voice_id: str) -> None:
        await run_sync(self._delete_voice_sync, voice_id)

    def _delete_voice_sync(self, voice_id: str) -> None:
        with self._lock:
            try:
                stored = self._custom_voices.pop(voice_id)
            except KeyError as exc:
                raise ValueError(f"Custom voice not found: {voice_id}") from exc
            self._voice_prompts.pop(voice_id, None)
            shutil.rmtree(stored.directory)

    async def synthesize(self, text: str, *, voice: str) -> SpeechAudio:
        return await run_sync(self._synthesize_sync, text, voice)

    def _synthesize_sync(self, text: str, voice: str) -> SpeechAudio:
        try:
            generation_limit = min(
                self.max_new_tokens,
                max(96, len(text) * 12),
            )
            with self._lock:
                stored = self._custom_voices.get(voice)
                if stored is None:
                    waveforms, sample_rate = self._model.generate_custom_voice(
                        text=text,
                        language=self.language,
                        speaker=voice,
                        non_streaming_mode=False,
                        max_new_tokens=generation_limit,
                    )
                else:
                    waveforms, sample_rate = self._get_clone_model().generate_voice_clone(
                        text=text,
                        language=self.language,
                        voice_clone_prompt=self._create_clone_prompt(stored),
                        non_streaming_mode=False,
                        max_new_tokens=generation_limit,
                    )
            if not waveforms:
                raise SpeechProviderError("Qwen3-TTS returned no waveform")
            return SpeechAudio(
                data=_wav_bytes(waveforms[0], int(sample_rate)),
                media_type="audio/wav",
            )
        except SpeechProviderError:
            raise
        except Exception as exc:
            raise SpeechProviderError(f"Qwen3-TTS synthesis failed: {exc}") from exc
