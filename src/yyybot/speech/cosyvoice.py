"""Native bi-streaming voice cloning backed by Fun-CosyVoice3."""

from __future__ import annotations

import io
import json
import shutil
import sys
import threading
import wave
from array import array
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .._async import run_sync
from .base import SpeechAudio, SpeechChunk, SpeechProviderError, Voice

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
_END_OF_PROMPT = "<|endofprompt|>"
_END_OF_PROMPT_TOKEN = 151646
_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True, slots=True)
class _StoredVoice:
    voice: Voice
    directory: Path
    audio_path: Path
    reference_text: str | None


def _prompt_text(reference_text: str | None) -> str:
    return f"{_SYSTEM_PROMPT}{_END_OF_PROMPT}{reference_text or ''}"


def _pcm_bytes(samples: Any) -> bytes:
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().reshape(-1).tolist()
    pcm = array(
        "h",
        (
            round(max(-1.0, min(1.0, float(sample))) * 32767)
            for sample in samples
        ),
    )
    if sys.byteorder != "little":  # pragma: no cover - CI is little-endian
        pcm.byteswap()
    return pcm.tobytes()


def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    if not pcm:
        raise SpeechProviderError("CosyVoice returned empty audio")
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm)
    return output.getvalue()


def _normalize_reference_audio(audio: bytes) -> bytes:
    """Extract an uploaded audio/video track into 24 kHz mono PCM WAV."""

    try:
        import av
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "CosyVoice video/audio upload requires the 'local-speech' extra"
        ) from exc
    try:
        chunks: list[bytes] = []
        with av.open(io.BytesIO(audio)) as container:
            stream = next(
                (item for item in container.streams if item.type == "audio"),
                None,
            )
            if stream is None:
                raise SpeechProviderError("Upload did not contain an audio track")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=24_000)
            for frame in container.decode(stream):
                converted = resampler.resample(frame)
                for output_frame in converted:
                    chunks.append(output_frame.to_ndarray().reshape(-1).tobytes())
            for output_frame in resampler.resample(None):
                chunks.append(output_frame.to_ndarray().reshape(-1).tobytes())
        pcm = b"".join(chunks)
    except SpeechProviderError:
        raise
    except Exception as exc:
        raise SpeechProviderError(
            f"Could not decode reference audio or video: {exc}"
        ) from exc
    duration = len(pcm) / (24_000 * 2)
    if duration < 1:
        raise ValueError("Reference voice must be at least 1 second")
    if duration > 30:
        raise ValueError("Reference voice must not exceed 30 seconds")
    return _wav_bytes(pcm, 24_000)


class CosyVoiceSpeechSynthesizer:
    """Clone persisted personal voices with CosyVoice3 bi-streaming inference."""

    def __init__(
        self,
        *,
        model_id: str | Path,
        repo_path: str | Path,
        voices_dir: str | Path,
        device: str = "cuda:1",
        fp16: bool = True,
        default_voice: str | None = None,
        model: Any | None = None,
        audio_normalizer: Callable[[bytes], bytes] = _normalize_reference_audio,
    ) -> None:
        self.model_id = str(model_id)
        self.repo_path = Path(repo_path).expanduser()
        self.voices_dir = Path(voices_dir).expanduser()
        self.device = device
        self._device_index = (
            int(device.partition(":")[2] or "0")
            if device.startswith("cuda")
            else None
        )
        self.fp16 = fp16
        self._audio_normalizer = audio_normalizer
        self._lock = threading.RLock()
        self._stream_error_lock = threading.Lock()
        self._stream_error: Exception | None = None
        self._custom_voices: dict[str, _StoredVoice] = {}
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self._load_stored_voices()
        self._model = model or self._create_model()
        self._guard_native_streaming()
        self._sample_rate = int(self._model.sample_rate)
        if self._sample_rate < 1:
            raise RuntimeError("CosyVoice returned an invalid sample rate")
        self._preload_voices()
        available = {voice.voice_id for voice in self.voices}
        if default_voice in available:
            self._default_voice = str(default_voice)
        elif available:
            self._default_voice = next(iter(self._custom_voices))
        else:
            self._default_voice = ""

    def _guard_native_streaming(self) -> None:
        """Prevent upstream bi-stream failures from wedging the web service."""

        engine = getattr(self._model, "model", None)
        llm = getattr(engine, "llm", None)
        sampling_ids = getattr(llm, "sampling_ids", None)
        eos_token = getattr(llm, "eos_token", None)
        if callable(sampling_ids) and isinstance(eos_token, int):
            def guarded_sampling(
                weighted_scores,
                decoded_tokens,
                sampling,
                ignore_eos=True,
            ):
                if ignore_eos:
                    weighted_scores[eos_token] = -float("inf")
                return sampling_ids(
                    weighted_scores,
                    decoded_tokens,
                    sampling,
                    ignore_eos,
                )

            llm.sampling_ids = guarded_sampling

        llm_job = getattr(engine, "llm_job", None)
        if not callable(llm_job):
            return

        def guarded_llm_job(*args) -> None:
            inference_id = args[-1]
            try:
                llm_job(*args)
            except Exception as exc:
                with self._stream_error_lock:
                    self._stream_error = exc
            finally:
                engine.llm_end_dict[inference_id] = True

        engine.llm_job = guarded_llm_job

    def _create_model(self) -> Any:
        if not self.repo_path.is_dir():
            raise RuntimeError(f"CosyVoice repository not found: {self.repo_path}")
        matcha_path = self.repo_path / "third_party" / "Matcha-TTS"
        for path in (matcha_path, self.repo_path):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        try:
            import torch
            from cosyvoice.cli.cosyvoice import AutoModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "CosyVoice dependencies are not installed in the server environment"
            ) from exc
        if self._device_index is not None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for CosyVoice")
            torch.cuda.set_device(self._device_index)
        try:
            return AutoModel(model_dir=self.model_id, fp16=self.fp16)
        except Exception as exc:
            raise RuntimeError(f"Could not load CosyVoice model: {exc}") from exc

    @property
    def voices(self) -> tuple[Voice, ...]:
        with self._lock:
            return tuple(item.voice for item in self._custom_voices.values())

    @property
    def default_voice(self) -> str:
        return self._default_voice

    @property
    def voice_upload_enabled(self) -> bool:
        return True

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _load_stored_voices(self) -> None:
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
                self._custom_voices[voice_id] = _StoredVoice(
                    voice=Voice(voice_id, f"{name} · 个人音色", custom=True),
                    directory=metadata_path.parent,
                    audio_path=audio_path,
                    reference_text=payload.get("reference_text") or None,
                )
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue

    def _register_voice(self, stored: _StoredVoice) -> None:
        self._activate_device()
        succeeded = self._model.add_zero_shot_spk(
            _prompt_text(stored.reference_text),
            str(stored.audio_path),
            stored.voice.voice_id,
        )
        if succeeded is not True:
            raise SpeechProviderError(
                f"CosyVoice could not register voice: {stored.voice.voice_id}"
            )
        self._isolate_reference_content(stored.voice.voice_id)

    def _isolate_reference_content(self, voice_id: str) -> None:
        """Keep acoustic voice conditioning out of the streaming text model.

        CosyVoice3's bi-streaming path interleaves the reference transcript and
        reference speech tokens with incoming text.  On short replies that can
        make the model speak words from the uploaded sample.  The flow model
        still receives the full acoustic prompt and speaker embedding, while
        the language model receives only its system prompt.
        """

        info = self._model.frontend.spk2info[voice_id]
        required = {
            "prompt_text",
            "prompt_text_len",
            "llm_prompt_speech_token",
            "llm_prompt_speech_token_len",
        }
        if not required.issubset(info):
            raise SpeechProviderError(
                f"CosyVoice returned incomplete voice data: {voice_id}"
            )
        prompt_text = info["prompt_text"]
        prompt_ids = prompt_text.flatten().tolist()
        try:
            eop_index = prompt_ids.index(_END_OF_PROMPT_TOKEN)
        except ValueError as exc:
            raise SpeechProviderError(
                f"CosyVoice reference prompt is invalid: {voice_id}"
            ) from exc

        info["prompt_text"] = prompt_text[:, : eop_index + 1]
        prompt_text_len = info["prompt_text_len"]
        info["prompt_text_len"] = prompt_text_len.new_full(
            prompt_text_len.shape,
            eop_index + 1,
        )
        llm_prompt = info["llm_prompt_speech_token"]
        info["llm_prompt_speech_token"] = llm_prompt[:, :0]
        llm_prompt_len = info["llm_prompt_speech_token_len"]
        info["llm_prompt_speech_token_len"] = llm_prompt_len.new_zeros(
            llm_prompt_len.shape
        )

    def _activate_device(self) -> None:
        if self._device_index is None:
            return
        import torch

        torch.cuda.set_device(self._device_index)

    def _preload_voices(self) -> None:
        with self._lock:
            for stored in self._custom_voices.values():
                self._register_voice(stored)

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
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Voice name is empty")
        if len(clean_name) > 80:
            raise ValueError("Voice name is too long")
        if not audio:
            raise ValueError("Reference audio is empty")
        if content_type not in _AUDIO_SUFFIXES:
            raise ValueError("Unsupported reference audio type")
        normalized = self._audio_normalizer(audio)
        reference_text = reference_text.strip() if reference_text else None
        voice_id = f"personal-{uuid4().hex}"
        pending_dir = self.voices_dir / f".{voice_id}.pending"
        final_dir = self.voices_dir / voice_id
        audio_path = pending_dir / "reference.wav"
        stored = _StoredVoice(
            voice=Voice(voice_id, f"{clean_name} · 个人音色", custom=True),
            directory=final_dir,
            audio_path=final_dir / audio_path.name,
            reference_text=reference_text,
        )
        try:
            pending_dir.mkdir(parents=False)
            audio_path.write_bytes(normalized)
            pending_stored = _StoredVoice(
                voice=stored.voice,
                directory=pending_dir,
                audio_path=audio_path,
                reference_text=reference_text,
            )
            with self._lock:
                self._register_voice(pending_stored)
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
                self._custom_voices[voice_id] = stored
                if not self._default_voice:
                    self._default_voice = voice_id
            return stored.voice
        except Exception as exc:
            shutil.rmtree(pending_dir, ignore_errors=True)
            if isinstance(exc, (SpeechProviderError, ValueError)):
                raise
            raise SpeechProviderError(
                f"Could not create CosyVoice personal voice: {exc}"
            ) from exc

    async def delete_voice(self, voice_id: str) -> None:
        await run_sync(self._delete_voice_sync, voice_id)

    def _delete_voice_sync(self, voice_id: str) -> None:
        with self._lock:
            try:
                stored = self._custom_voices.pop(voice_id)
            except KeyError as exc:
                raise ValueError(f"Custom voice not found: {voice_id}") from exc
            self._model.frontend.spk2info.pop(voice_id, None)
            shutil.rmtree(stored.directory)
            if self._default_voice == voice_id:
                self._default_voice = next(iter(self._custom_voices), "")

    def stream_synthesize(
        self,
        texts: Iterable[str],
        *,
        voice: str,
    ) -> Iterator[SpeechChunk]:
        if voice not in self._custom_voices:
            raise ValueError(f"Unsupported voice: {voice}")

        def text_generator():
            for text in texts:
                if text:
                    yield text

        try:
            self._activate_device()
            with self._lock:
                with self._stream_error_lock:
                    self._stream_error = None
                for output in self._model.inference_zero_shot(
                    text_generator(),
                    "",
                    "",
                    zero_shot_spk_id=voice,
                    stream=True,
                ):
                    pcm = _pcm_bytes(output["tts_speech"])
                    if pcm:
                        yield SpeechChunk(pcm, self.sample_rate)
                with self._stream_error_lock:
                    stream_error = self._stream_error
                    self._stream_error = None
                if stream_error is not None:
                    raise stream_error
        except (SpeechProviderError, ValueError):
            raise
        except Exception as exc:
            raise SpeechProviderError(
                f"CosyVoice streaming synthesis failed: {exc}"
            ) from exc

    async def synthesize(self, text: str, *, voice: str) -> SpeechAudio:
        return await run_sync(self._synthesize_sync, text, voice)

    def _synthesize_sync(self, text: str, voice: str) -> SpeechAudio:
        pcm = b"".join(
            chunk.data for chunk in self.stream_synthesize((text,), voice=voice)
        )
        return SpeechAudio(_wav_bytes(pcm, self.sample_rate), "audio/wav")
