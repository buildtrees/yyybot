"""Local SenseVoice speech-to-text adapter backed by sherpa-onnx."""

from __future__ import annotations

import io
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .._async import run_sync
from .base import SpeechProviderError

AudioDecoder = Callable[[bytes], tuple[int, Any]]


def _decode_audio(audio: bytes) -> tuple[int, Any]:
    """Decode browser audio into 16 kHz mono float32 samples."""

    try:
        import av
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError(
            "Local speech transcription requires the 'local-speech' extra"
        ) from exc

    chunks = []
    try:
        with av.open(io.BytesIO(audio)) as container:
            stream = next(
                (item for item in container.streams if item.type == "audio"),
                None,
            )
            if stream is None:
                raise SpeechProviderError(
                    "Audio recording did not contain an audio stream"
                )
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=16_000)
            for frame in container.decode(stream):
                for converted in resampler.resample(frame):
                    chunks.append(converted.to_ndarray().reshape(-1))
            for converted in resampler.resample(None):
                chunks.append(converted.to_ndarray().reshape(-1))
    except SpeechProviderError:
        raise
    except Exception as exc:
        raise SpeechProviderError(f"Could not decode audio recording: {exc}") from exc

    if not chunks:
        raise SpeechProviderError("Audio recording did not contain decodable samples")
    return 16_000, np.concatenate(chunks).astype(np.float32, copy=False)


class SherpaSpeechTranscriber:
    """Transcribe audio locally with a shared sherpa-onnx SenseVoice model."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        tokens_path: str | Path,
        language: str = "zh",
        num_threads: int = 4,
        provider: str = "cpu",
        recognizer: Any | None = None,
        decoder: AudioDecoder = _decode_audio,
    ) -> None:
        self.model_path = Path(model_path).expanduser()
        self.tokens_path = Path(tokens_path).expanduser()
        self.language = language
        self.num_threads = num_threads
        self.provider = provider
        if self.num_threads < 1:
            raise ValueError("YYYBOT_STT_THREADS must be at least 1")
        if self.language not in {"auto", "zh", "en", "ja", "ko", "yue"}:
            raise ValueError(f"Unsupported SenseVoice language: {self.language}")
        self._decoder = decoder
        self._lock = threading.Lock()
        self._recognizer = recognizer or self._create_recognizer()

    def _create_recognizer(self) -> Any:
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise RuntimeError(
                "Local speech transcription requires the 'local-speech' extra"
            ) from exc

        if not self.model_path.is_file():
            raise RuntimeError(f"SenseVoice model not found: {self.model_path}")
        if not self.tokens_path.is_file():
            raise RuntimeError(f"SenseVoice tokens not found: {self.tokens_path}")
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(self.model_path),
            tokens=str(self.tokens_path),
            num_threads=self.num_threads,
            provider=self.provider,
            language=self.language,
            use_itn=True,
        )

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str,
        content_type: str,
        language: str | None = None,
    ) -> str:
        del filename, content_type, language
        return await run_sync(self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: bytes) -> str:
        try:
            sample_rate, samples = self._decoder(audio)
            with self._lock:
                stream = self._recognizer.create_stream()
                stream.accept_waveform(sample_rate, samples)
                self._recognizer.decode_stream(stream)
                result = stream.result
            text = result if isinstance(result, str) else getattr(result, "text", None)
            if not isinstance(text, str):
                raise SpeechProviderError(
                    "SenseVoice transcription response did not contain text"
                )
            return text
        except SpeechProviderError:
            raise
        except Exception as exc:
            raise SpeechProviderError(
                f"SenseVoice transcription failed: {exc}"
            ) from exc
