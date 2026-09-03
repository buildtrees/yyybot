from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from yyybot.speech import (
    CosyVoiceSpeechSynthesizer,
    OpenAISpeechProvider,
    Qwen3SpeechSynthesizer,
    SherpaSpeechTranscriber,
    SpeechProviderError,
    SpeechService,
    Voice,
)


class FakeEndpoint:
    def __init__(self, response):
        self.response = response
        self.requests = []

    async def create(self, **request):
        self.requests.append(request)
        return self.response


def speech_client(*, transcript="你好", audio=b"mp3"):
    transcriptions = FakeEndpoint(SimpleNamespace(text=transcript))
    speech = FakeEndpoint(SimpleNamespace(content=audio))
    client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=transcriptions, speech=speech)
    )
    return client, transcriptions, speech


def test_openai_speech_provider_maps_transcription_and_synthesis_requests():
    client, transcriptions, speech = speech_client()
    provider = OpenAISpeechProvider(
        client=client,
        transcription_model="stt-model",
        synthesis_model="tts-model",
        default_voice="coral",
    )

    transcript = asyncio.run(
        provider.transcribe(
            b"webm-data",
            filename="recording.webm",
            content_type="audio/webm",
            language="zh",
        )
    )
    generated = asyncio.run(provider.synthesize("回答", voice="marin"))

    assert transcript == "你好"
    assert transcriptions.requests == [
        {
            "model": "stt-model",
            "file": ("recording.webm", b"webm-data", "audio/webm"),
            "language": "zh",
        }
    ]
    assert speech.requests == [
        {
            "model": "tts-model",
            "voice": "marin",
            "input": "回答",
            "response_format": "mp3",
        }
    ]
    assert generated.data == b"mp3"
    assert generated.media_type == "audio/mpeg"


def test_speech_service_rejects_unknown_voice_and_empty_transcript():
    client, _, _ = speech_client(transcript="  ")
    provider = OpenAISpeechProvider(
        client=client,
        voices=(Voice("test", "Test"),),
        default_voice="test",
    )
    service = SpeechService(transcriber=provider, synthesizer=provider)

    with pytest.raises(SpeechProviderError, match="No speech"):
        asyncio.run(
            service.transcribe(
                b"audio",
                filename="recording.webm",
                content_type="audio/webm",
            )
        )
    with pytest.raises(ValueError, match="Unsupported voice"):
        asyncio.run(service.synthesize("hello", voice="missing"))


def test_sherpa_transcriber_decodes_audio_off_the_event_loop():
    accepted = []

    class FakeStream:
        result = SimpleNamespace(text="本地识别结果")

        def accept_waveform(self, sample_rate, samples):
            accepted.append((sample_rate, samples))

    class FakeRecognizer:
        def create_stream(self):
            return FakeStream()

        def decode_stream(self, stream):
            assert isinstance(stream, FakeStream)

    provider = SherpaSpeechTranscriber(
        model_path="unused.onnx",
        tokens_path="unused.txt",
        recognizer=FakeRecognizer(),
        decoder=lambda audio: (16_000, [len(audio)]),
    )

    text = asyncio.run(
        provider.transcribe(
            b"browser-audio",
            filename="recording.webm",
            content_type="audio/webm",
            language="zh",
        )
    )

    assert text == "本地识别结果"
    assert accepted == [(16_000, [13])]


def test_qwen3_synthesizer_generates_browser_playable_wav():
    requests = []

    class FakeModel:
        def generate_custom_voice(self, **request):
            requests.append(request)
            return [[-1.0, 0.0, 0.5, 1.0]], 24_000

    provider = Qwen3SpeechSynthesizer(model=FakeModel())
    generated = asyncio.run(provider.synthesize("你好", voice="Serena"))

    assert requests == [
        {
            "text": "你好",
            "language": "Chinese",
            "speaker": "Serena",
            "non_streaming_mode": False,
            "max_new_tokens": 96,
        }
    ]
    assert generated.media_type == "audio/wav"
    with wave.open(io.BytesIO(generated.data), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 24_000
        assert audio.getnframes() == 4


def test_qwen3_synthesizer_persists_uses_and_deletes_personal_voice(tmp_path):
    clone_requests = []

    class FakeCustomVoiceModel:
        def generate_custom_voice(self, **request):
            return [[0.0]], 24_000

    class FakeCloneModel:
        def create_voice_clone_prompt(self, **request):
            clone_requests.append(("prompt", request))
            return ["cached-prompt"]

        def generate_voice_clone(self, **request):
            clone_requests.append(("generate", request))
            return [[-0.25, 0.25]], 24_000

    provider = Qwen3SpeechSynthesizer(
        model=FakeCustomVoiceModel(),
        clone_model=FakeCloneModel(),
        clone_model_id="base-model",
        voices_dir=tmp_path,
    )
    voice = asyncio.run(
        provider.add_voice(
            name="我的声音",
            audio=b"reference-audio",
            filename="sample.wav",
            content_type="audio/wav",
            reference_text="这是一段参考文字",
        )
    )

    assert voice.custom is True
    assert voice.name == "我的声音 · 个人音色"
    assert voice.voice_id.startswith("personal-")
    assert (tmp_path / voice.voice_id / "reference.wav").read_bytes() == b"reference-audio"

    reloaded = Qwen3SpeechSynthesizer(
        model=FakeCustomVoiceModel(),
        clone_model=FakeCloneModel(),
        clone_model_id="base-model",
        voices_dir=tmp_path,
    )
    assert reloaded.voices[-1] == voice
    generated = asyncio.run(reloaded.synthesize("使用个人声音", voice=voice.voice_id))
    assert generated.media_type == "audio/wav"
    assert clone_requests[-2][0] == "prompt"
    assert clone_requests[-2][1]["ref_text"] == "这是一段参考文字"
    assert clone_requests[-1] == (
        "generate",
        {
            "text": "使用个人声音",
            "language": "Chinese",
            "voice_clone_prompt": ["cached-prompt"],
            "non_streaming_mode": False,
            "max_new_tokens": 96,
        },
    )

    asyncio.run(reloaded.delete_voice(voice.voice_id))
    assert voice not in reloaded.voices
    assert not (tmp_path / voice.voice_id).exists()


def test_cosyvoice_synthesizer_reuses_personal_voices_and_streams_pcm(tmp_path):
    calls = []

    class FakeTensor:
        def __init__(self, values):
            self.values = values

        @property
        def shape(self):
            if self.values and isinstance(self.values[0], list):
                return (len(self.values), len(self.values[0]))
            return (len(self.values),)

        def flatten(self):
            values = self.values[0] if self.values and isinstance(self.values[0], list) else self.values
            return FakeTensor(list(values))

        def tolist(self):
            return list(self.values)

        def __getitem__(self, key):
            row_key, column_key = key
            rows = self.values[row_key]
            if rows and not isinstance(rows[0], list):
                rows = [rows]
            return FakeTensor([row[column_key] for row in rows])

        def new_full(self, shape, value):
            return FakeTensor([value] * shape[0])

        def new_zeros(self, shape):
            return FakeTensor([0] * shape[0])

    class FakeFrontend:
        spk2info = {}

    class FakeCosyVoice:
        sample_rate = 24_000
        frontend = FakeFrontend()

        def add_zero_shot_spk(self, prompt_text, prompt_wav, voice_id):
            calls.append(("register", prompt_text, Path(prompt_wav).name, voice_id))
            self.frontend.spk2info[voice_id] = {
                "prompt_text": FakeTensor([[10, 151646, 20, 21]]),
                "prompt_text_len": FakeTensor([4]),
                "llm_prompt_speech_token": FakeTensor([[30, 31, 32]]),
                "llm_prompt_speech_token_len": FakeTensor([3]),
                "flow_prompt_speech_token": FakeTensor([[30, 31, 32]]),
            }
            return True

        def inference_zero_shot(self, texts, *args, **kwargs):
            calls.append(("stream", list(texts), args, kwargs))
            yield {"tts_speech": [-1.0, 0.0, 1.0]}

    provider = CosyVoiceSpeechSynthesizer(
        model_id="unused",
        repo_path="unused",
        voices_dir=tmp_path,
        device="cpu",
        model=FakeCosyVoice(),
        audio_normalizer=lambda audio: b"normalized-wav",
    )
    voice = asyncio.run(
        provider.add_voice(
            name="我的声音",
            audio=b"uploaded-video",
            filename="voice.mp4",
            content_type="video/mp4",
            reference_text="参考文字",
        )
    )

    chunks = list(provider.stream_synthesize(iter(["你", "好"]), voice=voice.voice_id))

    assert voice.custom is True
    assert (tmp_path / voice.voice_id / "reference.wav").read_bytes() == b"normalized-wav"
    assert "<|endofprompt|>参考文字" in calls[0][1]
    assert chunks[0].sample_rate == 24_000
    assert len(chunks[0].data) == 6
    assert calls[-1][0:2] == ("stream", ["你", "好"])
    assert calls[-1][3] == {
        "zero_shot_spk_id": voice.voice_id,
        "stream": True,
    }
    cached = provider._model.frontend.spk2info[voice.voice_id]
    assert cached["prompt_text"].values == [[10, 151646]]
    assert cached["prompt_text_len"].values == [2]
    assert cached["llm_prompt_speech_token"].values == [[]]
    assert cached["llm_prompt_speech_token_len"].values == [0]
    assert cached["flow_prompt_speech_token"].values == [[30, 31, 32]]
