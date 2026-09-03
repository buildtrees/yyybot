from __future__ import annotations

import asyncio
from pathlib import Path

from yyybot import Agent, AgentResult, Message, Model, ModelResponse, Provider
from yyybot import runtime
from yyybot.execution import current_execution_directory
from yyybot.service import ChatService
from yyybot.workspace import WorkspaceManager


class EchoProvider(Provider):
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.requests: list[tuple[Message, ...]] = []

    async def complete(self, *, model_id, messages, tools=(), options=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.requests.append(tuple(messages))
        await asyncio.sleep(0.01)
        self.active -= 1
        user = next(message for message in reversed(messages) if message.role == "user")
        return ModelResponse(
            Message(role="assistant", content=f"answer:{user.content}"),
            "stop",
            {"total_tokens": 3},
        )


def test_service_serializes_runs_and_persists_fresh_session_context(tmp_path):
    provider = EchoProvider()
    model = Model("test-model", provider)
    service = ChatService(
        WorkspaceManager(tmp_path),
        lambda on_event: Agent(model, on_event=on_event),
    )
    workspace = service.create_workspace(name="Test", workspace_id="test")
    service.create_session(workspace.workspace_id, session_id="chat")

    async def run_both():
        return await asyncio.gather(
            service.run("test", "chat", "first"),
            service.run("test", "chat", "second"),
        )

    first, second = asyncio.run(run_both())

    assert first.output == "answer:first"
    assert second.output == "answer:second"
    assert provider.max_active == 1
    assert Message(role="assistant", content="answer:first") in provider.requests[1]
    persisted = service.get_session("test", "chat")
    assert len(persisted.turns) == 2
    assert persisted.messages[-1].content == "answer:second"


def test_bash_tool_is_temporarily_disabled_for_all_entry_points():
    enabled_names = {spec.name for spec in runtime.create_tools(enable_bash=True).specs}
    disabled_names = {
        spec.name for spec in runtime.create_tools(enable_bash=False).specs
    }

    assert "bash" not in enabled_names
    assert enabled_names == disabled_names == {"web_search", "web_fetch"}


def test_speech_environment_uses_separate_provider_configuration(monkeypatch):
    created = []

    class FakeSpeechProvider:
        def __init__(self, **options):
            created.append(options)

    monkeypatch.setattr(runtime, "OpenAISpeechProvider", FakeSpeechProvider)

    service = runtime.create_speech_service_from_env(
        {
            "YYYBOT_PROVIDER": "ollama",
            "YYYBOT_SPEECH_PROVIDER": "openai",
            "YYYBOT_SPEECH_API_KEY": "speech-key",
            "YYYBOT_SPEECH_BASE_URL": "https://speech.example/v1",
            "YYYBOT_STT_MODEL": "stt-model",
            "YYYBOT_TTS_MODEL": "tts-model",
            "YYYBOT_TTS_VOICE": "coral",
        }
    )

    assert service is not None
    assert service.transcriber is service.synthesizer
    assert created == [
        {
            "api_key": "speech-key",
            "base_url": "https://speech.example/v1",
            "transcription_model": "stt-model",
            "synthesis_model": "tts-model",
            "default_voice": "coral",
        }
    ]


def test_local_chat_does_not_enable_cloud_speech_implicitly():
    assert runtime.create_speech_service_from_env(
        {"YYYBOT_PROVIDER": "ollama"}
    ) is None
    assert runtime.create_speech_service_from_env(
        {
            "YYYBOT_PROVIDER": "openai",
            "YYYBOT_BASE_URL": "http://localhost:8080/v1",
        }
    ) is None


def test_speech_environment_can_mix_local_stt_with_openai_tts(monkeypatch):
    local_options = []
    openai_options = []

    class FakeLocalTranscriber:
        def __init__(self, **options):
            local_options.append(options)

    class FakeOpenAIProvider:
        def __init__(self, **options):
            openai_options.append(options)

    monkeypatch.setattr(runtime, "SherpaSpeechTranscriber", FakeLocalTranscriber)
    monkeypatch.setattr(runtime, "OpenAISpeechProvider", FakeOpenAIProvider)

    service = runtime.create_speech_service_from_env(
        {
            "YYYBOT_PROVIDER": "openai",
            "YYYBOT_STT_PROVIDER": "sherpa",
            "YYYBOT_TTS_PROVIDER": "openai",
            "YYYBOT_STT_MODEL_DIR": "/models/sensevoice",
            "YYYBOT_STT_THREADS": "2",
            "YYYBOT_API_KEY": "chat-key",
        }
    )

    assert service is not None
    assert isinstance(service.transcriber, FakeLocalTranscriber)
    assert isinstance(service.synthesizer, FakeOpenAIProvider)
    assert local_options == [
        {
            "model_path": "/models/sensevoice/model.int8.onnx",
            "tokens_path": "/models/sensevoice/tokens.txt",
            "language": "zh",
            "num_threads": 2,
            "provider": "cpu",
        }
    ]
    assert openai_options[0]["api_key"] == "chat-key"


def test_speech_environment_can_mix_local_stt_with_qwen3_tts(monkeypatch):
    qwen_options = []

    class FakeLocalTranscriber:
        def __init__(self, **options):
            pass

    class FakeQwenSynthesizer:
        def __init__(self, **options):
            qwen_options.append(options)

    monkeypatch.setattr(runtime, "SherpaSpeechTranscriber", FakeLocalTranscriber)
    monkeypatch.setattr(runtime, "Qwen3SpeechSynthesizer", FakeQwenSynthesizer)

    service = runtime.create_speech_service_from_env(
        {
            "YYYBOT_STT_PROVIDER": "sherpa",
            "YYYBOT_TTS_PROVIDER": "qwen3",
            "YYYBOT_TTS_MODEL_DIR": "/models/qwen3-tts",
            "YYYBOT_TTS_CLONE_MODEL_DIR": "/models/qwen3-base",
            "YYYBOT_HOME": "/data/yyybot",
            "YYYBOT_TTS_DEVICE": "cuda:1",
            "YYYBOT_TTS_VOICE": "Serena",
        }
    )

    assert service is not None
    assert isinstance(service.synthesizer, FakeQwenSynthesizer)
    assert qwen_options == [
        {
            "model_id": Path("/models/qwen3-tts"),
            "device": "cuda:1",
            "dtype": "bfloat16",
            "attention": "sdpa",
            "language": "Chinese",
            "max_new_tokens": 1024,
            "default_voice": "Serena",
            "clone_model_id": Path("/models/qwen3-base"),
            "voices_dir": Path("/data/yyybot/voices"),
            "preload_custom_voices": True,
        }
    ]


def test_speech_environment_can_use_cosyvoice_streaming_clone(monkeypatch):
    cosy_options = []

    class FakeCosyVoiceSynthesizer:
        def __init__(self, **options):
            cosy_options.append(options)

    monkeypatch.setattr(
        runtime,
        "CosyVoiceSpeechSynthesizer",
        FakeCosyVoiceSynthesizer,
    )

    service = runtime.create_speech_service_from_env(
        {
            "YYYBOT_STT_PROVIDER": "off",
            "YYYBOT_TTS_PROVIDER": "cosyvoice3",
            "YYYBOT_TTS_MODEL_DIR": "/models/cosyvoice3",
            "YYYBOT_COSYVOICE_REPO": "/vendor/CosyVoice",
            "YYYBOT_TTS_VOICES_DIR": "/data/voices",
            "YYYBOT_TTS_DEVICE": "cuda:1",
            "YYYBOT_TTS_VOICE": "personal-test",
        }
    )

    assert service is not None
    assert isinstance(service.synthesizer, FakeCosyVoiceSynthesizer)
    assert cosy_options == [
        {
            "model_id": Path("/models/cosyvoice3"),
            "repo_path": Path("/vendor/CosyVoice"),
            "voices_dir": Path("/data/voices"),
            "device": "cuda:1",
            "fp16": True,
            "default_voice": "personal-test",
        }
    ]


def test_sherpa_speech_provider_enables_only_transcription(monkeypatch):
    class FakeLocalTranscriber:
        def __init__(self, **options):
            pass

    monkeypatch.setattr(runtime, "SherpaSpeechTranscriber", FakeLocalTranscriber)

    service = runtime.create_speech_service_from_env(
        {
            "YYYBOT_SPEECH_PROVIDER": "sherpa",
            "YYYBOT_STT_MODEL_DIR": "/models/sensevoice",
        }
    )

    assert service is not None
    assert isinstance(service.transcriber, FakeLocalTranscriber)
    assert service.synthesizer is None


def test_service_binds_agent_execution_to_selected_workspace(tmp_path):
    observed_directories = []

    class RecordingAgent:
        async def run(self, messages):
            observed_directories.append(current_execution_directory())
            final = Message(role="assistant", content="done")
            return AgentResult(
                final_message=final,
                messages=(*messages, final),
                responses=(ModelResponse(final),),
            )

    service = ChatService(
        WorkspaceManager(tmp_path),
        lambda on_event: RecordingAgent(),
    )
    workspace = service.create_workspace(name="Files", workspace_id="files")
    service.create_session(workspace.workspace_id, session_id="chat")

    asyncio.run(service.run("files", "chat", "run here"))

    assert observed_directories == [workspace.directory.resolve()]
