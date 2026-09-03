"""Factories for assembling the default yyybot runtime."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from .agent import Agent, EventHandler
from .models import Model
from .providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    Provider,
    VLLMProvider,
)
from .service import ChatService
from .speech import (
    CosyVoiceSpeechSynthesizer,
    OpenAISpeechProvider,
    Qwen3SpeechSynthesizer,
    SherpaSpeechTranscriber,
    SpeechService,
)
from .tools import ToolRegistry
from .web import web_fetch, web_search
from .workspace import WorkspaceManager


def create_provider(
    provider: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Provider:
    options: dict[str, str] = {}
    if api_key:
        options["api_key"] = api_key
    if base_url:
        options["base_url"] = base_url
    if provider == "openai":
        return OpenAIProvider(**options)
    if provider == "anthropic":
        return AnthropicProvider(**options)
    if provider == "vllm":
        return VLLMProvider(**options)
    if provider == "ollama":
        options.pop("api_key", None)
        return OllamaProvider(**options)
    raise ValueError(f"Unsupported provider: {provider}")


def create_tools(*, enable_bash: bool) -> ToolRegistry:
    tools = ToolRegistry()
    tools.add(web_search)
    tools.add(web_fetch)
    # Bash registration is temporarily disabled for every entry point.
    # if enable_bash:
    #     tools.add(bash)
    return tools


def create_chat_service(
    *,
    model_id: str,
    provider: str = "openai",
    api_key: str | None = None,
    base_url: str | None = None,
    yyybot_home: str | Path | None = None,
    enable_bash: bool = False,
) -> ChatService:
    model = Model(
        model_id=model_id,
        provider=create_provider(provider, api_key=api_key, base_url=base_url),
    )
    tools = create_tools(enable_bash=enable_bash)

    def agent_factory(on_event: EventHandler | None = None) -> Agent:
        return Agent(model, tools=tools, on_event=on_event)

    return ChatService(WorkspaceManager(yyybot_home), agent_factory)


def create_chat_service_from_env(
    environment: Mapping[str, str] | None = None,
) -> ChatService:
    env = os.environ if environment is None else environment
    model_id = env.get("YYYBOT_MODEL")
    if not model_id:
        raise RuntimeError("Set YYYBOT_MODEL before starting the server.")
    enable_bash = env.get("YYYBOT_ENABLE_BASH", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    return create_chat_service(
        model_id=model_id,
        provider=env.get("YYYBOT_PROVIDER", "openai"),
        api_key=env.get("YYYBOT_API_KEY"),
        base_url=env.get("YYYBOT_BASE_URL"),
        yyybot_home=env.get("YYYBOT_HOME"),
        enable_bash=enable_bash,
    )


def create_speech_service_from_env(
    environment: Mapping[str, str] | None = None,
) -> SpeechService | None:
    """Build the optional speech adapter without coupling it to the Agent."""

    env = os.environ if environment is None else environment
    configured_provider = env.get("YYYBOT_SPEECH_PROVIDER")
    if configured_provider is None:
        configured_provider = (
            "openai"
            if (
                env.get("YYYBOT_PROVIDER", "openai") == "openai"
                and not env.get("YYYBOT_BASE_URL")
            )
            else "off"
        )
    disabled = {"", "0", "false", "no", "off", "none"}
    default_provider = configured_provider.strip().lower()
    stt_provider = env.get("YYYBOT_STT_PROVIDER", default_provider).strip().lower()
    default_tts_provider = (
        "off" if default_provider in {"sherpa", "sensevoice"} else default_provider
    )
    tts_provider = env.get("YYYBOT_TTS_PROVIDER", default_tts_provider).strip().lower()
    if stt_provider in disabled and tts_provider in disabled:
        return None

    openai_provider: OpenAISpeechProvider | None = None

    def get_openai_provider() -> OpenAISpeechProvider:
        nonlocal openai_provider
        if openai_provider is None:
            openai_provider = OpenAISpeechProvider(
                api_key=(
                    env.get("YYYBOT_SPEECH_API_KEY")
                    or env.get("YYYBOT_API_KEY")
                    or env.get("OPENAI_API_KEY")
                ),
                base_url=env.get(
                    "YYYBOT_SPEECH_BASE_URL",
                    "https://api.openai.com/v1",
                ),
                transcription_model=env.get("YYYBOT_STT_MODEL", "gpt-transcribe"),
                synthesis_model=env.get("YYYBOT_TTS_MODEL", "gpt-4o-mini-tts"),
                default_voice=env.get("YYYBOT_TTS_VOICE", "marin"),
            )
        return openai_provider

    local_transcriber_options: dict[str, str | int] | None = None
    if stt_provider in disabled:
        transcriber = None
    elif stt_provider == "openai":
        transcriber = get_openai_provider()
    elif stt_provider in {"sherpa", "sensevoice"}:
        yyybot_home = Path(env.get("YYYBOT_HOME", Path.home() / ".yyybot"))
        default_model_dir = yyybot_home / "models" / "sensevoice-small-int8"
        model_dir = Path(
            env.get("YYYBOT_STT_MODEL_DIR", str(default_model_dir))
        ).expanduser()
        local_transcriber_options = {
            "model_path": env.get(
                "YYYBOT_STT_MODEL_PATH", str(model_dir / "model.int8.onnx")
            ),
            "tokens_path": env.get(
                "YYYBOT_STT_TOKENS_PATH", str(model_dir / "tokens.txt")
            ),
            "language": env.get("YYYBOT_STT_LANGUAGE", "zh"),
            "num_threads": int(env.get("YYYBOT_STT_THREADS", "4")),
            "provider": env.get("YYYBOT_STT_EXECUTION_PROVIDER", "cpu"),
        }
        transcriber = None
    else:
        raise ValueError(f"Unsupported STT provider: {stt_provider}")

    if tts_provider in disabled:
        synthesizer = None
    elif tts_provider == "openai":
        synthesizer = get_openai_provider()
    elif tts_provider in {"qwen", "qwen3", "qwen3-tts"}:
        configured_model = env.get("YYYBOT_TTS_MODEL_DIR")
        configured_clone_model = env.get("YYYBOT_TTS_CLONE_MODEL_DIR")
        yyybot_home = Path(env.get("YYYBOT_HOME", Path.home() / ".yyybot"))
        synthesizer = Qwen3SpeechSynthesizer(
            model_id=(
                Path(configured_model).expanduser()
                if configured_model
                else env.get(
                    "YYYBOT_TTS_MODEL",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                )
            ),
            device=env.get("YYYBOT_TTS_DEVICE", "cuda:1"),
            dtype=env.get("YYYBOT_TTS_DTYPE", "bfloat16"),
            attention=env.get("YYYBOT_TTS_ATTENTION", "sdpa"),
            language=env.get("YYYBOT_TTS_LANGUAGE", "Chinese"),
            max_new_tokens=int(env.get("YYYBOT_TTS_MAX_NEW_TOKENS", "1024")),
            default_voice=env.get("YYYBOT_TTS_VOICE", "Serena"),
            clone_model_id=(
                Path(configured_clone_model).expanduser()
                if configured_clone_model
                else env.get(
                    "YYYBOT_TTS_CLONE_MODEL",
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                )
            ),
            voices_dir=Path(
                env.get("YYYBOT_TTS_VOICES_DIR", str(yyybot_home / "voices"))
            ).expanduser(),
            preload_custom_voices=env.get(
                "YYYBOT_TTS_PRELOAD_CUSTOM_VOICES", "1"
            ).lower()
            not in {"0", "false", "no", "off"},
        )
    elif tts_provider in {"cosy", "cosyvoice", "cosyvoice3"}:
        yyybot_home = Path(env.get("YYYBOT_HOME", Path.home() / ".yyybot"))
        project_root = Path(__file__).resolve().parents[2]
        synthesizer = CosyVoiceSpeechSynthesizer(
            model_id=Path(
                env.get(
                    "YYYBOT_TTS_MODEL_DIR",
                    str(yyybot_home / "models" / "Fun-CosyVoice3-0.5B"),
                )
            ).expanduser(),
            repo_path=Path(
                env.get(
                    "YYYBOT_COSYVOICE_REPO",
                    str(project_root / ".vendor" / "CosyVoice"),
                )
            ).expanduser(),
            voices_dir=Path(
                env.get("YYYBOT_TTS_VOICES_DIR", str(yyybot_home / "voices"))
            ).expanduser(),
            device=env.get("YYYBOT_TTS_DEVICE", "cuda:1"),
            fp16=env.get("YYYBOT_TTS_FP16", "1").lower()
            not in {"0", "false", "no", "off"},
            default_voice=env.get("YYYBOT_TTS_VOICE"),
        )
    else:
        raise ValueError(f"Unsupported TTS provider: {tts_provider}")

    # Load CosyVoice's ONNX Runtime sessions before sherpa-onnx. Loading them in
    # the reverse order can make the two native runtimes collide during startup.
    if local_transcriber_options is not None:
        transcriber = SherpaSpeechTranscriber(**local_transcriber_options)

    return SpeechService(transcriber=transcriber, synthesizer=synthesizer)
