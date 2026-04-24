from speaches.config import Config
from speaches.executors.shared.registry import ExecutorRegistry
from speaches.executors.whisper import WhisperModelManager
from speaches.executors.whisper_cpp import WhisperCppModelManager


def test_faster_whisper_backend_wires_faster_whisper_manager() -> None:
    config = Config(whisper_backend="faster_whisper")
    registry = ExecutorRegistry(config)
    assert isinstance(registry._whisper_executor.model_manager, WhisperModelManager)  # noqa: SLF001


def test_whisper_cpp_backend_wires_whisper_cpp_manager() -> None:
    config = Config(whisper_backend="whisper_cpp")
    registry = ExecutorRegistry(config)
    assert isinstance(registry._whisper_executor.model_manager, WhisperCppModelManager)  # noqa: SLF001


def test_transcription_property_exposes_selected_whisper_executor() -> None:
    config = Config(whisper_backend="whisper_cpp")
    registry = ExecutorRegistry(config)
    (whisper_exec, *_) = registry.transcription
    assert whisper_exec is registry._whisper_executor  # noqa: SLF001
    assert isinstance(whisper_exec.model_manager, WhisperCppModelManager)


def test_translation_property_exposes_selected_whisper_executor() -> None:
    config = Config(whisper_backend="whisper_cpp")
    registry = ExecutorRegistry(config)
    (whisper_exec,) = registry.translation
    assert isinstance(whisper_exec.model_manager, WhisperCppModelManager)
