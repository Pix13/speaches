from speaches.config import Config
from speaches.executors.shared.registry import ExecutorRegistry


def test_faster_whisper_backend_exposes_ctranslate2_registry() -> None:
    config = Config(whisper_backend="faster_whisper")
    registry = ExecutorRegistry(config)
    assert registry._whisper_executor.model_registry.hf_model_filter.library_name == "ctranslate2"


def test_whisper_cpp_backend_exposes_whisper_cpp_registry() -> None:
    config = Config(whisper_backend="whisper_cpp")
    registry = ExecutorRegistry(config)
    assert registry._whisper_executor.model_registry.hf_model_filter.library_name == "whisper.cpp"
