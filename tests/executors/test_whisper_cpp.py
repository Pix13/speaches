from speaches.executors.whisper_cpp import (
    WhisperCppModelFiles,
    whisper_cpp_model_registry,
)


def test_registry_exists() -> None:
    assert whisper_cpp_model_registry is not None


def test_model_files_fields() -> None:
    fields = set(WhisperCppModelFiles.model_fields)
    assert fields == {"model"}
