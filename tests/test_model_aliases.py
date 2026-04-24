from speaches.model_aliases import resolve_model_id_alias_for_backend


def test_flat_string_alias_applies_to_all_backends() -> None:
    aliases = {"tts-1": "speaches-ai/Kokoro-82M-v1.0-ONNX"}
    assert (
        resolve_model_id_alias_for_backend("tts-1", "faster_whisper", aliases=aliases)
        == "speaches-ai/Kokoro-82M-v1.0-ONNX"
    )
    assert (
        resolve_model_id_alias_for_backend("tts-1", "whisper_cpp", aliases=aliases)
        == "speaches-ai/Kokoro-82M-v1.0-ONNX"
    )


def test_per_backend_mapping_picks_correct_variant() -> None:
    aliases = {
        "whisper-1": {
            "faster_whisper": "Systran/faster-whisper-large-v3",
            "whisper_cpp": "ggerganov/whisper.cpp/ggml-large-v3.bin",
        }
    }
    assert (
        resolve_model_id_alias_for_backend("whisper-1", "faster_whisper", aliases=aliases)
        == "Systran/faster-whisper-large-v3"
    )
    assert (
        resolve_model_id_alias_for_backend("whisper-1", "whisper_cpp", aliases=aliases)
        == "ggerganov/whisper.cpp/ggml-large-v3.bin"
    )


def test_unknown_alias_returns_unchanged() -> None:
    assert resolve_model_id_alias_for_backend("custom/model", "faster_whisper", aliases={}) == "custom/model"


def test_missing_backend_in_mapping_returns_unchanged() -> None:
    aliases = {"whisper-1": {"faster_whisper": "Systran/faster-whisper-large-v3"}}
    assert resolve_model_id_alias_for_backend("whisper-1", "whisper_cpp", aliases=aliases) == "whisper-1"
