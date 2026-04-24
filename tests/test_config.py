import pytest

from speaches.config import Config


def test_whisper_backend_defaults_to_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WHISPER_BACKEND", raising=False)
    cfg = Config()
    assert cfg.whisper_backend == "faster_whisper"


def test_whisper_backend_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WHISPER_BACKEND", "whisper_cpp")
    cfg = Config()
    assert cfg.whisper_backend == "whisper_cpp"


def test_whisper_backend_rejects_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    import pydantic

    monkeypatch.setenv("WHISPER_BACKEND", "nope")
    try:
        Config()
    except pydantic.ValidationError:
        return
    raise AssertionError("expected ValidationError")
