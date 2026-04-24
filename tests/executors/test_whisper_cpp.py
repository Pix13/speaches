from unittest.mock import MagicMock, patch

import numpy as np
import openai.types.audio

from speaches.audio import Audio
from speaches.executors.shared.handler_protocol import TranscriptionRequest
from speaches.executors.silero_vad_v5 import VadOptions
from speaches.executors.whisper_cpp import (
    WhisperCppModelFiles,
    WhisperCppModelManager,
    whisper_cpp_model_registry,
)


def test_registry_exists() -> None:
    assert whisper_cpp_model_registry is not None


def test_model_files_fields() -> None:
    fields = set(WhisperCppModelFiles.model_fields)
    assert fields == {"model"}


def _make_request(stream: bool = False) -> TranscriptionRequest:
    audio = Audio(data=np.zeros(16000, dtype=np.float32), sample_rate=16000)
    return TranscriptionRequest(
        audio=audio,
        model="ggerganov/whisper.cpp/ggml-tiny.en.bin",
        stream=stream,
        language="en",
        prompt=None,
        response_format="json",
        temperature=0.0,
        hotwords=None,
        timestamp_granularities=["segment"],
        speech_segments=[],
        vad_options=VadOptions(),
        without_timestamps=True,
    )


def _fake_segment(text: str, start: float = 0.0, end: float = 1.0) -> MagicMock:
    s = MagicMock()
    s.text = text
    s.t0 = int(start * 100)
    s.t1 = int(end * 100)
    return s


def test_non_streaming_transcription_returns_json() -> None:
    mgr = WhisperCppModelManager(ttl=-1)
    fake_model = MagicMock()
    fake_model.transcribe.return_value = [_fake_segment(" hello world")]

    with patch.object(WhisperCppModelManager, "_load_fn", return_value=fake_model):
        resp = mgr.handle_non_streaming_transcription_request(_make_request())

    assert isinstance(resp, openai.types.audio.Transcription)
    assert resp.text.strip() == "hello world"


def test_streaming_transcription_emits_delta_then_done() -> None:
    mgr = WhisperCppModelManager(ttl=-1)
    fake_model = MagicMock()
    fake_model.transcribe.return_value = [
        _fake_segment(" hello", 0.0, 0.5),
        _fake_segment(" world", 0.5, 1.0),
    ]

    with patch.object(WhisperCppModelManager, "_load_fn", return_value=fake_model):
        events = list(mgr.handle_streaming_transcription_request(_make_request(stream=True)))

    assert [e.type for e in events] == ["transcript.text.delta", "transcript.text.delta", "transcript.text.done"]
    assert events[-1].text.strip() == "hello world"


def test_translation_returns_json() -> None:
    from speaches.executors.shared.handler_protocol import TranslationRequest
    from speaches.executors.silero_vad_v5 import VadOptions

    audio = Audio(data=np.zeros(16000, dtype=np.float32), sample_rate=16000)
    req = TranslationRequest(
        audio=audio,
        model="ggerganov/whisper.cpp/ggml-tiny.en.bin",
        prompt=None,
        response_format="json",
        temperature=0.0,
        speech_segments=[],
        vad_options=VadOptions(),
    )

    mgr = WhisperCppModelManager(ttl=-1)
    fake_model = MagicMock()
    fake_model.transcribe.return_value = [_fake_segment(" bonjour")]
    with patch.object(WhisperCppModelManager, "_load_fn", return_value=fake_model):
        resp = mgr.handle_translation_request(req)

    assert isinstance(resp, openai.types.audio.Translation)
    assert resp.text.strip() == "bonjour"
    assert fake_model.transcribe.call_args.kwargs["translate"] is True
