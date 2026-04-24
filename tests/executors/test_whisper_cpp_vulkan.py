from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speaches.audio import Audio
from speaches.executors.shared.handler_protocol import TranscriptionRequest
from speaches.executors.silero_vad_v5 import VadOptions
from speaches.executors.whisper_cpp import WhisperCppModelManager


@pytest.mark.requires_vulkan
def test_tiny_model_transcribes_sample_audio() -> None:
    audio_path = Path(__file__).resolve().parents[2] / "audio.wav"
    data, sr = sf.read(str(audio_path), dtype="float32")
    assert sr == 16000, f"sample audio.wav expected at 16kHz, got {sr}"
    if data.ndim == 2:
        data = data.mean(axis=1).astype(np.float32)

    audio = Audio(data, sample_rate=16000)

    mgr = WhisperCppModelManager(ttl=0)
    req = TranscriptionRequest(
        audio=audio,
        model="ggerganov/whisper.cpp/ggml-tiny.en.bin",
        stream=False,
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
    resp = mgr.handle_non_streaming_transcription_request(req)
    assert resp.text.strip() != ""
