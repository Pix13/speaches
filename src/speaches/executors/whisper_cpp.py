from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING

import huggingface_hub
import openai.types.audio
from pydantic import BaseModel

from speaches.api_types import Model
from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.hf_utils import HfModelFilter
from speaches.model_registry import ModelRegistry
from speaches.text_utils import format_as_srt, format_as_vtt
from speaches.tracing import traced, traced_generator

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from speaches.executors.shared.handler_protocol import (
        NonStreamingTranscriptionResponse,
        StreamingTranscriptionEvent,
        TranscriptionRequest,
        TranslationRequest,
        TranslationResponse,
    )

LIBRARY_NAME = "whisper.cpp"
TASK_NAME_TAG = "automatic-speech-recognition"
GGML_REPO_ID = "ggerganov/whisper.cpp"

logger = logging.getLogger(__name__)

hf_model_filter = HfModelFilter(
    library_name=LIBRARY_NAME,
    task=TASK_NAME_TAG,
)


class WhisperCppModelFiles(BaseModel):
    model: Path


class WhisperCppModelRegistry(ModelRegistry[Model, WhisperCppModelFiles]):
    def list_remote_models(self) -> Generator[Model]:
        info = huggingface_hub.model_info(GGML_REPO_ID, files_metadata=True)
        assert info.created_at is not None
        for sibling in info.siblings or []:
            name = sibling.rfilename
            if not (name.startswith("ggml-") and name.endswith(".bin")):
                continue
            yield Model(
                id=f"{GGML_REPO_ID}/{name}",
                created=int(info.created_at.timestamp()),
                owned_by=GGML_REPO_ID.split("/")[0],
                language=[],
                task=TASK_NAME_TAG,
            )

    def list_local_models(self) -> Generator[Model]:
        cache = Path(huggingface_hub.constants.HF_HUB_CACHE)
        repo_dir = cache / f"models--{GGML_REPO_ID.replace('/', '--')}"
        if not repo_dir.exists():
            return
        for bin_file in repo_dir.rglob("ggml-*.bin"):
            yield Model(
                id=f"{GGML_REPO_ID}/{bin_file.name}",
                created=int(bin_file.stat().st_mtime),
                owned_by=GGML_REPO_ID.split("/")[0],
                language=[],
                task=TASK_NAME_TAG,
            )

    def get_model_files(self, model_id: str) -> WhisperCppModelFiles:
        repo_id, filename = self._split(model_id)
        path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)
        return WhisperCppModelFiles(model=Path(path))

    def download_model_files(self, model_id: str) -> None:
        repo_id, filename = self._split(model_id)
        huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)

    @staticmethod
    def _split(model_id: str) -> tuple[str, str]:
        parts = model_id.rsplit("/", 1)
        if len(parts) != 2 or not parts[1].endswith(".bin"):
            raise ValueError(f"whisper_cpp model id must be '<repo>/<file>.bin', got {model_id!r}")
        return parts[0], parts[1]


whisper_cpp_model_registry = WhisperCppModelRegistry(hf_model_filter=hf_model_filter)


def _segment_text(segments: Iterable) -> str:
    return "".join(s.text for s in segments).strip()


def _segment_start(seg: object) -> float:
    return float(seg.t0) / 100.0  # type: ignore[attr-defined]


def _segment_end(seg: object) -> float:
    return float(seg.t1) / 100.0  # type: ignore[attr-defined]


def _segments_to_translation_response(
    segments: list,
    *,
    language: str,
    duration: float,
    response_format: openai.types.AudioResponseFormat,
) -> TranslationResponse:
    text = _segment_text(segments)
    match response_format:
        case "text":
            return text, "text/plain"
        case "json":
            return openai.types.audio.Translation(text=text)
        case "verbose_json":
            return openai.types.audio.TranslationVerbose(
                language=language,
                duration=duration,
                text=text,
                segments=[
                    openai.types.audio.TranscriptionSegment(
                        id=i,
                        seek=0,
                        start=_segment_start(s),
                        end=_segment_end(s),
                        text=s.text,
                        tokens=[],
                        temperature=0.0,
                        avg_logprob=0.0,
                        compression_ratio=0.0,
                        no_speech_prob=0.0,
                    )
                    for i, s in enumerate(segments)
                ],
            )
        case "vtt":
            return (
                "".join(format_as_vtt(s.text, _segment_start(s), _segment_end(s), i) for i, s in enumerate(segments)),
                "text/vtt",
            )
        case "srt":
            return (
                "".join(format_as_srt(s.text, _segment_start(s), _segment_end(s), i) for i, s in enumerate(segments)),
                "text/plain",
            )


def _segments_to_transcription_response(
    segments: list,
    *,
    language: str,
    duration: float,
    response_format: openai.types.AudioResponseFormat,
    word_timestamps: bool,  # noqa: ARG001
) -> NonStreamingTranscriptionResponse:
    text = _segment_text(segments)
    match response_format:
        case "text":
            return text, "text/plain"
        case "json":
            return openai.types.audio.Transcription(text=text)
        case "verbose_json":
            return openai.types.audio.TranscriptionVerbose(
                language=language,
                duration=duration,
                text=text,
                segments=[
                    openai.types.audio.TranscriptionSegment(
                        id=i,
                        seek=0,
                        start=_segment_start(s),
                        end=_segment_end(s),
                        text=s.text,
                        tokens=[],
                        temperature=0.0,
                        avg_logprob=0.0,
                        compression_ratio=0.0,
                        no_speech_prob=0.0,
                    )
                    for i, s in enumerate(segments)
                ],
                words=None,
            )
        case "vtt":
            return (
                "".join(format_as_vtt(s.text, _segment_start(s), _segment_end(s), i) for i, s in enumerate(segments)),
                "text/vtt",
            )
        case "srt":
            return (
                "".join(format_as_srt(s.text, _segment_start(s), _segment_end(s), i) for i, s in enumerate(segments)),
                "text/plain",
            )


class WhisperCppModelManager(BaseModelManager[object]):
    def __init__(self, ttl: int) -> None:
        super().__init__(ttl)

    def _load_fn(self, model_id: str) -> object:
        from pywhispercpp.model import Model as WhisperCppModel

        model_files = whisper_cpp_model_registry.get_model_files(model_id)
        return WhisperCppModel(model=str(model_files.model))

    @traced()
    def handle_non_streaming_transcription_request(
        self,
        request: TranscriptionRequest,
        **_kwargs,
    ) -> NonStreamingTranscriptionResponse:
        if request.response_format == "diarized_json":
            raise NotImplementedError(
                f"'{request.response_format}' response format is not supported for '{request.model}' model."
            )
        started = time.perf_counter()
        with self.load_model(request.model) as model:
            segments = list(
                model.transcribe(
                    request.audio.data,
                    language=request.language or "auto",
                    initial_prompt=request.prompt or "",
                    temperature=request.temperature,
                    translate=False,
                    token_timestamps="word" in request.timestamp_granularities,
                )
            )
        res = _segments_to_transcription_response(
            segments,
            language=request.language or "auto",
            duration=request.audio.duration,
            response_format=request.response_format,
            word_timestamps="word" in request.timestamp_granularities,
        )
        logger.info(f"Transcribed {request.audio.duration} seconds of audio in {time.perf_counter() - started} seconds")
        return res

    @traced_generator()
    def handle_streaming_transcription_request(
        self,
        request: TranscriptionRequest,
        **_kwargs,
    ) -> Generator[StreamingTranscriptionEvent]:
        started = time.perf_counter()
        with self.load_model(request.model) as model:
            segments = model.transcribe(
                request.audio.data,
                language=request.language or "auto",
                initial_prompt=request.prompt or "",
                temperature=request.temperature,
                translate=False,
                token_timestamps="word" in request.timestamp_granularities,
            )
            collected: list[str] = []
            for segment in segments:
                collected.append(segment.text)
                yield openai.types.audio.TranscriptionTextDeltaEvent(
                    type="transcript.text.delta", delta=segment.text, logprobs=None
                )
            yield openai.types.audio.TranscriptionTextDoneEvent(
                type="transcript.text.done", text="".join(collected), logprobs=None
            )
        logger.info(f"Transcribed {request.audio.duration} seconds of audio in {time.perf_counter() - started} seconds")

    def handle_transcription_request(
        self,
        request: TranscriptionRequest,
        **kwargs,
    ) -> NonStreamingTranscriptionResponse | Generator[StreamingTranscriptionEvent]:
        if request.stream:
            return self.handle_streaming_transcription_request(request, **kwargs)
        return self.handle_non_streaming_transcription_request(request, **kwargs)

    @traced()
    def handle_translation_request(
        self,
        request: TranslationRequest,
        **_kwargs,
    ) -> TranslationResponse:
        if request.response_format == "diarized_json":
            raise NotImplementedError(
                f"'{request.response_format}' response format is not supported for '{request.model}' model."
            )
        with self.load_model(request.model) as model:
            segments = list(
                model.transcribe(
                    request.audio.data,
                    language="auto",
                    initial_prompt=request.prompt or "",
                    temperature=request.temperature,
                    translate=True,
                    token_timestamps=False,
                )
            )
            return _segments_to_translation_response(
                segments,
                language="en",
                duration=request.audio.duration,
                response_format=request.response_format,
            )
