# Vulkan Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `whisper.cpp`-powered STT executor that supports Vulkan (plus CPU/CUDA), selectable via a new `WHISPER_BACKEND` config, and ship a Vulkan-flavored Docker image so a single install serves STT-on-Vulkan and CPU-backed TTS on non-NVIDIA hardware.

**Architecture:** Add a sibling executor `WhisperCppModelManager` that implements the existing `TranscriptionHandler` / `TranslationHandler` / model-registry protocols. Selection happens at DI time in `dependencies.py` from a new `Config.whisper_backend` field; only the chosen executor is registered. Model aliases schema is bumped to per-backend. A new `Dockerfile.vulkan` + `compose.vulkan.yaml` builds `pywhispercpp` from source with `-DGGML_VULKAN=1`.

**Tech Stack:** Python 3.12, FastAPI, `pydantic_settings`, `faster-whisper` (existing), `pywhispercpp` (new), `huggingface_hub`, Docker, Vulkan SDK / Mesa Vulkan drivers, pytest (with custom `requires_vulkan` marker).

**Spec:** `docs/superpowers/specs/2026-04-24-vulkan-backend-design.md`

---

## File Structure

Files created in this plan:

- `src/speaches/executors/whisper_cpp.py` — new executor: `WhisperCppModelRegistry`, `WhisperCppModelManager`, result adapters.
- `tests/executors/test_whisper_cpp.py` — CPU-mode unit tests for the new executor.
- `tests/executors/test_whisper_cpp_vulkan.py` — Vulkan-marked integration tests.
- `Dockerfile.vulkan` — Vulkan-capable image with source-built `pywhispercpp`.
- `compose.vulkan.yaml` — compose flavor pointing at the Vulkan image + device mounts.
- `docs/installation/vulkan.md` — install / deploy docs.
- `docs/configuration/whisper-backend.md` — backend selection docs + param matrix.

Files modified in this plan:

- `pyproject.toml` — add `[project.optional-dependencies].vulkan`.
- `src/speaches/config.py` — add `whisper_backend` field.
- `src/speaches/model_aliases.py` — support per-backend alias values.
- `model_aliases.json` — add `whisper_cpp` variants for whisper aliases.
- `src/speaches/dependencies.py` — select & register Whisper executor by backend.
- `conftest.py` (root) — register `requires_vulkan` pytest marker + env gate.
- `README.md` / `mkdocs.yml` nav — link new docs.

---

## Task 1: Add `whisper_backend` config setting

**Files:**
- Modify: `src/speaches/config.py`
- Test: `tests/test_config.py` (create if missing)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
import os
from speaches.config import Config


def test_whisper_backend_defaults_to_faster_whisper(monkeypatch) -> None:
    monkeypatch.delenv("WHISPER_BACKEND", raising=False)
    cfg = Config()
    assert cfg.whisper_backend == "faster_whisper"


def test_whisper_backend_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("WHISPER_BACKEND", "whisper_cpp")
    cfg = Config()
    assert cfg.whisper_backend == "whisper_cpp"


def test_whisper_backend_rejects_invalid(monkeypatch) -> None:
    import pydantic
    monkeypatch.setenv("WHISPER_BACKEND", "nope")
    try:
        Config()
    except pydantic.ValidationError:
        return
    raise AssertionError("expected ValidationError")
```

- [ ] **Step 2: Run test — expect failure**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `whisper_backend` attribute does not exist.

- [ ] **Step 3: Add the field**

In `src/speaches/config.py`, near the top type aliases add:

```python
type WhisperBackend = Literal["faster_whisper", "whisper_cpp"]
```

Inside class `Config(BaseSettings)` (alphabetically near other top-level fields), add:

```python
    whisper_backend: WhisperBackend = "faster_whisper"
    """
    Which Whisper inference backend to use.
    - `faster_whisper` (default): CTranslate2-based, CPU/CUDA.
    - `whisper_cpp`: whisper.cpp via pywhispercpp, CPU/CUDA/Vulkan.
    """
```

- [ ] **Step 4: Run test — expect pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 5: Format and lint**

Run: `uv run ruff format src/speaches/config.py tests/test_config.py && uv run ruff check --fix src/speaches/config.py tests/test_config.py`

- [ ] **Step 6: Commit**

```bash
git add src/speaches/config.py tests/test_config.py
git commit -m "feat: add WHISPER_BACKEND config setting"
```

---

## Task 2: Extend `model_aliases.json` schema for per-backend values

**Files:**
- Modify: `src/speaches/model_aliases.py`
- Modify: `model_aliases.json`
- Test: `tests/test_model_aliases.py` (create)

- [ ] **Step 1: Write failing tests**

Create `tests/test_model_aliases.py`:

```python
from speaches.model_aliases import resolve_model_id_alias_for_backend


def test_flat_string_alias_applies_to_all_backends() -> None:
    aliases = {"tts-1": "speaches-ai/Kokoro-82M-v1.0-ONNX"}
    assert resolve_model_id_alias_for_backend("tts-1", "faster_whisper", aliases=aliases) == "speaches-ai/Kokoro-82M-v1.0-ONNX"
    assert resolve_model_id_alias_for_backend("tts-1", "whisper_cpp", aliases=aliases) == "speaches-ai/Kokoro-82M-v1.0-ONNX"


def test_per_backend_mapping_picks_correct_variant() -> None:
    aliases = {
        "whisper-1": {
            "faster_whisper": "Systran/faster-whisper-large-v3",
            "whisper_cpp": "ggerganov/whisper.cpp/ggml-large-v3.bin",
        }
    }
    assert resolve_model_id_alias_for_backend("whisper-1", "faster_whisper", aliases=aliases) == "Systran/faster-whisper-large-v3"
    assert resolve_model_id_alias_for_backend("whisper-1", "whisper_cpp", aliases=aliases) == "ggerganov/whisper.cpp/ggml-large-v3.bin"


def test_unknown_alias_returns_unchanged() -> None:
    assert resolve_model_id_alias_for_backend("custom/model", "faster_whisper", aliases={}) == "custom/model"


def test_missing_backend_in_mapping_returns_unchanged() -> None:
    aliases = {"whisper-1": {"faster_whisper": "Systran/faster-whisper-large-v3"}}
    assert resolve_model_id_alias_for_backend("whisper-1", "whisper_cpp", aliases=aliases) == "whisper-1"
```

- [ ] **Step 2: Run tests — expect failure**

Run: `uv run pytest tests/test_model_aliases.py -v`
Expected: FAIL — `resolve_model_id_alias_for_backend` does not exist.

- [ ] **Step 3: Rewrite `model_aliases.py`**

Replace the contents of `src/speaches/model_aliases.py` with:

```python
from functools import lru_cache
import json
from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, Field

from speaches.config import Config

MODEL_ID_ALIASES_PATH = Path("model_aliases.json")

type AliasValue = str | dict[str, str]


@lru_cache
def load_model_id_aliases() -> dict[str, AliasValue]:
    return json.loads(MODEL_ID_ALIASES_PATH.read_text())


def resolve_model_id_alias_for_backend(
    model_id: str,
    backend: str,
    *,
    aliases: dict[str, AliasValue] | None = None,
) -> str:
    table = load_model_id_aliases() if aliases is None else aliases
    value = table.get(model_id)
    if value is None:
        return model_id
    if isinstance(value, str):
        return value
    return value.get(backend, model_id)


def _resolve_current(model_id: str) -> str:
    backend = Config().whisper_backend
    return resolve_model_id_alias_for_backend(model_id, backend)


ModelId = Annotated[
    str,
    BeforeValidator(_resolve_current),
    Field(
        min_length=1,
        description="The ID of the model. You can get a list of available models by calling `/v1/models`.",
        examples=[
            "Systran/faster-distil-whisper-large-v3",
            "bofenghuang/whisper-large-v2-cv11-french-ct2",
        ],
    ),
]
```

- [ ] **Step 4: Update `model_aliases.json`**

Replace contents with:

```json
{
    "tts-1": "speaches-ai/Kokoro-82M-v1.0-ONNX",
    "tts-1-hd": "speaches-ai/Kokoro-82M-v1.0-ONNX",
    "whisper-1": {
        "faster_whisper": "Systran/faster-whisper-large-v3",
        "whisper_cpp": "ggerganov/whisper.cpp/ggml-large-v3.bin"
    }
}
```

- [ ] **Step 5: Run tests — expect pass**

Run: `uv run pytest tests/test_model_aliases.py tests/test_config.py -v`
Expected: all pass.

- [ ] **Step 6: Ensure no regression in callers**

Run: `uv run pytest -x -q`
Expected: no new failures. If any test depended on `resolve_model_id_alias` (the old function), update the import to `resolve_model_id_alias_for_backend` or the `ModelId` annotation (which still works transparently).

- [ ] **Step 7: Format, lint, commit**

```bash
uv run ruff format src/speaches/model_aliases.py tests/test_model_aliases.py
uv run ruff check --fix src/speaches/model_aliases.py tests/test_model_aliases.py
git add src/speaches/model_aliases.py tests/test_model_aliases.py model_aliases.json
git commit -m "feat: per-backend model alias resolution"
```

---

## Task 3: Add `vulkan` optional-dependency group

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the group**

In `pyproject.toml`, after the `[dependency-groups]` block, add:

```toml
[project.optional-dependencies]
vulkan = [
    "pywhispercpp>=1.3.0",
]
```

- [ ] **Step 2: Verify `uv` accepts it**

Run: `uv sync --extra vulkan --dry-run`
Expected: resolves without error. (On a CPU-only host, this resolves but will not attempt a Vulkan build — the Vulkan build flag is applied via env in the Dockerfile.)

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add vulkan optional-dependency group with pywhispercpp"
```

---

## Task 4: Skeleton `WhisperCppModelRegistry`

**Files:**
- Create: `src/speaches/executors/whisper_cpp.py`
- Test: `tests/executors/test_whisper_cpp.py`

- [ ] **Step 1: Write failing test for registry basics**

Create `tests/executors/test_whisper_cpp.py`:

```python
from speaches.executors.whisper_cpp import (
    WhisperCppModelFiles,
    whisper_cpp_model_registry,
)


def test_registry_exists() -> None:
    assert whisper_cpp_model_registry is not None


def test_model_files_fields() -> None:
    fields = set(WhisperCppModelFiles.model_fields)
    assert fields == {"model"}
```

- [ ] **Step 2: Run — expect import error**

Run: `uv run pytest tests/executors/test_whisper_cpp.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Create skeleton module**

Create `src/speaches/executors/whisper_cpp.py`:

```python
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import huggingface_hub
from pydantic import BaseModel

from speaches.api_types import Model
from speaches.hf_utils import HfModelFilter
from speaches.model_registry import ModelRegistry

if TYPE_CHECKING:
    from collections.abc import Generator

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
            raise ValueError(
                f"whisper_cpp model id must be '<repo>/<file>.bin', got {model_id!r}"
            )
        return parts[0], parts[1]


whisper_cpp_model_registry = WhisperCppModelRegistry(hf_model_filter=hf_model_filter)
```

- [ ] **Step 4: Run tests — expect pass**

Run: `uv run pytest tests/executors/test_whisper_cpp.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/speaches/executors/whisper_cpp.py tests/executors/test_whisper_cpp.py
uv run ruff check --fix src/speaches/executors/whisper_cpp.py tests/executors/test_whisper_cpp.py
git add src/speaches/executors/whisper_cpp.py tests/executors/test_whisper_cpp.py
git commit -m "feat: add WhisperCppModelRegistry skeleton"
```

---

## Task 5: `WhisperCppModelManager` — non-streaming transcription

**Files:**
- Modify: `src/speaches/executors/whisper_cpp.py`
- Test: `tests/executors/test_whisper_cpp.py`

- [ ] **Step 1: Write failing test**

Append to `tests/executors/test_whisper_cpp.py`:

```python
from unittest.mock import MagicMock, patch

import numpy as np
import openai.types.audio

from speaches.audio import Audio
from speaches.executors.shared.handler_protocol import TranscriptionRequest
from speaches.executors.silero_vad_v5 import VadOptions
from speaches.executors.whisper_cpp import WhisperCppModelManager


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


def _fake_segment(text: str, start: float = 0.0, end: float = 1.0):
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
```

- [ ] **Step 2: Run test — expect failure**

Run: `uv run pytest tests/executors/test_whisper_cpp.py::test_non_streaming_transcription_returns_json -v`
Expected: FAIL — `WhisperCppModelManager` does not exist.

- [ ] **Step 3: Implement the manager (non-streaming path)**

Append to `src/speaches/executors/whisper_cpp.py`:

```python
import time
from collections.abc import Generator, Iterable

import openai.types.audio

from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.executors.shared.handler_protocol import (
    NonStreamingTranscriptionResponse,
    StreamingTranscriptionEvent,
    TranscriptionRequest,
    TranslationRequest,
    TranslationResponse,
)
from speaches.text_utils import format_as_srt, format_as_vtt
from speaches.tracing import traced, traced_generator


def _segment_text(segments: Iterable) -> str:
    return "".join(s.text for s in segments).strip()


def _segment_start(seg) -> float:
    # pywhispercpp exposes `t0` / `t1` in centiseconds.
    return float(seg.t0) / 100.0


def _segment_end(seg) -> float:
    return float(seg.t1) / 100.0


class WhisperCppModelManager(BaseModelManager):
    def __init__(self, ttl: int) -> None:
        super().__init__(ttl)

    def _load_fn(self, model_id: str) -> object:
        from pywhispercpp.model import Model as WhisperCppModel  # lazy import

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
            logger.info(
                f"Transcribed {request.audio.duration} seconds of audio in {time.perf_counter() - started} seconds"
            )
            return res

    def handle_transcription_request(
        self,
        request: TranscriptionRequest,
        **kwargs,
    ) -> NonStreamingTranscriptionResponse | Generator[StreamingTranscriptionEvent]:
        if request.stream:
            return self.handle_streaming_transcription_request(request, **kwargs)
        return self.handle_non_streaming_transcription_request(request, **kwargs)


def _segments_to_transcription_response(
    segments: list,
    *,
    language: str,
    duration: float,
    response_format,
    word_timestamps: bool,
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
                "".join(
                    format_as_vtt(s.text, _segment_start(s), _segment_end(s), i)
                    for i, s in enumerate(segments)
                ),
                "text/vtt",
            )
        case "srt":
            return (
                "".join(
                    format_as_srt(s.text, _segment_start(s), _segment_end(s), i)
                    for i, s in enumerate(segments)
                ),
                "text/plain",
            )
```

Note: `BaseModelManager` is generic; we parameterize implicitly with `object` here to avoid importing `pywhispercpp` at module top. If `pyrefly` complains, change the class to `BaseModelManager[object]`.

- [ ] **Step 4: Run test — expect pass**

Run: `uv run pytest tests/executors/test_whisper_cpp.py -v`
Expected: all pass.

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format src/speaches/executors/whisper_cpp.py tests/executors/test_whisper_cpp.py
uv run ruff check --fix src/speaches/executors/whisper_cpp.py tests/executors/test_whisper_cpp.py
git add src/speaches/executors/whisper_cpp.py tests/executors/test_whisper_cpp.py
git commit -m "feat: whisper_cpp executor — non-streaming transcription"
```

---

## Task 6: Streaming transcription handler

**Files:**
- Modify: `src/speaches/executors/whisper_cpp.py`
- Test: `tests/executors/test_whisper_cpp.py`

- [ ] **Step 1: Write failing test**

Append to `tests/executors/test_whisper_cpp.py`:

```python
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
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/executors/test_whisper_cpp.py::test_streaming_transcription_emits_delta_then_done -v`
Expected: FAIL — `handle_streaming_transcription_request` not implemented.

- [ ] **Step 3: Implement the streaming handler**

Append to `WhisperCppModelManager` in `src/speaches/executors/whisper_cpp.py`:

```python
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
        logger.info(
            f"Transcribed {request.audio.duration} seconds of audio in {time.perf_counter() - started} seconds"
        )
```

- [ ] **Step 4: Run test — expect pass**

Run: `uv run pytest tests/executors/test_whisper_cpp.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/speaches/executors/whisper_cpp.py
uv run ruff check --fix src/speaches/executors/whisper_cpp.py
git add src/speaches/executors/whisper_cpp.py tests/executors/test_whisper_cpp.py
git commit -m "feat: whisper_cpp executor — streaming transcription"
```

---

## Task 7: Translation handler

**Files:**
- Modify: `src/speaches/executors/whisper_cpp.py`
- Test: `tests/executors/test_whisper_cpp.py`

- [ ] **Step 1: Write failing test**

Append to `tests/executors/test_whisper_cpp.py`:

```python
def test_translation_returns_json() -> None:
    from speaches.executors.shared.handler_protocol import TranslationRequest

    audio = Audio(data=np.zeros(16000, dtype=np.float32), sample_rate=16000)
    req = TranslationRequest(
        audio=audio,
        model="ggerganov/whisper.cpp/ggml-tiny.en.bin",
        prompt=None,
        response_format="json",
        temperature=0.0,
    )

    mgr = WhisperCppModelManager(ttl=-1)
    fake_model = MagicMock()
    fake_model.transcribe.return_value = [_fake_segment(" bonjour")]
    with patch.object(WhisperCppModelManager, "_load_fn", return_value=fake_model):
        resp = mgr.handle_translation_request(req)

    assert isinstance(resp, openai.types.audio.Translation)
    assert resp.text.strip() == "bonjour"
    # must have been called with translate=True
    assert fake_model.transcribe.call_args.kwargs["translate"] is True
```

- [ ] **Step 2: Run — expect failure**

Run: `uv run pytest tests/executors/test_whisper_cpp.py::test_translation_returns_json -v`
Expected: FAIL.

- [ ] **Step 3: Implement translation**

Append to `WhisperCppModelManager`:

```python
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
```

Append helper at module bottom:

```python
def _segments_to_translation_response(
    segments: list,
    *,
    language: str,
    duration: float,
    response_format,
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
                "".join(
                    format_as_vtt(s.text, _segment_start(s), _segment_end(s), i)
                    for i, s in enumerate(segments)
                ),
                "text/vtt",
            )
        case "srt":
            return (
                "".join(
                    format_as_srt(s.text, _segment_start(s), _segment_end(s), i)
                    for i, s in enumerate(segments)
                ),
                "text/plain",
            )
```

- [ ] **Step 4: Run — expect pass**

Run: `uv run pytest tests/executors/test_whisper_cpp.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/speaches/executors/whisper_cpp.py
uv run ruff check --fix src/speaches/executors/whisper_cpp.py
git add src/speaches/executors/whisper_cpp.py tests/executors/test_whisper_cpp.py
git commit -m "feat: whisper_cpp executor — translation"
```

---

## Task 8: Wire executor selection into DI

**Files:**
- Modify: `src/speaches/dependencies.py`
- Test: new file `tests/test_dependencies_backend.py`

- [ ] **Step 1: Read current wiring**

Run: `grep -n "WhisperModelManager\|executor_registry\|ExecutorRegistry" src/speaches/dependencies.py`

Identify the block where `WhisperModelManager` is instantiated and registered. Make a mental note of surrounding structure before editing.

- [ ] **Step 2: Write failing test**

Create `tests/test_dependencies_backend.py`:

```python
import importlib

from speaches.config import Config


def _reload_deps(monkeypatch, backend: str):
    monkeypatch.setenv("WHISPER_BACKEND", backend)
    import speaches.dependencies as deps
    importlib.reload(deps)
    return deps


def test_faster_whisper_backend_registers_faster_whisper_manager(monkeypatch) -> None:
    deps = _reload_deps(monkeypatch, "faster_whisper")
    mgr = deps.get_whisper_model_manager()
    assert type(mgr).__name__ == "WhisperModelManager"


def test_whisper_cpp_backend_registers_whisper_cpp_manager(monkeypatch) -> None:
    deps = _reload_deps(monkeypatch, "whisper_cpp")
    mgr = deps.get_whisper_model_manager()
    assert type(mgr).__name__ == "WhisperCppModelManager"
```

*(Adjust the `get_whisper_model_manager` accessor name to match what actually exists — this task includes exposing a small accessor if one is not already present.)*

- [ ] **Step 3: Run — expect failure**

Run: `uv run pytest tests/test_dependencies_backend.py -v`
Expected: FAIL.

- [ ] **Step 4: Update `dependencies.py`**

At the point where `WhisperModelManager` is constructed today, replace with a factory selecting by config. Sketch (adapt to the real file shape):

```python
from speaches.config import Config
from speaches.executors.whisper import WhisperModelManager, whisper_model_registry
from speaches.executors.whisper_cpp import (
    WhisperCppModelManager,
    whisper_cpp_model_registry,
)


def _build_whisper_manager(config: Config):
    if config.whisper_backend == "whisper_cpp":
        return WhisperCppModelManager(ttl=config.stt_model_ttl)
    return WhisperModelManager(ttl=config.stt_model_ttl, whisper_config=config.whisper)


def _build_whisper_registry(config: Config):
    if config.whisper_backend == "whisper_cpp":
        return whisper_cpp_model_registry
    return whisper_model_registry
```

Wire these into wherever the executor registry is assembled, replacing the direct reference to `WhisperModelManager` / `whisper_model_registry`. Ensure only one is instantiated per process.

Expose a small accessor for testing:

```python
def get_whisper_model_manager():
    return _whisper_manager_singleton  # or however the rest of the module exposes it
```

- [ ] **Step 5: Run — expect pass**

Run: `uv run pytest tests/test_dependencies_backend.py tests/test_config.py tests/test_model_aliases.py -v`
Expected: all pass.

- [ ] **Step 6: Run full test suite — expect no regressions**

Run: `uv run pytest -x -q -m "not requires_openai and not requires_gated_hf_model"`
Expected: green (modulo prior state).

- [ ] **Step 7: Commit**

```bash
uv run ruff format src/speaches/dependencies.py tests/test_dependencies_backend.py
uv run ruff check --fix src/speaches/dependencies.py tests/test_dependencies_backend.py
git add src/speaches/dependencies.py tests/test_dependencies_backend.py
git commit -m "feat: select whisper executor by WHISPER_BACKEND"
```

---

## Task 9: Startup diagnostics (Vulkan device log)

**Files:**
- Modify: `src/speaches/main.py` (or whichever module owns the lifespan startup)
- Test: `tests/test_startup_vulkan_log.py` (create)

- [ ] **Step 1: Locate the startup hook**

Run: `grep -n "lifespan\|create_app\|startup" src/speaches/main.py`
Expected: find the lifespan / startup hook where we can run a one-shot diagnostic.

- [ ] **Step 2: Write failing test**

Create `tests/test_startup_vulkan_log.py`:

```python
import logging
from unittest.mock import patch

from speaches.config import Config
from speaches.main import log_vulkan_summary_if_enabled


def test_skips_when_backend_is_faster_whisper(caplog) -> None:
    cfg = Config(whisper_backend="faster_whisper")
    with caplog.at_level(logging.INFO):
        log_vulkan_summary_if_enabled(cfg)
    assert "vulkaninfo" not in caplog.text


def test_runs_vulkaninfo_when_backend_whisper_cpp(caplog) -> None:
    cfg = Config(whisper_backend="whisper_cpp")
    with patch("speaches.main.subprocess.run") as run, caplog.at_level(logging.INFO):
        run.return_value.stdout = "Vulkan Instance Version: 1.3.0\n"
        run.return_value.returncode = 0
        log_vulkan_summary_if_enabled(cfg)
    run.assert_called_once()
    assert "Vulkan Instance Version" in caplog.text
```

- [ ] **Step 3: Run — expect failure**

Run: `uv run pytest tests/test_startup_vulkan_log.py -v`
Expected: FAIL (function does not exist).

- [ ] **Step 4: Implement**

Add to `src/speaches/main.py`:

```python
import subprocess

from speaches.config import Config


def log_vulkan_summary_if_enabled(config: Config) -> None:
    if config.whisper_backend != "whisper_cpp":
        return
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        logger.warning("vulkaninfo not found; cannot report Vulkan device info")
        return
    if result.returncode != 0:
        logger.warning(f"vulkaninfo exited with status {result.returncode}")
        return
    logger.info("Vulkan summary:\n%s", result.stdout)
```

Call it from the FastAPI lifespan / `create_app` startup path with the current `Config` instance.

- [ ] **Step 5: Run — expect pass**

Run: `uv run pytest tests/test_startup_vulkan_log.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
uv run ruff format src/speaches/main.py tests/test_startup_vulkan_log.py
uv run ruff check --fix src/speaches/main.py tests/test_startup_vulkan_log.py
git add src/speaches/main.py tests/test_startup_vulkan_log.py
git commit -m "feat: log vulkaninfo summary at startup when backend=whisper_cpp"
```

---

## Task 10: `Dockerfile.vulkan`

**Files:**
- Create: `Dockerfile.vulkan`

- [ ] **Step 1: Author the Dockerfile**

Create `Dockerfile.vulkan` with:

```dockerfile
ARG BASE_IMAGE=ubuntu:24.04
# hadolint ignore=DL3006
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/speaches-ai/speaches"
LABEL org.opencontainers.image.licenses="MIT"

# hadolint ignore=DL3008
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates curl ffmpeg \
        mesa-vulkan-drivers libvulkan1 vulkan-tools \
        libshaderc-dev glslc \
        build-essential cmake git pkg-config

RUN useradd --create-home --shell /bin/bash --uid 1000 ubuntu || true
USER ubuntu
ENV HOME=/home/ubuntu \
    PATH=/home/ubuntu/.local/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/home/ubuntu/.cache/uv \
    UV_PYTHON_CACHE_DIR=/home/ubuntu/.cache/uv/python
WORKDIR $HOME/speaches

COPY --chown=ubuntu --from=ghcr.io/astral-sh/uv:0.10 /uv /bin/uv

# Source-build pywhispercpp with Vulkan backend
ENV CMAKE_ARGS="-DGGML_VULKAN=1"
ENV FORCE_CMAKE=1

RUN --mount=type=cache,target=/home/ubuntu/.cache/uv,uid=1000,gid=1000 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --compile-bytecode --no-install-project --no-dev --extra vulkan

COPY --chown=ubuntu . .
RUN --mount=type=cache,target=/home/ubuntu/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --compile-bytecode --no-dev --extra vulkan

# Build-time smoke test
RUN .venv/bin/python -c "from pywhispercpp.model import Model; print('pywhispercpp import OK')"

RUN mkdir -p $HOME/.cache/huggingface/hub

ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
ENV PATH="$HOME/speaches/.venv/bin:$PATH"
ENV DO_NOT_TRACK=1
ENV GRADIO_ANALYTICS_ENABLED="False"
ENV DISABLE_TELEMETRY=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV PYANNOTE_METRICS_ENABLED=0
ENV WHISPER_BACKEND=whisper_cpp
EXPOSE 8000
CMD ["uvicorn", "--factory", "speaches.main:create_app"]
```

- [ ] **Step 2: Local build**

Run: `docker build -f Dockerfile.vulkan -t speaches:vulkan-local .`
Expected: build succeeds, smoke-test `print('pywhispercpp import OK')` line shown.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile.vulkan
git commit -m "feat: Dockerfile.vulkan — Vulkan-capable image with pywhispercpp"
```

---

## Task 11: `compose.vulkan.yaml`

**Files:**
- Create: `compose.vulkan.yaml`

- [ ] **Step 1: Author the compose file**

Create `compose.vulkan.yaml`:

```yaml
# include:
#   - compose.observability.yaml
services:
  speaches:
    extends:
      file: compose.yaml
      service: speaches
    image: ghcr.io/speaches-ai/speaches:latest-vulkan
    build:
      context: .
      dockerfile: Dockerfile.vulkan
    environment:
      - WHISPER_BACKEND=whisper_cpp
    devices:
      - /dev/dri:/dev/dri
    group_add:
      - video
      - render
    volumes:
      - hf-hub-cache:/home/ubuntu/.cache/huggingface/hub
volumes:
  hf-hub-cache:
```

- [ ] **Step 2: Lint compose**

Run: `docker compose -f compose.vulkan.yaml config > /dev/null`
Expected: no errors; prints merged config when `> /dev/null` removed.

- [ ] **Step 3: Commit**

```bash
git add compose.vulkan.yaml
git commit -m "feat: compose.vulkan.yaml flavor"
```

---

## Task 12: `requires_vulkan` pytest marker

**Files:**
- Modify: `pyproject.toml` (add marker registration)
- Create or modify: `conftest.py` at repo root
- Create: `tests/executors/test_whisper_cpp_vulkan.py`

- [ ] **Step 1: Register the marker**

In `pyproject.toml` under `[tool.pytest.ini_options].markers`, add:

```toml
  "requires_vulkan",
```

- [ ] **Step 2: Add conftest gate**

At repo root, create or append to `conftest.py`:

```python
import os

import pytest


def pytest_collection_modifyitems(config, items) -> None:  # noqa: ARG001
    if os.environ.get("SPEACHES_HAS_VULKAN") == "1":
        return
    skip = pytest.mark.skip(reason="requires Vulkan device; set SPEACHES_HAS_VULKAN=1 to run")
    for item in items:
        if "requires_vulkan" in item.keywords:
            item.add_marker(skip)
```

- [ ] **Step 3: Add a Vulkan integration test**

Create `tests/executors/test_whisper_cpp_vulkan.py`:

```python
import numpy as np
import pytest

from speaches.audio import Audio
from speaches.executors.shared.handler_protocol import TranscriptionRequest
from speaches.executors.silero_vad_v5 import VadOptions
from speaches.executors.whisper_cpp import WhisperCppModelManager


@pytest.mark.requires_vulkan
def test_tiny_model_transcribes_sample_audio() -> None:
    mgr = WhisperCppModelManager(ttl=0)
    audio_path = "audio.wav"  # sample shipped in repo root
    import soundfile as sf

    data, sr = sf.read(audio_path, dtype="float32")
    if sr != 16000:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    audio = Audio(data=data, sample_rate=16000)
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
```

- [ ] **Step 4: Verify gating**

Run: `uv run pytest tests/executors/test_whisper_cpp_vulkan.py -v`
Expected: skipped (on a non-Vulkan host).

Run: `SPEACHES_HAS_VULKAN=1 uv run pytest tests/executors/test_whisper_cpp_vulkan.py --collect-only`
Expected: shows the test as collected (it will still fail to actually execute on a host without Vulkan, which is fine — the point is collection gating).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml conftest.py tests/executors/test_whisper_cpp_vulkan.py
git commit -m "test: add requires_vulkan marker and integration test"
```

---

## Task 13: `/v1/models` listing filtered to active backend

**Files:**
- Modify: `src/speaches/dependencies.py` (or wherever the executor registry lists models)
- Verify: `src/speaches/routers/models.py`

- [ ] **Step 1: Inspect current listing**

Run: `grep -n "list_local_models\|list_remote_models\|/v1/models" -r src/speaches/routers src/speaches/dependencies.py`
Expected: locate the code that enumerates available executors' model registries.

- [ ] **Step 2: Confirm single-executor registration already filters**

If Task 8 correctly registers only one of `WhisperModelManager` / `WhisperCppModelManager`, the `/v1/models` output for Whisper will naturally show only backend-appropriate repos — no extra filter needed.

- [ ] **Step 3: Add a regression test**

Create `tests/routers/test_models_listing.py`:

```python
import importlib


def test_faster_whisper_backend_lists_ct2_repos(monkeypatch) -> None:
    monkeypatch.setenv("WHISPER_BACKEND", "faster_whisper")
    import speaches.dependencies as deps
    importlib.reload(deps)
    registry = deps._build_whisper_registry(deps.Config())
    # Ask for one cached model id shape. On a cold machine we may have none;
    # the important thing is that the registry's library_name is ctranslate2.
    assert registry.hf_model_filter.library_name == "ctranslate2"


def test_whisper_cpp_backend_lists_ggml_repos(monkeypatch) -> None:
    monkeypatch.setenv("WHISPER_BACKEND", "whisper_cpp")
    import speaches.dependencies as deps
    importlib.reload(deps)
    registry = deps._build_whisper_registry(deps.Config())
    assert registry.hf_model_filter.library_name == "whisper.cpp"
```

- [ ] **Step 4: Run the test — expect pass**

Run: `uv run pytest tests/routers/test_models_listing.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/routers/test_models_listing.py
git commit -m "test: /v1/models listing filters by active whisper backend"
```

---

## Task 14: Documentation

**Files:**
- Create: `docs/installation/vulkan.md`
- Create: `docs/configuration/whisper-backend.md`
- Modify: `mkdocs.yml`

- [ ] **Step 1: Installation doc**

Create `docs/installation/vulkan.md`:

```markdown
# Vulkan (non-NVIDIA GPUs)

The Vulkan image uses `whisper.cpp` via `pywhispercpp` for speech-to-text,
enabling GPU acceleration on AMD, Intel Arc / iGPU, and other Vulkan-capable
hardware. Text-to-speech continues to run on CPU.

## Requirements

- A Vulkan 1.2+ capable GPU with a working ICD loader on the host.
- Verify with `vulkaninfo --summary` on the host.
- Docker with `/dev/dri` device access.

## Run

```bash
docker compose -f compose.vulkan.yaml up
```

## Configuration

Set `WHISPER_BACKEND=whisper_cpp`. Models are pulled from
`ggerganov/whisper.cpp` as GGML `.bin` files. Aliases map `whisper-1` to
`ggerganov/whisper.cpp/ggml-large-v3.bin` on this backend.

## Troubleshooting

- `vulkaninfo` logged at startup lists detected devices. Empty output means
  the container does not see a Vulkan device — check `/dev/dri` mount and
  `group_add` (`video`, `render`).
- CPU fallback: `pywhispercpp` built with `-DGGML_VULKAN=1` still runs on
  CPU when no Vulkan device is present, but slowly.
```

- [ ] **Step 2: Configuration doc**

Create `docs/configuration/whisper-backend.md`:

```markdown
# Whisper backend selection

`WHISPER_BACKEND` controls which Whisper implementation serves STT:

| Value             | Library            | Devices              | Model format     |
|-------------------|--------------------|----------------------|------------------|
| `faster_whisper`  | faster-whisper     | CPU, CUDA            | CTranslate2      |
| `whisper_cpp`     | pywhispercpp       | CPU, CUDA, Vulkan    | GGML `.bin`      |

Default: `faster_whisper`. Changing backends requires restarting the server.

## Parameter support

| Param                  | `faster_whisper` | `whisper_cpp`          |
|------------------------|------------------|-------------------------|
| `language`             | yes              | yes                     |
| `prompt`               | yes              | yes                     |
| `temperature`          | yes              | yes                     |
| `word_timestamps`      | yes              | yes (token-level)       |
| `hotwords`             | yes              | ignored (logged)        |
| `vad_filter`           | yes (internal)   | external Silero VAD     |
| `beam_size`            | yes              | ignored (logged)        |
| `response_format=diarized_json` | not supported | not supported |
```

- [ ] **Step 3: Update `mkdocs.yml` nav**

In `mkdocs.yml`, add entries under the existing `Installation` and
`Configuration` sections pointing at the new docs.

- [ ] **Step 4: Build docs locally**

Run: `uv run mkdocs build --strict`
Expected: builds without warnings.

- [ ] **Step 5: Commit**

```bash
git add docs/installation/vulkan.md docs/configuration/whisper-backend.md mkdocs.yml
git commit -m "docs: vulkan backend install and configuration"
```

---

## Task 15: CI build job for Vulkan image

**Files:**
- Modify: `.github/workflows/*.yml` (whichever handles image builds)

- [ ] **Step 1: Locate the existing CUDA/CPU image build workflow**

Run: `grep -rn "compose.cuda.yaml\|latest-cuda\|docker build" .github/workflows/`

- [ ] **Step 2: Add a Vulkan matrix entry or parallel job**

Mirror the CUDA job, swapping:

- `-f compose.cuda.yaml` → `-f compose.vulkan.yaml`
- tag `latest-cuda-12.6.3` → `latest-vulkan`
- no `buildx` CUDA-specific args

Example addition (adapt to the repo's workflow style):

```yaml
  build-vulkan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: Build Vulkan image
        run: docker build -f Dockerfile.vulkan -t speaches:vulkan-ci .
```

No run step — CI has no Vulkan GPU. The Vulkan-marked integration tests are
executed manually on the SSH-accessible Vulkan host.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/
git commit -m "ci: add vulkan image build job"
```

---

## Task 16: End-to-end verification on Vulkan host

This task runs on the SSH-accessible Vulkan machine. Not automated.

- [ ] **Step 1: Rsync / checkout the branch on the host**

- [ ] **Step 2: Build and run**

```bash
docker compose -f compose.vulkan.yaml up --build
```

Expected: startup log shows `Vulkan summary` with the device.

- [ ] **Step 3: Smoke transcription**

From the host:

```bash
curl -F "file=@audio.wav" -F "model=whisper-1" http://localhost:8000/v1/audio/transcriptions
```

Expected: non-empty `text` field.

- [ ] **Step 4: Run Vulkan-marked tests in-container**

```bash
docker compose -f compose.vulkan.yaml exec speaches \
  env SPEACHES_HAS_VULKAN=1 uv run pytest tests/executors/test_whisper_cpp_vulkan.py -v
```

Expected: pass.

- [ ] **Step 5: Smoke TTS (confirm CPU TTS still works in same install)**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "content-type: application/json" \
  -d '{"model":"tts-1","voice":"af_sky","input":"hello"}' --output out.wav
```

Expected: valid WAV file written.

- [ ] **Step 6: Realtime API smoke**

Connect the `realtime-console/` UI to the deployed server and verify a short
spoken round-trip.

- [ ] **Step 7: Capture results in the PR description**

Paste the `vulkaninfo` summary, transcription sample, and timing numbers into
the PR.

---

## Task 17: Open the PR

- [ ] **Step 1: Rebase / push**

```bash
git fetch origin master
git rebase origin/master
git push -u origin feat/vulkan-backend
```

- [ ] **Step 2: Open PR**

Title: `feat: Vulkan STT backend via whisper.cpp`

Body: summary, link to spec (`docs/superpowers/specs/2026-04-24-vulkan-backend-design.md`), results from Task 16, list of follow-ups (TTS Vulkan, Parakeet, auto-conversion — explicitly deferred).
