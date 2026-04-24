# Vulkan Backend Support — Design

**Status:** Draft
**Date:** 2026-04-24
**Branch:** `feat/vulkan-backend`

## Goal

Add Vulkan GPU acceleration for Whisper-based speech-to-text so that `speaches`
runs on non-NVIDIA hardware (AMD, Intel Arc / iGPU, etc.) from a single install.
TTS continues to run on CPU; a single deployment covers STT-on-Vulkan plus
CPU-backed TTS.

## Non-goals

- Vulkan acceleration for TTS (Kokoro / Piper remain on CPU / ONNX Runtime).
- Vulkan acceleration for Parakeet, Silero VAD, or pyannote diarization.
- Auto-conversion of CTranslate2 models to GGML.
- A combined CUDA + Vulkan image.
- Full parameter parity with `faster-whisper` (see "Parity" below).

## Constraints

- Must not change behavior for existing CPU / CUDA users. `faster-whisper`
  remains the default backend.
- Single installation / image serves STT-on-Vulkan and CPU TTS concurrently.
- Must fit the existing executor + model-registry + compose-per-flavor
  patterns.
- Realtime API and SSE streaming endpoints must keep working on the Vulkan
  backend.

## Architecture

Introduce a pluggable Whisper backend layer. The router and model-manager code
becomes backend-agnostic:

```
routers/*  ->  model_manager  ->  WhisperBackend (Protocol)
                                  |-- FasterWhisperBackend  (current code, refactored)
                                  \-- WhisperCppBackend     (new, via pywhispercpp)
```

Selection is a new `pydantic_settings` field
`whisper_backend: Literal["faster_whisper", "whisper_cpp"]`, default
`faster_whisper`. The factory lazy-imports only the selected backend so CPU /
CUDA images never load `pywhispercpp` and the Vulkan image never loads
CTranslate2.

### Module layout

```
src/speaches/executors/whisper/
├── __init__.py            # factory: get_backend(config) -> WhisperBackend
├── backend.py             # Protocol + shared Segment / Word / params types
├── faster_whisper.py      # today's executors/whisper.py, refactored
└── whisper_cpp.py         # new
```

Routers stop importing `faster_whisper` types directly and use the shared
types from `backend.py`. This is the one refactor-in-flight justified by the
feature work — without it the abstraction leaks.

### `WhisperBackend` Protocol

Surface scaled to what routers actually call today:

- `transcribe(audio, *, language, task, vad_filter, beam_size, temperature,
  initial_prompt, word_timestamps, ...) -> Iterator[Segment]`
- `detect_language(audio) -> tuple[str, float]`
- `load()` / `unload()` — participates in the existing dynamic load/offload
  lifecycle.
- `info` property exposing model id, backend name, device.

Shared `Segment` and `Word` types are Pydantic models, replacing the current
direct use of `faster_whisper.transcribe.Segment` in the streaming layer.

## Model resolution

Extend the existing registry rather than fork it.

- `model_aliases.json` schema bumped from flat `alias -> repo` to a per-backend
  map. A plain string value is preserved as back-compat shorthand meaning
  "same repo for all backends" (which in practice continues to mean
  faster-whisper).

  ```json
  {
    "whisper-large-v3": {
      "faster_whisper": "Systran/faster-whisper-large-v3",
      "whisper_cpp": "ggerganov/whisper.cpp/ggml-large-v3.bin"
    }
  }
  ```

- `model_registry.resolve(model_id, backend)` returns the backend-appropriate
  `(repo_id, filename | None)`. For whisper.cpp the "model" is a single
  `.bin` file inside `ggerganov/whisper.cpp`; `hf_utils.hf_hub_download`
  already supports the file-level API.
- User-supplied repo IDs forward to the active backend unchanged. No silent
  cross-backend translation.
- HF cache volume (`hf-hub-cache`) is shared — CT2 and GGML artifacts live
  under different repos and coexist.
- `/v1/models` listing filters by active backend.

## Streaming and realtime integration

- Segment streaming: `WhisperCppBackend` wraps `pywhispercpp`'s
  `new_segment_callback` into the iterator contract used by the SSE layer.
  SSE code is untouched thanks to the shared `Segment` type.
- Realtime / chunked input: the realtime path feeds audio in chunks. The
  whisper.cpp adapter uses the existing Silero VAD executor
  (`executors/silero_vad_v5.py`) as the gating / windowing mechanism
  regardless of the `vad_filter` param. faster-whisper's internal VAD is
  not reused.
- Word timestamps: mapped from `token_timestamps=True`. Accuracy differs
  from faster-whisper; acceptable under functional parity.
- Language detection: exposed via whisper.cpp's `lang_auto_detect`.

### Parity

Functional parity, not full parity. Params without a whisper.cpp equivalent
are accepted and logged at `debug` as ignored. A documentation page lists
which params are honored / mapped / ignored per backend.

## Deployment

### Files

- `compose.vulkan.yaml` — mirrors `compose.cuda.yaml`. Sets
  `WHISPER_BACKEND=whisper_cpp`, mounts `/dev/dri` (and `/dev/kfd` where
  needed) into the container, references the Vulkan image.
- `Dockerfile.vulkan` — separate Dockerfile. Keeps the CUDA Dockerfile clean
  and makes the Vulkan-specific build steps explicit.

### Vulkan image

- Base: `ubuntu:24.04`.
- System packages: `mesa-vulkan-drivers`, `libvulkan1`, `vulkan-tools`,
  `libshaderc-dev`, `glslc`, `cmake`, `g++`.
- `pywhispercpp` installed via `uv pip install` under
  `CMAKE_ARGS="-DGGML_VULKAN=1"` and `FORCE_CMAKE=1`.
- Build-time smoke check: `python -c "from pywhispercpp.model import Model"`.
- Runtime: log `vulkaninfo --summary` once at startup under a debug flag.

### `pyproject.toml`

Add a `vulkan` optional-dependency group so `pywhispercpp` is only pulled in
the Vulkan image:

```toml
[project.optional-dependencies]
vulkan = ["pywhispercpp>=1.3.0"]
```

### Configuration

- New setting: `WHISPER_BACKEND` (`faster_whisper` | `whisper_cpp`), default
  `faster_whisper`.
- Documented in `docs/configuration/`.

## Testing

- `WhisperCppBackend` exercised on CPU in CI — whisper.cpp builds and runs on
  CPU, so the adapter is covered without a GPU.
- Vulkan-only tests gated by a new `requires_vulkan` pytest marker, following
  the existing `requires_openai` / `requires_gated_hf_model` pattern.
  Run manually on the SSH-accessible Vulkan host.
- Vulkan image gets a CI build job (build only, no run).
- Existing realtime conversation tests run against both backends where
  possible.

## Documentation

- `docs/installation/vulkan.md` — device requirements, Docker device flags,
  configuration.
- `docs/configuration/` updated for `WHISPER_BACKEND`.
- Per-backend parameter support matrix.

## Risks

- **pywhispercpp Vulkan build fragility** — requires source build with
  specific CMake flags. Mitigated by locking versions and a build-time import
  check.
- **Realtime windowing behavior drift** — moving from faster-whisper's
  internal VAD to Silero-gated windows changes segmentation timing. Covered
  by the shared realtime test suite running against both backends.
- **Model alias schema migration** — existing consumers of
  `model_aliases.json` need to handle both shapes. Addressed by the
  back-compat shorthand.
