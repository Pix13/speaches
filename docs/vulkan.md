# Vulkan (non-NVIDIA GPUs)

The Vulkan image accelerates speech-to-text on AMD, Intel Arc / iGPU, and
other Vulkan-capable hardware by running Whisper through
[`whisper.cpp`](https://github.com/ggerganov/whisper.cpp) via the
[`pywhispercpp`](https://github.com/absadiki/pywhispercpp) bindings.
Text-to-speech continues to run on CPU, so a single installation serves
both.

## Requirements

- A Vulkan 1.2+ capable GPU with a working ICD loader on the host.
- Verify on the host: `vulkaninfo --summary`.
- Docker with `/dev/dri` device access.

## Run

```bash
docker compose -f compose.vulkan.yaml up
```

At startup the container logs `vulkaninfo --summary` so you can confirm the
device is visible inside the container.

## Configuration

Set `WHISPER_BACKEND=whisper_cpp` (the Vulkan compose file does this by
default). Models are pulled from `ggerganov/whisper.cpp` as GGML `.bin`
files. The alias `whisper-1` resolves to
`ggerganov/whisper.cpp/ggml-large-v3.bin` on this backend, and to
`Systran/faster-whisper-large-v3` on the default `faster_whisper` backend.

## Backend selection

`WHISPER_BACKEND` controls which Whisper implementation serves STT.

| Value             | Library            | Devices              | Model format     |
|-------------------|--------------------|----------------------|------------------|
| `faster_whisper`  | faster-whisper     | CPU, CUDA            | CTranslate2      |
| `whisper_cpp`     | pywhispercpp       | CPU, CUDA, Vulkan    | GGML `.bin`      |

Default: `faster_whisper`. Changing backends requires restarting the server.

## Parameter support

| Param                              | `faster_whisper` | `whisper_cpp`          |
|------------------------------------|------------------|-------------------------|
| `language`                         | yes              | yes                     |
| `prompt`                           | yes              | yes                     |
| `temperature`                      | yes              | yes                     |
| `word_timestamps`                  | yes              | yes (token-level)       |
| `hotwords`                         | yes              | ignored                 |
| `vad_filter`                       | yes (internal)   | external Silero VAD     |
| `beam_size`                        | yes              | ignored                 |
| `response_format=diarized_json`    | not supported    | not supported           |

## Troubleshooting

- Empty `vulkaninfo` output in startup logs means the container cannot see
  a Vulkan device. Check `/dev/dri` is mounted and `group_add` includes
  `video` and `render` (already configured in `compose.vulkan.yaml`).
- `pywhispercpp` built with `-DGGML_VULKAN=1` still runs on CPU when no
  Vulkan device is present, but slowly.
