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
            raise ValueError(f"whisper_cpp model id must be '<repo>/<file>.bin', got {model_id!r}")
        return parts[0], parts[1]


whisper_cpp_model_registry = WhisperCppModelRegistry(hf_model_filter=hf_model_filter)
