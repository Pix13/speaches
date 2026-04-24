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
