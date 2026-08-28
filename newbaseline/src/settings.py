"""Central, typed access to versioned configuration and local secrets."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = WORKSPACE_ROOT / "newbaseline" / "config.toml"
ENV_PATH = WORKSPACE_ROOT / "newbaseline" / ".env"


@dataclass(frozen=True)
class Settings:
    """Non-secret project settings, with paths derived from one workspace root."""

    values: dict[str, Any]

    @property
    def workspace_root(self) -> Path:
        return WORKSPACE_ROOT

    @property
    def dataset_dir(self) -> Path:
        return WORKSPACE_ROOT / self.get("paths", "dataset_dir")

    def path(self, key: str) -> Path:
        """Resolve a workspace-relative path declared in ``[paths]``."""
        value = Path(self.get("paths", key))
        return value if value.is_absolute() else WORKSPACE_ROOT / value

    @property
    def release(self) -> str:
        return self.get("corpus", "release")

    @property
    def release_dir(self) -> Path:
        return self.path("raw_corpus_dir")

    @property
    def metadata_dir(self) -> Path:
        return self.path("metadata_dir")

    @property
    def chunk_dir(self) -> Path:
        return self.path("chunk_dir")

    @property
    def reference_dir(self) -> Path:
        return self.path("reference_dir")

    @property
    def embedding_dir(self) -> Path:
        return self.path("embedding_dir")

    def embedding_root(self, selection_id: str) -> Path:
        return self.embedding_dir / selection_id

    def get(self, section: str, key: str) -> Any:
        try:
            return self.values[section][key]
        except KeyError as error:
            raise KeyError(f"Missing [{section}].{key} in {CONFIG_PATH}") from error


def load_settings() -> Settings:
    """Load versioned configuration and make `.env` secrets available to callers."""
    with CONFIG_PATH.open("rb") as config_file:
        values = tomllib.load(config_file)
    load_dotenv(ENV_PATH, override=False)
    return Settings(values=values)


def require_secret(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    raise RuntimeError(f"Set {name} in {ENV_PATH} before running this command.")
