from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Set

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_root: Path = Field(
        # Default stays at repo /data; override via DATA_ROOT in backend/.env or env vars.
        default_factory=lambda: (Path(__file__).resolve().parents[2] / "data")
    )
    # Per-dataset SQLite filename under each dataset directory.
    db_filename: str = Field(default="annotations.db")
    # If deploying under a subpath (e.g., "/fish"), set this to that path.
    # FastAPI will serve all routes under this root and generate correct URLs.
    root_path: str = Field(default="")
    image_exts: List[str] = Field(default_factory=lambda: ["jpg", "jpeg", "png"])
    stats_cache_seconds: int = Field(default=60, ge=0)
    dataset_cache_seconds: int = Field(default=60, ge=0)
    classes_cache_seconds: int = Field(default=60, ge=0)
    audit_log_filename: str = Field(default="audit_log.jsonl")
    # Idle-backup settings
    idle_backup_enabled: bool = Field(default=True)
    idle_backup_seconds: int = Field(default=21600, ge=60)  # 6 hours
    idle_check_interval_seconds: int = Field(default=60, ge=5)
    backup_dirname: str = Field(default="backup")
    # Auth keys file (one key per line) under data_root by default
    auth_keys_filename: str = Field(default="auth_keys.txt")

    model_config = SettingsConfigDict(
        env_prefix="",
        # Load environment variables from backend/.env by default,
        # or use ENV_FILE to point to a custom path.
        env_file=os.getenv("ENV_FILE") or str(Path(__file__).resolve().parents[1] / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("data_root", mode="before")
    def _expand_data_root(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    @property
    def image_extensions(self) -> Set[str]:
        return {f".{ext.lower().lstrip('.')}" for ext in self.image_exts}

    @property
    def audit_log_path(self) -> Path:
        return self.data_root / self.audit_log_filename

    @property
    def auth_keys_path(self) -> Path:
        return self.data_root / self.auth_keys_filename


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
