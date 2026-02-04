from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..config import Settings, get_settings
from . import datasets as datasets_service
from . import storage as storage_service

_settings = get_settings()


def _ensure_settings(settings: Optional[Settings]) -> Settings:
    return settings or _settings


def backup_root(settings: Optional[Settings] = None) -> Path:
    settings = _ensure_settings(settings)
    return settings.data_root / settings.backup_dirname


def _ymd() -> str:
    # Use UTC to keep behavior stable across hosts.
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _marker_path(settings: Settings) -> Path:
    return backup_root(settings) / ".last_backup_ymd"


def _read_last_marker(settings: Settings) -> str:
    path = _marker_path(settings)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _write_last_marker(value: str, settings: Settings) -> None:
    mrk = _marker_path(settings)
    mrk.parent.mkdir(parents=True, exist_ok=True)
    tmp = mrk.with_suffix(".tmp")
    tmp.write_text(f"{value}\n", encoding="utf-8")
    os.replace(tmp, mrk)


def perform_backup(settings: Optional[Settings] = None) -> list[Path]:
    settings = _ensure_settings(settings)
    root = backup_root(settings)
    root.mkdir(parents=True, exist_ok=True)

    ts = _ymd()
    outputs: list[Path] = []

    datasets = datasets_service.list_datasets(settings)
    for ds in datasets:
        ds_dir = datasets_service.resolve_dataset_path(ds, settings)
        src_db = storage_service.get_db_path(ds_dir, settings)
        if not src_db.exists():
            continue
        dst = root / f"{ds}_{ts}.db"
        shutil.copy2(src_db, dst)
        outputs.append(dst)

    return outputs


@dataclass
class IdleBackupState:
    last_backup_ymd: str


async def idle_backup_worker(settings: Optional[Settings] = None) -> None:
    """Background loop that snapshots each dataset DB once per day.

    Writes backups to: `data_root/backup/{dataset}_{YYYYMMDD}.db`.
    """
    settings = _ensure_settings(settings)
    if not settings.idle_backup_enabled:
        return

    state = IdleBackupState(last_backup_ymd=_read_last_marker(settings))

    while True:
        try:
            today = _ymd()
            if today and today != state.last_backup_ymd:
                perform_backup(settings)
                state.last_backup_ymd = today
                _write_last_marker(today, settings)
        except Exception:
            # Swallow exceptions to keep the loop alive
            pass

        await asyncio.sleep(max(5, int(settings.idle_check_interval_seconds)))
