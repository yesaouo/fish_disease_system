from __future__ import annotations

import asyncio
import json
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


def _timestamp() -> str:
    # Use UTC to make backups predictable across hosts
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _last_activity_mtime(settings: Settings) -> float:
    log_path = settings.audit_log_path
    if log_path.exists():
        try:
            return log_path.stat().st_mtime
        except Exception:
            return 0.0
    # If there is no audit log yet, treat as recent activity to avoid eager backups
    return float(datetime.now(timezone.utc).timestamp())


def _marker_path(settings: Settings) -> Path:
    return backup_root(settings) / ".last_activity_mtime"


def _read_last_marker(settings: Settings) -> float:
    path = _marker_path(settings)
    if not path.exists():
        return -1.0
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except Exception:
        return -1.0


def _write_last_marker(value: float, settings: Settings) -> None:
    mrk = _marker_path(settings)
    mrk.parent.mkdir(parents=True, exist_ok=True)
    tmp = mrk.with_suffix(".tmp")
    tmp.write_text(f"{value}\n", encoding="utf-8")
    os.replace(tmp, mrk)


def _write_meta_file(dest_dir: Path, settings: Settings, activity_mtime: float) -> None:
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "activity_log_mtime": activity_mtime,
        "data_root": str(settings.data_root),
    }
    (dest_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def perform_backup(settings: Optional[Settings] = None) -> Path:
    settings = _ensure_settings(settings)
    root = backup_root(settings)
    root.mkdir(parents=True, exist_ok=True)

    ts = _timestamp()
    temp_dir = root / f"{ts}.tmp"
    final_dir = root / ts
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Copy annotations from each dataset
    datasets = datasets_service.list_datasets(settings)
    for ds in datasets:
        ds_dir = datasets_service.resolve_dataset_path(ds, settings)
        src_json_dir = storage_service.get_json_dir(ds_dir)
        if not src_json_dir.exists():
            continue
        # Create destination annotations dir
        dst_ds_dir = temp_dir / ds / "annotations"
        dst_ds_dir.mkdir(parents=True, exist_ok=True)
        # Copy .json files only
        for src_path in sorted(src_json_dir.glob("*.json")):
            shutil.copy2(src_path, dst_ds_dir / src_path.name)

    # Write meta after copies complete
    _write_meta_file(temp_dir, settings, _last_activity_mtime(settings))

    # Atomically finalize backup folder
    os.replace(temp_dir, final_dir)
    return final_dir


@dataclass
class IdleBackupState:
    last_backed_activity_mtime: float


async def idle_backup_worker(settings: Optional[Settings] = None) -> None:
    """Background loop that snapshots annotations when idle duration elapses.

    Criteria: time since audit log mtime exceeds settings.idle_backup_seconds.
    Ensures at most one backup per unchanged activity mtime by recording a marker file.
    """
    settings = _ensure_settings(settings)
    if not settings.idle_backup_enabled:
        return

    state = IdleBackupState(last_backed_activity_mtime=_read_last_marker(settings))

    while True:
        try:
            act_mtime = _last_activity_mtime(settings)
            now_ts = datetime.now(timezone.utc).timestamp()
            idle_seconds = now_ts - act_mtime

            if (
                idle_seconds >= settings.idle_backup_seconds
                and act_mtime > 0
                and act_mtime != state.last_backed_activity_mtime
            ):
                # Perform backup
                perform_backup(settings)
                state.last_backed_activity_mtime = act_mtime
                _write_last_marker(act_mtime, settings)
        except Exception:
            # Swallow exceptions to keep the loop alive
            pass

        await asyncio.sleep(max(5, int(settings.idle_check_interval_seconds)))

