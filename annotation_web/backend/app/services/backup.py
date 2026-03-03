from __future__ import annotations

import asyncio
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

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


def _ymd_for_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d")


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


def _db_activity_mtime(dataset_dir: Path, settings: Settings) -> float | None:
    db_path = storage_service.get_db_path(dataset_dir, settings)
    candidates = [db_path, db_path.parent / f"{db_path.name}-wal"]
    mtimes: list[float] = []

    for path in candidates:
        try:
            mtimes.append(path.stat().st_mtime)
        except FileNotFoundError:
            continue
        except Exception:
            continue

    return max(mtimes) if mtimes else None


def datasets_changed_today(settings: Optional[Settings] = None) -> list[str]:
    settings = _ensure_settings(settings)
    today = _ymd()
    changed: list[str] = []

    for ds in datasets_service.list_datasets(settings):
        ds_dir = datasets_service.resolve_dataset_path(ds, settings)
        mtime = _db_activity_mtime(ds_dir, settings)
        if mtime is None:
            continue
        if _ymd_for_timestamp(mtime) == today:
            changed.append(ds)

    return changed


def _copy_sqlite_backup(src_db: Path, dst_db: Path) -> None:
    tmp_dst = dst_db.with_suffix(f"{dst_db.suffix}.tmp")
    if tmp_dst.exists():
        tmp_dst.unlink()

    src_conn = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True, timeout=30.0)
    try:
        dst_conn = sqlite3.connect(str(tmp_dst), timeout=30.0)
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
        os.replace(tmp_dst, dst_db)
    finally:
        src_conn.close()
        if tmp_dst.exists():
            tmp_dst.unlink(missing_ok=True)


def perform_backup(
    settings: Optional[Settings] = None,
    datasets: Optional[Iterable[str]] = None,
) -> list[Path]:
    settings = _ensure_settings(settings)
    root = backup_root(settings)
    root.mkdir(parents=True, exist_ok=True)

    ts = _ymd()
    outputs: list[Path] = []

    target_datasets = list(datasets) if datasets is not None else datasets_service.list_datasets(settings)
    for ds in target_datasets:
        ds_dir = datasets_service.resolve_dataset_path(ds, settings)
        src_db = storage_service.get_db_path(ds_dir, settings)
        if not src_db.exists():
            continue
        dst = root / f"{ds}_{ts}.db"
        _copy_sqlite_backup(src_db, dst)
        outputs.append(dst)

    return outputs


@dataclass
class DailyBackupState:
    last_backup_ymd: str


async def daily_backup_worker(settings: Optional[Settings] = None) -> None:
    """Background loop that snapshots changed dataset DBs at most once per day.

    Writes backups to: `data_root/backup/{dataset}_{YYYYMMDD}.db`.
    """
    settings = _ensure_settings(settings)
    if not settings.daily_backup_enabled:
        return

    state = DailyBackupState(last_backup_ymd=_read_last_marker(settings))

    while True:
        try:
            today = _ymd()
            if today and today != state.last_backup_ymd:
                changed = datasets_changed_today(settings)
                if changed:
                    perform_backup(settings, changed)
                    state.last_backup_ymd = today
                    _write_last_marker(today, settings)
        except Exception:
            # Swallow exceptions to keep the loop alive
            pass

        await asyncio.sleep(max(5, int(settings.daily_check_interval_seconds)))


# Backward-compatible import alias for older internal references.
idle_backup_worker = daily_backup_worker
