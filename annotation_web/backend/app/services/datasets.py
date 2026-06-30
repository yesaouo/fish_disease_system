from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from fastapi import HTTPException, status

from ..config import Settings, get_settings
from ..utils.cache import TTLCache

_DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")

# Marker file written into datasets created through the diagnosis writeback flow.
# The original (training) datasets have no marker → treated as locked / read-only
# targets. Only marked datasets can receive imported diagnosis cases.
DATASET_META_FILENAME = "meta.json"

_settings = get_settings()
_datasets_cache = TTLCache[str, List[str]](_settings.dataset_cache_seconds)
_classes_cache = TTLCache[str, Tuple[str, float, List[str]]](_settings.classes_cache_seconds)
_label_map_cache = TTLCache[str, Tuple[str, float, Dict[str, str]]](_settings.classes_cache_seconds)
_evidence_options_cache = TTLCache[str, Tuple[str, float, Dict[str, List[str]]]](_settings.classes_cache_seconds)


def _ensure_settings(settings: Settings | None) -> Settings:
    return settings or _settings


def _resolve_symptoms_path(dataset_dir: Path, settings: Settings) -> Path | None:
    dataset_symptoms_path = dataset_dir / "symptoms.json"
    if dataset_symptoms_path.exists():
        return dataset_symptoms_path

    global_symptoms_path = settings.data_root / "symptoms.json"
    if global_symptoms_path.exists():
        return global_symptoms_path
    return None


def list_datasets(settings: Settings | None = None) -> List[str]:
    settings = _ensure_settings(settings)

    def loader() -> List[str]:
        if not settings.data_root.exists():
            return []

        global_symptoms_exists = (settings.data_root / "symptoms.json").exists()
        items: list[str] = []
        for p in settings.data_root.iterdir():
            if not p.is_dir():
                continue
            if p.name.startswith("."):
                continue
            if p.name == settings.backup_dirname:
                continue

            if (p / "symptoms.json").exists():
                items.append(p.name)
                continue

            # When dataset-local symptoms is missing, allow global DATA_ROOT/symptoms.json.
            # Restrict to dataset-like folders to avoid listing utility dirs (e.g. backup/).
            if global_symptoms_exists and (
                (p / "images").is_dir() or (p / "healthy_images").is_dir()
            ):
                items.append(p.name)

        return sorted(items)

    return _datasets_cache.get_or_set("datasets", loader)


def load_global_symptoms(settings: Settings | None = None) -> Dict:
    """Classes + zh label map + evidence options straight from the global
    symptoms.json (data_root/symptoms.json), independent of any dataset.

    Used by the diagnosis draft editor, which edits against a dataset that may
    not exist yet (created only on submit) — so it can't use the dataset-scoped
    loaders that require the folder to exist.
    """
    settings = _ensure_settings(settings)
    path = settings.data_root / "symptoms.json"
    empty = {"classes": [], "label_map_zh": {}, "evidence_options_zh": {}}
    if not path.exists():
        return empty
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return empty
    label_map = raw.get("label_map") or {}
    data = raw.get("data") or {}

    def _key(k: str) -> int:
        try:
            return int(k)
        except Exception:
            return 0

    classes: List[str] = []
    label_map_zh: Dict[str, str] = {}
    evidence: Dict[str, List[str]] = {}
    for _id in sorted(label_map.keys(), key=_key):
        entry = label_map.get(_id) or {}
        if not isinstance(entry, dict):
            continue
        en = (entry.get("en") or "").strip() if isinstance(entry.get("en"), str) else ""
        if not en:
            continue
        classes.append(en)
        zh = entry.get("zh")
        if isinstance(zh, str) and zh:
            label_map_zh[en] = zh
        data_entry = data.get(_id) if isinstance(data, dict) else None
        caps: List[str] = []
        if isinstance(data_entry, dict):
            raw_caps = data_entry.get("captions_zh")
            if not isinstance(raw_caps, list) or not raw_caps:
                raw_caps = data_entry.get("captions_en")
            if isinstance(raw_caps, list):
                for cap in raw_caps:
                    if cap is None:
                        continue
                    text = str(cap).replace("\n", " ").strip()
                    if text:
                        caps.append(text)
        evidence[en] = caps
    return {"classes": classes, "label_map_zh": label_map_zh, "evidence_options_zh": evidence}


def load_dataset_meta(dataset: str, settings: Settings | None = None) -> Optional[Dict]:
    """Read a dataset's meta.json marker, or None if it has none (locked)."""
    settings = _ensure_settings(settings)
    meta_path = settings.data_root / dataset / DATASET_META_FILENAME
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def is_writable(dataset: str, settings: Settings | None = None) -> bool:
    """A dataset is a valid diagnosis-import target only if it carries a marker."""
    return load_dataset_meta(dataset, settings) is not None


def list_datasets_with_meta(settings: Settings | None = None) -> List[Dict]:
    """List datasets with lock/status info for the picker. Wraps list_datasets()
    (kept as List[str] for stats/backup) and joins each name's meta marker."""
    settings = _ensure_settings(settings)
    out: List[Dict] = []
    for name in list_datasets(settings):
        meta = load_dataset_meta(name, settings)
        out.append({
            "name": name,
            "locked": meta is None,
            "status": (meta or {}).get("status") if meta else None,
        })
    return out


def create_dataset(name: str, created_by: str, settings: Settings | None = None) -> Dict:
    """Create a new (writable) dataset folder for diagnosis writeback.

    Scaffolds images/ + healthy_images/, writes the meta.json marker, and inits
    the per-dataset SQLite DB. Taxonomy comes from the global symptoms.json
    fallback, so no per-dataset symptoms file is needed.
    """
    settings = _ensure_settings(settings)
    name = (name or "").strip()
    if not _DATASET_NAME_PATTERN.match(name):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="名稱僅能使用英數字、底線、連字號")
    target = (settings.data_root / name).resolve()
    if not str(target).startswith(str(settings.data_root.resolve())):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="非法資料集名稱")
    if target.exists():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="資料集已存在")

    (target / "images").mkdir(parents=True, exist_ok=True)
    (target / "healthy_images").mkdir(parents=True, exist_ok=True)
    meta = {
        "created_via": "diagnosis",
        "status": "pending",
        "created_by": created_by,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (target / DATASET_META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Lazy import to avoid a circular import (storage imports this module).
    from . import storage as storage_service
    storage_service.ensure_db(target, settings)

    _datasets_cache.clear("datasets")  # surface the new dataset immediately
    return {"name": name, "locked": False, "status": "pending"}


def remove_dataset(dataset: str, settings: Settings | None = None) -> None:
    """Delete a writable (diagnosis-created) dataset folder entirely. Refuses to
    touch locked/official datasets. Used to auto-clean a dataset emptied of all
    its tasks."""
    settings = _ensure_settings(settings)
    if not is_writable(dataset, settings):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="此資料集已鎖定，無法移除")
    target = resolve_dataset_path(dataset, settings)
    import shutil

    shutil.rmtree(target, ignore_errors=True)
    _datasets_cache.clear("datasets")


def resolve_dataset_path(dataset: str, settings: Settings | None = None) -> Path:
    settings = _ensure_settings(settings)
    if not _DATASET_NAME_PATTERN.match(dataset):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="非法資料集名稱")
    base = settings.data_root
    target = (base / dataset).resolve()
    if not str(target).startswith(str(base.resolve())) or not target.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="資料集不存在")
    return target


def load_classes(dataset: str, settings: Settings | None = None) -> List[str]:
    settings = _ensure_settings(settings)
    dataset_dir = resolve_dataset_path(dataset, settings)
    symptoms_path = _resolve_symptoms_path(dataset_dir, settings)
    if symptoms_path is None:
        return []

    def loader() -> Tuple[str, float, List[str]]:
        try:
            raw = json.loads(symptoms_path.read_text(encoding="utf-8"))
            label_map = raw.get("label_map") or {}

            # Sort by numeric key if possible to keep stable ordering.
            def key_fn(k: str) -> int:
                try:
                    return int(k)
                except Exception:
                    return 0

            classes = []
            for k in sorted(label_map.keys(), key=key_fn):
                entry = label_map.get(k) or {}
                en = entry.get("en")
                if isinstance(en, str):
                    en = en.strip()
                if en:
                    classes.append(en)
            return (str(symptoms_path), symptoms_path.stat().st_mtime, classes)
        except Exception:
            # On any error, fall back to empty list so API remains responsive.
            mtime = symptoms_path.stat().st_mtime if symptoms_path.exists() else 0.0
            return (str(symptoms_path), mtime, [])

    cached = _classes_cache.get_or_set(dataset, loader)
    source, mtime, classes = cached

    current_source = str(symptoms_path)
    current_mtime = symptoms_path.stat().st_mtime if symptoms_path.exists() else 0.0
    if current_source != source or current_mtime != mtime:
        _classes_cache.clear(dataset)
        _source, _mtime, classes = _classes_cache.get_or_set(dataset, loader)

    return classes


def load_label_map_zh(dataset: str, settings: Settings | None = None) -> Dict[str, str]:
    """Load mapping from English class to Chinese display text for a dataset.

    Reads dataset-level `symptoms.json` (or DATA_ROOT fallback) and converts
    its label_map section (which maps numeric ids to {en, zh}) into a dict
    mapping `en` -> `zh` for convenient lookup on the frontend.
    """
    settings = _ensure_settings(settings)
    dataset_dir = resolve_dataset_path(dataset, settings)
    json_path = _resolve_symptoms_path(dataset_dir, settings)
    if json_path is None:
        return {}

    if not json_path.exists():
        return {}

    def loader() -> Tuple[str, float, Dict[str, str]]:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            label_map = data.get("label_map") or {}
            result: Dict[str, str] = {}
            for _id, entry in label_map.items():
                if not isinstance(entry, dict):
                    continue
                en = entry.get("en")
                zh = entry.get("zh")
                if isinstance(en, str) and isinstance(zh, str) and en:
                    result[en] = zh
            return (str(json_path), json_path.stat().st_mtime, result)
        except Exception:
            # On error, fall back to empty mapping so frontend can still work
            mtime = json_path.stat().st_mtime if json_path.exists() else 0.0
            return (str(json_path), mtime, {})

    cached = _label_map_cache.get_or_set(dataset, loader)
    source, mtime, mapping = cached

    current_source = str(json_path)
    current_mtime = json_path.stat().st_mtime if json_path.exists() else 0.0
    if current_source != source or current_mtime != mtime:
        _label_map_cache.clear(dataset)
        _source, _mtime, mapping = _label_map_cache.get_or_set(dataset, loader)

    return mapping


def load_evidence_options_zh(dataset: str, settings: Settings | None = None) -> Dict[str, List[str]]:
    """Load evidence caption options (Chinese preferred) for a dataset.

    Reads dataset-level `symptoms.json` (or DATA_ROOT fallback) and converts its label_map + data
    sections into a mapping: `en_label` -> `captions_zh[]` (falling back to captions_en).
    """
    settings = _ensure_settings(settings)
    dataset_dir = resolve_dataset_path(dataset, settings)
    json_path = _resolve_symptoms_path(dataset_dir, settings)
    if json_path is None:
        return {}

    if not json_path.exists():
        return {}

    def loader() -> Tuple[str, float, Dict[str, List[str]]]:
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            label_map = raw.get("label_map") or {}
            data = raw.get("data") or {}

            result: Dict[str, List[str]] = {}
            for _id, entry in label_map.items():
                if not isinstance(entry, dict):
                    continue
                en = entry.get("en")
                if not isinstance(en, str):
                    continue
                en = en.strip()
                if not en:
                    continue

                data_entry = data.get(_id) if isinstance(data, dict) else None
                captions: List[str] = []

                if isinstance(data_entry, dict):
                    raw_caps = data_entry.get("captions_zh")
                    if not isinstance(raw_caps, list) or not raw_caps:
                        raw_caps = data_entry.get("captions_en")
                    if isinstance(raw_caps, list):
                        for cap in raw_caps:
                            if cap is None:
                                continue
                            text = str(cap).replace("\n", " ").strip()
                            if text:
                                captions.append(text)

                result[en] = captions

            return (str(json_path), json_path.stat().st_mtime, result)
        except Exception:
            # On error, fall back to empty mapping so frontend can still work
            mtime = json_path.stat().st_mtime if json_path.exists() else 0.0
            return (str(json_path), mtime, {})

    cached = _evidence_options_cache.get_or_set(dataset, loader)
    source, mtime, mapping = cached

    current_source = str(json_path)
    current_mtime = json_path.stat().st_mtime if json_path.exists() else 0.0
    if current_source != source or current_mtime != mtime:
        _evidence_options_cache.clear(dataset)
        _source, _mtime, mapping = _evidence_options_cache.get_or_set(dataset, loader)

    return mapping
