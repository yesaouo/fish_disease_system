from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple
import json

from fastapi import HTTPException, status

from ..config import Settings, get_settings
from ..utils.cache import TTLCache

_DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")

_settings = get_settings()
_datasets_cache = TTLCache[str, List[str]](_settings.dataset_cache_seconds)
_classes_cache = TTLCache[str, Tuple[float, List[str]]](_settings.classes_cache_seconds)
_label_map_cache = TTLCache[str, Tuple[float, Dict[str, str]]](_settings.classes_cache_seconds)


def _ensure_settings(settings: Settings | None) -> Settings:
    return settings or _settings


def list_datasets(settings: Settings | None = None) -> List[str]:
    settings = _ensure_settings(settings)

    def loader() -> List[str]:
        if not settings.data_root.exists():
            return []
        items = [
            p.name
            for p in settings.data_root.iterdir()
            if p.is_dir()
            and not p.name.startswith(".")
            and ((p / "symptoms.json").exists() or (p / "classes.txt").exists())
        ]
        return sorted(items)

    return _datasets_cache.get_or_set("datasets", loader)


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
    symptoms_path = dataset_dir / "symptoms.json"
    classes_path = dataset_dir / "classes.txt"

    # Prefer symptoms.json as the single source of truth for class names.
    # Fall back to classes.txt if symptoms.json is missing, for backward compatibility.
    if symptoms_path.exists():
        source_path = symptoms_path

        def loader() -> Tuple[float, List[str]]:
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
                return (symptoms_path.stat().st_mtime, classes)
            except Exception:
                # On any error, fall back to empty list so API remains responsive.
                return (symptoms_path.stat().st_mtime if symptoms_path.exists() else 0.0, [])
    else:
        source_path = classes_path
        if not classes_path.exists():
            return []

        def loader() -> Tuple[float, List[str]]:
            text = classes_path.read_text(encoding="utf-8")
            classes = [line.strip() for line in text.splitlines() if line.strip()]
            return (classes_path.stat().st_mtime, classes)

    cached = _classes_cache.get_or_set(dataset, loader)
    mtime, classes = cached

    current_mtime = source_path.stat().st_mtime
    if current_mtime != mtime:
        _classes_cache.clear(dataset)
        mtime, classes = _classes_cache.get_or_set(dataset, loader)

    return classes


def load_label_map_zh(dataset: str, settings: Settings | None = None) -> Dict[str, str]:
    """Load mapping from English class to Chinese display text for a dataset.

    Reads symptoms.json under the dataset directory and converts
    its label_map section (which maps numeric ids to {en, zh}) into a dict
    mapping `en` -> `zh` for convenient lookup on the frontend.
    """
    settings = _ensure_settings(settings)
    dataset_dir = resolve_dataset_path(dataset, settings)
    json_path = dataset_dir / "symptoms.json"
    if not json_path.exists():
        return {}

    def loader() -> Tuple[float, Dict[str, str]]:
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
            return (json_path.stat().st_mtime, result)
        except Exception:
            # On error, fall back to empty mapping so frontend can still work
            return (json_path.stat().st_mtime if json_path.exists() else 0.0, {})

    cached = _label_map_cache.get_or_set(dataset, loader)
    mtime, mapping = cached

    current_mtime = json_path.stat().st_mtime
    if current_mtime != mtime:
        _label_map_cache.clear(dataset)
        mtime, mapping = _label_map_cache.get_or_set(dataset, loader)

    return mapping
