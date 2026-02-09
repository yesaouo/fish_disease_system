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
                (p / "images").is_dir() or (p / "healthy_images").is_dir() or (p / "annotations").is_dir()
            ):
                items.append(p.name)

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
