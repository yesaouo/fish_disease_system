from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Tuple, Optional

from fastapi import HTTPException, status

from ..config import Settings, get_settings
from . import datasets as datasets_service
from ..models import TaskDocument

_settings = get_settings()


def _ensure_settings(settings: Settings | None) -> Settings:
    return settings or _settings


def get_json_dir(dataset_dir: Path) -> Path:
    return dataset_dir / "annotations"


def get_image_dir(dataset_dir: Path) -> Path:
    return dataset_dir / "images"


def get_versions_dir(dataset_dir: Path) -> Path:
    return dataset_dir / "annotations_versions"


def _infer_image_size(image_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Try to read image dimensions, falling back silently if unavailable."""
    try:
        from PIL import Image  # type: ignore

        with Image.open(image_path) as img:
            width, height = img.size
            return int(width), int(height)
    except Exception:
        return (None, None)


def read_raw_json(path: Path) -> Dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def _normalize_task(
    raw: Dict,
    dataset: str,
    stem: str,
    image_filename: str,
    dataset_dir: Optional[Path] = None,
    settings: Settings | None = None,
) -> TaskDocument:
    settings = _ensure_settings(settings)
    dataset_dir = dataset_dir or datasets_service.resolve_dataset_path(dataset, settings)
    image_path = get_image_dir(dataset_dir) / image_filename
    raw = dict(raw)

    def _coerce_dim(value: Any) -> Optional[int]:
        try:
            v = int(round(float(value)))
            return v if v > 0 else None
        except Exception:
            return None

    width = _coerce_dim(raw.get("image_width"))
    height = _coerce_dim(raw.get("image_height"))
    if (width is None or height is None) and image_path.exists():
        inferred_w, inferred_h = _infer_image_size(image_path)
        width = width or inferred_w
        height = height or inferred_h
    raw["image_width"] = width or 1000
    raw["image_height"] = height or 1000

    raw.setdefault("dataset", dataset)
    raw["dataset"] = dataset
    raw.setdefault("image_filename", image_filename)
    raw.setdefault("last_modified_at", datetime.now(timezone.utc).isoformat())
    raw.setdefault("overall", {})
    raw.setdefault("detections", [])
    raw.setdefault("global_causes_zh", [])
    raw.setdefault("global_treatments_zh", [])
    raw.setdefault("comments", [])
    raw.setdefault("generated_by", raw.get("generated_by"))
    # New schema only; default role editors to None
    raw.setdefault("general_editor", None)
    raw.setdefault("expert_editor", None)

    # Legacy compatibility: map box_2d -> box_xyxy (x1,y1,x2,y2 in pixel space)
    try:
        dets = raw.get("detections", [])
        if isinstance(dets, list):
            max_dim = max(raw["image_width"], raw["image_height"])

            def _clamp(val: float, low: float, high: float) -> float:
                try:
                    return min(max(val, low), high)
                except Exception:
                    return val

            for d in dets:
                if not isinstance(d, dict):
                    continue
                if "box_xyxy" not in d and d.get("box_2d") is not None:
                    box = d.get("box_2d")
                    try:
                        y1, x1, y2, x2 = [float(v) for v in box]
                    except Exception:
                        continue
                    x1_new, y1_new, x2_new, y2_new = x1, y1, x2, y2
                    # If legacy values exceed image dims, assume 0-1000 scale and rescale.
                    if max_dim and any(coord > max_dim for coord in (x1_new, x2_new, y1_new, y2_new)):
                        y1_new = (y1_new / 1000.0) * raw["image_height"]
                        y2_new = (y2_new / 1000.0) * raw["image_height"]
                        x1_new = (x1_new / 1000.0) * raw["image_width"]
                        x2_new = (x2_new / 1000.0) * raw["image_width"]
                    x1_new = _clamp(x1_new, 0, raw["image_width"])
                    x2_new = _clamp(x2_new, 0, raw["image_width"])
                    y1_new = _clamp(y1_new, 0, raw["image_height"])
                    y2_new = _clamp(y2_new, 0, raw["image_height"])
                    d["box_xyxy"] = [x1_new, y1_new, x2_new, y2_new]
                d.pop("box_2d", None)
    except Exception:
        pass

    # Backward/forward compatibility: map main/sub to label/evidence if present
    try:
        dets = raw.get("detections", [])
        if isinstance(dets, list):
            for d in dets:
                if isinstance(d, dict):
                    if not d.get("label") and d.get("main_category"):
                        d["label"] = str(d.get("main_category") or "")
                    if ("evidence_zh" not in d) and d.get("sub_category") is not None:
                        d["evidence_zh"] = str(d.get("sub_category") or "")
    except Exception:
        pass

    return TaskDocument.model_validate(raw)


def _determine_image_filename(
    dataset: str,
    dataset_dir: Path,
    stem: str,
    settings: Settings,
) -> str:
    image_dir = get_image_dir(dataset_dir)
    if not image_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="影像資料夾不存在")
    for ext in settings.image_extensions:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate.name
    # fallback search ignoring ext set
    matches = list(image_dir.glob(f"{stem}.*"))
    if matches:
        return matches[0].name
    append_audit_log(
        {
            "who": "system",
            "when": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "task_id": stem,
            "action": "missing_image",
        },
        settings,
    )
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="影像不存在")


def iter_task_files(dataset_dir: Path) -> Iterable[Path]:
    json_dir = get_json_dir(dataset_dir)
    if not json_dir.exists():
        return []
    return sorted(json_dir.glob("*.json"))


def load_task(
    dataset: str,
    stem: str,
    settings: Settings | None = None,
) -> TaskDocument:
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    json_dir = get_json_dir(dataset_dir)
    json_path = json_dir / f"{stem}.json"
    if not json_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="JSON 不存在")
    raw = read_raw_json(json_path)
    image_filename = _determine_image_filename(dataset, dataset_dir, stem, settings)
    return _normalize_task(raw, dataset, stem, image_filename, dataset_dir, settings)


def load_all_tasks(
    dataset: str,
    settings: Settings | None = None,
) -> Generator[Tuple[str, TaskDocument], None, None]:
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    json_dir = get_json_dir(dataset_dir)
    if not json_dir.exists():
        return
    for json_path in json_dir.glob("*.json"):
        stem = json_path.stem
        try:
            raw = read_raw_json(json_path)
            image_filename = _determine_image_filename(dataset, dataset_dir, stem, settings)
            doc = _normalize_task(raw, dataset, stem, image_filename, dataset_dir, settings)
            yield stem, doc
        except HTTPException:
            continue
        except Exception:
            continue


def atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_file:
            json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
            tmp_file.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_task(
    document: TaskDocument,
    settings: Settings | None = None,
) -> None:
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(document.dataset, settings)
    json_dir = get_json_dir(dataset_dir)
    json_path = json_dir / f"{Path(document.image_filename).stem}.json"
    payload = document.model_dump(mode="json")
    atomic_write_json(json_path, payload)


def save_task_version(
    document: TaskDocument,
    settings: Settings | None = None,
) -> None:
    """Persist a snapshot of the task as a versioned JSON file.

    Files are written under `<dataset>/annotations_versions` with the pattern
    `<image_stem>_vN.json`, where N is a monotonically increasing integer
    per image, incremented only on submit.
    """
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(document.dataset, settings)
    versions_dir = get_versions_dir(dataset_dir)

    image_stem = Path(document.image_filename).stem

    # Scan existing versions for this image stem to determine next N
    max_version = 0
    if versions_dir.exists():
        for p in versions_dir.glob(f"{image_stem}_v*.json"):
            stem = p.stem
            # Expect suffix `_vN`
            if "_v" not in stem:
                continue
            base, suffix = stem.rsplit("_v", 1)
            if base != image_stem:
                continue
            try:
                n = int(suffix)
            except ValueError:
                continue
            if n > max_version:
                max_version = n

    next_version = max_version + 1
    target = versions_dir / f"{image_stem}_v{next_version}.json"
    payload = document.model_dump(mode="json")
    atomic_write_json(target, payload)


def append_audit_log(entry: Dict[str, str], settings: Settings | None = None) -> None:
    settings = _ensure_settings(settings)
    log_path = settings.audit_log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with open(log_path, "a", encoding="utf-8") as fp:
        fp.write(line + "\n")
