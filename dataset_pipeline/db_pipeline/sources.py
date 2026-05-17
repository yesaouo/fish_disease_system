from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from .._bbox import bbox_contains


HEALTHY_LABEL = "healthy_region"


@dataclass
class DatasetSource:
    name: str
    db_path: Path
    dir: Path


@dataclass
class TaskRecord:
    dataset: str
    task_id: str
    image_filename: str
    image_path: Path
    is_healthy: bool
    doc: dict
    image_id: int = 0
    split: str = ""


@dataclass
class FilterStats:
    dropped_by_comments: int = 0
    dropped_by_submit_filter: int = 0
    dropped_by_missing_image: int = 0
    skipped_bboxes_unknown_label: Counter = field(default_factory=Counter)
    used_labels: Counter = field(default_factory=Counter)


def find_datasets(annotation_root: Path) -> list[DatasetSource]:
    out: list[DatasetSource] = []
    if not annotation_root.is_dir():
        return out
    for child in sorted(annotation_root.iterdir()):
        if not child.is_dir() or child.name.startswith("_") or child.name == "backup":
            continue
        db = child / "annotations.db"
        if db.is_file():
            out.append(DatasetSource(name=child.name, db_path=db, dir=child))
    return out


def resolve_image_path(dataset_dir: Path, filename: str) -> Optional[Path]:
    for sub in ("images", "healthy_images"):
        p = dataset_dir / sub / filename
        if p.is_file():
            return p
    return None


def iter_db_rows(db_path: Path) -> Iterator[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT task_id, image_filename, is_healthy, comments_count,
                   general_editors_json, expert_editors_json, doc_json
            FROM tasks
            ORDER BY sort_index ASC
            """
        ).fetchall()
    finally:
        conn.close()
    yield from rows


def _normalize_detections(
    detections: list,
    name_to_cat_id: dict[str, int],
    unknown_counter: Counter,
    used_counter: Counter,
) -> list[dict]:
    valid: list[dict] = []
    for det in detections or []:
        if not isinstance(det, dict):
            continue
        label = (det.get("label") or "").strip()
        if not label:
            continue
        box = det.get("box_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            coords = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        x1, y1, x2, y2 = coords
        if x2 <= x1 or y2 <= y1:
            continue
        if label not in name_to_cat_id:
            unknown_counter[label] += 1
            continue
        used_counter[label] += 1
        det = dict(det)
        det["label"] = label
        det["box_xyxy"] = coords
        valid.append(det)

    healthy_indices = [i for i, d in enumerate(valid) if d["label"] == HEALTHY_LABEL]
    wound_boxes = [d["box_xyxy"] for d in valid if d["label"] != HEALTHY_LABEL]
    to_remove: set[int] = set()
    for h_idx in healthy_indices:
        h_box = valid[h_idx]["box_xyxy"]
        for w_box in wound_boxes:
            if bbox_contains(w_box, h_box, fmt="xyxy"):
                to_remove.add(h_idx)
                break
    return [d for i, d in enumerate(valid) if i not in to_remove]


def load_tasks(
    sources: list[DatasetSource],
    name_to_cat_id: dict[str, int],
    require_submit: bool,
    require_expert_submit: bool,
    progress=None,
) -> tuple[list[TaskRecord], FilterStats]:
    stats = FilterStats()
    out: list[TaskRecord] = []

    for src in sources:
        rows = list(iter_db_rows(src.db_path))
        iterator = progress(rows, desc=f"db:{src.name}", leave=False) if progress else rows
        for row in iterator:
            if int(row["comments_count"] or 0) > 0:
                stats.dropped_by_comments += 1
                continue

            general_eds = json.loads(row["general_editors_json"] or "[]")
            expert_eds = json.loads(row["expert_editors_json"] or "[]")

            if require_expert_submit and not expert_eds:
                stats.dropped_by_submit_filter += 1
                continue
            if require_submit and not general_eds and not expert_eds:
                stats.dropped_by_submit_filter += 1
                continue

            img_path = resolve_image_path(src.dir, row["image_filename"])
            if img_path is None:
                stats.dropped_by_missing_image += 1
                continue

            try:
                doc = json.loads(row["doc_json"])
            except (ValueError, TypeError):
                continue

            valid_dets = _normalize_detections(
                doc.get("detections") or [],
                name_to_cat_id,
                stats.skipped_bboxes_unknown_label,
                stats.used_labels,
            )
            doc["detections"] = valid_dets
            non_healthy = sum(1 for d in valid_dets if d["label"] != HEALTHY_LABEL)
            is_healthy = (len(valid_dets) == 0) or (non_healthy == 0)
            doc["is_healthy"] = is_healthy

            out.append(
                TaskRecord(
                    dataset=src.name,
                    task_id=row["task_id"],
                    image_filename=row["image_filename"],
                    image_path=img_path,
                    is_healthy=is_healthy,
                    doc=doc,
                )
            )

    return out, stats
