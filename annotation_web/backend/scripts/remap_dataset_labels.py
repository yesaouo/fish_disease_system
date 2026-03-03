#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is importable when running as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.app.config import get_settings
from backend.app.services import datasets as datasets_service
from backend.app.services import storage as storage_service


@dataclass
class RunSummary:
    scanned_tasks: int = 0
    changed_tasks: int = 0
    changed_detections: int = 0
    json_decode_errors: int = 0
    doc_not_object: int = 0
    detections_not_list: int = 0
    detection_not_object: int = 0
    mapped_labels: int = 0
    cleared_labels: int = 0
    removed_evidence_index: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass
    return conn


def _derive_is_healthy(raw: dict[str, Any]) -> bool:
    dets = raw.get("detections")
    if not isinstance(dets, list) or len(dets) == 0:
        return True
    for d in dets:
        if not isinstance(d, dict):
            return False
        if str(d.get("label") or "").strip() != storage_service.HEALTHY_LABEL:
            return False
    return True


def _load_mapping_file(mapping_path: Path, *, allowed_labels: set[str]) -> dict[str, str]:
    if not mapping_path.exists():
        raise SystemExit(f"Mapping file not found: {mapping_path}")

    mapping: dict[str, str] = {}

    with mapping_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) != 2:
                raise SystemExit(
                    f"Invalid mapping format at line {lineno}: {line.rstrip()!r} "
                    f"(expected: '<old_label> <new_label>')"
                )

            old_label, new_label = parts

            if old_label in mapping and mapping[old_label] != new_label:
                raise SystemExit(
                    f"Conflicting mapping for {old_label!r}: "
                    f"{mapping[old_label]!r} vs {new_label!r}"
                )

            if new_label not in allowed_labels:
                raise SystemExit(
                    f"Mapped target label {new_label!r} at line {lineno} "
                    f"is not in new dataset classes."
                )

            mapping[old_label] = new_label

    return mapping


def _normalize_label(value: Any) -> str:
    return str(value or "").strip()


def _remap_detection(
    det: dict[str, Any],
    *,
    allowed_labels: set[str],
    mapping: dict[str, str],
    summary: RunSummary,
    detail_counts: Counter[str],
) -> bool:
    """
    Returns True if this detection changed.
    """
    before = json.dumps(det, ensure_ascii=False, sort_keys=True)

    raw_label = det.get("label", "")
    label = _normalize_label(raw_label)

    # 規則：
    # 1) 同名且新版仍存在 -> 保留
    # 2) 同名不存在，但 mapping 有對應 -> 改成新名，移除 evidence_index
    # 3) 同名不存在，也無 mapping -> label="", 移除 evidence_index
    if label in allowed_labels:
        target_label = label
        should_remove_evidence = False
        reason = "kept"
    elif label in mapping:
        target_label = mapping[label]
        should_remove_evidence = True
        reason = "mapped"
    else:
        target_label = ""
        should_remove_evidence = True
        reason = "cleared"

    det["label"] = target_label

    if should_remove_evidence and "evidence_index" in det:
        det.pop("evidence_index", None)
        summary.removed_evidence_index += 1
        detail_counts["evidence_index.removed"] += 1

    after = json.dumps(det, ensure_ascii=False, sort_keys=True)
    changed = before != after

    if changed:
        summary.changed_detections += 1

        if reason == "mapped":
            summary.mapped_labels += 1
            detail_counts["label.mapped"] += 1
        elif reason == "cleared":
            summary.cleared_labels += 1
            detail_counts["label.cleared_to_empty"] += 1
        else:
            # 例如 trim 空白、補上缺失的 label key 等
            detail_counts["label.normalized_but_kept"] += 1

    return changed


def _print_counter(title: str, counts: Counter[str]) -> None:
    print(f"\n{title}")
    if not counts:
        print("  (none)")
        return
    for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  - {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remap detection labels in dataset DB tasks according to new dataset classes "
            "and an old->new label mapping file."
        )
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Dataset folder name under DATA_ROOT.",
    )
    parser.add_argument(
        "--mapping-file",
        "-m",
        required=True,
        help="Path to mapping file. Each line: '<old_label> <new_label>'",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["check", "apply"],
        help="check: only report changes; apply: write DB changes.",
    )
    parser.add_argument(
        "--updated-by",
        default="system_label_remap",
        help="history.updated_by value in apply mode.",
    )
    parser.add_argument(
        "--action-name",
        default="remap_labels_to_new_classes",
        help="history.action value in apply mode.",
    )
    parser.add_argument(
        "--show-task-examples",
        type=int,
        default=10,
        help="Max number of changed task IDs to print.",
    )
    args = parser.parse_args()

    settings = get_settings()
    dataset = str(args.dataset).strip()
    do_apply = str(args.mode).strip().lower() == "apply"

    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    db_path = storage_service.ensure_db(dataset_dir, settings)

    classes = datasets_service.load_classes(dataset, settings)
    allowed_labels = set(classes)
    if not allowed_labels:
        raise SystemExit("New classes is empty; refusing to continue.")

    mapping_path = Path(args.mapping_file).expanduser().resolve()
    mapping = _load_mapping_file(mapping_path, allowed_labels=allowed_labels)

    rows = storage_service.list_task_rows(dataset, settings)

    summary = RunSummary(scanned_tasks=len(rows))
    detail_counts: Counter[str] = Counter()
    task_change_counter: Counter[str] = Counter()

    updates: list[tuple[str, str, str, int, int, str, str]] = []
    # tuple: (task_id, updated_at, updated_by, is_healthy, comments_count, old_json, new_json)

    for row in rows:
        task_id = str(row["task_id"])
        old_json = str(row["doc_json"] or "")

        try:
            raw = json.loads(old_json)
        except Exception:
            summary.json_decode_errors += 1
            detail_counts["doc_json.invalid_json"] += 1
            continue

        if not isinstance(raw, dict):
            summary.doc_not_object += 1
            detail_counts["doc_json.not_object"] += 1
            continue

        dets = raw.get("detections")
        if not isinstance(dets, list):
            summary.detections_not_list += 1
            detail_counts["detections.not_list"] += 1
            continue

        before_doc_sorted = json.dumps(raw, ensure_ascii=False, sort_keys=True)
        changed_in_task = 0

        for det in dets:
            if not isinstance(det, dict):
                summary.detection_not_object += 1
                detail_counts["detections.item_not_object"] += 1
                continue

            changed = _remap_detection(
                det,
                allowed_labels=allowed_labels,
                mapping=mapping,
                summary=summary,
                detail_counts=detail_counts,
            )
            if changed:
                changed_in_task += 1

        after_doc_sorted = json.dumps(raw, ensure_ascii=False, sort_keys=True)
        if before_doc_sorted == after_doc_sorted:
            continue

        summary.changed_tasks += 1
        task_change_counter[task_id] = changed_in_task

        if do_apply:
            updated_at = _now_iso()
            comments = raw.get("comments")
            comments_count = len(comments) if isinstance(comments, list) else 0
            is_healthy = 1 if _derive_is_healthy(raw) else 0
            new_json = json.dumps(raw, ensure_ascii=False)

            updates.append(
                (
                    task_id,
                    updated_at,
                    str(args.updated_by).strip() or "system_label_remap",
                    is_healthy,
                    comments_count,
                    old_json,
                    new_json,
                )
            )

    if do_apply and updates:
        conn = _connect(db_path)
        try:
            conn.execute("BEGIN IMMEDIATE;")
            for task_id, updated_at, updated_by, is_healthy, comments_count, old_doc, new_doc in updates:
                conn.execute(
                    """
                    INSERT INTO history(task_id, updated_at, updated_by, action, old_json, new_json)
                    VALUES(?,?,?,?,?,?);
                    """,
                    (task_id, updated_at, updated_by, str(args.action_name), old_doc, new_doc),
                )
                conn.execute(
                    """
                    UPDATE tasks
                    SET is_healthy=?,
                        last_modified_at=?,
                        comments_count=?,
                        doc_json=?
                    WHERE task_id=?;
                    """,
                    (is_healthy, updated_at, comments_count, new_doc, task_id),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    print(f"Dataset:      {dataset}")
    print(f"Mode:         {args.mode}")
    print(f"DB:           {db_path}")
    print(f"Mapping file: {mapping_path}")
    print(f"New classes:  {len(classes)}")
    print(f"Mappings:     {len(mapping)}")
    print(f"Rows:         {summary.scanned_tasks}")
    print(f"Changed tasks:       {summary.changed_tasks}")
    print(f"Changed detections:  {summary.changed_detections}")
    print(f"Mapped labels:       {summary.mapped_labels}")
    print(f"Cleared labels:      {summary.cleared_labels}")
    print(f"Removed evidence_index: {summary.removed_evidence_index}")
    print(f"JSON decode errors:  {summary.json_decode_errors}")
    print(f"doc_json not object: {summary.doc_not_object}")
    print(f"detections not list: {summary.detections_not_list}")
    print(f"detection not object: {summary.detection_not_object}")

    _print_counter("Detail counts", detail_counts)

    limit = max(int(args.show_task_examples), 0)
    if limit > 0 and task_change_counter:
        print(f"\nChanged task examples (top {limit})")
        for task_id, n_changes in task_change_counter.most_common(limit):
            print(f"  - {task_id}: {n_changes}")


if __name__ == "__main__":
    main()