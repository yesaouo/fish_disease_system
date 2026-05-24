#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HEALTHY_LABEL = "healthy_region"


@dataclass
class Summary:
    rows: int = 0
    changed_tasks: int = 0
    changed_detections: int = 0
    label_mapped: int = 0
    label_cleared: int = 0
    evidence_mapped: int = 0
    evidence_removed: int = 0
    evidence_missing: int = 0
    json_errors: int = 0
    doc_not_object: int = 0
    detections_not_list: int = 0
    detection_not_object: int = 0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def parse_int(value: Any) -> int | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed >= 0 else None


def load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return data


def load_new_symptoms(path: Path) -> tuple[dict[str, str], dict[str, list[str]], set[str]]:
    raw = load_json_object(path)
    label_map = raw.get("label_map") or {}
    data = raw.get("data") or {}

    id_to_label: dict[str, str] = {}
    label_to_captions: dict[str, list[str]] = {}
    allowed: set[str] = set()

    for raw_id, entry in label_map.items():
        if not isinstance(entry, dict):
            continue
        label_id = str(raw_id)
        en = entry.get("en")
        if not isinstance(en, str) or not en.strip():
            continue
        en = en.strip()
        id_to_label[label_id] = en
        allowed.add(en)

        captions: list[str] = []
        data_entry = data.get(label_id) if isinstance(data, dict) else None
        if isinstance(data_entry, dict):
            raw_caps = data_entry.get("captions_zh")
            if not isinstance(raw_caps, list) or not raw_caps:
                raw_caps = data_entry.get("captions_en")
            if isinstance(raw_caps, list):
                captions = [
                    str(c).replace("\n", " ").strip()
                    for c in raw_caps
                    if c is not None and str(c).strip()
                ]
        label_to_captions[en] = captions

    if not allowed:
        raise SystemExit(f"No classes found in {path}")
    return id_to_label, label_to_captions, allowed


def load_remap(
    path: Path,
    id_to_new_label: dict[str, str],
) -> tuple[dict[str, str | None], dict[tuple[str, int], tuple[str | None, int | None]]]:
    raw = load_json_object(path)
    class_remap_raw = raw.get("class_remap") or {}
    caption_remap_raw = raw.get("caption_remap") or []

    class_remap: dict[str, str | None] = {}
    old_label_to_id: dict[str, str] = {}

    for old_id, entry in class_remap_raw.items():
        if not isinstance(entry, dict):
            continue
        old_label = entry.get("old_label_en")
        if not isinstance(old_label, str) or not old_label.strip():
            continue
        old_label = old_label.strip()
        old_label_to_id[old_label] = str(old_id)

        if entry.get("status") == "mapped":
            new_id = entry.get("new_label_id")
            new_label = id_to_new_label.get(str(new_id)) if new_id is not None else None
            if not new_label:
                raw_new_label = entry.get("new_label_en")
                new_label = str(raw_new_label).strip() if raw_new_label else None
            class_remap[old_label] = new_label
        else:
            class_remap[old_label] = None

    if not isinstance(caption_remap_raw, list):
        raise SystemExit(f"{path}: caption_remap must be a list")

    caption_remap: dict[tuple[str, int], tuple[str | None, int | None]] = {}
    for entry in caption_remap_raw:
        if not isinstance(entry, dict):
            continue
        old_label = entry.get("old_label_en")
        old_idx = parse_int(entry.get("old_caption_index"))
        if not isinstance(old_label, str) or old_idx is None:
            continue
        old_label = old_label.strip()

        if entry.get("status") != "mapped":
            caption_remap[(old_label, old_idx - 1)] = (None, None)
            continue

        new_id = entry.get("new_label_id")
        new_label = id_to_new_label.get(str(new_id)) if new_id is not None else None
        if not new_label:
            raw_new_label = entry.get("new_label_en")
            new_label = str(raw_new_label).strip() if raw_new_label else None

        new_idx = parse_int(entry.get("new_caption_index"))
        caption_remap[(old_label, old_idx - 1)] = (
            new_label,
            (new_idx - 1) if new_idx is not None else None,
        )

    missing = [label for label in old_label_to_id if label not in class_remap]
    if missing:
        raise SystemExit(f"Class remap missing labels: {missing[:10]}")

    return class_remap, caption_remap


def derive_is_healthy(raw: dict[str, Any]) -> bool:
    dets = raw.get("detections")
    if not isinstance(dets, list) or not dets:
        return True
    return all(
        isinstance(d, dict) and str(d.get("label") or "").strip() == HEALTHY_LABEL
        for d in dets
    )


def required_fields_ok(raw: dict[str, Any]) -> bool:
    if derive_is_healthy(raw):
        return True

    overall = raw.get("overall") if isinstance(raw.get("overall"), dict) else {}
    if not str(overall.get("colloquial_zh") or "").strip():
        return False
    if not str(overall.get("medical_zh") or "").strip():
        return False

    dets = raw.get("detections")
    if not isinstance(dets, list) or not dets:
        return False
    for det in dets:
        if not isinstance(det, dict):
            return False
        label = str(det.get("label") or "").strip()
        if not label:
            return False
        if label != HEALTHY_LABEL and det.get("evidence_index") in (None, ""):
            return False

    causes = raw.get("global_causes_zh")
    treatments = raw.get("global_treatments_zh")
    return isinstance(causes, list) and bool(causes) and isinstance(treatments, list) and bool(treatments)


def compute_expert_complete(raw: dict[str, Any]) -> bool:
    experts = raw.get("expert_editor") or []
    if not isinstance(experts, list) or not experts:
        return False
    comments = raw.get("comments") or []
    if isinstance(comments, list) and comments:
        return True
    return required_fields_ok(raw)


def set_new_evidence_text(
    det: dict[str, Any],
    label: str,
    idx: int | None,
    label_to_captions: dict[str, list[str]],
    *,
    replace_evidence_zh: bool,
) -> None:
    if not replace_evidence_zh:
        return
    if idx is None:
        det["evidence_zh"] = ""
        return
    options = label_to_captions.get(label) or []
    det["evidence_zh"] = options[idx] if 0 <= idx < len(options) else ""


def remap_detection(
    det: dict[str, Any],
    *,
    allowed_labels: set[str],
    class_remap: dict[str, str | None],
    caption_remap: dict[tuple[str, int], tuple[str | None, int | None]],
    label_to_captions: dict[str, list[str]],
    replace_evidence_zh: bool,
    summary: Summary,
    detail: Counter[str],
) -> bool:
    before = json.dumps(det, ensure_ascii=False, sort_keys=True)
    old_label = str(det.get("label") or "").strip()
    old_idx = parse_int(det.get("evidence_index"))

    if old_label in allowed_labels:
        target_label: str | None = old_label
        target_idx = old_idx
    elif old_idx is not None and (old_label, old_idx) in caption_remap:
        target_label, target_idx = caption_remap[(old_label, old_idx)]
        if target_label:
            summary.evidence_mapped += 1
            detail["evidence_index.mapped_by_caption"] += 1
        else:
            summary.evidence_removed += 1
            detail["evidence_index.removed_by_caption"] += 1
    else:
        target_label = class_remap.get(old_label)
        target_idx = None
        if old_idx is not None:
            summary.evidence_missing += 1
            detail["evidence_index.removed_no_caption_mapping"] += 1

    if target_label and target_label in allowed_labels:
        det["label"] = target_label
        if old_label != target_label:
            summary.label_mapped += 1
            detail[f"label.mapped.{old_label}->{target_label}"] += 1
        det["evidence_index"] = int(target_idx) if target_idx is not None else None
        set_new_evidence_text(
            det,
            target_label,
            target_idx,
            label_to_captions,
            replace_evidence_zh=replace_evidence_zh,
        )
    else:
        det["label"] = ""
        det.pop("evidence_index", None)
        if replace_evidence_zh:
            det["evidence_zh"] = ""
        if old_label:
            summary.label_cleared += 1
            detail[f"label.cleared.{old_label}"] += 1
        if old_idx is not None:
            summary.evidence_removed += 1

    changed = before != json.dumps(det, ensure_ascii=False, sort_keys=True)
    if changed:
        summary.changed_detections += 1
    return changed


def backup_db(source: sqlite3.Connection, db_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.bak.{stamp}")
    backup_conn = sqlite3.connect(str(backup_path))
    try:
        source.backup(backup_conn)
    finally:
        backup_conn.close()
    return backup_path


def print_counter(title: str, counter: Counter[str], *, limit: int = 30) -> None:
    print(f"\n{title}")
    if not counter:
        print("  (none)")
        return
    for idx, (key, value) in enumerate(counter.most_common()):
        if idx >= limit:
            print(f"  ... {len(counter) - limit} more")
            break
        print(f"  - {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remap annotation_web SQLite task labels/evidence_index using "
            "symptoms_caption_remap.json."
        )
    )
    parser.add_argument("--db", required=True, type=Path, help="Path to annotations.db.")
    parser.add_argument(
        "--symptoms",
        required=True,
        type=Path,
        help="Path to the new symptoms.json.",
    )
    parser.add_argument(
        "--caption-remap",
        required=True,
        type=Path,
        help="Path to symptoms_caption_remap.json.",
    )
    parser.add_argument("--mode", choices=["check", "apply"], required=True)
    parser.add_argument("--updated-by", default="system_caption_remap")
    parser.add_argument("--action-name", default="remap_labels_and_evidence_from_caption_remap")
    parser.add_argument(
        "--replace-evidence-zh",
        action="store_true",
        help="Replace evidence_zh with the new caption text when evidence_index is remapped.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a SQLite backup before apply mode.",
    )
    parser.add_argument("--show-task-examples", type=int, default=15)
    args = parser.parse_args()

    db_path = args.db.expanduser().resolve()
    symptoms_path = args.symptoms.expanduser().resolve()
    remap_path = args.caption_remap.expanduser().resolve()
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    id_to_new_label, label_to_captions, allowed = load_new_symptoms(symptoms_path)
    class_remap, caption_remap = load_remap(remap_path, id_to_new_label)

    summary = Summary()
    detail: Counter[str] = Counter()
    changed_task_counts: Counter[str] = Counter()
    updates: list[tuple[str, str, str, int, int, str, int]] = []

    conn = connect(db_path)
    backup_path: Path | None = None
    try:
        cols = {str(row["name"]) for row in conn.execute("PRAGMA table_info(tasks);").fetchall()}
        has_version = "version" in cols
        has_expert_complete = "expert_complete" in cols

        rows = conn.execute("SELECT task_id, doc_json FROM tasks ORDER BY sort_index ASC;").fetchall()
        summary.rows = len(rows)

        for row in rows:
            task_id = str(row["task_id"])
            old_doc = str(row["doc_json"] or "")
            try:
                raw = json.loads(old_doc)
            except Exception:
                summary.json_errors += 1
                detail["doc_json.invalid_json"] += 1
                continue
            if not isinstance(raw, dict):
                summary.doc_not_object += 1
                detail["doc_json.not_object"] += 1
                continue

            dets = raw.get("detections")
            if not isinstance(dets, list):
                summary.detections_not_list += 1
                detail["detections.not_list"] += 1
                continue

            changed_in_task = 0
            for det in dets:
                if not isinstance(det, dict):
                    summary.detection_not_object += 1
                    detail["detections.item_not_object"] += 1
                    continue
                if remap_detection(
                    det,
                    allowed_labels=allowed,
                    class_remap=class_remap,
                    caption_remap=caption_remap,
                    label_to_captions=label_to_captions,
                    replace_evidence_zh=bool(args.replace_evidence_zh),
                    summary=summary,
                    detail=detail,
                ):
                    changed_in_task += 1

            if changed_in_task == 0:
                continue

            updated_at = now_iso()
            raw["last_modified_at"] = updated_at
            comments = raw.get("comments")
            comments_count = len(comments) if isinstance(comments, list) else 0
            is_healthy = 1 if derive_is_healthy(raw) else 0
            expert_complete = 1 if compute_expert_complete(raw) else 0
            new_doc = json.dumps(raw, ensure_ascii=False)

            summary.changed_tasks += 1
            changed_task_counts[task_id] = changed_in_task
            updates.append((task_id, updated_at, old_doc, is_healthy, comments_count, new_doc, expert_complete))

        if args.mode == "apply" and updates:
            if not args.no_backup:
                backup_path = backup_db(conn, db_path)

            conn.execute("BEGIN IMMEDIATE;")
            try:
                for task_id, updated_at, old_doc, is_healthy, comments_count, new_doc, expert_complete in updates:
                    conn.execute(
                        """
                        INSERT INTO history(task_id, updated_at, updated_by, action, old_json, new_json)
                        VALUES(?,?,?,?,?,?);
                        """,
                        (task_id, updated_at, str(args.updated_by), str(args.action_name), old_doc, new_doc),
                    )

                    assignments = [
                        "is_healthy=?",
                        "last_modified_at=?",
                        "comments_count=?",
                        "doc_json=?",
                    ]
                    values: list[Any] = [is_healthy, updated_at, comments_count, new_doc]
                    if has_version:
                        assignments.append("version=version+1")
                    if has_expert_complete:
                        assignments.append("expert_complete=?")
                        values.append(expert_complete)
                    values.append(task_id)
                    conn.execute(
                        f"UPDATE tasks SET {', '.join(assignments)} WHERE task_id=?;",
                        values,
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    finally:
        conn.close()

    if backup_path:
        print(f"Backup: {backup_path}")
    print(f"Mode: {args.mode}")
    print(f"DB: {db_path}")
    print(f"Symptoms: {symptoms_path}")
    print(f"Caption remap: {remap_path}")
    print(f"New classes: {len(allowed)}")
    print(f"Class mappings: {len(class_remap)}")
    print(f"Caption mappings: {len(caption_remap)}")
    print(f"Rows: {summary.rows}")
    print(f"Changed tasks: {summary.changed_tasks}")
    print(f"Changed detections: {summary.changed_detections}")
    print(f"Label mapped: {summary.label_mapped}")
    print(f"Label cleared: {summary.label_cleared}")
    print(f"Evidence mapped: {summary.evidence_mapped}")
    print(f"Evidence removed: {summary.evidence_removed}")
    print(f"Evidence removed without exact caption mapping: {summary.evidence_missing}")
    print(f"JSON errors: {summary.json_errors}")
    print(f"Doc not object: {summary.doc_not_object}")
    print(f"Detections not list: {summary.detections_not_list}")
    print(f"Detection item not object: {summary.detection_not_object}")

    print_counter("Detail counts", detail)
    if changed_task_counts and args.show_task_examples > 0:
        print_counter("Changed task examples", changed_task_counts, limit=args.show_task_examples)


if __name__ == "__main__":
    main()
