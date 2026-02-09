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
    tasks_with_issues: int = 0
    changed_tasks: int = 0
    changed_detections: int = 0
    json_decode_errors: int = 0


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


def _parse_evidence_index(value: Any) -> tuple[str, int | None]:
    """Return (status, parsed_value). status in: none, ok, not_integer, negative."""
    if value is None or value == "":
        return ("none", None)
    if isinstance(value, bool):
        return ("not_integer", None)
    try:
        idx = int(value)
    except Exception:
        return ("not_integer", None)
    if idx < 0:
        return ("negative", idx)
    return ("ok", idx)


def _analyze_and_clean_detection(
    det: dict[str, Any],
    *,
    allowed_labels: set[str],
    evidence_options: dict[str, list[str]],
    do_clean: bool,
    issue_counts: Counter[str],
    fix_counts: Counter[str],
) -> tuple[bool, bool]:
    """
    Returns: (has_issue, changed)
    """
    has_issue = False
    changed = False

    before = json.dumps(det, ensure_ascii=False, sort_keys=True)

    # Evaluate current values first (for issue counting).
    label_key_exists = "label" in det
    raw_label = det.get("label", "")
    label = str(raw_label or "").strip()
    label_is_healthy = label == storage_service.HEALTHY_LABEL
    label_is_blank = label == ""
    label_is_valid = (label in allowed_labels) or label_is_healthy

    if label_is_blank:
        issue_counts["detections.label.missing_or_blank"] += 1
        has_issue = True
    elif not label_is_valid:
        issue_counts["detections.label.not_in_classes"] += 1
        has_issue = True

    raw_idx = det.get("evidence_index")
    idx_status, idx_value = _parse_evidence_index(raw_idx)
    if idx_status == "not_integer":
        issue_counts["detections.evidence_index.not_integer"] += 1
        has_issue = True
    elif idx_status == "negative":
        issue_counts["detections.evidence_index.negative"] += 1
        has_issue = True
    elif idx_status == "ok":
        if label_is_blank or (not label_is_valid):
            issue_counts["detections.evidence_index.without_valid_label"] += 1
            has_issue = True
        elif not label_is_healthy:
            options = evidence_options.get(label) or []
            if not options:
                issue_counts["detections.evidence_index.no_options_for_label"] += 1
                has_issue = True
            elif idx_value is not None and idx_value >= len(options):
                issue_counts["detections.evidence_index.out_of_range"] += 1
                has_issue = True

    if do_clean:
        # Keep label key stable for downstream model compatibility.
        if not label_key_exists:
            det["label"] = ""
            fix_counts["fixes.detections.label.set_empty_for_missing"] += 1
        else:
            normalized_label = label
            if not label_is_valid:
                normalized_label = ""
                fix_counts["fixes.detections.label.cleared_invalid"] += 1
            elif str(raw_label or "") != label:
                fix_counts["fixes.detections.label.trimmed"] += 1
            det["label"] = normalized_label

        # Re-check with cleaned label.
        cleaned_label = str(det.get("label") or "").strip()
        cleaned_label_valid = (cleaned_label in allowed_labels) or (cleaned_label == storage_service.HEALTHY_LABEL)

        remove_idx = False
        if cleaned_label == "" and ("evidence_index" in det):
            remove_idx = True
            fix_counts["fixes.detections.evidence_index.removed_due_to_empty_label"] += 1
        elif idx_status in ("not_integer", "negative"):
            remove_idx = True
            fix_counts["fixes.detections.evidence_index.removed_invalid_type_or_negative"] += 1
        elif idx_status == "ok":
            if cleaned_label == "" or (not cleaned_label_valid):
                remove_idx = True
                fix_counts["fixes.detections.evidence_index.removed_without_valid_label"] += 1
            elif cleaned_label != storage_service.HEALTHY_LABEL:
                options = evidence_options.get(cleaned_label) or []
                if not options:
                    remove_idx = True
                    fix_counts["fixes.detections.evidence_index.removed_no_options_for_label"] += 1
                elif idx_value is not None and idx_value >= len(options):
                    remove_idx = True
                    fix_counts["fixes.detections.evidence_index.removed_out_of_range"] += 1
                else:
                    det["evidence_index"] = int(idx_value) if idx_value is not None else None
                    if raw_idx != det["evidence_index"]:
                        fix_counts["fixes.detections.evidence_index.normalized_integer"] += 1

        if remove_idx:
            if "evidence_index" in det:
                det.pop("evidence_index", None)

    after = json.dumps(det, ensure_ascii=False, sort_keys=True)
    if before != after:
        changed = True

    return has_issue, changed


def _analyze_and_maybe_clean_doc(
    raw: dict[str, Any],
    *,
    allowed_labels: set[str],
    evidence_options: dict[str, list[str]],
    do_clean: bool,
    issue_counts: Counter[str],
    fix_counts: Counter[str],
) -> tuple[bool, bool, int]:
    """
    Returns: (has_issue, changed, changed_detection_count)
    """
    has_issue = False
    changed = False
    changed_detection_count = 0

    dets = raw.get("detections")
    if not isinstance(dets, list):
        issue_counts["detections.not_list"] += 1
        has_issue = True
        if do_clean:
            raw["detections"] = []
            changed = True
            fix_counts["fixes.detections.reset_to_empty_list"] += 1
        return has_issue, changed, changed_detection_count

    cleaned_dets: list[Any] = []
    removed_non_object = 0
    for det in dets:
        if not isinstance(det, dict):
            issue_counts["detections.item_not_object"] += 1
            has_issue = True
            if do_clean:
                removed_non_object += 1
                continue
            cleaned_dets.append(det)
            continue

        det_has_issue, det_changed = _analyze_and_clean_detection(
            det,
            allowed_labels=allowed_labels,
            evidence_options=evidence_options,
            do_clean=do_clean,
            issue_counts=issue_counts,
            fix_counts=fix_counts,
        )
        if det_has_issue:
            has_issue = True
        if det_changed:
            changed = True
            changed_detection_count += 1
        cleaned_dets.append(det)

    if do_clean and removed_non_object > 0:
        raw["detections"] = cleaned_dets
        changed = True
        fix_counts["fixes.detections.removed_non_object"] += removed_non_object

    return has_issue, changed, changed_detection_count


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
            "Check or clean invalid annotation fields in dataset DB tasks. "
            "Rules are based on dataset symptoms.json classes and evidence options."
        )
    )
    parser.add_argument("--dataset", "-d", required=True, help="Dataset folder name under DATA_ROOT.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["check", "clean"],
        help="check: only report issues; clean: apply cleanup and write DB.",
    )
    parser.add_argument(
        "--allow-empty-classes",
        action="store_true",
        help="Allow running clean mode when classes list is empty.",
    )
    parser.add_argument(
        "--updated-by",
        default="system_cleanup",
        help="history.updated_by value in clean mode.",
    )
    parser.add_argument(
        "--action-name",
        default="clean_invalid_fields",
        help="history.action value in clean mode.",
    )
    parser.add_argument(
        "--show-task-examples",
        type=int,
        default=10,
        help="Max number of task IDs with issues to print.",
    )
    args = parser.parse_args()

    settings = get_settings()
    dataset = str(args.dataset).strip()
    mode = str(args.mode).strip().lower()
    do_clean = mode == "clean"

    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    db_path = storage_service.ensure_db(dataset_dir, settings)

    classes = datasets_service.load_classes(dataset, settings)
    evidence_options = datasets_service.load_evidence_options_zh(dataset, settings)
    allowed_labels = set(classes)

    if do_clean and (not allowed_labels) and (not args.allow_empty_classes):
        raise SystemExit(
            "Refusing to run clean mode: classes is empty. "
            "Use --allow-empty-classes to override."
        )

    rows = storage_service.list_task_rows(dataset, settings)
    summary = RunSummary(scanned_tasks=len(rows))
    issue_counts: Counter[str] = Counter()
    fix_counts: Counter[str] = Counter()
    task_issue_counter: Counter[str] = Counter()

    updates: list[tuple[str, str, str, int, int, str, str]] = []
    # tuple: (task_id, updated_at, updated_by, is_healthy, comments_count, old_json, new_json)

    for row in rows:
        task_id = str(row["task_id"])
        old_json = str(row["doc_json"] or "")

        try:
            raw = json.loads(old_json)
        except Exception:
            summary.json_decode_errors += 1
            summary.tasks_with_issues += 1
            issue_counts["doc_json.invalid_json"] += 1
            task_issue_counter[task_id] += 1
            continue

        if not isinstance(raw, dict):
            summary.tasks_with_issues += 1
            issue_counts["doc_json.not_object"] += 1
            task_issue_counter[task_id] += 1
            continue

        before_doc = json.dumps(raw, ensure_ascii=False, sort_keys=True)
        task_issue_before = sum(issue_counts.values())

        has_issue, changed, changed_det_count = _analyze_and_maybe_clean_doc(
            raw,
            allowed_labels=allowed_labels,
            evidence_options=evidence_options,
            do_clean=do_clean,
            issue_counts=issue_counts,
            fix_counts=fix_counts,
        )

        if has_issue:
            summary.tasks_with_issues += 1

        task_issue_after = sum(issue_counts.values())
        if task_issue_after > task_issue_before:
            task_issue_counter[task_id] += task_issue_after - task_issue_before

        if not do_clean:
            continue

        if changed:
            summary.changed_detections += changed_det_count
            updated_at = _now_iso()
            raw["last_modified_at"] = updated_at
            comments = raw.get("comments")
            comments_count = len(comments) if isinstance(comments, list) else 0
            is_healthy = 1 if _derive_is_healthy(raw) else 0
            after_doc = json.dumps(raw, ensure_ascii=False)
            if after_doc != old_json:
                summary.changed_tasks += 1
                updates.append(
                    (
                        task_id,
                        updated_at,
                        str(args.updated_by).strip() or "system_cleanup",
                        is_healthy,
                        comments_count,
                        old_json,
                        after_doc,
                    )
                )

        else:
            # Count document-level change edge case when serialization differs but no detection changed.
            after_doc = json.dumps(raw, ensure_ascii=False, sort_keys=True)
            if before_doc != after_doc:
                summary.changed_tasks += 1

    if do_clean and updates:
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

    print(f"Dataset: {dataset}")
    print(f"Mode:    {mode}")
    print(f"DB:      {db_path}")
    print(f"Classes: {len(classes)}")
    print(f"Rows:    {summary.scanned_tasks}")
    print(f"Tasks with issues: {summary.tasks_with_issues}")
    print(f"JSON decode errors: {summary.json_decode_errors}")

    if do_clean:
        print(f"Changed tasks: {summary.changed_tasks}")
        print(f"Changed detections: {summary.changed_detections}")

    _print_counter("Issue counts", issue_counts)
    if do_clean:
        _print_counter("Fix counts", fix_counts)

    if task_issue_counter:
        limit = max(int(args.show_task_examples), 0)
        if limit > 0:
            print(f"\nTask examples (top {limit} by issue count)")
            for task_id, n_issues in task_issue_counter.most_common(limit):
                print(f"  - {task_id}: {n_issues}")


if __name__ == "__main__":
    main()
