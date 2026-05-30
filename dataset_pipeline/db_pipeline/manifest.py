from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_sha() -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return sha or None
    except Exception:
        return None


def write_manifest(
    path: Path,
    *,
    version: str,
    command: str,
    sources: list[dict],
    symptoms_sha: str,
    labels_sha: str | None,
    detection_label_mode: str,
    split_ratios: list,
    strict_categories: bool,
    require_submit: bool,
    require_expert_submit: bool,
    filter_stats,
    per_split_counts: dict,
    per_dataset_counts: dict,
    new_image_ids_range: tuple[int, int] | None,
    total_images_in_registry: int,
) -> dict:
    manifest = {
        "version": version,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "git_sha": get_git_sha(),
        "command": command,
        "sources": sources,
        "config": {
            "symptoms_json_sha256": symptoms_sha,
            "labels_txt_sha256": labels_sha,
            "detection_label_mode": detection_label_mode,
            "split_ratios": split_ratios,
            "strict_categories": strict_categories,
            "require_submit": require_submit,
            "require_expert_submit": require_expert_submit,
        },
        "stats": {
            "dropped_by_comments": filter_stats.dropped_by_comments,
            "dropped_by_submit_filter": filter_stats.dropped_by_submit_filter,
            "dropped_by_missing_image": filter_stats.dropped_by_missing_image,
            "used_labels": [
                {"label": k, "count": v}
                for k, v in sorted(
                    filter_stats.used_labels.items(),
                    key=lambda x: (-x[1], x[0]),
                )
            ],
            "skipped_bboxes_unknown_label": [
                {"label": k, "count": v}
                for k, v in sorted(
                    filter_stats.skipped_bboxes_unknown_label.items(),
                    key=lambda x: (-x[1], x[0]),
                )
            ],
            "kept_tasks_per_split": dict(per_split_counts),
            "kept_tasks_per_dataset": dict(per_dataset_counts),
            "new_image_ids_this_run": list(new_image_ids_range) if new_image_ids_range else None,
            "total_images_in_registry": total_images_in_registry,
        },
        "views": ["full", "detection"],
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def write_image_index(path: Path, version: str, tasks_per_split: dict) -> None:
    payload = {
        "version": version,
        "splits": {
            split: sorted(task.image_id for task in tasks)
            for split, tasks in tasks_per_split.items()
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_category_diff(
    path: Path,
    unknown_in_data: Counter,
    defined_names: set[str],
    used_names: Counter,
) -> None:
    used = set(used_names.keys()) | set(unknown_in_data.keys())
    unused = sorted(defined_names - used)
    unknown = sorted(unknown_in_data.keys())
    used_sorted = sorted(used_names.items(), key=lambda x: (-x[1], x[0]))

    lines: list[str] = []
    lines.append("# Category diff report")
    lines.append("")
    lines.append(f"## Used in data (defined in symptoms.json) ({len(used_sorted)})")
    for name, count in used_sorted:
        lines.append(f"  - {name}  (count={count})")
    lines.append("")
    lines.append(f"## Defined in symptoms.json but never used in data ({len(unused)})")
    for name in unused:
        lines.append(f"  - {name}")
    lines.append("")
    lines.append(f"## Used in data but not defined in symptoms.json ({len(unknown)})")
    for name in unknown:
        lines.append(f"  - {name}  (count={unknown_in_data[name]})")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
