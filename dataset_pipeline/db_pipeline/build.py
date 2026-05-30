"""Build a versioned dataset snapshot from live annotation_web SQLite DBs.

Run from repo root:
    python -m dataset_pipeline.db_pipeline.build [options]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from .manifest import (
    sha256_file,
    write_category_diff,
    write_image_index,
    write_manifest,
)
from .registry import ImageRegistry
from .sources import find_datasets, load_tasks


def _load_symptoms(path: Path) -> tuple[list[dict], dict[str, int]]:
    """Return (coco_categories, name -> category_id)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    label_map = data.get("label_map") or {}
    categories: list[dict] = []
    name_to_id: dict[str, int] = {}
    for key_id, info in label_map.items():
        cat_id = int(key_id)
        name = info.get("en") or "unknown"
        categories.append(
            {"id": cat_id, "name": name, "supercategory": info.get("zh") or "unknown"}
        )
        name_to_id[name] = cat_id
    categories.sort(key=lambda c: c["id"])
    return categories, name_to_id


def _load_labels_txt(path: Path) -> tuple[dict[int, int], list[dict]]:
    """Re-index labels.txt per process_coco.py: dedup names, assign sequential new IDs from 0.

    Returns (orig_cat_id -> new_id, new_categories_list).
    """
    entries: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        try:
            entries.append((int(parts[0]), parts[1].strip()))
        except ValueError:
            continue
    entries.sort(key=lambda x: x[0])

    name_to_new_id: dict[str, int] = {}
    id_map: dict[int, int] = {}
    new_categories: list[dict] = []
    for orig_id, name in entries:
        if name not in name_to_new_id:
            new_id = len(new_categories)
            name_to_new_id[name] = new_id
            new_categories.append({"id": new_id, "name": name, "supercategory": name})
        id_map[orig_id] = name_to_new_id[name]
    return id_map, new_categories


def _default_detection_labels(full_categories: list[dict]) -> tuple[dict[int, int], list[dict]]:
    """Default detection view: drop category id 0 and merge all other ids as ABNORMAL."""
    id_map: dict[int, int] = {}
    for category in full_categories:
        cat_id = int(category["id"])
        if cat_id == 0:
            continue
        id_map[cat_id] = 0
    categories = [{"id": 0, "name": "ABNORMAL", "supercategory": "ABNORMAL"}]
    return id_map, categories


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=Path("data/processed"))
    p.add_argument("--annotation-root", type=Path, default=Path("data/annotation"))
    p.add_argument("--symptoms-json", type=Path, default=Path("data/annotation/symptoms.json"))
    p.add_argument(
        "--labels-txt",
        type=Path,
        default=None,
        help=(
            "Optional labels.txt override. By default, detection view filters category id 0 "
            "and maps every other symptoms.json category id to ABNORMAL."
        ),
    )
    p.add_argument("--version-tag", default=None)
    p.add_argument("--split-ratios", type=float, nargs=3, default=[8, 1, 1], metavar=("TRAIN", "VALID", "TEST"))
    p.add_argument("--strict-categories", action="store_true")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--require-submit", action="store_true")
    grp.add_argument("--require-expert-submit", action="store_true")
    return p.parse_args()


def _update_current_symlink(output_root: Path, version: str) -> None:
    current = output_root / "current"
    if current.is_symlink() or current.exists():
        current.unlink()
    current.symlink_to(version)


def _print_summary(
    *,
    version_dir: Path,
    manifest: dict,
    new_image_ids_range: tuple[int, int] | None,
    aborted: bool = False,
) -> None:
    s = manifest["stats"]
    print()
    print(f"version_dir : {version_dir}")
    print(f"git_sha     : {manifest['git_sha']}")
    print(f"generated_at: {manifest['generated_at']}")
    print()
    print("Filter stats:")
    print(f"  dropped_by_comments       : {s['dropped_by_comments']}")
    print(f"  dropped_by_submit_filter  : {s['dropped_by_submit_filter']}")
    print(f"  dropped_by_missing_image  : {s['dropped_by_missing_image']}")
    if s["skipped_bboxes_unknown_label"]:
        total_skipped = sum(x["count"] for x in s["skipped_bboxes_unknown_label"])
        print(f"  skipped_bboxes_unknown_label: {total_skipped} (see category_diff.txt)")
    print()
    print("Kept tasks per split:")
    for k, v in s["kept_tasks_per_split"].items():
        print(f"  {k:6s}: {v}")
    print()
    print("Kept tasks per dataset:")
    for k, v in s["kept_tasks_per_dataset"].items():
        print(f"  {k}: {v}")
    print()
    if new_image_ids_range:
        print(f"new image IDs this run: {new_image_ids_range[0]}..{new_image_ids_range[1]}")
    print(f"total images in registry: {s['total_images_in_registry']}")
    if aborted:
        print()
        print("ABORTED due to --strict-categories with unknown labels.")


def main() -> int:
    args = _parse_args()

    if not args.symptoms_json.is_file():
        sys.exit(f"symptoms.json not found: {args.symptoms_json}")
    if args.labels_txt is not None and not args.labels_txt.is_file():
        sys.exit(f"labels.txt not found: {args.labels_txt}")
    if not args.annotation_root.is_dir():
        sys.exit(f"annotation root not found: {args.annotation_root}")

    full_categories, name_to_cat_id = _load_symptoms(args.symptoms_json)
    if args.labels_txt is not None:
        labels_id_map, detection_categories = _load_labels_txt(args.labels_txt)
        labels_sha = sha256_file(args.labels_txt)
        detection_label_mode = "labels_txt"
    else:
        labels_id_map, detection_categories = _default_detection_labels(full_categories)
        labels_sha = None
        detection_label_mode = "auto_abnormal_except_id0"

    sources = find_datasets(args.annotation_root)
    if not sources:
        sys.exit(f"no datasets with annotations.db under {args.annotation_root}")

    version = args.version_tag or datetime.now().strftime("%Y-%m-%d_%H%M")
    output_root: Path = args.output
    version_dir = output_root / version
    if version_dir.exists():
        sys.exit(f"version dir already exists: {version_dir}")

    tasks, stats = load_tasks(
        sources,
        name_to_cat_id,
        require_submit=args.require_submit,
        require_expert_submit=args.require_expert_submit,
        progress=tqdm,
    )

    if args.strict_categories and stats.skipped_bboxes_unknown_label:
        version_dir.mkdir(parents=True, exist_ok=False)
        write_category_diff(
            version_dir / "category_diff.txt",
            stats.skipped_bboxes_unknown_label,
            set(name_to_cat_id.keys()),
            stats.used_labels,
        )
        manifest = write_manifest(
            version_dir / "MANIFEST.json",
            version=version,
            command=" ".join(sys.argv),
            sources=[],
            symptoms_sha=sha256_file(args.symptoms_json),
            labels_sha=labels_sha,
            detection_label_mode=detection_label_mode,
            split_ratios=args.split_ratios,
            strict_categories=True,
            require_submit=args.require_submit,
            require_expert_submit=args.require_expert_submit,
            filter_stats=stats,
            per_split_counts={},
            per_dataset_counts={},
            new_image_ids_range=None,
            total_images_in_registry=0,
        )
        _print_summary(version_dir=version_dir, manifest=manifest, new_image_ids_range=None, aborted=True)
        return 1

    version_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.symptoms_json, version_dir / "symptoms.json")
    if args.labels_txt is not None:
        shutil.copy2(args.labels_txt, version_dir / "labels.txt")

    registry = ImageRegistry.load(output_root / "image_registry.json")
    first_new_id: int | None = None
    last_new_id: int | None = None
    for task in tqdm(tasks, desc="register", leave=False):
        img_id, is_new = registry.get_or_create(
            task.dataset,
            task.image_filename,
            task.task_id,
            tuple(args.split_ratios),
        )
        task.image_id = img_id
        task.split = registry.split_of(img_id)
        if is_new:
            if first_new_id is None:
                first_new_id = img_id
            last_new_id = img_id

    tasks_per_split: dict[str, list] = defaultdict(list)
    for task in tasks:
        tasks_per_split[task.split].append(task)
    for k in ("train", "valid", "test"):
        tasks_per_split.setdefault(k, [])

    from .views import write_views
    write_views(
        version_dir,
        tasks_per_split,
        name_to_cat_id,
        full_categories,
        labels_id_map,
        detection_categories,
        progress=tqdm,
    )

    registry.save()
    write_image_index(version_dir / "image_index.json", version, tasks_per_split)

    if stats.used_labels or stats.skipped_bboxes_unknown_label:
        write_category_diff(
            version_dir / "category_diff.txt",
            stats.skipped_bboxes_unknown_label,
            set(name_to_cat_id.keys()),
            stats.used_labels,
        )

    per_split_counts = {k: len(v) for k, v in tasks_per_split.items()}
    per_dataset_counts: Counter = Counter()
    for task in tasks:
        per_dataset_counts[task.dataset] += 1

    sources_meta = [
        {
            "dataset": src.name,
            "db_path": str(src.db_path),
            "db_sha256": sha256_file(src.db_path),
            "db_mtime": datetime.fromtimestamp(src.db_path.stat().st_mtime).astimezone().isoformat(),
        }
        for src in sources
    ]

    new_range = (first_new_id, last_new_id) if first_new_id is not None else None
    manifest = write_manifest(
        version_dir / "MANIFEST.json",
        version=version,
        command=" ".join(sys.argv),
        sources=sources_meta,
        symptoms_sha=sha256_file(args.symptoms_json),
        labels_sha=labels_sha,
        detection_label_mode=detection_label_mode,
        split_ratios=args.split_ratios,
        strict_categories=args.strict_categories,
        require_submit=args.require_submit,
        require_expert_submit=args.require_expert_submit,
        filter_stats=stats,
        per_split_counts=per_split_counts,
        per_dataset_counts=dict(per_dataset_counts),
        new_image_ids_range=new_range,
        total_images_in_registry=len(registry.images),
    )

    _update_current_symlink(output_root, version)
    _print_summary(version_dir=version_dir, manifest=manifest, new_image_ids_range=new_range)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
