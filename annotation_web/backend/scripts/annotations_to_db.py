#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is importable when running as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.app.config import get_settings
from backend.app.services import datasets as datasets_service
from backend.app.services import storage as storage_service


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert legacy annotations/*.json into per-dataset SQLite DB.")
    parser.add_argument("--dataset", "-d", required=True, help="Dataset folder name under DATA_ROOT.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing DB file if present.",
    )
    args = parser.parse_args()

    settings = get_settings()
    dataset = str(args.dataset).strip()
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    annotations_dir = storage_service.get_annotations_dir(dataset_dir)
    if not annotations_dir.exists():
        raise SystemExit(f"Missing annotations dir: {annotations_dir}")

    db_path = storage_service.get_db_path(dataset_dir, settings)
    if db_path.exists():
        if not args.overwrite:
            raise SystemExit(f"DB already exists: {db_path} (use --overwrite)")
        db_path.unlink()

    storage_service.ensure_db(dataset_dir, settings)

    files = sorted(annotations_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found under: {annotations_dir}")

    imported = 0
    skipped = 0
    errors: list[str] = []

    for idx, json_path in enumerate(files, start=1):
        stem = json_path.stem
        try:
            raw = _load_json(json_path)
            # Image location is determined by filesystem (images/ vs healthy_images/).
            image_filename, _ = storage_service.find_image_filename(
                dataset, dataset_dir, stem, settings, is_healthy=None
            )

            doc = storage_service._normalize_task(  # type: ignore[attr-defined]
                raw=raw,
                dataset=dataset,
                stem=stem,
                image_filename=image_filename,
                dataset_dir=dataset_dir,
                settings=settings,
            )
            storage_service.upsert_task(
                dataset,
                stem,
                sort_index=idx,
                document=doc,
                updated_by="system",
                action="import",
                settings=settings,
            )
            imported += 1
        except Exception as exc:
            skipped += 1
            errors.append(f"{stem}: {type(exc).__name__}: {exc}")

    print(f"DB: {db_path}")
    print(f"Imported: {imported}")
    print(f"Skipped:  {skipped}")
    if errors:
        print("\nErrors:")
        for line in errors[:50]:
            print(f"  - {line}")
        if len(errors) > 50:
            print(f"  ... ({len(errors) - 50} more)")


if __name__ == "__main__":
    main()
