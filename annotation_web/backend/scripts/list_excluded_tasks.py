#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pydantic import ValidationError

# Ensure project root is importable when running as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Reuse backend services and models
from backend.app.config import get_settings, Settings
from backend.app.services import datasets as datasets_service
from backend.app.services import storage as storage_service
from backend.app.models import TaskDocument


def scan_dataset(dataset: str, settings: Settings) -> Tuple[int, int, List[Tuple[str, str]]]:
    """Return (total_json, included_count, excluded_list[(stem, reason)])"""
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    json_dir = storage_service.get_annotations_dir(dataset_dir)
    if not json_dir.exists():
        return (0, 0, [])

    excluded: List[Tuple[str, str]] = []
    included = 0
    files = sorted(json_dir.glob("*.json"))
    for json_path in files:
        stem = json_path.stem
        # Parse JSON
        try:
            raw: Dict = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            excluded.append((stem, f"invalid_json: {type(e).__name__}"))
            continue

        # Locate image
        try:
            image_filename, _is_healthy = storage_service.find_image_filename(
                dataset, dataset_dir, stem, settings, is_healthy=None
            )
        except Exception:
            excluded.append((stem, "missing_image"))
            continue

        # Validate payload against TaskDocument using same normalization
        try:
            _ = storage_service._normalize_task(  # type: ignore[attr-defined]
                raw=raw, dataset=dataset, stem=stem, image_filename=image_filename
            )
        except ValidationError as e:
            # Summarize first few validation messages for clarity
            msgs = []
            for err in e.errors()[:3]:
                loc = ".".join(str(x) for x in err.get("loc", []))
                msg = err.get("msg", "validation error")
                if loc:
                    msgs.append(f"{loc}: {msg}")
                else:
                    msgs.append(msg)
            reason = "; ".join(msgs) if msgs else "ValidationError"
            excluded.append((stem, f"validation_error: {reason}"))
            continue
        except Exception as e:
            excluded.append((stem, f"validation_error: {type(e).__name__}"))
            continue

        included += 1

    return (len(files), included, excluded)


def main() -> None:
    parser = argparse.ArgumentParser(description="List tasks excluded from total count.")
    parser.add_argument(
        "--dataset",
        "-d",
        help="Dataset name to scan (default: all)",
        default=None,
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    args = parser.parse_args()

    settings = get_settings()

    datasets: List[str]
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = datasets_service.list_datasets(settings)

    results = []
    for ds in datasets:
        total_json, included, excluded = scan_dataset(ds, settings)
        results.append(
            {
                "dataset": ds,
                "total_json_files": total_json,
                "included_in_total": included,
                "excluded_count": len(excluded),
                "excluded_items": [
                    {"task_id": stem, "reason": reason} for stem, reason in excluded
                ],
            }
        )

    if args.format == "json":
        print(json.dumps({"datasets": results}, ensure_ascii=False, indent=2))
        return

    # Human-readable text output
    for r in results:
        print(f"Dataset: {r['dataset']}")
        print(f"  JSON files:        {r['total_json_files']}")
        print(f"  Included in total: {r['included_in_total']}")
        print(f"  Excluded count:    {r['excluded_count']}")
        if r["excluded_items"]:
            print("  Excluded items:")
            for item in r["excluded_items"]:
                print(f"    - {item['task_id']}  [{item['reason']}]")
        print()


if __name__ == "__main__":
    main()
