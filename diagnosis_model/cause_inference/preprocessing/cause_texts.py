from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List


def collect_cause_strings_from_coco(
    coco_paths: Iterable[str | Path],
    skip_healthy: bool = True,
    verbose: bool = True,
) -> List[str]:
    """Collect a deduplicated list of non-empty cause strings from COCO files."""
    all_strings: List[str] = []
    seen: set[str] = set()
    n_images_kept = 0
    n_images_skipped = 0
    paths = [Path(p) for p in coco_paths]

    for coco_path in paths:
        if not coco_path.exists():
            print(f"[WARN] COCO file not found, skipping: {coco_path}", file=sys.stderr)
            continue

        with coco_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        for img in coco.get("images", []):
            if skip_healthy and bool(img.get("isHealthy", False)):
                n_images_skipped += 1
                continue

            causes = img.get("global_causes_zh")
            if not causes:
                n_images_skipped += 1
                continue

            n_images_kept += 1
            for item in causes:
                if not isinstance(item, str):
                    continue
                text = item.strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                all_strings.append(text)

    if verbose:
        print(
            f"[collect] images kept={n_images_kept}, skipped={n_images_skipped}, "
            f"unique cause strings={len(all_strings)}, from {len(paths)} coco file(s)"
        )

    return all_strings

