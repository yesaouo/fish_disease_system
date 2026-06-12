"""Shared helpers for the build_pipeline orchestrators."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SPLITS = ["train", "valid", "test"]


def run_step(name: str, module: str, args, dry_run: bool = False):
    """Invoke `python -m <module> <args>` as a subprocess (check=True)."""
    argv = [sys.executable, "-m", module, *map(str, args)]
    print(f"\n===== [{name}] {' '.join(argv[2:])} =====", flush=True)
    if dry_run:
        return
    subprocess.run(argv, check=True)


def detection_splits(det_root) -> list:
    """Splits whose detection COCO exists (so new-data `test` is picked up, old isn't)."""
    det_root = Path(det_root)
    return [s for s in SPLITS if (det_root / s / "_annotations.coco.json").exists()]


def casedb_splits(case_db_dir) -> list:
    """Splits present in a case_db (whatever build_raw produced)."""
    case_db_dir = Path(case_db_dir)
    return [s for s in SPLITS if (case_db_dir / f"{s}_cases.pt").exists()]
