"""Slice Phase 1 sweep results by lesion count to see if lesion-set retrieval
matters more in multi-lesion cases.

Reads per_query_results.jsonl from each sweep output dir, buckets queries by
query_lesion_count (1, 2, ≥3), and prints per-bucket semantic metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def bucket_of(n: int) -> str:
    if n == 1:
        return "N=1"
    if n == 2:
        return "N=2"
    return "N>=3"


def aggregate_per_bucket(rows: list, ks=(1, 5, 10, 20)) -> Dict[str, dict]:
    """Returns {bucket: {n_queries, n_gt, MRR, R@K, coverage}}.

    semantic_ranks <= pool_size means covered.
    """
    by_bucket: Dict[str, list] = defaultdict(list)
    for r in rows:
        b = bucket_of(int(r["query_lesion_count"]))
        by_bucket[b].append(r)

    out: Dict[str, dict] = {}
    for b, br in by_bucket.items():
        sem_ranks: List[int] = []
        cov: List[int] = []
        for r in br:
            pool = int(r["candidate_pool_size"])
            for sr in r["gt_semantic_ranks_in_pool"]:
                sem_ranks.append(int(sr))
                cov.append(1 if int(sr) <= pool else 0)
        if not sem_ranks:
            continue
        arr = np.array(sem_ranks, dtype=np.float64)
        rec = {
            "n_queries": len(br),
            "n_gt": int(arr.size),
            "MRR": float((1.0 / arr).mean()),
            "median_rank": float(np.median(arr)),
            "coverage": float(np.mean(cov)),
        }
        for k in ks:
            rec[f"R@{k}"] = float((arr <= k).mean())
        out[b] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/phase1_sweep")
    ap.add_argument("--configs", nargs="+", default=[
        "base_K10_a25_d95",
        "K10_a00_d95_lesion",
        "K10_a10_d95_global",
        "K20_a25_d95",
    ])
    args = ap.parse_args()

    sweep = Path(args.sweep_dir)
    results: Dict[str, Dict[str, dict]] = {}
    for cfg in args.configs:
        path = sweep / cfg / "per_query_results.jsonl"
        if not path.exists():
            print(f"[skip] {cfg}: missing {path}")
            continue
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        results[cfg] = aggregate_per_bucket(rows)

    buckets = ["N=1", "N=2", "N>=3"]
    metrics = ["n_queries", "n_gt", "coverage", "MRR", "R@1", "R@5", "R@10", "R@20", "median_rank"]

    for b in buckets:
        print(f"\n=== Bucket {b} ===")
        header = f"{'config':<28}" + "".join(f"{m:>12}" for m in metrics)
        print(header)
        print("-" * len(header))
        for cfg in args.configs:
            if cfg not in results or b not in results[cfg]:
                continue
            row = results[cfg][b]
            cells = []
            for m in metrics:
                v = row[m]
                if m in ("n_queries", "n_gt"):
                    cells.append(f"{int(v):>12}")
                else:
                    cells.append(f"{v:>12.4f}")
            print(f"{cfg:<28}" + "".join(cells))


if __name__ == "__main__":
    main()
