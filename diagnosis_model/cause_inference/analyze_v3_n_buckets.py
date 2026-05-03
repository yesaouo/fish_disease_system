"""Analyze CEAH v3 retrieval + attribution behavior bucketed by lesion count.

Reads per_query.jsonl from v3 eval (γ=0.5 hybrid), buckets queries into
N=1, N=2, N>=3, and reports:
  - Retrieval: sem_MRR, sem_R@K, sem_coverage per bucket
  - Attribution: mean global/text/lesion α per bucket
  - Lesion concentration: max-lesion-α / sum-lesion-α (= 1 if all on one lesion,
    = 1/N if uniform)
  - Cause-type agreement: % of lesion-type predictions where lesion-sum α > global α
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


GLOBAL_KEYS = ["水質", "緊迫", "營養", "環境", "水溫", "免疫力", "應激"]
LESION_KEYS = ["潰瘍", "出血", "紅腫", "寄生蟲", "棉絮", "創傷", "腫脹", "黴", "結痂", "撕裂", "蛀"]


def classify(t: str) -> str:
    g = sum(1 for k in GLOBAL_KEYS if k in t)
    l = sum(1 for k in LESION_KEYS if k in t)
    if g > l: return "global"
    if l > g: return "lesion"
    return "mixed"


def bucket_of(n: int) -> str:
    if n == 1: return "N=1"
    if n == 2: return "N=2"
    return "N>=3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=str, required=True)
    ap.add_argument("--case_db", type=str, required=True)
    ap.add_argument("--gamma_tag", type=str, default="g=0.50")
    args = ap.parse_args()

    cause_pack = torch.load(Path(args.case_db) / "cause_text_embs.pt", weights_only=False)
    texts = cause_pack["texts"]
    with open(Path(args.eval_dir) / "per_query.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    # ---- 1. Retrieval per bucket ----
    sem_ranks_b: dict = defaultdict(list)
    pool_b: dict = defaultdict(list)
    cov_b: dict = defaultdict(list)
    for r in rows:
        b = bucket_of(int(r["lesion_count"]))
        pool = int(r["pool_size"])
        pool_b[b].append(pool)
        for sr in r["ranks_per_gamma"][args.gamma_tag]["sem_ranks"]:
            sem_ranks_b[b].append(int(sr))
            cov_b[b].append(1 if int(sr) <= pool else 0)

    print("=" * 100)
    print("RETRIEVAL by lesion bucket  (γ=0.5 hybrid, v3 CEAH)")
    print("=" * 100)
    print(f'{"bucket":<10}{"queries":>10}{"GTs":>8}{"pool":>8}{"cov":>8}'
          f'{"MRR":>8}{"R@1":>8}{"R@5":>8}{"R@10":>8}{"R@20":>8}{"med_rank":>10}')
    for b in ["N=1", "N=2", "N>=3"]:
        if not sem_ranks_b[b]: continue
        arr = np.array(sem_ranks_b[b], dtype=np.float64)
        n_q = sum(1 for r in rows if bucket_of(int(r["lesion_count"])) == b)
        cov = np.mean(cov_b[b])
        pool = np.mean(pool_b[b])
        mrr = float((1.0 / arr).mean())
        med = float(np.median(arr))
        print(
            f'{b:<10}{n_q:>10}{int(arr.size):>8}{pool:>8.1f}{cov:>8.3f}'
            f'{mrr:>8.4f}{(arr<=1).mean():>8.4f}{(arr<=5).mean():>8.4f}'
            f'{(arr<=10).mean():>8.4f}{(arr<=20).mean():>8.4f}{med:>10.0f}'
        )

    # ---- 2. Attribution distribution per bucket (top-1 prediction) ----
    print()
    print("=" * 100)
    print("ATTRIBUTION (top-1 prediction's α) by lesion bucket")
    print("=" * 100)
    print(f'{"bucket":<10}{"n":>6}{"global α":>12}{"text α":>10}'
          f'{"lesion sum":>12}{"lesion max":>12}{"concentration":>15}')

    bucket_alpha = defaultdict(list)
    for r in rows:
        if not r["predicted_top_n"]: continue
        n = r["lesion_count"]
        if n == 0: continue
        b = bucket_of(n)
        a = r["predicted_top_n"][0]["alpha"]
        g, t = a[0], a[1]
        les_a = a[2:2 + n]
        ls = sum(les_a)
        lm = max(les_a)
        # concentration: max / sum (ignore very small denominators)
        conc = (lm / ls) if ls > 1e-6 else 0.0
        bucket_alpha[b].append((g, t, ls, lm, conc))

    for b in ["N=1", "N=2", "N>=3"]:
        if not bucket_alpha[b]: continue
        arr = np.array(bucket_alpha[b])
        print(
            f'{b:<10}{len(arr):>6}{arr[:, 0].mean():>12.3f}{arr[:, 1].mean():>10.3f}'
            f'{arr[:, 2].mean():>12.3f}{arr[:, 3].mean():>12.3f}{arr[:, 4].mean():>15.3f}'
        )

    # ---- 3. Cause-type agreement: where does α land for each cause type, per bucket? ----
    print()
    print("=" * 100)
    print("ATTRIBUTION × cause type × bucket  (mean α, lesion-sum / global)")
    print("=" * 100)
    print(f'{"bucket":<10}{"cause-type":<14}{"n":>5}{"global":>10}'
          f'{"text":>8}{"lesion sum":>12}{"l/g ratio":>12}')

    table = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not r["predicted_top_n"]: continue
        n = r["lesion_count"]
        if n == 0: continue
        b = bucket_of(n)
        p = r["predicted_top_n"][0]
        a = p["alpha"]
        ctype = classify(texts[p["cause_idx"]])
        ls = sum(a[2:2 + n])
        table[b][ctype].append((a[0], a[1], ls))

    for b in ["N=1", "N=2", "N>=3"]:
        for ctype in ["global", "lesion", "mixed"]:
            arr = table[b].get(ctype, [])
            if not arr: continue
            arr = np.array(arr)
            g_m = arr[:, 0].mean()
            t_m = arr[:, 1].mean()
            ls_m = arr[:, 2].mean()
            ratio = ls_m / max(g_m, 1e-9)
            print(f'{b:<10}{ctype:<14}{len(arr):>5}{g_m:>10.3f}'
                  f'{t_m:>8.3f}{ls_m:>12.3f}{ratio:>12.3f}')
        print()

    # ---- 4. Lesion winner asymmetry: in N>=2, does the model concentrate or spread? ----
    print("=" * 100)
    print("LESION WINNER ASYMMETRY (N>=2 only)  — fraction of α on the top lesion")
    print("=" * 100)

    for b in ["N=2", "N>=3"]:
        if not bucket_alpha[b]: continue
        arr = np.array(bucket_alpha[b])
        concentrations = arr[:, 4]  # max/sum
        n_uniform = (concentrations < 0.55).sum()
        n_concentrated = (concentrations > 0.75).sum()
        n_total = len(concentrations)
        print(f'{b:<10}n={n_total}  '
              f'mean_concentration={concentrations.mean():.3f}  '
              f'concentrated (>0.75): {n_concentrated} ({100*n_concentrated/n_total:.1f}%)  '
              f'uniform (<0.55): {n_uniform} ({100*n_uniform/n_total:.1f}%)')


if __name__ == "__main__":
    main()
