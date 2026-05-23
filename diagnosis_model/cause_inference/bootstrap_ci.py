#!/usr/bin/env python3
"""Paired query-level bootstrap CI between two cause-ranking runs.

Metric unit = GT-cause-occurrence (matches phase1_baseline/eval_ceah). Misses = inf.
Bootstrap unit = query (resample queries w/ replacement; both runs use the SAME
resampled query indices → paired). Δ = B - A.

Usage:
  $PY -m diagnosis_model.cause_inference.bootstrap_ci \
      --a <per_query> --a_kind {phase1,ceah} --a_name NAME \
      --b <per_query> --b_kind {phase1,ceah} --b_name NAME [--tag g=0.00]
"""
import argparse, json
import numpy as np

KS = [1, 5, 10, 20]
B = 10000
SEED = 0
INF = float("inf")


def _ranks(x):
    return np.array([INF if r is None else float(r) for r in x], dtype=np.float64)


def load(path, kind, tag):
    out = {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if kind == "phase1":
            out[int(d["query_image_id"])] = {
                "sem": _ranks(d["gt_semantic_ranks_in_pool"]),
                "cl":  _ranks(d.get("gt_cluster_ranks_in_pool", [])),
            }
        else:  # ceah
            r = d["ranks_per_gamma"][tag]
            out[int(d["case_id"])] = {
                "sem": _ranks(r["sem_ranks"]),
                "cl":  _ranks(r["cluster_ranks"]),
            }
    return out


def metrics(sem, cl):
    m = {}
    for k in KS:
        m[f"sem_R@{k}"] = float((sem <= k).mean()) if sem.size else 0.0
    m["sem_MRR"] = float(np.where(np.isfinite(sem), 1.0 / sem, 0.0).mean()) if sem.size else 0.0
    for k in [1, 10, 20]:
        m[f"cl_R@{k}"] = float((cl <= k).mean()) if cl.size else 0.0
    m["cl_MRR"] = float(np.where(np.isfinite(cl), 1.0 / cl, 0.0).mean()) if cl.size else 0.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--a_kind", required=True)
    ap.add_argument("--b", required=True); ap.add_argument("--b_kind", required=True)
    ap.add_argument("--a_name", default="A"); ap.add_argument("--b_name", default="B")
    ap.add_argument("--tag", default="g=0.00")
    args = ap.parse_args()

    A, Bd = load(args.a, args.a_kind, args.tag), load(args.b, args.b_kind, args.tag)
    ids = sorted(set(A) & set(Bd))
    print(f"# {args.b_name}  −  {args.a_name}   (Δ = B − A)")
    print(f"aligned queries: {len(ids)}  (A={len(A)}, B={len(Bd)})")
    mism = sum(1 for i in ids if len(A[i]["sem"]) != len(Bd[i]["sem"]))
    print(f"queries w/ mismatched sem occurrence count: {mism} (0 ⇒ same pool, clean paired)")

    Asem = [A[i]["sem"] for i in ids]; Acl = [A[i]["cl"] for i in ids]
    Bsem = [Bd[i]["sem"] for i in ids]; Bcl = [Bd[i]["cl"] for i in ids]
    full_a = metrics(np.concatenate(Asem), np.concatenate(Acl))
    full_b = metrics(np.concatenate(Bsem), np.concatenate(Bcl))

    rng = np.random.default_rng(SEED)
    n = len(ids)
    keys = list(full_a.keys())
    diffs = {k: np.empty(B) for k in keys}
    for bi in range(B):
        idx = rng.integers(0, n, n)
        ma = metrics(np.concatenate([Asem[j] for j in idx]), np.concatenate([Acl[j] for j in idx]))
        mb = metrics(np.concatenate([Bsem[j] for j in idx]), np.concatenate([Bcl[j] for j in idx]))
        for k in keys:
            diffs[k][bi] = mb[k] - ma[k]

    print(f"\n{'metric':<10} {args.a_name[:8]:>8} {args.b_name[:8]:>8} {'Δ':>8} {'95% CI':>20} {'p':>8}  sig")
    print("-" * 76)
    for k in keys:
        d = diffs[k]; is_mrr = k.endswith("MRR"); sc = 1.0 if is_mrr else 100.0
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        f = "{:.4f}" if is_mrr else "{:.2f}"; fd = "{:+.4f}" if is_mrr else "{:+.2f}"
        print(f"{k:<10} {f.format(full_a[k]*sc):>8} {f.format(full_b[k]*sc):>8} "
              f"{fd.format((full_b[k]-full_a[k])*sc):>8} "
              f"[{fd.format(lo*sc)}, {fd.format(hi*sc)}]".rjust(20) + f" {p:>8.4f}  {sig}")
    print(f"\nB={B} query-level paired, seed={SEED}. R@K & Δ in pp; MRR raw. ns ⇒ CI crosses 0.")


if __name__ == "__main__":
    main()
