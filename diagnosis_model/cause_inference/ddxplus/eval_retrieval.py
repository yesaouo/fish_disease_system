"""DDXPlus text-only Phase 1 retrieval evaluation.

This script reuses the FaCE-R Phase 1 case retrieval machinery but reports
DDXPlus-specific metrics:

  - strict PATHOLOGY exact R@K / MRR / candidate-pool coverage
  - differential-diagnosis NDCG@K and relevance-mass coverage

It expects a case DB produced by
`diagnosis_model.cause_inference.ddxplus.build_case_database`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.phase1_baseline import (
    MISS_RANK,
    add_recall_at_ks,
    build_candidate_pool,
    compute_case_similarities,
    load_cases,
    load_train_bank,
    load_train_cases_minimal,
    offsets_to_case_ids,
    score_candidates,
    select_positive_top_cases,
    stream_top_k_cases,
    summarize_rank_metric,
)


def first_rank(sorted_indices: Iterable[int], targets: Iterable[int]) -> float:
    target_set = {int(t) for t in targets}
    for pos, idx in enumerate(sorted_indices, start=1):
        if int(idx) in target_set:
            return float(pos)
    return MISS_RANK


def build_ddx_relevance(q: dict, cause_text_to_idx: Dict[str, int],
                        fallback_rank_relevance: bool = True) -> Dict[int, float]:
    rel: Dict[int, float] = {}
    for rank, item in enumerate(q.get("ddx", []) or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name not in cause_text_to_idx:
            continue
        prob = item.get("prob")
        try:
            value = float(prob)
        except Exception:
            value = 1.0 / float(rank + 1) if fallback_rank_relevance else 1.0
        if not math.isfinite(value) or value <= 0:
            continue
        idx = int(cause_text_to_idx[name])
        rel[idx] = max(rel.get(idx, 0.0), value)
    return rel


def dcg(relevances: List[float]) -> float:
    total = 0.0
    for i, rel in enumerate(relevances):
        total += float(rel) / math.log2(i + 2)
    return total


def ndcg_at_k(sorted_indices: Iterable[int], rel_by_idx: Dict[int, float], k: int) -> float | None:
    if not rel_by_idx:
        return None
    top = list(sorted_indices)[:k]
    gains = [rel_by_idx.get(int(idx), 0.0) for idx in top]
    ideal = sorted(rel_by_idx.values(), reverse=True)[:k]
    denom = dcg(ideal)
    if denom <= 0:
        return None
    return dcg(gains) / denom


def relevance_mass_at_k(sorted_indices: Iterable[int], rel_by_idx: Dict[int, float], k: int) -> float | None:
    if not rel_by_idx:
        return None
    denom = sum(rel_by_idx.values())
    if denom <= 0:
        return None
    top = set(int(idx) for idx in list(sorted_indices)[:k])
    return sum(rel for idx, rel in rel_by_idx.items() if idx in top) / denom


def relevance_mass_coverage(candidate_indices: Iterable[int], rel_by_idx: Dict[int, float]) -> float | None:
    if not rel_by_idx:
        return None
    denom = sum(rel_by_idx.values())
    if denom <= 0:
        return None
    pool = {int(idx) for idx in candidate_indices}
    return sum(rel for idx, rel in rel_by_idx.items() if idx in pool) / denom


def mean_optional(values: List[float | None]) -> float | None:
    arr = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not arr:
        return None
    return float(np.mean(arr))


def main():
    ap = argparse.ArgumentParser(description="Evaluate DDXPlus case-based retrieval.")
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--query_split", type=str, default="valid", choices=["valid", "test"])
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--top_n_causes", type=int, default=20)
    ap.add_argument("--alpha_global", type=float, default=0.25)
    ap.add_argument("--beta_lesion", type=float, default=0.75)
    ap.add_argument("--lesion_match", type=str, default="max_mean",
                    choices=["hungarian", "max_mean", "max_mean_normalized"])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    ap.add_argument("--max_queries", type=int, default=-1)
    ap.add_argument("--bank_dtype", type=str, default="bf16",
                    choices=["fp32", "fp16", "bf16"],
                    help="On-device storage dtype of the train bank stacks. "
                         "Default 'bf16' halves VRAM vs fp32; 'fp32' is needed "
                         "for fish case_dbs.")
    ap.add_argument("--max_train_cases", type=int, default=200000,
                    help="Cap on retained train-bank cases via uniform random "
                         "per-shard subsampling. The 1M-case DDXPlus bank is "
                         "~31 GB even in bf16 and overflows a 32 GB GPU; 200k "
                         "still leaves ~4k cases per condition for the 49-way "
                         "taxonomy. Use 0 to disable (full bank). Ignored when "
                         "--stream is set.")
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--stream", action="store_true",
                    help="Use shard-streaming retrieval (numerically equivalent "
                         "to the full bank within bf16 precision). Outer loop "
                         "iterates shards, inner loop batches queries — each "
                         "shard loaded to GPU exactly once. Required for the "
                         "full 1M DDXPlus bank on a single 32 GB GPU.")
    ap.add_argument("--stream_query_batch", type=int, default=64,
                    help="Query batch size for --stream mode. Memory peak is "
                         "dominated by [B*avg_lesions, m_shard_lesions] bf16. "
                         "Tune up (e.g. 128) on larger GPUs.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    bank_dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    bank_dtype = bank_dtype_map[args.bank_dtype]
    max_train_cases = args.max_train_cases if args.max_train_cases > 0 else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    case_db_dir = Path(args.case_db_dir)
    if args.stream:
        train_cases = load_train_cases_minimal(case_db_dir)
    else:
        train_cases, train_global_stack, train_lesion_stack, train_offsets = load_train_bank(
            case_db_dir, device, bank_dtype=bank_dtype,
            max_cases=max_train_cases, sample_seed=args.sample_seed,
        )
    queries = load_cases(case_db_dir, args.query_split)
    cause_pack = torch.load(case_db_dir / "cause_text_embs.pt", weights_only=False)
    meta_path = case_db_dir / "meta.json"
    meta = json.load(meta_path.open("r", encoding="utf-8")) if meta_path.exists() else {}
    cause_texts = list(cause_pack["texts"])
    cause_text_to_idx = {text: i for i, text in enumerate(cause_texts)}
    # .float() upcasts DDXPlus bf16/fp16 storage to fp32; no-op for fp32 builds.
    cause_table_embs = F.normalize(cause_pack["embeddings"].to(device).float(), dim=-1)

    query_cases = queries if args.max_queries <= 0 else queries[: args.max_queries]
    mode_tag = "stream(full bank)" if args.stream else (
        f"subsample={len(train_cases)}" if not args.stream and max_train_cases else "full bank"
    )
    print(
        f"[load] train={len(train_cases)} {args.query_split}={len(query_cases)} "
        f"causes={len(cause_texts)} dim={cause_table_embs.size(-1)} "
        f"dataset={meta.get('dataset', 'unknown')} mode={mode_tag}"
    )
    print(
        f"[eval] K={args.top_k_cases} alpha={args.alpha_global} "
        f"beta={args.beta_lesion} match={args.lesion_match}"
    )

    pathology_ranks: List[float] = []
    pathology_cov: List[int] = []
    pool_sizes: List[int] = []
    retained_case_counts: List[int] = []
    ddx_ndcg: Dict[int, List[float | None]] = {k: [] for k in args.ks}
    ddx_mass_at_k: Dict[int, List[float | None]] = {k: [] for k in args.ks}
    ddx_pool_mass_cov: List[float | None] = []
    per_query: List[dict] = []

    # --- Top-K case selection ---------------------------------------------
    # Streaming mode: shard-by-shard outer loop, all queries top-K precomputed.
    # Non-streaming mode: full bank on GPU, per-query loop (legacy path).
    if args.stream:
        print(f"[stream] computing top-K for {len(query_cases)} queries "
              f"against full bank (batch={args.stream_query_batch})...")
        stream_top_idx, stream_top_w, stream_top_raw = stream_top_k_cases(
            query_cases, case_db_dir,
            top_k_cases=args.top_k_cases,
            alpha=args.alpha_global, beta=args.beta_lesion,
            lesion_match=args.lesion_match,
            device=device, bank_dtype=bank_dtype,
            query_batch_size=args.stream_query_batch,
            verbose=True,
        )
    else:
        train_case_ids = offsets_to_case_ids(train_offsets, train_lesion_stack.device)

    t0 = time.time()
    for qi, q in enumerate(query_cases):
        if args.stream:
            top_case_idx = stream_top_idx[qi]
            top_case_w = stream_top_w[qi]
            top_case_raw_w = stream_top_raw[qi]
        else:
            q_global = F.normalize(q["global_emb"].to(device), dim=-1)
            q_evidence = F.normalize(q["lesion_embs"].to(device), dim=-1)

            sims = compute_case_similarities(
                q_global, q_evidence,
                train_global_stack, train_lesion_stack, train_offsets,
                args.alpha_global, args.beta_lesion,
                lesion_match=args.lesion_match,
                train_case_ids=train_case_ids,
            )
            top_case_idx, top_case_w, top_case_raw_w = select_positive_top_cases(
                sims, args.top_k_cases,
            )
        retained_case_counts.append(int(len(top_case_idx)))
        candidate_indices = build_candidate_pool(top_case_idx, train_cases)
        pool_sizes.append(len(candidate_indices))

        scores = score_candidates(
            candidate_indices, top_case_idx, top_case_w,
            train_cases, cause_table_embs,
        )
        if candidate_indices:
            sorted_local = torch.argsort(scores, descending=True).detach().cpu().numpy()
            sorted_global = np.asarray(candidate_indices, dtype=np.int64)[sorted_local]
        else:
            sorted_local = np.empty(0, dtype=np.int64)
            sorted_global = np.empty(0, dtype=np.int64)

        # DDXPlus pathology_exact = rank of the single strict pathology
        # (pathology_emb_idx is set by build_case_database). Fish builds
        # without this field fall back to cause_emb_indices (multi-cause GT).
        # Using the expanded cause_emb_indices on DDXPlus would over-count
        # DDX alternatives as pathology hits.
        pidx = q.get("pathology_emb_idx")
        if pidx is not None:
            gt_indices = [int(pidx)]
        else:
            gt_indices = [int(x) for x in q.get("cause_emb_indices", [])]
        rank = first_rank(sorted_global, gt_indices)
        pathology_ranks.append(rank)
        pathology_cov.append(1 if math.isfinite(rank) else 0)

        rel_by_idx = build_ddx_relevance(q, cause_text_to_idx)
        ddx_pool_mass_cov.append(relevance_mass_coverage(candidate_indices, rel_by_idx))
        for k in args.ks:
            ddx_ndcg[k].append(ndcg_at_k(sorted_global, rel_by_idx, k))
            ddx_mass_at_k[k].append(relevance_mass_at_k(sorted_global, rel_by_idx, k))

        top_n = min(args.top_n_causes, len(sorted_global))
        top_local = sorted_local[:top_n]
        top_global = sorted_global[:top_n]
        top_scores = (
            scores[torch.from_numpy(top_local).to(device)].detach().cpu().tolist()
            if top_n else []
        )
        per_query.append({
            "query_id": int(q.get("image_id", qi)),
            "file_name": q.get("file_name", f"query/{qi}"),
            "pathology": list(q.get("causes", [])),
            "pathology_rank": None if not math.isfinite(rank) else int(rank),
            "candidate_pool_size": int(len(candidate_indices)),
            "retained_positive_case_count": int(len(top_case_idx)),
            "evidence_count": int(q["lesion_embs"].size(0)),
            "ddx": q.get("ddx", []),
            "ddx_relevance_mass_coverage": ddx_pool_mass_cov[-1],
            "retrieved_cases": [
                {
                    "case_id": int(case_i),
                    "similarity_raw": float(top_case_raw_w[j]),
                    "similarity_weight_normalized": float(top_case_w[j]),
                    "pathology": list(train_cases[int(case_i)].get("causes", [])),
                }
                for j, case_i in enumerate(top_case_idx.tolist())
            ],
            "predicted_top_n": [
                {
                    "cause_table_idx": int(idx),
                    "text": cause_texts[int(idx)],
                    "score": float(score),
                    "ddx_relevance": float(rel_by_idx.get(int(idx), 0.0)),
                }
                for idx, score in zip(top_global.tolist(), top_scores)
            ],
        })

        if (qi + 1) % 50 == 0 or qi + 1 == len(query_cases):
            elapsed = time.time() - t0
            rate = (qi + 1) / max(elapsed, 1e-9)
            eta = (len(query_cases) - qi - 1) / max(rate, 1e-9)
            print(f"[eval] {qi+1}/{len(query_cases)} rate={rate:.2f} q/s ETA={eta/60:.1f} min")

    ranks = np.asarray(pathology_ranks, dtype=np.float64)
    path_metrics = summarize_rank_metric(ranks, pathology_cov)
    add_recall_at_ks(path_metrics, ranks, args.ks)

    metrics = {
        "dataset": meta.get("dataset", "DDXPlus"),
        "n_queries": len(query_cases),
        "n_train_cases": len(train_cases),
        "n_causes": len(cause_texts),
        "config": vars(args),
        "retrieved_cases": {
            "requested_top_k": int(args.top_k_cases),
            "positive_similarity_only": True,
            "mean_retained": float(np.mean(retained_case_counts)) if retained_case_counts else 0.0,
            "median_retained": float(np.median(retained_case_counts)) if retained_case_counts else 0.0,
        },
        "candidate_pool": {
            "mean": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
            "median": float(np.median(pool_sizes)) if pool_sizes else 0.0,
            "min": int(min(pool_sizes)) if pool_sizes else 0,
            "max": int(max(pool_sizes)) if pool_sizes else 0,
        },
        "pathology_exact": path_metrics,
        "differential_diagnosis": {
            "n_evaluable_queries": int(sum(v is not None for v in ddx_pool_mass_cov)),
            "pool_relevance_mass_coverage": mean_optional(ddx_pool_mass_cov),
            "NDCG": {f"@{k}": mean_optional(ddx_ndcg[k]) for k in args.ks},
            "relevance_mass_at_k": {f"@{k}": mean_optional(ddx_mass_at_k[k]) for k in args.ks},
        },
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "per_query_results.jsonl").open("w", encoding="utf-8") as f:
        for row in per_query:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n=== DDXPlus retrieval metrics ===")
    print(f"  n_queries={metrics['n_queries']}  train={metrics['n_train_cases']}  causes={metrics['n_causes']}")
    print(
        "  pathology: "
        f"coverage={path_metrics['coverage']:.4f} "
        f"MRR={path_metrics['MRR']:.4f} "
        + " ".join(f"R@{k}={path_metrics[f'R@{k}']:.4f}" for k in args.ks)
    )
    ddx = metrics["differential_diagnosis"]
    print(
        "  ddx: "
        f"pool_mass_cov={ddx['pool_relevance_mass_coverage']} "
        + " ".join(f"NDCG@{k}={ddx['NDCG'][f'@{k}']}" for k in args.ks)
    )
    print(f"[save] metrics.json + per_query_results.jsonl -> {out_dir}")


if __name__ == "__main__":
    main()

