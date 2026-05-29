"""DDXPlus Phase 3 evaluation: case encoder retrieval → pathology / DDX metrics.

Runs the trained DeepSets case encoder on a DDXPlus case_db (sharded layout
supported), then for each query case scores every train case by a single
case-to-case cosine ``z_q · z_train``. The downstream
``select_positive_top_cases`` → ``build_candidate_pool`` → ``score_candidates``
pipeline is shared with ``ddxplus/eval_retrieval.py``, so the reported
``pathology_exact`` and ``differential_diagnosis`` blocks are directly
comparable to the Phase 1 retrieval numbers.

CLI from repo root (SDM env):

    /home/lab603/anaconda3/envs/SDM/bin/python \
        -m diagnosis_model.cause_inference.ddxplus.eval_phase3 \
        --encoder_ckpt diagnosis_model/cause_inference/outputs/ddxplus_encoder/best_encoder.pt \
        --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
        --output_dir diagnosis_model/cause_inference/outputs/ddxplus_phase3_eval \
        --max_train_cases 200000 --sample_seed 42 \
        --max_query_cases 5000 \
        --top_k_cases 20 --top_n_causes 20 \
        --ks 1 5 10 20 50
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.ddxplus.eval_retrieval import (
    build_ddx_relevance,
    first_rank,
    mean_optional,
    ndcg_at_k,
    relevance_mass_at_k,
    relevance_mass_coverage,
)
from diagnosis_model.cause_inference.phase1_baseline import (
    add_recall_at_ks,
    build_candidate_pool,
    load_cases,
    score_candidates,
    select_positive_top_cases,
    summarize_rank_metric,
)
from diagnosis_model.cause_inference.rvq_rerank.fit_rvq import load_encoder
from diagnosis_model.cause_inference.train_case_encoder import encode_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", type=str, required=True)
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--query_split", type=str, default="valid",
                    choices=["valid", "test"])
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--top_n_causes", type=int, default=20)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    ap.add_argument("--max_train_cases", type=int, default=200000,
                    help="Cap retained train cases (RVQ-aligned bank size). "
                         "Must match the value used in train_case_encoder.py "
                         "so the encoder is queried on the same bank it saw "
                         "at training time.")
    ap.add_argument("--max_query_cases", type=int, default=-1,
                    help="Cap query cases via subsample. -1 = full split "
                         "(132k valid / 134k test). 5000 is enough for "
                         "headline numbers in ~minute scale.")
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--encode_batch", type=int, default=256)
    ap.add_argument("--query_batch", type=int, default=256,
                    help="Query batch for the [Nq, Nt] sim matmul. Memory "
                         "peak is dominated by [query_batch, Nt] fp32 sim "
                         "(at Nt=200k, 256 queries = 200 MB).")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    case_db_dir = Path(args.case_db_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # 1. Load encoder
    encoder, enc_cfg = load_encoder(Path(args.encoder_ckpt), device)
    print(f"[encoder] type={enc_cfg.encoder_type}  D={enc_cfg.d_model}")

    # 2. Load case db (train bank + query split + cause embeddings)
    max_train = args.max_train_cases if args.max_train_cases > 0 else None
    max_q = args.max_query_cases if args.max_query_cases > 0 else None
    train_cases = load_cases(
        case_db_dir, "train",
        max_cases=max_train, sample_seed=args.sample_seed,
    )
    query_cases = load_cases(
        case_db_dir, args.query_split,
        max_cases=max_q, sample_seed=args.sample_seed,
    )
    cause_pack = torch.load(
        case_db_dir / "cause_text_embs.pt", weights_only=False, map_location="cpu",
    )
    cause_texts = list(cause_pack["texts"])
    cause_table_embs = F.normalize(
        cause_pack["embeddings"].to(device).float(), dim=-1,
    )
    cause_text_to_idx = {t: i for i, t in enumerate(cause_texts)}
    meta = json.load((case_db_dir / "meta.json").open())
    print(
        f"[data] train={len(train_cases)} {args.query_split}={len(query_cases)} "
        f"causes={len(cause_texts)} D={meta['global_dim']} "
        f"dataset={meta.get('dataset', 'unknown')}"
    )

    # 3. Encode bank + queries
    print("[encode] train bank ...")
    t0 = time.time()
    z_train = encode_all(
        encoder, train_cases, device, batch_size=args.encode_batch,
    ).to(device).float()
    z_train = F.normalize(z_train, dim=-1)
    print(f"  z_train={tuple(z_train.shape)}  t={time.time()-t0:.1f}s")
    t0 = time.time()
    z_query = encode_all(
        encoder, query_cases, device, batch_size=args.encode_batch,
    ).to(device).float()
    z_query = F.normalize(z_query, dim=-1)
    print(f"  z_query={tuple(z_query.shape)}  t={time.time()-t0:.1f}s")

    # 4. Sims = z_q @ z_train.T (batched over queries)
    Nq = z_query.size(0)
    Nt = z_train.size(0)
    print(f"[score] {Nq} queries × {Nt} train cases (batch={args.query_batch})")

    pathology_ranks: List[float] = []
    pathology_cov: List[int] = []
    pool_sizes: List[int] = []
    retained_case_counts: List[int] = []
    ddx_ndcg: Dict[int, List] = {k: [] for k in args.ks}
    ddx_mass_at_k: Dict[int, List] = {k: [] for k in args.ks}
    ddx_pool_mass_cov: List = []
    per_query: List[dict] = []

    z_train_T = z_train.T
    t0 = time.time()
    for qs in range(0, Nq, args.query_batch):
        qe = min(qs + args.query_batch, Nq)
        sims_b = (z_query[qs:qe] @ z_train_T).cpu().numpy()        # [bv, Nt]
        for bi in range(qe - qs):
            qi = qs + bi
            q = query_cases[qi]
            sims = sims_b[bi]

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

            pidx = q.get("pathology_emb_idx")
            if pidx is not None:
                gt_indices = [int(pidx)]
            else:
                gt_indices = [int(x) for x in q.get("cause_emb_indices", [])]
            rank = first_rank(sorted_global, gt_indices)
            pathology_ranks.append(rank)
            pathology_cov.append(1 if math.isfinite(rank) else 0)

            rel_by_idx = build_ddx_relevance(q, cause_text_to_idx)
            ddx_pool_mass_cov.append(
                relevance_mass_coverage(candidate_indices, rel_by_idx)
            )
            for k in args.ks:
                ddx_ndcg[k].append(ndcg_at_k(sorted_global, rel_by_idx, k))
                ddx_mass_at_k[k].append(
                    relevance_mass_at_k(sorted_global, rel_by_idx, k)
                )

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

        if (qe % (args.query_batch * 4) == 0) or qe == Nq:
            elapsed = time.time() - t0
            rate = qe / max(elapsed, 1e-9)
            eta = (Nq - qe) / max(rate, 1e-9)
            print(f"[eval] {qe}/{Nq} rate={rate:.1f} q/s ETA={eta/60:.1f} min")

    ranks = np.asarray(pathology_ranks, dtype=np.float64)
    path_metrics = summarize_rank_metric(ranks, pathology_cov)
    add_recall_at_ks(path_metrics, ranks, args.ks)

    metrics = {
        "dataset": meta.get("dataset", "DDXPlus"),
        "phase": 3,
        "encoder": str(args.encoder_ckpt),
        "n_queries": int(Nq),
        "n_train_cases": int(Nt),
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

    print("\n=== DDXPlus Phase 3 retrieval metrics ===")
    print(f"  encoder: {args.encoder_ckpt}")
    print(f"  n_queries={metrics['n_queries']}  train={metrics['n_train_cases']}")
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
