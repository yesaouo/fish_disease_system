"""DDXPlus Phase 4 evaluation: RVQ-compressed case bank + optional reranker.

Stacks on top of Phase 3: encodes the train bank and queries via the same
DeepSets case encoder, then replaces dense ``z_q · z_train`` with the
production ABQ path ``z_q · ẑ_train`` (RVQ-only) and optionally the Light
cross-attention reranker that learns Δ ≈ q·e on the top-K_top candidates.

For each scorer in ``--methods`` the script runs the shared DDXPlus
``select_positive_top_cases`` → ``build_candidate_pool`` →
``score_candidates`` pipeline and reports pathology R@K + DDX NDCG. The
``rvq_only`` row is the production point (Phase 1/2 use the same
case_db_raw); ``light`` shows where the reranker recovers stress-regime
gap; ``dense`` is the uncompressed reference.

CLI from repo root (SDM env):

    /home/lab603/anaconda3/envs/SDM/bin/python \
        -m diagnosis_model.cause_inference.ddxplus.eval_phase4 \
        --encoder_ckpt diagnosis_model/cause_inference/outputs/ddxplus_encoder/best_encoder.pt \
        --rvq_dir diagnosis_model/cause_inference/outputs/ddxplus_rvq/rvq_M4_K256 \
        --reranker_ckpt diagnosis_model/cause_inference/outputs/ddxplus_rvq/reranker_M4_K256_light/best.pt \
        --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
        --output_dir diagnosis_model/cause_inference/outputs/ddxplus_phase4_eval \
        --max_train_cases 200000 --sample_seed 42 \
        --max_query_cases 5000 \
        --methods dense rvq_only light \
        --top_k_cases 20 --top_n_causes 20 \
        --ks 1 5 10 20 50
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

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
from diagnosis_model.cause_inference.rvq_rerank.reranker import (
    Reranker, RerankerConfig,
)
from diagnosis_model.cause_inference.rvq_rerank.rvq import RVQCodebook
from diagnosis_model.cause_inference.train_case_encoder import encode_all


def _load_rvq(rvq_dir: Path, D: int, device: torch.device) -> RVQCodebook:
    pkg = torch.load(
        rvq_dir / "codebooks.pt", weights_only=False, map_location=device,
    )
    M = int(pkg["config"]["M"])
    K = int(pkg["config"]["K"])
    rvq = RVQCodebook(M=M, K=K, D=D).to(device)
    rvq.codebooks.copy_(pkg["codebooks"].to(device))
    rvq.fitted.copy_(pkg["fitted"].to(device))
    return rvq


def _load_reranker(ckpt_path: Path, device: torch.device) -> tuple[Reranker, RerankerConfig]:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg = RerankerConfig(**dict(ckpt["reranker_config"]))
    reranker = Reranker(cfg).to(device)
    reranker.load_state_dict(ckpt["reranker_state"])
    reranker.eval()
    return reranker, cfg


@torch.no_grad()
def _compute_sims(
    method: str,
    z_q_batch: torch.Tensor,                     # [B, D]
    z_train: torch.Tensor,                       # [Nt, D]
    z_hat_train: Optional[torch.Tensor],         # [Nt, D]
    e_train: Optional[torch.Tensor],             # [Nt, D]
    codes_train: Optional[torch.Tensor],         # [Nt, M]
    e_norm_train: Optional[torch.Tensor],        # [Nt]
    reranker: Optional[Reranker],
    reranker_variant: str,
    K_top: int,
) -> torch.Tensor:
    """Return [B, Nt] sim row for the requested method."""
    if method == "dense":
        return z_q_batch @ z_train.T
    if method == "rvq_only":
        return z_q_batch @ z_hat_train.T
    if method == "full_analytic":
        s_first = z_q_batch @ z_hat_train.T
        s_final = s_first.clone()
        s_first_top, top_idx = s_first.topk(K_top, dim=-1)
        e_top = e_train[top_idx]                                       # [B, K, D]
        delta = (z_q_batch.unsqueeze(1) * e_top).sum(dim=-1)           # [B, K]
        s_final.scatter_(1, top_idx, s_first_top + delta)
        return s_final
    if method == "light":
        assert reranker is not None, "light method requires --reranker_ckpt"
        s_first = z_q_batch @ z_hat_train.T
        s_final = s_first.clone()
        s_first_top, top_idx = s_first.topk(K_top, dim=-1)
        z_hat_top = z_hat_train[top_idx]
        codes_top = codes_train[top_idx]
        e_norm_top = e_norm_train[top_idx]
        z_top = z_train[top_idx] if reranker_variant == "full" else None
        e_top = e_train[top_idx] if reranker_variant == "full" else None
        delta = reranker(
            z_q_batch, z_hat_top, codes_top, s_first_top, e_norm_top,
            z=z_top, e=e_top,
        )
        s_final.scatter_(1, top_idx, s_first_top + delta)
        return s_final
    raise ValueError(f"unknown method {method!r}")


def _evaluate_method(
    method: str,
    z_query: torch.Tensor,
    z_train: torch.Tensor,
    z_hat_train: Optional[torch.Tensor],
    e_train: Optional[torch.Tensor],
    codes_train: Optional[torch.Tensor],
    e_norm_train: Optional[torch.Tensor],
    reranker: Optional[Reranker],
    reranker_variant: str,
    query_cases: List[dict],
    train_cases: List[dict],
    cause_table_embs: torch.Tensor,
    cause_text_to_idx: Dict[str, int],
    cause_texts: List[str],
    args,
    device: torch.device,
) -> Dict:
    pathology_ranks: List[float] = []
    pathology_cov: List[int] = []
    pool_sizes: List[int] = []
    retained_case_counts: List[int] = []
    ddx_ndcg: Dict[int, List] = {k: [] for k in args.ks}
    ddx_mass_at_k: Dict[int, List] = {k: [] for k in args.ks}
    ddx_pool_mass_cov: List = []
    per_query: List[dict] = []

    Nq = z_query.size(0)
    t0 = time.time()
    for qs in range(0, Nq, args.query_batch):
        qe = min(qs + args.query_batch, Nq)
        sims_b = _compute_sims(
            method, z_query[qs:qe],
            z_train, z_hat_train, e_train, codes_train, e_norm_train,
            reranker, reranker_variant, args.K_top,
        ).cpu().numpy()
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
                "method": method,
                "query_id": int(q.get("image_id", qi)),
                "file_name": q.get("file_name", f"query/{qi}"),
                "pathology": list(q.get("causes", [])),
                "pathology_rank": None if not math.isfinite(rank) else int(rank),
                "candidate_pool_size": int(len(candidate_indices)),
                "retained_positive_case_count": int(len(top_case_idx)),
                "ddx_relevance_mass_coverage": ddx_pool_mass_cov[-1],
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
            print(f"[{method}] {qe}/{Nq}  rate={rate:.1f} q/s ETA={eta/60:.1f} min")

    ranks = np.asarray(pathology_ranks, dtype=np.float64)
    path_metrics = summarize_rank_metric(ranks, pathology_cov)
    add_recall_at_ks(path_metrics, ranks, args.ks)
    return {
        "method": method,
        "retrieved_cases": {
            "requested_top_k": int(args.top_k_cases),
            "positive_similarity_only": True,
            "mean_retained": float(np.mean(retained_case_counts)) if retained_case_counts else 0.0,
        },
        "candidate_pool": {
            "mean": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
            "max": int(max(pool_sizes)) if pool_sizes else 0,
        },
        "pathology_exact": path_metrics,
        "differential_diagnosis": {
            "n_evaluable_queries": int(sum(v is not None for v in ddx_pool_mass_cov)),
            "pool_relevance_mass_coverage": mean_optional(ddx_pool_mass_cov),
            "NDCG": {f"@{k}": mean_optional(ddx_ndcg[k]) for k in args.ks},
            "relevance_mass_at_k": {f"@{k}": mean_optional(ddx_mass_at_k[k]) for k in args.ks},
        },
        "per_query": per_query,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", type=str, required=True)
    ap.add_argument("--rvq_dir", type=str, required=True,
                    help="Dir holding codebooks.pt (e.g. .../rvq_M4_K256/).")
    ap.add_argument("--reranker_ckpt", type=str, default="",
                    help="Optional Light/Full reranker best.pt. Required if "
                         "--methods includes 'light'.")
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--query_split", type=str, default="valid",
                    choices=["valid", "test"])
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["dense", "rvq_only", "light"],
                    choices=["dense", "rvq_only", "light", "full_analytic"])
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--top_n_causes", type=int, default=20)
    ap.add_argument("--K_top", type=int, default=50,
                    help="Reranker top-K_top (must match training)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    ap.add_argument("--max_train_cases", type=int, default=200000,
                    help="Cap retained train cases. Must match fit_rvq.py "
                         "and train_case_encoder.py.")
    ap.add_argument("--max_query_cases", type=int, default=-1)
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--encode_batch", type=int, default=256)
    ap.add_argument("--query_batch", type=int, default=128)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    case_db_dir = Path(args.case_db_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    encoder, enc_cfg = load_encoder(Path(args.encoder_ckpt), device)
    D = enc_cfg.d_model
    print(f"[encoder] type={enc_cfg.encoder_type}  D={D}")

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
        f"causes={len(cause_texts)}"
    )

    # Encode bank + queries
    print("[encode] z_train ...")
    z_train = encode_all(encoder, train_cases, device, batch_size=args.encode_batch).to(device).float()
    z_train = F.normalize(z_train, dim=-1)
    print(f"  z_train={tuple(z_train.shape)}")
    print("[encode] z_query ...")
    z_query = encode_all(encoder, query_cases, device, batch_size=args.encode_batch).to(device).float()
    z_query = F.normalize(z_query, dim=-1)
    print(f"  z_query={tuple(z_query.shape)}")

    # RVQ encode
    rvq = _load_rvq(Path(args.rvq_dir), D, device)
    codes_train, z_hat_train, e_train = rvq.encode(z_train)
    e_norm_train = e_train.norm(dim=-1)
    M = int(rvq.codebooks.size(0))
    K = int(rvq.codebooks.size(1))
    bits_per_code = max(1, math.ceil(math.log2(max(K, 2))))
    compression_x = (D * 32) / (M * bits_per_code)
    print(f"[rvq] M={M} K={K} compression≈{compression_x:.0f}×  "
          f"||e|| mean={e_norm_train.mean().item():.4f}")

    # Optional reranker
    reranker = None
    reranker_variant = "light"
    if args.reranker_ckpt:
        reranker, rcfg = _load_reranker(Path(args.reranker_ckpt), device)
        reranker_variant = rcfg.variant
        print(f"[reranker] variant={reranker_variant} loaded from {args.reranker_ckpt}")
    elif "light" in args.methods:
        raise ValueError(
            "method 'light' requires --reranker_ckpt; drop 'light' from --methods "
            "or pass a checkpoint."
        )

    rows: List[Dict] = []
    for method in args.methods:
        print(f"\n=== method: {method} ===")
        result = _evaluate_method(
            method, z_query, z_train, z_hat_train, e_train,
            codes_train, e_norm_train, reranker, reranker_variant,
            query_cases, train_cases, cause_table_embs,
            cause_text_to_idx, cause_texts, args, device,
        )
        rows.append(result)
        path = result["pathology_exact"]
        ddx = result["differential_diagnosis"]
        print(
            "  pathology: "
            f"coverage={path['coverage']:.4f} "
            f"MRR={path['MRR']:.4f} "
            + " ".join(f"R@{k}={path[f'R@{k}']:.4f}" for k in args.ks)
        )
        print(
            "  ddx: "
            f"pool_mass_cov={ddx['pool_relevance_mass_coverage']} "
            + " ".join(f"NDCG@{k}={ddx['NDCG'][f'@{k}']}" for k in args.ks)
        )

    metrics = {
        "dataset": meta.get("dataset", "DDXPlus"),
        "phase": 4,
        "encoder": str(args.encoder_ckpt),
        "rvq": {"dir": str(args.rvq_dir), "M": M, "K": K, "compression_x": compression_x},
        "reranker": str(args.reranker_ckpt) if args.reranker_ckpt else None,
        "n_queries": int(z_query.size(0)),
        "n_train_cases": int(z_train.size(0)),
        "n_causes": len(cause_texts),
        "config": vars(args),
        "methods": [
            {k: v for k, v in r.items() if k != "per_query"} for r in rows
        ],
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            for entry in row["per_query"]:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n[save] metrics.json + per_query.jsonl -> {out_dir}")


if __name__ == "__main__":
    main()
