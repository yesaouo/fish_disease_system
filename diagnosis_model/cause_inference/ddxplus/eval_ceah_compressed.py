"""DDXPlus: does coarse-stage RVQ compression propagate through CEAH?

This is the DDXPlus analogue of the fish ``eval_ceah_compressed.py``. The fish
script measures whether RVQ damage on the Phase-3 coarse case ranking survives
the CEAH fine stage (gamma=0, pure CEAH). DDXPlus differs in two ways that this
script accounts for:

  1. There is a single ``ddxplus_case_db`` (no raw/fine split — DDXPlus is
     tabular, no VLM fine-tune), so coarse z and CEAH share one DB.
  2. The strongest DDXPlus operating point is gamma=0.75 (coarse-dominant),
     not gamma=0. We therefore sweep gamma and report the whole curve so the
     reranker's effect can be read at the production point, not just pure-CEAH.

For each coarse case-similarity source (dense / rvq_only / light / full_analytic)
we run the *same* CEAH over the resulting candidate pool, mix Phase-1 aggregation
score with CEAH score at every gamma, and report strict PATHOLOGY R@K + MRR and
DDX NDCG. Decisive comparisons at the production point (gamma=0.75, top_k=20):

  (a) rvq_only vs dense       -> does compression damage survive CEAH?
  (b) light    vs rvq_only    -> does the reranker recover/hurt after CEAH?
  (c) full_analytic vs light  -> is the gap an architecture limit (oracle Delta)?

Coarse sims mirror eval_phase4.py exactly (encoder_v2 z, F.normalize, RVQ encode).

Run (repo root, SDM env):
  $PY -m diagnosis_model.cause_inference.ddxplus.eval_ceah_compressed \
    --encoder_ckpt diagnosis_model/cause_inference/outputs/ddxplus_encoder_v2/best_encoder.pt \
    --rvq_dir      diagnosis_model/cause_inference/outputs/ddxplus_rvq_v2/rvq_M4_K256 \
    --reranker_ckpt diagnosis_model/cause_inference/outputs/ddxplus_rvq_v2/reranker_M4_K256_light/best.pt \
    --case_db_dir  diagnosis_model/cause_inference/outputs/ddxplus_case_db \
    --ceah_ckpt    diagnosis_model/cause_inference/outputs/ddxplus_ceah/best_ceah.pt \
    --output_dir   diagnosis_model/cause_inference/outputs/ddxplus_ceah_compressed_eval \
    --methods dense rvq_only light full_analytic \
    --max_query_cases 5000
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

from diagnosis_model.cause_inference.ddxplus.eval_ceah import (
    ceah_forward_for_pool,
    minmax_norm,
)
from diagnosis_model.cause_inference.ddxplus.eval_phase4 import (
    _compute_sims,
    _load_rvq,
    _load_reranker,
)
from diagnosis_model.cause_inference.ddxplus.eval_retrieval import (
    build_ddx_relevance,
    first_rank,
    mean_optional,
    ndcg_at_k,
    relevance_mass_at_k,
)
from diagnosis_model.cause_inference.models import CEAH
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
    ap.add_argument("--rvq_dir", type=str, required=True,
                    help="Dir holding codebooks.pt (e.g. .../rvq_M4_K256/).")
    ap.add_argument("--reranker_ckpt", type=str, default="",
                    help="Light reranker best.pt; required if 'light' in --methods.")
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--ceah_ckpt", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--query_split", type=str, default="valid",
                    choices=["valid", "test"])
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["dense", "rvq_only", "light", "full_analytic"],
                    choices=["dense", "rvq_only", "light", "full_analytic"])
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="hybrid = gamma*Phase1 + (1-gamma)*CEAH")
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--K_top", type=int, default=50,
                    help="Reranker top-K_top (must match training).")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    ap.add_argument("--text_kind", type=str, default="none",
                    choices=["medical", "colloquial", "none"])
    ap.add_argument("--common_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attribution_mode", type=str, default="softmax",
                    choices=["sigmoid", "softmax"])
    ap.add_argument("--scoring_mode", type=str, default="multiplicative",
                    choices=["single", "multiplicative"])
    ap.add_argument("--max_train_cases", type=int, default=200000,
                    help="Cap retained train cases; must match fit_rvq.py and "
                         "train_case_encoder.py.")
    ap.add_argument("--max_query_cases", type=int, default=5000,
                    help="Cap query set (CEAH is per-query; full 132k is slow).")
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--encode_batch", type=int, default=256)
    ap.add_argument("--query_batch", type=int, default=128)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_db_dir = Path(args.case_db_dir)

    # ---- encoder + data ----
    encoder, enc_cfg = load_encoder(Path(args.encoder_ckpt), device)
    D = enc_cfg.d_model
    max_train = args.max_train_cases if args.max_train_cases > 0 else None
    max_q = args.max_query_cases if args.max_query_cases > 0 else None
    train_cases = load_cases(case_db_dir, "train",
                             max_cases=max_train, sample_seed=args.sample_seed)
    query_cases = load_cases(case_db_dir, args.query_split,
                             max_cases=max_q, sample_seed=args.sample_seed)
    cause_pack = torch.load(case_db_dir / "cause_text_embs.pt",
                            weights_only=False, map_location="cpu")
    cause_texts = list(cause_pack["texts"])
    cause_text_to_idx = {t: i for i, t in enumerate(cause_texts)}
    cause_table_embs = F.normalize(
        cause_pack["embeddings"].to(device).float(), dim=-1,
    )
    in_dim = cause_table_embs.size(-1)
    meta = json.load((case_db_dir / "meta.json").open())
    print(f"[data] train={len(train_cases)} {args.query_split}={len(query_cases)} "
          f"causes={len(cause_texts)} D={D}")

    # ---- coarse z (mirror eval_phase4: encode -> float -> L2-normalize) ----
    z_train = encode_all(encoder, train_cases, device, batch_size=args.encode_batch).to(device).float()
    z_train = F.normalize(z_train, dim=-1)
    z_query = encode_all(encoder, query_cases, device, batch_size=args.encode_batch).to(device).float()
    z_query = F.normalize(z_query, dim=-1)

    # ---- RVQ + reranker ----
    rvq = _load_rvq(Path(args.rvq_dir), D, device)
    codes_train, z_hat_train, e_train = rvq.encode(z_train)
    e_norm_train = e_train.norm(dim=-1)
    M = int(rvq.codebooks.size(0))
    K = int(rvq.codebooks.size(1))
    bits = max(1, math.ceil(math.log2(max(K, 2))))
    comp_x = (D * 32) / (M * bits)
    print(f"[rvq] M={M} K={K} comp≈{comp_x:.0f}×")

    reranker = None
    reranker_variant = "light"
    if args.reranker_ckpt:
        reranker, rcfg = _load_reranker(Path(args.reranker_ckpt), device)
        reranker_variant = rcfg.variant
        print(f"[reranker] variant={reranker_variant} <- {args.reranker_ckpt}")
    elif "light" in args.methods:
        raise ValueError("method 'light' requires --reranker_ckpt")

    # ---- CEAH ----
    ceah = CEAH(
        global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
        common_dim=args.common_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
        attribution_mode=args.attribution_mode, scoring_mode=args.scoring_mode,
    ).to(device)
    ceah.load_state_dict(torch.load(args.ceah_ckpt, map_location=device))
    ceah.eval()
    print(f"[ceah] {args.ceah_ckpt}")

    Nq = z_query.size(0)
    all_rows: List[Dict] = []

    for method in args.methods:
        print(f"\n=== method: {method} ===")
        ranks_by_g: Dict[str, List[float]] = {f"g={g:.2f}": [] for g in args.gammas}
        cov_by_g: Dict[str, List[int]] = {f"g={g:.2f}": [] for g in args.gammas}
        ndcg_by_g: Dict[str, Dict[int, List]] = {
            f"g={g:.2f}": {k: [] for k in args.ks} for g in args.gammas
        }
        mass_by_g: Dict[str, Dict[int, List]] = {
            f"g={g:.2f}": {k: [] for k in args.ks} for g in args.gammas
        }
        pool_sizes: List[int] = []
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

                top_case_idx, top_case_w, _ = select_positive_top_cases(
                    sims, args.top_k_cases,
                )
                candidate_indices = build_candidate_pool(top_case_idx, train_cases)
                pool_sizes.append(len(candidate_indices))

                s1 = score_candidates(
                    candidate_indices, top_case_idx, top_case_w,
                    train_cases, cause_table_embs,
                )
                s_ceah, _, _ = ceah_forward_for_pool(
                    ceah, q, candidate_indices, cause_table_embs,
                    in_dim, device, args.text_kind,
                )
                s1_n = minmax_norm(s1)
                sc_n = minmax_norm(s_ceah)

                pidx = q.get("pathology_emb_idx")
                if pidx is not None:
                    gt_indices = [int(pidx)]
                else:
                    gt_indices = [int(x) for x in q.get("cause_emb_indices", [])]
                rel_by_idx = build_ddx_relevance(q, cause_text_to_idx)

                for g in args.gammas:
                    tag = f"g={g:.2f}"
                    hybrid = g * s1_n + (1.0 - g) * sc_n
                    if candidate_indices:
                        sorted_local = torch.argsort(hybrid, descending=True).cpu().numpy()
                        sorted_global = np.asarray(candidate_indices, dtype=np.int64)[sorted_local]
                    else:
                        sorted_global = np.empty(0, dtype=np.int64)
                    rank = first_rank(sorted_global, gt_indices)
                    ranks_by_g[tag].append(rank)
                    cov_by_g[tag].append(1 if math.isfinite(rank) else 0)
                    for k in args.ks:
                        ndcg_by_g[tag][k].append(ndcg_at_k(sorted_global, rel_by_idx, k))
                        mass_by_g[tag][k].append(relevance_mass_at_k(sorted_global, rel_by_idx, k))

            if (qe % (args.query_batch * 4) == 0) or qe == Nq:
                rate = qe / max(time.time() - t0, 1e-9)
                print(f"[{method}] {qe}/{Nq} {rate:.1f} q/s "
                      f"ETA={(Nq-qe)/max(rate,1e-9)/60:.1f} min")

        metrics_per_gamma = {}
        for g in args.gammas:
            tag = f"g={g:.2f}"
            arr = np.asarray(ranks_by_g[tag], dtype=np.float64)
            block = summarize_rank_metric(arr, cov_by_g[tag])
            add_recall_at_ks(block, arr, args.ks)
            metrics_per_gamma[tag] = {
                "pathology_exact": block,
                "differential_diagnosis": {
                    "NDCG": {f"@{k}": mean_optional(ndcg_by_g[tag][k]) for k in args.ks},
                    "relevance_mass_at_k": {
                        f"@{k}": mean_optional(mass_by_g[tag][k]) for k in args.ks
                    },
                },
            }
        all_rows.append({
            "method": method,
            "mean_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
            "metrics_per_gamma": metrics_per_gamma,
        })

        print(f"  {'gamma':>6} {'MRR':>8} " + " ".join(f"R@{k}" for k in args.ks))
        for g in args.gammas:
            p = metrics_per_gamma[f"g={g:.2f}"]["pathology_exact"]
            print(f"  {g:>6.2f} {p['MRR']:>8.4f} "
                  + " ".join(f"{p[f'R@{k}']:.4f}" for k in args.ks))

    summary = {
        "dataset": meta.get("dataset", "DDXPlus"),
        "phase": "4+CEAH",
        "encoder": str(args.encoder_ckpt),
        "rvq": {"dir": str(args.rvq_dir), "M": M, "K": K, "compression_x": comp_x},
        "reranker": str(args.reranker_ckpt) if args.reranker_ckpt else None,
        "ceah": str(args.ceah_ckpt),
        "n_queries": int(Nq),
        "n_train_cases": int(z_train.size(0)),
        "config": vars(args),
        "methods": all_rows,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[save] -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
