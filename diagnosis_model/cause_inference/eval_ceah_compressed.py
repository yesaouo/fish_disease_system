"""Experiment: does coarse-stage compression propagate through the CEAH fine stage?

Production cascade is: coarse case retrieval (encoder_raw + optional RVQ) -> top-K
cases -> candidate cause pool -> CEAH re-rank (gamma=0). The existing Phase-4 evals
measure compression damage on *retrieval-only* metrics (eval_final.py). This script
asks the downstream question: after CEAH re-scores the candidate pool, does RVQ
compression of the coarse case ranking still move the final cause R@K?

For each coarse case-similarity source we run the *same* CEAH (gamma=0) on the
resulting candidate pool and report final sem / cluster R@K:

  dense           : q . z_train            (no compression, reference)
  rvq_only M,K    : q . z_hat_train         (lossy, no reranker)
  light  M,K      : s_first + Delta_light   (residual reranker on top-K_top)

Decisive comparisons:
  (a) rvq_only vs dense at each (M,K)  -> does compression damage survive CEAH?
  (b) light    vs rvq_only at each (M,K) -> does the reranker recover anything
                                            *after* CEAH + aggregation buffer?

Coarse z is computed from encoder_raw on case_db_raw (production coarse path); CEAH
runs on the fine case_db. Train/valid case indices are aligned between the two DBs
(verified: identical image_id ordering), so top-K indices from the raw coarse
ranking directly select fine case_db cases.

Run (repo root, SDM env):
  $PY -m diagnosis_model.cause_inference.eval_ceah_compressed \
    --coarse_case_db_dir outputs/case_db_raw \
    --encoder_ckpt       outputs/encoder_raw/best_encoder.pt \
    --rvq_root           outputs/rvq_rerank_raw \
    --fine_case_db_dir   outputs/case_db \
    --ceah_ckpt          outputs/ceah_v3/best_ceah.pt \
    --cluster_json       outputs/cause_clusters_llm.json \
    --attribution_mode softmax --scoring_mode multiplicative
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.models import CEAH
from diagnosis_model.cause_inference.eval_ceah import (
    ceah_forward_for_pool,
    minmax_norm,
)
from diagnosis_model.cause_inference.phase1_baseline import (
    MISS_RANK,
    add_recall_at_ks,
    build_candidate_pool,
    select_positive_top_cases,
    summarize_rank_metric,
)
from diagnosis_model.cause_inference.train_case_encoder import encode_all
from diagnosis_model.cause_inference.rvq_rerank.fit_rvq import load_encoder
from diagnosis_model.cause_inference.rvq_rerank.rvq import RVQCodebook
from diagnosis_model.cause_inference.rvq_rerank.reranker import Reranker, RerankerConfig
from diagnosis_model.cause_inference.rvq_rerank.eval_final import _rerank_sim


@torch.no_grad()
def eval_one_source(
    sim_fn: Callable[[int], torch.Tensor],   # qi -> [N_train] case similarity
    valid_cases: list,
    fine_train_cases: list,
    cause_table_embs: torch.Tensor,
    cluster_id_array: np.ndarray | None,
    ceah,
    args,
    device: str,
    in_dim: int,
) -> Dict[str, float]:
    """Run gamma=0 CEAH on candidate pools selected by sim_fn; return aggregate metrics."""
    sem_ranks_all: List[float] = []
    sem_cov_all: List[int] = []
    cl_ranks_all: List[float] = []
    cl_cov_all: List[int] = []
    pool_sizes: List[int] = []

    for qi, q in enumerate(valid_cases):
        sims = sim_fn(qi)
        if isinstance(sims, torch.Tensor):
            sims = sims.detach().cpu().numpy()
        top_k_idx, _, _ = select_positive_top_cases(sims, args.top_k_cases)
        candidate_indices = build_candidate_pool(top_k_idx, fine_train_cases)
        pool_size = len(candidate_indices)
        pool_sizes.append(pool_size)
        n_gt = len(q["cause_emb_indices"])

        if pool_size == 0:
            for _ in range(n_gt):
                sem_ranks_all.append(MISS_RANK); sem_cov_all.append(0)
                if cluster_id_array is not None:
                    cl_ranks_all.append(MISS_RANK); cl_cov_all.append(0)
            continue

        s_ceah, _, _ = ceah_forward_for_pool(
            ceah, q, candidate_indices, cause_table_embs, in_dim, device,
            use_text_kind=args.text_kind,
        )
        # gamma = 0  ->  pure CEAH ranking
        sc_n = minmax_norm(s_ceah)
        raw_sorted_local = torch.argsort(sc_n, descending=True).cpu().numpy()

        cand_idx_t = torch.tensor(candidate_indices, device=device, dtype=torch.long)
        cand_embs = cause_table_embs.index_select(0, cand_idx_t)
        raw_sorted_cand_embs = cand_embs[torch.from_numpy(raw_sorted_local).to(device)]

        gt_idx_t = torch.tensor(q["cause_emb_indices"], device=device, dtype=torch.long)
        gt_embs = cause_table_embs.index_select(0, gt_idx_t)
        cos_sorted = gt_embs @ raw_sorted_cand_embs.T
        sem_match = cos_sorted >= args.semantic_threshold
        for gi in range(sem_match.size(0)):
            hits = torch.nonzero(sem_match[gi], as_tuple=False)
            if hits.numel() > 0:
                sem_ranks_all.append(float(int(hits[0].item()) + 1)); sem_cov_all.append(1)
            else:
                sem_ranks_all.append(MISS_RANK); sem_cov_all.append(0)

        if cluster_id_array is not None:
            raw_sorted_global = np.array(candidate_indices)[raw_sorted_local]
            raw_sorted_clusters = cluster_id_array[raw_sorted_global]
            for gi in q["cause_emb_indices"]:
                cid = int(cluster_id_array[int(gi)])
                hits = np.flatnonzero(raw_sorted_clusters == cid)
                if hits.size > 0:
                    cl_ranks_all.append(float(int(hits[0]) + 1)); cl_cov_all.append(1)
                else:
                    cl_ranks_all.append(MISS_RANK); cl_cov_all.append(0)

    sa = np.asarray(sem_ranks_all, dtype=np.float64)
    sem_block = summarize_rank_metric(sa, sem_cov_all)
    m: Dict[str, float] = {"sem_MRR": sem_block["MRR"], "sem_coverage": sem_block["coverage"]}
    sem_recall: Dict[str, float] = {}
    add_recall_at_ks(sem_recall, sa, args.ks)
    for k, v in sem_recall.items():
        m[f"sem_{k}"] = v

    if cluster_id_array is not None and cl_ranks_all:
        ca = np.asarray(cl_ranks_all, dtype=np.float64)
        cl_block = summarize_rank_metric(ca, cl_cov_all)
        m["cl_MRR"] = cl_block["MRR"]
        cl_recall: Dict[str, float] = {}
        add_recall_at_ks(cl_recall, ca, args.ks)
        for k, v in cl_recall.items():
            m[f"cl_{k}"] = v

    m["mean_pool_size"] = float(np.mean(pool_sizes))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse_case_db_dir", type=str, required=True,
                    help="case_db used to compute coarse z (raw, production coarse path)")
    ap.add_argument("--encoder_ckpt", type=str, required=True)
    ap.add_argument("--rvq_root", type=str, required=True)
    ap.add_argument("--fine_case_db_dir", type=str, required=True,
                    help="fine case_db for CEAH (lesion_embs + cause_text_embs)")
    ap.add_argument("--ceah_ckpt", type=str, required=True)
    ap.add_argument("--cluster_json", type=str,
                    default="diagnosis_model/cause_inference/outputs/cause_clusters_llm.json")
    ap.add_argument("--output_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/ceah_compressed_eval")
    ap.add_argument("--query_split", type=str, default="valid",
                    choices=["valid", "test"],
                    help="which split to use as the query set (default valid)")
    # (M, K, light_subdir)
    ap.add_argument("--configs", nargs="+", type=str,
                    default=["4,256,reranker_M4_K256_light",
                             "2,64,reranker_M2_K64_light",
                             "1,16,reranker_M1_K16_light"])
    ap.add_argument("--K_top", type=int, default=50)
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--text_kind", type=str, default="medical",
                    choices=["medical", "colloquial", "none"])
    ap.add_argument("--common_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attribution_mode", type=str, default="softmax",
                    choices=["sigmoid", "softmax"])
    ap.add_argument("--scoring_mode", type=str, default="multiplicative",
                    choices=["single", "multiplicative"])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- coarse path: encoder + z (from raw case_db) ----
    coarse_dir = Path(args.coarse_case_db_dir)
    coarse_train = torch.load(coarse_dir / "train_cases.pt", weights_only=False)
    coarse_valid = torch.load(
        coarse_dir / f"{args.query_split}_cases.pt", weights_only=False,
    )
    encoder, _ = load_encoder(Path(args.encoder_ckpt), device)
    z_train = encode_all(encoder, coarse_train, device).to(device).float()
    z_valid = encode_all(encoder, coarse_valid, device).to(device).float()
    D = z_train.size(-1)
    print(f"[coarse] z_train={tuple(z_train.shape)} z_valid={tuple(z_valid.shape)}")

    # ---- fine path: CEAH inputs ----
    fine_dir = Path(args.fine_case_db_dir)
    fine_train = torch.load(fine_dir / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(
        fine_dir / f"{args.query_split}_cases.pt", weights_only=False,
    )
    cause_pack = torch.load(fine_dir / "cause_text_embs.pt", weights_only=False)
    cause_table_embs = F.normalize(cause_pack["embeddings"].to(device), dim=-1)
    cause_texts = cause_pack["texts"]
    in_dim = cause_table_embs.size(-1)
    assert len(coarse_valid) == len(valid_cases), \
        f"{args.query_split} index misalignment between coarse and fine case_db"
    assert len(coarse_train) == len(fine_train), "train index misalignment"

    cluster_id_array = None
    if args.cluster_json:
        with open(args.cluster_json, encoding="utf-8") as f:
            cl = json.load(f)
        o2c = cl["original_to_cause_id"]
        cluster_id_array = np.array([int(o2c[t]) for t in cause_texts], dtype=np.int64)
        print(f"[cluster] {len(set(cluster_id_array.tolist()))} clusters")

    ceah = CEAH(
        global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
        common_dim=args.common_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
        attribution_mode=args.attribution_mode, scoring_mode=args.scoring_mode,
    ).to(device)
    ceah.load_state_dict(torch.load(args.ceah_ckpt, map_location=device))
    ceah.eval()
    print(f"[ceah] loaded {args.ceah_ckpt}")

    rvq_root = Path(args.rvq_root)
    rows = []

    def run(name, M, K, comp, sim_fn):
        t0 = time.time()
        m = eval_one_source(sim_fn, valid_cases, fine_train, cause_table_embs,
                            cluster_id_array, ceah, args, device, in_dim)
        m.update({"method": name, "M": M, "K": K, "compression_x": comp})
        rows.append(m)
        print(f"[{name:>16}] sem R@10={m.get('sem_R@10',0):.4f} "
              f"cl R@10={m.get('cl_R@10',0):.4f} MRR={m['sem_MRR']:.4f} "
              f"pool={m['mean_pool_size']:.1f}  ({time.time()-t0:.0f}s)")

    # dense reference
    sim_dense = z_valid @ z_train.T
    run("dense", -1, -1, 1.0, lambda qi: sim_dense[qi])

    for spec in args.configs:
        M_s, K_s, sub = spec.split(",")
        M, K = int(M_s), int(K_s)
        rvq = RVQCodebook(M=M, K=K, D=D).to(device)
        pkg = torch.load(rvq_root / f"rvq_M{M}_K{K}" / "codebooks.pt",
                         weights_only=False, map_location=device)
        rvq.codebooks.copy_(pkg["codebooks"].to(device))
        rvq.fitted.copy_(pkg["fitted"].to(device))
        codes_train, z_hat_train, e_train = rvq.encode(z_train)
        e_norm_train = e_train.norm(dim=-1)
        comp = (D * 32) / (M * max(1, math.ceil(math.log2(max(K, 2)))))

        sim_rvq = z_valid @ z_hat_train.T
        run(f"rvq_only_M{M}K{K}", M, K, comp, lambda qi, s=sim_rvq: s[qi])

        light_ckpt = rvq_root / sub / "best.pt"
        if light_ckpt.exists():
            ckpt = torch.load(light_ckpt, weights_only=False, map_location=device)
            cfg = RerankerConfig(**dict(ckpt["reranker_config"]))
            reranker = Reranker(cfg).to(device)
            reranker.load_state_dict(ckpt["reranker_state"])
            sim_light = _rerank_sim(
                reranker, z_valid, z_hat_train, z_train, e_train,
                codes_train, e_norm_train, K_top=args.K_top, variant=cfg.variant,
            )
            run(f"light_M{M}K{K}", M, K, comp, lambda qi, s=sim_light: s[qi])
        else:
            print(f"[light_M{M}K{K}] no checkpoint at {light_ckpt}, skipping")

    with (out_dir / "ceah_compressed_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "rows": rows}, f, ensure_ascii=False, indent=2)

    print("\n=== gamma=0 CEAH on compressed coarse pools (LLM clusters) ===")
    hdr = f"{'method':>16} {'comp×':>8} {'pool':>6} {'sem_R@1':>8} {'sem_R@10':>9} {'sem_MRR':>8} {'cl_R@10':>8} {'cl_MRR':>7}"
    print(hdr)
    for r in rows:
        print(f"{r['method']:>16} {r['compression_x']:>8.0f} {r['mean_pool_size']:>6.1f} "
              f"{r.get('sem_R@1',0):>8.4f} {r.get('sem_R@10',0):>9.4f} {r['sem_MRR']:>8.4f} "
              f"{r.get('cl_R@10',0):>8.4f} {r.get('cl_MRR',0):>7.4f}")
    print(f"\n[save] -> {out_dir}/ceah_compressed_metrics.json")


if __name__ == "__main__":
    main()
