"""End-to-end hybrid eval: case-retrieval (Phase 1 OR trained encoder) +
CEAH attribution + γ-weighted re-rank.

Pipeline is identical to eval_ceah.py except the case-similarity scorer can
be swapped. The downstream candidate-pool aggregation, CEAH forward pass,
γ-hybrid scoring, and semantic-rank metric are all reused unchanged.

Compared methods:
    - phase1-hungarian + CEAH    (paper baseline, ~78.65% sem_R@10 at γ=0.75)
    - <encoder> + CEAH           (Mamba / mean / DeepSets via distillation)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.models import CEAH
from diagnosis_model.cause_inference.models.mamba_encoder import (
    EncoderConfig, build_encoder,
)
from diagnosis_model.cause_inference.phase1_baseline import (
    build_candidate_pool,
    compute_case_similarities,
    load_case_db,
    score_candidates,
    select_positive_top_cases,
    stack_train_lesions,
)
from diagnosis_model.cause_inference.train_mamba_encoder import encode_all
from diagnosis_model.cause_inference.eval_ceah import (
    ceah_forward_for_pool,
    minmax_norm,
)


MISS_RANK = float("inf")


def evaluate_hybrid(
    score_fn: Callable[[int, dict], np.ndarray],
    valid_cases: list,
    train_cases: list,
    cause_table_embs: torch.Tensor,
    ceah,
    in_dim: int,
    gammas: List[float],
    top_k_cases: int = 20,
    semantic_threshold: float = 0.95,
    text_kind: str = "medical",
    Ks: List[int] = (1, 5, 10, 20, 100),
    device: torch.device = torch.device("cuda"),
) -> Dict[str, dict]:
    cause_table_embs = F.normalize(cause_table_embs.to(device), dim=-1)
    out_per_gamma = {f"g={g:.2f}": {"ranks": [], "cov": []} for g in gammas}
    pool_sizes = []

    t0 = time.time()
    for qi, q in enumerate(valid_cases):
        sims = score_fn(qi, q)                                        # [N_train]
        top_k_idx, top_k_w, _ = select_positive_top_cases(sims, top_k_cases)
        candidate_indices = build_candidate_pool(top_k_idx, train_cases)
        pool_size = len(candidate_indices)
        pool_sizes.append(pool_size)

        gt_cause_idx = q["cause_emb_indices"]

        if pool_size == 0:
            for g in gammas:
                key = f"g={g:.2f}"
                for _ in gt_cause_idx:
                    out_per_gamma[key]["ranks"].append(MISS_RANK)
                    out_per_gamma[key]["cov"].append(0)
            continue

        cand_idx_t = torch.tensor(candidate_indices, device=device, dtype=torch.long)
        cand_embs = cause_table_embs.index_select(0, cand_idx_t)

        s1 = score_candidates(
            candidate_indices, top_k_idx, top_k_w, train_cases, cause_table_embs,
        )
        s_ceah, _alpha, _ev_mask = ceah_forward_for_pool(
            ceah, q, candidate_indices, cause_table_embs, in_dim, device,
            use_text_kind=text_kind,
        )
        s1_n = minmax_norm(s1)
        sc_n = minmax_norm(s_ceah)

        gt_idx_t = torch.tensor(gt_cause_idx, device=device, dtype=torch.long)
        gt_embs = cause_table_embs.index_select(0, gt_idx_t)

        for g in gammas:
            key = f"g={g:.2f}"
            hybrid = g * s1_n + (1.0 - g) * sc_n                       # [P]
            sorted_local = torch.argsort(hybrid, descending=True).cpu().numpy()
            sorted_cand_embs = cand_embs[
                torch.from_numpy(sorted_local).to(device)
            ]
            cos_sorted = gt_embs @ sorted_cand_embs.T
            sem_match = cos_sorted >= semantic_threshold

            for gi in range(sem_match.size(0)):
                hits = torch.nonzero(sem_match[gi], as_tuple=False)
                if hits.numel() > 0:
                    out_per_gamma[key]["ranks"].append(float(hits[0].item()) + 1)
                    out_per_gamma[key]["cov"].append(1)
                else:
                    out_per_gamma[key]["ranks"].append(MISS_RANK)
                    out_per_gamma[key]["cov"].append(0)

    elapsed = time.time() - t0

    summary = {"mean_pool_size": float(np.mean(pool_sizes)),
               "eval_time_s": elapsed,
               "per_query_ms": elapsed / max(1, len(valid_cases)) * 1000,
               "per_gamma": {}}
    for key, agg in out_per_gamma.items():
        ranks = np.array(agg["ranks"], dtype=np.float64)
        finite = np.isfinite(ranks)
        reciprocal = np.where(finite, 1.0 / np.where(finite, ranks, 1.0), 0.0)
        row = {f"sem_R@{k}": float((ranks <= k).mean()) for k in Ks}
        row["sem_MRR"] = float(reciprocal.mean())
        row["coverage"] = float(np.mean(agg["cov"]))
        summary["per_gamma"][key] = row
    return summary


def make_phase1_scorer(train_cases, alpha=0.25, beta=0.75, lesion_match="hungarian",
                       device=torch.device("cuda")):
    G_t = F.normalize(torch.stack([c["global_emb"] for c in train_cases]).to(device), dim=-1)
    L_t, off = stack_train_lesions(train_cases)
    L_t = F.normalize(L_t.to(device), dim=-1)

    def score(qi, q):
        g = F.normalize(q["global_emb"].to(device), dim=-1)
        L = q["lesion_embs"].to(device)
        if L.size(0) > 0:
            L = F.normalize(L, dim=-1)
        return compute_case_similarities(g, L, G_t, L_t, off, alpha, beta, lesion_match)
    return score


def make_encoder_scorer(checkpoint_path, train_cases, valid_cases, device):
    pkg = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    cfg_dict = pkg["encoder_config"]
    cfg_dict["dtype"] = torch.bfloat16
    cfg = EncoderConfig(**cfg_dict)
    enc = build_encoder(cfg).to(device)
    enc.load_state_dict(pkg["encoder_state"])
    enc.eval()

    H_train = F.normalize(encode_all(enc, train_cases, device).float(), dim=-1).to(device)
    H_valid = F.normalize(encode_all(enc, valid_cases, device).float(), dim=-1).to(device)
    sim_full = (H_valid @ H_train.T).cpu().numpy()

    def score(qi, q):
        return sim_full[qi]
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db")
    ap.add_argument("--ceah_ckpt", type=str,
                    default="diagnosis_model/cause_inference/outputs/ceah_v3/best_ceah.pt")
    ap.add_argument("--include_phase1", action="store_true")
    ap.add_argument("--checkpoints", type=str, nargs="+", default=[],
                    help="space-separated name=path pairs for encoder checkpoints")
    # CEAH hyperparams (must match training config of ceah_v3)
    ap.add_argument("--common_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attribution_mode", type=str, default="softmax")
    ap.add_argument("--scoring_mode", type=str, default="multiplicative")
    ap.add_argument("--text_kind", type=str, default="medical",
                    choices=["medical", "colloquial", "none"])
    # Eval hyperparams
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 5, 10, 20, 100])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--min_query_lesions", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_cases, valid_cases, cause_pkg, _ = load_case_db(Path(args.case_db_dir))
    cause_embs = cause_pkg["embeddings"]
    in_dim = cause_embs.size(-1)
    if args.min_query_lesions > 0:
        before = len(valid_cases)
        valid_cases = [c for c in valid_cases
                       if c["lesion_embs"].size(0) >= args.min_query_lesions]
        print(f"[filter] valid >= {args.min_query_lesions} lesions: "
              f"{len(valid_cases)} / {before}")
    print(f"train={len(train_cases)} valid={len(valid_cases)}  K={args.top_k_cases}  "
          f"text_kind={args.text_kind}")

    ceah = CEAH(
        global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
        common_dim=args.common_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
        attribution_mode=args.attribution_mode, scoring_mode=args.scoring_mode,
    ).to(device)
    ceah.load_state_dict(torch.load(args.ceah_ckpt, map_location=device))
    ceah.eval()
    print(f"[ceah] loaded {args.ceah_ckpt}")

    results: Dict[str, dict] = {}

    if args.include_phase1:
        print("\n[phase1-hungarian] running ...")
        scorer = make_phase1_scorer(train_cases, device=device)
        results["phase1+ceah"] = evaluate_hybrid(
            scorer, valid_cases, train_cases, cause_embs, ceah, in_dim,
            gammas=args.gammas, top_k_cases=args.top_k_cases,
            semantic_threshold=args.semantic_threshold, text_kind=args.text_kind,
            Ks=tuple(args.Ks), device=device,
        )

    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        print(f"\n[{name}+ceah] loading {path} ...")
        scorer = make_encoder_scorer(Path(path), train_cases, valid_cases, device)
        results[f"{name}+ceah"] = evaluate_hybrid(
            scorer, valid_cases, train_cases, cause_embs, ceah, in_dim,
            gammas=args.gammas, top_k_cases=args.top_k_cases,
            semantic_threshold=args.semantic_threshold, text_kind=args.text_kind,
            Ks=tuple(args.Ks), device=device,
        )

    # Pretty-print
    print("\n" + "=" * 88)
    print(f"{'method':<22s} {'γ':>5s} | "
          + " ".join(f"R@{k:<3d}" for k in args.Ks) + f" |  MRR  | per-q ms")
    print("-" * 88)
    for name, m in results.items():
        for g in args.gammas:
            key = f"g={g:.2f}"
            row = m["per_gamma"][key]
            cells = " ".join(f"{row[f'sem_R@{k}']:.3f}" for k in args.Ks)
            print(f"{name:<22s} {g:>5.2f} | {cells} | {row['sem_MRR']:.3f} | "
                  f"{m['per_query_ms']:6.1f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
