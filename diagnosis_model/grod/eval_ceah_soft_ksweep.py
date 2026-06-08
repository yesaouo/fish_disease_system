"""Eval-time sweep over the CEAH lesion-evidence selection, holding the soft
retrieval (encoder_grod_soft + bank_z_soft) and the trained soft CEAH FIXED.

For the SAME soft CEAH + same candidate pools, vary only the evidence fed to
CEAH:
  thresh@<t> : keep queries with w>t, weight=1 (mimics gpu_infer.py hard select)
  topK       : top-K queries by w, soft weights (training used top-32)
  all        : all 300 queries, soft weights

Caveat: the model was *trained* at top-32 soft, so thresh/all are slightly
out-of-distribution for it — this tests robustness of the trained model, not a
per-regime-retrained design comparison. γ-free cascade.

Run: $PY -m diagnosis_model.grod.eval_ceah_soft_ksweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.models import CEAH
from diagnosis_model.cause_inference.phase1_baseline import (
    MISS_RANK, add_recall_at_ks, build_candidate_pool, summarize_rank_metric,
)
from diagnosis_model.cause_inference.models.case_encoder import EncoderConfig, build_encoder
from diagnosis_model.grod.train_case_encoder_soft import encode_all_soft, load_soft


def select(z_all, w, mode):
    """-> (z_sel[k,D], w_sel[k]) per mode."""
    if mode.startswith("thresh"):
        t = float(mode.split("@")[1])
        keep = w > t
        if keep.sum() == 0:
            keep = torch.zeros_like(w, dtype=torch.bool); keep[w.argmax()] = True
        return z_all[keep], torch.ones(int(keep.sum()), device=w.device)   # binary weight
    if mode == "all":
        return z_all, w
    K = min(int(mode[3:]), w.numel())                                      # "topK"
    idx = w.topk(K).indices
    return z_all[idx], w[idx]


@torch.no_grad()
def eval_mode(ceah, H_va, bank_z, train_cases, valid_cases, cause_embs,
              g_va, z_va, w_va, device, mode, top_k_cases=20, max_queries=300,
              semantic_threshold=0.95, ks=(1, 5, 10, 20, 100)):
    nq = min(max_queries, len(valid_cases))
    sem_ranks, cov, n_les = [], [], []
    for qi in range(nq):
        gt = valid_cases[qi]["cause_emb_indices"]
        if not gt:
            continue
        sims = H_va[qi:qi + 1].to(device) @ bank_z.T
        top_idx = sims[0].topk(top_k_cases).indices.cpu().numpy()
        cand_idx = build_candidate_pool(top_idx, train_cases)
        P = len(cand_idx)
        if P == 0:
            for _ in gt:
                sem_ranks.append(MISS_RANK); cov.append(0)
            continue
        cand_embs = cause_embs[torch.as_tensor(cand_idx, device=device)]
        z_s, w_s = select(z_va[qi].float().to(device), w_va[qi].float().to(device), mode)
        K = z_s.size(0); n_les.append(K)
        scores, _, _ = ceah(
            g_va[qi].float().to(device).unsqueeze(0).expand(P, -1),
            torch.zeros(P, cand_embs.size(-1), device=device),
            torch.zeros(P, dtype=torch.bool, device=device),
            z_s.unsqueeze(0).expand(P, -1, -1).contiguous(),
            torch.ones(P, K, dtype=torch.bool, device=device),
            cand_embs, lesion_weights=w_s.unsqueeze(0).expand(P, -1).contiguous())
        order = scores.argsort(descending=True).cpu().numpy()
        sorted_cand = cand_embs[torch.as_tensor(order[: max(ks)], device=device)]
        gt_e = F.normalize(cause_embs[torch.as_tensor(gt, device=device)], dim=-1)
        cos = gt_e @ F.normalize(sorted_cand, dim=-1).T
        match = cos >= semantic_threshold
        for gi in range(match.size(0)):
            hit = torch.nonzero(match[gi], as_tuple=False)
            if hit.numel() > 0:
                sem_ranks.append(float(hit[0].item()) + 1.0); cov.append(1)
            else:
                sem_ranks.append(MISS_RANK); cov.append(0)
    arr = np.asarray(sem_ranks, dtype=np.float64)
    block = summarize_rank_metric(arr, cov)
    m = {"sem_MRR": block["MRR"], "cov": block["coverage"],
         "avg_les": float(np.mean(n_les)) if n_les else 0.0}
    add_recall_at_ks(m, arr, list(ks))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", default="diagnosis_model/cause_inference/outputs/case_db_jointDistRawP")
    ap.add_argument("--soft_dir", default="diagnosis_model/grod/outputs/soft_inputs")
    ap.add_argument("--encoder_ckpt", default="diagnosis_model/cause_inference/outputs/encoder_grod_soft/best_encoder.pt")
    ap.add_argument("--bank_path", default="diagnosis_model/cause_inference/outputs/encoder_grod_soft/bank_z_soft.pt")
    ap.add_argument("--ceah_ckpt", default="diagnosis_model/cause_inference/outputs/ceah_grod_soft/best_ceah.pt")
    ap.add_argument("--modes", nargs="+", default=["thresh@0.3", "top16", "top32", "top64", "all"])
    ap.add_argument("--max_queries", type=int, default=300)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    train_cases = torch.load(Path(args.case_db_dir) / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(Path(args.case_db_dir) / "valid_cases.pt", weights_only=False)
    cause_embs = torch.load(Path(args.case_db_dir) / "cause_text_embs.pt", weights_only=False)["embeddings"].float().to(device)
    in_dim = cause_embs.size(-1)
    g_va, z_va, w_va, _ = load_soft(Path(args.soft_dir) / "valid.pt")

    enc_pkg = torch.load(args.encoder_ckpt, weights_only=False, map_location="cpu")
    encoder = build_encoder(EncoderConfig(**enc_pkg["encoder_config"])).to(device).eval()
    encoder.load_state_dict(enc_pkg["encoder_state"])
    bank_z = torch.load(args.bank_path, weights_only=False)["bank_z"].to(device)
    ceah = CEAH(global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
                common_dim=256, hidden_dim=512, dropout=0.0,
                attribution_mode="softmax", scoring_mode="multiplicative").to(device).eval()
    ceah.load_state_dict(torch.load(args.ceah_ckpt, map_location=device))

    H_va = encode_all_soft(encoder, g_va, z_va, w_va, device)              # retrieval fixed across modes

    print(f"{'mode':<12}{'avg_les':>8}{'sem_MRR':>9}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'R@20':>8}{'cov':>8}")
    for mode in args.modes:
        m = eval_mode(ceah, H_va, bank_z, train_cases, valid_cases, cause_embs,
                      g_va, z_va, w_va, device, mode, max_queries=args.max_queries)
        print(f"{mode:<12}{m['avg_les']:>8.1f}{m['sem_MRR']:>9.4f}{m['R@1']:>8.4f}"
              f"{m['R@5']:>8.4f}{m['R@10']:>8.4f}{m['R@20']:>8.4f}{m['cov']:>8.4f}")


if __name__ == "__main__":
    main()
