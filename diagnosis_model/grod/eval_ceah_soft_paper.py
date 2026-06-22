"""Paper-grade full-valid eval for the OAVLE soft cascade (Region-Gate weighted).

Unlike eval_ceah_soft_ksweep.py (300-query evidence-selection robustness probe),
this runs the production soft cascade on the FULL valid set and reports the
metrics the thesis Ch5 tables need:

  retrieval  : encoder(g, z, w) -> bank_z top-k cases -> candidate cause pool
  prior s1   : phase1 score_candidates over retrieved cases (the gamma prior)
  CEAH s_c   : soft CEAH (lesion_weights = Region-Gate w, top-K by w)
  ranking    : hybrid = gamma * minmax(s1) + (1-gamma) * minmax(s_c)

Metrics per gamma: sem R@{1,3,5,10,20} + sem MRR, cluster R@{1,10,20} + cluster
MRR (cause_clusters_llm.json), and NDCG@5 (binary relevance = sem-match to any GT).
Conventions (L2-norm, miss=+inf, occurrence-level cluster) mirror eval_ceah.py /
phase1_baseline.py so the numbers are comparable to the rest of the paper.

Run: $PY -m diagnosis_model.grod.eval_ceah_soft_paper
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.models import CEAH
from diagnosis_model.cause_inference.models.case_encoder import EncoderConfig, build_encoder
from diagnosis_model.cause_inference.phase1_baseline import (
    MISS_RANK, add_recall_at_ks, build_candidate_pool, score_candidates,
    select_positive_top_cases, summarize_rank_metric,
)
from diagnosis_model.grod.train_case_encoder_soft import encode_all_soft, load_soft


def minmax(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


def ndcg_at_k(rel_in_order: np.ndarray, k: int) -> float:
    """rel_in_order: binary relevance of candidates in predicted rank order."""
    rel = rel_in_order[:k].astype(np.float64)
    if rel.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float((rel * discounts).sum())
    ideal = np.sort(rel_in_order)[::-1][:k].astype(np.float64)
    idcg = float((ideal * discounts[: ideal.size]).sum())
    return dcg / idcg if idcg > 0 else 0.0


def select_lesions(z_all, w, top_k):
    K = min(top_k, w.numel())
    idx = w.topk(K).indices
    return z_all[idx], w[idx]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", default="diagnosis_model/cause_inference/outputs/case_db_jointDistRawP")
    ap.add_argument("--soft_dir", default="diagnosis_model/grod/outputs/soft_inputs")
    ap.add_argument("--encoder_ckpt", default="diagnosis_model/cause_inference/outputs/encoder_grod_soft/best_encoder.pt")
    ap.add_argument("--bank_path", default="diagnosis_model/cause_inference/outputs/encoder_grod_soft/bank_z_soft.pt")
    ap.add_argument("--ceah_ckpt", default="diagnosis_model/cause_inference/outputs/ceah_grod_soft/best_ceah.pt")
    ap.add_argument("--cluster_json", default="diagnosis_model/cause_inference/outputs/cause_clusters_llm.json")
    ap.add_argument("--output_dir", default="diagnosis_model/cause_inference/outputs/ceah_grod_soft/paper_eval")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--text", choices=["none", "medical", "colloquial"], default="medical",
                    help="CEAH text-evidence slot. 'none' = vision-only (Image+Lesion row).")
    ap.add_argument("--mask_lesions", action="store_true",
                    help="Drop lesion evidence from CEAH scoring (Image+Text row). "
                         "Retrieval pool is unchanged (encoder still uses lesions).")
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--top_k_lesions", type=int, default=32)
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--max_queries", type=int, default=-1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    cdir = Path(args.case_db_dir)
    train_cases = torch.load(cdir / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(cdir / "valid_cases.pt", weights_only=False)
    pack = torch.load(cdir / "cause_text_embs.pt", weights_only=False)
    cause_texts = pack["texts"]
    cause_embs = pack["embeddings"].float().to(device)
    cause_embs_n = F.normalize(cause_embs, dim=-1)
    in_dim = cause_embs.size(-1)

    cluster_id_array = None
    if args.cluster_json:
        o2c = json.load(open(args.cluster_json, encoding="utf-8"))["original_to_cause_id"]
        cluster_id_array = np.array([int(o2c[t]) for t in cause_texts], dtype=np.int64)
        print(f"[cluster] {len(set(cluster_id_array.tolist()))} clusters")

    g_va, z_va, w_va, _ = load_soft(Path(args.soft_dir) / "valid.pt")

    enc_pkg = torch.load(args.encoder_ckpt, weights_only=False, map_location="cpu")
    encoder = build_encoder(EncoderConfig(**enc_pkg["encoder_config"])).to(device).eval()
    encoder.load_state_dict(enc_pkg["encoder_state"])
    bank_z = torch.load(args.bank_path, weights_only=False)["bank_z"].to(device)
    ceah = CEAH(global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
                common_dim=256, hidden_dim=512, dropout=0.0,
                attribution_mode="softmax", scoring_mode="multiplicative").to(device).eval()
    ceah.load_state_dict(torch.load(args.ceah_ckpt, map_location=device))

    H_va = encode_all_soft(encoder, g_va, z_va, w_va, device).to(device)

    nq = len(valid_cases) if args.max_queries < 0 else min(args.max_queries, len(valid_cases))
    gtags = [f"g={g:.2f}" for g in args.gammas]
    sem_ranks = {t: [] for t in gtags}
    sem_cov = {t: [] for t in gtags}
    cl_ranks = {t: [] for t in gtags}
    cl_cov = {t: [] for t in gtags}
    ndcg = {t: [] for t in gtags}

    for qi in range(nq):
        gt = valid_cases[qi]["cause_emb_indices"]
        if not gt:
            continue
        sims = (H_va[qi:qi + 1] @ bank_z.T)[0].cpu().numpy()
        top_idx, top_w, _ = select_positive_top_cases(sims, args.top_k_cases)
        cand = build_candidate_pool(top_idx, train_cases)
        n_gt = len(gt)
        if len(cand) == 0:
            for t in gtags:
                sem_ranks[t] += [MISS_RANK] * n_gt; sem_cov[t] += [0] * n_gt
                cl_ranks[t] += [MISS_RANK] * n_gt; cl_cov[t] += [0] * n_gt
                ndcg[t].append(0.0)
            continue

        cand_t = torch.tensor(cand, device=device, dtype=torch.long)
        cand_embs_n = cause_embs_n.index_select(0, cand_t)
        s1 = score_candidates(cand, top_idx, top_w, train_cases, cause_embs)
        z_sel, w_sel = select_lesions(z_va[qi].float().to(device),
                                      w_va[qi].float().to(device), args.top_k_lesions)
        K = z_sel.size(0)
        P = len(cand)
        if args.text == "none":
            text_emb = torch.zeros(P, in_dim, device=device)
            text_present = torch.zeros(P, dtype=torch.bool, device=device)
        else:
            t = valid_cases[qi][f"text_{args.text}_emb"].float().to(device)
            text_emb = t.unsqueeze(0).expand(P, -1)
            text_present = torch.ones(P, dtype=torch.bool, device=device)
        lesion_mask = torch.full((P, K), not args.mask_lesions, dtype=torch.bool, device=device)
        s_ceah, _, _ = ceah(
            g_va[qi].float().to(device).unsqueeze(0).expand(P, -1),
            text_emb, text_present,
            z_sel.unsqueeze(0).expand(P, -1, -1).contiguous(),
            lesion_mask,
            cause_embs.index_select(0, cand_t),
            lesion_weights=w_sel.unsqueeze(0).expand(P, -1).contiguous())

        s1n, scn = minmax(s1), minmax(s_ceah)
        gt_embs_n = cause_embs_n.index_select(0, torch.tensor(gt, device=device, dtype=torch.long))
        cand_global = np.array(cand)

        for t, g in zip(gtags, args.gammas):
            hybrid = g * s1n + (1.0 - g) * scn
            order = torch.argsort(hybrid, descending=True).cpu().numpy()
            sorted_embs_n = cand_embs_n[torch.from_numpy(order).to(device)]
            cos = gt_embs_n @ sorted_embs_n.T            # [n_gt, P]
            match = (cos >= args.semantic_threshold).cpu().numpy()
            for gi in range(n_gt):
                hit = np.flatnonzero(match[gi])
                if hit.size:
                    sem_ranks[t].append(float(hit[0]) + 1.0); sem_cov[t].append(1)
                else:
                    sem_ranks[t].append(MISS_RANK); sem_cov[t].append(0)
            # candidate-level relevance for NDCG (any-GT sem match), in rank order
            rel = (match.max(axis=0)).astype(np.float64)
            ndcg[t].append(ndcg_at_k(rel, 5))
            if cluster_id_array is not None:
                sorted_clusters = cluster_id_array[cand_global[order]]
                for gi in gt:
                    cid = int(cluster_id_array[int(gi)])
                    hits = np.flatnonzero(sorted_clusters == cid)
                    if hits.size:
                        cl_ranks[t].append(float(hits[0]) + 1.0); cl_cov[t].append(1)
                    else:
                        cl_ranks[t].append(MISS_RANK); cl_cov[t].append(0)

    out = {}
    for t in gtags:
        arr = np.asarray(sem_ranks[t], dtype=np.float64)
        m = summarize_rank_metric(arr, sem_cov[t])
        block = {"sem_MRR": m["MRR"], "sem_coverage": m["coverage"]}
        sem_m = {}
        add_recall_at_ks(sem_m, arr, [1, 3, 5, 10, 20])
        block.update({f"sem_{k}": v for k, v in sem_m.items()})
        block["NDCG@5"] = float(np.mean(ndcg[t])) if ndcg[t] else 0.0
        if cluster_id_array is not None:
            carr = np.asarray(cl_ranks[t], dtype=np.float64)
            cm = summarize_rank_metric(carr, cl_cov[t])
            cl_m = {}
            add_recall_at_ks(cl_m, carr, [1, 10, 20])
            block["cl_MRR"] = cm["MRR"]
            block.update({f"cl_{k}": v for k, v in cl_m.items()})
        out[t] = block

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    json.dump({"n_queries": nq, "metrics_per_gamma": out, "args": vars(args)},
              open(Path(args.output_dir) / "metrics_gammas.json", "w"), indent=2)

    cols = ["sem_R@1", "sem_R@3", "sem_R@5", "sem_R@10", "sem_R@20", "sem_MRR",
            "NDCG@5", "cl_R@1", "cl_R@10", "cl_MRR"]
    print(f"\n{'gamma':<8}" + "".join(f"{c:>10}" for c in cols))
    for t in gtags:
        b = out[t]
        print(f"{t:<8}" + "".join(f"{b.get(c, float('nan')):>10.4f}" for c in cols))


if __name__ == "__main__":
    main()
