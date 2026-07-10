"""Table 14 (整合式架構與區域門控消融) — all three rows in ONE run.

Computes sem R@10 + cluster R@10 (and R@1/5/20, MRRs, NDCG@5) for the three
settings compared in the thesis integration ablation, under ONE shared protocol
(learned single case-vector aggregation -> bank retrieval -> CEAM cause scoring,
cascade gamma=0 by default), so the numbers are directly comparable:

  分離式基準 (base)  : case_db_base + encoder_base + ceah_base (VLM lesion re-encode,
                       no Region-Gate weights). Standard DeepSets encode_all path.
  OAVLE-Hard (hard)  : encoder_grod_soft + ceah_grod_soft, query built from HARD
                       objectness (sigmoid(obj) > display_thresh -> {0,1}); the SOFT
                       bank_z_soft is reused (the demo's byte-exact hard-gate degenerate).
  OAVLE (soft, main) : encoder_grod_soft + ceah_grod_soft, Region-Gate (gated) weights.

The metric loop is identical to eval_ceah_soft_paper.py; only the per-query
evidence source differs per backend. Conventions (L2-norm, miss=+inf,
occurrence-level cluster, semantic_threshold=0.95, cause_clusters_llm) mirror
eval_ceah.py / phase1_baseline.py.

Run (SDM env, repo root):
  $PY -m diagnosis_model.grod.eval_integration_ablation
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
from diagnosis_model.cause_inference.train_case_encoder import encode_all
from diagnosis_model.cause_inference.phase1_baseline import (
    MISS_RANK, add_recall_at_ks, build_candidate_pool, score_candidates,
    select_positive_top_cases, summarize_rank_metric,
)
from diagnosis_model.grod.train_case_encoder_soft import encode_all_soft, load_soft

ART = Path("data/processed/current/artifacts")


def minmax(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


def ndcg_at_k(rel_in_order: np.ndarray, k: int) -> float:
    rel = rel_in_order[:k].astype(np.float64)
    if rel.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float((rel * discounts).sum())
    ideal = np.sort(rel_in_order)[::-1][:k].astype(np.float64)
    idcg = float((ideal * discounts[: ideal.size]).sum())
    return dcg / idcg if idcg > 0 else 0.0


def _load_ceah(ckpt, in_dim, device):
    ceah = CEAH(global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
                common_dim=256, hidden_dim=512, dropout=0.0,
                attribution_mode="softmax", scoring_mode="multiplicative").to(device).eval()
    ceah.load_state_dict(torch.load(ckpt, map_location=device))
    return ceah


def _load_encoder(ckpt, device):
    pkg = torch.load(ckpt, weights_only=False, map_location="cpu")
    enc = build_encoder(EncoderConfig(**pkg["encoder_config"])).to(device).eval()
    enc.load_state_dict(pkg["encoder_state"])
    return enc


# ----------------------------------------------------------------------------
# Backends: each returns everything the shared metric loop needs. A backend
# exposes a `query(qi)` closure -> (gt, g_slot[D], text_emb[D] or None,
# z_sel[n,D], w_sel[n] or None) for the CEAM evidence of query qi.
# ----------------------------------------------------------------------------
def build_soft_like(name, w_dir, binarize_thresh, top_k_lesions, device):
    """soft / hard: shared encoder_grod_soft + ceah_grod_soft + gated bank."""
    cdir = ART / "db/case_db_jointDistRawP"
    train_cases = torch.load(cdir / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(cdir / "valid_cases.pt", weights_only=False)
    pack = torch.load(cdir / "cause_text_embs.pt", weights_only=False)
    cause_embs = pack["embeddings"].float().to(device)
    in_dim = cause_embs.size(-1)

    enc = _load_encoder(ART / "models/encoder_grod_soft/best_encoder.pt", device)
    ceah = _load_ceah(ART / "models/ceah_grod_soft/best_ceah.pt", in_dim, device)
    bank_z = torch.load(ART / "models/encoder_grod_soft/bank_z_soft.pt",
                        weights_only=False)["bank_z"].to(device)

    g_va, z_va, w_va, _ = load_soft(ART / f"db/{w_dir}/valid.pt")
    if binarize_thresh is not None:
        w_va = (w_va.float() > binarize_thresh).float()  # hard {0,1}
    H_va = encode_all_soft(enc, g_va, z_va, w_va, device).to(device)

    def query(qi):
        gt = valid_cases[qi]["cause_emb_indices"]
        g_slot = g_va[qi].float().to(device)
        text_emb = valid_cases[qi]["text_medical_emb"].float().to(device)
        K = min(top_k_lesions, w_va.size(1))
        idx = w_va[qi].float().to(device).topk(K).indices
        z_sel = z_va[qi].float().to(device).index_select(0, idx)
        w_sel = w_va[qi].float().to(device).index_select(0, idx)
        return gt, g_slot, text_emb, z_sel, w_sel

    return dict(name=name, train_cases=train_cases, n_valid=len(valid_cases),
                cause_embs=cause_embs, cause_texts=pack["texts"], in_dim=in_dim,
                bank_z=bank_z, H_va=H_va, ceah=ceah, query=query)


def build_base(name, top_k_lesions, device):
    """分離式基準: case_db_base + encoder_base + ceah_base (no gate weights)."""
    cdir = ART / "db/case_db_base"
    train_cases = torch.load(cdir / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(cdir / "valid_cases.pt", weights_only=False)
    pack = torch.load(cdir / "cause_text_embs.pt", weights_only=False)
    cause_embs = pack["embeddings"].float().to(device)
    in_dim = cause_embs.size(-1)

    enc = _load_encoder(ART / "models/encoder_base/best_encoder.pt", device)
    ceah = _load_ceah(ART / "models/ceah_base/best_ceah.pt", in_dim, device)
    bank_z = encode_all(enc, train_cases, device).to(device)
    H_va = encode_all(enc, valid_cases, device).to(device)

    def query(qi):
        q = valid_cases[qi]
        gt = q["cause_emb_indices"]
        g_slot = q["global_emb"].float().to(device)
        text_emb = q["text_medical_emb"].float().to(device)
        z_sel = q["lesion_embs"].float().to(device)[:top_k_lesions]  # all case lesions
        return gt, g_slot, text_emb, z_sel, None  # no Region-Gate weights

    return dict(name=name, train_cases=train_cases, n_valid=len(valid_cases),
                cause_embs=cause_embs, cause_texts=pack["texts"], in_dim=in_dim,
                bank_z=bank_z, H_va=H_va, ceah=ceah, query=query)


# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(be, gammas, top_k_cases, sem_thresh, cluster_json, max_queries, device):
    train_cases = be["train_cases"]
    cause_embs = be["cause_embs"]
    cause_embs_n = F.normalize(cause_embs, dim=-1)
    ceah = be["ceah"]
    in_dim = be["in_dim"]
    bank_z = F.normalize(be["bank_z"].float(), dim=-1)
    H_va = F.normalize(be["H_va"].float(), dim=-1)

    cluster_id_array = None
    if cluster_json:
        o2c = json.load(open(cluster_json, encoding="utf-8"))["original_to_cause_id"]
        cluster_id_array = np.array([int(o2c[t]) for t in be["cause_texts"]], dtype=np.int64)

    nq = be["n_valid"] if max_queries < 0 else min(max_queries, be["n_valid"])
    gtags = [f"g={g:.2f}" for g in gammas]
    sem_ranks = {t: [] for t in gtags}; sem_cov = {t: [] for t in gtags}
    cl_ranks = {t: [] for t in gtags}; cl_cov = {t: [] for t in gtags}
    ndcg = {t: [] for t in gtags}

    for qi in range(nq):
        gt, g_slot, text_emb, z_sel, w_sel = be["query"](qi)
        if not gt:
            continue
        n_gt = len(gt)
        sims = (H_va[qi:qi + 1] @ bank_z.T)[0].cpu().numpy()
        top_idx, top_w, _ = select_positive_top_cases(sims, top_k_cases)
        cand = build_candidate_pool(top_idx, train_cases)
        if len(cand) == 0:
            for t in gtags:
                sem_ranks[t] += [MISS_RANK] * n_gt; sem_cov[t] += [0] * n_gt
                cl_ranks[t] += [MISS_RANK] * n_gt; cl_cov[t] += [0] * n_gt
                ndcg[t].append(0.0)
            continue

        cand_t = torch.tensor(cand, device=device, dtype=torch.long)
        cand_embs_n = cause_embs_n.index_select(0, cand_t)
        s1 = score_candidates(cand, top_idx, top_w, train_cases, cause_embs)

        P = len(cand); K = z_sel.size(0)
        text_present = torch.ones(P, dtype=torch.bool, device=device)
        text_slot = text_emb.unsqueeze(0).expand(P, -1)
        lesion_mask = torch.ones((P, K), dtype=torch.bool, device=device)
        lw = None if w_sel is None else w_sel.unsqueeze(0).expand(P, -1).contiguous()
        s_ceah, _, _ = ceah(
            g_slot.unsqueeze(0).expand(P, -1), text_slot, text_present,
            z_sel.unsqueeze(0).expand(P, -1, -1).contiguous(), lesion_mask,
            cause_embs.index_select(0, cand_t), lesion_weights=lw)

        s1n, scn = minmax(s1), minmax(s_ceah)
        gt_embs_n = cause_embs_n.index_select(0, torch.tensor(gt, device=device, dtype=torch.long))
        cand_global = np.array(cand)

        for t, g in zip(gtags, gammas):
            hybrid = g * s1n + (1.0 - g) * scn
            order = torch.argsort(hybrid, descending=True).cpu().numpy()
            sorted_embs_n = cand_embs_n[torch.from_numpy(order).to(device)]
            cos = gt_embs_n @ sorted_embs_n.T
            match = (cos >= sem_thresh).cpu().numpy()
            for gi in range(n_gt):
                hit = np.flatnonzero(match[gi])
                if hit.size:
                    sem_ranks[t].append(float(hit[0]) + 1.0); sem_cov[t].append(1)
                else:
                    sem_ranks[t].append(MISS_RANK); sem_cov[t].append(0)
            ndcg[t].append(ndcg_at_k(match.max(axis=0).astype(np.float64), 5))
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
        sem_m = {}; add_recall_at_ks(sem_m, arr, [1, 3, 5, 10, 20])
        block.update({f"sem_{k}": v for k, v in sem_m.items()})
        block["NDCG@5"] = float(np.mean(ndcg[t])) if ndcg[t] else 0.0
        if cluster_id_array is not None:
            carr = np.asarray(cl_ranks[t], dtype=np.float64)
            cm = summarize_rank_metric(carr, cl_cov[t])
            cl_m = {}; add_recall_at_ks(cl_m, carr, [1, 10, 20])
            block["cl_MRR"] = cm["MRR"]; block.update({f"cl_{k}": v for k, v in cl_m.items()})
        out[t] = block
    return {"n_queries": nq, "metrics_per_gamma": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0],
                    help="Cascade uses gamma=0 (pure CEAM). Pass more for a scan.")
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--top_k_lesions", type=int, default=32)
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--cluster_json", default=str(ART / "cause_clusters_llm.json"))
    ap.add_argument("--display_thresh", type=float, default=None,
                    help="Hard-gate objectness threshold. Default: read thresholds.json.")
    ap.add_argument("--max_queries", type=int, default=-1)
    ap.add_argument("--settings", nargs="+", default=["base", "hard", "soft"],
                    choices=["base", "hard", "soft"])
    ap.add_argument("--output_dir", default=str(ART / "models/ceah_grod_soft/integration_ablation"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    disp = args.display_thresh
    if disp is None:
        disp = json.load(open("data/processed/current/thresholds.json"))["display_thresh"]

    builders = {
        "base": lambda: build_base("分離式基準", args.top_k_lesions, dev),
        "hard": lambda: build_soft_like("OAVLE-Hard", "soft_inputs", disp, args.top_k_lesions, dev),
        "soft": lambda: build_soft_like("OAVLE (soft)", "soft_inputs_gated", None, args.top_k_lesions, dev),
    }

    results = {}
    rows = []
    for key in args.settings:
        be = builders[key]()
        res = evaluate(be, args.gammas, args.top_k_cases, args.semantic_threshold,
                       args.cluster_json, args.max_queries, dev)
        results[key] = {"label": be["name"], **res}
        rows.append((be["name"], res))
        print(f"[{be['name']}] n_valid={res['n_queries']}  "
              f"(cause space: {be['cause_embs'].size(0)} causes)")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    json.dump({"args": {k: v for k, v in vars(args).items()},
               "display_thresh_used": disp, "results": results},
              open(Path(args.output_dir) / "metrics.json", "w"), indent=2, ensure_ascii=False)

    cols = ["sem_R@1", "sem_R@5", "sem_R@10", "sem_R@20", "sem_MRR",
            "cl_R@1", "cl_R@10", "cl_MRR"]
    for gi, g in enumerate(args.gammas):
        t = f"g={g:.2f}"
        print(f"\n=== gamma={g:.2f}  (top_k_cases={args.top_k_cases}, sem_thr={args.semantic_threshold}) ===")
        print(f"{'setting':<16}" + "".join(f"{c:>10}" for c in cols))
        for name, res in rows:
            b = res["metrics_per_gamma"][t]
            print(f"{name:<16}" + "".join(f"{b.get(c, float('nan')):>10.4f}" for c in cols))
    print(f"\nsaved -> {Path(args.output_dir) / 'metrics.json'}")


if __name__ == "__main__":
    main()
