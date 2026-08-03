"""Correct framework comparison (same task):

  CLIP/SigLIP2 baseline = image->image case retrieval + IMAGE<->CAUSE-TEXT
  alignment to rank the pooled causes (this is what the 1:1 / 1:many contrastive
  loss actually trains).

  Ours (OAVLE+case-based) = production DeepSets retrieval + case-based cause
  aggregation (score_candidates).

R/P/F1 @ n (# causes shown), swept k in {3,5}. Relevance in fixed SigLIP2 768-d
space (cos>=0.95) AND cluster (LLM taxonomy).
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.grod.eval_integration_ablation import build_soft_like
from diagnosis_model.cause_inference.phase1_baseline import (
    load_cases, load_train_bank, compute_case_similarities,
    select_positive_top_cases, build_candidate_pool, score_candidates,
)

DB = Path("data/processed/current/artifacts/db")
ART = Path("data/processed/current/artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEM_THR = 0.95
NS = [1, 2, 3, 4, 5, 6, 8, 10]
KS_CASES = [3, 5]
OUT = Path(__file__).parent / "rpf1_imgtext.json"

FIX = torch.load(DB / "case_db_raw" / "cause_text_embs.pt", weights_only=False)
FIX_TEXTS = FIX["texts"]
FIX_EMB = F.normalize(FIX["embeddings"].float().to(DEV), dim=-1)
_o2c = json.load(open(ART / "cause_clusters_llm.json"))["original_to_cause_id"]
CLUSTER = np.array([int(_o2c[t]) for t in FIX_TEXTS], dtype=np.int64)

BASELINES = [("SigLIP2-B", "case_db_raw"), ("CLIP-B", "case_db_raw__clip-vit-base-patch16")]


def metrics_from_ranked(ranked_global, gt):
    gt_e = FIX_EMB[torch.tensor(gt, device=DEV)]
    rk_e = FIX_EMB[torch.tensor(ranked_global, device=DEV)]
    cos = rk_e @ gt_e.T
    rk_clu = CLUSTER[ranked_global]; gt_clu = CLUSTER[np.asarray(gt)]
    gt_clu_set = set(gt_clu.tolist()); pool = len(ranked_global)
    res = {}
    for n in NS:
        m = min(n, pool); tc = cos[:m]
        p_s = (tc.max(1).values >= SEM_THR).float().mean().item()
        r_s = (tc.max(0).values >= SEM_THR).sum().item() / len(gt)
        top_clu = rk_clu[:m]; top_set = set(top_clu.tolist())
        p_c = sum(1 for ci in top_clu if ci in gt_clu_set) / m
        r_c = sum(1 for gc in gt_clu if gc in top_set) / len(gt)
        res[n] = (p_s, r_s, p_c, r_c)
    return res


def agg(pq):
    f = lambda p, r: 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    out = {"semantic": {}, "cluster": {}}
    for n in NS:
        ps = np.mean([q[n][0] for q in pq]); rs = np.mean([q[n][1] for q in pq])
        pc = np.mean([q[n][2] for q in pq]); rc = np.mean([q[n][3] for q in pq])
        out["semantic"][n] = {"P": float(ps), "R": float(rs), "F1": float(f(ps, rs))}
        out["cluster"][n] = {"P": float(pc), "R": float(rc), "F1": float(f(pc, rc))}
    return out


def eval_ours(k):
    be = build_soft_like("OAVLE", "soft_inputs_gated", None, 32, DEV)
    train_cases = be.train_cases; cause_embs = be.cause_embs
    bank_z = F.normalize(be.bank_z.float(), dim=-1); H_va = F.normalize(be.H.float(), dim=-1)
    pq = []
    for qi in range(be.n_queries):
        gt, *_ = be.query(qi)
        if not gt:
            continue
        sims = (H_va[qi:qi + 1] @ bank_z.T)[0].cpu().numpy()
        ti, tw, _ = select_positive_top_cases(sims, k)
        cand = build_candidate_pool(ti, train_cases)
        if not cand:
            continue
        s = score_candidates(cand, ti, tw, train_cases, cause_embs)
        order = torch.argsort(s, descending=True).cpu().numpy()
        pq.append(metrics_from_ranked(np.asarray(cand)[order], gt))
    return agg(pq)


def eval_imgtext(case_db, k):
    """Image->image retrieval + image<->cause-text alignment ranking."""
    pack = torch.load(DB / case_db / "cause_text_embs.pt", weights_only=False)
    cause_txt = F.normalize(pack["embeddings"].float().to(DEV), dim=-1)   # that VLM's text space
    tr_cases, G, Lst, offs = load_train_bank(DB / case_db, DEV)           # G = normalized image globals
    queries = load_cases(DB / case_db, "valid")
    pq = []
    for q in queries:
        gt = list(q["cause_emb_indices"])
        if not gt:
            continue
        qg = F.normalize(q["global_emb"].float().to(DEV), dim=-1)
        # (1) image->image case retrieval (global only)
        sims = compute_case_similarities(qg, qg.unsqueeze(0), G, Lst, offs, 1.0, 0.0, lesion_match="max_mean")
        ti, _, _ = select_positive_top_cases(sims, k)
        cand = build_candidate_pool(ti, tr_cases)
        if not cand:
            continue
        cand_arr = np.asarray(cand, dtype=np.int64)
        # (2) rank pooled causes by IMAGE <-> CAUSE-TEXT alignment
        align = (cause_txt.index_select(0, torch.tensor(cand_arr, device=DEV)) @ qg)  # [pool]
        order = torch.argsort(align, descending=True).cpu().numpy()
        pq.append(metrics_from_ranked(cand_arr[order], gt))
    return agg(pq)


def main():
    res = {"ns": NS, "ks": KS_CASES, "methods": {}}
    for k in KS_CASES:
        print(f"===== k={k} =====")
        res["methods"][f"Ours (OAVLE) k={k}"] = eval_ours(k)
        for label, cdb in BASELINES:
            res["methods"][f"{label} k={k}"] = eval_imgtext(cdb, k)
        for m in [f"Ours (OAVLE) k={k}", f"SigLIP2-B k={k}", f"CLIP-B k={k}"]:
            d = res["methods"][m]
            print(f"  {m:20s} F1(clu)@n: " + " ".join(f"{n}:{d['cluster'][n]['F1']:.3f}" for n in NS))
    json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
