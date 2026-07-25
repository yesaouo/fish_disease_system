"""k-sweep with CEAM ranking (does the top_k_cases optimum change vs score_cand?).
Sweep top_k_cases, rank pooled causes with production CEAM (no text), R/P/F1 @ n.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.grod.eval_integration_ablation import build_soft_like
from diagnosis_model.cause_inference.phase1_baseline import (
    select_positive_top_cases, build_candidate_pool,
)

DB = Path("data/processed/current/artifacts/db")
ART = Path("data/processed/current/artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
KS = [1, 2, 3, 4, 5, 7, 10, 15, 20]
NS = [3, 5, 6]
SEM_THR = 0.95
OUT = Path(__file__).parent / "rpf1_k_ceam.json"

FIX = torch.load(DB / "case_db_raw" / "cause_text_embs.pt", weights_only=False)
FIX_EMB = F.normalize(FIX["embeddings"].float().to(DEV), dim=-1)
_o2c = json.load(open(ART / "cause_clusters_llm.json"))["original_to_cause_id"]
CLUSTER = np.array([int(_o2c[t]) for t in FIX["texts"]], dtype=np.int64)


def metric_at(ranked, gt, n):
    m = min(n, len(ranked))
    rk = ranked[:m]
    gt_clu = CLUSTER[np.asarray(gt)]; gset = set(gt_clu.tolist())
    rk_clu = CLUSTER[rk]; rset = set(rk_clu.tolist())
    pc = sum(1 for c in rk_clu if c in gset) / m
    rc = sum(1 for g in gt_clu if g in rset) / len(gt)
    return pc, rc


def main():
    be = build_soft_like("OAVLE", "soft_inputs_gated", None, 32, DEV)
    train_cases = be["train_cases"]; cause_embs = be["cause_embs"]; ceah = be["ceah"]
    bank_z = F.normalize(be["bank_z"].float(), dim=-1); H_va = F.normalize(be["H_va"].float(), dim=-1)
    acc = {k: {n: {"p": [], "r": []} for n in NS} for k in KS}
    for qi in range(be["n_valid"]):
        gt, g_slot, text_emb, z_sel, w_sel = be["query"](qi)
        if not gt:
            continue
        sims = (H_va[qi:qi + 1] @ bank_z.T)[0].cpu().numpy()
        for k in KS:
            ti, tw, _ = select_positive_top_cases(sims, k)
            cand = build_candidate_pool(ti, train_cases)
            if not cand:
                continue
            cand_arr = np.asarray(cand, dtype=np.int64)
            ce = cause_embs.index_select(0, torch.tensor(cand_arr, device=DEV))
            P = len(cand); Kl = z_sel.size(0)
            s, _, _ = ceah(g_slot.unsqueeze(0).expand(P, -1),
                           text_emb.unsqueeze(0).expand(P, -1),
                           torch.zeros(P, dtype=torch.bool, device=DEV),
                           z_sel.unsqueeze(0).expand(P, -1, -1).contiguous(),
                           torch.ones((P, Kl), dtype=torch.bool, device=DEV), ce,
                           lesion_weights=w_sel.unsqueeze(0).expand(P, -1).contiguous())
            ranked = cand_arr[torch.argsort(s, descending=True).cpu().numpy()]
            for n in NS:
                pc, rc = metric_at(ranked, gt, n)
                acc[k][n]["p"].append(pc); acc[k][n]["r"].append(rc)
    res = {}
    for k in KS:
        res[k] = {}
        for n in NS:
            p = np.mean(acc[k][n]["p"]); r = np.mean(acc[k][n]["r"])
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            res[k][n] = {"P": float(p), "R": float(r), "F1": float(f1)}
    json.dump(res, open(OUT, "w"), indent=2)
    print("CEAM 排序 k-sweep, 群集 F1:")
    print("  k   " + " ".join(f"n={n}" for n in NS))
    for k in KS:
        print(f"  {k:>2}  " + " ".join(f"{res[k][n]['F1']:.3f}" for n in NS))
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
