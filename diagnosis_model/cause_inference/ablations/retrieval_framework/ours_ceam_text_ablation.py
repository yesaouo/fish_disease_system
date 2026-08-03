"""Does adding clinical TEXT help ours? Production CEAM (has a text slot) as the
cause scorer, with vs without text_medical_emb, at k=3. Compared to ours
(score_candidates, image-only) already on disk.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.grod.eval_integration_ablation import build_soft_like
from diagnosis_model.cause_inference.phase1_baseline import (
    select_positive_top_cases, build_candidate_pool, score_candidates,
)

DB = Path("data/processed/current/artifacts/db")
ART = Path("data/processed/current/artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 3
SEM_THR = 0.95
NS = [1, 2, 3, 4, 5, 6, 8, 10]
OUT = Path(__file__).parent / "rpf1_text.json"

FIX = torch.load(DB / "case_db_raw" / "cause_text_embs.pt", weights_only=False)
FIX_EMB = F.normalize(FIX["embeddings"].float().to(DEV), dim=-1)
_o2c = json.load(open(ART / "cause_clusters_llm.json"))["original_to_cause_id"]
CLUSTER = np.array([int(_o2c[t]) for t in FIX["texts"]], dtype=np.int64)


def metrics(ranked, gt):
    gt_e = FIX_EMB[torch.tensor(gt, device=DEV)]
    cos = FIX_EMB[torch.tensor(ranked, device=DEV)] @ gt_e.T
    rk_clu = CLUSTER[ranked]; gt_clu = CLUSTER[np.asarray(gt)]; gset = set(gt_clu.tolist())
    r = {}
    for n in NS:
        m = min(n, len(ranked)); tc = cos[:m]
        ps = (tc.max(1).values >= SEM_THR).float().mean().item()
        rs = (tc.max(0).values >= SEM_THR).sum().item() / len(gt)
        tclu = rk_clu[:m]; tset = set(tclu.tolist())
        pc = sum(1 for c in tclu if c in gset) / m
        rc = sum(1 for g in gt_clu if g in tset) / len(gt)
        r[n] = (ps, rs, pc, rc)
    return r


def agg(pq):
    f = lambda p, r: 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    out = {"semantic": {}, "cluster": {}}
    for n in NS:
        ps = np.mean([q[n][0] for q in pq]); rs = np.mean([q[n][1] for q in pq])
        pc = np.mean([q[n][2] for q in pq]); rc = np.mean([q[n][3] for q in pq])
        out["semantic"][n] = {"P": float(ps), "R": float(rs), "F1": float(f(ps, rs))}
        out["cluster"][n] = {"P": float(pc), "R": float(rc), "F1": float(f(pc, rc))}
    return out


def main():
    be = build_soft_like("OAVLE", "soft_inputs_gated", None, 32, DEV)
    train_cases = be.train_cases; cause_embs = be.cause_embs; ceah = be.ceah
    bank_z = F.normalize(be.bank_z.float(), dim=-1); H_va = F.normalize(be.H.float(), dim=-1)
    pq_sc, pq_notext, pq_text = [], [], []
    for qi in range(be.n_queries):
        gt, g_slot, text_emb, z_sel, w_sel = be.query(qi)
        if not gt:
            continue
        sims = (H_va[qi:qi + 1] @ bank_z.T)[0].cpu().numpy()
        ti, tw, _ = select_positive_top_cases(sims, K)
        cand = build_candidate_pool(ti, train_cases)
        if not cand:
            continue
        cand_arr = np.asarray(cand, dtype=np.int64)
        cand_t = torch.tensor(cand_arr, device=DEV)
        ce = cause_embs.index_select(0, cand_t)
        P = len(cand); Kl = z_sel.size(0)
        g = g_slot.unsqueeze(0).expand(P, -1)
        z = z_sel.unsqueeze(0).expand(P, -1, -1).contiguous()
        lm = torch.ones((P, Kl), dtype=torch.bool, device=DEV)
        lw = w_sel.unsqueeze(0).expand(P, -1).contiguous()
        tslot = text_emb.unsqueeze(0).expand(P, -1)
        # score_candidates (image-only, current "ours")
        s_sc = score_candidates(cand, ti, tw, train_cases, cause_embs)
        pq_sc.append(metrics(cand_arr[torch.argsort(s_sc, descending=True).cpu().numpy()], gt))
        # CEAM no text
        s0, _, _ = ceah(g, tslot, torch.zeros(P, dtype=torch.bool, device=DEV), z, lm, ce, lesion_weights=lw)
        pq_notext.append(metrics(cand_arr[torch.argsort(s0, descending=True).cpu().numpy()], gt))
        # CEAM + text
        s1, _, _ = ceah(g, tslot, torch.ones(P, dtype=torch.bool, device=DEV), z, lm, ce, lesion_weights=lw)
        pq_text.append(metrics(cand_arr[torch.argsort(s1, descending=True).cpu().numpy()], gt))

    res = {"Ours (score_cand)": agg(pq_sc), "Ours CEAM no-text": agg(pq_notext),
           "Ours CEAM +text": agg(pq_text)}
    json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)
    for m, d in res.items():
        print(f"{m:20s} F1(clu)@n: " + " ".join(f"{n}:{d['cluster'][n]['F1']:.3f}" for n in NS))
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
