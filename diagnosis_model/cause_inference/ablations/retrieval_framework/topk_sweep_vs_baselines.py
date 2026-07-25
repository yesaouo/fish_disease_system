"""retrieval_framework_rpf1 圖的多-k 版本資料：ours(DeepSets+CEAM) vs 兩條
直接圖文匹配 baseline(影像↔影像檢索 + 影像↔病因文字對齊)，R/P/F1@n(語意)，
k in {3,5,10,20}。輸出 rpf1_multi_k.json 供繪圖。"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.grod.eval_integration_ablation import build_soft_like
from diagnosis_model.cause_inference.phase1_baseline import (
    load_cases, load_train_bank, compute_case_similarities,
    select_positive_top_cases, build_candidate_pool,
)

DB = Path("data/processed/current/artifacts/db")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
KS = [1, 3, 5, 10, 20]
NS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SEM = 0.95
OUT = Path(__file__).parent / "rpf1_multi_k.json"

FIX = torch.load(DB / "case_db_raw" / "cause_text_embs.pt", weights_only=False)
FIX_EMB = F.normalize(FIX["embeddings"].float().to(DEV), dim=-1)
BASE = [("SigLIP2-B", "case_db_raw"), ("CLIP-B", "case_db_raw__clip-vit-base-patch16")]


def rpf(ranked, gt):
    gt_e = FIX_EMB[torch.tensor(gt, device=DEV)]
    cos = FIX_EMB[torch.tensor(ranked, device=DEV)] @ gt_e.T
    out = {}
    for n in NS:
        m = min(n, len(ranked)); tc = cos[:m]
        p = (tc.max(1).values >= SEM).float().mean().item()
        r = (tc.max(0).values >= SEM).sum().item() / len(gt)
        out[n] = (p, r)
    return out


def agg(pq):
    f = lambda p, r: 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    o = {}
    for n in NS:
        p = np.mean([q[n][0] for q in pq]); r = np.mean([q[n][1] for q in pq])
        o[n] = {"P": float(p), "R": float(r), "F1": float(f(p, r))}
    return o


def eval_ours():
    be = build_soft_like("OAVLE", "soft_inputs_gated", None, 32, DEV)
    tc = be["train_cases"]; ce_all = be["cause_embs"]; ceah = be["ceah"]
    bank_z = F.normalize(be["bank_z"].float(), dim=-1); H = F.normalize(be["H_va"].float(), dim=-1)
    pq = {k: [] for k in KS}
    for qi in range(be["n_valid"]):
        gt, g_slot, text_emb, z_sel, w_sel = be["query"](qi)
        if not gt:
            continue
        sims = (H[qi:qi + 1] @ bank_z.T)[0].cpu().numpy()
        for k in KS:
            ti, tw, _ = select_positive_top_cases(sims, k)
            cand = build_candidate_pool(ti, tc)
            if not cand:
                continue
            ca = np.asarray(cand, dtype=np.int64)
            ce = ce_all.index_select(0, torch.tensor(ca, device=DEV))
            P = len(cand); Kl = z_sel.size(0)
            s, _, _ = ceah(g_slot.unsqueeze(0).expand(P, -1),
                           text_emb.unsqueeze(0).expand(P, -1),
                           torch.zeros(P, dtype=torch.bool, device=DEV),
                           z_sel.unsqueeze(0).expand(P, -1, -1).contiguous(),
                           torch.ones((P, Kl), dtype=torch.bool, device=DEV), ce,
                           lesion_weights=w_sel.unsqueeze(0).expand(P, -1).contiguous())
            ranked = ca[torch.argsort(s, descending=True).cpu().numpy()]
            pq[k].append(rpf(ranked, gt))
    return {k: agg(pq[k]) for k in KS}


def eval_imgtext(case_db):
    pack = torch.load(DB / case_db / "cause_text_embs.pt", weights_only=False)
    ctxt = F.normalize(pack["embeddings"].float().to(DEV), dim=-1)
    trc, G, L, offs = load_train_bank(DB / case_db, DEV)
    queries = load_cases(DB / case_db, "valid")
    pq = {k: [] for k in KS}
    for q in queries:
        gt = list(q["cause_emb_indices"])
        if not gt:
            continue
        qg = F.normalize(q["global_emb"].float().to(DEV), dim=-1)
        sims = compute_case_similarities(qg, qg.unsqueeze(0), G, L, offs, 1.0, 0.0, lesion_match="max_mean")
        for k in KS:
            ti, _, _ = select_positive_top_cases(sims, k)
            cand = build_candidate_pool(ti, trc)
            if not cand:
                continue
            ca = np.asarray(cand, dtype=np.int64)
            align = ctxt.index_select(0, torch.tensor(ca, device=DEV)) @ qg
            ranked = ca[torch.argsort(align, descending=True).cpu().numpy()]
            pq[k].append(rpf(ranked, gt))
    return {k: agg(pq[k]) for k in KS}


def main():
    res = {"ns": NS, "ks": KS, "methods": {}}
    print("[ours]"); res["methods"]["OAVLE＋CEAM（40M）"] = eval_ours()
    for lab, cdb in BASE:
        print(f"[{lab}]"); res["methods"][lab] = eval_imgtext(cdb)
    json.dump(res, open(OUT, "w"), indent=2, ensure_ascii=False)
    for k in KS:
        print(f"k={k}: " + " ".join(
            f"{m.split('（')[0]} F1@5={res['methods'][m][k]['5']['F1']:.3f}"
            for m in res["methods"]))
    print(f"[save] {OUT}")


if __name__ == "__main__":
    main()
