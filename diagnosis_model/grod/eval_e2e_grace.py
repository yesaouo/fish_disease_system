"""Compare end-to-end variants vs baseline sigmoid on case->cause retrieval.

Paper-protocol retrieval-only eval: scores candidate causes with phase1
``score_candidates`` over the FULL valid set, sem R@k (cos>=0.95 coverage) + MRR
and LLM-cluster R@k (cause_clusters_llm.json) — the SAME conventions as
eval_ceah_soft_paper.py / phase1_baseline.py, so numbers are directly comparable
to the Ch5 retrieval table (OAVLE soft single-vector sem R@10 = 45.46%).

Incremental end-to-end scope (paper §multi-task joint training ablation):

  baseline      = encoder_grod_soft        + frozen sigmoid   (= production soft start)
  sigmoidctrl   = grace_e2e_sigmoidctrl agg + frozen sigmoid   (aggregator trained more)
  L1            = grace_e2e agg            + Region Gate       (production e2e)
  L2            = grace_e2e_l2 agg+heads   + Region Gate       (heads on hs; ablation)

  baseline -> sigmoidctrl : effect of extra aggregator training alone
  sigmoidctrl -> L1       : ISOLATED Region Gate effect
  L2 vs L1                : whether end-to-end should extend into the detector heads

Run both regimes:
  $PY -m diagnosis_model.grod.eval_e2e_grace --top_k_cases 20   # production (buffered)
  $PY -m diagnosis_model.grod.eval_e2e_grace --top_k_cases 1    # stress (no agg buffer)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.models.case_encoder import EncoderConfig, build_encoder
from diagnosis_model.cause_inference.train_case_encoder import encode_all
from diagnosis_model.cause_inference.phase1_baseline import (
    MISS_RANK, add_recall_at_ks, build_candidate_pool, score_candidates,
    select_positive_top_cases, summarize_rank_metric,
)
from diagnosis_model.grod.train_case_encoder_soft import load_soft
from diagnosis_model.grod.region_gate import RegionGate
from diagnosis_model.grod.finetune_e2e_grace import recover_logits, encode_bank
from diagnosis_model.grod.finetune_e2e_grace_l2 import clone_heads, encode_bank as encode_bank_l2


def paper_retrieval_eval(bank, H_va, train_cases, valid_cases, cause_embs, cause_embs_n,
                         cluster_id_array, device, top_k_cases, sem_thresh=0.95):
    """Retrieval-only cause ranking (s1 / score_candidates), full valid set.
    Mirrors eval_ceah_soft_paper.py gamma=1.0 path (no CEAH)."""
    sem_ranks, sem_cov, cl_ranks, cl_cov = [], [], [], []
    for qi in range(len(valid_cases)):
        gt = valid_cases[qi]["cause_emb_indices"]
        if not gt:
            continue
        n_gt = len(gt)
        sims = (H_va[qi:qi + 1].to(device) @ bank.T.to(device))[0].cpu().numpy()
        top_idx, top_w, _ = select_positive_top_cases(sims, top_k_cases)
        cand = build_candidate_pool(top_idx, train_cases)
        if len(cand) == 0:
            sem_ranks += [MISS_RANK] * n_gt; sem_cov += [0] * n_gt
            cl_ranks += [MISS_RANK] * n_gt; cl_cov += [0] * n_gt
            continue
        s1 = score_candidates(cand, top_idx, top_w, train_cases, cause_embs)
        order = torch.argsort(s1, descending=True).cpu().numpy()
        cand_global = np.array(cand)
        cand_embs_n = cause_embs_n.index_select(0, torch.tensor(cand, device=device, dtype=torch.long))
        sorted_embs_n = cand_embs_n[torch.from_numpy(order).to(device)]
        gt_embs_n = cause_embs_n.index_select(0, torch.tensor(gt, device=device, dtype=torch.long))
        match = (gt_embs_n @ sorted_embs_n.T >= sem_thresh).cpu().numpy()  # [n_gt, P]
        for gi in range(n_gt):
            hit = np.flatnonzero(match[gi])
            sem_ranks.append(float(hit[0]) + 1.0 if hit.size else MISS_RANK)
            sem_cov.append(1 if hit.size else 0)
        if cluster_id_array is not None:
            sorted_clusters = cluster_id_array[cand_global[order]]
            for gi in gt:
                cid = int(cluster_id_array[int(gi)])
                hits = np.flatnonzero(sorted_clusters == cid)
                cl_ranks.append(float(hits[0]) + 1.0 if hits.size else MISS_RANK)
                cl_cov.append(1 if hits.size else 0)

    arr = np.asarray(sem_ranks, dtype=np.float64)
    m = summarize_rank_metric(arr, sem_cov)
    sem_m = {}; add_recall_at_ks(sem_m, arr, [1, 10])
    out = {"sem_R@1": sem_m["R@1"], "sem_R@10": sem_m["R@10"], "sem_MRR": m["MRR"]}
    if cluster_id_array is not None:
        carr = np.asarray(cl_ranks, dtype=np.float64)
        cm = {}; add_recall_at_ks(cm, carr, [10])
        out["cl_R@10"] = cm["R@10"]
    return out


def cluster_array(cause_texts, cluster_json):
    if not (cluster_json and Path(cluster_json).exists()):
        return None
    o2c = json.load(open(cluster_json, encoding="utf-8"))["original_to_cause_id"]
    return np.array([int(o2c[t]) for t in cause_texts], dtype=np.int64)


def encode_bank_w(enc, g, z, w, device, bs=256):
    """Encode cases with explicit per-lesion weights w [N, Q] (e.g. hard {0,1})."""
    enc.eval(); out = []
    with torch.no_grad():
        for s in range(0, g.size(0), bs):
            e = min(s + bs, g.size(0))
            lens = torch.full((e - s,), z.size(1), device=device, dtype=torch.long)
            h = enc(g[s:e].float().to(device), z[s:e].float().to(device), lens,
                    lesion_weights=w[s:e].float().to(device))
            out.append(h.float().cpu())
    enc.train()
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--art", default=ART)
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--case_db_base", default=f"{ART}/db/case_db_base")
    ap.add_argument("--encoder_base", default=f"{ART}/models/encoder_base/best_encoder.pt")
    ap.add_argument("--soft_dir", default=f"{ART}/db/soft_inputs")
    ap.add_argument("--hs_dir", default=f"{ART}/db/hs_soft")
    ap.add_argument("--cluster_json", default=f"{ART}/cause_clusters_llm.json")
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    ART, dev = args.art, args.device

    cdir = Path(args.case_db_dir)
    train_cases = torch.load(cdir / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(cdir / "valid_cases.pt", weights_only=False)
    pack = torch.load(cdir / "cause_text_embs.pt", weights_only=False)
    cause_texts = pack["texts"]
    cause_embs = pack["embeddings"].float().to(dev)
    cause_embs_n = F.normalize(cause_embs, dim=-1)

    cluster_id_array = cluster_array(cause_texts, args.cluster_json)
    if cluster_id_array is not None:
        print(f"[cluster] {len(set(cluster_id_array.tolist()))} clusters")

    g_tr, z_tr, w_tr, _ = load_soft(Path(args.soft_dir) / "train.pt")
    g_va, z_va, w_va, _ = load_soft(Path(args.soft_dir) / "valid.pt")
    o_tr, o_va = recover_logits(w_tr), recover_logits(w_va)

    def load_enc(ckpt):
        pkg = torch.load(ckpt, weights_only=False, map_location="cpu")
        enc = build_encoder(EncoderConfig(**pkg["encoder_config"])).to(dev)
        enc.load_state_dict(pkg["encoder_state"])
        return enc, pkg

    def report(tag, bank, H_va):
        m = paper_retrieval_eval(bank, H_va, train_cases, valid_cases, cause_embs,
                                 cause_embs_n, cluster_id_array, dev, args.top_k_cases)
        cl = f"  clR@10={m['cl_R@10']:.4f}" if "cl_R@10" in m else ""
        print(f"  {tag:16s} semR@1={m['sem_R@1']:.4f}  semR@10={m['sem_R@10']:.4f}  "
              f"semMRR={m['sem_MRR']:.4f}{cl}")

    def run_l1(tag, enc, gate, use_gate):
        bank = encode_bank(enc, gate, g_tr, z_tr, o_tr, dev, use_gate=use_gate)
        H_va = encode_bank(enc, gate, g_va, z_va, o_va, dev, use_gate=use_gate)
        report(tag, bank, H_va)

    print(f"[paper-protocol retrieval-only, full valid n={len(valid_cases)}, "
          f"top_k_cases={args.top_k_cases}]\n")

    enc_b, _ = load_enc(f"{ART}/models/encoder_grod_soft/best_encoder.pt")
    run_l1("baseline", enc_b, RegionGate().to(dev), use_gate=False)

    enc_s, _ = load_enc(f"{ART}/models/grace_e2e_sigmoidctrl/best.pt")
    run_l1("sigmoidctrl", enc_s, RegionGate().to(dev), use_gate=False)

    enc1, pkg1 = load_enc(f"{ART}/models/grace_e2e/best.pt")
    gate1 = RegionGate(init_temp=pkg1.get("gate_init_temp", 0.3)).to(dev)
    gate1.load_state_dict(pkg1["gate_state"])
    run_l1("L1(gate+agg)", enc1, gate1, use_gate=True)

    # L2: aggregator + gate + trained heads applied to cached hs
    pkg2 = torch.load(f"{ART}/models/grace_e2e_l2/best.pt", weights_only=False, map_location="cpu")
    enc2 = build_encoder(EncoderConfig(**pkg2["encoder_config"])).to(dev)
    enc2.load_state_dict(pkg2["encoder_state"])
    gate2 = RegionGate(init_temp=0.3).to(dev); gate2.load_state_dict(pkg2["gate_state"])
    obj_head, sem_head = clone_heads(
        f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth",
        f"{ART}/models/distilled_global_rawP/global_embed_state_dict.pt",
        f"{ART}/models/text_anchors.pt", dev)
    obj_head.load_state_dict(pkg2["obj_head_state"]); sem_head.load_state_dict(pkg2["sem_head_state"])
    hs_tr = torch.load(Path(args.hs_dir) / "hs_train.pt", weights_only=False)["hs"]
    hs_va = torch.load(Path(args.hs_dir) / "hs_valid.pt", weights_only=False)["hs"]
    bank = encode_bank_l2(enc2, obj_head, sem_head, gate2, g_tr, hs_tr, dev, use_gate=True)
    H_va = encode_bank_l2(enc2, obj_head, sem_head, gate2, g_va, hs_va, dev, use_gate=True)
    report("L2(+det heads)", bank, H_va)

    print("\n[ablation rows — paper §joint-training narrative]")
    # hard: OAVLE soft model fed {0,1} hard inputs (= hard lesion selection),
    # encoder_grod_soft aggregator. w_hard = (sigmoid(o) >= 0.5) = (o >= 0).
    wh_tr = (w_tr >= 0.5).float(); wh_va = (w_va >= 0.5).float()
    bank = encode_bank_w(enc_b, g_tr, z_tr, wh_tr, dev)
    H_va = encode_bank_w(enc_b, g_va, z_va, wh_va, dev)
    report("OAVLE-hard", bank, H_va)
    report("OAVLE-soft(=baseline)", *(  # alias for narrative clarity
        (encode_bank(enc_b, RegionGate().to(dev), g_tr, z_tr, o_tr, dev, use_gate=False),
         encode_bank(enc_b, RegionGate().to(dev), g_va, z_va, o_va, dev, use_gate=False))))

    # base: separated baseline — detect -> crop -> finetuned SigLIP2 -> dense
    # DeepSets (encoder_base) -> CEAH. Own case_db / cause table; cluster R@10
    # comparable via shared cluster_json (text mapping), same 1584 valid queries.
    bcdir = Path(args.case_db_base)
    btr = torch.load(bcdir / "train_cases.pt", weights_only=False)
    bva = torch.load(bcdir / "valid_cases.pt", weights_only=False)
    bpack = torch.load(bcdir / "cause_text_embs.pt", weights_only=False)
    bce = bpack["embeddings"].float().to(dev)
    bce_n = F.normalize(bce, dim=-1)
    bcl = cluster_array(bpack["texts"], args.cluster_json)
    benc, _ = load_enc(args.encoder_base)
    bbank = encode_all(benc, btr, dev).to(dev)
    bH = encode_all(benc, bva, dev).to(dev)
    m = paper_retrieval_eval(bbank, bH, btr, bva, bce, bce_n, bcl, dev, args.top_k_cases)
    cl = f"  clR@10={m['cl_R@10']:.4f}" if "cl_R@10" in m else ""
    print(f"  {'base(separated)':16s} semR@1={m['sem_R@1']:.4f}  semR@10={m['sem_R@10']:.4f}  "
          f"semMRR={m['sem_MRR']:.4f}{cl}")


if __name__ == "__main__":
    main()
