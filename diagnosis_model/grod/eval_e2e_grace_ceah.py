"""through-CEAH eval of e2e variants: final cause ranking (gamma=0 production).

Companion to eval_e2e_grace.py (retrieval-only). Here we run the FULL production
cascade per query (gamma=0 = pure CEAH ranking, the fish operating point), on the
full valid set, with the SAME conventions as eval_ceah_soft_paper.py — so numbers
are comparable to the Ch5 cause-ranking table (§多病因診斷排序).

  variant -> encoder(g,z,w) -> bank top-k -> candidate pool
          -> CEAH(g, text, z_sel, w_sel, cand) -> rank by s_ceah (gamma=0)

CAVEAT: the production CEAH (ceah_grod_soft) was trained with the sigmoid gate's
w and the encoder_grod_soft z. baseline is matched; L1/L2 feed Region-Gate w
(and L2 also retrained sem-head z) into that same fixed CEAH — a deliberate
"plug the e2e encoder/gate into production CEAH without retraining" test. If flat,
it confirms e2e fine-tune yields no final-metric gain absent CEAH retraining.

Run:  $PY -m diagnosis_model.grod.eval_e2e_grace_ceah --top_k_cases 20
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
    MISS_RANK, add_recall_at_ks, build_candidate_pool,
    select_positive_top_cases, summarize_rank_metric,
)
from diagnosis_model.grod.train_case_encoder_soft import load_soft
from diagnosis_model.grod.region_gate import RegionGate
from diagnosis_model.grod.eval_ceah_soft_paper import select_lesions
from diagnosis_model.grod.finetune_e2e_grace import recover_logits, encode_bank
from diagnosis_model.grod.finetune_e2e_grace_l2 import (
    clone_heads, heads_forward, encode_bank as encode_bank_l2,
)


@torch.no_grad()
def ceah_rank_eval(ceah, bank, H_va, g_va, z_va_used, w_va_used, valid_cases, train_cases,
                   cause_embs, cause_embs_n, cluster_id_array, device,
                   top_k_cases, top_k_lesions=32, text_kind="medical", sem_thresh=0.95):
    in_dim = cause_embs.size(-1)
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
        cand_t = torch.tensor(cand, device=device, dtype=torch.long)
        P = len(cand)
        z_sel, w_sel = select_lesions(z_va_used[qi].float().to(device),
                                      w_va_used[qi].float().to(device), top_k_lesions)
        K = z_sel.size(0)
        t = valid_cases[qi][f"text_{text_kind}_emb"].float().to(device)
        text_emb = t.unsqueeze(0).expand(P, -1)
        text_present = torch.ones(P, dtype=torch.bool, device=device)
        lesion_mask = torch.ones((P, K), dtype=torch.bool, device=device)
        s_ceah, _, _ = ceah(
            g_va[qi].float().to(device).unsqueeze(0).expand(P, -1),
            text_emb, text_present,
            z_sel.unsqueeze(0).expand(P, -1, -1).contiguous(),
            lesion_mask,
            cause_embs.index_select(0, cand_t),
            lesion_weights=w_sel.unsqueeze(0).expand(P, -1).contiguous())
        order = torch.argsort(s_ceah, descending=True).cpu().numpy()
        cand_global = np.array(cand)
        cand_embs_n = cause_embs_n.index_select(0, cand_t)
        sorted_embs_n = cand_embs_n[torch.from_numpy(order).to(device)]
        gt_embs_n = cause_embs_n.index_select(0, torch.tensor(gt, device=device, dtype=torch.long))
        match = (gt_embs_n @ sorted_embs_n.T >= sem_thresh).cpu().numpy()
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
    sem_m = {}; add_recall_at_ks(sem_m, arr, [1, 5, 10])
    out = {"sem_R@1": sem_m["R@1"], "sem_R@5": sem_m["R@5"],
           "sem_R@10": sem_m["R@10"], "sem_MRR": m["MRR"]}
    if cluster_id_array is not None:
        carr = np.asarray(cl_ranks, dtype=np.float64)
        cm = {}; add_recall_at_ks(cm, carr, [10])
        out["cl_R@10"] = cm["R@10"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--art", default=ART)
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--soft_dir", default=f"{ART}/db/soft_inputs")
    ap.add_argument("--hs_dir", default=f"{ART}/db/hs_soft")
    ap.add_argument("--ceah_ckpt", default=f"{ART}/models/ceah_grod_soft/best_ceah.pt")
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
    in_dim = cause_embs.size(-1)

    cluster_id_array = None
    if Path(args.cluster_json).exists():
        o2c = json.load(open(args.cluster_json, encoding="utf-8"))["original_to_cause_id"]
        cluster_id_array = np.array([int(o2c[t]) for t in cause_texts], dtype=np.int64)
        print(f"[cluster] {len(set(cluster_id_array.tolist()))} clusters")

    ceah = CEAH(global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
                common_dim=256, hidden_dim=512, dropout=0.0,
                attribution_mode="softmax", scoring_mode="multiplicative").to(dev).eval()
    ceah.load_state_dict(torch.load(args.ceah_ckpt, map_location=dev))

    g_tr, z_tr, w_tr, _ = load_soft(Path(args.soft_dir) / "train.pt")
    g_va, z_va, w_va, _ = load_soft(Path(args.soft_dir) / "valid.pt")
    o_tr, o_va = recover_logits(w_tr), recover_logits(w_va)

    def load_enc(ckpt):
        pkg = torch.load(ckpt, weights_only=False, map_location="cpu")
        enc = build_encoder(EncoderConfig(**pkg["encoder_config"])).to(dev)
        enc.load_state_dict(pkg["encoder_state"])
        return enc, pkg

    def report(tag, bank, H_va, z_va_used, w_va_used):
        m = ceah_rank_eval(ceah, bank, H_va, g_va, z_va_used, w_va_used, valid_cases,
                           train_cases, cause_embs, cause_embs_n, cluster_id_array, dev,
                           args.top_k_cases)
        cl = f"  clR@10={m['cl_R@10']:.4f}" if "cl_R@10" in m else ""
        print(f"  {tag:16s} semR@1={m['sem_R@1']:.4f}  semR@5={m['sem_R@5']:.4f}  "
              f"semR@10={m['sem_R@10']:.4f}  semMRR={m['sem_MRR']:.4f}{cl}")

    print(f"[through-CEAH gamma=0, full valid n={len(valid_cases)}, "
          f"top_k_cases={args.top_k_cases}]\n")

    # baseline: sigmoid w + encoder_grod_soft (matched to CEAH training)
    enc_b, _ = load_enc(f"{ART}/models/encoder_grod_soft/best_encoder.pt")
    g0 = RegionGate().to(dev)
    bank = encode_bank(enc_b, g0, g_tr, z_tr, o_tr, dev, use_gate=False)
    H_va = encode_bank(enc_b, g0, g_va, z_va, o_va, dev, use_gate=False)
    report("baseline", bank, H_va, z_va, w_va)

    # sigmoidctrl: sigmoid w + aggregator trained more
    enc_s, _ = load_enc(f"{ART}/models/grace_e2e_sigmoidctrl/best.pt")
    bank = encode_bank(enc_s, g0, g_tr, z_tr, o_tr, dev, use_gate=False)
    H_va = encode_bank(enc_s, g0, g_va, z_va, o_va, dev, use_gate=False)
    report("sigmoidctrl", bank, H_va, z_va, w_va)

    # L1: Region Gate w + grace_e2e aggregator
    enc1, pkg1 = load_enc(f"{ART}/models/grace_e2e/best.pt")
    gate1 = RegionGate(init_temp=pkg1.get("gate_init_temp", 0.3)).to(dev)
    gate1.load_state_dict(pkg1["gate_state"])
    bank = encode_bank(enc1, gate1, g_tr, z_tr, o_tr, dev, use_gate=True)
    H_va = encode_bank(enc1, gate1, g_va, z_va, o_va, dev, use_gate=True)
    w_va_l1 = gate1(o_va.to(dev)).cpu()
    report("L1(gate+agg)", bank, H_va, z_va, w_va_l1)

    # L2: Region Gate w + trained heads (z and w from hs)
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
    w_va_l2, z_va_l2 = heads_forward(obj_head, sem_head, gate2, hs_va, dev, use_gate=True)
    report("L2(+det heads)", bank, H_va, z_va_l2.cpu(), w_va_l2.cpu())


if __name__ == "__main__":
    main()
