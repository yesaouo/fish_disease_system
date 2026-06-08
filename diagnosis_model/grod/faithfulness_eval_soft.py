"""Soft-pipeline retrain — step #4b: faithfulness gate for soft CEAH.

Mirrors faithfulness_eval.py but on the soft cascade: soft-encode the query →
retrieve top-k over bank_z_soft → soft CEAH (top-K lesion z + objectness w,
lesion_weights gate). For the top-1 predicted cause we mask evidence groups via
CEAH.force_mask and measure the score drop (baseline - masked):

  no_global   — zero the global token
  no_lesion   — zero all lesion tokens     (HEADLINE: must stay POSITIVE)
  no_top_α    — zero the single highest-α token (should be the largest drop)
  no_random   — zero a random other token  (control)

split by cause type (lesion-type vs global-type). This is the gate that decides
whether soft evidence preserves CEAH's lesion-grounding (vs reversing it).

γ-free cascade (ranking = pure CEAH score, matches production).

Run from repo root (SDM env):
    $PY -m diagnosis_model.grod.faithfulness_eval_soft
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import torch

from diagnosis_model.cause_inference.models import CEAH
from diagnosis_model.cause_inference.phase1_baseline import build_candidate_pool
from diagnosis_model.cause_inference.faithfulness_eval import classify_cause
from diagnosis_model.cause_inference.models.case_encoder import EncoderConfig, build_encoder
from diagnosis_model.grod.train_case_encoder_soft import encode_all_soft, load_soft
from diagnosis_model.grod.train_ceah_soft import topk_by_w


def n_bucket(n):
    return "N=1" if n == 1 else ("N=2" if n == 2 else "N>=3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", default="diagnosis_model/cause_inference/outputs/case_db_jointDistRawP")
    ap.add_argument("--soft_dir", default="diagnosis_model/grod/outputs/soft_inputs")
    ap.add_argument("--encoder_ckpt", default="diagnosis_model/cause_inference/outputs/encoder_grod_soft/best_encoder.pt")
    ap.add_argument("--bank_path", default="diagnosis_model/cause_inference/outputs/encoder_grod_soft/bank_z_soft.pt")
    ap.add_argument("--ceah_ckpt", default="diagnosis_model/cause_inference/outputs/ceah_grod_soft/best_ceah.pt")
    ap.add_argument("--output_dir", default="diagnosis_model/cause_inference/outputs/ceah_grod_soft")
    ap.add_argument("--top_k_lesions", type=int, default=32)
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--max_queries", type=int, default=300)
    ap.add_argument("--common_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--attribution_mode", default="softmax")
    ap.add_argument("--scoring_mode", default="multiplicative")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    np.random.seed(args.seed)
    device = args.device
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    train_cases = torch.load(Path(args.case_db_dir) / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(Path(args.case_db_dir) / "valid_cases.pt", weights_only=False)
    cpack = torch.load(Path(args.case_db_dir) / "cause_text_embs.pt", weights_only=False)
    cause_embs = cpack["embeddings"].float().to(device)
    cause_texts = cpack["texts"]
    in_dim = cause_embs.size(-1)

    g_va, z_va, w_va, _ = load_soft(Path(args.soft_dir) / "valid.pt")

    enc_pkg = torch.load(args.encoder_ckpt, weights_only=False, map_location="cpu")
    encoder = build_encoder(EncoderConfig(**enc_pkg["encoder_config"])).to(device).eval()
    encoder.load_state_dict(enc_pkg["encoder_state"])
    bank_z = torch.load(args.bank_path, weights_only=False)["bank_z"].to(device)

    ceah = CEAH(global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
                common_dim=args.common_dim, hidden_dim=args.hidden_dim, dropout=0.0,
                attribution_mode=args.attribution_mode, scoring_mode=args.scoring_mode).to(device).eval()
    ceah.load_state_dict(torch.load(args.ceah_ckpt, map_location=device))

    H_va = encode_all_soft(encoder, g_va, z_va, w_va, device)
    nq = min(args.max_queries, len(valid_cases))
    K = args.top_k_lesions
    drops_total = defaultdict(list)
    drops_bucket = defaultdict(lambda: defaultdict(list))
    bucket_count = defaultdict(int)

    @torch.no_grad()
    def run(qi):
        sims = H_va[qi:qi + 1].to(device) @ bank_z.T
        top_idx = sims[0].topk(args.top_k_cases).indices.cpu().numpy()
        cand_idx = build_candidate_pool(top_idx, train_cases)
        P = len(cand_idx)
        if P == 0:
            return
        cand_embs = cause_embs[torch.as_tensor(cand_idx, device=device)]      # [P, D]
        z_k, w_k = topk_by_w(z_va[qi].float().to(device), w_va[qi].float().to(device), K)
        g_e = g_va[qi].float().to(device).unsqueeze(0).expand(P, -1)
        l_e = z_k.unsqueeze(0).expand(P, -1, -1).contiguous()
        l_w = w_k.unsqueeze(0).expand(P, -1).contiguous()
        l_m = torch.ones(P, z_k.size(0), dtype=torch.bool, device=device)
        t_e = torch.zeros(P, in_dim, device=device)
        t_p = torch.zeros(P, dtype=torch.bool, device=device)
        scores, alphas, ev_mask = ceah(g_e, t_e, t_p, l_e, l_m, cand_embs, lesion_weights=l_w)
        top1 = int(scores.argmax().item())
        baseline = float(scores[top1])
        base_alpha = alphas[top1].cpu().numpy()
        max_Ne = ev_mask.size(1)
        bucket = classify_cause(cause_texts[cand_idx[top1]])
        bucket_count[bucket] += 1

        def mask_score(positions):
            fm = torch.ones(P, max_Ne, dtype=torch.bool, device=device)
            for p in positions:
                if p < max_Ne:
                    fm[:, p] = False
            s, _, _ = ceah(g_e, t_e, t_p, l_e, l_m, cand_embs, force_mask=fm, lesion_weights=l_w)
            return float(s[top1])

        lesion_pos = list(range(2, 2 + z_k.size(0)))
        s_no_global = mask_score([0])
        s_no_lesion = mask_score(lesion_pos)
        # top-α among valid positions (global + lesions; text absent)
        valid_pos = [0] + lesion_pos
        top_a = max(valid_pos, key=lambda p: base_alpha[p])
        s_no_top = mask_score([top_a])
        others = [p for p in valid_pos if p != top_a]
        s_rand = mask_score([int(np.random.choice(others))]) if others else baseline

        for cond, val in [("no_global", baseline - s_no_global),
                          ("no_lesion", baseline - s_no_lesion),
                          ("no_top_α", baseline - s_no_top),
                          ("no_random", baseline - s_rand)]:
            drops_total[cond].append(val)
            drops_bucket[bucket][cond].append(val)

    for qi in range(nq):
        run(qi)

    conds = ["no_global", "no_lesion", "no_top_α", "no_random"]
    print(f"\n=== SOFT CEAH faithfulness (score drop = baseline - masked, n={nq}) ===")
    print(f"{'condition':<12}{'all':>10}{'global-type':>14}{'lesion-type':>14}")
    summary = {"score_drop_by_condition": {}, "score_drop_by_bucket": {}, "bucket_counts": dict(bucket_count)}
    for cond in conds:
        allv = float(np.mean(drops_total[cond]))
        gt = drops_bucket.get("global-type", {}).get(cond, [])
        lt = drops_bucket.get("lesion-type", {}).get(cond, [])
        gtv = float(np.mean(gt)) if gt else 0.0
        ltv = float(np.mean(lt)) if lt else 0.0
        print(f"{cond:<12}{allv:>10.4f}{gtv:>14.4f}{ltv:>14.4f}")
        summary["score_drop_by_condition"][cond] = {"mean": allv, "n": len(drops_total[cond])}
        summary["score_drop_by_bucket"][cond] = {"global-type": gtv, "lesion-type": ltv}
    print(f"bucket counts: {dict(bucket_count)}")
    json.dump(summary, open(out_dir / "faithfulness_soft.json", "w"), ensure_ascii=False, indent=2)
    print(f"[save] -> {out_dir}/faithfulness_soft.json")
    print("\nHEADLINE: no_lesion mean must be POSITIVE (masking lesions lowers the "
          "score) and lesion-type > global-type for it. Negative = faithfulness reversed.")


if __name__ == "__main__":
    main()
