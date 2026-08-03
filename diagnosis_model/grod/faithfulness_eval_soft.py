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

**Defaults are the production operating point**: current artifact tree
(`artifacts.ART`), **gated** soft inputs, `--split test`, `--top_k_cases 3`,
`--max_queries -1` (= whole split). A bare run reproduces the thesis table.
`--top_k_cases` matters: k=20 flips the sign of `no_global` (−0.0038 vs +0.0036).

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

from diagnosis_model.cause_inference.phase1_baseline import build_candidate_pool
from diagnosis_model.cause_inference.faithfulness_eval import classify_cause
from diagnosis_model.grod.artifacts import ART, load_case_db, load_ceah, load_encoder, load_bank
from diagnosis_model.grod.soft_eval_common import faithfulness_drops, make_ceah_batch
from diagnosis_model.grod.train_case_encoder_soft import encode_all_soft, load_soft
from diagnosis_model.grod.train_ceah_soft import topk_by_w


def n_bucket(n):
    return "N=1" if n == 1 else ("N=2" if n == 2 else "N>=3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", default=str(ART / "db/case_db_jointDistRawP"))
    ap.add_argument("--soft_dir", default=str(ART / "db/soft_inputs_gated"))
    ap.add_argument("--encoder_ckpt", default=str(ART / "models/encoder_grod_soft/best_encoder.pt"))
    ap.add_argument("--bank_path", default=str(ART / "models/encoder_grod_soft/bank_z_soft.pt"))
    ap.add_argument("--ceah_ckpt", default=str(ART / "models/ceah_grod_soft/best_ceah.pt"))
    ap.add_argument("--output_dir", default=str(ART / "models/ceah_grod_soft"))
    ap.add_argument("--top_k_lesions", type=int, default=32)
    ap.add_argument("--top_k_cases", type=int, default=3,
                    help="production operating point. k=20 flips the sign of no_global.")
    ap.add_argument("--split", default="test", choices=["valid", "test"],
                    help="query split: {split}_cases.pt + <soft_dir>/{split}.pt")
    ap.add_argument("--max_queries", type=int, default=-1,
                    help="-1 = whole split (the paper setting)")
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

    db = load_case_db(args.case_db_dir, args.split, device)
    train_cases, valid_cases = db.train_cases, db.query_cases
    cause_embs, cause_texts = db.cause_embs, db.cause_texts

    g_va, z_va, w_va, _ = load_soft(Path(args.soft_dir) / f"{args.split}.pt")

    encoder = load_encoder(args.encoder_ckpt, device)
    bank_z = load_bank(args.bank_path, device)
    ceah = load_ceah(args.ceah_ckpt, db.in_dim, device,
                     common_dim=args.common_dim, hidden_dim=args.hidden_dim,
                     attribution_mode=args.attribution_mode, scoring_mode=args.scoring_mode)

    H_va = encode_all_soft(encoder, g_va, z_va, w_va, device)
    nq = len(valid_cases) if args.max_queries < 0 else min(args.max_queries, len(valid_cases))
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
        batch = make_ceah_batch(ceah, g_va[qi].float().to(device), z_k, w_k, cand_embs, device)
        top1, baseline, drops = faithfulness_drops(batch, device)
        baselines.append(baseline)
        bucket = classify_cause(cause_texts[cand_idx[top1]])
        bucket_count[bucket] += 1

        for cond, val in drops.items():
            drops_total[cond].append(val)
            drops_bucket[bucket][cond].append(val)

    baselines: list[float] = []
    for qi in range(nq):
        run(qi)

    conds = ["no_global", "no_lesion", "no_top_α", "no_random"]
    print(f"\n=== SOFT CEAH faithfulness (score drop = baseline - masked, n={nq}) ===")
    print(f"{'condition':<12}{'all':>10}{'global-type':>14}{'lesion-type':>14}")
    mean_base = float(np.mean(baselines)) if baselines else 0.0
    print(f"mean top-1 support score (baseline) = {mean_base:.4f}")
    summary = {"mean_top1_support": mean_base,
               "score_drop_by_condition": {}, "score_drop_by_bucket": {}, "bucket_counts": dict(bucket_count)}
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
