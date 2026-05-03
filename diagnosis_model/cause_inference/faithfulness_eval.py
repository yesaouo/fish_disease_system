"""Faithfulness verification for CEAH attributions.

For each valid query and its top-1 predicted cause (under hybrid γ=0.5),
compute the CEAH score under five evidence-masking conditions:

  full       — no mask (baseline score)
  no_global  — zero out the global token
  no_text    — zero out the text token
  no_lesion  — zero out all lesion tokens
  no_top_alpha — zero out the single highest-α evidence token

If CEAH attributions are faithful, masking the highest-α token (no_top_alpha)
should produce the largest score drop — and bigger than masking a random token.
We additionally bucket scores by predicted-cause type (global-type vs lesion-type)
to see whether lesion-type causes depend more on lesion evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from diagnosis_model.cause_inference.models import CEAH
from diagnosis_model.cause_inference.phase1_baseline import (
    build_candidate_pool, compute_case_similarities,
    diversify, score_candidates, stack_train_lesions,
)


GLOBAL_KEYS = ["水質", "緊迫", "營養", "環境", "水溫", "免疫力", "應激"]
LESION_KEYS = ["潰瘍", "出血", "紅腫", "寄生蟲", "棉絮", "創傷", "腫脹", "黴", "結痂", "撕裂", "蛀"]


def classify_cause(text: str) -> str:
    g = sum(1 for k in GLOBAL_KEYS if k in text)
    l = sum(1 for k in LESION_KEYS if k in text)
    if g > l:
        return "global-type"
    if l > g:
        return "lesion-type"
    return "mixed"


@torch.no_grad()
def ceah_forward(
    ceah, q, candidate_indices, cause_table_embs, in_dim, device,
    use_text_kind="medical", force_mask=None,
):
    P = len(candidate_indices)
    if P == 0:
        return None
    n_les = q["lesion_embs"].size(0)
    global_emb = q["global_emb"].unsqueeze(0).expand(P, -1).to(device)
    lesion_embs = q["lesion_embs"].unsqueeze(0).expand(P, -1, -1).to(device)
    lesion_mask = torch.ones(P, n_les, dtype=torch.bool, device=device)

    text_key = f"text_{use_text_kind}_emb"
    t = q.get(text_key)
    if t is not None:
        text_emb = t.unsqueeze(0).expand(P, -1).to(device)
        text_present = torch.ones(P, dtype=torch.bool, device=device)
    else:
        text_emb = torch.zeros(P, in_dim, device=device)
        text_present = torch.zeros(P, dtype=torch.bool, device=device)

    cand_idx_t = torch.tensor(candidate_indices, device=device, dtype=torch.long)
    cand_embs = cause_table_embs.index_select(0, cand_idx_t)

    return ceah(global_emb, text_emb, text_present, lesion_embs, lesion_mask,
                cand_embs, force_mask=force_mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--ceah_ckpt", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--alpha_global", type=float, default=0.25)
    ap.add_argument("--beta_lesion", type=float, default=0.75)
    ap.add_argument("--lesion_match", type=str, default="hungarian")
    ap.add_argument("--diversify_threshold", type=float, default=0.95)
    ap.add_argument("--text_kind", type=str, default="medical")
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="hybrid mix to pick top-1 prediction for each query")
    ap.add_argument("--common_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--attribution_mode", type=str, default="sigmoid",
                    choices=["sigmoid", "softmax"])
    ap.add_argument("--scoring_mode", type=str, default="single",
                    choices=["single", "multiplicative"])
    ap.add_argument("--max_queries", type=int, default=-1)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    case_db_dir = Path(args.case_db_dir)
    train_cases = torch.load(case_db_dir / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(case_db_dir / "valid_cases.pt", weights_only=False)
    cause_pack = torch.load(case_db_dir / "cause_text_embs.pt", weights_only=False)
    cause_table_embs = cause_pack["embeddings"].to(device)
    cause_texts = cause_pack["texts"]

    train_global_stack = torch.stack([c["global_emb"] for c in train_cases]).to(device)
    train_lesion_stack, train_offsets = stack_train_lesions(train_cases)
    train_lesion_stack = train_lesion_stack.to(device)

    in_dim = cause_table_embs.size(-1)
    ceah = CEAH(
        global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
        common_dim=args.common_dim, hidden_dim=args.hidden_dim, dropout=0.0,
        attribution_mode=args.attribution_mode, scoring_mode=args.scoring_mode,
    ).to(device)
    ceah.load_state_dict(torch.load(args.ceah_ckpt, map_location=device))
    ceah.eval()
    print(f"[load] valid={len(valid_cases)}  ceah={args.ceah_ckpt}")

    queries = valid_cases if args.max_queries <= 0 else valid_cases[: args.max_queries]

    # Per-condition score deltas (full minus masked)
    drops_by_cond_total: Dict[str, List[float]] = defaultdict(list)
    drops_by_cond_bucket: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    drops_by_cond_n: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    bucket_count = defaultdict(int)
    n_bucket_count = defaultdict(int)

    def n_bucket(n: int) -> str:
        if n == 1: return "N=1"
        if n == 2: return "N=2"
        return "N>=3"

    t0 = time.time()
    for qi, q in enumerate(queries):
        q_global = q["global_emb"].to(device)
        q_lesions = q["lesion_embs"].to(device)
        n_les = q_lesions.size(0)
        if n_les == 0:
            continue

        sims = compute_case_similarities(
            q_global, q_lesions,
            train_global_stack, train_lesion_stack, train_offsets,
            args.alpha_global, args.beta_lesion, lesion_match=args.lesion_match,
        )
        top_k_idx = np.argsort(-sims)[: args.top_k_cases]
        top_k_w = sims[top_k_idx]
        candidate_indices = build_candidate_pool(top_k_idx, train_cases)
        if not candidate_indices:
            continue

        # Compute hybrid score to pick the top-1 prediction
        cand_idx_t = torch.tensor(candidate_indices, device=device, dtype=torch.long)
        cand_embs = cause_table_embs.index_select(0, cand_idx_t)
        s1 = score_candidates(candidate_indices, top_k_idx, top_k_w, train_cases, cause_table_embs)

        out_full = ceah_forward(ceah, q, candidate_indices, cause_table_embs,
                                in_dim, device, use_text_kind=args.text_kind)
        scores_full, alpha_full, ev_mask = out_full

        s1_n = (s1 - s1.min()) / (s1.max() - s1.min() + 1e-9)
        sc_n = (scores_full - scores_full.min()) / (scores_full.max() - scores_full.min() + 1e-9)
        hybrid = args.gamma * s1_n + (1 - args.gamma) * sc_n

        score_sorted = torch.argsort(hybrid, descending=True).cpu().numpy()
        sorted_local = diversify(score_sorted, cand_embs, args.diversify_threshold)
        if len(sorted_local) == 0:
            continue
        top1_local = int(sorted_local[0])
        top1_global_idx = candidate_indices[top1_local]
        top1_text = cause_texts[top1_global_idx]
        bucket = classify_cause(top1_text)
        bucket_count[bucket] += 1

        # Score (and alpha) for the top-1 prediction under various masks
        # Per-position semantics: 0=global, 1=text, 2..=lesions
        max_Ne = ev_mask.size(1)
        P = len(candidate_indices)
        device_t = ev_mask.device

        baseline = float(scores_full[top1_local].item())
        baseline_alpha = alpha_full[top1_local].cpu().numpy()  # length max_Ne

        def mask_and_score(positions_to_zero: List[int]) -> float:
            fm = torch.ones(P, max_Ne, dtype=torch.bool, device=device_t)
            for pos in positions_to_zero:
                if pos < max_Ne:
                    fm[:, pos] = False
            out = ceah_forward(ceah, q, candidate_indices, cause_table_embs,
                               in_dim, device, use_text_kind=args.text_kind, force_mask=fm)
            return float(out[0][top1_local].item())

        s_no_global  = mask_and_score([0])
        s_no_text    = mask_and_score([1])
        lesion_positions = list(range(2, 2 + n_les))
        s_no_lesion  = mask_and_score(lesion_positions)

        # mask the single highest-α among VALID positions
        valid_positions = [0]  # global always valid
        if q.get(f"text_{args.text_kind}_emb") is not None:
            valid_positions.append(1)
        valid_positions.extend(lesion_positions)
        valid_alpha_pairs = [(p, baseline_alpha[p]) for p in valid_positions]
        top_alpha_pos = max(valid_alpha_pairs, key=lambda x: x[1])[0]
        s_no_top_alpha = mask_and_score([top_alpha_pos])

        # Random-baseline: mask one randomly-chosen valid position (other than the top-α one)
        other_positions = [p for p in valid_positions if p != top_alpha_pos]
        if other_positions:
            rnd_pos = int(np.random.choice(other_positions))
            s_random = mask_and_score([rnd_pos])
        else:
            s_random = baseline

        nb = n_bucket(n_les)
        n_bucket_count[nb] += 1
        for cond, val in [
            ("no_global",   baseline - s_no_global),
            ("no_text",     baseline - s_no_text),
            ("no_lesion",   baseline - s_no_lesion),
            ("no_top_α",    baseline - s_no_top_alpha),
            ("no_random",   baseline - s_random),
        ]:
            drops_by_cond_total[cond].append(val)
            drops_by_cond_bucket[bucket][cond].append(val)
            drops_by_cond_n[nb][cond].append(val)

        if (qi + 1) % 200 == 0 or qi + 1 == len(queries):
            elapsed = time.time() - t0
            rate = (qi + 1) / elapsed
            eta = (len(queries) - qi - 1) / max(rate, 1e-9)
            print(f"[eval] {qi+1}/{len(queries)}  "
                  f"rate={rate:.1f} q/s  ETA={eta/60:.1f} min")

    # Aggregate
    summary = {
        "n_queries_used": sum(len(drops_by_cond_total[c]) for c in ["no_global"]) // 1,
        "bucket_counts": dict(bucket_count),
        "n_bucket_counts": dict(n_bucket_count),
        "score_drop_by_condition": {
            cond: {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "n": len(vals),
            }
            for cond, vals in drops_by_cond_total.items()
        },
        "score_drop_by_bucket": {
            bucket: {
                cond: {
                    "mean": float(np.mean(vals)) if vals else 0.0,
                    "median": float(np.median(vals)) if vals else 0.0,
                    "n": len(vals),
                }
                for cond, vals in conds.items()
            }
            for bucket, conds in drops_by_cond_bucket.items()
        },
        "score_drop_by_n_bucket": {
            nb: {
                cond: {
                    "mean": float(np.mean(vals)) if vals else 0.0,
                    "median": float(np.median(vals)) if vals else 0.0,
                    "n": len(vals),
                }
                for cond, vals in conds.items()
            }
            for nb, conds in drops_by_cond_n.items()
        },
        "config": vars(args),
    }
    with (out_dir / "faithfulness.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Score drop (baseline minus masked, mean over queries) ===")
    print(f"{'condition':<14}{'all':>10}{'global-type':>14}{'lesion-type':>14}{'mixed':>10}")
    for cond in ["no_global", "no_text", "no_lesion", "no_top_α", "no_random"]:
        all_v = summary["score_drop_by_condition"][cond]["mean"]
        gt_v = summary["score_drop_by_bucket"].get("global-type", {}).get(cond, {}).get("mean", 0.0)
        lt_v = summary["score_drop_by_bucket"].get("lesion-type", {}).get(cond, {}).get("mean", 0.0)
        mx_v = summary["score_drop_by_bucket"].get("mixed", {}).get(cond, {}).get("mean", 0.0)
        print(f"{cond:<14}{all_v:>10.4f}{gt_v:>14.4f}{lt_v:>14.4f}{mx_v:>10.4f}")
    print(f"\nbucket counts: {dict(bucket_count)}")

    print("\n=== Score drop by lesion-count bucket ===")
    print(f"{'condition':<14}{'all':>10}{'N=1':>10}{'N=2':>10}{'N>=3':>10}")
    for cond in ["no_global", "no_text", "no_lesion", "no_top_α", "no_random"]:
        all_v = summary["score_drop_by_condition"][cond]["mean"]
        v1 = summary["score_drop_by_n_bucket"].get("N=1", {}).get(cond, {}).get("mean", 0.0)
        v2 = summary["score_drop_by_n_bucket"].get("N=2", {}).get(cond, {}).get("mean", 0.0)
        v3 = summary["score_drop_by_n_bucket"].get("N>=3", {}).get(cond, {}).get("mean", 0.0)
        print(f"{cond:<14}{all_v:>10.4f}{v1:>10.4f}{v2:>10.4f}{v3:>10.4f}")
    print(f"\nN-bucket counts: {dict(n_bucket_count)}")
    print(f"[save] -> {out_dir}/faithfulness.json")


if __name__ == "__main__":
    main()
