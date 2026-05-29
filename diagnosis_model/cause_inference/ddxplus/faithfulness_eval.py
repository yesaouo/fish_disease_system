"""DDXPlus evidence faithfulness evaluation for CEAH.

For each query, choose the top-1 predicted disease from the Phase1->CEAH
cascade, then rescore that same candidate under evidence masks:

  full               baseline CEAH score
  no_global          mask patient-summary token
  no_text            mask optional text token
  no_evidence        mask all symptom/antecedent evidence tokens
  no_top_alpha       mask the highest-alpha valid token
  no_top_evidence    mask the highest-alpha evidence token only
  no_random_evidence mask one random evidence token

Score drop = full_score - masked_score. Positive drops indicate that the masked
token group supported the prediction.
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
import torch.nn.functional as F

from diagnosis_model.cause_inference.ddxplus.eval_ceah import (
    ceah_forward_for_pool,
    minmax_norm,
    summarize_alpha,
)
from diagnosis_model.cause_inference.models import CEAH
from diagnosis_model.cause_inference.phase1_baseline import (
    build_candidate_pool,
    compute_case_similarities,
    load_cases,
    load_train_bank,
    load_train_cases_minimal,
    offsets_to_case_ids,
    score_candidates,
    select_positive_top_cases,
    stream_top_k_cases,
)


def evidence_bucket(n: int) -> str:
    if n <= 2:
        return "N<=2"
    if n <= 5:
        return "N=3-5"
    return "N>=6"


def mean_stats(values: List[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "n": int(arr.size),
    }


@torch.no_grad()
def masked_top1_score(
    ceah: CEAH,
    q: dict,
    candidate_indices: List[int],
    cause_table_embs: torch.Tensor,
    in_dim: int,
    device: str,
    text_kind: str,
    top1_local: int,
    positions_to_zero: List[int],
) -> float:
    P = len(candidate_indices)
    n_ev = q["lesion_embs"].size(0)
    max_ne = 2 + n_ev
    force_mask = torch.ones(P, max_ne, dtype=torch.bool, device=device)
    for pos in positions_to_zero:
        if 0 <= pos < max_ne:
            force_mask[:, pos] = False

    n = q["lesion_embs"].size(0)
    global_emb = q["global_emb"].unsqueeze(0).expand(P, -1).contiguous().to(device)
    evidence_embs = q["lesion_embs"].unsqueeze(0).expand(P, -1, -1).contiguous().to(device)
    evidence_mask = torch.ones(P, n, dtype=torch.bool, device=device)

    if text_kind == "none":
        text_emb = torch.zeros(P, in_dim, device=device)
        text_present = torch.zeros(P, dtype=torch.bool, device=device)
    else:
        t = q.get(f"text_{text_kind}_emb")
        if t is not None:
            text_emb = t.unsqueeze(0).expand(P, -1).contiguous().to(device)
            text_present = torch.ones(P, dtype=torch.bool, device=device)
        else:
            text_emb = torch.zeros(P, in_dim, device=device)
            text_present = torch.zeros(P, dtype=torch.bool, device=device)

    cand_idx_t = torch.tensor(candidate_indices, device=device, dtype=torch.long)
    cand_embs = cause_table_embs.index_select(0, cand_idx_t)
    scores, _, _ = ceah(
        global_emb,
        text_emb,
        text_present,
        evidence_embs,
        evidence_mask,
        cand_embs,
        force_mask=force_mask,
    )
    return float(scores[top1_local].item())


def main():
    ap = argparse.ArgumentParser(description="Evaluate DDXPlus CEAH evidence faithfulness.")
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--ceah_ckpt", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--query_split", type=str, default="valid", choices=["valid", "test"])
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="hybrid score for choosing top-1: gamma*Phase1 + (1-gamma)*CEAH")
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--top_n_dump", type=int, default=1)
    ap.add_argument("--alpha_global", type=float, default=0.25)
    ap.add_argument("--beta_lesion", type=float, default=0.75)
    ap.add_argument("--lesion_match", type=str, default="max_mean",
                    choices=["hungarian", "max_mean", "max_mean_normalized"])
    ap.add_argument("--text_kind", type=str, default="medical",
                    choices=["medical", "colloquial", "none"])
    ap.add_argument("--common_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attribution_mode", type=str, default="sigmoid",
                    choices=["sigmoid", "softmax"])
    ap.add_argument("--scoring_mode", type=str, default="single",
                    choices=["single", "multiplicative"])
    ap.add_argument("--max_queries", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bank_dtype", type=str, default="bf16",
                    choices=["fp32", "fp16", "bf16"],
                    help="On-device storage dtype of the train bank stacks. "
                         "Default 'bf16' halves VRAM vs fp32. CEAH itself stays "
                         "fp32 because cand_embs come from cause_table_embs "
                         "(loaded as fp32), not the bank.")
    ap.add_argument("--max_train_cases", type=int, default=200000,
                    help="Cap on retained train-bank cases via uniform random "
                         "per-shard subsampling. The 1M-case DDXPlus bank is "
                         "~31 GB even in bf16 and overflows a 32 GB GPU; 200k "
                         "still leaves ~4k cases per condition. Use 0 to disable. "
                         "Ignored when --stream is set.")
    ap.add_argument("--sample_seed", type=int, default=42)
    ap.add_argument("--stream", action="store_true",
                    help="Use shard-streaming Phase 1 retrieval (full 1M bank).")
    ap.add_argument("--stream_query_batch", type=int, default=64,
                    help="Query batch size for --stream mode.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    bank_dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    bank_dtype = bank_dtype_map[args.bank_dtype]
    max_train_cases = args.max_train_cases if args.max_train_cases > 0 else None

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    case_db_dir = Path(args.case_db_dir)

    if args.stream:
        train_cases = load_train_cases_minimal(case_db_dir)
    else:
        train_cases, train_global_stack, train_lesion_stack, train_offsets = load_train_bank(
            case_db_dir, device, bank_dtype=bank_dtype,
            max_cases=max_train_cases, sample_seed=args.sample_seed,
        )
    queries = load_cases(case_db_dir, args.query_split)
    cause_pack = torch.load(case_db_dir / "cause_text_embs.pt", weights_only=False)
    cause_texts = list(cause_pack["texts"])
    # .float() upcasts DDXPlus bf16/fp16 storage so CEAH stays fp32.
    cause_table_embs = F.normalize(cause_pack["embeddings"].to(device).float(), dim=-1)
    in_dim = cause_table_embs.size(-1)

    ceah = CEAH(
        global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
        common_dim=args.common_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
        attribution_mode=args.attribution_mode, scoring_mode=args.scoring_mode,
    ).to(device)
    ceah.load_state_dict(torch.load(args.ceah_ckpt, map_location=device))
    ceah.eval()

    query_cases = queries if args.max_queries <= 0 else queries[: args.max_queries]
    print(f"[load] train={len(train_cases)} {args.query_split}={len(query_cases)} causes={len(cause_texts)}")
    print(f"[ceah] {args.ceah_ckpt}")

    cond_order = [
        "no_global",
        "no_text",
        "no_evidence",
        "no_top_alpha",
        "no_top_evidence",
        "no_random_evidence",
    ]
    drops_total: Dict[str, List[float]] = defaultdict(list)
    drops_by_n: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_query: List[dict] = []

    if args.stream:
        print(f"[stream] computing Phase 1 top-K for {len(query_cases)} queries "
              f"against full bank (batch={args.stream_query_batch})...")
        stream_top_idx, stream_top_w, _stream_top_raw = stream_top_k_cases(
            query_cases, case_db_dir,
            top_k_cases=args.top_k_cases,
            alpha=args.alpha_global, beta=args.beta_lesion,
            lesion_match=args.lesion_match,
            device=device, bank_dtype=bank_dtype,
            query_batch_size=args.stream_query_batch,
            verbose=True,
        )
    else:
        train_case_ids = offsets_to_case_ids(train_offsets, train_lesion_stack.device)

    t0 = time.time()
    for qi, q in enumerate(query_cases):
        n_ev = int(q["lesion_embs"].size(0))
        if n_ev <= 0:
            continue

        if args.stream:
            top_case_idx = stream_top_idx[qi]
            top_case_w = stream_top_w[qi]
        else:
            q_global = F.normalize(q["global_emb"].to(device), dim=-1)
            q_evidence = F.normalize(q["lesion_embs"].to(device), dim=-1)
            sims = compute_case_similarities(
                q_global, q_evidence,
                train_global_stack, train_lesion_stack, train_offsets,
                args.alpha_global, args.beta_lesion,
                lesion_match=args.lesion_match,
                train_case_ids=train_case_ids,
            )
            top_case_idx, top_case_w, _ = select_positive_top_cases(sims, args.top_k_cases)
        candidate_indices = build_candidate_pool(top_case_idx, train_cases)
        if not candidate_indices:
            continue

        s1 = score_candidates(
            candidate_indices, top_case_idx, top_case_w, train_cases, cause_table_embs,
        )
        s_ceah, alpha, _ = ceah_forward_for_pool(
            ceah, q, candidate_indices, cause_table_embs, in_dim, device, args.text_kind,
        )
        hybrid = args.gamma * minmax_norm(s1) + (1.0 - args.gamma) * minmax_norm(s_ceah)
        if hybrid.numel() == 0:
            continue
        top_sorted = torch.argsort(hybrid, descending=True).detach().cpu().numpy()
        top1_local = int(top_sorted[0])
        top1_cause_idx = int(candidate_indices[top1_local])
        baseline = float(s_ceah[top1_local].item())
        alpha_row = alpha[top1_local].detach().cpu().tolist() if alpha.numel() else []

        valid_positions = [0]
        if args.text_kind != "none" and q.get(f"text_{args.text_kind}_emb") is not None:
            valid_positions.append(1)
        evidence_positions = list(range(2, 2 + n_ev))
        valid_positions.extend(evidence_positions)

        top_alpha_pos = max(valid_positions, key=lambda p: alpha_row[p] if p < len(alpha_row) else -1.0)
        top_evidence_pos = max(
            evidence_positions,
            key=lambda p: alpha_row[p] if p < len(alpha_row) else -1.0,
        )
        random_evidence_pos = int(rng.choice(evidence_positions))

        condition_masks = {
            "no_global": [0],
            "no_text": [1],
            "no_evidence": evidence_positions,
            "no_top_alpha": [top_alpha_pos],
            "no_top_evidence": [top_evidence_pos],
            "no_random_evidence": [random_evidence_pos],
        }

        b = evidence_bucket(n_ev)
        drops_for_query = {}
        for cond in cond_order:
            masked_score = masked_top1_score(
                ceah, q, candidate_indices, cause_table_embs,
                in_dim, device, args.text_kind, top1_local, condition_masks[cond],
            )
            drop = baseline - masked_score
            drops_total[cond].append(drop)
            drops_by_n[b][cond].append(drop)
            drops_for_query[cond] = {
                "masked_score": masked_score,
                "drop": drop,
                "masked_positions": condition_masks[cond],
            }

        per_query.append({
            "query_id": int(q.get("image_id", qi)),
            "file_name": q.get("file_name", f"query/{qi}"),
            "pathology": list(q.get("causes", [])),
            "predicted_cause_idx": top1_cause_idx,
            "predicted_cause": cause_texts[top1_cause_idx],
            "baseline_ceah_score": baseline,
            "hybrid_score": float(hybrid[top1_local].item()),
            "candidate_pool_size": int(len(candidate_indices)),
            "evidence_count": n_ev,
            "top_alpha_position": int(top_alpha_pos),
            "top_evidence_position": int(top_evidence_pos),
            "random_evidence_position": int(random_evidence_pos),
            "alpha": summarize_alpha(q, alpha_row, args.text_kind),
            "mask_results": drops_for_query,
        })

        if (qi + 1) % 50 == 0 or qi + 1 == len(query_cases):
            elapsed = time.time() - t0
            rate = (qi + 1) / max(elapsed, 1e-9)
            eta = (len(query_cases) - qi - 1) / max(rate, 1e-9)
            print(f"[eval] {qi+1}/{len(query_cases)} rate={rate:.2f} q/s ETA={eta/60:.1f} min")

    summary = {
        "n_queries_used": len(per_query),
        "score_drop_by_condition": {
            cond: mean_stats(drops_total[cond]) for cond in cond_order
        },
        "score_drop_by_evidence_count": {
            bucket: {cond: mean_stats(vals.get(cond, [])) for cond in cond_order}
            for bucket, vals in drops_by_n.items()
        },
        "config": vars(args),
    }
    with (out_dir / "faithfulness.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for row in per_query:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n=== DDXPlus evidence faithfulness ===")
    for cond in cond_order:
        stats = summary["score_drop_by_condition"][cond]
        mean = stats["mean"]
        mean_s = "NA" if mean is None else f"{mean:.4f}"
        print(f"{cond:<20} mean_drop={mean_s} n={stats['n']}")
    print(f"[save] faithfulness.json + per_query.jsonl -> {out_dir}")


if __name__ == "__main__":
    main()

