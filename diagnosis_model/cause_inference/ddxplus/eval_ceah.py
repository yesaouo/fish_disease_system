"""DDXPlus CEAH reranking evaluation.

This is the DDXPlus analogue of eval_ceah.py:
  1. Retrieve similar train cases with Phase 1.
  2. Build a candidate disease pool from retrieved cases.
  3. Score candidates with CEAH, optionally mixed with Phase 1 scores.
  4. Report strict PATHOLOGY and differential-diagnosis metrics.

The case DB should be produced by ddxplus/build_case_database.py.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.ddxplus.eval_retrieval import (
    build_ddx_relevance,
    first_rank,
    mean_optional,
    ndcg_at_k,
    relevance_mass_at_k,
    relevance_mass_coverage,
)
from diagnosis_model.cause_inference.models import CEAH
from diagnosis_model.cause_inference.phase1_baseline import (
    add_recall_at_ks,
    build_candidate_pool,
    compute_case_similarities,
    load_cases,
    load_train_bank,
    load_train_cases_minimal,
    offsets_to_case_ids,
    score_candidates,
    select_positive_top_cases,
    stream_top_k_cases,
    summarize_rank_metric,
)


def minmax_norm(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    lo = x.min()
    hi = x.max()
    if hi - lo < 1e-9:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


@torch.no_grad()
def ceah_forward_for_pool(
    ceah: CEAH,
    q: dict,
    candidate_indices: List[int],
    cause_table_embs: torch.Tensor,
    in_dim: int,
    device: str,
    text_kind: str,
):
    P = len(candidate_indices)
    if P == 0:
        return (
            torch.zeros(0, device=device),
            torch.zeros(0, 0, device=device),
            torch.zeros(0, 0, dtype=torch.bool, device=device),
        )

    n_ev = q["lesion_embs"].size(0)
    global_emb = q["global_emb"].unsqueeze(0).expand(P, -1).contiguous().to(device)
    evidence_embs = q["lesion_embs"].unsqueeze(0).expand(P, -1, -1).contiguous().to(device)
    evidence_mask = torch.ones(P, n_ev, dtype=torch.bool, device=device)

    if text_kind == "none":
        text_emb = torch.zeros(P, in_dim, device=device)
        text_present = torch.zeros(P, dtype=torch.bool, device=device)
    else:
        text_key = f"text_{text_kind}_emb"
        t = q.get(text_key)
        if t is not None:
            text_emb = t.unsqueeze(0).expand(P, -1).contiguous().to(device)
            text_present = torch.ones(P, dtype=torch.bool, device=device)
        else:
            text_emb = torch.zeros(P, in_dim, device=device)
            text_present = torch.zeros(P, dtype=torch.bool, device=device)

    cand_idx_t = torch.tensor(candidate_indices, device=device, dtype=torch.long)
    cand_embs = cause_table_embs.index_select(0, cand_idx_t)
    return ceah(global_emb, text_emb, text_present, evidence_embs, evidence_mask, cand_embs)


def summarize_alpha(q: dict, alpha: List[float], text_kind: str) -> dict:
    evidence_texts = list(q.get("evidence_texts", []))
    evidence_alpha = []
    for i, text in enumerate(evidence_texts):
        pos = 2 + i
        evidence_alpha.append({
            "index": i,
            "text": text,
            "alpha": float(alpha[pos]) if pos < len(alpha) else 0.0,
        })
    evidence_alpha.sort(key=lambda x: x["alpha"], reverse=True)
    return {
        "global_alpha": float(alpha[0]) if len(alpha) > 0 else 0.0,
        "text_alpha": float(alpha[1]) if len(alpha) > 1 and text_kind != "none" else 0.0,
        "evidence_alpha": evidence_alpha,
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate DDXPlus CEAH reranking.")
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--ceah_ckpt", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--query_split", type=str, default="valid", choices=["valid", "test"])
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="hybrid score = gamma * Phase1 + (1-gamma) * CEAH")
    ap.add_argument("--dump_gamma", type=float, default=0.0)
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--top_n_causes", type=int, default=20)
    ap.add_argument("--alpha_global", type=float, default=0.25)
    ap.add_argument("--beta_lesion", type=float, default=0.75)
    ap.add_argument("--lesion_match", type=str, default="max_mean",
                    choices=["hungarian", "max_mean", "max_mean_normalized"])
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 50])
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
                    help="Use shard-streaming Phase 1 retrieval (full 1M bank, "
                         "shard-by-shard outer loop; numerically equivalent to "
                         "monolithic-bank retrieval within bf16 precision).")
    ap.add_argument("--stream_query_batch", type=int, default=64,
                    help="Query batch size for --stream mode.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    bank_dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    bank_dtype = bank_dtype_map[args.bank_dtype]
    max_train_cases = args.max_train_cases if args.max_train_cases > 0 else None

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
    meta_path = case_db_dir / "meta.json"
    meta = json.load(meta_path.open("r", encoding="utf-8")) if meta_path.exists() else {}

    cause_texts = list(cause_pack["texts"])
    cause_text_to_idx = {text: i for i, text in enumerate(cause_texts)}
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
    print(
        f"[load] train={len(train_cases)} {args.query_split}={len(query_cases)} "
        f"causes={len(cause_texts)} dim={in_dim} dataset={meta.get('dataset', 'unknown')}"
    )
    print(f"[ceah] {args.ceah_ckpt}")

    ranks_by_gamma: Dict[str, List[float]] = {f"g={g:.2f}": [] for g in args.gammas}
    cov_by_gamma: Dict[str, List[int]] = {f"g={g:.2f}": [] for g in args.gammas}
    ddx_ndcg: Dict[str, Dict[int, List[float | None]]] = {
        f"g={g:.2f}": {k: [] for k in args.ks} for g in args.gammas
    }
    ddx_mass_at_k: Dict[str, Dict[int, List[float | None]]] = {
        f"g={g:.2f}": {k: [] for k in args.ks} for g in args.gammas
    }
    ddx_pool_mass_cov: List[float | None] = []
    pool_sizes: List[int] = []
    retained_case_counts: List[int] = []
    per_query: List[dict] = []

    dump_tag = f"g={args.dump_gamma:.2f}"

    if args.stream:
        print(f"[stream] computing Phase 1 top-K for {len(query_cases)} queries "
              f"against full bank (batch={args.stream_query_batch})...")
        stream_top_idx, stream_top_w, stream_top_raw = stream_top_k_cases(
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
        if args.stream:
            top_case_idx = stream_top_idx[qi]
            top_case_w = stream_top_w[qi]
            top_case_raw_w = stream_top_raw[qi]
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
            top_case_idx, top_case_w, top_case_raw_w = select_positive_top_cases(
                sims, args.top_k_cases,
            )
        retained_case_counts.append(int(len(top_case_idx)))
        candidate_indices = build_candidate_pool(top_case_idx, train_cases)
        pool_sizes.append(len(candidate_indices))

        s1 = score_candidates(
            candidate_indices, top_case_idx, top_case_w, train_cases, cause_table_embs,
        )
        s_ceah, alpha, _ = ceah_forward_for_pool(
            ceah, q, candidate_indices, cause_table_embs, in_dim, device, args.text_kind,
        )
        s1_n = minmax_norm(s1)
        sc_n = minmax_norm(s_ceah)
        rel_by_idx = build_ddx_relevance(q, cause_text_to_idx)
        ddx_pool_mass_cov.append(relevance_mass_coverage(candidate_indices, rel_by_idx))
        # Strict pathology rank — see eval_retrieval.py for full reasoning.
        pidx = q.get("pathology_emb_idx")
        if pidx is not None:
            gt_indices = [int(pidx)]
        else:
            gt_indices = [int(x) for x in q.get("cause_emb_indices", [])]

        per_gamma_ranks = {}
        per_gamma_top = {}
        for g in args.gammas:
            tag = f"g={g:.2f}"
            hybrid = g * s1_n + (1.0 - g) * sc_n
            if candidate_indices:
                sorted_local = torch.argsort(hybrid, descending=True).detach().cpu().numpy()
                sorted_global = np.asarray(candidate_indices, dtype=np.int64)[sorted_local]
            else:
                sorted_local = np.empty(0, dtype=np.int64)
                sorted_global = np.empty(0, dtype=np.int64)

            rank = first_rank(sorted_global, gt_indices)
            ranks_by_gamma[tag].append(rank)
            cov_by_gamma[tag].append(1 if math.isfinite(rank) else 0)
            per_gamma_ranks[tag] = None if not math.isfinite(rank) else int(rank)
            for k in args.ks:
                ddx_ndcg[tag][k].append(ndcg_at_k(sorted_global, rel_by_idx, k))
                ddx_mass_at_k[tag][k].append(relevance_mass_at_k(sorted_global, rel_by_idx, k))

            if tag == dump_tag:
                top_n = min(args.top_n_causes, len(sorted_global))
                top_local = sorted_local[:top_n]
                top_global = sorted_global[:top_n]
                top_scores = (
                    hybrid[torch.from_numpy(top_local).to(device)].detach().cpu().tolist()
                    if top_n else []
                )
                rows = []
                for local_i, idx, score in zip(top_local.tolist(), top_global.tolist(), top_scores):
                    alpha_row = alpha[local_i].detach().cpu().tolist() if alpha.numel() else []
                    rows.append({
                        "cause_table_idx": int(idx),
                        "text": cause_texts[int(idx)],
                        "score": float(score),
                        "phase1_score": float(s1[local_i].item()) if s1.numel() else 0.0,
                        "ceah_score": float(s_ceah[local_i].item()) if s_ceah.numel() else 0.0,
                        "ddx_relevance": float(rel_by_idx.get(int(idx), 0.0)),
                        "alpha": summarize_alpha(q, alpha_row, args.text_kind),
                    })
                per_gamma_top[tag] = rows

        per_query.append({
            "query_id": int(q.get("image_id", qi)),
            "file_name": q.get("file_name", f"query/{qi}"),
            "pathology": list(q.get("causes", [])),
            "pathology_rank_per_gamma": per_gamma_ranks,
            "candidate_pool_size": int(len(candidate_indices)),
            "retained_positive_case_count": int(len(top_case_idx)),
            "evidence_count": int(q["lesion_embs"].size(0)),
            "evidence_texts": list(q.get("evidence_texts", [])),
            "ddx": q.get("ddx", []),
            "ddx_relevance_mass_coverage": ddx_pool_mass_cov[-1],
            "retrieved_cases": [
                {
                    "case_id": int(case_i),
                    "similarity_raw": float(top_case_raw_w[j]),
                    "similarity_weight_normalized": float(top_case_w[j]),
                    "pathology": list(train_cases[int(case_i)].get("causes", [])),
                }
                for j, case_i in enumerate(top_case_idx.tolist())
            ],
            "predicted_top_n": per_gamma_top.get(dump_tag, []),
        })

        if (qi + 1) % 50 == 0 or qi + 1 == len(query_cases):
            elapsed = time.time() - t0
            rate = (qi + 1) / max(elapsed, 1e-9)
            eta = (len(query_cases) - qi - 1) / max(rate, 1e-9)
            print(f"[eval] {qi+1}/{len(query_cases)} rate={rate:.2f} q/s ETA={eta/60:.1f} min")

    metrics_by_gamma = {}
    for tag, ranks in ranks_by_gamma.items():
        arr = np.asarray(ranks, dtype=np.float64)
        block = summarize_rank_metric(arr, cov_by_gamma[tag])
        add_recall_at_ks(block, arr, args.ks)
        metrics_by_gamma[tag] = {
            "pathology_exact": block,
            "differential_diagnosis": {
                "NDCG": {f"@{k}": mean_optional(ddx_ndcg[tag][k]) for k in args.ks},
                "relevance_mass_at_k": {
                    f"@{k}": mean_optional(ddx_mass_at_k[tag][k]) for k in args.ks
                },
            },
        }

    summary = {
        "dataset": meta.get("dataset", "DDXPlus"),
        "n_queries": len(query_cases),
        "n_train_cases": len(train_cases),
        "n_causes": len(cause_texts),
        "candidate_pool": {
            "mean": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
            "median": float(np.median(pool_sizes)) if pool_sizes else 0.0,
            "min": int(min(pool_sizes)) if pool_sizes else 0,
            "max": int(max(pool_sizes)) if pool_sizes else 0,
        },
        "retrieved_cases": {
            "requested_top_k": int(args.top_k_cases),
            "positive_similarity_only": True,
            "mean_retained": float(np.mean(retained_case_counts)) if retained_case_counts else 0.0,
            "median_retained": float(np.median(retained_case_counts)) if retained_case_counts else 0.0,
        },
        "differential_diagnosis_pool_relevance_mass_coverage": mean_optional(ddx_pool_mass_cov),
        "metrics_per_gamma": metrics_by_gamma,
        "config": vars(args),
    }

    with (out_dir / "metrics_gammas.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (out_dir / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for row in per_query:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n=== DDXPlus CEAH rerank metrics ===")
    print(f"{'gamma':>6} {'MRR':>8} " + " ".join(f"R@{k:>2}" for k in args.ks))
    for g in args.gammas:
        tag = f"g={g:.2f}"
        path = metrics_by_gamma[tag]["pathology_exact"]
        recalls = " ".join(f"{path[f'R@{k}']:.4f}" for k in args.ks)
        print(f"{g:>6.2f} {path['MRR']:>8.4f} {recalls}")
    print(f"[save] metrics_gammas.json + per_query.jsonl -> {out_dir}")


if __name__ == "__main__":
    main()

