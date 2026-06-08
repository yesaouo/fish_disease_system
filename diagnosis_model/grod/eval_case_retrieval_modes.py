"""Compare case-retrieval accuracy for demo/app_gradio.py modes.

This is a standalone offline evaluator for the retrieval stage used by the
Gradio demo. It compares:

  base      case_db_base + encoder_base
  grod      case_db_jointDistRawP + encoder_grod
  grod_soft soft_inputs valid queries + encoder_grod_soft + bank_z_soft

The metrics are retrieval-centric:

  case_*_R@K:
      Query-level hit if any retrieved train case shares an exact/semantic
      cause with the query.

  pool_*_cov:
      Occurrence-level GT-cause coverage after expanding top_k_cases into the
      candidate cause pool. If this is low, downstream CEAH cannot recover.

Example:

  /home/lab603/anaconda3/envs/SDM/bin/python \
      -m diagnosis_model.grod.eval_case_retrieval_modes \
      --modes base grod grod_soft --top_k_cases 20
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from diagnosis_model.cause_inference.models.case_encoder import (  # noqa: E402
    EncoderConfig,
    build_encoder,
)
from diagnosis_model.cause_inference.train_case_encoder import encode_all  # noqa: E402
from diagnosis_model.grod.train_case_encoder_soft import (  # noqa: E402
    encode_all_soft,
    load_soft,
)


DEFAULT_ART = REPO_ROOT / "data/processed/current/artifacts"


@dataclass(frozen=True)
class ModeSpec:
    name: str
    case_db_dir: Path
    encoder_ckpt: Path
    soft_dir: Optional[Path] = None
    soft_bank_path: Optional[Path] = None


def mode_specs(art_root: Path) -> dict[str, ModeSpec]:
    return {
        "base": ModeSpec(
            name="base",
            case_db_dir=art_root / "db/case_db_base",
            encoder_ckpt=art_root / "models/encoder_base/best_encoder.pt",
        ),
        "grod": ModeSpec(
            name="grod",
            case_db_dir=art_root / "db/case_db_jointDistRawP",
            encoder_ckpt=art_root / "models/encoder_grod/best_encoder.pt",
        ),
        "grod_soft": ModeSpec(
            name="grod_soft",
            case_db_dir=art_root / "db/case_db_jointDistRawP",
            encoder_ckpt=art_root / "models/encoder_grod_soft/best_encoder.pt",
            soft_dir=art_root / "db/soft_inputs",
            soft_bank_path=art_root / "models/encoder_grod_soft/bank_z_soft.pt",
        ),
    }


def choose_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def load_cases(case_db_dir: Path) -> tuple[list, list, dict]:
    train_cases = torch.load(case_db_dir / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(case_db_dir / "valid_cases.pt", weights_only=False)
    cause_pack = torch.load(case_db_dir / "cause_text_embs.pt", weights_only=False)
    return train_cases, valid_cases, cause_pack


def load_encoder(
    ckpt: Path,
    device: torch.device,
    dtype_override: Optional[torch.dtype] = None,
) -> torch.nn.Module:
    pkg = torch.load(ckpt, weights_only=False, map_location="cpu")
    cfg_dict = dict(pkg["encoder_config"])
    if dtype_override is not None:
        cfg_dict["dtype"] = dtype_override
    enc = build_encoder(EncoderConfig(**cfg_dict)).to(device).eval()
    enc.load_state_dict(pkg["encoder_state"])
    return enc


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1).contiguous()


def maybe_limit(xs: list, max_items: int) -> list:
    return xs if max_items <= 0 else xs[:max_items]


@torch.no_grad()
def encode_mode(
    spec: ModeSpec,
    device: torch.device,
    batch_size: int,
    max_queries: int,
    recompute_soft_bank: bool,
) -> tuple[list, list, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return train/valid cases, cause embeddings, H_train, H_valid."""
    train_cases, valid_cases, cause_pack = load_cases(spec.case_db_dir)
    valid_cases = maybe_limit(valid_cases, max_queries)
    cause_embs = normalize_rows(cause_pack["embeddings"])

    if spec.name == "grod_soft":
        if spec.soft_dir is None or spec.soft_bank_path is None:
            raise ValueError("grod_soft spec must include soft_dir and soft_bank_path")
        enc = load_encoder(spec.encoder_ckpt, device)

        g_valid, z_valid, w_valid, _ = load_soft(spec.soft_dir / "valid.pt")
        if max_queries > 0:
            g_valid = g_valid[:max_queries]
            z_valid = z_valid[:max_queries]
            w_valid = w_valid[:max_queries]
        H_valid = encode_all_soft(
            enc, g_valid, z_valid, w_valid, device, batch_size=batch_size,
        )

        if spec.soft_bank_path.exists() and not recompute_soft_bank:
            H_train = torch.load(spec.soft_bank_path, weights_only=False)["bank_z"]
        else:
            g_train, z_train, w_train, _ = load_soft(spec.soft_dir / "train.pt")
            H_train = encode_all_soft(
                enc, g_train, z_train, w_train, device, batch_size=batch_size,
            )
    else:
        # Match app_gradio.py: base forces fp32; grod follows the bf16 encoder
        # config used by gpu_infer.py.
        dtype = torch.float32 if spec.name == "base" else torch.bfloat16
        enc = load_encoder(spec.encoder_ckpt, device, dtype_override=dtype)
        H_train = encode_all(enc, train_cases, device, batch_size=batch_size)
        H_valid = encode_all(enc, valid_cases, device, batch_size=batch_size)

    return (
        train_cases,
        valid_cases,
        cause_embs,
        normalize_rows(H_train),
        normalize_rows(H_valid),
    )


@torch.no_grad()
def retrieve_top_cases(
    H_train: torch.Tensor,
    H_valid: torch.Tensor,
    device: torch.device,
    top_k: int,
    batch_size: int,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    H_train = normalize_rows(H_train).to(device)
    H_valid = normalize_rows(H_valid).to(device)
    top_indices: List[np.ndarray] = []
    top_scores: List[np.ndarray] = []
    k = min(top_k, H_train.size(0))
    for start in range(0, H_valid.size(0), batch_size):
        end = min(start + batch_size, H_valid.size(0))
        sims = H_valid[start:end] @ H_train.T
        scores, indices = sims.topk(k, dim=1)
        top_indices.extend(indices.cpu().numpy())
        top_scores.extend(scores.float().cpu().numpy())
    return top_indices, top_scores


def first_hit_rank(
    top_idx: Sequence[int],
    train_cases: list,
    gt_idx: Sequence[int],
    cause_embs: torch.Tensor,
    semantic_threshold: float,
    semantic: bool,
) -> float:
    gt_set = set(int(x) for x in gt_idx)
    if not gt_set:
        return float("inf")
    gt_tensor = torch.tensor(sorted(gt_set), dtype=torch.long)
    gt_embs = cause_embs.index_select(0, gt_tensor)

    for rank, train_i in enumerate(top_idx, 1):
        case_cidx = [int(x) for x in train_cases[int(train_i)].get("cause_emb_indices", [])]
        if not case_cidx:
            continue
        if not semantic:
            if gt_set.intersection(case_cidx):
                return float(rank)
            continue
        case_tensor = torch.tensor(case_cidx, dtype=torch.long)
        case_embs = cause_embs.index_select(0, case_tensor)
        if float((gt_embs @ case_embs.T).max().item()) >= semantic_threshold:
            return float(rank)
    return float("inf")


def pool_coverage(
    top_idx: Sequence[int],
    train_cases: list,
    gt_idx: Sequence[int],
    cause_embs: torch.Tensor,
    semantic_threshold: float,
) -> dict:
    seen: set[int] = set()
    pool: List[int] = []
    for train_i in top_idx:
        for cidx in train_cases[int(train_i)].get("cause_emb_indices", []):
            cidx = int(cidx)
            if cidx not in seen:
                seen.add(cidx)
                pool.append(cidx)

    gt_idx = [int(x) for x in gt_idx]
    if not gt_idx:
        return {
            "pool_size": len(pool),
            "n_gt": 0,
            "exact_hits": 0,
            "sem_hits": 0,
            "exact_any": 0,
            "sem_any": 0,
            "exact_all": 0,
            "sem_all": 0,
        }

    exact_hits = sum(1 for g in gt_idx if g in seen)
    sem_hits = 0
    if pool:
        pool_tensor = torch.tensor(pool, dtype=torch.long)
        pool_embs = cause_embs.index_select(0, pool_tensor)
        for g in gt_idx:
            gt_emb = cause_embs[g].unsqueeze(0)
            sem_hits += int(float((gt_emb @ pool_embs.T).max().item()) >= semantic_threshold)

    return {
        "pool_size": len(pool),
        "n_gt": len(gt_idx),
        "exact_hits": exact_hits,
        "sem_hits": sem_hits,
        "exact_any": int(exact_hits > 0),
        "sem_any": int(sem_hits > 0),
        "exact_all": int(exact_hits == len(gt_idx)),
        "sem_all": int(sem_hits == len(gt_idx)),
    }


def mrr(ranks: np.ndarray) -> float:
    if ranks.size == 0:
        return 0.0
    finite = np.isfinite(ranks)
    reciprocal = np.zeros_like(ranks, dtype=np.float64)
    reciprocal[finite] = 1.0 / ranks[finite]
    return float(reciprocal.mean())


def summarize_mode(
    mode: str,
    train_cases: list,
    valid_cases: list,
    cause_embs: torch.Tensor,
    top_indices: List[np.ndarray],
    top_scores: List[np.ndarray],
    ks: Sequence[int],
    top_k_cases: int,
    semantic_threshold: float,
    elapsed_s: float,
) -> tuple[dict, List[dict]]:
    cause_embs = normalize_rows(cause_embs.cpu())
    exact_ranks: List[float] = []
    sem_ranks: List[float] = []
    pool_sizes: List[int] = []
    top1_max_cause_cos: List[float] = []
    skipped_empty_gt = 0
    pool_gt_total = 0
    pool_exact_hits = 0
    pool_sem_hits = 0
    pool_exact_any = 0
    pool_sem_any = 0
    pool_exact_all = 0
    pool_sem_all = 0
    per_query: List[dict] = []

    for qi, (case, topi, topw) in enumerate(zip(valid_cases, top_indices, top_scores)):
        gt_idx = [int(x) for x in case.get("cause_emb_indices", [])]
        if not gt_idx:
            skipped_empty_gt += 1
            continue

        exact_rank = first_hit_rank(
            topi, train_cases, gt_idx, cause_embs, semantic_threshold, semantic=False,
        )
        sem_rank = first_hit_rank(
            topi, train_cases, gt_idx, cause_embs, semantic_threshold, semantic=True,
        )
        exact_ranks.append(exact_rank)
        sem_ranks.append(sem_rank)

        pool = pool_coverage(
            topi[:top_k_cases], train_cases, gt_idx, cause_embs, semantic_threshold,
        )
        pool_sizes.append(pool["pool_size"])
        pool_gt_total += pool["n_gt"]
        pool_exact_hits += pool["exact_hits"]
        pool_sem_hits += pool["sem_hits"]
        pool_exact_any += pool["exact_any"]
        pool_sem_any += pool["sem_any"]
        pool_exact_all += pool["exact_all"]
        pool_sem_all += pool["sem_all"]

        if len(topi) > 0:
            first_case_idx = [int(x) for x in train_cases[int(topi[0])].get("cause_emb_indices", [])]
            if first_case_idx:
                gt_embs = cause_embs.index_select(0, torch.tensor(gt_idx, dtype=torch.long))
                case_embs = cause_embs.index_select(0, torch.tensor(first_case_idx, dtype=torch.long))
                top1_max_cause_cos.append(float((gt_embs @ case_embs.T).max().item()))

        per_query.append(
            {
                "mode": mode,
                "query_index": qi,
                "file_name": case.get("file_name"),
                "gt_cause_emb_indices": gt_idx,
                "top_case_indices": [int(x) for x in topi[:top_k_cases].tolist()],
                "top_case_scores": [float(x) for x in topw[:top_k_cases].tolist()],
                "rank_exact": None if not np.isfinite(exact_rank) else int(exact_rank),
                "rank_sem": None if not np.isfinite(sem_rank) else int(sem_rank),
                "pool_size": pool["pool_size"],
            }
        )

    exact_np = np.asarray(exact_ranks, dtype=np.float64)
    sem_np = np.asarray(sem_ranks, dtype=np.float64)
    n_eval = int(sem_np.size)
    summary = {
        "mode": mode,
        "n_queries": len(valid_cases),
        "n_eval_queries": n_eval,
        "skipped_empty_gt": int(skipped_empty_gt),
        "top_k_cases": int(top_k_cases),
        "semantic_threshold": float(semantic_threshold),
        "case_exact_MRR": mrr(exact_np),
        "case_sem_MRR": mrr(sem_np),
        "pool_exact_cov": float(pool_exact_hits / max(1, pool_gt_total)),
        "pool_sem_cov": float(pool_sem_hits / max(1, pool_gt_total)),
        "pool_exact_any": float(pool_exact_any / max(1, n_eval)),
        "pool_sem_any": float(pool_sem_any / max(1, n_eval)),
        "pool_exact_all": float(pool_exact_all / max(1, n_eval)),
        "pool_sem_all": float(pool_sem_all / max(1, n_eval)),
        "mean_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "mean_top1_max_cause_cos": (
            float(np.mean(top1_max_cause_cos)) if top1_max_cause_cos else 0.0
        ),
        "eval_time_s": float(elapsed_s),
        "per_query_ms": float(elapsed_s / max(1, len(valid_cases)) * 1000.0),
    }
    for k in ks:
        summary[f"case_exact_R@{k}"] = float((exact_np <= k).mean()) if n_eval else 0.0
        summary[f"case_sem_R@{k}"] = float((sem_np <= k).mean()) if n_eval else 0.0
    return summary, per_query


def write_outputs(output_dir: Path, summaries: List[dict], per_query: List[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    if summaries:
        fields: List[str] = []
        for row in summaries:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summaries)
    with (output_dir / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for row in per_query:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_table(summaries: List[dict], ks: Sequence[int]) -> None:
    if not summaries:
        return
    header = (
        f"{'mode':<10} "
        + " ".join(f"semR@{k:<3}" for k in ks)
        + " semMRR poolSem poolExact poolSize ms/q"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in summaries:
        recalls = " ".join(f"{row[f'case_sem_R@{k}']:.3f}" for k in ks)
        print(
            f"{row['mode']:<10} {recalls} "
            f"{row['case_sem_MRR']:.3f}  "
            f"{row['pool_sem_cov']:.3f}   "
            f"{row['pool_exact_cov']:.3f}    "
            f"{row['mean_pool_size']:.1f}   "
            f"{row['per_query_ms']:.1f}"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art_root", type=Path, default=DEFAULT_ART)
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["base", "grod", "grod_soft"],
        choices=["base", "grod", "grod_soft"],
    )
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 50])
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="0 means all valid queries; use a small value for smoke tests.",
    )
    ap.add_argument(
        "--recompute_soft_bank",
        action="store_true",
        help="Encode soft train inputs instead of loading bank_z_soft.pt.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_ART / "eval/case_retrieval_modes",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    specs = mode_specs(args.art_root)
    retrieve_k = max(max(args.ks), args.top_k_cases)
    summaries: List[dict] = []
    all_per_query: List[dict] = []

    print(
        f"[eval] device={device} retrieve_k={retrieve_k} "
        f"top_k_cases={args.top_k_cases} sem_thr={args.semantic_threshold}"
    )
    for mode in args.modes:
        spec = specs[mode]
        print(f"\n[{mode}] case_db={spec.case_db_dir}")
        t0 = time.time()
        train_cases, valid_cases, cause_embs, H_train, H_valid = encode_mode(
            spec=spec,
            device=device,
            batch_size=args.batch_size,
            max_queries=args.max_queries,
            recompute_soft_bank=args.recompute_soft_bank,
        )
        topi, topw = retrieve_top_cases(
            H_train=H_train,
            H_valid=H_valid,
            device=device,
            top_k=retrieve_k,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - t0
        summary, per_query = summarize_mode(
            mode=mode,
            train_cases=train_cases,
            valid_cases=valid_cases,
            cause_embs=cause_embs,
            top_indices=topi,
            top_scores=topw,
            ks=args.ks,
            top_k_cases=args.top_k_cases,
            semantic_threshold=args.semantic_threshold,
            elapsed_s=elapsed,
        )
        summaries.append(summary)
        all_per_query.extend(per_query)
        print(
            f"[{mode}] sem_R@10={summary.get('case_sem_R@10', 0.0):.4f} "
            f"pool_sem_cov={summary['pool_sem_cov']:.4f} "
            f"time={elapsed:.1f}s"
        )

    print_table(summaries, args.ks)
    write_outputs(args.output_dir, summaries, all_per_query)
    print(f"\n[save] {args.output_dir / 'summary.csv'}")
    print(f"[save] {args.output_dir / 'per_query.jsonl'}")


if __name__ == "__main__":
    main()
