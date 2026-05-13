"""Pre-compute the full Phase 1 case-to-case teacher score matrix for distillation.

For every train case treated as anchor:
  s_teacher(i, j) = alpha * cos(g_i, g_j) + beta * lesion_set_score(L_i, L_j)

where lesion_set_score uses Hungarian (paper best config) by default. The result
is the input for listwise-KL distillation when training the Mamba master-slave
encoder (see train_case_encoder.py).

Output:
  outputs/case_db/teacher_train_train.pt = {
      'scores': fp16 tensor [N, N]      # diagonal set to NaN
      'config': dict with alpha/beta/lesion_match/normalize flags
  }

Storage: 12780^2 * 2 bytes ~= 311 MiB.
Single-process runtime: ~15-20 minutes on the existing dataset (see pre-flight
benchmark in design doc: 30 anchors x 12780 = 2.6s -> 12780 x 12780 ~= 18 min).
Run from repo root:
    $PY -m diagnosis_model.cause_inference.preprocessing.build_teacher_table
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from diagnosis_model.cause_inference.phase1_baseline import (
    compute_case_similarities,
    load_case_db,
    stack_train_lesions,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db")
    ap.add_argument("--output_path", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db/"
                            "teacher_train_train.pt")
    ap.add_argument("--alpha_global", type=float, default=0.25)
    ap.add_argument("--beta_lesion", type=float, default=0.75)
    ap.add_argument("--lesion_match", type=str, default="hungarian",
                    choices=["hungarian", "max_mean"])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--report_every", type=int, default=200)
    args = ap.parse_args()

    case_db_dir = Path(args.case_db_dir)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_cases, valid_cases, _cause, _meta = load_case_db(case_db_dir)
    N = len(train_cases)
    print(f"Loaded {N} train cases (valid={len(valid_cases)} ignored here)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Stack and L2-normalize globals + lesion bank.
    global_stack = torch.stack([c["global_emb"] for c in train_cases]).to(device)
    global_stack = global_stack / global_stack.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    lesion_stack, offsets = stack_train_lesions(train_cases)
    lesion_stack = lesion_stack.to(device)
    lesion_stack = lesion_stack / lesion_stack.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    # Pre-normalize each case's own lesion view (small, so cache).
    case_lesions_normed = []
    for i in range(N):
        s, e = offsets[i], offsets[i + 1]
        case_lesions_normed.append(lesion_stack[s:e])

    print(f"alpha={args.alpha_global}  beta={args.beta_lesion}  "
          f"lesion_match={args.lesion_match}")
    print(f"Computing teacher scores for {N} anchors x {N} candidates "
          f"(~{N * N / 1e6:.1f}M pairs, fp16 -> {N * N * 2 / 1024**2:.0f} MiB)")

    scores = np.full((N, N), np.nan, dtype=np.float32)
    t0 = time.time()
    for i in range(N):
        sims = compute_case_similarities(
            global_stack[i],
            case_lesions_normed[i],
            global_stack,
            lesion_stack,
            offsets,
            alpha=args.alpha_global,
            beta=args.beta_lesion,
            lesion_match=args.lesion_match,
        )
        sims[i] = np.nan
        scores[i] = sims

        if (i + 1) % args.report_every == 0 or i == N - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N - i - 1)
            print(f"  anchor {i + 1}/{N}  elapsed={elapsed:6.1f}s  "
                  f"eta={eta:6.1f}s  rate={(i + 1) / elapsed:.1f} anchors/s")

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Score stats (excluding diagonal):")
    valid = ~np.isnan(scores)
    flat = scores[valid]
    print(f"  N={len(flat)}  mean={flat.mean():.3f}  std={flat.std():.3f}  "
          f"min={flat.min():.3f}  max={flat.max():.3f}")

    payload = {
        "scores": torch.from_numpy(scores).to(torch.float16),
        "config": {
            "alpha_global": args.alpha_global,
            "beta_lesion": args.beta_lesion,
            "lesion_match": args.lesion_match,
            "n_train_cases": N,
            "global_normalized": True,
            "lesion_normalized": True,
            "diagonal": "NaN",
        },
    }
    torch.save(payload, out_path)
    print(f"\nSaved {out_path}  ({out_path.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
