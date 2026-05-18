"""Per-case weights for ranking-aware and aggregation-aware RVQ.

Three weight vectors over the train case bank, all sum-normalized to N:

    w_vanilla[i] = 1                                (uniform — baseline)
    w_ranking[i] = retrieval_frequency(i)           (how often i appears in
                                                     top-K of any train query
                                                     under dense scoring)
    w_agg[i]     = retrieval_frequency(i) · isolation(i)
                                                    (up-weights cases in
                                                     sparse feature regions
                                                     whose ranking error is
                                                     NOT absorbed by similar
                                                     co-retrieved peers)

Isolation for case i:
    isolation(i) = 1 − mean(z_i · z_j for j in z's top-50 neighbours)
Rationale: aggregation buffers ranking noise only when many similar cases
co-retrieve and provide overlapping causes; an isolated case has no peers
to absorb its quantization error.

(A cause-uniqueness signal was tried first and collapsed because 94.7% of
causes in this dataset are singletons; isolation in feature space captures
the same aggregation insight without that pathology.)

CLI from repo root:
    $PY -m diagnosis_model.cause_inference.rvq_rerank.compute_rvq_weights
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diagnosis_model.cause_inference.phase1_baseline import load_case_db
from diagnosis_model.cause_inference.train_case_encoder import encode_all
from diagnosis_model.cause_inference.rvq_rerank.fit_rvq import load_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", type=str,
                    default="diagnosis_model/cause_inference/outputs/"
                            "encoder_final/best_encoder.pt")
    ap.add_argument("--case_db_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db")
    ap.add_argument("--top_k", type=int, default=20,
                    help="top-K used for retrieval frequency stats (matches "
                         "production aggregation window).")
    ap.add_argument("--out", type=str,
                    default="diagnosis_model/cause_inference/outputs/"
                            "rvq_rerank/case_weights.pt")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    encoder, _ = load_encoder(Path(args.encoder_ckpt), device)
    train_cases, _, _, meta = load_case_db(Path(args.case_db_dir))
    N = len(train_cases)
    D = meta["global_dim"]
    print(f"[data] train cases = {N}  dim = {D}  top_k = {args.top_k}")

    z_train = encode_all(encoder, train_cases, device).to(device).float()  # [N, D]
    z_train = torch.nn.functional.normalize(z_train, dim=-1)

    sim = z_train @ z_train.T                                              # [N, N]
    sim.fill_diagonal_(float("-inf"))                                      # exclude self

    top_idx = sim.topk(args.top_k, dim=-1).indices                         # [N, K]
    print(f"[topk] per-query top-{args.top_k} extracted")

    # --- Ranking weight: count appearances across all query rows ---
    appearances = torch.zeros(N, device=device, dtype=torch.float32)
    appearances.scatter_add_(
        0, top_idx.reshape(-1),
        torch.ones(top_idx.numel(), device=device, dtype=torch.float32),
    )
    w_ranking = appearances / appearances.sum() * N                        # mean = 1
    print(f"[ranking] appearances stats: min={appearances.min().item():.0f}  "
          f"median={appearances.median().item():.0f}  "
          f"max={appearances.max().item():.0f}  "
          f"zero_count={(appearances == 0).sum().item()}")

    # --- Aggregation weight: feature-space isolation ---
    # isolation(i) = 1 − mean cosine to top-50 nearest neighbours in z space
    K_nbr = 50
    nbr_sim, _ = sim.topk(K_nbr, dim=-1)                                   # [N, 50]
    mean_nbr_sim = nbr_sim.mean(dim=-1)                                    # [N]
    isolation = (1.0 - mean_nbr_sim).clamp_min(0.0)                        # [N]
    print(f"[isolation] mean_nbr_sim stats: "
          f"min={mean_nbr_sim.min().item():.3f}  "
          f"median={mean_nbr_sim.median().item():.3f}  "
          f"max={mean_nbr_sim.max().item():.3f}")
    print(f"[isolation] isolation stats:    "
          f"min={isolation.min().item():.3f}  "
          f"median={isolation.median().item():.3f}  "
          f"max={isolation.max().item():.3f}")

    w_agg_raw = w_ranking * isolation
    if w_agg_raw.sum() > 0:
        w_agg = w_agg_raw / w_agg_raw.sum() * N
    else:
        w_agg = w_agg_raw
    print(f"[w_agg] min={w_agg.min().item():.4f}  "
          f"median={w_agg.median().item():.4f}  "
          f"max={w_agg.max().item():.4f}  "
          f"zero={(w_agg == 0).sum().item()}")

    # --- Inverse aggregation weight: density (= 1 - isolation) ---
    # Up-weights cases in DENSE co-retrieval regions; tests whether the
    # absorption-buffer intuition was inverted.
    density = mean_nbr_sim.clamp_min(0.0)
    w_agg_inv_raw = w_ranking * density
    if w_agg_inv_raw.sum() > 0:
        w_agg_inv = w_agg_inv_raw / w_agg_inv_raw.sum() * N
    else:
        w_agg_inv = w_agg_inv_raw
    print(f"[w_agg_inv] min={w_agg_inv.min().item():.4f}  "
          f"median={w_agg_inv.median().item():.4f}  "
          f"max={w_agg_inv.max().item():.4f}  "
          f"zero={(w_agg_inv == 0).sum().item()}")

    # Floor non-retrieved cases at 1/N so they aren't fully starved during
    # k-means (otherwise centroid collapse on those points).
    floor = 1.0 / N
    w_ranking_floored = w_ranking.clamp_min(floor)
    w_agg_floored = w_agg.clamp_min(floor)
    w_agg_inv_floored = w_agg_inv.clamp_min(floor)
    # Renormalize back so mean = 1
    w_ranking_floored = w_ranking_floored / w_ranking_floored.sum() * N
    w_agg_floored = w_agg_floored / w_agg_floored.sum() * N
    w_agg_inv_floored = w_agg_inv_floored / w_agg_inv_floored.sum() * N

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pkg = {
        "w_vanilla":  torch.ones(N, dtype=torch.float32),
        "w_ranking":  w_ranking_floored.cpu(),
        "w_agg":      w_agg_floored.cpu(),
        "w_agg_inv":  w_agg_inv_floored.cpu(),
        "top_k_used": args.top_k,
        "appearances": appearances.cpu(),
        "isolation":  isolation.cpu(),
    }
    torch.save(pkg, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
