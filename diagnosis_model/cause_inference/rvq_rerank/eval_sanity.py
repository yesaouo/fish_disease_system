"""CRR-DeepRVQ sanity check: dense (q·z) vs RVQ-only (q·ẑ).

Reuses the existing eval_phase1_aligned.evaluate() pipeline with two
pluggable scorers:

    dense    : s(q, i) = z_q · z_i             (the current Phase 3 baseline)
    rvq_only : s(q, i) = z_q · ẑ_i             (case bank quantized; query fresh)

This defines the recovery ceiling for the eventual reranker: if rvq_only
already matches dense within noise on sem_R@K, the reranker has no room.

CLI from repo root (SDM env):
    /home/lab603/anaconda3/envs/SDM/bin/python \\
        -m diagnosis_model.cause_inference.rvq_rerank.eval_sanity \\
        --encoder_ckpt diagnosis_model/cause_inference/outputs/encoder_final/best_encoder.pt \\
        --case_db_dir diagnosis_model/cause_inference/outputs/case_db \\
        --rvq_dir diagnosis_model/cause_inference/outputs/rvq_rerank/rvq_M4_K256 \\
        --top_k_cases 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diagnosis_model.cause_inference.eval_phase1_aligned import evaluate
from diagnosis_model.cause_inference.phase1_baseline import load_case_db
from diagnosis_model.cause_inference.train_case_encoder import encode_all
from diagnosis_model.cause_inference.rvq_rerank.fit_rvq import load_encoder
from diagnosis_model.cause_inference.rvq_rerank.rvq import RVQCodebook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", type=str, required=True)
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--rvq_dir", type=str, required=True,
                    help="Dir containing codebooks.pt (from fit_rvq.py)")
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 5, 10, 20, 100])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output_json", type=str, default="",
                    help="Optional path to dump the report JSON.")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rvq_dir = Path(args.rvq_dir)

    # 1. Load RVQ codebook
    pkg = torch.load(rvq_dir / "codebooks.pt", weights_only=False, map_location=device)
    cfg = pkg["config"]
    M, K, D = cfg["M"], cfg["K"], cfg["D"]
    rvq = RVQCodebook(M=M, K=K, D=D).to(device)
    rvq.codebooks.copy_(pkg["codebooks"].to(device))
    rvq.fitted.copy_(pkg["fitted"].to(device))
    print(f"[rvq] M={M}  K={K}  D={D}")

    # 2. Load encoder
    encoder, _ = load_encoder(Path(args.encoder_ckpt), device)

    # 3. Load case database
    train_cases, valid_cases, cause_pkg, _ = load_case_db(Path(args.case_db_dir))
    cause_embs = cause_pkg["embeddings"]
    print(f"[data] train={len(train_cases)}  valid={len(valid_cases)}")

    # 4. Encode (encoder head already L2-normalizes)
    z_train = encode_all(encoder, train_cases, device).to(device).float()
    z_valid = encode_all(encoder, valid_cases, device).to(device).float()

    # 5. Quantize ONLY the case bank (query stays in fp32, simulates deployment)
    _, z_hat_train, e_train = rvq.encode(z_train)
    zhat_norms = z_hat_train.norm(dim=-1)
    print(f"||z_hat_train||  mean={zhat_norms.mean().item():.4f}  "
          f"min={zhat_norms.min().item():.4f}  max={zhat_norms.max().item():.4f}")

    # 6. Precompute the two score matrices once
    sim_dense = (z_valid @ z_train.T).cpu().numpy()        # [Nv, Nt]
    sim_rvq = (z_valid @ z_hat_train.T).cpu().numpy()      # [Nv, Nt]
    diff = sim_dense - sim_rvq
    print(f"[diff sim_dense - sim_rvq]  mean={float(diff.mean()):+.5f}  "
          f"std={float(diff.std()):.5f}  "
          f"abs_max={float(abs(diff).max()):.5f}")

    def dense_score(qi, q):
        return sim_dense[qi]

    def rvq_score(qi, q):
        return sim_rvq[qi]

    # 7. Run the standard Phase 1 evaluation pipeline with each scorer
    Ks = tuple(args.Ks)
    print("\n[eval dense]")
    m_dense = evaluate(
        dense_score, valid_cases, train_cases, cause_embs,
        top_k_cases=args.top_k_cases,
        semantic_threshold=args.semantic_threshold,
        Ks=Ks, device=device,
    )
    print("[eval rvq_only]")
    m_rvq = evaluate(
        rvq_score, valid_cases, train_cases, cause_embs,
        top_k_cases=args.top_k_cases,
        semantic_threshold=args.semantic_threshold,
        Ks=Ks, device=device,
    )

    # 8. Side-by-side table
    print("\n" + "=" * 84)
    header_Ks = " ".join(f"R@{k:>3d}" for k in args.Ks)
    print(f"{'method':<14s} | {header_Ks} |  MRR  |  cov  | per-q ms")
    print("-" * 84)
    for name, m in [("dense", m_dense), ("rvq_only", m_rvq)]:
        cells = " ".join(f"{m[f'sem_R@{k}']:.3f}" for k in args.Ks)
        print(f"{name:<14s} | {cells} | "
              f"{m['sem_MRR']:.3f} | {m['coverage']:.3f} | "
              f"{m['per_query_ms']:6.2f}")
    print("-" * 84)
    delta_cells = " ".join(
        f"{(m_dense[f'sem_R@{k}'] - m_rvq[f'sem_R@{k}']) * 100:+5.2f}pp"
        for k in args.Ks
    )
    print(f"{'Δ (dense-rvq)':<14s} | {delta_cells}")
    print("=" * 84)

    # 9. Optional dump
    if args.output_json:
        report = {
            "config": {
                "encoder_ckpt": str(args.encoder_ckpt),
                "rvq_dir": str(args.rvq_dir),
                "rvq_M": M, "rvq_K": K, "D": D,
                "top_k_cases": args.top_k_cases,
                "semantic_threshold": args.semantic_threshold,
                "Ks": list(args.Ks),
            },
            "stats": {
                "zhat_norm_mean": zhat_norms.mean().item(),
                "zhat_norm_min": zhat_norms.min().item(),
                "zhat_norm_max": zhat_norms.max().item(),
                "sim_diff_mean": float(diff.mean()),
                "sim_diff_std": float(diff.std()),
                "sim_diff_abs_max": float(abs(diff).max()),
            },
            "dense": m_dense,
            "rvq_only": m_rvq,
            "delta_pp": {
                f"sem_R@{k}": (m_dense[f"sem_R@{k}"] - m_rvq[f"sem_R@{k}"]) * 100
                for k in args.Ks
            },
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
