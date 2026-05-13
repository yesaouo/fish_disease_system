"""Evaluate a trained Mamba (or baseline) case encoder against Phase 1.

Reports:
  - sem_R@K, sem_MRR  (semantic exact-match across train cases' GT cause sets)
  - retrieval latency: per-query µs, comparing single-vector cosine vs Phase 1
    hungarian (the teacher).

Usage:
    CC=/usr/bin/gcc-12 \
    /home/lab603/anaconda3/envs/mamba3/bin/python \
        -m diagnosis_model.cause_inference.eval_mamba_encoder \
        --checkpoint diagnosis_model/cause_inference/outputs/mamba_encoder/mamba_v1/best_encoder.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from diagnosis_model.cause_inference.models.case_encoder import (
    EncoderConfig,
    build_encoder,
)
from diagnosis_model.cause_inference.phase1_baseline import (
    compute_case_similarities,
    load_case_db,
    stack_train_lesions,
)
from diagnosis_model.cause_inference.train_case_encoder import (
    encode_all,
    retrieval_metrics,
)


def _load_encoder(checkpoint_path: Path, device: torch.device):
    pkg = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    cfg_dict = pkg["encoder_config"]
    cfg_dict["dtype"] = torch.bfloat16              # serialized as enum int sometimes
    cfg = EncoderConfig(**cfg_dict)
    encoder = build_encoder(cfg).to(device)
    encoder.load_state_dict(pkg["encoder_state"])
    encoder.eval()
    return encoder, cfg, pkg.get("metrics", None)


def _phase1_latency(
    valid_cases, train_cases, alpha=0.25, beta=0.75, n_queries=200
):
    """Wall-clock per-query time for the existing Phase 1 hungarian retrieval."""
    G_t = torch.stack([c["global_emb"] for c in train_cases])
    G_t = G_t / G_t.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    L_t, off = stack_train_lesions(train_cases)
    L_t = L_t / L_t.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_t = G_t.to(device)
    L_t = L_t.to(device)

    t0 = time.time()
    n = min(n_queries, len(valid_cases))
    for q in valid_cases[:n]:
        g = q["global_emb"].to(device)
        g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        L = q["lesion_embs"].to(device)
        if L.size(0) > 0:
            L = L / L.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        _ = compute_case_similarities(g, L, G_t, L_t, off, alpha, beta, "hungarian")
    return (time.time() - t0) / n


def _student_latency(H_train, encoder, valid_cases, device, n_queries=200):
    """Per-query time for student: encode query + cosine top-K against H_train."""
    from diagnosis_model.cause_inference.train_case_encoder import (
        CaseEncoderDataset, make_collate
    )
    ds = CaseEncoderDataset(valid_cases[:n_queries])
    collate = make_collate(D=H_train.size(1))
    H_train_dev = H_train.to(device).float()

    t0 = time.time()
    for i in range(len(ds)):
        batch = collate([ds[i]])
        with torch.no_grad():
            z = encoder(
                batch["global_emb"].to(device),
                batch["lesion_pad"].to(device),
                batch["lesion_lens"].to(device),
            )
            _ = (z.float() @ H_train_dev.T).argsort(dim=1, descending=True)[:, :100]
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--case_db_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--latency_queries", type=int, default=200)
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 5, 10, 20, 100])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_cases, valid_cases, _, _ = load_case_db(Path(args.case_db_dir))
    print(f"train={len(train_cases)} valid={len(valid_cases)}")

    encoder, cfg, train_metrics = _load_encoder(Path(args.checkpoint), device)
    print(f"loaded encoder_type={cfg.encoder_type}  "
          f"saved_metrics={train_metrics}")

    # Retrieval metrics
    H_train = encode_all(encoder, train_cases, device)
    H_valid = encode_all(encoder, valid_cases, device)
    metrics = retrieval_metrics(H_valid, H_train, valid_cases, train_cases,
                                Ks=tuple(args.Ks))
    print("\n=== Retrieval metrics ===")
    for k, v in metrics.items():
        print(f"  {k:>12s} = {v:.4f}")

    # Latency comparison
    print("\n=== Latency (per query) ===")
    p1 = _phase1_latency(valid_cases, train_cases, n_queries=args.latency_queries)
    st = _student_latency(H_train, encoder, valid_cases, device,
                          n_queries=args.latency_queries)
    print(f"  Phase 1 (hungarian):   {p1*1000:.2f} ms/query")
    print(f"  Student (single-vec):  {st*1000:.2f} ms/query")
    print(f"  speedup:               {p1/st:.1f}x")


if __name__ == "__main__":
    main()
