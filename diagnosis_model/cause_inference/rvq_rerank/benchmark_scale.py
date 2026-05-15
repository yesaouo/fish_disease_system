"""Stage A5: latency / memory / throughput scaling benchmark.

For each N (case bank size), synthesizes z vectors by resampling real
z_train + Gaussian noise + L2-renormalize, quantizes via the existing
RVQ codebook (no refit), and benchmarks four retrieval methods on real
valid queries:

    dense              : brute-force q @ z.T + topk             (memory: N·D fp16)
    rvq_only           : LUT scoring + topk                     (memory: N·M bytes)
    rvq_light          : RVQ first-stage + Light reranker top-K
    rvq_full_analytic  : RVQ first-stage + dense rerank top-K

NO recall measured (synth cases have no GT). Reports per-query latency
p50/p99, throughput, and storage footprint. The point is to show how the
methods scale; quality numbers come from eval_final.py at real-N=12780.

CLI from repo root:
    /home/lab603/anaconda3/envs/SDM/bin/python \\
        -m diagnosis_model.cause_inference.rvq_rerank.benchmark_scale \\
        --rvq_dir diagnosis_model/cause_inference/outputs/rvq_rerank/rvq_M4_K256 \\
        --light_reranker_ckpt diagnosis_model/cause_inference/outputs/rvq_rerank/reranker_M4_K256_light/best.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.phase1_baseline import load_case_db
from diagnosis_model.cause_inference.train_case_encoder import encode_all
from diagnosis_model.cause_inference.rvq_rerank.fit_rvq import load_encoder
from diagnosis_model.cause_inference.rvq_rerank.rvq import RVQCodebook
from diagnosis_model.cause_inference.rvq_rerank.reranker import (
    Reranker, RerankerConfig,
)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def synthesize_z(
    real_z: torch.Tensor,
    N: int,
    noise_std: float = 0.02,
    seed: int = 42,
    chunk: int = 200_000,
) -> torch.Tensor:
    """Chunked sampling + Gaussian noise + L2-renorm. Returns [N, D] same device."""
    device = real_z.device
    D = real_z.size(1)
    out = torch.empty(N, D, device=device, dtype=real_z.dtype)
    g = torch.Generator(device=device).manual_seed(seed)
    pos = 0
    while pos < N:
        n = min(chunk, N - pos)
        idx = torch.randint(0, real_z.size(0), (n,), generator=g, device=device)
        base = real_z[idx]
        noise = torch.randn(n, D, generator=g, device=device,
                            dtype=real_z.dtype) * noise_std
        out[pos:pos + n] = F.normalize(base + noise, dim=-1)
        pos += n
    return out


def quantize_chunked(
    z: torch.Tensor,             # [N, D] fp16/fp32
    codebooks: torch.Tensor,     # [M, K, D]
    chunk: int = 50_000,
) -> tuple:
    """Encode z with given codebooks. Returns (codes [N, M] long,
    e_norm [N] fp32)."""
    N, D = z.shape
    M = codebooks.size(0)
    codes = torch.empty(N, M, dtype=torch.long, device=z.device)
    e_norm = torch.empty(N, dtype=torch.float32, device=z.device)
    pos = 0
    while pos < N:
        n = min(chunk, N - pos)
        residual = z[pos:pos + n].float().clone()
        for m in range(M):
            dists = torch.cdist(residual, codebooks[m])  # [n, K]
            k = dists.argmin(dim=-1)
            codes[pos:pos + n, m] = k
            residual = residual - codebooks[m][k]
        e_norm[pos:pos + n] = residual.norm(dim=-1)
        pos += n
    return codes, e_norm


# ---------------------------------------------------------------------------
# Inference kernels
# ---------------------------------------------------------------------------

def kernel_dense(q, z_bank, K_top):
    sims = q @ z_bank.T.float() if z_bank.dtype != q.dtype else q @ z_bank.T
    return sims.topk(K_top, dim=-1)


def kernel_rvq_only(q, codebooks, codes, K_top):
    """LUT scoring."""
    M = codebooks.size(0)
    lut = torch.einsum("bd,mkd->bmk", q, codebooks)   # [B, M, K]
    N = codes.size(0)
    sims = torch.zeros(q.size(0), N, device=q.device, dtype=q.dtype)
    for m in range(M):
        sims = sims + lut[:, m, :].index_select(1, codes[:, m])
    return sims.topk(K_top, dim=-1)


def kernel_rvq_light(q, codebooks, codes, e_norm, reranker, K_top_rerank, K_top_final):
    M = codebooks.size(0)
    D = codebooks.size(2)
    lut = torch.einsum("bd,mkd->bmk", q, codebooks)
    N = codes.size(0)
    sims = torch.zeros(q.size(0), N, device=q.device, dtype=q.dtype)
    for m in range(M):
        sims = sims + lut[:, m, :].index_select(1, codes[:, m])
    s_top, top_idx = sims.topk(K_top_rerank, dim=-1)
    # Reconstruct ẑ for top candidates
    z_hat_top = torch.zeros(q.size(0), K_top_rerank, D,
                            device=q.device, dtype=q.dtype)
    codes_top = codes[top_idx]                                       # [B, K_top, M]
    for m in range(M):
        z_hat_top = z_hat_top + codebooks[m][codes_top[:, :, m]]
    e_norm_top = e_norm[top_idx]
    delta = reranker(q, z_hat_top, codes_top, s_top, e_norm_top)
    s_final = s_top + delta
    return s_final.topk(K_top_final, dim=-1)


def kernel_rvq_full_analytic(q, codebooks, codes, z_bank, K_top_rerank, K_top_final):
    M = codebooks.size(0)
    lut = torch.einsum("bd,mkd->bmk", q, codebooks)
    N = codes.size(0)
    sims = torch.zeros(q.size(0), N, device=q.device, dtype=q.dtype)
    for m in range(M):
        sims = sims + lut[:, m, :].index_select(1, codes[:, m])
    s_top, top_idx = sims.topk(K_top_rerank, dim=-1)
    z_top = z_bank[top_idx].float()                                  # [B, K_top, D]
    s_final = (q.unsqueeze(1) * z_top).sum(dim=-1)
    return s_final.topk(K_top_final, dim=-1)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_op(fn: Callable, n_warmup: int = 5, n_runs: int = 50) -> List[float]:
    """Returns per-call latency in ms (n_runs samples)."""
    for _ in range(n_warmup):
        _ = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def storage_bytes(N: int, D: int, M: int, K: int) -> dict:
    """Per-method index storage at scale N."""
    bits_per_code = max(1, math.ceil(math.log2(max(K, 2))))
    return {
        "dense_fp32_bytes": N * D * 4,
        "dense_fp16_bytes": N * D * 2,
        "rvq_codes_bytes": math.ceil(N * M * bits_per_code / 8),
        "rvq_codebooks_bytes": M * K * D * 4,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", type=str,
                    default="diagnosis_model/cause_inference/outputs/"
                            "encoder_final/best_encoder.pt")
    ap.add_argument("--case_db_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db")
    ap.add_argument("--rvq_dir", type=str, required=True,
                    help="dir containing codebooks.pt")
    ap.add_argument("--light_reranker_ckpt", type=str, required=True)
    ap.add_argument("--output_path", type=str,
                    default="diagnosis_model/cause_inference/outputs/"
                            "rvq_rerank/benchmark_scale.json")
    ap.add_argument("--N_list", nargs="+", type=int,
                    default=[12780, 50000, 100000, 500000, 1000000])
    ap.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 32])
    ap.add_argument("--K_top_rerank", type=int, default=50)
    ap.add_argument("--K_top_final", type=int, default=20)
    ap.add_argument("--noise_std", type=float, default=0.02)
    ap.add_argument("--n_warmup", type=int, default=5)
    ap.add_argument("--n_runs", type=int, default=50)
    ap.add_argument("--dense_max_N", type=int, default=2_000_000,
                    help="Skip dense and full_analytic above this N "
                         "(z bank too big for GPU).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Load real z + reranker ---
    print("[setup] loading encoder and real z ...")
    encoder, _ = load_encoder(Path(args.encoder_ckpt), device)
    train_cases, valid_cases, _, meta = load_case_db(Path(args.case_db_dir))
    z_real_train = encode_all(encoder, train_cases, device).to(device).float()
    z_q_real = encode_all(encoder, valid_cases, device).to(device).float()
    D = z_real_train.size(1)
    del encoder
    torch.cuda.empty_cache()

    pkg = torch.load(Path(args.rvq_dir) / "codebooks.pt",
                     weights_only=False, map_location=device)
    M, K = pkg["config"]["M"], pkg["config"]["K"]
    codebooks = pkg["codebooks"].to(device).float()
    print(f"[rvq] M={M} K={K} D={D}")

    ckpt = torch.load(args.light_reranker_ckpt, weights_only=False, map_location=device)
    cfg = RerankerConfig(**dict(ckpt["reranker_config"]))
    reranker = Reranker(cfg).to(device)
    reranker.load_state_dict(ckpt["reranker_state"])
    reranker.eval()
    print(f"[reranker] variant={cfg.variant}  K_top={args.K_top_rerank}")

    results = []

    for N in args.N_list:
        print(f"\n=== N = {N:,} ===")
        if N <= z_real_train.size(0):
            z_bank = z_real_train[:N].clone()
        else:
            z_bank = synthesize_z(z_real_train, N, args.noise_std, args.seed)
        z_bank = z_bank.half()
        gc.collect(); torch.cuda.empty_cache()

        codes, e_norm = quantize_chunked(z_bank, codebooks)
        codes_u8 = codes.to(torch.uint8) if K <= 256 else codes.to(torch.int16)
        gc.collect(); torch.cuda.empty_cache()

        gpu_mem_after_build = torch.cuda.memory_allocated(device) / 1e9
        print(f"  GPU mem after build: {gpu_mem_after_build:.2f} GB")

        skip_dense = N > args.dense_max_N

        for bs in args.batch_sizes:
            q = z_q_real[:bs].clone()
            q_half = q.half()
            row = {
                "N": N, "batch_size": bs,
                "M": M, "K": K, "D": D,
                "gpu_mem_gb_index": gpu_mem_after_build,
            }

            methods = []
            if not skip_dense:
                methods.append(("dense", lambda q=q_half, z=z_bank:
                                kernel_dense(q.float(), z, args.K_top_final)))
            methods.append(("rvq_only", lambda q=q:
                            kernel_rvq_only(q, codebooks, codes,
                                            args.K_top_final)))
            methods.append(("rvq_light", lambda q=q:
                            kernel_rvq_light(q, codebooks, codes, e_norm,
                                             reranker, args.K_top_rerank,
                                             args.K_top_final)))
            if not skip_dense:
                methods.append(("rvq_full_analytic", lambda q=q, z=z_bank:
                                kernel_rvq_full_analytic(q, codebooks, codes,
                                                         z, args.K_top_rerank,
                                                         args.K_top_final)))

            line = [f"  bs={bs:>3d}:"]
            for name, fn in methods:
                try:
                    times = time_op(fn, n_warmup=args.n_warmup,
                                    n_runs=args.n_runs)
                    arr = np.array(times)
                    row[f"{name}_ms_p50"] = float(np.percentile(arr, 50))
                    row[f"{name}_ms_p99"] = float(np.percentile(arr, 99))
                    row[f"{name}_ms_mean"] = float(arr.mean())
                    row[f"{name}_qps"] = bs / (arr.mean() / 1000)
                    line.append(f"{name}={row[f'{name}_ms_p50']:.2f}ms "
                                f"({row[f'{name}_qps']:.0f}q/s)")
                except RuntimeError as e:
                    line.append(f"{name}=OOM")
                    row[f"{name}_ms_mean"] = -1
                    row[f"{name}_error"] = str(e)[:200]
                    torch.cuda.empty_cache()
            print(" ".join(line))

            # Storage size accounting
            row.update(storage_bytes(N, D, M, K))
            results.append(row)

        # Cleanup big tensors before next N
        del z_bank, codes, codes_u8, e_norm
        gc.collect(); torch.cuda.empty_cache()

    # ---- Persist + summary table ----
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({
            "config": {
                "rvq_dir": str(args.rvq_dir),
                "light_reranker_ckpt": str(args.light_reranker_ckpt),
                "M": M, "K": K, "D": D,
                "K_top_rerank": args.K_top_rerank,
                "K_top_final": args.K_top_final,
                "noise_std": args.noise_std,
                "n_warmup": args.n_warmup,
                "n_runs": args.n_runs,
            },
            "results": results,
        }, f, indent=2)

    # Pretty-print summary at bs=1
    print("\n\n" + "=" * 108)
    print(f"--- Latency @ batch_size=1 ---")
    print(f"{'N':>10s} | "
          f"{'dense':>10s} {'rvq_only':>10s} {'rvq_light':>10s} "
          f"{'rvq_full':>10s} | "
          f"{'idx (rvq/dense)':>20s}")
    print("-" * 108)
    for r in results:
        if r["batch_size"] != 1:
            continue
        n = f"{r['N']:>10,}"
        cells = []
        for name in ["dense", "rvq_only", "rvq_light", "rvq_full_analytic"]:
            v = r.get(f"{name}_ms_p50", -1)
            cells.append(f"{v:.2f}ms" if v >= 0 else "  OOM ")
        idx = (f"{r['rvq_codes_bytes']/1e6:.1f}MB / "
               f"{r['dense_fp16_bytes']/1e6:.0f}MB")
        print(f"{n} | " + " ".join(f"{c:>10s}" for c in cells)
              + f" | {idx:>20s}")
    print("=" * 108)

    print(f"\nSaved: {args.output_path}")


if __name__ == "__main__":
    main()
