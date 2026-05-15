"""Residual Vector Quantization for CRR-DeepRVQ Phase 3.

Stateless fit via sequential k-means on residuals. Stateful encode / decode /
LUT-based query scoring on a frozen codebook. Designed for post-hoc
compression of L2-normed DeepSets case embeddings (768-dim).

Math:
    Level m operates on r_{m-1}, the residual after levels 0..m-1.
    r_{-1} = z; cb_m = kmeans(r_{m-1}); k_m = argmin_k ||r_{m-1} - cb_m[k]||;
    r_m   = r_{m-1} - cb_m[k_m].
    ẑ = Σ_m cb_m[k_m];  e = z - ẑ = r_{M-1}.

LUT scoring:
    For a query q, precompute LUT[m, k] = q · cb_m[k]  (M·K dot products).
    Then s_first(q, i) = Σ_m LUT[m, k_{i,m}] — only M adds per candidate.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------------
# K-means (plain Lloyd, with dead-code reinit)
# ---------------------------------------------------------------------------

def _kmeans(
    x: torch.Tensor,                # [N, D] float
    K: int,
    n_iters: int = 25,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Lloyd k-means in PyTorch. Re-inits empty clusters from random points.

    Returns:
        centroids: [K, D] same dtype/device as x
        assigns:   [N]    long
        stats:     dict {recon_mse, n_reinits_total}
    """
    N, D = x.shape
    assert K <= N, f"K={K} > N={N}"
    g = torch.Generator(device=x.device).manual_seed(seed)
    perm = torch.randperm(N, generator=g, device=x.device)[:K]
    centroids = x[perm].clone()
    n_reinits = 0
    for _ in range(n_iters):
        dists = torch.cdist(x, centroids)                   # [N, K]
        assigns = dists.argmin(dim=-1)
        new_c = torch.zeros_like(centroids)
        counts = torch.zeros(K, device=x.device, dtype=x.dtype)
        new_c.index_add_(0, assigns, x)
        counts.index_add_(
            0, assigns, torch.ones(N, device=x.device, dtype=x.dtype),
        )
        mask = counts > 0
        new_c[mask] = new_c[mask] / counts[mask].unsqueeze(-1)
        dead = ~mask
        n_dead = int(dead.sum().item())
        if n_dead > 0:
            n_reinits += n_dead
            new_perm = torch.randperm(N, generator=g, device=x.device)[:n_dead]
            new_c[dead] = x[new_perm]
        centroids = new_c

    dists = torch.cdist(x, centroids)
    assigns = dists.argmin(dim=-1)
    err = x - centroids[assigns]
    mse = err.pow(2).mean().item()
    return centroids, assigns, {"recon_mse": mse, "n_reinits_total": n_reinits}


# ---------------------------------------------------------------------------
# RVQ module
# ---------------------------------------------------------------------------

class RVQCodebook(nn.Module):
    """M-level Residual Vector Quantization, K codes per level."""

    def __init__(self, M: int, K: int, D: int):
        super().__init__()
        self.M = M
        self.K = K
        self.D = D
        self.register_buffer("codebooks", torch.zeros(M, K, D))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(
        self,
        z: torch.Tensor,                # [N, D]
        n_iters: int = 25,
        seed: int = 0,
        verbose: bool = True,
    ) -> list:
        """Sequentially fit M codebooks on z and its residuals.

        Returns a list of per-level stats dicts:
            {level, level_recon_mse, cumulative_recon_mse,
             usage_rate, n_dead_reinits}
        """
        assert z.dim() == 2 and z.size(1) == self.D, \
            f"expected [N, {self.D}], got {tuple(z.shape)}"
        residual = z.clone().float()
        stats = []
        for m in range(self.M):
            cb, assigns, km_stats = _kmeans(
                residual, self.K, n_iters=n_iters, seed=seed + m,
            )
            self.codebooks[m] = cb
            usage = assigns.unique().numel() / self.K
            residual = residual - cb[assigns]
            cum_mse = residual.pow(2).mean().item()
            stats.append({
                "level": m,
                "level_recon_mse": km_stats["recon_mse"],
                "cumulative_recon_mse": cum_mse,
                "usage_rate": usage,
                "n_dead_reinits": km_stats["n_reinits_total"],
            })
            if verbose:
                print(
                    f"[RVQ fit] level {m}: "
                    f"level_mse={km_stats['recon_mse']:.6e}  "
                    f"cum_mse={cum_mse:.6e}  "
                    f"usage={usage:.2%}  "
                    f"reinits={km_stats['n_reinits_total']}"
                )
        self.fitted.fill_(True)
        return stats

    @torch.no_grad()
    def encode(
        self, z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """z [N, D] -> (codes [N, M] long, z_hat [N, D] float, e [N, D] float)."""
        assert bool(self.fitted.item()), "RVQ not fitted yet"
        residual = z.float().clone()
        codes_list = []
        for m in range(self.M):
            dists = torch.cdist(residual, self.codebooks[m])    # [N, K]
            k = dists.argmin(dim=-1)                            # [N]
            codes_list.append(k)
            residual = residual - self.codebooks[m][k]
        codes = torch.stack(codes_list, dim=-1)                 # [N, M]
        z_hat = z.float() - residual
        e = residual
        return codes, z_hat, e

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [N, M] -> z_hat [N, D]."""
        N = codes.size(0)
        z_hat = torch.zeros(N, self.D, device=codes.device, dtype=self.codebooks.dtype)
        for m in range(self.M):
            z_hat = z_hat + self.codebooks[m][codes[:, m].long()]
        return z_hat

    @torch.no_grad()
    def lut_scores(
        self,
        q: torch.Tensor,                # [D] or [B, D]
        codes: torch.Tensor,            # [N, M]
    ) -> torch.Tensor:
        """LUT-based dot product q · ẑ_i for all i.

        Returns: [N] if q is 1-D else [B, N].
        """
        single = q.dim() == 1
        if single:
            q = q.unsqueeze(0)
        B = q.size(0)
        N = codes.size(0)
        # LUT[b, m, k] = q[b] · cb[m, k]
        lut = torch.einsum(
            "bd,mkd->bmk", q.float(), self.codebooks.float(),
        )                                                       # [B, M, K]
        codes_long = codes.long()
        scores = torch.zeros(B, N, device=q.device)
        for m in range(self.M):
            # lut[:, m, :]: [B, K]; codes_long[:, m]: [N]
            scores = scores + lut[:, m, :].index_select(1, codes_long[:, m])
        return scores.squeeze(0) if single else scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def codes_to_compact(codes: torch.Tensor, K: int) -> torch.Tensor:
    """Cast codes [N, M] long -> uint8 / int16 / int32 based on K."""
    if K <= 256:
        return codes.to(torch.uint8)
    if K <= 32768:
        return codes.to(torch.int16)
    return codes.to(torch.int32)
