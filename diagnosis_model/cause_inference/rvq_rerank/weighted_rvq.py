"""Weighted RVQ for ranking-aware / aggregation-aware variants.

Standard k-means minimises  Σ_i ‖x_i − c_{k_i}‖².
Weighted k-means minimises  Σ_i w_i ‖x_i − c_{k_i}‖².

Assignment is unchanged (still nearest centroid); the centroid update step
becomes a weighted mean instead of a plain mean. Lloyd's algorithm still
converges monotonically in the weighted objective.

Reuses the RVQCodebook container so downstream code (encode / decode / LUT /
eval_absorption_surface) works without modification — only the fit step
differs.
"""

from __future__ import annotations

from typing import Tuple

import torch

from diagnosis_model.cause_inference.rvq_rerank.rvq import RVQCodebook


@torch.no_grad()
def _weighted_kmeans(
    x: torch.Tensor,                # [N, D]
    w: torch.Tensor,                # [N]
    K: int,
    n_iters: int = 25,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Lloyd's k-means with per-point weights w_i ≥ 0.

    Returns (centroids [K, D], assigns [N], stats).
    """
    N, D = x.shape
    assert w.shape == (N,), f"weights shape {w.shape} != ({N},)"
    assert K <= N, f"K={K} > N={N}"
    g = torch.Generator(device=x.device).manual_seed(seed)
    # Seed centroids by weight-proportional sampling so heavy points are
    # more likely to be initial centroids (helps when weight mass is skewed).
    probs = w.clamp_min(0).double()
    probs = probs / probs.sum()
    init_idx = torch.multinomial(probs, K, replacement=False, generator=g)
    centroids = x[init_idx].clone()

    n_reinits = 0
    for _ in range(n_iters):
        dists = torch.cdist(x, centroids)                    # [N, K]
        assigns = dists.argmin(dim=-1)                       # [N]
        new_c = torch.zeros_like(centroids)
        wsums = torch.zeros(K, device=x.device, dtype=x.dtype)
        # Weighted accumulation
        new_c.index_add_(0, assigns, x * w.unsqueeze(-1))
        wsums.index_add_(0, assigns, w)
        mask = wsums > 0
        new_c[mask] = new_c[mask] / wsums[mask].unsqueeze(-1)
        dead = ~mask
        n_dead = int(dead.sum().item())
        if n_dead > 0:
            n_reinits += n_dead
            new_perm = torch.multinomial(
                probs, n_dead, replacement=False, generator=g,
            )
            new_c[dead] = x[new_perm]
        centroids = new_c

    dists = torch.cdist(x, centroids)
    assigns = dists.argmin(dim=-1)
    err = x - centroids[assigns]
    # Both mean (unweighted, for comparability with vanilla) and weighted mse
    mse_uniform = err.pow(2).mean().item()
    mse_weighted = (w.unsqueeze(-1) * err.pow(2)).sum().item() / max(w.sum().item(), 1e-12)
    return centroids, assigns, {
        "recon_mse_uniform": mse_uniform,
        "recon_mse_weighted": mse_weighted,
        "n_reinits_total": n_reinits,
    }


class WeightedRVQCodebook(RVQCodebook):
    """RVQCodebook with weighted-k-means fit. Same encode/decode/lut as parent."""

    @torch.no_grad()
    def fit_weighted(
        self,
        z: torch.Tensor,                # [N, D]
        w: torch.Tensor,                # [N]
        n_iters: int = 25,
        seed: int = 0,
        verbose: bool = True,
    ) -> list:
        """Sequentially fit M codebooks on weighted residuals."""
        assert z.dim() == 2 and z.size(1) == self.D, \
            f"expected [N, {self.D}], got {tuple(z.shape)}"
        assert w.shape == (z.size(0),), \
            f"weights shape {w.shape} != ({z.size(0)},)"
        residual = z.clone().float()
        w = w.to(z.device).float()
        stats = []
        for m in range(self.M):
            cb, assigns, km_stats = _weighted_kmeans(
                residual, w, self.K, n_iters=n_iters, seed=seed + m,
            )
            self.codebooks[m] = cb
            usage = assigns.unique().numel() / self.K
            residual = residual - cb[assigns]
            cum_mse_u = residual.pow(2).mean().item()
            cum_mse_w = (
                (w.unsqueeze(-1) * residual.pow(2)).sum().item()
                / max(w.sum().item(), 1e-12)
            )
            stats.append({
                "level": m,
                "level_recon_mse_uniform": km_stats["recon_mse_uniform"],
                "level_recon_mse_weighted": km_stats["recon_mse_weighted"],
                "cum_recon_mse_uniform": cum_mse_u,
                "cum_recon_mse_weighted": cum_mse_w,
                "usage_rate": usage,
                "n_dead_reinits": km_stats["n_reinits_total"],
            })
            if verbose:
                print(
                    f"[wRVQ fit] L{m}: "
                    f"mse_u={cum_mse_u:.4e}  mse_w={cum_mse_w:.4e}  "
                    f"usage={usage:.2%}  reinits={km_stats['n_reinits_total']}"
                )
        self.fitted.fill_(True)
        return stats
