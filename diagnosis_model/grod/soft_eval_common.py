"""Shared CEAM cascade metric loop + faithfulness masking for the `grod/` evals.

This is the single place a metric convention may change; keep it that way or the
thesis tables stop being comparable across scripts.

  `evaluate(be, ...)`  — the ranking-metric loop. `be` is a `Backend`: it owns
      the artifacts and a `query(qi)` closure returning that query's CEAM
      evidence, so callers differ only in how the evidence is produced (soft /
      hard-binarized / separated-baseline).

  `CeahBatch` + `faithfulness_drops` — the evidence-masking probe (no_global /
      no_lesion / no_top_α / no_random score drops for the top-1 cause).

Conventions (L2-norm, miss=+inf, occurrence-level cluster) mirror
`eval_ceah.py` / `phase1_baseline.py`. Note `evaluate` does NOT normalize
`bank_z` / `H` — a backend that wants that does it when it builds them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.phase1_baseline import (
    MISS_RANK, add_recall_at_ks, build_candidate_pool, score_candidates,
    select_positive_top_cases, summarize_rank_metric,
)

PREC_NS = [1, 3, 5, 10]


def minmax(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


def ndcg_at_k(rel_in_order: np.ndarray, k: int) -> float:
    """rel_in_order: binary relevance of candidates in predicted rank order."""
    rel = rel_in_order[:k].astype(np.float64)
    if rel.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float((rel * discounts).sum())
    ideal = np.sort(rel_in_order)[::-1][:k].astype(np.float64)
    idcg = float((ideal * discounts[: ideal.size]).sum())
    return dcg / idcg if idcg > 0 else 0.0


def topk_lesions(z_all, w, top_k):
    """-> (z[k,D], w[k]) keeping the top-`top_k` queries by gate weight."""
    K = min(top_k, w.numel())
    idx = w.topk(K).indices
    return z_all[idx], w[idx]


@dataclass
class Backend:
    """One evaluated setting: artifacts + a per-query evidence closure.

    `query(qi)` -> (gt, g[D], text_emb[D] or None, z_sel[K,D], w_sel[K] or None)
    where `text_emb=None` means the CEAM text slot is absent (vision-only) and
    `w_sel=None` means no Region-Gate weighting.
    """
    name: str
    train_cases: list
    n_queries: int
    cause_embs: torch.Tensor
    cause_texts: List[str]
    bank_z: torch.Tensor
    H: torch.Tensor
    ceah: object
    query: Callable
    mask_lesions: bool = False

    @property
    def in_dim(self) -> int:
        return self.cause_embs.size(-1)


@torch.no_grad()
def evaluate(be: Backend, gammas, top_k_cases, sem_thresh, cluster_id_array,
             max_queries, device) -> dict:
    """Retrieval -> candidate pool -> CEAM -> ranking metrics, per gamma.

    `cluster_id_array` is the [V] per-cause cluster id (artifacts.load_cluster_ids),
    or None to skip the cluster metrics.
    """
    cause_embs = be.cause_embs
    cause_embs_n = F.normalize(cause_embs, dim=-1)
    in_dim = be.in_dim

    nq = be.n_queries if max_queries < 0 else min(max_queries, be.n_queries)
    gtags = [f"g={g:.2f}" for g in gammas]
    sem_ranks = {t: [] for t in gtags}
    sem_cov = {t: [] for t in gtags}
    cl_ranks = {t: [] for t in gtags}
    cl_cov = {t: [] for t in gtags}
    ndcg = {t: [] for t in gtags}
    # Per-query (macro) P@n / R@n for the P/F1 triple. The sem_R@k columns stay
    # micro (per GT occurrence, eq. Recall@k); these are macro so F1@n reconciles
    # with its own P@n / R@n, matching ablations/retrieval_framework's convention.
    sem_prec = {t: {n: [] for n in PREC_NS} for t in gtags}
    sem_rec_q = {t: {n: [] for n in PREC_NS} for t in gtags}

    for qi in range(nq):
        gt, g_slot, text_emb, z_sel, w_sel = be.query(qi)
        if not gt:
            continue
        n_gt = len(gt)
        sims = (be.H[qi:qi + 1] @ be.bank_z.T)[0].cpu().numpy()
        top_idx, top_w, _ = select_positive_top_cases(sims, top_k_cases)
        cand = build_candidate_pool(top_idx, be.train_cases)
        if len(cand) == 0:
            for t in gtags:
                sem_ranks[t] += [MISS_RANK] * n_gt; sem_cov[t] += [0] * n_gt
                cl_ranks[t] += [MISS_RANK] * n_gt; cl_cov[t] += [0] * n_gt
                ndcg[t].append(0.0)
                for n in PREC_NS:
                    sem_prec[t][n].append(0.0); sem_rec_q[t][n].append(0.0)
            continue

        cand_t = torch.tensor(cand, device=device, dtype=torch.long)
        cand_embs_n = cause_embs_n.index_select(0, cand_t)
        s1 = score_candidates(cand, top_idx, top_w, be.train_cases, cause_embs)

        P = len(cand)
        K = z_sel.size(0)
        if text_emb is None:
            text_slot = torch.zeros(P, in_dim, device=device)
            text_present = torch.zeros(P, dtype=torch.bool, device=device)
        else:
            text_slot = text_emb.unsqueeze(0).expand(P, -1)
            text_present = torch.ones(P, dtype=torch.bool, device=device)
        lesion_mask = torch.full((P, K), not be.mask_lesions, dtype=torch.bool, device=device)
        lw = None if w_sel is None else w_sel.unsqueeze(0).expand(P, -1).contiguous()
        s_ceah, _, _ = be.ceah(
            g_slot.unsqueeze(0).expand(P, -1), text_slot, text_present,
            z_sel.unsqueeze(0).expand(P, -1, -1).contiguous(), lesion_mask,
            cause_embs.index_select(0, cand_t), lesion_weights=lw)

        s1n, scn = minmax(s1), minmax(s_ceah)
        gt_embs_n = cause_embs_n.index_select(0, torch.tensor(gt, device=device, dtype=torch.long))
        cand_global = np.array(cand)

        for t, g in zip(gtags, gammas):
            hybrid = g * s1n + (1.0 - g) * scn
            order = torch.argsort(hybrid, descending=True).cpu().numpy()
            sorted_embs_n = cand_embs_n[torch.from_numpy(order).to(device)]
            cos = gt_embs_n @ sorted_embs_n.T            # [n_gt, P]
            match = (cos >= sem_thresh).cpu().numpy()
            for gi in range(n_gt):
                hit = np.flatnonzero(match[gi])
                if hit.size:
                    sem_ranks[t].append(float(hit[0]) + 1.0); sem_cov[t].append(1)
                else:
                    sem_ranks[t].append(MISS_RANK); sem_cov[t].append(0)
            # candidate-level relevance for NDCG (any-GT sem match), in rank order
            rel = match.max(axis=0).astype(np.float64)
            ndcg[t].append(ndcg_at_k(rel, 5))
            for n in PREC_NS:
                mn = min(n, rel.size)
                sem_prec[t][n].append(float(rel[:mn].mean()) if mn else 0.0)
                sem_rec_q[t][n].append(float(match[:, :mn].any(axis=1).mean()))
            if cluster_id_array is not None:
                sorted_clusters = cluster_id_array[cand_global[order]]
                for gi in gt:
                    cid = int(cluster_id_array[int(gi)])
                    hits = np.flatnonzero(sorted_clusters == cid)
                    if hits.size:
                        cl_ranks[t].append(float(hits[0]) + 1.0); cl_cov[t].append(1)
                    else:
                        cl_ranks[t].append(MISS_RANK); cl_cov[t].append(0)

    out = {}
    for t in gtags:
        arr = np.asarray(sem_ranks[t], dtype=np.float64)
        m = summarize_rank_metric(arr, sem_cov[t])
        block = {"sem_MRR": m["MRR"], "sem_coverage": m["coverage"]}
        sem_m = {}
        add_recall_at_ks(sem_m, arr, [1, 3, 5, 10, 20])
        block.update({f"sem_{k}": v for k, v in sem_m.items()})
        block["NDCG@5"] = float(np.mean(ndcg[t])) if ndcg[t] else 0.0
        for n in PREC_NS:
            p = float(np.mean(sem_prec[t][n])) if sem_prec[t][n] else 0.0
            r = float(np.mean(sem_rec_q[t][n])) if sem_rec_q[t][n] else 0.0
            block[f"sem_P@{n}"] = p
            block[f"sem_Rq@{n}"] = r
            block[f"sem_F1@{n}"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        if cluster_id_array is not None:
            carr = np.asarray(cl_ranks[t], dtype=np.float64)
            cm = summarize_rank_metric(carr, cl_cov[t])
            cl_m = {}
            add_recall_at_ks(cl_m, carr, [1, 10, 20])
            block["cl_MRR"] = cm["MRR"]
            block.update({f"cl_{k}": v for k, v in cl_m.items()})
        out[t] = block
    return {"n_queries": nq, "metrics_per_gamma": out}


# ---------------------------------------------------------------------------
# Faithfulness (evidence masking)
# ---------------------------------------------------------------------------
@dataclass
class CeahBatch:
    """CEAM inputs for one query broadcast over its P candidate causes."""
    ceah: object
    g_e: torch.Tensor
    t_e: torch.Tensor
    t_p: torch.Tensor
    l_e: torch.Tensor
    l_m: torch.Tensor
    l_w: Optional[torch.Tensor]
    cand_embs: torch.Tensor

    def score(self, force_mask=None):
        return self.ceah(self.g_e, self.t_e, self.t_p, self.l_e, self.l_m,
                         self.cand_embs, force_mask=force_mask, lesion_weights=self.l_w)

    @property
    def n_lesions(self) -> int:
        return self.l_e.size(1)


def make_ceah_batch(ceah, g, z_sel, w_sel, cand_embs, device) -> CeahBatch:
    """Vision-only CEAM batch (text slot absent) — the faithfulness protocol."""
    P = cand_embs.size(0)
    K = z_sel.size(0)
    return CeahBatch(
        ceah=ceah,
        g_e=g.unsqueeze(0).expand(P, -1),
        t_e=torch.zeros(P, cand_embs.size(-1), device=device),
        t_p=torch.zeros(P, dtype=torch.bool, device=device),
        l_e=z_sel.unsqueeze(0).expand(P, -1, -1).contiguous(),
        l_m=torch.ones(P, K, dtype=torch.bool, device=device),
        l_w=w_sel.unsqueeze(0).expand(P, -1).contiguous(),
        cand_embs=cand_embs,
    )


@torch.no_grad()
def faithfulness_drops(batch: CeahBatch, device):
    """Score drops (baseline - masked) for the top-1 cause of `batch`.

    -> (top1, baseline, drops) with drops keyed no_global / no_lesion /
    no_top_α / no_random. `no_random` consumes exactly one np.random draw, last.
    """
    scores, alphas, ev_mask = batch.score()
    top1 = int(scores.argmax().item())
    baseline = float(scores[top1])
    base_alpha = alphas[top1].cpu().numpy()
    P, max_Ne = batch.cand_embs.size(0), ev_mask.size(1)

    def mask_score(positions):
        fm = torch.ones(P, max_Ne, dtype=torch.bool, device=device)
        for p in positions:
            if p < max_Ne:
                fm[:, p] = False
        s, _, _ = batch.score(force_mask=fm)
        return float(s[top1])

    lesion_pos = list(range(2, 2 + batch.n_lesions))
    s_no_global = mask_score([0])
    s_no_lesion = mask_score(lesion_pos)
    # top-α among valid positions (global + lesions; text absent)
    valid_pos = [0] + lesion_pos
    top_a = max(valid_pos, key=lambda p: base_alpha[p])
    s_no_top = mask_score([top_a])
    others = [p for p in valid_pos if p != top_a]
    s_rand = mask_score([int(np.random.choice(others))]) if others else baseline

    drops = {"no_global": baseline - s_no_global,
             "no_lesion": baseline - s_no_lesion,
             "no_top_α": baseline - s_no_top,
             "no_random": baseline - s_rand}
    return top1, baseline, drops
