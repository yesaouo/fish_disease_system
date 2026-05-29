"""Phase 1 baseline: zero-training case-based cause retrieval + candidate-restricted cause scoring.

Pipeline per query (valid case):
  1. Combined case similarity vs every train case
       sim(q, c) = α · cos(q.global, c.global)
                 + β · lesion-set cosine score
  2. Top-K retrieved cases, restricted to positive similarity only
  3. Build CANDIDATE POOL: union of deduped cause embedding indices
     appearing in those positive-similarity top-K retrieved cases.
  4. Score each candidate c in the pool using Option A:
       score(c) = Σ_{j in top-K-positive} w_j · max_g cos(c, e_{j,g})
     where w_j is normalized over the retained positive retrieved cases.
  5. Raw ranking is used for metrics.
  6. Diversified ranking is used only for predicted_top_n inspection output.

Metrics:
  - exact: rank of each GT cause within the raw candidate-pool ranking.
           Misses are represented internally as +inf for metrics and as null
           in per_query_results.jsonl, so R@K never counts uncovered GTs.
  - semantic (cosine ≥ threshold): rank of first candidate semantically
           equivalent to GT within the raw candidate-pool ranking
  - cluster (HDBSCAN): rank of first candidate in the same cluster as each GT
                         cause occurrence within the raw candidate-pool ranking
  - coverage_*: per-GT-cause-occurrence fraction of "GT covered by pool"
                under each match type

Notes:
  - Embeddings are explicitly L2-normalized before dot products are used as cosine.
  - Retrieved train cases with similarity <= 0 are excluded from both candidate-pool
    construction and candidate scoring.
  - Cluster metric is computed per GT cause occurrence, matching exact and
    semantic metric denominators. Duplicate GT causes/clusters in a query each
    contribute one occurrence.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

# Embedding fields that DDXPlus builds may store as fp16/bf16 to halve shard
# disk; downstream code (CEAH's fp32 nn.Linear, F.normalize, dot products) is
# fp32-native, so upcast at load time. No-op for legacy fp32 fish case_dbs.
_EMB_KEYS = ("global_emb", "text_colloquial_emb", "text_medical_emb", "lesion_embs")
_HALF_DTYPES = (torch.float16, torch.bfloat16)


def _upcast_case_dict(c: dict) -> None:
    for k in _EMB_KEYS:
        t = c.get(k)
        if isinstance(t, torch.Tensor) and t.dtype in _HALF_DTYPES:
            c[k] = t.float()


def _upcast_tensor(t: torch.Tensor) -> torch.Tensor:
    return t.float() if t.dtype in _HALF_DTYPES else t


def load_cases(
    case_db_dir: Path,
    split: str,
    max_cases: Optional[int] = None,
    sample_seed: int = 42,
) -> List[dict]:
    """Load all cases for a split, transparently handling sharded layouts.

    Sharded layout (DDXPlus large builds): meta.json lists ``{split}_shards``
    pointing to per-shard ``.pt`` files. Legacy layout: a single
    ``{split}_cases.pt`` next to ``meta.json``.

    ``max_cases`` enables uniform per-shard random subsampling (proportional
    fraction = ``max_cases / n_{split}_cases`` from meta.json). Sampling
    happens **before** :func:`_upcast_case_dict` so dropped cases never pay
    the bf16→fp32 cost. Same RNG construction (seeded by ``sample_seed``,
    called once per shard via ``np.random.default_rng(seed).choice``) as
    :func:`load_train_bank` — passing the same ``sample_seed`` to both
    produces identical case subsets in identical order, which keeps
    :class:`train_ceah.CEAHDataset` aligned with ``train_candidate_pool.pt``
    indices. For 200k DDXPlus subsample this drops peak CPU RAM from
    ~64 GB (fp32 full-train upcast) to ~5 GB (per-shard ~300 MB peak).

    ``None`` (default) loads everything — fish workflows are unchanged.
    """
    meta_path = case_db_dir / "meta.json"
    shards: List[str] = []
    n_split_meta: Optional[int] = None
    if meta_path.exists():
        meta = json.load(meta_path.open())
        shards = meta.get(f"{split}_shards") or []
        n_split_meta = meta.get(f"n_{split}_cases")
        if isinstance(n_split_meta, int) and n_split_meta <= 0:
            n_split_meta = None

    keep_frac: Optional[float] = None
    if (
        max_cases is not None
        and n_split_meta is not None
        and max_cases < n_split_meta
    ):
        keep_frac = float(max_cases) / float(n_split_meta)
    rng = np.random.default_rng(sample_seed) if keep_frac is not None else None

    def _maybe_subsample(cases: List[dict]) -> List[dict]:
        if keep_frac is None:
            return cases
        n_shard = len(cases)
        k = max(1, int(round(n_shard * keep_frac)))
        sel = rng.choice(n_shard, size=k, replace=False)
        sel.sort()
        return [cases[int(i)] for i in sel]

    if shards:
        out: List[dict] = []
        for shard in shards:
            cases = torch.load(case_db_dir / shard, weights_only=False)
            cases = _maybe_subsample(cases)
            for c in cases:
                _upcast_case_dict(c)
            out.extend(cases)
            del cases
            gc.collect()
        return out

    cases = torch.load(case_db_dir / f"{split}_cases.pt", weights_only=False)
    cases = _maybe_subsample(cases)
    for c in cases:
        _upcast_case_dict(c)
    return cases


@torch.no_grad()
def load_train_bank(
    case_db_dir: Path,
    device: str,
    keep_keys: Sequence[str] = ("cause_emb_indices", "causes", "pathology_emb_idx"),
    bank_dtype: Optional[torch.dtype] = None,
    max_cases: Optional[int] = None,
    sample_seed: int = 42,
) -> Tuple[List[dict], torch.Tensor, torch.Tensor, List[int]]:
    """Memory-efficient train-bank loader for Phase 1 / CEAH eval.

    Streams shards (or a single legacy file), stacks ``global_emb`` and
    ``lesion_embs`` on ``device`` (L2-normalized), and reduces each case dict
    to ``keep_keys``. Per-case fields outside ``keep_keys`` — including the
    source embedding tensors, ``text_*_emb`` duplicates, ``lesion_boxes_xywh``
    zeros, and verbose strings (``evidence_texts`` / ``global_text`` / ``ddx``)
    — are dropped so they don't pin CPU RAM during eval. Returns
    ``(minimal_cases, global_stack, lesion_stack, offsets)``; offsets match
    ``stack_train_lesions``.

    For a ~130k DDXPlus train bank this drops the resident train-list cost
    from ~5 GB of dicts to ~10–20 MB (stacks live on ``device``). Required
    fields for the existing scoring loop (``cause_emb_indices``, ``causes``)
    are the default; pass extra ``keep_keys`` if you also need ``image_id``
    or other debug fields in per-query output.

    ``bank_dtype`` controls the on-device storage dtype of the stacked
    embeddings. ``None`` (default) keeps the legacy behavior of upcasting
    half-precision storage (DDXPlus bf16/fp16) to fp32 — required for fish
    case_dbs whose downstream consumers assume fp32. Pass ``torch.bfloat16`` /
    ``torch.float16`` for DDXPlus banks where fp32 (~63 GB lesion stack at 1M
    cases) does not fit on a single 32 GB GPU. The bank is only used by
    :func:`compute_case_similarities` (cosine matmul, dtype-agnostic on
    modern GPUs) and by :func:`build_candidate_pool` / :func:`score_candidates`
    which read per-case ``cause_emb_indices`` int lists, not embedding tensors,
    so half-precision banks do not affect CEAH's fp32 path.

    ``max_cases`` caps the retained train-bank size via uniform random per-shard
    subsampling (proportional fraction = ``max_cases / n_train_cases`` from
    meta.json). At full 1M DDXPlus scale even bf16 (~31 GB) overflows a 32 GB
    GPU; ``max_cases=200000`` keeps the per-condition stratification intact
    (DDXPlus has 49 conditions, ~20k cases each — 200k leaves ~4k per
    condition) while halving bank VRAM. ``None`` keeps all cases. Sampling
    happens inside the shard-load loop so the dropped cases never reach the
    stack/cat path.
    """
    meta_path = case_db_dir / "meta.json"
    shards: List[str] = []
    n_train_meta: Optional[int] = None
    if meta_path.exists():
        meta = json.load(meta_path.open())
        shards = list(meta.get("train_shards") or [])
        n_train_meta = meta.get("n_train_cases")
        if isinstance(n_train_meta, int) and n_train_meta <= 0:
            n_train_meta = None
    sources = (
        [case_db_dir / s for s in shards]
        if shards else [case_db_dir / "train_cases.pt"]
    )

    keep_frac: Optional[float] = None
    if (
        max_cases is not None
        and n_train_meta is not None
        and max_cases < n_train_meta
    ):
        keep_frac = float(max_cases) / float(n_train_meta)
    rng = np.random.default_rng(sample_seed) if keep_frac is not None else None

    minimal_cases: List[dict] = []
    global_chunks: List[torch.Tensor] = []
    lesion_chunks: List[torch.Tensor] = []
    offsets: List[int] = [0]

    def _to_bank(t: torch.Tensor) -> torch.Tensor:
        if bank_dtype is not None:
            return t.to(device=device, dtype=bank_dtype, non_blocking=True)
        return _upcast_tensor(t).to(device, non_blocking=True)

    for src in sources:
        shard = torch.load(src, weights_only=False)
        if keep_frac is not None:
            n_shard = len(shard)
            k = max(1, int(round(n_shard * keep_frac)))
            sel = rng.choice(n_shard, size=k, replace=False)
            sel.sort()
            shard = [shard[int(i)] for i in sel]
        # Stack/cat on CPU, then move to device immediately so the CPU-side
        # copies are GC-able by next iteration. Without the eager .to(device),
        # chunks accumulate on CPU and the final torch.cat doubles peak RAM.
        g_chunk = _to_bank(torch.stack([c["global_emb"] for c in shard]))
        l_chunk = _to_bank(torch.cat([c["lesion_embs"] for c in shard], dim=0))
        for c in shard:
            offsets.append(offsets[-1] + int(c["lesion_embs"].size(0)))
            minimal_cases.append({k: c[k] for k in keep_keys if k in c})
        global_chunks.append(g_chunk)
        lesion_chunks.append(l_chunk)
        del shard, g_chunk, l_chunk
        gc.collect()

    global_stack = F.normalize(torch.cat(global_chunks, dim=0), dim=-1)
    del global_chunks
    lesion_stack = F.normalize(torch.cat(lesion_chunks, dim=0), dim=-1)
    del lesion_chunks
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()
    return minimal_cases, global_stack, lesion_stack, offsets


@torch.no_grad()
def load_train_cases_minimal(
    case_db_dir: Path,
    keep_keys: Sequence[str] = ("cause_emb_indices", "causes", "pathology_emb_idx"),
) -> List[dict]:
    """Load per-case metadata (``cause_emb_indices`` / ``causes``) without
    materializing any embedding tensor on GPU.

    Pairs with :func:`stream_top_k_cases`: streaming retrieval bypasses
    :func:`load_train_bank` (which won't fit when the lesion stack exceeds
    GPU VRAM), but downstream :func:`build_candidate_pool` /
    :func:`score_candidates` still need the per-case cause-index lists. This
    loader provides exactly that — for 1M DDXPlus cases it produces ~100 MB
    of CPU dicts, compared with ~30 GB of GPU embeddings from load_train_bank.
    """
    meta_path = case_db_dir / "meta.json"
    shards: List[str] = []
    if meta_path.exists():
        meta = json.load(meta_path.open())
        shards = list(meta.get("train_shards") or [])
    sources = (
        [case_db_dir / s for s in shards]
        if shards else [case_db_dir / "train_cases.pt"]
    )
    out: List[dict] = []
    for src in sources:
        shard = torch.load(src, weights_only=False)
        for c in shard:
            out.append({k: c[k] for k in keep_keys if k in c})
        del shard
        gc.collect()
    return out


@torch.no_grad()
def stream_top_k_cases(
    queries: List[dict],
    case_db_dir: Path,
    top_k_cases: int,
    alpha: float,
    beta: float,
    lesion_match: str = "max_mean",
    device: str = "cuda",
    bank_dtype: torch.dtype = torch.bfloat16,
    query_batch_size: int = 64,
    verbose: bool = False,
    max_cases: Optional[int] = None,
    sample_seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Shard-streaming top-K case retrieval for banks larger than GPU VRAM.

    Replaces the per-query :func:`compute_case_similarities` +
    :func:`select_positive_top_cases` flow when the full bank does not fit on
    a single GPU (the DDXPlus 1M-case bank is ~30 GB in bf16, exceeding a
    32 GB GPU after accounting for query / working buffers).

    Algorithm (outer shard, inner query batch — each shard loaded exactly once
    per call; total PCIe transfer = bank size in ``bank_dtype``):

      1. Pre-stack queries on GPU (``q_global``, ``q_lesion_stack``).
         ~5 GB for 132K DDXPlus valid queries in bf16.
      2. For each shard ``s``:
         a. Load shard global + lesion stacks to GPU (~2 GB).
         b. For each query batch of size ``query_batch_size``:
            - Compute ``α·g_sim + β·max_mean(lesion_sim)`` per
              (query, shard_case) — fully vectorized scatter_reduce.
            - Top-K-merge per query into a running ``[N_q, K]`` accumulator
              (sims + globally-numbered case indices).
         c. Free shard.
      3. Filter top-K to strictly positive sims per query (matches
         :func:`select_positive_top_cases` semantics) and normalize weights.

    Numerically equivalent to a monolithic-bank
    :func:`compute_case_similarities` call within bf16 precision (cosine
    matmul in bf16 is accurate to ~1e-3; ranking impact on top-K is
    negligible since case-similarity gaps are typically >> 0.01).

    Returns ``(top_k_idx_list, top_k_w_list, top_k_raw_w_list)`` with one
    entry per query; arrays may be shorter than ``top_k_cases`` if fewer
    positive-similarity cases were found.

    Only ``lesion_match='max_mean'`` is supported — Hungarian needs a
    per-pair scipy LP loop and is incompatible with batched scatter_reduce
    (and is anyway infeasible at 1M-case bank scale).

    ``max_cases`` / ``sample_seed`` mirror :func:`load_train_bank`'s
    subsampling for apples-to-apples equivalence tests (stream-vs-non-stream
    on the same retained subset). ``None`` (default) uses all cases.
    """
    if lesion_match != "max_mean":
        raise ValueError(
            f"stream_top_k_cases supports only lesion_match='max_mean'; "
            f"got {lesion_match!r}. The non-streaming path supports hungarian "
            f"but is only viable at <~10k case banks."
        )
    n_q = len(queries)
    if n_q == 0:
        return [], [], []

    # ---- Pre-stack queries on GPU -----------------------------------------
    q_global = F.normalize(
        torch.stack([q["global_emb"] for q in queries]).to(device, dtype=bank_dtype),
        dim=-1,
    )  # [N_q, D]
    q_lesion_chunks = [q["lesion_embs"] for q in queries]
    q_offsets_py: List[int] = [0]
    for chunk in q_lesion_chunks:
        q_offsets_py.append(q_offsets_py[-1] + int(chunk.size(0)))
    q_lesion_stack = F.normalize(
        torch.cat(q_lesion_chunks, dim=0).to(device, dtype=bank_dtype),
        dim=-1,
    )  # [Q_total_lesions, D]
    q_lesion_case_ids_full = offsets_to_case_ids(q_offsets_py, device)
    del q_lesion_chunks

    # ---- Running top-K accumulator ----------------------------------------
    running_sims = torch.full(
        (n_q, top_k_cases), float("-inf"), device=device, dtype=torch.float32,
    )
    running_idxs = torch.full(
        (n_q, top_k_cases), -1, device=device, dtype=torch.long,
    )

    # ---- Resolve shard sources --------------------------------------------
    meta_path = case_db_dir / "meta.json"
    meta = json.load(meta_path.open()) if meta_path.exists() else {}
    shards_list = list(meta.get("train_shards") or [])
    sources = (
        [case_db_dir / s for s in shards_list]
        if shards_list else [case_db_dir / "train_cases.pt"]
    )
    n_train_meta = meta.get("n_train_cases")
    if isinstance(n_train_meta, int) and n_train_meta <= 0:
        n_train_meta = None
    keep_frac: Optional[float] = None
    if (
        max_cases is not None
        and n_train_meta is not None
        and max_cases < n_train_meta
    ):
        keep_frac = float(max_cases) / float(n_train_meta)
    rng_sub = np.random.default_rng(sample_seed) if keep_frac is not None else None

    case_offset = 0
    t0 = time.time()
    for src_idx, src in enumerate(sources):
        shard = torch.load(src, weights_only=False)
        if keep_frac is not None:
            n_full = len(shard)
            k = max(1, int(round(n_full * keep_frac)))
            sel = rng_sub.choice(n_full, size=k, replace=False)
            sel.sort()
            shard = [shard[int(i)] for i in sel]
        n_shard = len(shard)
        shard_global = F.normalize(
            torch.stack([c["global_emb"] for c in shard]).to(device, dtype=bank_dtype),
            dim=-1,
        )
        shard_lesion_list = [c["lesion_embs"] for c in shard]
        shard_offsets_py: List[int] = [0]
        for chunk in shard_lesion_list:
            shard_offsets_py.append(shard_offsets_py[-1] + int(chunk.size(0)))
        shard_lesion = F.normalize(
            torch.cat(shard_lesion_list, dim=0).to(device, dtype=bank_dtype),
            dim=-1,
        )
        m_shard = shard_lesion.size(0)
        shard_case_ids = offsets_to_case_ids(shard_offsets_py, device)
        del shard, shard_lesion_list
        gc.collect()

        if verbose:
            elapsed = time.time() - t0
            print(
                f"[stream] shard {src_idx+1}/{len(sources)} loaded "
                f"n_shard={n_shard} m_shard={m_shard} elapsed={elapsed:.1f}s"
            )

        for qs in range(0, n_q, query_batch_size):
            qe = min(qs + query_batch_size, n_q)
            B = qe - qs
            q_les_start = q_offsets_py[qs]
            q_les_end = q_offsets_py[qe]

            # g_sim: [B, n_shard], fp32 for accumulation
            g_sim = (q_global[qs:qe] @ shard_global.T).float()

            q_les_subset = q_lesion_stack[q_les_start:q_les_end]
            q_local_ids = q_lesion_case_ids_full[q_les_start:q_les_end] - qs
            B_les_total = q_les_subset.size(0)

            if B_les_total == 0 or m_shard == 0:
                les_score = torch.zeros((B, n_shard), device=device, dtype=torch.float32)
            else:
                # les_sim [B_les_total, m_shard] in bank dtype — the largest
                # transient buffer. Sized by query_batch_size to fit VRAM.
                les_sim = q_les_subset @ shard_lesion.T

                # Step 1: max over each train-case's lesions for each query-lesion row.
                #   t_max_per_row[i, t] = max_{j in t_case lesions} les_sim[i, j]
                t_max_per_row = torch.full(
                    (B_les_total, n_shard), float("-inf"),
                    device=device, dtype=les_sim.dtype,
                )
                shard_case_ids_exp = shard_case_ids.unsqueeze(0).expand(B_les_total, -1)
                t_max_per_row.scatter_reduce_(
                    1, shard_case_ids_exp, les_sim,
                    reduce="amax", include_self=False,
                )
                t_max_per_row = torch.where(
                    t_max_per_row == float("-inf"),
                    torch.zeros_like(t_max_per_row),
                    t_max_per_row,
                )

                # Step 2a (forward direction): mean over q's lesions of t_max_per_row,
                # grouped by q_case (q_local_ids).
                forward_sum = torch.zeros(
                    (B, n_shard), device=device, dtype=torch.float32,
                )
                q_local_exp_T = q_local_ids.unsqueeze(-1).expand(-1, n_shard)
                forward_sum.scatter_add_(0, q_local_exp_T, t_max_per_row.float())
                q_counts = torch.zeros(B, device=device, dtype=torch.float32)
                q_counts.scatter_add_(
                    0, q_local_ids,
                    torch.ones_like(q_local_ids, dtype=torch.float32),
                )
                forward_mean = forward_sum / q_counts.clamp_min(1.0).unsqueeze(-1)

                # Step 2b (backward direction): max over q's lesions per t_lesion,
                # then mean over t's lesions per t_case.
                q_max_per_col = torch.full(
                    (B, m_shard), float("-inf"),
                    device=device, dtype=les_sim.dtype,
                )
                q_local_exp_M = q_local_ids.unsqueeze(-1).expand(-1, m_shard)
                q_max_per_col.scatter_reduce_(
                    0, q_local_exp_M, les_sim,
                    reduce="amax", include_self=False,
                )
                q_max_per_col = torch.where(
                    q_max_per_col == float("-inf"),
                    torch.zeros_like(q_max_per_col),
                    q_max_per_col,
                )
                backward_sum = torch.zeros(
                    (B, n_shard), device=device, dtype=torch.float32,
                )
                shard_case_ids_exp_B = shard_case_ids.unsqueeze(0).expand(B, -1)
                backward_sum.scatter_add_(
                    1, shard_case_ids_exp_B, q_max_per_col.float(),
                )
                t_counts = torch.zeros(n_shard, device=device, dtype=torch.float32)
                t_counts.scatter_add_(
                    0, shard_case_ids,
                    torch.ones_like(shard_case_ids, dtype=torch.float32),
                )
                backward_mean = backward_sum / t_counts.clamp_min(1.0).unsqueeze(0)

                les_score = 0.5 * (forward_mean + backward_mean)
                del les_sim, t_max_per_row, forward_sum, q_counts, forward_mean
                del q_max_per_col, backward_sum, t_counts, backward_mean

            sim_block = alpha * g_sim + beta * les_score  # [B, n_shard]
            del g_sim, les_score

            # Top-K-within-shard, then merge with running.
            top_k_in_shard = min(top_k_cases, n_shard)
            shard_top_sims, shard_top_local = sim_block.topk(top_k_in_shard, dim=1)
            shard_top_global = shard_top_local + case_offset
            del sim_block, shard_top_local

            combined_sims = torch.cat([running_sims[qs:qe], shard_top_sims], dim=1)
            combined_idxs = torch.cat([running_idxs[qs:qe], shard_top_global], dim=1)
            top_sims, top_pos = combined_sims.topk(top_k_cases, dim=1)
            top_idxs = torch.gather(combined_idxs, 1, top_pos)
            running_sims[qs:qe] = top_sims
            running_idxs[qs:qe] = top_idxs
            del shard_top_sims, shard_top_global, combined_sims, combined_idxs
            del top_sims, top_pos, top_idxs

        case_offset += n_shard
        del shard_global, shard_lesion, shard_case_ids
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

    # ---- Extract per-query results (positive-similarity filter) ----------
    sims_np = running_sims.detach().cpu().numpy()
    idxs_np = running_idxs.detach().cpu().numpy()
    top_k_idx_list: List[np.ndarray] = []
    top_k_w_list: List[np.ndarray] = []
    top_k_raw_w_list: List[np.ndarray] = []
    for qi in range(n_q):
        sims = sims_np[qi]
        idxs = idxs_np[qi]
        # topk preserves descending order — keep it.
        mask = (sims > 0) & (idxs >= 0)
        sims_kept = sims[mask].astype(np.float32)
        idxs_kept = idxs[mask].astype(np.int64)
        if sims_kept.size > 0:
            w = sims_kept / (sims_kept.sum() + 1e-8)
        else:
            w = np.empty(0, dtype=np.float32)
        top_k_idx_list.append(idxs_kept)
        top_k_w_list.append(w.astype(np.float32))
        top_k_raw_w_list.append(sims_kept)
    return top_k_idx_list, top_k_w_list, top_k_raw_w_list


def load_case_db(case_db_dir: Path, query_split: str = "valid"):
    """Load train + query split cases. `query_split` selects which `<split>_cases.pt`
    is returned in the 2nd slot (default `valid` preserves prior behavior)."""
    if query_split not in ("valid", "test"):
        raise ValueError(f"query_split must be 'valid' or 'test', got {query_split!r}")
    train = load_cases(case_db_dir, "train")
    queries = load_cases(case_db_dir, query_split)
    cause = torch.load(case_db_dir / "cause_text_embs.pt", weights_only=False)
    if isinstance(cause, dict) and isinstance(cause.get("embeddings"), torch.Tensor):
        cause["embeddings"] = _upcast_tensor(cause["embeddings"])
    meta = json.load((case_db_dir / "meta.json").open())
    return train, queries, cause, meta


def stack_train_lesions(train_cases) -> Tuple[torch.Tensor, List[int]]:
    """Concatenate all train lesion embs; return (stacked, per-case offsets)."""
    pieces = [c["lesion_embs"] for c in train_cases]
    offsets = [0]
    for p in pieces:
        offsets.append(offsets[-1] + p.size(0))
    return torch.cat(pieces, dim=0), offsets


# ---------------------------------------------------------------------------
# Case similarity
# ---------------------------------------------------------------------------

def hungarian_set_score(sim_block: np.ndarray) -> float:
    """sim_block: [N_q, M_c] cosine matrix → mean of matched cosines / max(N_q, M_c).

    Empty sides return 0. The /max(N,M) normalizer penalizes lesion-count mismatch.
    """
    n, m = sim_block.shape
    if n == 0 or m == 0:
        return 0.0
    if n == 1 or m == 1:
        return float(sim_block.max() / max(n, m))
    row, col = linear_sum_assignment(-sim_block)
    return float(sim_block[row, col].sum() / max(n, m))


def max_mean_set_score(sim_block: np.ndarray) -> float:
    """Symmetric max-mean: 0.5 * (mean_i max_j sim_ij + mean_j max_i sim_ij).

    Differentiable analog of Hungarian (used at training time). Not penalized by
    size mismatch — slight asymmetry vs Hungarian.
    """
    n, m = sim_block.shape
    if n == 0 or m == 0:
        return 0.0
    forward = float(sim_block.max(axis=1).mean())
    backward = float(sim_block.max(axis=0).mean())
    return 0.5 * (forward + backward)


def max_mean_normalized_set_score(sim_block: np.ndarray) -> float:
    """Symmetric max-mean with the same /max(N,M) penalty as Hungarian.

    Isolates the aggregation operator (one-to-one Hungarian assignment vs
    bidirectional soft MaxSim) from the size-mismatch normalization that
    `hungarian_set_score` applies. Used for the aggregation ablation.
    """
    n, m = sim_block.shape
    if n == 0 or m == 0:
        return 0.0
    forward = float(sim_block.max(axis=1).sum())
    backward = float(sim_block.max(axis=0).sum())
    return 0.5 * (forward + backward) / max(n, m)


_LESION_MATCH_FNS = {
    "hungarian":           hungarian_set_score,
    "max_mean":            max_mean_set_score,
    "max_mean_normalized": max_mean_normalized_set_score,
}

_BATCHED_LESION_MATCHES = {"max_mean", "max_mean_normalized"}


def offsets_to_case_ids(train_offsets: Sequence[int], device) -> torch.Tensor:
    """Expand per-case offsets into a ``[total_lesions]`` long tensor mapping
    each lesion/evidence row to its train-case index. Pre-compute once and
    reuse across queries in tight loops (``build_train_candidate_pool`` /
    eval scripts) to skip the ~10 ms/query setup cost."""
    n_train = len(train_offsets) - 1
    lengths = torch.tensor(
        [int(train_offsets[i + 1] - train_offsets[i]) for i in range(n_train)],
        device=device, dtype=torch.long,
    )
    return torch.arange(n_train, device=device, dtype=torch.long).repeat_interleave(lengths)


@torch.no_grad()
def _batched_max_mean(
    les_sim: torch.Tensor,    # [N_q, L]
    case_ids: torch.Tensor,   # [L] long
    n_train: int,
    normalized: bool,
) -> torch.Tensor:
    """Fully vectorized GPU implementation of ``max_mean_set_score`` (and the
    ``max_mean_normalized`` variant) across all train cases simultaneously.

    Replaces the per-case Python loop in ``compute_case_similarities`` —
    critical for 1M-scale banks where the loop alone is days/weeks.
    """
    N_q, L = les_sim.shape
    device = les_sim.device
    dtype = les_sim.dtype
    neg_inf = float("-inf")

    # Forward direction: max over evidences belonging to each case, per query.
    forward_max = les_sim.new_full((N_q, n_train), neg_inf)
    case_ids_exp = case_ids.unsqueeze(0).expand(N_q, -1)
    forward_max.scatter_reduce_(1, case_ids_exp, les_sim, reduce="amax", include_self=False)
    forward_max = torch.where(
        forward_max == neg_inf, torch.zeros_like(forward_max), forward_max,
    )
    forward_sums = forward_max.sum(dim=0)  # [n_train], sum over queries

    # Backward direction: max over queries per evidence, then aggregate per case.
    evidence_max = les_sim.amax(dim=0)     # [L]
    backward_sums = torch.zeros(n_train, device=device, dtype=dtype)
    backward_sums.scatter_add_(0, case_ids, evidence_max)

    counts = torch.zeros(n_train, device=device, dtype=dtype)
    counts.scatter_add_(0, case_ids, torch.ones(L, device=device, dtype=dtype))
    valid = counts > 0

    if normalized:
        max_ne = torch.maximum(counts, counts.new_full((n_train,), float(N_q))).clamp_min(1.0)
        score = 0.5 * (forward_sums + backward_sums) / max_ne
    else:
        forward_mean = forward_sums / float(N_q)
        backward_mean = backward_sums / counts.clamp_min(1.0)
        score = 0.5 * (forward_mean + backward_mean)
    return torch.where(valid, score, torch.zeros_like(score))


def compute_case_similarities(
    q_global: torch.Tensor,           # [D], already normalized
    q_lesions: torch.Tensor,          # [N_q, D], already normalized
    train_global_stack: torch.Tensor, # [n_train, D], already normalized
    train_lesion_stack: torch.Tensor, # [total_lesions, D], already normalized
    train_offsets: Sequence[int],
    alpha: float,
    beta: float,
    lesion_match: str = "max_mean",
    train_case_ids: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Return [n_train] combined similarity scores.

    All embeddings are L2-normalized before this is called, so matmul =
    cosine. For ``lesion_match in {max_mean, max_mean_normalized}`` the lesion
    score is computed fully vectorized on the lesion stack's device (the
    1M-case-bank fast path). ``hungarian`` falls back to the per-case Python
    loop because ``scipy.linear_sum_assignment`` cannot be vectorized.

    Pass ``train_case_ids`` (from :func:`offsets_to_case_ids`) when calling in
    a tight loop to avoid recomputing the mapping per query.
    """
    # Cast queries to the bank's dtype so half-precision banks (DDXPlus bf16/
    # fp16, used to fit the 1M-case lesion stack in 32 GB VRAM) match the
    # matmul operand dtype. fp32 banks (fish) are a no-op cast.
    bank_dtype = train_global_stack.dtype
    if q_global.dtype != bank_dtype:
        q_global = q_global.to(bank_dtype)
    if q_lesions.dtype != bank_dtype:
        q_lesions = q_lesions.to(bank_dtype)

    g_sim = (q_global.unsqueeze(0) @ train_global_stack.T).squeeze(0).float()
    n_train = len(train_offsets) - 1

    if q_lesions.size(0) == 0 or train_lesion_stack.size(0) == 0:
        return (alpha * g_sim.detach().cpu().numpy()
                + np.zeros(n_train, dtype=np.float32))

    if lesion_match in _BATCHED_LESION_MATCHES:
        if train_case_ids is None:
            train_case_ids = offsets_to_case_ids(train_offsets, train_lesion_stack.device)
        les_sim = q_lesions @ train_lesion_stack.T
        l_score_t = _batched_max_mean(
            les_sim, train_case_ids, n_train,
            normalized=(lesion_match == "max_mean_normalized"),
        )
        del les_sim
        l_score = l_score_t.float().detach().cpu().numpy()
        return alpha * g_sim.detach().cpu().numpy() + beta * l_score

    # Hungarian fallback (per-case Python loop). Only viable at <~10k cases.
    match_fn = _LESION_MATCH_FNS[lesion_match]
    les_sim = (q_lesions @ train_lesion_stack.T).float().detach().cpu().numpy()
    l_score = np.zeros(n_train, dtype=np.float32)
    for i in range(n_train):
        s, e = train_offsets[i], train_offsets[i + 1]
        if e > s:
            l_score[i] = match_fn(les_sim[:, s:e])
    return alpha * g_sim.detach().cpu().numpy() + beta * l_score


# ---------------------------------------------------------------------------
# Candidate-restricted cause scoring
# ---------------------------------------------------------------------------

def select_positive_top_cases(
    sims: np.ndarray,
    top_k_cases: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select top-K train cases with strictly positive similarity.

    Returns:
      top_k_idx: selected train-case indices
      top_k_w: normalized positive weights for scoring
      top_k_raw_w: raw positive similarities, kept for inspection output
    """
    ranked_idx = np.argsort(-sims)
    positive_idx = ranked_idx[sims[ranked_idx] > 0]
    top_k_idx = positive_idx[:top_k_cases]

    top_k_raw_w = sims[top_k_idx].astype(np.float32)
    if top_k_raw_w.size > 0:
        top_k_w = top_k_raw_w / (top_k_raw_w.sum() + 1e-8)
    else:
        top_k_w = np.empty(0, dtype=np.float32)

    return top_k_idx, top_k_w.astype(np.float32), top_k_raw_w


def build_candidate_pool(
    top_case_idx: np.ndarray,
    train_cases: list,
) -> List[int]:
    """Unique cause-table indices appearing in the retained retrieved cases.

    The order preserves first-seen order across retrieved cases.
    """
    seen: set = set()
    pool: List[int] = []
    for case_i in top_case_idx.tolist():
        for cidx in train_cases[int(case_i)]["cause_emb_indices"]:
            if cidx not in seen:
                seen.add(cidx)
                pool.append(int(cidx))
    return pool


def diversify(
    sorted_local: np.ndarray,    # local indices into candidate pool, sorted by score desc
    cand_embs: torch.Tensor,     # [pool_size, D]
    threshold: float,
) -> np.ndarray:
    """Greedy MMR-style dedup for output inspection only.

    Keep the highest-scored candidate, then suppress any later candidate whose
    cosine to any already-kept candidate is >= threshold.

    Returns kept local indices in score order. If threshold >= 1.0, returns
    sorted_local unchanged.
    """
    if sorted_local.size == 0 or threshold >= 1.0:
        return sorted_local

    kept_local: List[int] = [int(sorted_local[0])]
    kept_emb_rows: List[int] = [int(sorted_local[0])]

    for li in sorted_local[1:].tolist():
        e = cand_embs[int(li)]
        kept_t = cand_embs[torch.tensor(
            kept_emb_rows,
            device=cand_embs.device,
            dtype=torch.long,
        )]
        max_sim = (kept_t @ e).max().item()
        if max_sim < threshold:
            kept_local.append(int(li))
            kept_emb_rows.append(int(li))

    return np.array(kept_local, dtype=np.int64)


def score_candidates(
    candidate_indices: List[int],
    top_case_idx: np.ndarray,
    top_case_weights: np.ndarray,
    train_cases: list,
    cause_table_embs: torch.Tensor,
) -> torch.Tensor:
    """For each c in candidate_indices:

       score(c) = Σ_j w_j · max_g cos(emb[c], emb_{j,g})

    top_case_weights are expected to be non-negative and normalized.
    Returns [len(candidate_indices)] tensor on cause_table_embs' device.
    """
    device = cause_table_embs.device
    if not candidate_indices:
        return torch.zeros(0, device=device)

    cand_embs = cause_table_embs.index_select(
        0,
        torch.tensor(candidate_indices, device=device, dtype=torch.long),
    )  # [U_cand, D]

    score = torch.zeros(len(candidate_indices), device=device)

    for case_i, w in zip(top_case_idx.tolist(), top_case_weights.tolist()):
        case_cidx = train_cases[int(case_i)]["cause_emb_indices"]
        if not case_cidx:
            continue

        case_embs = cause_table_embs.index_select(
            0,
            torch.tensor(case_cidx, device=device, dtype=torch.long),
        )  # [G, D]

        sims = cand_embs @ case_embs.T  # [U_cand, G]
        score = score + float(w) * sims.max(dim=1).values

    return score


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

MISS_RANK = float("inf")


def summarize_rank_metric(
    ranks: np.ndarray,
    coverage_flags: List[int],
) -> dict:
    """Summarize occurrence-level ranking metrics with correct miss handling.

    Uncovered GT occurrences must not be encoded as pool_size + 1, because that
    makes R@K depend on candidate-pool size and can incorrectly count misses as
    hits for large K. Internally, misses are +inf; therefore:
      - R@K counts only finite ranks <= K.
      - MRR gives misses reciprocal rank 0.
      - mean/median rank are reported over covered occurrences only.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    finite_mask = np.isfinite(ranks)
    finite_ranks = ranks[finite_mask]

    reciprocal = np.zeros_like(ranks, dtype=np.float64)
    reciprocal[finite_mask] = 1.0 / finite_ranks

    return {
        "coverage": float(np.mean(coverage_flags)) if coverage_flags else 0.0,
        "MRR": float(reciprocal.mean()) if ranks.size else 0.0,
        "mean_rank": float(finite_ranks.mean()) if finite_ranks.size else None,
        "median_rank": float(np.median(finite_ranks)) if finite_ranks.size else None,
        "n_covered": int(finite_ranks.size),
        "n_missed": int(ranks.size - finite_ranks.size),
        "rank_meaning": (
            "mean_rank and median_rank are computed over covered occurrences only; "
            "misses contribute 0 to MRR and R@K."
        ),
    }


def add_recall_at_ks(metric_block: dict, ranks: np.ndarray, ks: List[int]) -> None:
    """Add R@K values. Misses are +inf, so they never satisfy rank <= K."""
    ranks = np.asarray(ranks, dtype=np.float64)
    for k in ks:
        metric_block[f"R@{k}"] = float((ranks <= k).mean()) if ranks.size else 0.0


def fmt_metric(value) -> str:
    """Pretty-print scalar metric values, including None for unavailable ranks."""
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--query_split", type=str, default="valid",
                    choices=["valid", "test"],
                    help="which split to use as the query set (default valid)")
    ap.add_argument("--top_k_cases", type=int, default=10)
    ap.add_argument("--top_n_causes", type=int, default=20)
    ap.add_argument("--alpha_global", type=float, default=0.25)
    ap.add_argument("--beta_lesion", type=float, default=0.75)
    ap.add_argument("--lesion_match", type=str, default="max_mean",
                    choices=["hungarian", "max_mean", "max_mean_normalized"],
                    help="lesion-set matching mode for case similarity")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_queries", type=int, default=-1)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100])
    ap.add_argument("--semantic_threshold", type=float, default=0.95,
                    help="cosine threshold for treating two cause strings as semantically equivalent")
    ap.add_argument("--cluster_json", type=str,
                    default="diagnosis_model/cause_inference/outputs/cause_clusters_llm.json",
                    help="Cluster taxonomy JSON (raw_string -> cluster_id). "
                         "Paper main: cause_clusters_llm.json (LLM, 466 topics). "
                         "Paper baseline: cause_clusters_hdbscan.json (HDBSCAN, 100). "
                         "Set to '' to disable cluster-level metrics.")
    ap.add_argument("--diversify_threshold", type=float, default=0.95,
                    help="For predicted_top_n output only: suppress any candidate whose cosine "
                         "to a previously kept candidate exceeds this. "
                         "Set to 1.0 to disable diversification.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    train_cases, valid_cases, cause_pack, meta = load_case_db(
        Path(args.case_db_dir), query_split=args.query_split,
    )

    # Explicit normalization: all following dot products are cosine similarities.
    cause_table_embs = F.normalize(cause_pack["embeddings"].to(device), dim=-1)
    cause_texts = cause_pack["texts"]

    print(f"[load] train={len(train_cases)}  {args.query_split}={len(valid_cases)}  "
          f"unique_causes={len(cause_texts)}  dim={cause_table_embs.size(-1)}")

    # Optional: cluster-based evaluation.
    # Cluster metrics are computed per GT cause occurrence, matching exact/semantic.
    cluster_id_array: np.ndarray | None = None
    if args.cluster_json:
        with open(args.cluster_json, encoding="utf-8") as f:
            cl = json.load(f)
        o2c = cl["original_to_cause_id"]
        cluster_id_array = np.array(
            [int(o2c[t]) for t in cause_texts], dtype=np.int64,
        )
        n_clusters = len(set(cluster_id_array.tolist()))
        print(f"[cluster] loaded {n_clusters} clusters from {args.cluster_json}")

    train_global_stack = torch.stack([c["global_emb"] for c in train_cases]).to(device)
    train_global_stack = F.normalize(train_global_stack, dim=-1)

    train_lesion_stack, train_offsets = stack_train_lesions(train_cases)
    train_lesion_stack = F.normalize(train_lesion_stack.to(device), dim=-1)

    print(f"[stack] train_globals={tuple(train_global_stack.shape)}  "
          f"train_lesions={tuple(train_lesion_stack.shape)}")

    queries = valid_cases if args.max_queries <= 0 else valid_cases[: args.max_queries]
    print(f"[eval] queries={len(queries)}  K={args.top_k_cases}  N={args.top_n_causes}  "
          f"alpha={args.alpha_global}  beta={args.beta_lesion}  "
          f"positive_case_only=True")

    per_query_results: list = []
    pool_sizes: List[int] = []
    retained_case_counts: List[int] = []

    all_gt_ranks: List[float] = []          # exact-index rank; MISS_RANK for uncovered GT
    all_gt_sem_ranks: List[float] = []      # semantic-cosine rank; MISS_RANK for uncovered GT
    all_gt_cluster_ranks: List[float] = []  # cluster-level rank; MISS_RANK for uncovered GT

    cov_exact: List[int] = []             # 1 if GT exactly in raw candidate pool
    cov_semantic: List[int] = []          # 1 if any raw pool cand >= threshold to GT
    cov_cluster: List[int] = []           # 1 if any raw pool cand in same cluster

    all_top1_max_cos: List[float] = []    # max cos(raw top-1 pred, any GT) per query

    t0 = time.time()
    for qi, q in enumerate(queries):
        q_global = F.normalize(q["global_emb"].to(device), dim=-1)
        q_lesions = F.normalize(q["lesion_embs"].to(device), dim=-1)

        sims = compute_case_similarities(
            q_global, q_lesions,
            train_global_stack, train_lesion_stack, train_offsets,
            args.alpha_global, args.beta_lesion,
            lesion_match=args.lesion_match,
        )

        # Only positive-similarity retrieved cases are allowed into candidate generation/scoring.
        top_k_idx, top_k_w, top_k_raw_w = select_positive_top_cases(
            sims=sims,
            top_k_cases=args.top_k_cases,
        )
        retained_case_counts.append(int(len(top_k_idx)))

        # Build candidate pool from the retained positive-similarity cases only.
        candidate_indices = build_candidate_pool(top_k_idx, train_cases)
        pool_size = len(candidate_indices)
        pool_sizes.append(pool_size)

        # Score the raw candidate pool only.
        cand_scores = score_candidates(
            candidate_indices, top_k_idx, top_k_w, train_cases, cause_table_embs,
        )  # [pool_size]

        if pool_size == 0:
            raw_sorted_local = np.empty(0, dtype=np.int64)
            raw_sorted_global = np.empty(0, dtype=np.int64)
            div_sorted_local = raw_sorted_local
            div_sorted_global = raw_sorted_global
            cand_embs = torch.empty(0, cause_table_embs.size(-1), device=device)
        else:
            cand_embs = cause_table_embs.index_select(
                0,
                torch.tensor(candidate_indices, device=device, dtype=torch.long),
            )

            # Raw sorted ranking: this is the ranking used for all metrics.
            raw_sorted_local = torch.argsort(cand_scores, descending=True).detach().cpu().numpy()
            raw_sorted_global = np.array(candidate_indices)[raw_sorted_local]

            # Diversified ranking: output inspection only, not used for metrics.
            div_sorted_local = diversify(raw_sorted_local, cand_embs, args.diversify_threshold)
            div_sorted_global = np.array(candidate_indices)[div_sorted_local]

        gt_cause_idx = q["cause_emb_indices"]

        # ---- Exact-index rank within raw pool ranking ----
        # Store misses as null in per_query_results and as +inf internally.
        # This prevents R@K from counting uncovered GTs as hits when K exceeds
        # the candidate-pool size.
        global_to_pool_pos = {int(g): i for i, g in enumerate(raw_sorted_global.tolist())}
        gt_ranks_local: List[int | None] = []
        for g in gt_cause_idx:
            pos = global_to_pool_pos.get(int(g))
            if pos is None:
                gt_ranks_local.append(None)
                all_gt_ranks.append(MISS_RANK)
                cov_exact.append(0)
            else:
                rank = int(pos) + 1
                gt_ranks_local.append(rank)
                all_gt_ranks.append(float(rank))
                cov_exact.append(1)

        # ---- Semantic-cosine rank within raw pool ranking ----
        gt_sem_ranks_local: List[int | None] = []
        if pool_size == 0:
            for _ in gt_cause_idx:
                gt_sem_ranks_local.append(None)
                all_gt_sem_ranks.append(MISS_RANK)
                cov_semantic.append(0)
        else:
            gt_idx_t = torch.tensor(gt_cause_idx, device=device, dtype=torch.long)
            gt_embs = cause_table_embs.index_select(0, gt_idx_t)  # [G, D]

            raw_sorted_cand_embs = cand_embs[
                torch.from_numpy(raw_sorted_local).to(device)
            ]  # [pool_size, D]

            cos_sorted = gt_embs @ raw_sorted_cand_embs.T  # [G, pool_size]
            sem_match = cos_sorted >= args.semantic_threshold

            for g_i in range(sem_match.size(0)):
                hits = torch.nonzero(sem_match[g_i], as_tuple=False)
                if hits.numel() > 0:
                    rank = int(hits[0].item()) + 1
                    gt_sem_ranks_local.append(rank)
                    all_gt_sem_ranks.append(float(rank))
                    cov_semantic.append(1)
                else:
                    gt_sem_ranks_local.append(None)
                    all_gt_sem_ranks.append(MISS_RANK)
                    cov_semantic.append(0)

        # ---- Cluster rank within raw pool ranking ----
        # Per GT cause occurrence, matching exact-index and semantic-cosine metrics.
        # If multiple GT causes map to the same cluster, each occurrence contributes
        # one rank/coverage item. This intentionally does NOT deduplicate clusters
        # within a query.
        gt_cluster_ranks_local: List[int | None] = []
        gt_clusters_local: List[int] = []
        if cluster_id_array is not None:
            raw_sorted_clusters = cluster_id_array[raw_sorted_global] if pool_size > 0 \
                                  else np.empty(0, dtype=np.int64)
            for g in gt_cause_idx:
                cid = int(cluster_id_array[int(g)])
                hits = np.flatnonzero(raw_sorted_clusters == cid) if pool_size > 0 \
                       else np.empty(0, dtype=np.int64)
                if hits.size > 0:
                    rank = int(hits[0]) + 1
                    gt_cluster_ranks_local.append(rank)
                    all_gt_cluster_ranks.append(float(rank))
                    cov_cluster.append(1)
                else:
                    gt_cluster_ranks_local.append(None)
                    all_gt_cluster_ranks.append(MISS_RANK)
                    cov_cluster.append(0)
                gt_clusters_local.append(cid)

        # ---- Raw top-1 max-cos diagnostic ----
        if pool_size > 0:
            top1_global = int(raw_sorted_global[0])
            gt_idx_t = torch.tensor(gt_cause_idx, device=device, dtype=torch.long)
            gt_embs = cause_table_embs.index_select(0, gt_idx_t)
            top1_cos = float((gt_embs @ cause_table_embs[top1_global]).max().item())
        else:
            top1_cos = 0.0
        all_top1_max_cos.append(top1_cos)

        # ---- Top-N predictions for inspection: diversified output ranking ----
        top_n_count = min(args.top_n_causes, len(div_sorted_global))
        top_n_global = div_sorted_global[:top_n_count].tolist()
        top_n_scores = (
            cand_scores[torch.from_numpy(div_sorted_local[:top_n_count]).to(device)]
            .detach().cpu().tolist()
        ) if top_n_count > 0 else []
        top_n_texts = [cause_texts[int(i)] for i in top_n_global]

        per_query_results.append({
            "query_image_id": int(q["image_id"]),
            "query_file_name": q["file_name"],
            "query_lesion_count": int(q["lesion_embs"].size(0)),
            "retained_positive_case_count": int(len(top_k_idx)),
            "candidate_pool_size": pool_size,
            "gt_causes": list(q["causes"]),
            "gt_cause_indices": list(gt_cause_idx),
            "gt_ranks_in_pool": gt_ranks_local,
            "gt_semantic_ranks_in_pool": gt_sem_ranks_local,
            "gt_clusters": gt_clusters_local,
            "gt_cluster_ranks_in_pool": gt_cluster_ranks_local,
            "top1_max_cos_to_gt": top1_cos,
            "retrieved_cases": [
                {
                    "case_id": int(top_k_idx[ki]),
                    "image_id": int(train_cases[int(top_k_idx[ki])]["image_id"]),
                    "similarity_raw": float(top_k_raw_w[ki]),
                    "similarity_weight_normalized": float(top_k_w[ki]),
                    "causes": list(train_cases[int(top_k_idx[ki])]["causes"]),
                }
                for ki in range(len(top_k_idx))
            ],
            "predicted_top_n": [
                {"cause_table_idx": int(i), "score": float(s), "text": t}
                for i, s, t in zip(top_n_global, top_n_scores, top_n_texts)
            ],
        })

        if (qi + 1) % 50 == 0 or qi + 1 == len(queries):
            elapsed = time.time() - t0
            rate = (qi + 1) / elapsed
            eta = (len(queries) - qi - 1) / max(rate, 1e-9)
            print(f"[eval] {qi+1}/{len(queries)}  "
                  f"rate={rate:.2f} q/s  ETA={eta/60:.1f} min")

    # Aggregate metrics
    ranks = np.array(all_gt_ranks, dtype=np.float64)
    sem_ranks = np.array(all_gt_sem_ranks, dtype=np.float64)
    pool_arr = np.array(pool_sizes, dtype=np.float64)
    retained_arr = np.array(retained_case_counts, dtype=np.float64)

    metrics = {
        "n_queries": len(queries),
        "n_gt_cause_occurrences": int(len(ranks)),
        "retrieved_cases": {
            "requested_top_k": int(args.top_k_cases),
            "positive_similarity_only": True,
            "mean_retained": float(retained_arr.mean()) if retained_arr.size else 0.0,
            "median_retained": float(np.median(retained_arr)) if retained_arr.size else 0.0,
            "min_retained": int(retained_arr.min()) if retained_arr.size else 0,
            "max_retained": int(retained_arr.max()) if retained_arr.size else 0,
        },
        "candidate_pool": {
            "mean_size": float(pool_arr.mean()) if pool_arr.size else 0.0,
            "median_size": float(np.median(pool_arr)) if pool_arr.size else 0.0,
            "min_size": int(pool_arr.min()) if pool_arr.size else 0,
            "max_size": int(pool_arr.max()) if pool_arr.size else 0,
        },
        "exact": summarize_rank_metric(ranks, cov_exact),
        "semantic": {
            "threshold": args.semantic_threshold,
            **summarize_rank_metric(sem_ranks, cov_semantic),
            "mean_top1_max_cos_to_gt": float(np.mean(all_top1_max_cos)) if all_top1_max_cos else 0.0,
        },
    }

    add_recall_at_ks(metrics["exact"], ranks, args.ks)
    add_recall_at_ks(metrics["semantic"], sem_ranks, args.ks)

    if all_gt_cluster_ranks:
        cl_ranks = np.array(all_gt_cluster_ranks, dtype=np.float64)
        metrics["cluster"] = {
            "n_gt_cause_occurrences": int(len(cl_ranks)),
            **summarize_rank_metric(cl_ranks, cov_cluster),
        }
        add_recall_at_ks(metrics["cluster"], cl_ranks, args.ks)

    config = {
        **vars(args),
        "case_db_meta": meta,
        "implementation_notes": {
            "embedding_normalization": "L2-normalize global, lesion, and cause embeddings before dot products.",
            "retrieved_case_filter": "Only train cases with combined similarity > 0 are retained for candidate-pool construction and scoring.",
            "case_weighting": "Positive retained similarities are normalized to sum to 1 before candidate scoring.",
            "metrics_ranking": "All metrics use raw candidate-pool score ranking, before diversification.",
            "output_ranking": "predicted_top_n uses diversified ranking for inspection only.",
            "cluster_metric": "Computed per GT cause occurrence; duplicate GT clusters within a query are not deduplicated.",
            "missing_rank_handling": "Misses are +inf internally, null in per_query_results, and never counted by R@K; mean/median rank are covered-only.",
        },
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "config": config}, f,
                  ensure_ascii=False, indent=2)

    with (out_dir / "per_query_results.jsonl").open("w", encoding="utf-8") as f:
        for r in per_query_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== Phase 1 baseline metrics (candidate-restricted, fixed) ===")
    print(f"  n_queries={metrics['n_queries']}  "
          f"n_gt_occ={metrics['n_gt_cause_occurrences']}")
    print(f"  retrieved_cases: positive_only=True  "
          f"requested_K={metrics['retrieved_cases']['requested_top_k']}  "
          f"mean_retained={metrics['retrieved_cases']['mean_retained']:.1f}  "
          f"median_retained={metrics['retrieved_cases']['median_retained']:.0f}  "
          f"min={metrics['retrieved_cases']['min_retained']}  "
          f"max={metrics['retrieved_cases']['max_retained']}")
    print(f"  candidate_pool: mean={metrics['candidate_pool']['mean_size']:.1f}  "
          f"median={metrics['candidate_pool']['median_size']:.0f}  "
          f"min={metrics['candidate_pool']['min_size']}  "
          f"max={metrics['candidate_pool']['max_size']}")

    print(f"\n  -- exact-string match --")
    print(f"    coverage: {metrics['exact']['coverage']:.4f}")
    print(f"    n_covered: {metrics['exact']['n_covered']}  "
          f"n_missed: {metrics['exact']['n_missed']}")
    for k in ["MRR", "median_rank", "mean_rank"]:
        print(f"    {k}: {fmt_metric(metrics['exact'][k])}")
    for k in args.ks:
        print(f"    R@{k}: {metrics['exact'][f'R@{k}']:.4f}")

    print(f"\n  -- semantic match (threshold={args.semantic_threshold}) --")
    print(f"    coverage: {metrics['semantic']['coverage']:.4f}")
    print(f"    n_covered: {metrics['semantic']['n_covered']}  "
          f"n_missed: {metrics['semantic']['n_missed']}")
    for k in ["MRR", "median_rank", "mean_rank", "mean_top1_max_cos_to_gt"]:
        print(f"    {k}: {fmt_metric(metrics['semantic'][k])}")
    for k in args.ks:
        print(f"    R@{k}: {metrics['semantic'][f'R@{k}']:.4f}")

    if "cluster" in metrics:
        print(f"\n  -- cluster-level match (HDBSCAN) --")
        print(f"    coverage: {metrics['cluster']['coverage']:.4f}")
        print(f"    n_covered: {metrics['cluster']['n_covered']}  "
              f"n_missed: {metrics['cluster']['n_missed']}")
        for k in ["MRR", "median_rank", "mean_rank"]:
            print(f"    {k}: {fmt_metric(metrics['cluster'][k])}")
        print(f"    n_gt_cause_occurrences: {metrics['cluster']['n_gt_cause_occurrences']}")
        for k in args.ks:
            print(f"    R@{k}: {metrics['cluster'][f'R@{k}']:.4f}")

    print(f"\n[save] metrics.json + per_query_results.jsonl -> {out_dir}")
    print(f"[done] total time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
