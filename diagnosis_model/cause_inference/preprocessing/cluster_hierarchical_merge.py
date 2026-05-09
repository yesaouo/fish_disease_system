from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .cause_cluster_json import normalize_cluster_json_keys


def _build_text_index(texts: List[str]) -> Dict[str, int]:
    return {text: idx for idx, text in enumerate(texts)}


def _normalize_cluster_state(clusters: Dict) -> Tuple[Dict[int, str], Dict[str, int], Dict[int, Dict]]:
    normalized = normalize_cluster_json_keys(clusters)
    canonical = dict(normalized["cause_id_to_canonical"])
    original_to_cause_id = dict(normalized["original_to_cause_id"])
    meta = {
        int(cid): {"size": int(info["size"]), "members": list(info["members"])}
        for cid, info in normalized["cluster_meta"].items()
    }
    return canonical, original_to_cause_id, meta


def _validate_members_in_cache(meta: Dict[int, Dict], text_to_index: Dict[str, int]) -> None:
    missing_members: List[str] = []
    for info in meta.values():
        for member in info["members"]:
            if member not in text_to_index:
                missing_members.append(member)

    if missing_members:
        preview = ", ".join(missing_members[:5])
        extra = "" if len(missing_members) <= 5 else f" ... (+{len(missing_members) - 5})"
        raise KeyError(f"Cluster member(s) missing from embedding cache: {preview}{extra}")


def _cluster_centroid(
    cid: int,
    meta: Dict[int, Dict],
    x: torch.Tensor,
    text_to_index: Dict[str, int],
) -> torch.Tensor:
    idx = torch.tensor([text_to_index[m] for m in meta[cid]["members"]], dtype=torch.long)
    return F.normalize(x[idx].mean(dim=0, keepdim=True), dim=-1).squeeze(0)


def _choose_merge_direction(meta: Dict[int, Dict], cid_a: int, cid_b: int) -> Tuple[int, int]:
    """Return (target_id, source_id). Smaller clusters are merged into larger clusters."""
    size_a = int(meta[cid_a]["size"])
    size_b = int(meta[cid_b]["size"])

    if size_a > size_b:
        return cid_a, cid_b
    if size_b > size_a:
        return cid_b, cid_a

    # Deterministic tie-break: keep the smaller cause_id.
    return (cid_a, cid_b) if cid_a < cid_b else (cid_b, cid_a)


def _find_best_merge_pair(
    active_ids: List[int],
    centroids: torch.Tensor,
    cosine_threshold: float,
    margin: float,
    batch_size: int,
) -> Optional[Tuple[int, int, float]]:
    n = len(active_ids)
    if n < 2:
        return None

    k = 2 if n > 2 else 1
    best_pair: Optional[Tuple[int, int, float]] = None
    best_sim = float("-inf")
    batch_size = max(1, int(batch_size))

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = centroids[start:end] @ centroids.t()

        # Exclude self-similarity.
        row_offsets = torch.arange(end - start, dtype=torch.long)
        col_indices = torch.arange(start, end, dtype=torch.long)
        sims[row_offsets, col_indices] = float("-inf")

        values, indices = torch.topk(sims, k=k, dim=1)
        best_sims = values[:, 0]
        if k == 2:
            second_sims = values[:, 1]
        else:
            second_sims = torch.full_like(best_sims, float("-inf"))

        eligible = (best_sims >= float(cosine_threshold)) & (
            (best_sims - second_sims) >= float(margin)
        )
        if not bool(eligible.any()):
            continue

        masked_best_sims = best_sims.masked_fill(~eligible, float("-inf"))
        local_best_sim, local_row = torch.max(masked_best_sims, dim=0)
        local_best_sim_float = float(local_best_sim.item())
        if local_best_sim_float > best_sim:
            row = start + int(local_row.item())
            col = int(indices[int(local_row.item()), 0].item())
            best_sim = local_best_sim_float
            best_pair = (active_ids[row], active_ids[col], best_sim)

    return best_pair


def merge_similar_clusters_hierarchical(
    clusters: Dict,
    texts: List[str],
    embeddings: torch.Tensor,
    cosine_threshold: float,
    margin: float = 0.0,
    min_merge_cluster_size: int = 1,
    batch_size: int = 4096,
    max_merge_rounds: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """
    Iteratively merge any two JSON clusters whose centroid cosine similarity passes the threshold.

    This is a separate post-processing step from singleton reassignment:
      1. compute one centroid per active cluster,
      2. find the most similar pair above cosine_threshold,
      3. merge the smaller cluster into the larger cluster,
      4. recompute centroids and repeat until no pair passes the threshold.
    """
    if min_merge_cluster_size < 1:
        raise ValueError("min_merge_cluster_size must be >= 1")
    if cosine_threshold > 1.0:
        raise ValueError("cosine_threshold must be <= 1.0")
    if margin < 0:
        raise ValueError("margin must be >= 0")

    canonical, original_to_cause_id, meta = _normalize_cluster_state(clusters)
    initial_clusters = len(meta)
    initial_singletons = sum(
        1 for info in meta.values()
        if int(info["size"]) == 1 and len(info["members"]) == 1
    )

    stats: Dict = {
        "n_initial_clusters": initial_clusters,
        "n_initial_singletons": initial_singletons,
        "n_final_clusters": initial_clusters,
        "n_final_singletons": initial_singletons,
        "n_merges": 0,
        "max_merged_similarity": None,
        "last_merged_similarity": None,
        "merge_history": [],
    }

    if len(meta) < 2:
        return {
            "cause_id_to_canonical": canonical,
            "original_to_cause_id": original_to_cause_id,
            "cluster_meta": meta,
        }, stats

    text_to_index = _build_text_index(texts)
    _validate_members_in_cache(meta, text_to_index)
    x = F.normalize(embeddings.float().cpu(), dim=-1)

    max_possible_rounds = max(0, len(meta) - 1)
    if max_merge_rounds is None:
        round_limit = max_possible_rounds
    else:
        round_limit = max(0, min(int(max_merge_rounds), max_possible_rounds))

    for round_idx in range(round_limit):
        active_ids = [
            cid for cid, info in sorted(meta.items())
            if int(info["size"]) >= int(min_merge_cluster_size)
        ]
        if len(active_ids) < 2:
            break

        centroids = torch.stack(
            [_cluster_centroid(cid, meta, x, text_to_index) for cid in active_ids],
            dim=0,
        )
        pair = _find_best_merge_pair(
            active_ids=active_ids,
            centroids=centroids,
            cosine_threshold=cosine_threshold,
            margin=margin,
            batch_size=batch_size,
        )
        if pair is None:
            break

        cid_a, cid_b, sim = pair
        target_id, source_id = _choose_merge_direction(meta, cid_a, cid_b)
        source_members = list(meta[source_id]["members"])

        meta[target_id]["members"].extend(source_members)
        meta[target_id]["size"] = len(meta[target_id]["members"])
        for member in source_members:
            original_to_cause_id[member] = target_id

        if target_id not in canonical and source_id in canonical:
            canonical[target_id] = canonical[source_id]
        canonical.pop(source_id, None)
        meta.pop(source_id, None)

        stats["merge_history"].append(
            {
                "round": round_idx + 1,
                "target_id": target_id,
                "source_id": source_id,
                "similarity": sim,
                "target_size_after_merge": int(meta[target_id]["size"]),
            }
        )
        stats["n_merges"] += 1
        stats["last_merged_similarity"] = sim
        if stats["max_merged_similarity"] is None or sim > float(stats["max_merged_similarity"]):
            stats["max_merged_similarity"] = sim

    stats["n_final_clusters"] = len(meta)
    stats["n_final_singletons"] = sum(
        1 for info in meta.values()
        if int(info["size"]) == 1 and len(info["members"]) == 1
    )

    return {
        "cause_id_to_canonical": canonical,
        "original_to_cause_id": original_to_cause_id,
        "cluster_meta": meta,
    }, stats
