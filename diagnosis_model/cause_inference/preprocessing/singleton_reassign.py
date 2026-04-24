from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .cause_cluster_json import normalize_cluster_json_keys


def _build_text_index(texts: List[str]) -> Dict[str, int]:
    return {text: idx for idx, text in enumerate(texts)}


def reassign_singletons_to_real_clusters(
    clusters: Dict,
    texts: List[str],
    embeddings: torch.Tensor,
    cosine_threshold: float,
    margin: float = 0.0,
    min_real_cluster_size: int = 2,
    batch_size: int = 4096,
) -> Tuple[Dict, Dict[str, int]]:
    """Attach singleton JSON clusters to existing real clusters by cosine similarity."""
    normalized = normalize_cluster_json_keys(clusters)
    canonical = dict(normalized["cause_id_to_canonical"])
    original_to_cause_id = dict(normalized["original_to_cause_id"])
    meta = {
        int(cid): {"size": int(info["size"]), "members": list(info["members"])}
        for cid, info in normalized["cluster_meta"].items()
    }

    singleton_ids = [
        cid for cid, info in sorted(meta.items())
        if int(info["size"]) == 1 and len(info["members"]) == 1
    ]
    anchor_ids = [
        cid for cid, info in sorted(meta.items())
        if int(info["size"]) >= int(min_real_cluster_size)
    ]
    stats = {
        "n_singleton_candidates": len(singleton_ids),
        "n_anchor_clusters": len(anchor_ids),
        "n_attached": 0,
        "n_kept_singleton": len(singleton_ids),
    }
    if cosine_threshold <= 0 or not singleton_ids or not anchor_ids:
        return normalized, stats

    text_to_index = _build_text_index(texts)
    missing_members: List[str] = []
    for cid in singleton_ids + anchor_ids:
        for member in meta[cid]["members"]:
            if member not in text_to_index:
                missing_members.append(member)
    if missing_members:
        preview = ", ".join(missing_members[:5])
        extra = "" if len(missing_members) <= 5 else f" ... (+{len(missing_members) - 5})"
        raise KeyError(f"Cluster member(s) missing from embedding cache: {preview}{extra}")

    x = F.normalize(embeddings.float().cpu(), dim=-1)
    anchor_centroids = []
    for cid in anchor_ids:
        idx = torch.tensor([text_to_index[m] for m in meta[cid]["members"]], dtype=torch.long)
        centroid = F.normalize(x[idx].mean(dim=0, keepdim=True), dim=-1)
        anchor_centroids.append(centroid.squeeze(0))
    anchor_matrix = torch.stack(anchor_centroids, dim=0)

    attached_pairs: List[Tuple[int, int]] = []
    singleton_texts = [meta[cid]["members"][0] for cid in singleton_ids]
    singleton_indices = torch.tensor([text_to_index[text] for text in singleton_texts], dtype=torch.long)
    singleton_vectors = x[singleton_indices]
    k = 2 if len(anchor_ids) >= 2 else 1

    for start in range(0, singleton_vectors.size(0), int(batch_size)):
        batch_vectors = singleton_vectors[start:start + int(batch_size)]
        sims = batch_vectors @ anchor_matrix.t()
        values, indices = torch.topk(sims, k=k, dim=1)
        best_sims = values[:, 0]
        if k == 2:
            second_sims = values[:, 1]
        else:
            second_sims = torch.full_like(best_sims, -1.0)

        chosen = (best_sims >= float(cosine_threshold)) & (
            (best_sims - second_sims) >= float(margin)
        )
        for row in torch.nonzero(chosen, as_tuple=False).flatten().tolist():
            singleton_id = singleton_ids[start + int(row)]
            anchor_id = anchor_ids[int(indices[row, 0].item())]
            attached_pairs.append((singleton_id, anchor_id))

    for singleton_id, anchor_id in attached_pairs:
        text = meta[singleton_id]["members"][0]
        meta[anchor_id]["members"].append(text)
        meta[anchor_id]["size"] = len(meta[anchor_id]["members"])
        original_to_cause_id[text] = anchor_id

        meta.pop(singleton_id, None)
        canonical.pop(singleton_id, None)

    stats["n_attached"] = len(attached_pairs)
    stats["n_kept_singleton"] = len(singleton_ids) - len(attached_pairs)

    return {
        "cause_id_to_canonical": canonical,
        "original_to_cause_id": original_to_cause_id,
        "cluster_meta": meta,
    }, stats

