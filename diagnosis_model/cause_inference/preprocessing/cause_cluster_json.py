from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F


def _pick_canonical(member_indices: List[int], embeddings: torch.Tensor, texts: List[str]) -> str:
    embs = embeddings[torch.tensor(member_indices, dtype=torch.long)]
    centroid = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)
    sims = (embs @ centroid.t()).squeeze(-1)
    best = int(torch.argmax(sims).item())
    return texts[member_indices[best]]


def assign_cause_ids(
    texts: List[str],
    embeddings: torch.Tensor,
    cluster_labels: np.ndarray,
    starting_id: int = 1,
) -> Dict:
    if not (len(texts) == int(embeddings.size(0)) == len(cluster_labels)):
        raise ValueError("texts, embeddings, and cluster_labels must have the same length")

    real_clusters: Dict[int, List[int]] = defaultdict(list)
    noise_indices: List[int] = []
    for idx, label in enumerate(cluster_labels):
        if int(label) < 0:
            noise_indices.append(idx)
        else:
            real_clusters[int(label)].append(idx)

    cause_id_to_canonical: Dict[int, str] = {}
    original_to_cause_id: Dict[str, int] = {}
    cluster_meta: Dict[int, Dict] = {}

    next_id = int(starting_id)
    for _, members in sorted(real_clusters.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        cause_id = next_id
        next_id += 1
        canonical = _pick_canonical(members, embeddings, texts)
        cause_id_to_canonical[cause_id] = canonical
        cluster_meta[cause_id] = {"size": len(members), "members": [texts[i] for i in members]}
        for idx in members:
            original_to_cause_id[texts[idx]] = cause_id

    for idx in noise_indices:
        cause_id = next_id
        next_id += 1
        cause_id_to_canonical[cause_id] = texts[idx]
        cluster_meta[cause_id] = {"size": 1, "members": [texts[idx]]}
        original_to_cause_id[texts[idx]] = cause_id

    return {
        "cause_id_to_canonical": cause_id_to_canonical,
        "original_to_cause_id": original_to_cause_id,
        "cluster_meta": cluster_meta,
    }


def normalize_cluster_json_keys(clusters: Dict) -> Dict:
    return {
        "cause_id_to_canonical": {
            int(k): v for k, v in clusters.get("cause_id_to_canonical", {}).items()
        },
        "original_to_cause_id": {
            str(k): int(v) for k, v in clusters.get("original_to_cause_id", {}).items()
        },
        "cluster_meta": {
            int(k): {
                "size": int(v.get("size", len(v.get("members", [])))),
                "members": list(v.get("members", [])),
            }
            for k, v in clusters.get("cluster_meta", {}).items()
        },
    }


def load_clusters_json(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return normalize_cluster_json_keys(json.load(f))


def save_clusters_json(clusters: Dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_cluster_json_keys(clusters)
    serializable = {
        "cause_id_to_canonical": {
            str(k): v for k, v in normalized["cause_id_to_canonical"].items()
        },
        "original_to_cause_id": normalized["original_to_cause_id"],
        "cluster_meta": {
            str(k): v for k, v in normalized["cluster_meta"].items()
        },
    }
    with output.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def compute_stats(clusters: Dict) -> Dict:
    meta = normalize_cluster_json_keys(clusters)["cluster_meta"]
    sizes = np.array([m["size"] for m in meta.values()], dtype=np.int64)
    n_clusters = int(sizes.size)
    n_strings = int(sizes.sum()) if sizes.size else 0
    n_singletons = int((sizes == 1).sum()) if sizes.size else 0
    n_real = int((sizes >= 2).sum()) if sizes.size else 0
    return {
        "n_clusters": n_clusters,
        "n_real_clusters": n_real,
        "n_singletons": n_singletons,
        "n_strings": n_strings,
        "compression": float(1.0 - n_clusters / max(1, n_strings)),
        "mean_size": float(sizes.mean()) if sizes.size else 0.0,
        "median_size": float(np.median(sizes)) if sizes.size else 0.0,
        "size_std": float(sizes.std()) if sizes.size else 0.0,
        "largest": int(sizes.max()) if sizes.size else 0,
    }


def quality_report(clusters: Dict, top_n: int = 15) -> str:
    normalized = normalize_cluster_json_keys(clusters)
    meta = normalized["cluster_meta"]
    canonical_map = normalized["cause_id_to_canonical"]
    sizes = [m["size"] for m in meta.values()]
    n_clusters = len(sizes)
    n_singletons = sum(1 for size in sizes if size == 1)
    n_strings = sum(sizes)

    buckets = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 10**9)]
    bucket_labels = ["1 (singleton)", "2", "3-5", "6-10", "11-20", "21+"]
    histogram = [sum(1 for size in sizes if lo <= size <= hi) for lo, hi in buckets]

    lines = [
        "=== Cluster quality report ===",
        f"total cause_ids       : {n_clusters}",
        f"singletons            : {n_singletons} ({n_singletons / max(1, n_clusters):.1%})",
        f"total cause strings   : {n_strings}",
        f"compression ratio     : {1 - n_clusters / max(1, n_strings):.1%}",
        f"avg cluster size      : {n_strings / max(1, n_clusters):.2f}",
    ]
    if sizes:
        lines.append(f"largest cluster size  : {max(sizes)}")

    lines.append("")
    lines.append("--- Cluster-size histogram ---")
    for label, count in zip(bucket_labels, histogram):
        lines.append(f"  size {label:<15s}: {count}")

    lines.append("")
    lines.append(f"--- Top {top_n} largest clusters ---")
    for cause_id, info in sorted(meta.items(), key=lambda kv: -kv[1]["size"])[:top_n]:
        lines.append(f"[cause_id={cause_id}, size={info['size']}] {canonical_map[cause_id]}")
        for member in info["members"][:3]:
            tag = " <-- CANON" if member == canonical_map[cause_id] else ""
            lines.append(f"   - {member}{tag}")
        if len(info["members"]) > 3:
            lines.append(f"   ... and {len(info['members']) - 3} more")

    return "\n".join(lines)


def map_image_causes_to_ids(
    cause_strings: Iterable[str],
    original_to_cause_id: Dict[str, int],
    strict: bool = True,
) -> List[int]:
    cause_ids: List[int] = []
    missing: List[str] = []
    for cause in cause_strings:
        cause_id = original_to_cause_id.get(cause)
        if cause_id is None:
            missing.append(cause)
            continue
        if cause_id not in cause_ids:
            cause_ids.append(int(cause_id))

    if missing and strict:
        preview = ", ".join(missing[:5])
        extra = "" if len(missing) <= 5 else f" ... (+{len(missing) - 5})"
        raise KeyError(f"Cause string(s) missing from cluster map: {preview}{extra}")

    return cause_ids

