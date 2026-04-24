from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def reduce_embeddings(
    embeddings: torch.Tensor,
    pca_components: Optional[int],
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_n_components: int,
    umap_metric: str,
    random_state: int,
    verbose: bool = True,
) -> np.ndarray:
    """Run optional PCA followed by UMAP."""
    x = embeddings.detach().cpu().float().contiguous().numpy()
    in_dim = int(x.shape[1])

    if pca_components is not None and 0 < int(pca_components) < in_dim:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=int(pca_components), random_state=int(random_state))
        x = pca.fit_transform(x).astype(np.float32)
        if verbose:
            evr = float(pca.explained_variance_ratio_.sum())
            print(f"[reduce] PCA {in_dim} -> {x.shape[1]} (explained_variance={evr:.3f})")

    import umap

    reducer = umap.UMAP(
        n_neighbors=int(umap_n_neighbors),
        min_dist=float(umap_min_dist),
        n_components=int(umap_n_components),
        metric=umap_metric,
        random_state=int(random_state),
    )
    reduced = reducer.fit_transform(x).astype(np.float32)
    if verbose:
        print(
            f"[reduce] UMAP {x.shape[1]} -> {reduced.shape[1]} "
            f"(n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}, "
            f"metric={umap_metric}, random_state={random_state})"
        )
    return reduced


def reduction_metadata(
    pca_components: Optional[int],
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_n_components: int,
    umap_metric: str,
    random_state: int,
) -> Dict[str, Any]:
    return {
        "pca_components": pca_components,
        "umap_n_neighbors": int(umap_n_neighbors),
        "umap_min_dist": float(umap_min_dist),
        "umap_n_components": int(umap_n_components),
        "umap_metric": umap_metric,
        "random_state": int(random_state),
    }


def save_reduced_embeddings(
    reduced: np.ndarray,
    output_path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, reduced.astype(np.float32))

    meta_path = path.with_suffix(path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(dict(metadata or {}), f, ensure_ascii=False, indent=2)


def load_reduced_embeddings(path: str | Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    p = Path(path)
    reduced = np.load(p).astype(np.float32)
    meta_path = p.with_suffix(p.suffix + ".meta.json")
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    return reduced, metadata

