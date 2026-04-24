from __future__ import annotations

from typing import Optional

import numpy as np


def cluster_hdbscan(
    reduced: np.ndarray,
    cluster_selection_method: str,
    min_cluster_size: int,
    min_samples: Optional[int],
) -> np.ndarray:
    """Run HDBSCAN on low-dimensional UMAP output."""
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=int(min_cluster_size),
        min_samples=(int(min_samples) if min_samples is not None else None),
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=-1,
    )
    return clusterer.fit_predict(reduced.astype(np.float32)).astype(np.int32)

