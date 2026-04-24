"""
Cause string clustering CLI.

Pipeline:
  cache text embeddings -> optional PCA -> UMAP -> HDBSCAN -> cause_clusters.json

The heavy encoder step is isolated in the cache command. The same cache can be
reused for parameter sweeps, JSON export, and singleton reassignment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from diagnosis_model.cause_inference.preprocessing.cause_cluster_json import (  # type: ignore
        assign_cause_ids,
        compute_stats,
        load_clusters_json,
        quality_report,
        save_clusters_json,
    )
    from diagnosis_model.cause_inference.preprocessing.dim_reduction import (  # type: ignore
        load_reduced_embeddings,
        reduce_embeddings,
        reduction_metadata,
        save_reduced_embeddings,
    )
    from diagnosis_model.cause_inference.preprocessing.hdbscan_clustering import cluster_hdbscan  # type: ignore
    from diagnosis_model.cause_inference.preprocessing.singleton_reassign import (  # type: ignore
        reassign_singletons_to_real_clusters,
    )
    from diagnosis_model.cause_inference.preprocessing.text_embedding_cache import (  # type: ignore
        build_text_embedding_cache,
        load_text_embedding_cache,
    )
else:
    from .cause_cluster_json import (
        assign_cause_ids,
        compute_stats,
        load_clusters_json,
        quality_report,
        save_clusters_json,
    )
    from .dim_reduction import (
        load_reduced_embeddings,
        reduce_embeddings,
        reduction_metadata,
        save_reduced_embeddings,
    )
    from .hdbscan_clustering import cluster_hdbscan
    from .singleton_reassign import reassign_singletons_to_real_clusters
    from .text_embedding_cache import build_text_embedding_cache, load_text_embedding_cache


DEFAULT_SWEEP_N_NEIGHBORS = [3, 5, 8, 12]
DEFAULT_SWEEP_MIN_CLUSTER_SIZE = [3, 6, 9, 12]


def _resolve_model_path(args: argparse.Namespace) -> str:
    if args.encoder_backend == "vlm":
        if not args.vlm_path:
            raise ValueError("--vlm_path is required when --encoder_backend=vlm")
        return args.vlm_path
    if not args.hf_model:
        raise ValueError("--hf_model is required when --encoder_backend is hf or sentence-transformers")
    return args.hf_model


def _load_or_reduce(args: argparse.Namespace, embeddings: torch.Tensor) -> np.ndarray:
    if args.reduced_path:
        reduced, meta = load_reduced_embeddings(args.reduced_path)
        print(f"[reduce] loaded {tuple(reduced.shape)} from {args.reduced_path}")
        if meta:
            print(f"[reduce] metadata: {meta}")
        return reduced

    reduced = reduce_embeddings(
        embeddings,
        pca_components=args.pca_components,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_n_components=args.umap_n_components,
        umap_metric=args.umap_metric,
        random_state=args.random_state,
    )
    if args.save_reduced:
        save_reduced_embeddings(
            reduced,
            args.save_reduced,
            metadata=reduction_metadata(
                args.pca_components,
                args.umap_n_neighbors,
                args.umap_min_dist,
                args.umap_n_components,
                args.umap_metric,
                args.random_state,
            ),
        )
        print(f"[reduce] saved reduced embeddings to {args.save_reduced}")
    return reduced


def _run_cluster_json(
    texts: List[str],
    embeddings: torch.Tensor,
    reduced: np.ndarray,
    cluster_selection_method: str,
    min_cluster_size: int,
    min_samples: Optional[int],
) -> Tuple[Dict, np.ndarray]:
    labels = cluster_hdbscan(
        reduced,
        cluster_selection_method=cluster_selection_method,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    clusters = assign_cause_ids(texts, embeddings, labels, starting_id=1)
    return clusters, labels


def cmd_cache(args: argparse.Namespace) -> None:
    model_name_or_path = _resolve_model_path(args)
    build_text_embedding_cache(
        coco_files=args.coco_files,
        cache_dir=args.cache_dir,
        backend=args.encoder_backend,
        model_name_or_path=model_name_or_path,
        device=args.device,
        text_batch_size=args.text_batch_size,
        max_length=args.max_length,
        use_amp=not args.no_amp,
        text_template=args.text_template,
        include_healthy=args.include_healthy,
        hf_pooling=args.hf_pooling,
        trust_remote_code=args.trust_remote_code,
    )


def cmd_reduce(args: argparse.Namespace) -> None:
    _, embeddings, _ = load_text_embedding_cache(args.cache_dir)
    reduced = reduce_embeddings(
        embeddings,
        pca_components=args.pca_components,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_n_components=args.umap_n_components,
        umap_metric=args.umap_metric,
        random_state=args.random_state,
    )
    save_reduced_embeddings(
        reduced,
        args.output,
        metadata=reduction_metadata(
            args.pca_components,
            args.umap_n_neighbors,
            args.umap_min_dist,
            args.umap_n_components,
            args.umap_metric,
            args.random_state,
        ),
    )
    print(f"[reduce] saved {tuple(reduced.shape)} to {args.output}")


def cmd_cluster(args: argparse.Namespace) -> None:
    texts, embeddings, _ = load_text_embedding_cache(args.cache_dir)
    print(f"[cluster] loaded {len(texts)} texts, embeddings={tuple(embeddings.shape)}")

    reduced = _load_or_reduce(args, embeddings)
    clusters, labels = _run_cluster_json(
        texts,
        embeddings,
        reduced,
        cluster_selection_method=args.cluster_selection_method,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    n_noise = int((labels < 0).sum())
    n_real = len(set(int(x) for x in labels if int(x) >= 0))
    print(
        f"[hdbscan] method={args.cluster_selection_method}, "
        f"min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}: "
        f"{n_real} real clusters, {n_noise} noise points"
    )

    save_clusters_json(clusters, args.output)
    print()
    print(quality_report(clusters, top_n=args.report_top_n))
    print(f"\nSaved: {args.output}")


def _format_sweep_table(rows: List[Dict]) -> str:
    header = (
        f"{'n_neigh':>7} {'n_comp':>6} {'mcs':>4} {'method':>6} | "
        f"{'clusters':>8} {'real':>6} {'single':>7} | "
        f"{'mean':>6} {'median':>6} {'std':>6} {'largest':>7} {'compress':>8}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    for row in rows:
        stats = row["stats"]
        lines.append(
            f"{row['umap_n_neighbors']:>7} {row['umap_n_components']:>6} "
            f"{row['min_cluster_size']:>4} "
            f"{row['cluster_selection_method']:>6} | "
            f"{stats['n_clusters']:>8} {stats['n_real_clusters']:>6} "
            f"{stats['n_singletons']:>7} | "
            f"{stats['mean_size']:>6.2f} {stats['median_size']:>6.1f} "
            f"{stats['size_std']:>6.2f} {stats['largest']:>7} "
            f"{stats['compression']:>8.1%}"
        )
    return "\n".join(lines)


def cmd_sweep(args: argparse.Namespace) -> None:
    texts, embeddings, _ = load_text_embedding_cache(args.cache_dir)
    print(f"[sweep] loaded {len(texts)} texts, embeddings={tuple(embeddings.shape)}")

    nn_grid = args.sweep_umap_n_neighbors or DEFAULT_SWEEP_N_NEIGHBORS
    n_components_grid = args.sweep_umap_n_components or [args.umap_n_components]
    mcs_grid = args.sweep_min_cluster_size or DEFAULT_SWEEP_MIN_CLUSTER_SIZE
    total = len(nn_grid) * len(n_components_grid) * len(mcs_grid)
    print(
        f"[sweep] grid: n_neighbors={nn_grid}, n_components={n_components_grid}, "
        f"min_cluster_size={mcs_grid}; method={args.cluster_selection_method} -> {total} combos"
    )

    output_path = Path(args.output)
    summary_path = (
        output_path.with_suffix(output_path.suffix + ".sweep.json")
        if output_path.suffix
        else output_path.with_name(output_path.name + ".sweep.json")
    )

    rows: List[Dict] = []
    combo_idx = 0
    for n_neighbors in nn_grid:
        for n_components in n_components_grid:
            reduced = reduce_embeddings(
                embeddings,
                pca_components=args.pca_components,
                umap_n_neighbors=n_neighbors,
                umap_min_dist=args.umap_min_dist,
                umap_n_components=n_components,
                umap_metric=args.umap_metric,
                random_state=args.random_state,
            )

            for min_cluster_size in mcs_grid:
                combo_idx += 1
                tag = f"nn{n_neighbors}_nc{n_components}_mcs{min_cluster_size}_{args.cluster_selection_method}"
                print(f"\n[sweep {combo_idx}/{total}] {tag}")
                clusters, _ = _run_cluster_json(
                    texts,
                    embeddings,
                    reduced,
                    cluster_selection_method=args.cluster_selection_method,
                    min_cluster_size=min_cluster_size,
                    min_samples=args.min_samples,
                )
                stats = compute_stats(clusters)
                row = {
                    "tag": tag,
                    "umap_n_neighbors": n_neighbors,
                    "umap_n_components": n_components,
                    "min_cluster_size": min_cluster_size,
                    "cluster_selection_method": args.cluster_selection_method,
                    "stats": stats,
                }
                rows.append(row)
                print(
                    f"  clusters={stats['n_clusters']} real={stats['n_real_clusters']} "
                    f"singletons={stats['n_singletons']} mean_size={stats['mean_size']:.2f} "
                    f"largest={stats['largest']} compression={stats['compression']:.1%}"
                )

                if not args.sweep_save_summary_only:
                    per_combo_path = output_path.with_name(
                        f"{output_path.stem}__{tag}{output_path.suffix or '.json'}"
                    )
                    save_clusters_json(clusters, per_combo_path)
                    row["saved_to"] = str(per_combo_path)

    print("\n=== Sweep summary ===")
    print(_format_sweep_table(rows))

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "grid": {
                    "umap_n_neighbors": list(nn_grid),
                    "umap_n_components": list(n_components_grid),
                    "min_cluster_size": list(mcs_grid),
                },
                "shared_params": {
                    "pca_components": args.pca_components,
                    "umap_min_dist": args.umap_min_dist,
                    "umap_metric": args.umap_metric,
                    "random_state": args.random_state,
                    "cluster_selection_method": args.cluster_selection_method,
                    "min_samples": args.min_samples,
                },
                "results": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[sweep] summary written to: {summary_path}")


def cmd_reassign_singletons(args: argparse.Namespace) -> None:
    texts, embeddings, _ = load_text_embedding_cache(args.cache_dir)
    clusters = load_clusters_json(args.input)
    reassigned, stats = reassign_singletons_to_real_clusters(
        clusters,
        texts,
        embeddings,
        cosine_threshold=args.cosine_threshold,
        margin=args.margin,
        min_real_cluster_size=args.min_real_cluster_size,
        batch_size=args.batch_size,
    )
    save_clusters_json(reassigned, args.output)
    print(
        f"[reassign-singletons] anchors={stats['n_anchor_clusters']}, "
        f"candidates={stats['n_singleton_candidates']}, "
        f"attached={stats['n_attached']}, kept_singleton={stats['n_kept_singleton']}"
    )
    print()
    print(quality_report(reassigned, top_n=args.report_top_n))
    print(f"\nSaved: {args.output}")


def add_encoder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--encoder_backend",
        choices=["vlm", "hf", "sentence-transformers"],
        default="vlm",
        help="Text encoder backend used to create the cache.",
    )
    parser.add_argument("--vlm_path", default=None, help="Path to trained VLM checkpoint.")
    parser.add_argument("--hf_model", default=None, help="Hugging Face model name/path.")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--text_batch_size", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument(
        "--text_template",
        type=str,
        default="{cap}。",
        help="Prompt wrapper used before text encoding.",
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--hf_pooling", choices=["mean", "cls", "pooler"], default="mean")
    parser.add_argument("--trust_remote_code", action="store_true")


def add_reduce_args(parser: argparse.ArgumentParser, allow_reduced_path: bool = False) -> None:
    if allow_reduced_path:
        parser.add_argument(
            "--reduced_path",
            default=None,
            help="Existing .npy reduced embedding file. If omitted, PCA+UMAP is run.",
        )
        parser.add_argument(
            "--save_reduced",
            default=None,
            help="Optional .npy path for saving newly computed reduced embeddings.",
        )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=None,
        help="Optional PCA dimensions before UMAP. Use 50 as a common acceleration setting.",
    )
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.0)
    parser.add_argument("--umap_n_components", type=int, default=5)
    parser.add_argument("--umap_metric", type=str, default="cosine")
    parser.add_argument("--random_state", type=int, default=42)


def add_hdbscan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cluster_selection_method", choices=["leaf", "eom"], default="eom")
    parser.add_argument("--min_cluster_size", type=int, default=5)
    parser.add_argument(
        "--min_samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples. Default inside HDBSCAN is min_cluster_size.",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cache", help="Collect cause strings and encode text embeddings.")
    p.add_argument("--coco_files", nargs="+", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--include_healthy", action="store_true")
    add_encoder_args(p)
    p.set_defaults(func=cmd_cache)

    p = sub.add_parser("reduce", help="Run PCA+UMAP from cached embeddings and save .npy.")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--output", required=True)
    add_reduce_args(p)
    p.set_defaults(func=cmd_reduce)

    p = sub.add_parser("cluster", help="Run reduction if needed, then HDBSCAN and JSON export.")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--output", required=True)
    add_reduce_args(p, allow_reduced_path=True)
    add_hdbscan_args(p)
    p.add_argument("--report_top_n", type=int, default=15)
    p.set_defaults(func=cmd_cluster)

    p = sub.add_parser("sweep", help="Grid sweep UMAP neighbors and HDBSCAN params.")
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--output", required=True)
    add_reduce_args(p)
    p.add_argument("--cluster_selection_method", choices=["leaf", "eom"], default="eom")
    p.add_argument("--min_samples", type=int, default=None)
    p.add_argument("--sweep_umap_n_neighbors", type=int, nargs="+", default=None)
    p.add_argument("--sweep_umap_n_components", type=int, nargs="+", default=None)
    p.add_argument("--sweep_min_cluster_size", type=int, nargs="+", default=None)
    p.add_argument(
        "--sweep_save_summary_only",
        action="store_true",
        help="Skip per-combo cluster JSON files and only write the sweep summary.",
    )
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser(
        "reassign-singletons",
        help="Attach singleton JSON clusters to existing real clusters by cosine similarity.",
    )
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cosine_threshold", type=float, required=True)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--min_real_cluster_size", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--report_top_n", type=int, default=15)
    p.set_defaults(func=cmd_reassign_singletons)

    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
