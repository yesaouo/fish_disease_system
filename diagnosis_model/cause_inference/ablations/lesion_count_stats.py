"""Lesion-count distribution for case_db (train + valid).

Reports per-split mean / median / quantiles and a 1..N+ histogram, plus a PNG
plot. Used to justify the "small-set regime" framing in the aggregation ablation.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def load_lesion_counts(pt_path: Path) -> np.ndarray:
    cases = torch.load(pt_path, weights_only=False, map_location="cpu")
    return np.array([int(c["lesion_embs"].shape[0]) for c in cases], dtype=np.int64)


def summarize(counts: np.ndarray) -> dict:
    return {
        "n_cases":  int(counts.size),
        "mean":     float(counts.mean()),
        "median":   float(np.median(counts)),
        "p25":      float(np.quantile(counts, 0.25)),
        "p75":      float(np.quantile(counts, 0.75)),
        "p90":      float(np.quantile(counts, 0.90)),
        "p99":      float(np.quantile(counts, 0.99)),
        "max":      int(counts.max()),
        "min":      int(counts.min()),
        "histogram": dict(sorted(Counter(counts.tolist()).items())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db")
    ap.add_argument("--output_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/ablation_aggregation")
    args = ap.parse_args()

    case_db = Path(args.case_db_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split in ["train", "valid"]:
        pt = case_db / f"{split}_cases.pt"
        if not pt.exists():
            continue
        counts = load_lesion_counts(pt)
        splits[split] = {"counts": counts, "summary": summarize(counts)}
        s = splits[split]["summary"]
        print(f"[{split}] n={s['n_cases']}  mean={s['mean']:.3f}  median={s['median']}  "
              f"p25/p75/p90/p99={s['p25']}/{s['p75']}/{s['p90']}/{s['p99']}  max={s['max']}")
        bins = s["histogram"]
        print(f"  histogram: {bins}")

    json_out = {k: v["summary"] for k, v in splits.items()}
    (out_dir / "lesion_count_stats.json").write_text(json.dumps(json_out, indent=2))
    print(f"\nwrote {out_dir / 'lesion_count_stats.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        max_k = max(max(s["counts"].max() for s in splits.values()), 1)
        bins_edges = np.arange(0.5, max_k + 1.5, 1.0)
        for split, payload in splits.items():
            ax.hist(payload["counts"], bins=bins_edges, alpha=0.55, label=f"{split} (n={len(payload['counts'])})")
        ax.set_xlabel("# lesions per case")
        ax.set_ylabel("# cases")
        ax.set_title("Lesion-count distribution (case_db)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        png = out_dir / "lesion_count_hist.png"
        fig.tight_layout()
        fig.savefig(png, dpi=140)
        print(f"wrote {png}")
    except ImportError:
        print("matplotlib unavailable; skipping plot")


if __name__ == "__main__":
    main()
