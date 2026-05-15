"""Generate the CRR-DeepRVQ compression × quality Pareto plot.

Combines two data sources per encoder (raw / fine-tuned):
  - sweep_report.json   : 12-cell (M, K) grid of rvq_only at Regime B
  - final_eval_report   : 3 trained reranker configs (Light + Full analytic)

Output: a log-scale plot of sem R@10 vs compression ratio, with both raw and
fine-tuned curves overlaid. Stage 1 dense baselines are horizontal references.

CLI from repo root:
    /home/lab603/anaconda3/envs/SDM/bin/python \\
        -m diagnosis_model.cause_inference.rvq_rerank.plot_pareto

Output: outputs/rvq_rerank/pareto_compression_vs_R10.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_sweep_rvq_only(report_path: Path) -> list:
    """Return list of (compression_x, sem_R@10) from sweep_report.json."""
    data = json.loads(report_path.read_text())
    rows = data["results"]
    pts = [(r["compression_x"], r["sem_R@10"] * 100) for r in rows
           if r.get("M", -1) > 0]
    pts.sort(key=lambda x: x[0])
    return pts


def _load_reranker_points(report_path: Path) -> dict:
    """From final_eval_report.json (Regime B rows), return:
        {method: [(compression_x, sem_R@10%), ...]}
    for method in {rvq_only, light, full_analytic} at top_k=1.
    """
    data = json.loads(report_path.read_text())
    out = {"rvq_only": [], "light": [], "full_analytic": [], "dense_R10": None}
    for r in data["rows"]:
        if r["top_k"] != 1:
            continue
        if r["method"] == "dense":
            out["dense_R10"] = r["sem_R@10"] * 100
        elif r["method"] in ("rvq_only", "light", "full_analytic"):
            out[r["method"]].append((r["compression_x"], r["sem_R@10"] * 100))
    for k in ("rvq_only", "light", "full_analytic"):
        out[k].sort(key=lambda x: x[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_final",
                    default="diagnosis_model/cause_inference/outputs/"
                            "rvq_rerank_raw/final_eval_report.json")
    ap.add_argument("--fine_final",
                    default="diagnosis_model/cause_inference/outputs/"
                            "rvq_rerank/final_eval_report.json")
    ap.add_argument("--raw_sweep",
                    default="diagnosis_model/cause_inference/outputs/"
                            "rvq_rerank_raw/sweep_report.json")
    ap.add_argument("--fine_sweep",
                    default="diagnosis_model/cause_inference/outputs/"
                            "rvq_rerank/sweep_report.json")
    ap.add_argument("--output",
                    default="diagnosis_model/cause_inference/outputs/"
                            "rvq_rerank/pareto_compression_vs_R10.png")
    args = ap.parse_args()

    raw = _load_reranker_points(Path(args.raw_final))
    fine = _load_reranker_points(Path(args.fine_final))
    raw_sweep = _load_sweep_rvq_only(Path(args.raw_sweep))
    fine_sweep = _load_sweep_rvq_only(Path(args.fine_sweep))

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    RAW_COLOR = "#1f77b4"   # blue
    FINE_COLOR = "#d62728"  # red

    # Full sweep rvq_only (12 points per side, light-weight markers)
    x = [p[0] for p in raw_sweep]
    y = [p[1] for p in raw_sweep]
    ax.scatter(x, y, marker="x", color=RAW_COLOR, alpha=0.45, s=42, zorder=2,
               label="raw RVQ-only (12-cell sweep)")
    x = [p[0] for p in fine_sweep]
    y = [p[1] for p in fine_sweep]
    ax.scatter(x, y, marker="x", color=FINE_COLOR, alpha=0.45, s=42, zorder=2,
               label="fine-tuned RVQ-only (12-cell sweep)")

    # 3 trained-reranker configs: prominent connected lines
    x = [p[0] for p in raw["rvq_only"]]
    y = [p[1] for p in raw["rvq_only"]]
    ax.plot(x, y, "o--", color=RAW_COLOR, alpha=0.85, linewidth=1.6,
            markersize=8.5, zorder=3, label="raw RVQ-only (trained configs)")
    x = [p[0] for p in raw["light"]]
    y = [p[1] for p in raw["light"]]
    ax.plot(x, y, "s-", color=RAW_COLOR, linewidth=2.6, markersize=10, zorder=4,
            label="raw + Light reranker")
    x = [p[0] for p in raw["full_analytic"]]
    y = [p[1] for p in raw["full_analytic"]]
    ax.plot(x, y, "^:", color=RAW_COLOR, alpha=0.7, linewidth=1.8,
            markersize=9, zorder=3, label="raw + Full analytic")

    x = [p[0] for p in fine["rvq_only"]]
    y = [p[1] for p in fine["rvq_only"]]
    ax.plot(x, y, "o--", color=FINE_COLOR, alpha=0.85, linewidth=1.6,
            markersize=8.5, zorder=3,
            label="fine-tuned RVQ-only (trained configs)")
    x = [p[0] for p in fine["light"]]
    y = [p[1] for p in fine["light"]]
    ax.plot(x, y, "s-", color=FINE_COLOR, linewidth=2.6, markersize=10, zorder=4,
            label="fine-tuned + Light reranker")
    x = [p[0] for p in fine["full_analytic"]]
    y = [p[1] for p in fine["full_analytic"]]
    ax.plot(x, y, "^:", color=FINE_COLOR, alpha=0.7, linewidth=1.8,
            markersize=9, zorder=3,
            label="fine-tuned + Full analytic")

    # Dense reference lines
    ax.axhline(raw["dense_R10"], color=RAW_COLOR, linestyle="-",
               linewidth=1.0, alpha=0.30,
               label=f"raw dense (no compression): {raw['dense_R10']:.1f}%")
    ax.axhline(fine["dense_R10"], color=FINE_COLOR, linestyle="-",
               linewidth=1.0, alpha=0.30,
               label=f"fine-tuned dense (no compression): {fine['dense_R10']:.1f}%")

    ax.set_xscale("log")
    ax.set_xlabel("Compression ratio vs fp32 dense (× log scale)", fontsize=12)
    ax.set_ylabel("sem R@10 (%, Regime B: top_k_cases=1)", fontsize=12)
    ax.set_title(
        "CRR-DeepRVQ Pareto — compression × retrieval quality\n"
        "Regime B (no aggregation buffer); Light reranker recovers RVQ damage; "
        "Full analytic = oracle upper bound",
        fontsize=11.5,
    )
    ax.grid(True, which="major", linestyle="-", alpha=0.25)
    ax.grid(True, which="minor", linestyle=":", alpha=0.15)
    ax.legend(loc="lower left", fontsize=8.0, framealpha=0.92, ncol=2)

    # Annotate production point
    if raw["light"]:
        prod = raw["light"][-1]  # M=4 K=256 has lowest compression (rightmost in sweep)
        # find by max comp value in raw["light"]
        prod = max(raw["light"], key=lambda p: p[0])
        ax.annotate(
            "Production\n(raw + Light, M=4 K=256)",
            xy=prod,
            xytext=(prod[0] * 0.18, prod[1] - 5),
            arrowprops=dict(arrowstyle="->", color=RAW_COLOR, lw=1.4),
            fontsize=9.5, color=RAW_COLOR, fontweight="bold", ha="center",
        )

    # Annotate compression axis sweet points
    ax.set_ylim(top=max(raw["dense_R10"], fine["dense_R10"]) + 4)

    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Compact tabular summary for paper / inference.txt
    print("\n=== Regime B Pareto (sem R@10 %) — trained configs ===")
    print(f"{'Comp×':>8s} | {'raw rvq':>8s} {'raw L':>7s} {'raw F':>7s} | "
          f"{'fine rvq':>9s} {'fine L':>7s} {'fine F':>7s}")
    print("-" * 70)
    # Index by compression for alignment
    raw_rvq = dict(raw["rvq_only"])
    raw_lig = dict(raw["light"])
    raw_full = dict(raw["full_analytic"])
    fine_rvq = dict(fine["rvq_only"])
    fine_lig = dict(fine["light"])
    fine_full = dict(fine["full_analytic"])
    for comp in sorted(set(raw_rvq) | set(fine_rvq), reverse=True):
        rrvq = raw_rvq.get(comp, float("nan"))
        rlig = raw_lig.get(comp, float("nan"))
        rful = raw_full.get(comp, float("nan"))
        frvq = fine_rvq.get(comp, float("nan"))
        flig = fine_lig.get(comp, float("nan"))
        fful = fine_full.get(comp, float("nan"))
        print(f"{comp:>7.0f}× | {rrvq:>8.2f} {rlig:>7.2f} {rful:>7.2f} | "
              f"{frvq:>9.2f} {flig:>7.2f} {fful:>7.2f}")
    print(f"  Dense  | {raw['dense_R10']:>26.2f} | {fine['dense_R10']:>26.2f}")


if __name__ == "__main__":
    main()
