"""Render Phase 3 case study figures: fish image + lesion bboxes color-coded by α
plus a text panel with GT cause and CEAH top-1 attribution breakdown.

Usage:
  python -m diagnosis_model.cause_inference.case_study_viz \
    --eval_dir diagnosis_model/cause_inference/outputs/ceah_v3_eval_full \
    --case_db diagnosis_model/cause_inference/outputs/case_db \
    --image_root data/detection/coco/_merged/valid \
    --output_dir diagnosis_model/cause_inference/outputs/case_study_v3 \
    --row_indices 79 94 15 133 13
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import numpy as np
import torch
from PIL import Image


CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def get_font(size: int = 12) -> FontProperties:
    return FontProperties(fname=CJK_FONT, size=size)


def alpha_to_rgba(alpha: float, color="red", alpha_max: float = 1.0) -> tuple:
    """Map α ∈ [0, alpha_max] to a color with intensity-controlled stroke."""
    intensity = float(np.clip(alpha / max(alpha_max, 1e-6), 0.0, 1.0))
    base_colors = {
        "red":    (1.0, 0.15, 0.15),
        "orange": (1.0, 0.55, 0.0),
        "yellow": (1.0, 0.85, 0.0),
        "blue":   (0.15, 0.4, 1.0),
        "gray":   (0.6, 0.6, 0.6),
    }
    r, g, b = base_colors.get(color, base_colors["red"])
    # Stroke alpha: low α → faded, high α → bold
    return (r, g, b, 0.3 + 0.7 * intensity)


def render_one_case(
    image_path: Path,
    lesion_boxes_xywh: np.ndarray,    # [N, 4]
    alpha: List[float],               # full α vector: [g, t, l_0, ..., l_{N-1}, ...padding]
    n_lesions: int,
    text_panel_lines: List[tuple],    # list of (label, value, color)
    output_path: Path,
    title: str,
):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # Lesion alpha values
    g_a = alpha[0]
    t_a = alpha[1]
    les_a = alpha[2 : 2 + n_lesions]
    alpha_max = max([g_a, t_a] + list(les_a))

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.05)

    # --- Left: image with bbox overlays ---
    ax_im = fig.add_subplot(gs[0, 0])
    ax_im.imshow(img)
    ax_im.set_xticks([])
    ax_im.set_yticks([])
    ax_im.set_title(title, fontproperties=get_font(13), pad=8)

    for li, (bbox, a_val) in enumerate(zip(lesion_boxes_xywh, les_a)):
        x, y, w, h = [int(v) for v in bbox]
        intensity = float(np.clip(a_val / max(alpha_max, 1e-6), 0.0, 1.0))
        # Stroke width proportional to α
        lw = 1.0 + 4.0 * intensity
        edge_color = alpha_to_rgba(a_val, "red", alpha_max=alpha_max)
        rect = mpatches.Rectangle(
            (x, y), w, h, linewidth=lw, edgecolor=edge_color, facecolor="none",
        )
        ax_im.add_patch(rect)
        # Label
        label = f"L{li}\nα={a_val:.2f}"
        ax_im.text(
            x + 3, max(y - 6, 12), label,
            fontproperties=get_font(11),
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=edge_color[:3], alpha=0.85, edgecolor="none"),
        )

    # Global / text indicators in corners
    g_color = alpha_to_rgba(g_a, "blue", alpha_max=alpha_max)
    t_color = alpha_to_rgba(t_a, "orange", alpha_max=alpha_max)
    ax_im.text(
        8, 22, f"GLOBAL  α={g_a:.2f}",
        fontproperties=get_font(11), color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=g_color[:3], alpha=0.95, edgecolor="none"),
    )
    ax_im.text(
        8, 50, f"TEXT  α={t_a:.2f}",
        fontproperties=get_font(11), color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=t_color[:3], alpha=0.95, edgecolor="none"),
    )

    # --- Right: text panel ---
    ax_tx = fig.add_subplot(gs[0, 1])
    ax_tx.axis("off")
    y_pos = 0.98
    for label, value, color in text_panel_lines:
        ax_tx.text(
            0.0, y_pos, label, fontproperties=get_font(12),
            transform=ax_tx.transAxes, fontweight="bold", color=color,
        )
        y_pos -= 0.05
        # Wrap value at ~40 chars
        words = value.split("\n")
        for line in words:
            for chunk in [line[i:i+38] for i in range(0, len(line), 38)] or [""]:
                ax_tx.text(
                    0.02, y_pos, chunk, fontproperties=get_font(11),
                    transform=ax_tx.transAxes, color="black",
                )
                y_pos -= 0.04
        y_pos -= 0.02

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def classify_cause(text: str) -> str:
    GLOBAL_KEYS = ["水質", "緊迫", "營養", "環境", "水溫", "免疫力", "應激"]
    LESION_KEYS = ["潰瘍", "出血", "紅腫", "寄生蟲", "棉絮", "創傷", "腫脹", "黴", "結痂", "撕裂", "蛀"]
    g = sum(1 for k in GLOBAL_KEYS if k in text)
    l = sum(1 for k in LESION_KEYS if k in text)
    if g > l: return "global-type"
    if l > g: return "lesion-type"
    return "mixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", type=str, required=True)
    ap.add_argument("--case_db", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--row_indices", type=int, nargs="+", required=True,
                    help="Row indices in per_query.jsonl to render")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_cases = torch.load(Path(args.case_db) / "valid_cases.pt", weights_only=False)
    cause_pack = torch.load(Path(args.case_db) / "cause_text_embs.pt", weights_only=False)
    cause_texts = cause_pack["texts"]
    with open(Path(args.eval_dir) / "per_query.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    img_root = Path(args.image_root)

    for idx in args.row_indices:
        if idx >= len(rows) or idx >= len(valid_cases):
            print(f"[skip] index {idx} out of range")
            continue
        r = rows[idx]
        c = valid_cases[idx]
        if not r["predicted_top_n"]:
            print(f"[skip] index {idx}: no predictions")
            continue
        if r["case_id"] != int(c["image_id"]):
            print(f"[warn] index {idx}: row case_id {r['case_id']} != valid_cases image_id {c['image_id']}")

        n = r["lesion_count"]
        p = r["predicted_top_n"][0]
        alpha = p["alpha"]
        pred_text = cause_texts[p["cause_idx"]]
        pred_type = classify_cause(pred_text)

        sem_ranks = r["ranks_per_gamma"]["g=0.50"]["sem_ranks"]
        cl_ranks = r["ranks_per_gamma"]["g=0.50"]["cluster_ranks"]
        gt_min_rank = min(sem_ranks) if sem_ranks else "n/a"

        # Build text panel
        gt_block = "\n".join(f"  • {g}" for g in r["gt_causes"])
        top3_block = "\n".join(
            f"  {ti+1}. (s={pp['score']:.2f}) {cause_texts[pp['cause_idx']][:50]}"
            for ti, pp in enumerate(r["predicted_top_n"][:3])
        )
        alpha_block = (
            f"global α = {alpha[0]:.3f}\n"
            f"text   α = {alpha[1]:.3f}\n"
            + "\n".join(f"L{li} α = {alpha[2+li]:.3f}" for li in range(n))
        )

        text_lines = [
            ("GT causes:", gt_block, "darkgreen"),
            ("Top-3 predictions:", top3_block, "navy"),
            ("Top-1 cause type:", f"  {pred_type}", "purple"),
            ("Top-1 α breakdown:", alpha_block, "darkred"),
            ("GT semantic min-rank:", f"  {gt_min_rank}  (lower = better)", "gray"),
        ]

        title = f"Query #{idx}  ({n} lesions, pool={r['pool_size']})"
        img_path = img_root / r["file_name"]
        if not img_path.exists():
            print(f"[skip] {img_path} not found")
            continue
        out_path = out_dir / f"case_{idx:04d}_{pred_type.replace('-', '_')}.png"
        bboxes = c["lesion_boxes_xywh"].numpy() if hasattr(c["lesion_boxes_xywh"], "numpy") else np.array(c["lesion_boxes_xywh"])
        render_one_case(img_path, bboxes, alpha, n, text_lines, out_path, title)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
