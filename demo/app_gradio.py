"""Gradio demo — three GROD-family pipelines on the new dataset tree.

Thin Gradio UI over diagnosis_model.grod.pipeline (the shared inference + render
core, also used by the FastAPI REST service). This file holds only UI wiring:
handlers, the timing/params markdown, and build_ui.

Modes (UI dropdown, lazy-loaded, live-switchable):
  base       conventional separated baseline: RF-DETR + raw SigLIP2 global +
             standard-finetuned SigLIP2 lesion crops → DeepSets → dense retrieval → CEAH
  grod_soft  soft per-query weights (default)
  grod       same soft model + artifacts as grod_soft, but the per-query weights are
             hard-gated to {0,1} at display_thresh (objectness > τ) — the bytes-exact
             hard-gate degenerate of the soft path (input-selection ablation).

Display (detailed mode only):
  detection · per-lesion classification cards · retrieved cases · top-N causes + α
  + per-module parameter counts
  + per-stage latency averaged over N CUDA-synced runs (warm-up dropped)

Thresholds (two decoupled, dataset-calibrated — calibrate_thresholds.py writes
data/processed/current/thresholds.json; the sliders default to it):
  abstain_thresh : 健/病判定 (max objectness, Youden-optimal).
  display_thresh : 顯示/選框 (per-query, F2 recall-leaning) → feeds the ②
                   classification cards.
  Step ① shows the A objectness heatmap (all queries splatted), not boxes.

Run from repo root:
  /home/lab603/anaconda3/envs/SDM/bin/python demo/app_gradio.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import gradio as gr
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from diagnosis_model.grod.pipeline import (
    DEVICE, DET_VALID, ABSTAIN_DEFAULT, DISPLAY_DEFAULT,
    load_shared, get_pipeline, encode_text_slot,
    render_heatmap_image, make_lesion_card, make_missing_case_placeholder,
    make_alpha_attribution_image, make_alpha_breakdown_chart, _aggregate_timings,
)

MAX_TOPN_BUTTONS = 10
N_TIMING_RUNS = 20          # detailed mode: timed runs (first one dropped as warm-up)


# ---------------------------------------------------------------------------
# Detail tables (params + timing)
# ---------------------------------------------------------------------------

def _params_md(pipe) -> str:
    rows = ["", "**模組參數量**", "", "| 模組 | 參數量 |", "|---|---:|"]
    total = 0
    for name, n in pipe.params.items():
        if not name.startswith("  "):
            total += n
        rows.append(f"| {name} | {n:,} |")
    rows.append(f"| **總計 (不含子項重複)** | **{total:,}** |")
    return "\n".join(rows)


def _timing_md(stage_stats, n_runs) -> str:
    rows = ["", f"**各階段延遲 (CUDA-synced, {n_runs} 次平均, 丟首次 warm-up)**", "",
            "| 階段 | mean ms | ±std |", "|---|---:|---:|"]
    tot_mean = 0.0
    for label, mean, std in stage_stats:
        tot_mean += mean
        rows.append(f"| {label} | {mean:.2f} | {std:.2f} |")
    rows.append(f"| **總計** | **{tot_mean:.2f}** | — |")
    if tot_mean > 0:
        rows.append(f"\n吞吐 ≈ **{1000.0 / tot_mean:.1f} img/s**（單張、batch=1）")
    return "\n".join(rows)


# ===========================================================================
# Gradio handlers
# ===========================================================================

def _empty_buttons():
    return [gr.update(visible=False) for _ in range(MAX_TOPN_BUTTONS)]


def handler_run(mode, image, text, abstain_thresh, display_thresh, top_k_cases,
                top_n_causes):
    if image is None:
        return (None, None, None, "請先上傳或選一張圖。", [], "", None, None, "",
                *_empty_buttons())
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    pipe = get_pipeline(mode)
    text_emb = encode_text_slot(text)
    top_k, top_n = int(top_k_cases), int(top_n_causes)
    ath, dth = float(abstain_thresh), float(display_thresh)

    res = pipe.infer_rich(image, text_emb, top_k, top_n, ath, dth)

    detailed = True
    timing_md = ""
    if detailed and res["n_lesions"] > 0:
        runs = [res["timings"]]
        for _ in range(N_TIMING_RUNS):
            runs.append(pipe.infer_rich(image, text_emb, top_k, top_n, ath, dth)["timings"])
        stats = _aggregate_timings(runs[1:])    # drop warm-up (first)
        timing_md = _timing_md(stats, N_TIMING_RUNS)

    det_img = render_heatmap_image(res["image_pil"], res.get("obj_all"), res.get("boxes_all"))
    gallery = [(make_lesion_card(l), f"L{l['idx']}: {l['cls']['label_zh']}")
               for l in res["lesions"]]

    if res.get("abstain"):
        info = f"🟢 **abstain：最高 objectness 未達 abstain_thresh={ath:.2f}，判定為健康**，不進行病因推論。"
    elif not res["lesions"]:
        info = "**display_thresh 下未顯示病灶。**"
    else:
        thr = f"abstain={ath:.2f} / display={dth:.2f}"
        info = (f"模式 **{mode}** ｜ 顯示 **{res['n_lesions']}** 個病灶（{thr}）｜ "
                f"候選病因池 = **{res['pool_size']}** ｜ top-{len(res['top_n'])} 病因"
                + ("（含文字證據）" if res["text_used"] else "（vision-only）"))
    if detailed:
        info += "\n\n" + _params_md(pipe)
        if timing_md:
            info += "\n\n" + timing_md

    btns = []
    for i in range(MAX_TOPN_BUTTONS):
        if i < len(res["top_n"]):
            r = res["top_n"][i]; txt = r["text"][:48] + "…" if len(r["text"]) > 50 else r["text"]
            sup = f"  {r['support']}例" if r.get("support") else ""
            btns.append(gr.update(value=f"#{r['rank']}  s={r['score']:.2f}{sup}  {txt}", visible=True))
        else:
            btns.append(gr.update(visible=False))

    retr_gallery = []
    for i, r in enumerate(res["retrieved"], 1):
        cap = f"#{i}  sim={r['similarity']:.3f}"
        retr_gallery.append((r["image_path"], cap) if r.get("image_path")
                            else (make_missing_case_placeholder(i, r["similarity"]), cap + " (missing)"))

    state = {"image_pil": res["image_pil"], "boxes": [l["bbox_xywh"] for l in res["lesions"]],
             "n_lesions": res["n_lesions"], "top_n": res["top_n"], "text_used": res["text_used"]}
    return (det_img, gallery, state, info, retr_gallery, "", None, None, "", *btns)


def handler_select(idx, state):
    if state is None or idx >= len(state.get("top_n", [])):
        return None, None, ""
    r = state["top_n"][idx]; n = state["n_lesions"]
    show_text = state.get("text_used", False)
    bbox = np.array(state["boxes"], dtype=np.float32)
    ai = make_alpha_attribution_image(state["image_pil"], bbox, r["alpha"], n,
                                      r["text"], r["score"], show_text)
    bar = make_alpha_breakdown_chart(r["alpha"], n, show_text)
    members = r.get("members") or []
    sup_md = f"- 支持度：**{r['support']}** 個相似病例指向此病因\n" if r.get("support") else ""
    fold_md = ""
    if len(members) > 1:
        extra = "\n".join(f"  - {m[:60]}" for m in members[1:6])
        more = f"\n  - …+{len(members) - 6}" if len(members) > 6 else ""
        fold_md = f"\n\n<details><summary>已聚合 {len(members) - 1} 條相近病因</summary>\n\n{extra}{more}\n</details>"
    explain = (f"### Top-{r['rank']} 病因\n**{r['text']}**\n\n"
               + sup_md
               + f"- CEAH score: **{r['score']:.3f}**\n"
               + fold_md)
    return ai, bar, explain


# ===========================================================================
# UI
# ===========================================================================
DESCRIPTION = """
# 🐟 GROD 魚病診斷流水線 demo

三個流水線，UI 下拉即時切換：

| 模式 | 架構 |
|---|---|
| **base** | 常規分離式對照組：偵測 (RF-DETR) + 凍結 SigLIP2 全域 + 微調 SigLIP2 病灶 → Aggregator (DeepSets) → dense 檢索 → CEAH |
| **grod_soft** | Backbone(DINOv2)+DETR forward → 四 head（box / objectness / semantic / global）→ Aggregator (DeepSets, per-query 連續權重 w) → dense 檢索 → CEAH（預設） |
| **grod** | 同 grod_soft 同一顆模型/artifacts，僅把 per-query 權重在 display_thresh 二值化成 {0,1}（硬閘）。DeepSets/CEAH 在 {0,1} 權重下 bytes-exactly 退化成硬路徑 → grod vs grod_soft ＝同一模型上的輸入選取消融 |
"""


def build_ui():
    with gr.Blocks(title="GROD Fish Disease Demo") as demo:
        gr.Markdown(DESCRIPTION)
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Dropdown(["base", "grod", "grod_soft"], value="grod_soft", label="模式")
                inp_image = gr.Image(label="魚體輸入圖", type="pil", height=300)
                inp_text = gr.Textbox(label="選填：文字描述", lines=2,
                                      placeholder="例：體表潰瘍、紅腫，疑似感染；留空＝vision-only")
                with gr.Accordion("可調參數", open=False):
                    sld_abstain = gr.Slider(0.1, 0.9, value=ABSTAIN_DEFAULT, step=0.01,
                                            label="abstain_thresh")
                    sld_display = gr.Slider(0.1, 0.9, value=DISPLAY_DEFAULT, step=0.01,
                                            label="display_thresh")
                    sld_topk = gr.Slider(5, 50, value=20, step=1, label="top_k_cases")
                    sld_topn = gr.Slider(1, MAX_TOPN_BUTTONS, value=5, step=1, label="top_n_causes")
                btn_run = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                out_det = gr.Image(label="① 異常熱力圖", type="pil", height=300)
                out_info = gr.Markdown()

        gr.Markdown("---\n## ② 病灶分類（rep · symptom anchor）")
        out_gallery = gr.Gallery(columns=1, height=520, show_label=False, object_fit="contain")
        gr.Markdown("---\n## ③ 檢索到的相似 case")
        out_retr = gr.Gallery(columns=5, height=220, show_label=True, object_fit="contain")
        gr.Markdown("---\n## ④ Top-N 病因 + α 歸因（點按鈕看解釋）")
        buttons = []
        with gr.Row():
            with gr.Column(scale=1):
                for _ in range(MAX_TOPN_BUTTONS):
                    buttons.append(gr.Button(value="", visible=False, size="sm"))
                out_retr_md = gr.Markdown()
            with gr.Column(scale=2):
                out_explain = gr.Markdown()
                out_alpha = gr.Image(label="α attribution", type="pil", height=400)
                out_bar = gr.Image(label="α breakdown", type="pil", height=230)

        run_outputs = [out_det, out_gallery, state, out_info, out_retr,
                       out_retr_md, out_alpha, out_bar, out_explain, *buttons]
        btn_run.click(handler_run,
                      [mode, inp_image, inp_text, sld_abstain, sld_display, sld_topk,
                       sld_topn],
                      run_outputs)
        for i, b in enumerate(buttons):
            b.click(lambda st, idx=i: handler_select(idx, st), [state],
                    [out_alpha, out_bar, out_explain])

        ex = _diseased_examples(8)
        if ex:
            gr.Examples(examples=ex, inputs=[inp_image], examples_per_page=8,
                        label="範例（valid，含病灶的魚）")
    return demo


def _diseased_examples(n=8) -> List[List[str]]:
    """Pick valid images that actually have lesion boxes (most-annotated first),
    so the examples exercise the full pipeline rather than abstaining on healthy fish."""
    coco_path = DET_VALID / "_annotations.coco.json"
    if not coco_path.exists():
        return []
    coco = json.load(open(coco_path, encoding="utf-8"))
    id2fn = {im["id"]: im["file_name"] for im in coco["images"]}
    cnt = Counter(a["image_id"] for a in coco["annotations"])
    out = []
    for iid, _ in cnt.most_common():
        fn = id2fn.get(iid)
        if fn and (DET_VALID / fn).exists():
            out.append([str(DET_VALID / fn)])
        if len(out) >= n:
            break
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preload", nargs="*", default=[],
                    help="modes to load at startup (default: lazy on first use)")
    args = ap.parse_args()
    print(f"[init] device={DEVICE}")
    load_shared()
    for m in args.preload:
        get_pipeline(m)
    build_ui().queue(default_concurrency_limit=1).launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
