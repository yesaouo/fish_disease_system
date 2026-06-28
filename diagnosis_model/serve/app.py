"""FastAPI diagnosis service — REST front for diagnosis_model.grod.pipeline.

Same inference core as the Gradio demo (get_pipeline → infer_rich); this just
serializes the result dict to JSON with all figures rendered server-side as
base64 PNG data-URIs, so the React report page is pure layout.

Run from repo root (SDM env, GPU resident):
  /home/lab603/anaconda3/envs/SDM/bin/python -m diagnosis_model.serve.app
  (--preload grod_soft to warm a mode at startup; --port 8900 default)

Endpoints:
  GET  /health           → {"status","device","loaded"}
  GET  /modes            → ["base","grod","grod_soft"]
  POST /diagnose         → multipart(image, text, mode, thresholds, top_k/n) → report JSON

The 處置建議 / 專家覆核 report blocks (Table 4) are intentionally not produced
here — they are empty fields the React layer renders, matching the thesis framing
that the expert-feedback loop is future work.
"""

from __future__ import annotations

import argparse
import base64
import io
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from collections import OrderedDict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageOps

from diagnosis_model.grod.pipeline import (
    DEVICE, ABSTAIN_DEFAULT, DISPLAY_DEFAULT, _BUILDERS,
    load_shared, get_pipeline, encode_text_slot,
    render_heatmap_image, render_detection_image, make_missing_case_placeholder,
    make_alpha_breakdown_chart, make_combined_alpha_chart,
)

# GPU is single — serialize inference so concurrent requests don't race the device
# (mirrors the demo's default_concurrency_limit=1).
_GPU_LOCK = threading.Lock()

# Cache recent report payloads so the PDF endpoint can render from case_id alone
# (avoids round-tripping the large base64 JSON back through the upload hop → 413).
_REPORT_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_REPORT_CACHE_MAX = 64
_CACHE_LOCK = threading.Lock()

app = FastAPI(title="GROD Fish Disease Diagnosis Service")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _params_payload(pipe) -> dict:
    """{modules:[{name,count}], total} — total skips indented sub-rows (demo convention)."""
    mods, total = [], 0
    for name, n in pipe.params.items():
        if not name.startswith("  "):
            total += n
        mods.append({"name": name, "count": int(n)})
    return {"modules": mods, "total": int(total)}


def _serialize(res: dict, pipe, meta: dict) -> dict:
    """result dict (+ rendered figures) → JSON-safe report payload."""
    out = {
        "meta": meta,
        "abstain": bool(res.get("abstain", False)),
        "pool_size": int(res.get("pool_size", 0)),
        "text_used": bool(res.get("text_used", False)),
        "n_lesions": int(res.get("n_lesions", 0)),
        "image_size": [int(res["image_pil"].width), int(res["image_pil"].height)],
        "heatmap": _b64(render_heatmap_image(res["image_pil"], res.get("obj_all"),
                                             res.get("boxes_all"))),
        "params": _params_payload(pipe),
        "timings": [{"stage": s, "ms": round(float(ms), 3)} for s, ms in res.get("timings", [])],
        "lesions": [],
        "retrieved": [],
        "causes": [],
    }
    # 原圖（健康時無框＝純送檢影像，異常時疊紅框＋編號）；PDF 兩種報告都用得到
    out["boxes_image"] = _b64(render_detection_image(res["image_pil"], res.get("lesions", [])))
    if out["abstain"]:
        return out

    for les in res["lesions"]:
        out["lesions"].append({
            "idx": les["idx"], "bbox_xywh": [float(v) for v in les["bbox_xywh"]],
            "det_score": float(les["det_score"]),
            "label_id": les["cls"]["label_id"], "label_zh": les["cls"]["label_zh"],
            "cls_score": float(les["cls"]["score"]),
            "top_k": [{"label_zh": it["label_zh"], "score": float(it["score"]),
                       "prob": float(it["prob"])}
                      for it in les["cls"]["top_k"][:3]],
            "crop": _b64(les["crop"]),
        })

    for i, r in enumerate(res["retrieved"], 1):
        if r.get("image_path"):
            img = Image.open(r["image_path"])
        else:
            img = make_missing_case_placeholder(i, r["similarity"])
        out["retrieved"].append({
            "rank": i, "file_name": r["file_name"],
            "similarity": float(r["similarity"]),
            "exists": r.get("image_path") is not None,
            "image": _b64(img),
        })

    n = res["n_lesions"]
    show_text = res["text_used"]
    for c in res["top_n"]:
        out["causes"].append({
            "rank": c["rank"], "text": c["text"], "score": float(c["score"]),
            "support": c.get("support"), "members": c.get("members", []),
            "support_cases": c.get("support_cases", []),
            "alpha": [float(a) for a in c["alpha"]],
            "breakdown": _b64(make_alpha_breakdown_chart(c["alpha"], n, show_text)),
        })

    # 所有病因合成一張證據貢獻堆疊圖，供 PDF（三）
    if out["causes"]:
        out["causes_breakdown"] = _b64(make_combined_alpha_chart(out["causes"], n, show_text))
    return out


@app.get("/health")
def health():
    from diagnosis_model.grod import pipeline as P
    return {"status": "ok", "device": DEVICE, "loaded": list(P._PIPELINES.keys())}


@app.get("/modes")
def modes():
    return list(_BUILDERS.keys())


@app.post("/diagnose")
def diagnose(
    image: UploadFile = File(...),
    text: str = Form(""),
    mode: str = Form("grod_soft"),
    abstain_thresh: float = Form(ABSTAIN_DEFAULT),
    display_thresh: float = Form(DISPLAY_DEFAULT),
    top_k_cases: int = Form(3),
    top_n_causes: int = Form(6),   # 顯示上限；實際數量由 cause_score_thresh (τ) 自適應決定
):
    if mode not in _BUILDERS:
        raise HTTPException(400, f"unknown mode '{mode}'; choose {list(_BUILDERS.keys())}")
    try:
        # exif_transpose bakes the EXIF orientation into the pixels so server-side
        # work (bboxes, image_size, heatmap) matches the upright image the browser
        # shows from URL.createObjectURL (which applies EXIF). Without this, phone
        # photos tagged "rotate 90°" come out with boxes/heatmap mis-aligned.
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image.file.read()))).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"cannot read image: {e}")

    case_id = f"FD-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:6]}"
    meta = {
        "case_id": case_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "text": text,
        "thresholds": {"abstain": float(abstain_thresh), "display": float(display_thresh)},
    }
    with _GPU_LOCK:
        pipe = get_pipeline(mode)
        text_emb = encode_text_slot(text)
        res = pipe.infer_rich(img, text_emb, int(top_k_cases), int(top_n_causes),
                              float(abstain_thresh), float(display_thresh))
        payload = _serialize(res, pipe, meta)
    with _CACHE_LOCK:
        _REPORT_CACHE[case_id] = payload
        _REPORT_CACHE.move_to_end(case_id)
        while len(_REPORT_CACHE) > _REPORT_CACHE_MAX:
            _REPORT_CACHE.popitem(last=False)
    return payload


@app.get("/report/pdf")
def report_pdf(case_id: str):
    """以 case_id 從快取的報告渲染固定模板 PDF（不重跑推論，純排版）。"""
    from diagnosis_model.serve.report_pdf import render_report_pdf

    with _CACHE_LOCK:
        payload = _REPORT_CACHE.get(case_id)
    if payload is None:
        raise HTTPException(404, f"report for case_id '{case_id}' not found or expired")
    try:
        pdf = render_report_pdf(payload)
    except Exception as e:
        raise HTTPException(400, f"cannot render report: {e}")
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{case_id}.pdf"'},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--preload", nargs="*", default=[],
                    help="modes to load at startup (default: lazy on first request)")
    args = ap.parse_args()
    print(f"[init] device={DEVICE}")
    load_shared()
    for m in args.preload:
        get_pipeline(m)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
