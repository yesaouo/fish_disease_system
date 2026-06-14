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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from diagnosis_model.grod.pipeline import (
    DEVICE, ABSTAIN_DEFAULT, DISPLAY_DEFAULT, _BUILDERS,
    load_shared, get_pipeline, encode_text_slot,
    render_heatmap_image, make_missing_case_placeholder,
    make_alpha_attribution_image, make_alpha_breakdown_chart,
)

# GPU is single — serialize inference so concurrent requests don't race the device
# (mirrors the demo's default_concurrency_limit=1).
_GPU_LOCK = threading.Lock()

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

    boxes = [l["bbox_xywh"] for l in res["lesions"]]
    n = res["n_lesions"]
    show_text = res["text_used"]
    for c in res["top_n"]:
        out["causes"].append({
            "rank": c["rank"], "text": c["text"], "score": float(c["score"]),
            "support": c.get("support"), "members": c.get("members", []),
            "alpha": [float(a) for a in c["alpha"]],
            "attribution": _b64(make_alpha_attribution_image(
                res["image_pil"], boxes, c["alpha"], n, c["text"], c["score"], show_text)),
            "breakdown": _b64(make_alpha_breakdown_chart(c["alpha"], n, show_text)),
        })
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
    top_k_cases: int = Form(20),
    top_n_causes: int = Form(5),
):
    if mode not in _BUILDERS:
        raise HTTPException(400, f"unknown mode '{mode}'; choose {list(_BUILDERS.keys())}")
    try:
        img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
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
    return payload


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
