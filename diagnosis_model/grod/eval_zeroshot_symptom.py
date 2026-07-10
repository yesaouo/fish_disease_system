"""Zero-shot per-lesion symptom classification for a FROZEN image-text encoder.

For each GT lesion box we crop it, encode the crop with a frozen VL image encoder,
and classify against per-symptom text anchors by cosine argmax — no training. This
mirrors the anchor protocol of `eval_lesion_symptom_cls.py` (which does the same
for the DISTILLED joint model's z), so a candidate encoder's zero-shot number is
directly comparable to its distilled number.

Given-match by construction (every GT box is a known lesion region), so match_rate
is 1.0 and "given match" == "end-to-end".

Anchors are built from `symptoms.json` captions (must be the SAME symptom taxonomy
tree as the detection data's `symptom_category_id`).

Run from repo root (SDM env):
  $PY -m diagnosis_model.grod.eval_zeroshot_symptom --vlm google/siglip2-base-patch16-224
  $PY -m diagnosis_model.grod.eval_zeroshot_symptom --vlm openai/clip-vit-base-patch16 --langs en
  $PY -m diagnosis_model.grod.eval_zeroshot_symptom --vlm BAAI/AltCLIP   # trust_remote_code auto
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

VL_DIR = Path(__file__).resolve().parents[1] / "vl_classifier"
sys.path.insert(0, str(VL_DIR))
from common import load_flat_caption_bank, get_image_features, get_text_features  # noqa: E402


def scaled_rect_crop(img: Image.Image, bbox_xywh, scale: float = 1.0) -> Image.Image:
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2, y + h / 2
    w, h = w * scale, h * scale
    x1, y1 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    x2, y2 = min(img.width, int(cx + w / 2)), min(img.height, int(cy + h / 2))
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return img.crop((x1, y1, x2, y2))


def build_anchors(model, proc, symptoms, langs, device):
    b = load_flat_caption_bank(symptoms, langs=langs, text_mode="captions")
    tok = proc(text=b.texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        f = F.normalize(get_text_features(model, tok["input_ids"], tok.get("attention_mask")).float(), dim=-1).cpu()
    byc = defaultdict(list)
    for i, c in enumerate(b.label_ids):
        byc[int(c)].append(f[i])
    C = max(byc) + 1
    A = torch.zeros(C, f.size(1))
    for c, v in byc.items():
        A[c] = F.normalize(torch.stack(v).mean(0), dim=-1)   # per-category mean anchor
    return A


def main():
    ap = argparse.ArgumentParser()
    DET = "data/processed/current/detection"
    ap.add_argument("--vlm", required=True, help="HF image-text model id or finetuned dir")
    ap.add_argument("--symptoms", default="data/processed/current/symptoms.json")
    ap.add_argument("--coco", default=f"{DET}/valid/_annotations.coco.json")
    ap.add_argument("--image_root", default=f"{DET}/valid")
    ap.add_argument("--langs", default="both", choices=["en", "zh", "both"],
                    help="anchor caption language(s); english-only for english-native CLIP, etc.")
    ap.add_argument("--scale", type=float, default=1.0, help="bbox crop scale")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    langs = ("en", "zh") if args.langs == "both" else (args.langs,)

    try:
        model = AutoModel.from_pretrained(args.vlm).to(args.device).eval()
        proc = AutoProcessor.from_pretrained(args.vlm)
    except Exception:
        model = AutoModel.from_pretrained(args.vlm, trust_remote_code=True).to(args.device).eval()
        proc = AutoProcessor.from_pretrained(args.vlm, trust_remote_code=True)

    A = build_anchors(model, proc, args.symptoms, langs, args.device).to(args.device)
    cand = list(range(1, A.size(0)))                                # exclude 0 = healthy
    ct = torch.tensor(cand, device=args.device)

    coco = json.load(open(args.coco))
    id2fn = {im["id"]: im["file_name"] for im in coco["images"]}
    by = defaultdict(list)
    for a in coco["annotations"]:
        if a.get("bbox") and len(a["bbox"]) == 4:
            by[a["image_id"]].append((a["bbox"], a["symptom_category_id"]))

    crops, cats = [], []
    for iid, items in by.items():
        fn = id2fn.get(iid)
        if fn is None:
            continue
        img = Image.open(Path(args.image_root) / fn).convert("RGB")
        for b, c in items:
            crops.append(scaled_rect_crop(img, b, args.scale))
            cats.append(c)
        img.close()
    cats = torch.tensor(cats)

    feats = []
    with torch.no_grad():
        for i in range(0, len(crops), args.batch_size):
            px = proc(images=crops[i:i + args.batch_size], return_tensors="pt")["pixel_values"].to(args.device)
            feats.append(F.normalize(get_image_features(model, px).float(), dim=-1).cpu())
    feats = torch.cat(feats)

    sims = feats.to(args.device) @ A[ct].t()
    ranked = ct[sims.argsort(dim=1, descending=True)].cpu()          # [N, len(cand)]
    top1 = (ranked[:, 0] == cats).float().mean().item()
    top3 = torch.tensor([cats[i] in ranked[i, :3].tolist() for i in range(len(cats))]).float().mean().item()

    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for i in range(len(cats)):
        p, t = int(ranked[i, 0]), int(cats[i])
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1; fn[t] += 1
    f1s = []
    for c in cand:
        support = tp[c] + fn[c]
        if support == 0 and fp[c] == 0:
            continue
        pr = tp[c] / max(1, tp[c] + fp[c])
        rc = tp[c] / max(1, tp[c] + fn[c])
        if support > 0:
            f1s.append(2 * pr * rc / max(1e-9, pr + rc))
    mf1 = sum(f1s) / max(1, len(f1s))

    print(f"\n=== zero-shot symptom cls: {args.vlm} | anchors langs={langs} | dim={A.size(1)} ===")
    print(f"GT lesions = {len(cats)}  (given-match by construction)")
    print(f"top-1 = {top1:.4f}   top-3 = {top3:.4f}   macro-F1 = {mf1:.4f}")


if __name__ == "__main__":
    main()
