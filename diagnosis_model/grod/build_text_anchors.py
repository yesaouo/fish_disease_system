"""Build the [num_symptom_cats, D] frozen text-anchor table for loss_semantic.

For each symptom category, encode its captions (en+zh) with the FROZEN raw
SigLIP2 text encoder and average to one L2-normalized anchor. The joint
detector's semantic head is pulled toward these anchors during training.

Categories are indexed 0..C-1 by symptoms.json category id (contiguous), so
anchor_embs[c] is the anchor for symptom_category_id == c.

Run from repo root:
  $PY -m diagnosis_model.grod.build_text_anchors \
      --symptoms data/raw/symptoms.json \
      --out diagnosis_model/grod/outputs/text_anchors.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

VL_CLASSIFIER_DIR = Path(__file__).resolve().parents[1] / "vl_classifier"
if str(VL_CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(VL_CLASSIFIER_DIR))

from common import load_flat_caption_bank, get_text_features  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symptoms", type=str, default="data/raw/symptoms.json")
    ap.add_argument("--model_name", type=str, default="google/siglip2-base-patch16-224")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import AutoModel, AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(args.device).eval()

    bank = load_flat_caption_bank(args.symptoms, langs=("en", "zh"), text_mode="captions")
    labels = [int(x) for x in bank.label_ids]
    num_cats = max(labels) + 1
    D = None

    # encode all captions
    embs = []
    for i in range(0, len(bank.texts), 256):
        batch = bank.texts[i : i + 256]
        ti = proc(text=batch, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=args.max_length)
        ti = {k: v.to(args.device) for k, v in ti.items()}
        f = get_text_features(model, ti["input_ids"], ti.get("attention_mask"))
        embs.append(F.normalize(f.float(), dim=-1).cpu())
    cap_embs = torch.cat(embs, dim=0)
    D = cap_embs.size(-1)

    # average per category, then renormalize
    anchors = torch.zeros(num_cats, D)
    counts = torch.zeros(num_cats)
    for emb, lab in zip(cap_embs, labels):
        anchors[lab] += emb
        counts[lab] += 1
    counts = counts.clamp_min(1).unsqueeze(1)
    anchors = F.normalize(anchors / counts, dim=-1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"anchor_embs": anchors, "num_cats": num_cats, "dim": D,
                "model_name": args.model_name}, out)
    print(f"[done] anchors [{num_cats}, {D}] -> {out}")
    print(f"  per-cat caption counts: {counts.squeeze(1).int().tolist()}")


if __name__ == "__main__":
    main()
