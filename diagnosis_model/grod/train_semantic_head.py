"""Phase A (frozen) — train a Linear semantic head on frozen RF-DETR `hs`.

Plan B teacher: align each lesion's query feature `hs` to the FROZEN raw
SigLIP2 *text* space (symptom captions), via multi-positive contrastive. We
never regress to isolated-crop image embeddings (that would just copy the
non-faithful crop vector). The text anchors are shared with the old crop route,
so the only changed variable is the visual routing: isolated crop (old) vs
RF-DETR decoder query (this).

Pipeline:
  hs [256]  --Linear-->  z [768]  --L2norm-->  contrastive vs frozen SigLIP2
                                                text bank (symptom captions)

A lesion is positive to all caption-bank entries of its own symptom category
(both en+zh), negative to the rest. SigLIP sigmoid (logit_scale/bias) loss.

After training we rebuild a case_db whose `lesion_embs` are replaced by the
head's z (aligned back to the full case via kept_lesion_idx; unmatched lesions
fall back to the original raw lesion_emb so downstream shapes are unchanged).
That rebuilt case_db is what faithfulness_eval.py / eval_ceah.py consume.

Run from repo root (see bottom of file for the exact command)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

VL_CLASSIFIER_DIR = Path(__file__).resolve().parents[1] / "vl_classifier"
if str(VL_CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(VL_CLASSIFIER_DIR))

from common import load_flat_caption_bank, get_text_features  # noqa: E402


# ---------------------------------------------------------------------------
# Frozen SigLIP2 text bank
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_text_bank(model_name: str, symptoms_path: str, device: str,
                    max_length: int = 64):
    """Encode the symptom caption bank with FROZEN raw SigLIP2 text encoder.

    Returns:
      bank_embs   [M, D] L2-normalized
      bank_labels LongTensor[M]  symptom category_id per caption
      logit_scale, logit_bias    SigLIP calibration (frozen)
    """
    from transformers import AutoModel, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    bank = load_flat_caption_bank(symptoms_path, langs=("en", "zh"), text_mode="captions")
    texts = bank.texts
    labels = torch.tensor([int(x) for x in bank.label_ids], dtype=torch.long)

    embs = []
    for i in range(0, len(texts), 256):
        batch = texts[i : i + 256]
        ti = proc(text=batch, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=max_length)
        ti = {k: v.to(device) for k, v in ti.items()}
        f = get_text_features(model, ti["input_ids"], ti.get("attention_mask"))
        embs.append(F.normalize(f.float(), dim=-1).cpu())
    bank_embs = torch.cat(embs, dim=0)

    ls = getattr(model, "logit_scale", None)
    lb = getattr(model, "logit_bias", None)
    logit_scale = (ls.exp().item() if isinstance(ls, torch.Tensor) else 1.0 / 0.07)
    logit_bias = (lb.item() if isinstance(lb, torch.Tensor) else -10.0)
    logit_scale = min(logit_scale, 100.0)

    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return bank_embs, labels, float(logit_scale), float(logit_bias)


# ---------------------------------------------------------------------------
# hs cache -> flat training tensors
# ---------------------------------------------------------------------------

def flatten_hs(cache: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (hs [N, Hd], category [N]) over all matched lesions with a valid
    (>=0) symptom category."""
    hs_list, cat_list = [], []
    for c in cache:
        if c["hs"].numel() == 0:
            continue
        cats = c["lesion_category_id"]
        for j in range(c["hs"].size(0)):
            cat = int(cats[j])
            if cat < 0:
                continue
            hs_list.append(c["hs"][j])
            cat_list.append(cat)
    if not hs_list:
        raise RuntimeError("no matched lesions with valid category in cache")
    return torch.stack(hs_list), torch.tensor(cat_list, dtype=torch.long)


# ---------------------------------------------------------------------------
# Multi-positive sigmoid loss (lesion z vs frozen text bank)
# ---------------------------------------------------------------------------

def multipos_sigmoid_loss(
    z: torch.Tensor, bank: torch.Tensor, z_cat: torch.Tensor,
    bank_cat: torch.Tensor, logit_scale: float, logit_bias: float,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    logits = logit_scale * (z @ bank.t()) + logit_bias        # [B, M]
    pos = (z_cat.view(-1, 1) == bank_cat.view(1, -1))         # [B, M]
    targets = pos.float() * (1 - label_smoothing) + 0.5 * label_smoothing
    element = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pos_m = pos.float()
    neg_m = (~pos).float()
    pos_loss = (element * pos_m).sum(1) / pos_m.sum(1).clamp_min(1.0)
    neg_loss = (element * neg_m).sum(1) / neg_m.sum(1).clamp_min(1.0)
    return (0.5 * (pos_loss + neg_loss)).mean()


@torch.no_grad()
def retrieval_at_k(z: torch.Tensor, bank: torch.Tensor, z_cat: torch.Tensor,
                   bank_cat: torch.Tensor, k: int = 1) -> float:
    """Lesion->category R@k: is the lesion's true category among top-k bank hits?"""
    sims = z @ bank.t()                                       # [B, M]
    topk = sims.topk(min(k, bank.size(0)), dim=1).indices     # [B, k]
    hit = 0
    for i in range(z.size(0)):
        cats = bank_cat[topk[i]]
        if (cats == z_cat[i]).any():
            hit += 1
    return hit / z.size(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train Linear semantic head on frozen RF-DETR hs (plan B).")
    ap.add_argument("--hs_dir", type=str, required=True,
                    help="dir with hs_train.pt / hs_valid.pt from extract_hs.py")
    ap.add_argument("--symptoms", type=str, default="data/raw/symptoms.json")
    ap.add_argument("--model_name", type=str, default="google/siglip2-base-patch16-224")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    hs_dir = Path(args.hs_dir)

    train_cache = torch.load(hs_dir / "hs_train.pt", weights_only=False)
    valid_cache = torch.load(hs_dir / "hs_valid.pt", weights_only=False)
    hs_tr, cat_tr = flatten_hs(train_cache)
    hs_va, cat_va = flatten_hs(valid_cache)
    Hd = hs_tr.size(-1)
    print(f"[data] train lesions={hs_tr.size(0)} valid lesions={hs_va.size(0)} hidden_dim={Hd}")

    bank, bank_cat, logit_scale, logit_bias = build_text_bank(
        args.model_name, args.symptoms, device,
    )
    D = bank.size(-1)
    print(f"[text-bank] {bank.size(0)} captions, dim={D}, "
          f"logit_scale={logit_scale:.2f} logit_bias={logit_bias:.2f}")

    bank = bank.to(device)
    bank_cat = bank_cat.to(device)
    hs_tr, cat_tr = hs_tr.to(device), cat_tr.to(device)
    hs_va, cat_va = hs_va.to(device), cat_va.to(device)

    head = nn.Linear(Hd, D).to(device)
    nn.init.xavier_uniform_(head.weight)
    nn.init.zeros_(head.bias)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    N = hs_tr.size(0)
    best_va_r1 = -1.0
    best_state = None
    t0 = time.time()

    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(N, device=device)
        tot = 0.0
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            z = F.normalize(head(hs_tr[idx]), dim=-1)
            loss = multipos_sigmoid_loss(
                z, bank, cat_tr[idx], bank_cat, logit_scale, logit_bias,
                label_smoothing=args.label_smoothing,
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * idx.size(0)

        head.eval()
        with torch.no_grad():
            z_tr = F.normalize(head(hs_tr), dim=-1)
            z_va = F.normalize(head(hs_va), dim=-1)
            tr_r1 = retrieval_at_k(z_tr, bank, cat_tr, bank_cat, k=1)
            va_r1 = retrieval_at_k(z_va, bank, cat_va, bank_cat, k=1)
            va_r3 = retrieval_at_k(z_va, bank, cat_va, bank_cat, k=3)
        if va_r1 > best_va_r1:
            best_va_r1 = va_r1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:02d} loss={tot/N:.4f} tr_R@1={tr_r1:.3f} "
                  f"va_R@1={va_r1:.3f} va_R@3={va_r3:.3f}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "semantic_head.pt")
    meta = {
        "hidden_dim": Hd, "out_dim": D,
        "best_valid_lesion_R@1": best_va_r1,
        "n_train_lesions": int(hs_tr.size(0)),
        "n_valid_lesions": int(hs_va.size(0)),
        "model_name": args.model_name,
        "symptoms": args.symptoms,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "elapsed_sec": time.time() - t0,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\n[done] best valid lesion->category R@1={best_va_r1:.3f}")
    print(f"[save] semantic_head.pt -> {out_dir}")


if __name__ == "__main__":
    main()
