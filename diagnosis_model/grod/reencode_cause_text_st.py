"""Probe (text-encoder swap) — re-encode the unique cause-string table of an
existing case_db with a sentence-transformers model, as a drop-in replacement
for the SigLIP2-text `cause_text_embs.pt`. Texts and their index order are
preserved, so every case's `cause_emb_indices` stay valid.

Symmetric use (cause-vs-cause cosine in Phase 1 scoring + GT match), so we encode
with NO instruction prompt and L2-normalize.

Run from repo root:
  $PY -m diagnosis_model.grod.reencode_cause_text_st \
      --src_embs diagnosis_model/cause_inference/outputs/case_db_raw/cause_text_embs.pt \
      --model microsoft/harrier-oss-v1-270m \
      --out diagnosis_model/grod/outputs/cause_text_embs_harrier.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_embs", type=str, required=True,
                    help="existing cause_text_embs.pt with {'texts','embeddings'}")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--prompt_name", type=str, default=None,
                    help="sentence-transformers prompt name; default None = symmetric/no prompt")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    src = torch.load(args.src_embs, weights_only=False, map_location="cpu")
    texts = src["texts"]
    print(f"[load] {len(texts)} cause strings (src dim={tuple(src['embeddings'].shape)})")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device=args.device)
    print(f"[model] {args.model}  max_seq_len={model.max_seq_length}")

    t0 = time.time()
    embs = model.encode(
        texts, batch_size=args.batch_size, prompt_name=args.prompt_name,
        normalize_embeddings=True, convert_to_tensor=True,
        show_progress_bar=True,
    ).float().cpu()
    print(f"[encode] {tuple(embs.shape)}  norm={embs.norm(dim=-1).mean():.3f}  ({time.time()-t0:.0f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"texts": texts, "embeddings": embs,
                "model": args.model, "prompt_name": args.prompt_name}, out)
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
