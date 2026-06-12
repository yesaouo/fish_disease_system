"""build_raw — the frozen-VLM (A) layer of the artifact pipeline, one command.

Mirrors `db_pipeline`: produces the raw-VLM artifacts that need **no trained model**
— `text_anchors` + `case_db_raw` (encode the dataset with a frozen dual-encoder).
This is the model-swap ablation harness: pass `--vlm` (CLIP / other SigLIP2 sizes)
and a `__<tag>` suffix keeps each variant separate from production. The encode path
(`common.get_image_features/get_text_features`) is already model-agnostic, so any HF
dual-encoder (CLIP/SigLIP) works; the raw layer carries its own dim (the trained B
layer stays bound to the production VLM, so a CLIP raw_db is a Phase-1 ablation).

Covers BUILD_PIPELINE.md steps 1 + 4. Run from repo root:

  $PY -m diagnosis_model.build_pipeline.build_raw                      # production SigLIP2
  $PY -m diagnosis_model.build_pipeline.build_raw --vlm openai/clip-vit-base-patch32
  $PY -m diagnosis_model.build_pipeline.build_raw --dry-run
"""

from __future__ import annotations

import argparse
import re

from diagnosis_model.build_pipeline._util import detection_splits, run_step

DEFAULT_VLM = "google/siglip2-base-patch16-224"


def vlm_tag(vlm: str) -> str:
    """'' for the production VLM, else '__<sanitized last path component>'."""
    if vlm == DEFAULT_VLM:
        return ""
    return "__" + re.sub(r"[^0-9a-zA-Z]+", "-", vlm.split("/")[-1]).strip("-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default=DEFAULT_VLM, help="HF dual-encoder (CLIP/SigLIP) id")
    ap.add_argument("--art", default="data/processed/current/artifacts")
    ap.add_argument("--det", default="data/processed/current/detection")
    ap.add_argument("--sym", default="data/processed/current/symptoms.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tag = vlm_tag(args.vlm)
    anchors = f"{args.art}/models/text_anchors{tag}.pt"
    raw_db = f"{args.art}/db/case_db_raw{tag}"
    splits = detection_splits(args.det)
    print(f"[build_raw] vlm={args.vlm} tag='{tag or '(production)'}' splits={splits}")

    run_step("text_anchors", "diagnosis_model.grod.build_text_anchors",
             ["--model_name", args.vlm, "--symptoms", args.sym, "--out", anchors],
             args.dry_run)

    cdb = ["--vlm_global", args.vlm, "--vlm_lesion", args.vlm, "--raw_lesion",
           "--output_dir", raw_db,
           "--chunk_size", 64, "--img_batch_size", 64, "--text_batch_size", 256]
    for sp in splits:
        cdb += [f"--coco_{sp}", f"{args.det}/{sp}/_annotations.coco.json",
                f"--image_root_{sp}", f"{args.det}/{sp}"]
    run_step("case_db_raw", "diagnosis_model.cause_inference.preprocessing.build_case_database",
             cdb, args.dry_run)

    print(f"\n[done] raw layer for '{args.vlm}':\n  {anchors}\n  {raw_db}")


if __name__ == "__main__":
    main()
