"""build_case_dbs — assemble the retrieval / experiment case-DBs from trained models.

The deterministic glue of BUILD_PIPELINE.md's B layer (steps 5,6,8a-d,S1,S3): given
the trained models, (re)build every case-DB in dependency order. Model *training*
stays out (it's the GPU/hyperparam knob); this only assembles. Idempotent.

Training interleaves the assembly, so steps are grouped into 3 stages with training
gates between them — use `--from/--to` to stop before a gate:

  stage1  z_joint, dino_global              (needs rfdetr + joint trained)
     gate ↳ train distilled_global_rawP
  stage2  case_db_jointDistRaw → swap_global → case_db_jointDistRawP,
          candidate_pool, teacher_table, soft_inputs,  (needs distilled_global)
          + gating defaults: disease_perquery → lesion_threshold + thresholds.json
            + neck disease head (production 健/病 + 病灶門檻; need joint+distilled only)
     gate ↳ train encoder_grod, ceah_jointDistRawP, encoder_grod_soft, ceah_grod_soft
  stage3  soft bank_z_soft                   (needs encoder_grod_soft)

Typical from-scratch bring-up (run from repo root):
  build_raw
  〔train rfdetr, joint〕
  build_case_dbs --to stage1
  〔train distilled_global_rawP〕
  build_case_dbs --from stage2 --to stage2
  〔train encoder_grod / ceah_jointDistRawP / encoder_grod_soft / ceah_grod_soft〕
  build_case_dbs --from stage3
  (the production gating defaults — thresholds.json + neck disease head — are built
   inside stage2, not separately.)

  $PY -m diagnosis_model.build_pipeline.build_case_dbs --to stage1
  $PY -m diagnosis_model.build_pipeline.build_case_dbs --dry-run
"""

from __future__ import annotations

import argparse

from diagnosis_model.build_pipeline._util import casedb_splits, run_step

STAGES = ["stage1", "stage2", "stage3"]


def build_steps(art, det, splits):
    """Return [(stage, name, module, args), ...] in dependency order."""
    M, DB = f"{art}/models", f"{art}/db"
    raw = f"{DB}/case_db_raw"
    anchors = f"{M}/text_anchors.pt"
    joint = f"{M}/joint_rfdetr/checkpoint_best_regular.pth"
    rfdetr = f"{M}/rfdetr/checkpoint_best_total.pth"
    distilled = f"{M}/distilled_global_rawP"
    gsd = f"{distilled}/global_embed_state_dict.pt"
    jdr, jdrp = f"{DB}/case_db_jointDistRaw", f"{DB}/case_db_jointDistRawP"
    soft_in = f"{DB}/soft_inputs"
    soft_enc = f"{M}/encoder_grod_soft/best_encoder.pt"
    dperq, dhead = f"{DB}/disease_perquery", f"{M}/disease_head"

    steps = [
        ("stage1", "z_joint", "diagnosis_model.grod.extract_z_joint",
         ["--case_db_dir", raw, "--joint_ckpt", joint, "--anchors", anchors,
          "--image_root", det, "--output_dir", f"{DB}/z_joint", "--splits", *splits]),
    ]
    for sp in splits:
        steps.append(("stage1", f"dino_global:{sp}", "diagnosis_model.grod.extract_dino_global",
                      ["--case_db_dir", raw, "--split", sp, "--det_ckpt", rfdetr,
                       "--image_root", f"{det}/{sp}", "--output_dir", f"{DB}/dino_global"]))

    steps += [
        ("stage2", "rebuild_case_db", "diagnosis_model.grod.rebuild_case_db",
         ["--src_case_db", raw, "--hs_dir", f"{DB}/z_joint", "--from_joint", "--out_case_db", jdr]),
        ("stage2", "swap_global", "diagnosis_model.grod.build_case_db_swap_global",
         ["--src_db", jdr, "--global_dir", distilled, "--global_prefix", "distilled_global",
          "--dst_db", jdrp]),
        ("stage2", "candidate_pool",
         "diagnosis_model.cause_inference.preprocessing.build_train_candidate_pool",
         ["--case_db_dir", jdrp, "--output_path", f"{jdrp}/train_candidate_pool.pt",
          "--top_k_cases", 20, "--alpha_global", 0.25, "--beta_lesion", 0.75,
          "--lesion_match", "max_mean", "--semantic_threshold", 0.95]),
        ("stage2", "teacher_table",
         "diagnosis_model.cause_inference.preprocessing.build_teacher_table",
         ["--case_db_dir", jdrp, "--output_path", f"{jdrp}/teacher_train_train.pt",
          "--alpha_global", 0.25, "--beta_lesion", 0.75, "--lesion_match", "max_mean"]),
        ("stage2", "soft_inputs", "diagnosis_model.grod.extract_soft_inputs",
         ["--joint_ckpt", joint, "--global_sd", gsd,
          "--anchors", anchors, "--case_db_dir", jdrp, "--img_root", det, "--out_dir", soft_in]),
        # production gating defaults (health verdict + thresholds) — need joint + distilled only
        ("stage2", "disease_perquery", "diagnosis_model.grod.extract_disease_perquery",
         ["--joint_ckpt", joint, "--global_sd", gsd, "--anchors", anchors,
          "--det_root", det, "--out_dir", dperq]),
        ("stage2", "lesion_threshold", "diagnosis_model.grod.compute_lesion_threshold",
         ["--cache_dir", dperq, "--out", f"{dhead}/lesion_threshold.json"]),
        ("stage2", "thresholds", "diagnosis_model.grod.calibrate_thresholds",
         ["--cache", f"{dperq}/train.pt", "--out", "data/processed/current/thresholds.json"]),
        ("stage2", "abstain_head", "diagnosis_model.grod.train_abstain_head",
         ["--feat", "dino_neck", "--det_root", det, "--joint_ckpt", joint,
          "--global_sd", gsd, "--anchors", anchors,
          "--out", f"{dhead}/neck_disease_head.pt", "--cache", f"{dhead}/neck_features.pt"]),
        ("stage3", "soft_bank", "diagnosis_model.grod.build_soft_bank",
         ["--soft_dir", soft_in, "--encoder_ckpt", soft_enc,
          "--out", f"{M}/encoder_grod_soft/bank_z_soft.pt"]),
    ]
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="data/processed/current/artifacts")
    ap.add_argument("--det", default="data/processed/current/detection")
    ap.add_argument("--from", dest="from_stage", choices=STAGES, default="stage1")
    ap.add_argument("--to", dest="to_stage", choices=STAGES, default="stage3")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    splits = casedb_splits(f"{args.art}/db/case_db_raw")
    if not splits:
        raise SystemExit(f"[build_case_dbs] no case_db_raw splits under {args.art}/db/case_db_raw "
                         "— run build_raw first.")
    lo, hi = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
    print(f"[build_case_dbs] splits={splits}  stages {args.from_stage}..{args.to_stage}")

    for stage, name, module, sargs in build_steps(args.art, args.det, splits):
        if lo <= STAGES.index(stage) <= hi:
            run_step(f"{stage}/{name}", module, sargs, args.dry_run)

    print(f"\n[done] build_case_dbs {args.from_stage}..{args.to_stage}")


if __name__ == "__main__":
    main()
