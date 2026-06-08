"""Probe A — build a case_db that is identical to case_db_raw except `global_emb`
is replaced by the RF-DETR backbone global vector (from extract_dino_global.py).

Single-variable swap: lesion_embs / causes / cause_emb_indices / boxes untouched.
Heavy shared files (teacher table, candidate pool, cause_text_embs, meta) are
symlinked so we reuse the SigLIP2 teacher as the distillation ceiling and don't
duplicate GBs on disk.

Run from repo root (production: swap in the distilled global -> jointDistRawP):
  $PY -m diagnosis_model.grod.build_case_db_swap_global \
      --src_db data/processed/current/artifacts/db/case_db_jointDistRaw \
      --global_dir data/processed/current/artifacts/models/distilled_global_rawP \
      --global_prefix distilled_global \
      --dst_db data/processed/current/artifacts/db/case_db_jointDistRawP
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def swap_split(src_db: Path, dst_db: Path, gdir: Path, prefix: str, split: str):
    cases = torch.load(src_db / f"{split}_cases.pt", weights_only=False)
    pack = torch.load(gdir / f"{prefix}_{split}.pt", weights_only=False)
    g = pack["global"]                                   # [N, D]
    fns = pack["file_names"]
    assert len(cases) == g.size(0), f"{split}: {len(cases)} cases vs {g.size(0)} globals"
    # order sanity: extract preserved case order, verify file_names align
    mism = [i for i, c in enumerate(cases) if c["file_name"] != fns[i]]
    assert not mism, f"{split}: file_name order mismatch at {mism[:5]}"
    for i, c in enumerate(cases):
        c["global_emb"] = g[i].clone()
    torch.save(cases, dst_db / f"{split}_cases.pt")
    print(f"[{split}] swapped global_emb on {len(cases)} cases  dim={g.size(1)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_db", type=str, required=True)
    ap.add_argument("--global_dir", type=str, required=True)
    ap.add_argument("--global_prefix", type=str, default="dino_global")
    ap.add_argument("--dst_db", type=str, required=True)
    args = ap.parse_args()

    src_db, dst_db, gdir = Path(args.src_db), Path(args.dst_db), Path(args.global_dir)
    dst_db.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid"):
        swap_split(src_db, dst_db, gdir, args.global_prefix, split)

    # symlink shared files (reuse SigLIP2 teacher ceiling; no disk duplication)
    for fname in ("meta.json", "cause_text_embs.pt",
                  "teacher_train_train.pt", "train_candidate_pool.pt"):
        s = (src_db / fname).resolve()
        d = dst_db / fname
        if d.exists() or d.is_symlink():
            d.unlink()
        if s.exists():
            os.symlink(s, d)
            print(f"[link] {fname} -> {s}")
        else:
            print(f"[skip] {fname} not in src_db")
    print(f"[done] {dst_db}")


if __name__ == "__main__":
    main()
