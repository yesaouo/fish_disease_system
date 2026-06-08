"""Phase A (frozen) — rebuild a case_db whose lesion_embs are the semantic
head's z (RF-DETR hs -> Linear -> L2norm), aligned 1:1 to the source case_db.

Only the lesion *routing* changes: each matched lesion's `lesion_embs[g]` is
replaced by the head's z for that lesion's query feature. Everything else
(global_emb, text embs, causes, cause_text_embs.pt) is copied verbatim from the
source case_db so the only changed variable downstream is the lesion feature
source: isolated crop (source) vs RF-DETR decoder query (this).

Lesions that were unmatched (no IoU>=thresh query) or whose symptom category
could not be joined keep their ORIGINAL raw lesion_emb, so N is unchanged and
faithfulness_eval / train_ceah / build_train_candidate_pool need zero edits.

Run from repo root (see bottom for the command)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_head(head_path: str, hidden_dim: int, out_dim: int, device: str) -> nn.Linear:
    head = nn.Linear(hidden_dim, out_dim).to(device)
    state = torch.load(head_path, map_location=device)
    head.load_state_dict(state)
    head.eval()
    return head


@torch.no_grad()
def rebuild_split(
    src_cases: List[Dict], hs_cache: List[Dict], head, device: str,
) -> Dict[str, float]:
    """In-place replace lesion_embs for matched lesions. Returns stats.

    Two cache schemas are supported:
      - frozen probe: cache has 'hs' (raw query feat) + 'lesion_category_id';
        z = L2norm(head(hs)), and lesions with category < 0 are skipped.
      - joint:        cache has 'z' (already-trained, 768-d, L2-normalized) and
        no head (head is None); all matched lesions are replaced.
    """
    cache_by_case = {c["case_id"]: c for c in hs_cache}
    n_replaced = 0
    n_lesions = 0
    n_missing_cache = 0

    for case in src_cases:
        n_lesions += case["lesion_embs"].size(0)
        hc = cache_by_case.get(case["case_id"])
        if hc is None:
            n_missing_cache += 1
            continue

        if head is not None:  # frozen-probe schema
            if hc["hs"].numel() == 0:
                continue
            z = F.normalize(head(hc["hs"].to(device)), dim=-1).cpu().to(case["lesion_embs"].dtype)
            kept = hc["kept_lesion_idx"].tolist()
            cats = hc["lesion_category_id"].tolist()
            for row, (g, cat) in enumerate(zip(kept, cats)):
                if cat < 0:
                    continue
                if g < case["lesion_embs"].size(0):
                    case["lesion_embs"][g] = z[row]
                    n_replaced += 1
        else:                 # joint schema: z already final
            if hc["z"].numel() == 0:
                continue
            z = hc["z"].to(case["lesion_embs"].dtype)
            kept = hc["kept_lesion_idx"].tolist()
            for row, g in enumerate(kept):
                if g < case["lesion_embs"].size(0):
                    case["lesion_embs"][g] = z[row]
                    n_replaced += 1

    return {
        "n_lesions": n_lesions,
        "n_replaced": n_replaced,
        "replace_rate": n_replaced / max(1, n_lesions),
        "n_cases_missing_cache": n_missing_cache,
    }


def main():
    ap = argparse.ArgumentParser(description="Rebuild case_db with semantic-head z as lesion_embs.")
    ap.add_argument("--src_case_db", type=str, required=True,
                    help="source case_db dir (e.g. .../outputs/case_db_raw)")
    ap.add_argument("--hs_dir", type=str, required=True,
                    help="dir with cache: hs_{split}.pt (frozen probe) or z_{split}.pt (--from_joint)")
    ap.add_argument("--head_path", type=str, default=None,
                    help="semantic_head.pt (frozen probe only; omit with --from_joint)")
    ap.add_argument("--from_joint", action="store_true",
                    help="cache holds final trained z (z_{split}.pt from extract_z_joint.py); no head")
    ap.add_argument("--out_case_db", type=str, required=True)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    src = Path(args.src_case_db)
    hs_dir = Path(args.hs_dir)
    out = Path(args.out_case_db)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device

    cause_pack = torch.load(src / "cause_text_embs.pt", weights_only=False)
    out_dim = cause_pack["embeddings"].size(-1)

    # carry forward the dims downstream consumers read (train_case_encoder reads
    # meta["global_dim"]). global_emb is copied verbatim from source, so the
    # source meta's dims still hold.
    src_meta = json.load(open(src / "meta.json"))

    if args.from_joint:
        head = None
        hidden_dim = None
        cache_prefix = "z"
        print("[mode] joint: lesion_embs <- trained pred_semantic z (no head)")
    else:
        if args.head_path is None:
            raise ValueError("frozen-probe mode requires --head_path (or pass --from_joint)")
        head_meta = json.load(open(Path(args.head_path).parent / "meta.json"))
        hidden_dim = head_meta["hidden_dim"]
        head = load_head(args.head_path, hidden_dim, out_dim, device)
        cache_prefix = "hs"
        print(f"[head] Linear({hidden_dim} -> {out_dim}) loaded from {args.head_path}")

    stats = {}
    for split in ["train", "valid"]:
        src_cases = torch.load(src / f"{split}_cases.pt", weights_only=False)
        hs_cache = torch.load(hs_dir / f"{cache_prefix}_{split}.pt", weights_only=False)
        s = rebuild_split(src_cases, hs_cache, head, device)
        torch.save(src_cases, out / f"{split}_cases.pt")
        stats[split] = s
        print(f"[{split}] replaced {s['n_replaced']}/{s['n_lesions']} "
              f"lesion embs ({s['replace_rate']:.3f}); saved {split}_cases.pt")

    # copy the verbatim shared artifacts (control variables)
    shutil.copy(src / "cause_text_embs.pt", out / "cause_text_embs.pt")
    print("[copy] cause_text_embs.pt (verbatim)")

    meta = {
        "src_case_db": str(src),
        "hs_dir": str(hs_dir),
        "head_path": args.head_path,
        "from_joint": bool(args.from_joint),
        "hidden_dim": hidden_dim,
        "out_dim": out_dim,
        "global_dim": src_meta["global_dim"],
        "lesion_dim": src_meta["lesion_dim"],
        "note": ("lesion_embs replaced by trained joint pred_semantic z"
                 if args.from_joint else
                 "lesion_embs replaced by frozen-probe semantic-head z (RF-DETR hs routing)")
                + "; global_emb/text/cause table copied from source.",
        "stats": stats,
    }
    with (out / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[done] rebuilt case_db -> {out}")


if __name__ == "__main__":
    main()
