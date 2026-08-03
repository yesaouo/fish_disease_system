"""Faithfulness sweep over CEAH evidence selection (companion to
eval_ceah_soft_ksweep.py). Same fixed soft retrieval + soft CEAH; per mode
report the score-drop faithfulness signals:

  no_lesion  (must be POSITIVE) , no_top_α (should be largest) , no_random (~0)

so we can see whether feeding more/fewer lesion tokens dilutes lesion-grounding.

Run: $PY -m diagnosis_model.grod.eval_faithfulness_soft_sweep
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from diagnosis_model.cause_inference.phase1_baseline import build_candidate_pool
from diagnosis_model.cause_inference.faithfulness_eval import classify_cause
from diagnosis_model.grod.artifacts import ART, load_case_db, load_ceah, load_encoder, load_bank
from diagnosis_model.grod.soft_eval_common import faithfulness_drops, make_ceah_batch
from diagnosis_model.grod.train_case_encoder_soft import encode_all_soft, load_soft
from diagnosis_model.grod.eval_ceah_soft_ksweep import select


@torch.no_grad()
def faith_mode(ceah, H_va, bank_z, train_cases, valid_cases, cause_embs, cause_texts,
               g_va, z_va, w_va, device, mode, top_k_cases=20, max_queries=300):
    nq = min(max_queries, len(valid_cases))
    drops = defaultdict(list)
    drops_bucket = defaultdict(lambda: defaultdict(list))
    for qi in range(nq):
        sims = H_va[qi:qi + 1].to(device) @ bank_z.T
        top_idx = sims[0].topk(top_k_cases).indices.cpu().numpy()
        cand_idx = build_candidate_pool(top_idx, train_cases)
        if len(cand_idx) == 0:
            continue
        cand_embs = cause_embs[torch.as_tensor(cand_idx, device=device)]
        z_s, w_s = select(z_va[qi].float().to(device), w_va[qi].float().to(device), mode)
        batch = make_ceah_batch(ceah, g_va[qi].float().to(device), z_s, w_s, cand_embs, device)
        top1, _, d = faithfulness_drops(batch, device)
        bucket = classify_cause(cause_texts[cand_idx[top1]])
        for k, v in d.items():
            drops[k].append(v)
            drops_bucket[bucket][k].append(v)
    out = {k: float(np.mean(v)) for k, v in drops.items()}
    out["nl_lesion-type"] = float(np.mean(drops_bucket.get("lesion-type", {}).get("no_lesion", [0])))
    out["nl_global-type"] = float(np.mean(drops_bucket.get("global-type", {}).get("no_lesion", [0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", default=str(ART / "db/case_db_jointDistRawP"))
    ap.add_argument("--soft_dir", default=str(ART / "db/soft_inputs_gated"))
    ap.add_argument("--encoder_ckpt", default=str(ART / "models/encoder_grod_soft/best_encoder.pt"))
    ap.add_argument("--bank_path", default=str(ART / "models/encoder_grod_soft/bank_z_soft.pt"))
    ap.add_argument("--ceah_ckpt", default=str(ART / "models/ceah_grod_soft/best_ceah.pt"))
    ap.add_argument("--modes", nargs="+", default=["thresh@0.3", "top16", "top32", "top64", "all"])
    ap.add_argument("--max_queries", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    np.random.seed(args.seed)
    device = args.device
    db = load_case_db(args.case_db_dir, "valid", device)
    train_cases, valid_cases = db.train_cases, db.query_cases
    cause_embs, cause_texts = db.cause_embs, db.cause_texts
    g_va, z_va, w_va, _ = load_soft(Path(args.soft_dir) / "valid.pt")

    encoder = load_encoder(args.encoder_ckpt, device)
    bank_z = load_bank(args.bank_path, device)
    ceah = load_ceah(args.ceah_ckpt, db.in_dim, device)

    H_va = encode_all_soft(encoder, g_va, z_va, w_va, device)

    print(f"{'mode':<12}{'no_lesion':>11}{'no_top_α':>11}{'no_random':>11}{'no_global':>11}"
          f"{'nl:les-ty':>11}{'nl:glob-ty':>11}")
    for mode in args.modes:
        m = faith_mode(ceah, H_va, bank_z, train_cases, valid_cases, cause_embs, cause_texts,
                       g_va, z_va, w_va, device, mode, max_queries=args.max_queries)
        print(f"{mode:<12}{m['no_lesion']:>11.4f}{m['no_top_α']:>11.4f}{m['no_random']:>11.4f}"
              f"{m['no_global']:>11.4f}{m['nl_lesion-type']:>11.4f}{m['nl_global-type']:>11.4f}")
    print("\nfaithful = no_lesion POSITIVE & no_top_α largest & no_random ≈ 0")


if __name__ == "__main__":
    main()
