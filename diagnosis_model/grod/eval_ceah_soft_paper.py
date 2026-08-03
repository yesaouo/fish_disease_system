"""Paper-grade full-valid eval for the OAVLE soft cascade (Region-Gate weighted).

Unlike eval_ceah_soft_ksweep.py (300-query evidence-selection robustness probe),
this runs the production soft cascade on the FULL valid set and reports the
metrics the thesis Ch5 tables need:

  retrieval  : encoder(g, z, w) -> bank_z top-k cases -> candidate cause pool
  prior s1   : phase1 score_candidates over retrieved cases (the gamma prior)
  CEAH s_c   : soft CEAH (lesion_weights = Region-Gate w, top-K by w)
  ranking    : hybrid = gamma * minmax(s1) + (1-gamma) * minmax(s_c)

Metrics per gamma: sem R@{1,3,5,10,20} + sem MRR, cluster R@{1,10,20} + cluster
MRR (cause_clusters_llm.json), and NDCG@5 (binary relevance = sem-match to any GT).
Conventions (L2-norm, miss=+inf, occurrence-level cluster) mirror eval_ceah.py /
phase1_baseline.py so the numbers are comparable to the rest of the paper. The
metric loop itself lives in soft_eval_common.py, shared with
eval_integration_ablation.py.

**Defaults are the production operating point**: current artifact tree
(`artifacts.ART`), **gated** soft inputs (self-consistent with `bank_z_soft`;
raw `soft_inputs` does not match it), `--split test`, `--top_k_cases 3`. A bare
run reproduces the thesis Ch5 numbers; pass `--split valid` / other `--top_k_cases`
for the sweeps. See BUILD_PIPELINE.md §12.

Run: $PY -m diagnosis_model.grod.eval_ceah_soft_paper
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diagnosis_model.grod.artifacts import (
    ART, load_case_db, load_ceah, load_cluster_ids, load_encoder, load_bank,
)
from diagnosis_model.grod.soft_eval_common import Backend, evaluate, topk_lesions
from diagnosis_model.grod.train_case_encoder_soft import encode_all_soft, load_soft


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", default=str(ART / "db/case_db_jointDistRawP"))
    ap.add_argument("--soft_dir", default=str(ART / "db/soft_inputs_gated"))
    ap.add_argument("--encoder_ckpt", default=str(ART / "models/encoder_grod_soft/best_encoder.pt"))
    ap.add_argument("--bank_path", default=str(ART / "models/encoder_grod_soft/bank_z_soft.pt"))
    ap.add_argument("--ceah_ckpt", default=str(ART / "models/ceah_grod_soft/best_ceah.pt"))
    ap.add_argument("--cluster_json", default=str(ART / "cause_clusters_llm.json"))
    ap.add_argument("--output_dir", default=str(ART / "models/ceah_grod_soft/paper_eval"))
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--text", choices=["none", "medical", "colloquial"], default="medical",
                    help="CEAH text-evidence slot. 'none' = vision-only (Image+Lesion row).")
    ap.add_argument("--mask_lesions", action="store_true",
                    help="Drop lesion evidence from CEAH scoring (Image+Text row). "
                         "Retrieval pool is unchanged (encoder still uses lesions).")
    ap.add_argument("--top_k_cases", type=int, default=3,
                    help="production operating point (see README / Table H6). Pass 1/3/5/10/20 for the k-sweep table.")
    ap.add_argument("--split", default="test", choices=["valid", "test"],
                    help="query split: loads {split}_cases.pt + <soft_dir>/{split}.pt")
    ap.add_argument("--top_k_lesions", type=int, default=32)
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--max_queries", type=int, default=-1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    db = load_case_db(args.case_db_dir, args.split, device)
    cluster_id_array = load_cluster_ids(args.cluster_json, db.cause_texts)
    if cluster_id_array is not None:
        print(f"[cluster] {len(set(cluster_id_array.tolist()))} clusters")

    g_va, z_va, w_va, _ = load_soft(Path(args.soft_dir) / f"{args.split}.pt")

    encoder = load_encoder(args.encoder_ckpt, device)
    bank_z = load_bank(args.bank_path, device)
    ceah = load_ceah(args.ceah_ckpt, db.in_dim, device)
    H_va = encode_all_soft(encoder, g_va, z_va, w_va, device).to(device)

    def query(qi):
        gt = db.query_cases[qi]["cause_emb_indices"]
        z_sel, w_sel = topk_lesions(z_va[qi].float().to(device),
                                    w_va[qi].float().to(device), args.top_k_lesions)
        text_emb = (None if args.text == "none"
                    else db.query_cases[qi][f"text_{args.text}_emb"].float().to(device))
        return gt, g_va[qi].float().to(device), text_emb, z_sel, w_sel

    be = Backend(name="OAVLE (soft)", train_cases=db.train_cases,
                 n_queries=len(db.query_cases), cause_embs=db.cause_embs,
                 cause_texts=db.cause_texts, bank_z=bank_z, H=H_va, ceah=ceah,
                 query=query, mask_lesions=args.mask_lesions)

    res = evaluate(be, args.gammas, args.top_k_cases, args.semantic_threshold,
                   cluster_id_array, args.max_queries, device)
    out = res["metrics_per_gamma"]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    json.dump({"n_queries": res["n_queries"], "metrics_per_gamma": out, "args": vars(args)},
              open(Path(args.output_dir) / "metrics_gammas.json", "w"), indent=2)

    cols = ["sem_R@1", "sem_R@5", "sem_R@10", "sem_MRR", "NDCG@5",
            "sem_P@5", "sem_Rq@5", "sem_F1@5", "sem_P@10", "sem_F1@10",
            "cl_R@1", "cl_R@10", "cl_MRR"]
    print(f"\n{'gamma':<8}" + "".join(f"{c:>10}" for c in cols))
    for t, b in out.items():
        print(f"{t:<8}" + "".join(f"{b.get(c, float('nan')):>10.4f}" for c in cols))


if __name__ == "__main__":
    main()
