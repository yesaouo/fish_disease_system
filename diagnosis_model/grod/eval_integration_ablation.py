"""Table 14 (整合式架構與區域門控消融) — all three rows in ONE run.

Computes sem R@10 + cluster R@10 (and R@1/5/20, MRRs, NDCG@5) for the three
settings compared in the thesis integration ablation, under ONE shared protocol
(learned single case-vector aggregation -> bank retrieval -> CEAM cause scoring,
cascade gamma=0 by default), so the numbers are directly comparable:

  分離式基準 (base)  : case_db_base + encoder_base + ceah_base (VLM lesion re-encode,
                       no Region-Gate weights). Standard DeepSets encode_all path.
  OAVLE-Hard (hard)  : encoder_grod_soft + ceah_grod_soft, query built from HARD
                       objectness (sigmoid(obj) > display_thresh -> {0,1}); the SOFT
                       bank_z_soft is reused (the demo's byte-exact hard-gate degenerate).
  OAVLE (soft, main) : encoder_grod_soft + ceah_grod_soft, Region-Gate (gated) weights.

The metric loop is soft_eval_common.evaluate — the same one eval_ceah_soft_paper.py
uses; only the per-query evidence source differs per backend. Conventions
(L2-norm, miss=+inf, occurrence-level cluster, semantic_threshold=0.95,
cause_clusters_llm) mirror eval_ceah.py / phase1_baseline.py.

Defaults are the production operating point (`--split test --top_k_cases 3`,
gamma=0), so a bare run reproduces the thesis table.

Run (SDM env, repo root):
  $PY -m diagnosis_model.grod.eval_integration_ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.train_case_encoder import encode_all
from diagnosis_model.grod.artifacts import (
    ART, load_case_db, load_ceah, load_cluster_ids, load_encoder, load_bank,
)
from diagnosis_model.grod.soft_eval_common import Backend, evaluate
from diagnosis_model.grod.train_case_encoder_soft import encode_all_soft, load_soft


# ----------------------------------------------------------------------------
# Backends: each owns its artifacts and a `query(qi)` closure returning that
# query's CEAM evidence. See soft_eval_common.Backend.
# ----------------------------------------------------------------------------
def build_soft_like(name, w_dir, binarize_thresh, top_k_lesions, device, split="valid"):
    """soft / hard: shared encoder_grod_soft + ceah_grod_soft + gated bank."""
    db = load_case_db(ART / "db/case_db_jointDistRawP", split, device)

    enc = load_encoder(ART / "models/encoder_grod_soft/best_encoder.pt", device)
    ceah = load_ceah(ART / "models/ceah_grod_soft/best_ceah.pt", db.in_dim, device)
    bank_z = load_bank(ART / "models/encoder_grod_soft/bank_z_soft.pt", device)

    g_va, z_va, w_va, _ = load_soft(ART / f"db/{w_dir}/{split}.pt")
    if binarize_thresh is not None:
        w_va = (w_va.float() > binarize_thresh).float()  # hard {0,1}
    H_va = encode_all_soft(enc, g_va, z_va, w_va, device).to(device)

    def query(qi):
        gt = db.query_cases[qi]["cause_emb_indices"]
        g_slot = g_va[qi].float().to(device)
        text_emb = db.query_cases[qi]["text_medical_emb"].float().to(device)
        K = min(top_k_lesions, w_va.size(1))
        idx = w_va[qi].float().to(device).topk(K).indices
        z_sel = z_va[qi].float().to(device).index_select(0, idx)
        w_sel = w_va[qi].float().to(device).index_select(0, idx)
        return gt, g_slot, text_emb, z_sel, w_sel

    return Backend(name=name, train_cases=db.train_cases, n_queries=len(db.query_cases),
                   cause_embs=db.cause_embs, cause_texts=db.cause_texts,
                   bank_z=F.normalize(bank_z.float(), dim=-1),
                   H=F.normalize(H_va.float(), dim=-1), ceah=ceah, query=query)


def build_base(name, top_k_lesions, device, split="valid"):
    """分離式基準: case_db_base + encoder_base + ceah_base (no gate weights)."""
    cdir = ART / ("db/case_db_base" if split == "valid" else "db/case_db_base_test")
    db = load_case_db(cdir, split, device)

    enc = load_encoder(ART / "models/encoder_base/best_encoder.pt", device)
    ceah = load_ceah(ART / "models/ceah_base/best_ceah.pt", db.in_dim, device)
    bank_z = encode_all(enc, db.train_cases, device).to(device)
    H_va = encode_all(enc, db.query_cases, device).to(device)

    def query(qi):
        q = db.query_cases[qi]
        gt = q["cause_emb_indices"]
        g_slot = q["global_emb"].float().to(device)
        text_emb = q["text_medical_emb"].float().to(device)
        z_sel = q["lesion_embs"].float().to(device)[:top_k_lesions]  # all case lesions
        return gt, g_slot, text_emb, z_sel, None  # no Region-Gate weights

    return Backend(name=name, train_cases=db.train_cases, n_queries=len(db.query_cases),
                   cause_embs=db.cause_embs, cause_texts=db.cause_texts,
                   bank_z=F.normalize(bank_z.float(), dim=-1),
                   H=F.normalize(H_va.float(), dim=-1), ceah=ceah, query=query)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0],
                    help="Cascade uses gamma=0 (pure CEAM). Pass more for a scan.")
    ap.add_argument("--top_k_cases", type=int, default=3,
                    help="production operating point (see README / Table H6).")
    ap.add_argument("--split", default="test", choices=["valid", "test"],
                    help="query split; base uses case_db_base{,_test} accordingly")
    ap.add_argument("--top_k_lesions", type=int, default=32)
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--cluster_json", default=str(ART / "cause_clusters_llm.json"))
    ap.add_argument("--display_thresh", type=float, default=None,
                    help="Hard-gate objectness threshold. Default: read thresholds.json.")
    ap.add_argument("--max_queries", type=int, default=-1)
    ap.add_argument("--settings", nargs="+", default=["base", "hard", "soft"],
                    choices=["base", "hard", "soft"])
    ap.add_argument("--output_dir", default=str(ART / "models/ceah_grod_soft/integration_ablation"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    disp = args.display_thresh
    if disp is None:
        disp = json.load(open("data/processed/current/thresholds.json"))["display_thresh"]

    builders = {
        "base": lambda: build_base("分離式基準", args.top_k_lesions, dev, args.split),
        "hard": lambda: build_soft_like("OAVLE-Hard", "soft_inputs", disp, args.top_k_lesions, dev, args.split),
        "soft": lambda: build_soft_like("OAVLE (soft)", "soft_inputs_gated", None, args.top_k_lesions, dev, args.split),
    }

    results = {}
    rows = []
    for key in args.settings:
        be = builders[key]()
        res = evaluate(be, args.gammas, args.top_k_cases, args.semantic_threshold,
                       load_cluster_ids(args.cluster_json, be.cause_texts),
                       args.max_queries, dev)
        results[key] = {"label": be.name, **res}
        rows.append((be.name, res))
        print(f"[{be.name}] n_valid={res['n_queries']}  "
              f"(cause space: {be.cause_embs.size(0)} causes)")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    json.dump({"args": {k: v for k, v in vars(args).items()},
               "display_thresh_used": disp, "results": results},
              open(Path(args.output_dir) / "metrics.json", "w"), indent=2, ensure_ascii=False)

    cols = ["sem_R@1", "sem_R@5", "sem_R@10", "sem_MRR",
            "sem_P@5", "sem_Rq@5", "sem_F1@5", "sem_P@10", "sem_F1@10",
            "cl_R@1", "cl_R@10", "cl_MRR"]
    for g in args.gammas:
        t = f"g={g:.2f}"
        print(f"\n=== gamma={g:.2f}  (top_k_cases={args.top_k_cases}, sem_thr={args.semantic_threshold}) ===")
        print(f"{'setting':<16}" + "".join(f"{c:>10}" for c in cols))
        for name, res in rows:
            b = res["metrics_per_gamma"][t]
            print(f"{name:<16}" + "".join(f"{b.get(c, float('nan')):>10.4f}" for c in cols))
    print(f"\nsaved -> {Path(args.output_dir) / 'metrics.json'}")


if __name__ == "__main__":
    main()
