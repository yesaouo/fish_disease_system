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

from diagnosis_model.cause_inference.models import CEAH
from diagnosis_model.cause_inference.phase1_baseline import build_candidate_pool
from diagnosis_model.cause_inference.faithfulness_eval import classify_cause
from diagnosis_model.cause_inference.models.case_encoder import EncoderConfig, build_encoder
from diagnosis_model.grod.train_case_encoder_soft import encode_all_soft, load_soft
from diagnosis_model.grod.eval_ceah_soft_ksweep import select


@torch.no_grad()
def faith_mode(ceah, H_va, bank_z, train_cases, valid_cases, cause_embs, cause_texts,
               g_va, z_va, w_va, device, mode, top_k_cases=20, max_queries=300):
    nq = min(max_queries, len(valid_cases))
    in_dim = cause_embs.size(-1)
    drops = defaultdict(list)
    drops_bucket = defaultdict(lambda: defaultdict(list))
    for qi in range(nq):
        sims = H_va[qi:qi + 1].to(device) @ bank_z.T
        top_idx = sims[0].topk(top_k_cases).indices.cpu().numpy()
        cand_idx = build_candidate_pool(top_idx, train_cases)
        P = len(cand_idx)
        if P == 0:
            continue
        cand_embs = cause_embs[torch.as_tensor(cand_idx, device=device)]
        z_s, w_s = select(z_va[qi].float().to(device), w_va[qi].float().to(device), mode)
        K = z_s.size(0)
        g_e = g_va[qi].float().to(device).unsqueeze(0).expand(P, -1)
        l_e = z_s.unsqueeze(0).expand(P, -1, -1).contiguous()
        l_w = w_s.unsqueeze(0).expand(P, -1).contiguous()
        l_m = torch.ones(P, K, dtype=torch.bool, device=device)
        t_e = torch.zeros(P, in_dim, device=device)
        t_p = torch.zeros(P, dtype=torch.bool, device=device)
        scores, alphas, ev_mask = ceah(g_e, t_e, t_p, l_e, l_m, cand_embs, lesion_weights=l_w)
        top1 = int(scores.argmax().item())
        baseline = float(scores[top1]); base_alpha = alphas[top1].cpu().numpy()
        max_Ne = ev_mask.size(1)
        bucket = classify_cause(cause_texts[cand_idx[top1]])

        def mask_score(positions):
            fm = torch.ones(P, max_Ne, dtype=torch.bool, device=device)
            for p in positions:
                if p < max_Ne:
                    fm[:, p] = False
            s, _, _ = ceah(g_e, t_e, t_p, l_e, l_m, cand_embs, force_mask=fm, lesion_weights=l_w)
            return float(s[top1])

        lesion_pos = list(range(2, 2 + K))
        valid_pos = [0] + lesion_pos
        top_a = max(valid_pos, key=lambda p: base_alpha[p])
        others = [p for p in valid_pos if p != top_a]
        d = {
            "no_global": baseline - mask_score([0]),
            "no_lesion": baseline - mask_score(lesion_pos),
            "no_top_α": baseline - mask_score([top_a]),
            "no_random": baseline - (mask_score([int(np.random.choice(others))]) if others else baseline),
        }
        for k, v in d.items():
            drops[k].append(v)
            drops_bucket[bucket][k].append(v)
    out = {k: float(np.mean(v)) for k, v in drops.items()}
    out["nl_lesion-type"] = float(np.mean(drops_bucket.get("lesion-type", {}).get("no_lesion", [0])))
    out["nl_global-type"] = float(np.mean(drops_bucket.get("global-type", {}).get("no_lesion", [0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", default="diagnosis_model/cause_inference/outputs/case_db_jointDistRawP")
    ap.add_argument("--soft_dir", default="diagnosis_model/grod/outputs/soft_inputs")
    ap.add_argument("--encoder_ckpt", default="diagnosis_model/cause_inference/outputs/encoder_grod_soft/best_encoder.pt")
    ap.add_argument("--bank_path", default="diagnosis_model/cause_inference/outputs/encoder_grod_soft/bank_z_soft.pt")
    ap.add_argument("--ceah_ckpt", default="diagnosis_model/cause_inference/outputs/ceah_grod_soft/best_ceah.pt")
    ap.add_argument("--modes", nargs="+", default=["thresh@0.3", "top16", "top32", "top64", "all"])
    ap.add_argument("--max_queries", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    np.random.seed(args.seed)
    device = args.device
    train_cases = torch.load(Path(args.case_db_dir) / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(Path(args.case_db_dir) / "valid_cases.pt", weights_only=False)
    cpack = torch.load(Path(args.case_db_dir) / "cause_text_embs.pt", weights_only=False)
    cause_embs = cpack["embeddings"].float().to(device); cause_texts = cpack["texts"]
    in_dim = cause_embs.size(-1)
    g_va, z_va, w_va, _ = load_soft(Path(args.soft_dir) / "valid.pt")

    enc_pkg = torch.load(args.encoder_ckpt, weights_only=False, map_location="cpu")
    encoder = build_encoder(EncoderConfig(**enc_pkg["encoder_config"])).to(device).eval()
    encoder.load_state_dict(enc_pkg["encoder_state"])
    bank_z = torch.load(args.bank_path, weights_only=False)["bank_z"].to(device)
    ceah = CEAH(global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
                common_dim=256, hidden_dim=512, dropout=0.0,
                attribution_mode="softmax", scoring_mode="multiplicative").to(device).eval()
    ceah.load_state_dict(torch.load(args.ceah_ckpt, map_location=device))

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
