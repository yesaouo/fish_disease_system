"""End-to-end *soft* variant of gpu_infer.py.

`gpu_infer.py` (the hard baseline) hard-selects lesion queries with `det_thresh`
and abstains via "zero kept boxes". This version removes the hard cliff:

    w_i = sigmoid(objectness_i)          # per-query soft evidence, ALL Q kept
    Aggregator / CEAH consume w as a *continuous* mask (soft pooling / soft alpha)
    abstain  = max_i w_i < τ             # fixed threshold (grod/LESION_GATE.md)

The three soft generalizations reduce *exactly* to gpu_infer.py's hard behaviour
when w ∈ {0,1}: aggregator (weighted mean → hard mask), CEAH alpha
(s_logits += log(w) → masked_fill -inf), presence (max_i w_i → "any kept box").

Production soft cascade — all components retrained on soft inputs:
  - Aggregator : encoder_grod_soft  (DeepSets, native lesion_weights pooling)
  - bank       : bank_z_soft.pt     (soft-encoded train cases)
  - CEAH       : ceah_grod_soft      (soft alpha gate; top-K lesions by w)

Retrieval uses ALL Q soft-weighted queries (matches how bank_z_soft was built);
CEAH uses the top-K=32 queries by w (matches train_ceah_soft). A single
`w.topk(K)` serves both: scores[0] is the max-w (abstain check) and
(scores, indices) are the top-K lesion weights / queries fed to CEAH.

Run from repo root:
    $PY -m diagnosis_model.grod.gpu_infer_soft --image path/to.jpg
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# Fixed lesion-objectness threshold: abstain (healthy) iff max_i w_i < τ.
# Derived by compute_lesion_threshold.py; rationale in grod/LESION_GATE.md.
DEFAULT_LESION_THRESH = 0.5


class GpuPipelineSoft:
    def __init__(self, joint_ckpt, global_sd, anchors, enc_ckpt, ceah_ckpt,
                 case_db_dir, bank_path, top_k_lesions=32,
                 device="cuda"):
        self.dev = device
        self.top_k_lesions = top_k_lesions

        # --- GROD joint detector: decoder region heads (box/obj/semantic z)
        #     + backbone-tapped global branch (both enabled via env) ---
        os.environ["RFDETR_SEMANTIC_DIM"] = "768"
        os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(anchors)
        os.environ["RFDETR_GLOBAL_DIM"] = "768"
        from rfdetr import RFDETRMedium
        rf = RFDETRMedium(pretrain_weights=joint_ckpt, num_classes=1)
        self.net = rf.model.model.to(device).eval()
        self.net.global_embed.load_state_dict(torch.load(global_sd, map_location=device))
        self.res = int(rf.model.resolution)
        self.means, self.stds = list(rf.means), list(rf.stds)

        # --- Aggregator (DeepSets) — native soft pooling via lesion_weights ---
        from diagnosis_model.cause_inference.models.case_encoder import (
            EncoderConfig, build_encoder,
        )
        pkg = torch.load(enc_ckpt, weights_only=False, map_location="cpu")
        cfg_dict = pkg["encoder_config"]; cfg_dict["dtype"] = torch.bfloat16
        self.enc = build_encoder(EncoderConfig(**cfg_dict)).to(device).eval()
        self.enc.load_state_dict(pkg["encoder_state"])

        # --- CEAH — soft alpha gate via lesion_weights ---
        from diagnosis_model.cause_inference.models.ceah import CEAH
        self.ceah = CEAH(global_dim=768, text_dim=768, lesion_dim=768, cause_dim=768,
                         attribution_mode="softmax", scoring_mode="multiplicative").to(device).eval()
        self.ceah.load_state_dict(torch.load(ceah_ckpt, weights_only=False, map_location=device))

        # --- bank artifacts (precomputed soft bank, kept on GPU) ---
        self.bank_z = torch.load(bank_path, weights_only=False)["bank_z"].to(device)  # [Nt,768]
        cdir = Path(case_db_dir)
        cases = torch.load(cdir / "train_cases.pt", weights_only=False)
        cte = torch.load(cdir / "cause_text_embs.pt", weights_only=False)
        self.cause_embs = F.normalize(cte["embeddings"].float().to(device), dim=-1)  # [Ncause,768]
        self.cause_texts = cte["texts"]
        idx_lists = [c["cause_emb_indices"] for c in cases]
        max_c = max(len(x) for x in idx_lists)
        memb = torch.full((len(cases), max_c), -1, dtype=torch.long)
        mlen = torch.zeros(len(cases), dtype=torch.long)
        for i, xs in enumerate(idx_lists):
            memb[i, :len(xs)] = torch.tensor(xs, dtype=torch.long)
            mlen[i] = len(xs)
        self.memb = memb.to(device)
        self.mlen = mlen.to(device)
        print(f"[bank] z={tuple(self.bank_z.shape)} causes={tuple(self.cause_embs.shape)} "
              f"memb={tuple(self.memb.shape)}")

    @torch.no_grad()
    def infer(self, image: Image.Image, det_thresh=DEFAULT_LESION_THRESH,
              top_k_cases=20, top_n=10, verify=False):
        px = TF.normalize(TF.resize(TF.to_tensor(image), [self.res, self.res]),
                          self.means, self.stds).unsqueeze(0).to(self.dev)
        out = self.net(px)
        logits = out["pred_logits"][0][:, 0]                     # [Q] ABNORMAL logit (cat 0 -> col 0)
        z_all, g = out["pred_semantic"][0], out["pred_global"][0]  # [Q,768],[768]
        w = logits.sigmoid()                                     # [Q] soft evidence, all Q kept

        # one topk serves both: scores[0]=max_w (abstain) and (scores, lidx)=top-K (CEAH)
        scores, lidx = w.topk(self.top_k_lesions)                                 # descending; scores[0]=max_w

        # abstain: healthy iff no query clears the fixed threshold
        if scores[0].item() < det_thresh:
            return []

        # aggregate ALL Q soft-weighted queries -> query case vector (matches bank_z_soft)
        zq = self.enc(g.float().unsqueeze(0), z_all.float().unsqueeze(0),
                      torch.tensor([w.numel()], device=self.dev),
                      lesion_weights=w.float().unsqueeze(0))      # [1,768]

        # retrieval (dense, no RVQ) over the soft bank -> candidate cause pool
        s = zq @ self.bank_z.t()                                 # [1,Nt]
        _, cidx = s[0].topk(top_k_cases)
        rows, rlen = self.memb[cidx], self.mlen[cidx]
        cmask = torch.arange(rows.size(1), device=self.dev)[None] < rlen[:, None]
        cand = torch.unique(rows[cmask])
        cand = cand[cand >= 0]
        P = cand.numel()
        cand_embs = self.cause_embs[cand]                        # [P,768]

        # soft CEAH over candidates: top-K lesions (by w), scores as the soft alpha gate
        z_k = z_all[lidx]                                        # [K,768]
        g_e = g.float().unsqueeze(0).expand(P, -1)
        l_e = z_k.float().unsqueeze(0).expand(P, -1, -1).contiguous()
        l_w = scores.float().unsqueeze(0).expand(P, -1).contiguous()
        l_m = torch.ones(P, self.top_k_lesions, dtype=torch.bool, device=self.dev)
        t_e = torch.zeros(P, 768, device=self.dev)
        t_p = torch.zeros(P, dtype=torch.bool, device=self.dev)
        s_ceah, alphas, _ = self.ceah(g_e, t_e, t_p, l_e, l_m, cand_embs, lesion_weights=l_w)
        order = s_ceah.argsort(descending=True)[:top_n]

        if verify:
            for name, t in [("pred_global", g), ("lesion_z", z_all), ("weights", w),
                            ("query_zq", zq), ("sims", s), ("candidates", cand),
                            ("ceah_score", s_ceah), ("rank_order", order)]:
                assert t.is_cuda, f"{name} left CUDA!"
            print(f"[verify] Q={z_all.size(0)} topK_les={self.top_k_lesions} P_candidates={P} "
                  f"max_w={scores[0].item():.3f} — all on CUDA ✓")

        ids = cand[order].cpu().tolist()
        sc = s_ceah[order].cpu().tolist()
        return [(self.cause_texts[i], v) for i, v in zip(ids, sc)]


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"  # unified artifact root (under the current dataset version)
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--global_sd", default=f"{ART}/models/distilled_global_rawP/global_embed_state_dict.pt")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--enc_ckpt", default=f"{ART}/models/encoder_grod_soft/best_encoder.pt")
    ap.add_argument("--ceah_ckpt", default=f"{ART}/models/ceah_grod_soft/best_ceah.pt")
    ap.add_argument("--bank_path", default=f"{ART}/models/encoder_grod_soft/bank_z_soft.pt")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--top_k_lesions", type=int, default=32,
                    help="lesions fed to CEAH (top-K by w); must match train_ceah_soft.")
    ap.add_argument("--det_thresh", type=float, default=DEFAULT_LESION_THRESH)
    ap.add_argument("--image", required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    pipe = GpuPipelineSoft(args.joint_ckpt, args.global_sd, args.anchors,
                           args.enc_ckpt, args.ceah_ckpt, args.case_db_dir,
                           args.bank_path, top_k_lesions=args.top_k_lesions)
    res = pipe.infer(Image.open(args.image).convert("RGB"),
                     det_thresh=args.det_thresh, verify=args.verify)
    if not res:
        print("\nAbstain: no disease detected — out of scope for cause inference.")
        return
    print("\nTop causes:")
    for r, (txt, sc) in enumerate(res, 1):
        print(f"  {r:>2}. {sc:.3f}  {txt[:70]}")


if __name__ == "__main__":
    main()
