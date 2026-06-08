from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# Fixed lesion-selection threshold on objectness sigmoid(pred_logits[:,0]):
# keep a query as a lesion iff obj > τ, abstain (healthy) when none kept.
# Derived by compute_lesion_threshold.py; rationale in grod/LESION_GATE.md.
DEFAULT_LESION_THRESH = 0.5


class GpuPipeline:
    def __init__(self, joint_ckpt, global_sd, anchors, enc_ckpt, ceah_ckpt,
                 case_db_dir, device="cuda"):
        self.dev = device
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

        # --- Aggregator (DeepSets) ---
        from diagnosis_model.cause_inference.models.case_encoder import (
            EncoderConfig, build_encoder,
        )
        pkg = torch.load(enc_ckpt, weights_only=False, map_location="cpu")
        cfg_dict = pkg["encoder_config"]; cfg_dict["dtype"] = torch.bfloat16
        self.enc = build_encoder(EncoderConfig(**cfg_dict)).to(device).eval()
        self.enc.load_state_dict(pkg["encoder_state"])

        # --- CEAH ---
        from diagnosis_model.cause_inference.models.ceah import CEAH
        self.ceah = CEAH(global_dim=768, text_dim=768, lesion_dim=768, cause_dim=768,
                         attribution_mode="softmax", scoring_mode="multiplicative").to(device).eval()
        self.ceah.load_state_dict(torch.load(ceah_ckpt, weights_only=False, map_location=device))

        # --- bank artifacts (precomputed once, kept on GPU) ---
        cdir = Path(case_db_dir)
        cases = torch.load(cdir / "train_cases.pt", weights_only=False)
        from diagnosis_model.cause_inference.train_case_encoder import encode_all
        self.bank_z = encode_all(self.enc, cases, device).to(device)          # [Nt, 768]
        cte = torch.load(cdir / "cause_text_embs.pt", weights_only=False)
        self.cause_embs = F.normalize(cte["embeddings"].float().to(device), dim=-1)  # [Ncause, 768]
        self.cause_texts = cte["texts"]
        # case -> cause membership, padded [Nt, max_c] on GPU
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
              f"memb={tuple(self.memb.shape)} (all on {self.bank_z.device})")

    @torch.no_grad()
    def infer(self, image: Image.Image, det_thresh=DEFAULT_LESION_THRESH, top_k_cases=20, top_n=10,
              verify=False):
        px = TF.normalize(TF.resize(TF.to_tensor(image), [self.res, self.res]),
                          self.means, self.stds).unsqueeze(0).to(self.dev)
        out = self.net(px)
        logits = out["pred_logits"][0][:, 0]                         # [Q] object logit
        z_all, g = out["pred_semantic"][0], out["pred_global"][0]    # [Q,768],[768]
        obj = logits.sigmoid()                                       # [Q]

        # Lesion gate: keep queries above the fixed threshold; no box ⟹ healthy.
        keep = obj > det_thresh
        if keep.sum() == 0:
            return []
        z = z_all[keep]                                              # [N,768]
        N = z.size(0)

        # Aggregator -> query case vector
        zq = self.enc(g.unsqueeze(0), z.unsqueeze(0),
                      torch.tensor([N], device=self.dev))            # [1,768]
        # retrieval (dense, no RVQ)
        s = zq @ self.bank_z.t()                                     # [1,Nt]
        topw, topi = s[0].topk(top_k_cases)                          # [k]
        # candidate cause pool from top-k cases
        rows, rlen = self.memb[topi], self.mlen[topi]                # [k,max_c],[k]
        cmask = torch.arange(rows.size(1), device=self.dev)[None] < rlen[:, None]
        cand = torch.unique(rows[cmask])                             # [P], sorted
        cand = cand[cand >= 0]
        P = cand.numel()
        # CEAH over candidates (evidence replicated)
        cand_embs = self.cause_embs[cand]                            # [P,768]
        g_emb = g.unsqueeze(0).expand(P, -1)
        l_emb = z.unsqueeze(0).expand(P, -1, -1).contiguous()
        l_mask = torch.ones(P, N, dtype=torch.bool, device=self.dev)
        t_emb = torch.zeros(P, 768, device=self.dev)
        t_present = torch.zeros(P, dtype=torch.bool, device=self.dev)
        s_ceah, alphas, _ = self.ceah(g_emb, t_emb, t_present, l_emb, l_mask, cand_embs)
        order = s_ceah.argsort(descending=True)[:top_n]

        if verify:
            for name, t in [("pred_global", g), ("lesion_z", z), ("query_zq", zq),
                            ("sims", s), ("candidates", cand), ("ceah_score", s_ceah),
                            ("rank_order", order)]:
                assert t.is_cuda, f"{name} left CUDA!"
            print(f"[verify] N_lesions={N} P_candidates={P} — all compute tensors on CUDA ✓")

        ids = cand[order].cpu().tolist()                             # final result -> CPU
        scores = s_ceah[order].cpu().tolist()
        return [(self.cause_texts[i], sc) for i, sc in zip(ids, scores)]


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"  # unified artifact root (under the current dataset version)
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--global_sd", default=f"{ART}/models/distilled_global_rawP/global_embed_state_dict.pt")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--enc_ckpt", default=f"{ART}/models/encoder_grod/best_encoder.pt")
    ap.add_argument("--ceah_ckpt", default=f"{ART}/models/ceah_jointDistRawP/best_ceah.pt")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--image", required=True)
    ap.add_argument("--det_thresh", type=float, default=DEFAULT_LESION_THRESH)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    pipe = GpuPipeline(args.joint_ckpt, args.global_sd, args.anchors,
                       args.enc_ckpt, args.ceah_ckpt, args.case_db_dir)
    res = pipe.infer(Image.open(args.image).convert("RGB"),
                     det_thresh=args.det_thresh, verify=args.verify)
    if not res:
        print("\nAbstain: no lesion detected — out of scope for cause inference.")
        return
    print("\nTop causes:")
    for r, (txt, sc) in enumerate(res, 1):
        print(f"  {r:>2}. {sc:.3f}  {txt[:70]}")


if __name__ == "__main__":
    main()
