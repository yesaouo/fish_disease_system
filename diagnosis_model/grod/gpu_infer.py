"""OAVLE 最簡流程版本 (simplest main-flow reference).

單檔、可讀、獨立跑的生產 soft 主流程參考檔——一次 GROD joint forward →
Region-Gate 加權 DeepSets 聚合 → 密集檢索 base bank → CEAH 病因排序。

刻意精簡，只保留主流程，不含生產服務的週邊：
  - 閥值：定值常數 (DEFAULT_LESION_THRESH)，不做校準/每張自適應。
  - 檢索庫：只讀離線 base bank (bank_z_soft.pt + case_db)，
            **不含 base⊕delta 熱更新**（那套 HITL 機制在 grod/pipeline.py
            的 GpuPipelineSoft，由 serve 的 /bank/* 端點驅動）。
  - 無 provenance / 圖表 / 病因去重——那些在 grod/pipeline.py。

流程與 bank_z_soft 的建法一致：聚合吃「全部 Q 個 gate 加權 query」，
CEAH 吃「依 objectness 取 top-K 個病灶」。Region Gate (∅-sink 溫度 softmax)
權重驅動聚合+歸因；病灶選取/abstain 走原始 objectness sigmoid(logit)。

Run from repo root:
    $PY -m diagnosis_model.grod.gpu_infer --image path/to.jpg
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

# 定值閥值：max_i w_i < τ 即判健康 (abstain)。rationale: grod/LESION_GATE.md。
DEFAULT_LESION_THRESH = 0.5


class GpuPipeline:
    def __init__(self, joint_ckpt, global_sd, anchors, enc_ckpt, ceah_ckpt,
                 case_db_dir, bank_path, top_k_lesions=32, device="cuda"):
        self.dev = device
        self.top_k_lesions = top_k_lesions

        # --- GROD joint detector: box / objectness / semantic z / global，一次 forward。
        #     走 vendored decoder，推論不 import rfdetr；env 開關啟用 heads。
        os.environ["RFDETR_SEMANTIC_DIM"] = "768"
        os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(anchors)
        os.environ["RFDETR_GLOBAL_DIM"] = "768"
        from diagnosis_model.grod.build import OAVLE
        self.net = OAVLE.from_vendored(joint_ckpt, device=device)
        self.net.core.global_embed.load_state_dict(torch.load(global_sd, map_location=device))
        self.res = int(self.net.resolution)
        self.means, self.stds = list(self.net.means), list(self.net.stds)

        # --- Weighted DeepSets Aggregator ---
        from diagnosis_model.cause_inference.models.case_encoder import (
            EncoderConfig, build_encoder,
        )
        pkg = torch.load(enc_ckpt, weights_only=False, map_location="cpu")
        cfg_dict = pkg["encoder_config"]; cfg_dict["dtype"] = torch.bfloat16
        self.enc = build_encoder(EncoderConfig(**cfg_dict)).to(device).eval()
        self.enc.load_state_dict(pkg["encoder_state"])

        # --- Region Gate (∅-sink 溫度 softmax)，權重隨 encoder ckpt。
        #     缺 (legacy sigmoid ckpt) → gate=None，退回 sigmoid(obj)。
        self.gate = None
        if "gate_state" in pkg:
            from diagnosis_model.grod.region_gate import RegionGate
            self.gate = RegionGate(init_temp=pkg.get("gate_init_temp", 0.3)).to(device).eval()
            self.gate.load_state_dict(pkg["gate_state"])

        # --- CEAH ---
        from diagnosis_model.cause_inference.models.ceah import CEAH
        self.ceah = CEAH(global_dim=768, text_dim=768, lesion_dim=768, cause_dim=768,
                         attribution_mode="softmax", scoring_mode="multiplicative").to(device).eval()
        self.ceah.load_state_dict(torch.load(ceah_ckpt, weights_only=False, map_location=device))

        # --- base bank (離線預算，常駐 GPU；無 delta) ---
        self.bank_z = torch.load(bank_path, weights_only=False)["bank_z"].to(device)  # [Nt,768]
        cdir = Path(case_db_dir)
        cases = torch.load(cdir / "train_cases.pt", weights_only=False)
        cte = torch.load(cdir / "cause_text_embs.pt", weights_only=False)
        self.cause_embs = F.normalize(cte["embeddings"].float().to(device), dim=-1)   # [Ncause,768]
        self.cause_texts = list(cte["texts"])
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
              f"memb={tuple(self.memb.shape)} (base only, no delta)")

    @torch.no_grad()
    def infer(self, image: Image.Image, det_thresh=DEFAULT_LESION_THRESH,
              top_k_cases=20, top_n=10):
        px = TF.normalize(TF.resize(TF.to_tensor(image), [self.res, self.res]),
                          self.means, self.stds).unsqueeze(0).to(self.dev)
        out = self.net(px)
        logits = out["pred_logits"][0][:, 0]                     # [Q] abnormal logit
        z_all, g = out["pred_semantic"][0], out["pred_global"][0]  # [Q,768],[768]
        obj = logits.sigmoid()                                   # [Q] objectness: 選取 / abstain
        w = self.gate(logits.unsqueeze(0))[0] if self.gate is not None else obj

        sel, lidx = obj.topk(self.top_k_lesions)                 # top-K by objectness
        if sel[0].item() < det_thresh:                           # abstain: 全圖無病灶過閥
            return []

        # 全 Q 個 gate 加權 query 聚合 → query case 向量 (與 bank_z_soft 建法一致)
        zq = self.enc(g.float().unsqueeze(0), z_all.float().unsqueeze(0),
                      torch.tensor([w.numel()], device=self.dev),
                      lesion_weights=w.float().unsqueeze(0))      # [1,768]

        # 密集檢索 base bank → 候選病因池
        s = zq @ self.bank_z.t()                                 # [1,Nt]
        _, cidx = s[0].topk(top_k_cases)
        rows, rlen = self.memb[cidx], self.mlen[cidx]
        cmask = torch.arange(rows.size(1), device=self.dev)[None] < rlen[:, None]
        cand = torch.unique(rows[cmask]); cand = cand[cand >= 0]
        P = cand.numel()
        cand_embs = self.cause_embs[cand]                        # [P,768]

        # CEAH：top-K 病灶 (by objectness)，gate 權重當 soft alpha gate
        z_k = z_all[lidx]                                        # [K,768]
        g_e = g.float().unsqueeze(0).expand(P, -1)
        l_e = z_k.float().unsqueeze(0).expand(P, -1, -1).contiguous()
        l_w = w[lidx].float().unsqueeze(0).expand(P, -1).contiguous()
        l_m = torch.ones(P, self.top_k_lesions, dtype=torch.bool, device=self.dev)
        t_e = torch.zeros(P, 768, device=self.dev)
        t_p = torch.zeros(P, dtype=torch.bool, device=self.dev)
        s_ceah, _, _ = self.ceah(g_e, t_e, t_p, l_e, l_m, cand_embs, lesion_weights=l_w)
        order = s_ceah.argsort(descending=True)[:top_n]

        ids = cand[order].cpu().tolist()
        sc = s_ceah[order].cpu().tolist()
        return [(self.cause_texts[i], v) for i, v in zip(ids, sc)]


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"  # unified artifact root (current dataset version)
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--global_sd", default=f"{ART}/models/distilled_global_rawP/global_embed_state_dict.pt")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--enc_ckpt", default=f"{ART}/models/encoder_grod_soft/best_encoder.pt")
    ap.add_argument("--ceah_ckpt", default=f"{ART}/models/ceah_grod_soft/best_ceah.pt")
    ap.add_argument("--bank_path", default=f"{ART}/models/encoder_grod_soft/bank_z_soft.pt")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--top_k_lesions", type=int, default=32,
                    help="lesions fed to CEAH (top-K by objectness); must match train_ceah_soft.")
    ap.add_argument("--det_thresh", type=float, default=DEFAULT_LESION_THRESH)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()

    pipe = GpuPipeline(args.joint_ckpt, args.global_sd, args.anchors,
                       args.enc_ckpt, args.ceah_ckpt, args.case_db_dir,
                       args.bank_path, top_k_lesions=args.top_k_lesions)
    res = pipe.infer(Image.open(args.image).convert("RGB"), det_thresh=args.det_thresh)
    if not res:
        print("\nAbstain: no disease detected — out of scope for cause inference.")
        return
    print("\nTop causes:")
    for r, (txt, sc) in enumerate(res, 1):
        print(f"  {r:>2}. {sc:.3f}  {txt[:70]}")


if __name__ == "__main__":
    main()
