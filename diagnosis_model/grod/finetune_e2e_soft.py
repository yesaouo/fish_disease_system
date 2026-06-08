"""Lightweight co-adaptive end-to-end fine-tune of the soft cascade.

GROD frozen. Warm-start enc <- encoder_grod_soft, ceah <- ceah_grod_soft.
Each epoch (co-adaptation loop):
  1. re-encode the train bank with the CURRENT enc, rebuild the CEAH candidate
     pool from THAT soft retrieval (leave-one-out top-k -> candidate causes,
     positive if cos>=0.95 to a GT cause). This is the only achievable
     "end-to-end" coupling: CEAH trains on the Aggregator's own retrieval pool
     (zq is not consumed by CEAH and top-k is non-differentiable, so no gradient
     flows enc<-ceah; the link is the refreshed pool).
  2. fine-tune CEAH (BCE) on the refreshed soft pool, top-32 lesion evidence.
  3. fine-tune enc (listwise-KL teacher + InfoNCE) one pass.
  4. eval: soft-retrieval sem_MRR + faithfulness no_lesion (top-32).

Compares against the staged ckpts (encoder_grod_soft / ceah_grod_soft).
Output: outputs/e2e_soft/{best_encoder.pt,best_ceah.pt}

Run: $PY -m diagnosis_model.grod.finetune_e2e_soft
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from diagnosis_model.cause_inference.models import CEAH
from diagnosis_model.cause_inference.models.case_encoder import (
    EncoderConfig, build_encoder, listwise_kl_loss, case_cause_infonce_loss,
)
from diagnosis_model.cause_inference.phase1_baseline import build_candidate_pool
from diagnosis_model.grod.train_case_encoder_soft import encode_all_soft, load_soft
from diagnosis_model.grod.train_ceah_soft import (
    SoftCEAHDataset, make_soft_collate, soft_eval,
)
from diagnosis_model.grod.eval_faithfulness_soft_sweep import faith_mode


@torch.no_grad()
def build_soft_pool(bank, train_cases, cause_embs, top_k_cases, sem_thresh=0.95, chunk=2048):
    """LOO top-k over the soft bank -> per-case {candidate_cause_indices, positive_mask}."""
    N = bank.size(0)
    dev = bank.device
    pool = []
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sims = bank[s:e] @ bank.T                                   # [b, N]
        for r in range(e - s):
            qi = s + r
            sims[r, qi] = -1e30                                     # leave-one-out
            top_idx = sims[r].topk(top_k_cases).indices.cpu().numpy()
            cand = build_candidate_pool(top_idx, train_cases)
            if not cand:
                pool.append({"candidate_cause_indices": torch.empty(0, dtype=torch.long),
                             "positive_mask": torch.empty(0, dtype=torch.bool)})
                continue
            cand_t = torch.as_tensor(cand, device=dev)
            gt = train_cases[qi]["cause_emb_indices"]
            gt_e = cause_embs.index_select(0, torch.as_tensor(gt, device=dev))
            cos = gt_e @ cause_embs.index_select(0, cand_t).T
            pos = (cos >= sem_thresh).any(dim=0)
            pool.append({"candidate_cause_indices": cand_t.cpu(),
                         "positive_mask": pos.cpu()})
    return pool


def ceah_epoch(ceah, opt, loader, cause_embs, device, in_dim, K):
    ceah.train(); losses = []
    for b in loader:
        g = b["global_emb"].to(device); lz = b["lesion_z"].to(device)
        lw = b["lesion_w"].to(device); te = b["text_emb"].to(device)
        tp = b["text_present"].to(device); cand = b["cand_embs"].to(device)
        cm = b["cand_mask"].to(device); tgt = b["targets"].to(device)
        B, max_P, _ = cand.shape
        g_r = g.unsqueeze(1).expand(B, max_P, in_dim).reshape(B * max_P, in_dim)
        t_r = te.unsqueeze(1).expand(B, max_P, in_dim).reshape(B * max_P, in_dim)
        tp_r = tp.unsqueeze(1).expand(B, max_P).reshape(B * max_P)
        lz_r = lz.unsqueeze(1).expand(B, max_P, K, in_dim).reshape(B * max_P, K, in_dim)
        lw_r = lw.unsqueeze(1).expand(B, max_P, K).reshape(B * max_P, K)
        lm_r = torch.ones(B * max_P, K, dtype=torch.bool, device=device)
        s, _, _ = ceah(g_r, t_r, tp_r, lz_r, lm_r, cand.reshape(B * max_P, in_dim),
                       lesion_weights=lw_r)
        s = s.view(B, max_P); cmf = cm.float()
        loss = (F.binary_cross_entropy(s, tgt, reduction="none") * cmf).sum() / cmf.sum().clamp_min(1.0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ceah.parameters(), 1.0); opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def enc_epoch(enc, opt, teacher, g_tr, z_tr, w_tr, cidx_tr, cause_embs, device,
              batch_size=256, infonce_w=0.5, infonce_t=0.07):
    enc.train(); N = g_tr.size(0); perm = torch.randperm(N); losses = []
    for bs in range(0, N - batch_size + 1, batch_size):
        idx = perm[bs:bs + batch_size]; b = idx.size(0)
        lens = torch.full((b,), z_tr.size(1), device=device, dtype=torch.long)
        h = enc(g_tr[idx].float().to(device), z_tr[idx].float().to(device), lens,
                lesion_weights=w_tr[idx].float().to(device))
        tb = teacher[idx][:, idx].to(device).float()
        ld = listwise_kl_loss(h, tb, temp_target=0.1, temp_pred=0.1)
        V = cause_embs.size(0)
        pos = torch.zeros(b, V, dtype=torch.bool, device=device)
        for i, ci in enumerate(idx.tolist()):
            cc = cidx_tr[ci]
            if cc:
                pos[i, torch.tensor(cc, dtype=torch.long, device=device)] = True
        li = case_cause_infonce_loss(h, cause_embs, pos, temp=infonce_t)
        loss = ld + infonce_w * li
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", default="diagnosis_model/cause_inference/outputs/case_db_jointDistRawP")
    ap.add_argument("--soft_dir", default="diagnosis_model/grod/outputs/soft_inputs")
    ap.add_argument("--enc_ckpt", default="diagnosis_model/cause_inference/outputs/encoder_grod_soft/best_encoder.pt")
    ap.add_argument("--ceah_ckpt", default="diagnosis_model/cause_inference/outputs/ceah_grod_soft/best_ceah.pt")
    ap.add_argument("--teacher_path", default="diagnosis_model/cause_inference/outputs/case_db_jointDistRawP/teacher_train_train.pt")
    ap.add_argument("--output_dir", default="diagnosis_model/cause_inference/outputs/e2e_soft")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--top_k_lesions", type=int, default=32)
    ap.add_argument("--ceah_lr", type=float, default=5e-5)
    ap.add_argument("--enc_lr", type=float, default=1e-4)
    ap.add_argument("--freeze_enc", action="store_true", help="CEAH-only finetune on soft pool")
    ap.add_argument("--ceah_bs", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=2)

    train_cases = torch.load(Path(args.case_db_dir) / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(Path(args.case_db_dir) / "valid_cases.pt", weights_only=False)
    cpack = torch.load(Path(args.case_db_dir) / "cause_text_embs.pt", weights_only=False)
    cause_embs = F.normalize(cpack["embeddings"].float(), dim=-1).to(device)
    cause_embs_raw = cpack["embeddings"].float().to(device)   # CEAH cause_proj expects raw
    cause_texts = cpack["texts"]
    in_dim = cause_embs.size(-1)
    K = args.top_k_lesions
    suffix = "_smoke" if args.limit else ""
    g_tr, z_tr, w_tr, cidx_tr = load_soft(Path(args.soft_dir) / f"train{suffix}.pt")
    g_va, z_va, w_va, _ = load_soft(Path(args.soft_dir) / f"valid{suffix}.pt")
    teacher = torch.load(args.teacher_path, weights_only=False, map_location="cpu")["scores"]
    if args.limit:
        train_cases, valid_cases = train_cases[:args.limit], valid_cases[:args.limit]
        g_tr, z_tr, w_tr, cidx_tr = g_tr[:args.limit], z_tr[:args.limit], w_tr[:args.limit], cidx_tr[:args.limit]
        g_va, z_va, w_va = g_va[:args.limit], z_va[:args.limit], w_va[:args.limit]
    teacher = teacher[:g_tr.size(0), :g_tr.size(0)]

    enc_pkg = torch.load(args.enc_ckpt, weights_only=False, map_location="cpu")
    enc_cfg = EncoderConfig(**enc_pkg["encoder_config"])
    enc = build_encoder(enc_cfg).to(device); enc.load_state_dict(enc_pkg["encoder_state"])
    ceah = CEAH(global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
                common_dim=256, hidden_dim=512, dropout=0.1,
                attribution_mode="softmax", scoring_mode="multiplicative").to(device)
    ceah.load_state_dict(torch.load(args.ceah_ckpt, map_location=device))

    enc_opt = torch.optim.AdamW(enc.parameters(), lr=args.enc_lr, weight_decay=1e-2)
    ceah_opt = torch.optim.AdamW(ceah.parameters(), lr=args.ceah_lr, weight_decay=1e-2)
    rng = np.random.default_rng(0)

    def evaluate(tag):
        bank = encode_all_soft(enc, g_tr, z_tr, w_tr, device).to(device)
        H_va = encode_all_soft(enc, g_va, z_va, w_va, device)
        m = soft_eval(ceah, enc, bank, train_cases, valid_cases, cause_embs_raw,
                      g_va, z_va, w_va, device, K, top_k_cases=args.top_k_cases,
                      max_queries=min(300, len(valid_cases)))
        ceah.eval()   # soft_eval leaves train(); faithfulness must run dropout-off
        fm = faith_mode(ceah, H_va, bank, train_cases, valid_cases, cause_embs_raw, cause_texts,
                        g_va, z_va, w_va, device, "top%d" % K, max_queries=min(300, len(valid_cases)))
        print(f"  [{tag}] sem_MRR={m['sem_MRR']:.4f} R@10={m['sem_R@10']:.4f} "
              f"R@1={m['sem_R@1']:.4f} | no_lesion={fm['no_lesion']:+.4f} "
              f"no_top_α={fm['no_top_α']:+.4f} no_random={fm['no_random']:+.4f}")
        return m, bank

    print("[warm-start staged ckpts]"); _, bank = evaluate("ep0")
    best = -1.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        pool = build_soft_pool(bank, train_cases, cause_embs, args.top_k_cases)
        ds = SoftCEAHDataset(train_cases, pool, g_tr, z_tr, w_tr, K)
        loader = DataLoader(ds, batch_size=args.ceah_bs, shuffle=True, drop_last=True,
                            collate_fn=make_soft_collate(0.5, in_dim, cause_embs_raw, rng))
        lc = ceah_epoch(ceah, ceah_opt, loader, cause_embs_raw, device, in_dim, K)
        le = (enc_epoch(enc, enc_opt, teacher, g_tr, z_tr, w_tr, cidx_tr, cause_embs_raw, device)
              if not args.freeze_enc else 0.0)
        m, bank = evaluate(f"ep{ep}")
        print(f"  (ceah_loss={lc:.4f} enc_loss={le:.4f} dur={time.time()-t0:.0f}s)")
        if m["sem_MRR"] > best:
            best = m["sem_MRR"]
            torch.save({"encoder_state": {k: v.cpu() for k, v in enc.state_dict().items()},
                        "encoder_config": vars(enc_cfg)}, out / "best_encoder.pt")
            torch.save(ceah.state_dict(), out / "best_ceah.pt")
    print(f"[done] best sem_MRR={best:.4f} -> {out}")


if __name__ == "__main__":
    main()
