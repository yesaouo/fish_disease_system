"""Soft-pipeline retrain — step #4: train CEAH on soft GROD lesion evidence.

Same supervision as the hard CEAH (reuses train_candidate_pool.pt negatives +
cause table + BCE/sparsity loss); only the lesion evidence changes from clean
GT lesions to the soft detector outputs from extract_soft_inputs.py. To keep the
per-(query,candidate) replication tractable we feed the **top-K queries by
objectness w** (real lesions are few; w≈0 queries contribute ~0 through CEAH's
log-w gate). CEAH.forward gets the matching ``lesion_weights`` (added,
backward-compatible).

Eval (early-stop on sem_MRR) uses the production soft cascade: soft-encode the
valid query → retrieve top-k over bank_z_soft → CEAH soft-attribution rank.

Output: outputs/ceah_grod_soft/best_ceah.pt

Run from repo root (SDM env):
    $PY -m diagnosis_model.grod.train_ceah_soft
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from diagnosis_model.cause_inference.models import CEAH
from diagnosis_model.cause_inference.phase1_baseline import (
    MISS_RANK, add_recall_at_ks, build_candidate_pool, summarize_rank_metric,
)
from diagnosis_model.cause_inference.models.case_encoder import EncoderConfig, build_encoder
from diagnosis_model.grod.train_case_encoder_soft import encode_all_soft, load_soft


def topk_by_w(z_all: torch.Tensor, w: torch.Tensor, K: int):
    """z_all [Q,D], w [Q] -> (z_k [K,D], w_k [K]) keeping the K highest-w queries."""
    K = min(K, w.numel())
    idx = torch.topk(w, K).indices
    return z_all[idx], w[idx]


class SoftCEAHDataset(Dataset):
    def __init__(self, train_cases, train_pool, g, z, w, K):
        self.cases, self.pool = train_cases, train_pool
        self.g, self.z, self.w, self.K = g, z, w, K

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        c, p = self.cases[idx], self.pool[idx]
        z_k, w_k = topk_by_w(self.z[idx].float(), self.w[idx].float(), self.K)
        return {
            "global_emb": self.g[idx].float(),                 # pred_global
            "text_colloquial_emb": c.get("text_colloquial_emb"),
            "text_medical_emb": c.get("text_medical_emb"),
            "lesion_z": z_k,                                    # [K, D]
            "lesion_w": w_k,                                    # [K]
            "candidate_indices": p["candidate_cause_indices"],
            "positive_mask": p["positive_mask"],
        }


def make_soft_collate(text_dropout, in_dim, cause_table_embs, rng):
    def collate(batch: List[dict]) -> dict:
        B = len(batch)
        K = batch[0]["lesion_z"].size(0)
        max_P = max(int(b["candidate_indices"].numel()) for b in batch)
        D = in_dim
        global_pad = torch.zeros(B, D)
        lesion_pad = torch.zeros(B, K, D)
        lesion_w = torch.zeros(B, K)
        text_pad = torch.zeros(B, D)
        text_present = torch.zeros(B, dtype=torch.bool)
        cand_pad = torch.zeros(B, max_P, D)
        cand_mask = torch.zeros(B, max_P, dtype=torch.bool)
        target_pad = torch.zeros(B, max_P, dtype=torch.float32)
        for i, b in enumerate(batch):
            global_pad[i] = b["global_emb"]
            lesion_pad[i] = b["lesion_z"]
            lesion_w[i] = b["lesion_w"]
            opts = [t for t in (b.get("text_colloquial_emb"), b.get("text_medical_emb")) if t is not None]
            if opts and rng.random() >= text_dropout:
                text_pad[i] = opts[int(rng.integers(len(opts)))]
                text_present[i] = True
            ci = b["candidate_indices"]; P = int(ci.numel())
            if P > 0:
                cand_pad[i, :P] = cause_table_embs[ci]
            cand_mask[i, :P] = True
            target_pad[i, :P] = b["positive_mask"].float()
        return {"global_emb": global_pad, "lesion_z": lesion_pad, "lesion_w": lesion_w,
                "text_emb": text_pad, "text_present": text_present,
                "cand_embs": cand_pad, "cand_mask": cand_mask, "targets": target_pad}
    return collate


@torch.no_grad()
def soft_eval(ceah, encoder, bank_z, train_cases, valid_cases, cause_embs,
              g_va, z_va, w_va, device, K, top_k_cases=20, max_queries=300,
              semantic_threshold=0.95, ks=(1, 5, 10, 20, 100)):
    ceah.eval()
    H_va = encode_all_soft(encoder, g_va, z_va, w_va, device)         # [Nv, D]
    bank_z = bank_z.to(device)
    nq = min(max_queries, len(valid_cases))
    sem_ranks, cov = [], []
    for qi in range(nq):
        q = valid_cases[qi]
        gt = q["cause_emb_indices"]
        if not gt:
            continue
        sims = H_va[qi:qi + 1].to(device) @ bank_z.T                  # [1, Nt]
        top_idx = sims[0].topk(top_k_cases).indices.cpu().numpy()
        cand_idx = build_candidate_pool(top_idx, train_cases)         # np array
        P = len(cand_idx)
        if P == 0:
            for _ in gt:
                sem_ranks.append(MISS_RANK); cov.append(0)
            continue
        cand_embs = cause_embs[torch.as_tensor(cand_idx, device=device)]   # [P, D]
        z_k, w_k = topk_by_w(z_va[qi].float().to(device), w_va[qi].float().to(device), K)
        g_e = g_va[qi].float().to(device).unsqueeze(0).expand(P, -1)
        l_e = z_k.unsqueeze(0).expand(P, -1, -1).contiguous()
        l_w = w_k.unsqueeze(0).expand(P, -1).contiguous()
        l_m = torch.ones(P, z_k.size(0), dtype=torch.bool, device=device)
        t_e = torch.zeros(P, cand_embs.size(-1), device=device)
        t_p = torch.zeros(P, dtype=torch.bool, device=device)
        scores, _, _ = ceah(g_e, t_e, t_p, l_e, l_m, cand_embs, lesion_weights=l_w)
        order = scores.argsort(descending=True).cpu().numpy()
        sorted_cand_embs = cand_embs[torch.as_tensor(order[: max(ks)], device=device)]
        gt_e = F.normalize(cause_embs[torch.as_tensor(gt, device=device)], dim=-1)
        cos = gt_e @ F.normalize(sorted_cand_embs, dim=-1).T          # [G, <=maxK]
        match = cos >= semantic_threshold
        for gi in range(match.size(0)):
            hit = torch.nonzero(match[gi], as_tuple=False)
            if hit.numel() > 0:
                sem_ranks.append(float(hit[0].item()) + 1.0); cov.append(1)
            else:
                sem_ranks.append(MISS_RANK); cov.append(0)
    ceah.train()
    arr = np.asarray(sem_ranks, dtype=np.float64)
    block = summarize_rank_metric(arr, cov)
    m = {"sem_MRR": block["MRR"], "sem_coverage": block["coverage"]}
    add_recall_at_ks(m, arr, list(ks))
    for k in ks:
        m[f"sem_R@{k}"] = m.pop(f"R@{k}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--train_pool_path", default=f"{ART}/db/case_db_jointDistRawP/train_candidate_pool.pt")
    ap.add_argument("--soft_dir", default=f"{ART}/db/soft_inputs")
    ap.add_argument("--encoder_ckpt", default=f"{ART}/models/encoder_grod_soft/best_encoder.pt")
    ap.add_argument("--bank_path", default=f"{ART}/models/encoder_grod_soft/bank_z_soft.pt")
    ap.add_argument("--output_dir", default=f"{ART}/models/ceah_grod_soft")
    ap.add_argument("--top_k_lesions", type=int, default=32)
    # CEAH canonical (matches ceah_jointDistRawP)
    ap.add_argument("--common_dim", type=int, default=256)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attribution_mode", default="softmax")
    ap.add_argument("--scoring_mode", default="multiplicative")
    ap.add_argument("--lambda_sparsity", type=float, default=0.0)
    ap.add_argument("--text_dropout", type=float, default=0.5)
    # Rung 1: multi-positive listwise CE (cross-candidate competition) on top of BCE.
    # Ported from cause_inference/train_ceah.py; default 0.0 => loss byte-identical.
    ap.add_argument("--lambda_bce", type=float, default=1.0)
    ap.add_argument("--lambda_rank", type=float, default=0.0)
    ap.add_argument("--rank_temp", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--eval_top_k_cases", type=int, default=20)
    ap.add_argument("--eval_max_queries", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out_dir / "config.json", "w"), indent=2)
    device = args.device

    train_cases = torch.load(Path(args.case_db_dir) / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(Path(args.case_db_dir) / "valid_cases.pt", weights_only=False)
    train_pool = torch.load(args.train_pool_path, weights_only=False)["case_pool"]
    cause_embs_cpu = torch.load(Path(args.case_db_dir) / "cause_text_embs.pt", weights_only=False)["embeddings"].float()
    cause_embs = cause_embs_cpu.to(device)
    in_dim = cause_embs.size(-1)

    suffix = "_smoke" if args.limit else ""
    g_tr, z_tr, w_tr, _ = load_soft(Path(args.soft_dir) / f"train{suffix}.pt")
    g_va, z_va, w_va, _ = load_soft(Path(args.soft_dir) / f"valid{suffix}.pt")
    if args.limit:
        train_cases, train_pool = train_cases[:args.limit], train_pool[:args.limit]
        valid_cases = valid_cases[:args.limit]
        g_tr, z_tr, w_tr = g_tr[:args.limit], z_tr[:args.limit], w_tr[:args.limit]
        g_va, z_va, w_va = g_va[:args.limit], z_va[:args.limit], w_va[:args.limit]
    assert len(train_cases) == len(train_pool) == g_tr.size(0), "alignment mismatch"
    print(f"[load] train={len(train_cases)} valid={len(valid_cases)} K={args.top_k_lesions} "
          f"causes={in_dim and cause_embs.size(0)}")

    # soft encoder + bank for eval retrieval
    enc_pkg = torch.load(args.encoder_ckpt, weights_only=False, map_location="cpu")
    encoder = build_encoder(EncoderConfig(**enc_pkg["encoder_config"])).to(device).eval()
    encoder.load_state_dict(enc_pkg["encoder_state"])
    bank_z = torch.load(args.bank_path, weights_only=False)["bank_z"]
    if args.limit:
        bank_z = bank_z[:args.limit]

    ceah = CEAH(global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
                common_dim=args.common_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
                attribution_mode=args.attribution_mode, scoring_mode=args.scoring_mode).to(device)
    print(f"[model] CEAH params={sum(p.numel() for p in ceah.parameters()):,}")

    optim = torch.optim.AdamW(ceah.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = SoftCEAHDataset(train_cases, train_pool, g_tr, z_tr, w_tr, args.top_k_lesions)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        collate_fn=make_soft_collate(args.text_dropout, in_dim, cause_embs_cpu, rng))
    total_steps = len(loader) * args.epochs

    def lr_mult(step):
        if step < args.warmup_steps:
            return (step + 1) / max(args.warmup_steps, 1)
        prog = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))

    log_f = open(out_dir / "train_log.jsonl", "w")
    best_mrr, gstep = -1.0, 0
    for epoch in range(args.epochs):
        ceah.train(); losses = []; t0 = time.time()
        for batch in loader:
            g = batch["global_emb"].to(device); lz = batch["lesion_z"].to(device)
            lw = batch["lesion_w"].to(device); te = batch["text_emb"].to(device)
            tp = batch["text_present"].to(device); cand = batch["cand_embs"].to(device)
            cmask = batch["cand_mask"].to(device); targets = batch["targets"].to(device)
            B, max_P, _ = cand.shape; K = lz.size(1)
            g_r = g.unsqueeze(1).expand(B, max_P, in_dim).reshape(B * max_P, in_dim)
            t_r = te.unsqueeze(1).expand(B, max_P, in_dim).reshape(B * max_P, in_dim)
            tp_r = tp.unsqueeze(1).expand(B, max_P).reshape(B * max_P)
            lz_r = lz.unsqueeze(1).expand(B, max_P, K, in_dim).reshape(B * max_P, K, in_dim)
            lw_r = lw.unsqueeze(1).expand(B, max_P, K).reshape(B * max_P, K)
            lm_r = torch.ones(B * max_P, K, dtype=torch.bool, device=device)
            cand_f = cand.reshape(B * max_P, in_dim)
            scores, _, _ = ceah(g_r, t_r, tp_r, lz_r, lm_r, cand_f, lesion_weights=lw_r)
            scores = scores.view(B, max_P)
            cmf = cmask.float()
            bce = F.binary_cross_entropy(scores, targets, reduction="none")
            loss = args.lambda_bce * ((bce * cmf).sum() / cmf.sum().clamp_min(1.0))
            # Rung 1: multi-positive listwise CE — softmax(score/T) over the pool,
            # target = uniform over the binary positives (ported from train_ceah.py).
            if args.lambda_rank > 0:
                logits = (scores / args.rank_temp).masked_fill(~cmask, float("-inf"))
                logp = F.log_softmax(logits, dim=1)
                pos = (targets > 0.5) & cmask
                n_pos = pos.sum(dim=1)
                per_q = -torch.where(pos, logp, torch.zeros_like(logp)).sum(dim=1) / n_pos.clamp_min(1)
                valid_q = n_pos > 0
                if valid_q.any():
                    loss = loss + args.lambda_rank * per_q[valid_q].mean()
            for gp in optim.param_groups:
                gp["lr"] = args.lr * lr_mult(gstep)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ceah.parameters(), 1.0)
            optim.step()
            losses.append(loss.item()); gstep += 1

        row = {"epoch": epoch, "loss_cls": float(np.mean(losses)), "dur": round(time.time() - t0, 1)}
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            m = soft_eval(ceah, encoder, bank_z, train_cases, valid_cases, cause_embs,
                          g_va, z_va, w_va, device, args.top_k_lesions,
                          top_k_cases=args.eval_top_k_cases, max_queries=args.eval_max_queries)
            row.update(m)
            if m["sem_MRR"] > best_mrr:
                best_mrr = m["sem_MRR"]; row["saved_best"] = True
                torch.save(ceah.state_dict(), out_dir / "best_ceah.pt")
        torch.save(ceah.state_dict(), out_dir / "last_ceah.pt")
        log_f.write(json.dumps(row) + "\n"); log_f.flush()
        msg = f"[ep {epoch+1}/{args.epochs}] cls={row['loss_cls']:.4f} dur={row['dur']}s"
        if "sem_MRR" in row:
            msg += f" | sem_MRR={row['sem_MRR']:.3f} R@10={row.get('sem_R@10',0):.3f}"
            if row.get("saved_best"):
                msg += " *best*"
        print(msg)
    log_f.close()
    print(f"[done] best sem_MRR={best_mrr:.4f} -> {out_dir/'best_ceah.pt'}")


if __name__ == "__main__":
    main()
