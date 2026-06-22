"""End-to-end (L1) — train GRACE aggregator + Region Gate jointly.

The production soft pipeline freezes a sigmoid gate: extract_soft_inputs.py dumps
``w = sigmoid(obj_logits)`` to disk and the aggregator trains on that fixed w
([[project_region_gate_e2e]]). Here we put the gate back in the live graph:

  * recover the raw objectness logits exactly from the cached soft weights
    (o = logit(w)), so we DON'T have to re-run GROD;
  * replace the frozen sigmoid with a learnable RegionGate (∅-sink softmax_τ);
  * co-train RegionGate (τ, ∅) + DeepSets aggregator with the GRACE case-encoding
    objective (listwise-KL teacher distill + case-cause InfoNCE).

This is "L1" end-to-end: gradient reaches the gate (τ/∅) and the aggregator.
"L2" (gradient into the GROD objectness/semantic heads) additionally needs the
decoder query features hs — the hook point is the input to ``class_embed`` (see
extract_hs.py) — and is the follow-up, not done here.

GROD stays frozen; obj_logits are fixed inputs (we only learn how the gate maps
them to weights). Compares the learned RegionGate bank against the baseline
sigmoid-w bank on case->cause retrieval.

Run (smoke):  $PY -m diagnosis_model.grod.finetune_e2e_grace --limit 512 --epochs 3
Run (full):   $PY -m diagnosis_model.grod.finetune_e2e_grace
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.models.case_encoder import (
    EncoderConfig, build_encoder, listwise_kl_loss, case_cause_infonce_loss,
)
from diagnosis_model.cause_inference.phase1_baseline import build_candidate_pool
from diagnosis_model.grod.train_case_encoder_soft import load_soft
from diagnosis_model.grod.region_gate import RegionGate


def recover_logits(w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """o = logit(w); exact inverse of w = sigmoid(o) (clamped for w∈{0,1})."""
    w = w.float().clamp(eps, 1.0 - eps)
    return torch.log(w) - torch.log1p(-w)


def encode_bank(enc, gate, g, z, logits, device, bs=256, use_gate=True):
    """Encode all cases -> [N, D] L2-normed bank (no grad). use_gate=False keeps
    the original sigmoid weights (baseline)."""
    enc.eval()
    out = []
    with torch.no_grad():
        for s in range(0, g.size(0), bs):
            e = min(s + bs, g.size(0))
            lg = logits[s:e].to(device)
            w = gate(lg) if use_gate else lg.sigmoid()
            lens = torch.full((e - s,), z.size(1), device=device, dtype=torch.long)
            h = enc(g[s:e].float().to(device), z[s:e].float().to(device), lens,
                    lesion_weights=w)
            out.append(h.float().cpu())
    enc.train()
    return torch.cat(out)


@torch.no_grad()
def retrieval_eval(bank, H_va, train_cases, valid_cases, cause_embs, device,
                   top_k_cases=20, sem_thresh=0.95, max_q=300, ks=(1, 10)):
    """case->cause sem R@K: top-k cases -> sim-weighted candidate cause scores ->
    hit if a GT cause is within top-K ranked candidates (cos>=thresh)."""
    nq = min(max_q, H_va.size(0))
    hits = {k: 0 for k in ks}
    mrr = 0.0
    for qi in range(nq):
        sims = (H_va[qi:qi + 1].to(device) @ bank.T.to(device)).squeeze(0)  # [N]
        top = sims.topk(top_k_cases)
        cand = build_candidate_pool(top.indices.cpu().numpy(), train_cases)
        if not cand:
            continue
        # score each candidate cause = sum of case sims where it appears
        score = {}
        for ci, sj in zip(top.indices.tolist(), top.values.tolist()):
            for cc in train_cases[ci]["cause_emb_indices"]:
                score[cc] = score.get(cc, 0.0) + max(sj, 0.0)
        ranked = sorted(score, key=score.get, reverse=True)
        gt = valid_cases[qi]["cause_emb_indices"]
        gt_e = cause_embs.index_select(0, torch.as_tensor(gt, device=device))
        ranked_e = cause_embs.index_select(0, torch.as_tensor(ranked, device=device))
        cov = (gt_e @ ranked_e.T).max(dim=0).values  # [len(ranked)] best gt-cos per rank
        first = (cov >= sem_thresh).nonzero()
        if first.numel():
            r = first[0].item() + 1
            mrr += 1.0 / r
            for k in ks:
                if r <= k:
                    hits[k] += 1
    return {f"R@{k}": hits[k] / max(nq, 1) for k in ks} | {"MRR": mrr / max(nq, 1)}


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--soft_dir", default=f"{ART}/db/soft_inputs")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--teacher_path", default=f"{ART}/db/case_db_jointDistRawP/teacher_train_train.pt")
    ap.add_argument("--enc_ckpt", default=f"{ART}/models/encoder_grod_soft/best_encoder.pt",
                    help="warm-start aggregator; omit/empty to train from scratch")
    ap.add_argument("--output_dir", default=f"{ART}/models/grace_e2e")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gate_lr", type=float, default=1e-3)
    ap.add_argument("--init_temp", type=float, default=0.3)
    ap.add_argument("--infonce_weight", type=float, default=0.5)
    ap.add_argument("--infonce_temp", type=float, default=0.07)
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--gate_mode", choices=["region", "sigmoid"], default="region",
                    help="sigmoid = attribution control: frozen sigmoid gate, "
                         "aggregator trained the same #epochs (isolates Region Gate)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    region = args.gate_mode == "region"

    device = args.device
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=2)

    train_cases = torch.load(Path(args.case_db_dir) / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(Path(args.case_db_dir) / "valid_cases.pt", weights_only=False)
    cpack = torch.load(Path(args.case_db_dir) / "cause_text_embs.pt", weights_only=False)
    cause_embs = F.normalize(cpack["embeddings"].float(), dim=-1).to(device)

    g_tr, z_tr, w_tr, cidx_tr = load_soft(Path(args.soft_dir) / "train.pt")
    g_va, z_va, w_va, _ = load_soft(Path(args.soft_dir) / "valid.pt")
    teacher = torch.load(args.teacher_path, weights_only=False, map_location="cpu")["scores"]
    if args.limit:
        n = args.limit
        train_cases, valid_cases = train_cases[:n], valid_cases[:n]
        g_tr, z_tr, w_tr, cidx_tr = g_tr[:n], z_tr[:n], w_tr[:n], cidx_tr[:n]
        g_va, z_va, w_va = g_va[:n], z_va[:n], w_va[:n]
    teacher = teacher[:g_tr.size(0), :g_tr.size(0)]
    o_tr = recover_logits(w_tr)          # [N, Q] raw objectness logits
    o_va = recover_logits(w_va)
    N, Q = o_tr.shape
    print(f"[data] train={N} valid={g_va.size(0)} Q={Q}")

    # aggregator (warm-start) + fresh Region Gate
    if args.enc_ckpt and Path(args.enc_ckpt).exists():
        pkg = torch.load(args.enc_ckpt, weights_only=False, map_location="cpu")
        enc_cfg = EncoderConfig(**pkg["encoder_config"])
        enc = build_encoder(enc_cfg).to(device); enc.load_state_dict(pkg["encoder_state"])
        print(f"[warm-start] aggregator <- {args.enc_ckpt}")
    else:
        enc_cfg = EncoderConfig(); enc = build_encoder(enc_cfg).to(device)
        print("[scratch] aggregator")
    gate = RegionGate(init_temp=args.init_temp, init_sink=0.0).to(device)

    pgroups = [{"params": enc.parameters(), "lr": args.lr}]
    if region:
        pgroups.append({"params": gate.parameters(), "lr": args.gate_lr})
    opt = torch.optim.AdamW(pgroups, weight_decay=1e-2)

    cause_embs_raw = cpack["embeddings"].float().to(device)  # InfoNCE on normed below

    def evaluate(tag, use_gate=True):
        bank = encode_bank(enc, gate, g_tr, z_tr, o_tr, device, use_gate=use_gate)
        H_va = encode_bank(enc, gate, g_va, z_va, o_va, device, use_gate=use_gate)
        m = retrieval_eval(bank.to(device), H_va, train_cases, valid_cases,
                           cause_embs, device, top_k_cases=args.top_k_cases)
        print(f"  [{tag}] sem R@1={m['R@1']:.4f} R@10={m['R@10']:.4f} MRR={m['MRR']:.4f}"
              + ("" if use_gate else "  (baseline sigmoid w)"))
        return m, bank

    print(f"[mode] gate_mode={args.gate_mode}")
    print("[baseline] warm-start aggregator + frozen sigmoid gate:")
    evaluate("sigmoid", use_gate=False)
    print(f"[ep0] same aggregator, untrained gate (use_gate={region}):")
    _, bank = evaluate("ep0", use_gate=region)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        enc.train(); t0 = time.time(); perm = torch.randperm(N); losses = []
        gnorm = 0.0
        for bs in range(0, N - args.batch_size + 1, args.batch_size):
            idx = perm[bs:bs + args.batch_size]; b = idx.size(0)
            o_b = o_tr[idx].to(device)
            w = gate(o_b) if region else o_b.sigmoid()           # live gate vs frozen sigmoid
            lens = torch.full((b,), Q, device=device, dtype=torch.long)
            h = enc(g_tr[idx].float().to(device), z_tr[idx].float().to(device),
                    lens, lesion_weights=w)
            tb = teacher[idx][:, idx].to(device).float()
            ld = listwise_kl_loss(h, tb, temp_target=0.1, temp_pred=0.1)
            V = cause_embs.size(0)
            pos = torch.zeros(b, V, dtype=torch.bool, device=device)
            for i, ci in enumerate(idx.tolist()):
                cc = cidx_tr[ci]
                if cc:
                    pos[i, torch.tensor(cc, dtype=torch.long, device=device)] = True
            li = case_cause_infonce_loss(h, cause_embs, pos, temp=args.infonce_temp)
            loss = ld + args.infonce_weight * li
            opt.zero_grad(); loss.backward()
            if gate.log_temp.grad is not None:
                gnorm = float(torch.cat([p.grad.flatten() for p in gate.parameters()]).norm())
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(gate.parameters()), 1.0)
            opt.step(); losses.append(loss.item())
        m, bank = evaluate(f"ep{ep}", use_gate=region)
        print(f"    loss={np.mean(losses):.4f} τ={gate.temp.item():.4f} "
              f"∅={gate.sink.item():+.4f} grad‖gate‖={gnorm:.4f} dur={time.time()-t0:.0f}s")
        if m["MRR"] > best:
            best = m["MRR"]
            torch.save({"encoder_state": {k: v.cpu() for k, v in enc.state_dict().items()},
                        "encoder_config": vars(enc_cfg),
                        "gate_state": {k: v.cpu() for k, v in gate.state_dict().items()},
                        "gate_init_temp": args.init_temp},
                       out / "best.pt")
    print(f"[done] best sem MRR={best:.4f} -> {out/'best.pt'}")


if __name__ == "__main__":
    main()
