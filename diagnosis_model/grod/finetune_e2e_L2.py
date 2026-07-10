"""L2 end-to-end probe — unfreeze the GROD semantic + objectness heads.

The L1 finetune (``finetune_e2e_grace``) treats objectness logits as fixed
cached inputs: gradient reaches only the Region Gate + aggregator, and the
per-region semantic embeddings ``z`` are frozen. L2 runs the semantic head
(``semantic_embed``) and objectness head (``class_embed``) LIVE on the cached
decoder query features ``hs`` (from ``extract_hs_all``), so the retrieval
gradient additionally shapes those two heads. Backbone/decoder stay frozen
(``hs`` is cached), so the box predictions — and hence detection mAP — are
untouched by construction; this is the bbox-safe rung below full L3.

A/B (same warm-start aggregator + gate, same protocol):
  --freeze_heads      L1 control: heads frozen (live path reproduces cached z)
  (default)           L2: heads trainable, L_retr only
  --lambda_sym > 0    L2+sym: add symptom grounding to protect the head

In-loop metrics: sem R@10 / MRR (retrieval) + symptom top-1 on matched lesion
queries (does the semantic head drift off symptom grounding?). Faithfulness
(needs a CEAM retrain on the resulting bank) is a follow-up, not in this loop.

Run (smoke):  $PY -m diagnosis_model.grod.finetune_e2e_L2 --limit 512 --epochs 3
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.models.case_encoder import (
    EncoderConfig, build_encoder, listwise_kl_loss, case_cause_infonce_loss,
)
from diagnosis_model.grod.region_gate import RegionGate
from diagnosis_model.grod.finetune_e2e_grace import retrieval_eval
from diagnosis_model.grod.train_case_encoder_soft import load_soft


def load_heads(joint_ckpt, anchors_path, device):
    """Build the joint model, lift out semantic_embed (Linear Hd->768) and the
    class_embed (Linear Hd->C) whose col 0 is ABNORMAL objectness; drop the net."""
    os.environ["RFDETR_SEMANTIC_DIM"] = "768"
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(anchors_path)
    os.environ["RFDETR_GLOBAL_DIM"] = "768"
    from diagnosis_model.grod.extract_hs import load_detector, find_class_embed
    rf, net, dev, means, stds, res = load_detector(joint_ckpt, device)
    sem = copy.deepcopy(net.semantic_embed).to(device).float()
    cls = copy.deepcopy(find_class_embed(net)).to(device).float()
    del net, rf
    torch.cuda.empty_cache()
    return sem, cls


def heads_fwd(sem, cls, hs):
    """hs [B,Q,Hd] float -> z [B,Q,768] L2-normed, o [B,Q] objectness logit."""
    z = F.normalize(sem(hs), dim=-1)
    o = cls(hs)[..., 0]
    return z, o


def load_hs(path, limit=0):
    data = torch.load(path, weights_only=False)
    if limit:
        data = data[:limit]
    hs = torch.stack([d["hs_all"] for d in data])          # [N,Q,Hd] bf16
    mq = [d["matched_qidx"] for d in data]
    mc = [d["matched_cat"] for d in data]
    return hs, mq, mc


@torch.no_grad()
def encode_bank(enc, gate, sem, cls, g, hs, device, bs=256, use_gate=True):
    enc.eval(); sem.eval(); cls.eval()
    out = []
    for s in range(0, g.size(0), bs):
        e = min(s + bs, g.size(0))
        z, o = heads_fwd(sem, cls, hs[s:e].float().to(device))
        w = gate(o) if use_gate else o.sigmoid()
        lens = torch.full((e - s,), z.size(1), device=device, dtype=torch.long)
        h = enc(g[s:e].float().to(device), z, lens, lesion_weights=w)
        out.append(h.float().cpu())
    enc.train()
    return torch.cat(out)


@torch.no_grad()
def symptom_top1(sem, hs, mq, mc, anchors, device, bs=256):
    """argmax(z·anchor) == symptom category on matched lesion queries."""
    sem.eval()
    correct = tot = 0
    for s in range(0, hs.size(0), bs):
        e = min(s + bs, hs.size(0))
        z = F.normalize(sem(hs[s:e].float().to(device)), dim=-1)   # [b,Q,768]
        for j in range(e - s):
            qidx, cat = mq[s + j], mc[s + j]
            if len(qidx) == 0:
                continue
            logit = z[j, qidx.to(device)] @ anchors.T              # [n,15]
            correct += (logit.argmax(-1).cpu() == cat).sum().item()
            tot += len(cat)
    return correct / max(tot, 1)


def sym_ce_loss(z_b, idx, mq, mc, anchors, device, temp=0.07):
    """Cross-entropy of matched-query z·anchor vs symptom label (grounding)."""
    zs, tgt = [], []
    for j, ci in enumerate(idx.tolist()):
        qidx, cat = mq[ci], mc[ci]
        if len(qidx) == 0:
            continue
        zs.append(z_b[j, qidx.to(device)]); tgt.append(cat.to(device))
    if not zs:
        return torch.zeros((), device=device)
    logits = torch.cat(zs) @ anchors.T / temp
    return F.cross_entropy(logits, torch.cat(tgt))


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--soft_dir", default=f"{ART}/db/soft_inputs")
    ap.add_argument("--hs_dir", default=f"{ART}/db/hs_all")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--teacher_path", default=f"{ART}/db/case_db_jointDistRawP/teacher_train_train.pt")
    ap.add_argument("--enc_ckpt", default=f"{ART}/models/encoder_grod_soft/best_encoder.pt")
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--output_dir", default=f"{ART}/models/grace_e2e_L2")
    ap.add_argument("--freeze_heads", action="store_true", help="L1 control via live path")
    ap.add_argument("--lambda_sym", type=float, default=0.0, help=">0 adds symptom grounding")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gate_lr", type=float, default=1e-3)
    ap.add_argument("--head_lr", type=float, default=1e-5)
    ap.add_argument("--init_temp", type=float, default=0.3)
    ap.add_argument("--infonce_weight", type=float, default=0.5)
    ap.add_argument("--infonce_temp", type=float, default=0.07)
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0, help="0 = entropy (non-deterministic)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    device = args.device
    if args.seed:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=2)

    train_cases = torch.load(Path(args.case_db_dir) / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(Path(args.case_db_dir) / "valid_cases.pt", weights_only=False)
    cpack = torch.load(Path(args.case_db_dir) / "cause_text_embs.pt", weights_only=False)
    cause_embs = F.normalize(cpack["embeddings"].float(), dim=-1).to(device)
    anchors = F.normalize(torch.load(args.anchors, weights_only=False)["anchor_embs"].float(), dim=-1).to(device)

    g_tr, _z, _w, cidx_tr = load_soft(Path(args.soft_dir) / "train.pt")
    g_va, _zv, _wv, _ = load_soft(Path(args.soft_dir) / "valid.pt")
    hs_tr, mq_tr, mc_tr = load_hs(Path(args.hs_dir) / "hs_all_train.pt", args.limit)
    hs_va, mq_va, mc_va = load_hs(Path(args.hs_dir) / "hs_all_valid.pt", args.limit)
    teacher = torch.load(args.teacher_path, weights_only=False, map_location="cpu")["scores"]
    if args.limit:
        n = args.limit
        train_cases, valid_cases = train_cases[:n], valid_cases[:n]
        g_tr, cidx_tr, g_va = g_tr[:n], cidx_tr[:n], g_va[:n]
    teacher = teacher[:g_tr.size(0), :g_tr.size(0)]
    N, Q = hs_tr.size(0), hs_tr.size(1)
    assert hs_tr.size(0) == g_tr.size(0), f"hs/g misalign {hs_tr.size(0)} vs {g_tr.size(0)}"
    print(f"[data] train={N} valid={g_va.size(0)} Q={Q} freeze_heads={args.freeze_heads} lambda_sym={args.lambda_sym}")

    # warm-start aggregator + gate from production encoder
    pkg = torch.load(args.enc_ckpt, weights_only=False, map_location="cpu")
    enc_cfg = EncoderConfig(**pkg["encoder_config"])
    enc = build_encoder(enc_cfg).to(device); enc.load_state_dict(pkg["encoder_state"])
    gate = RegionGate(init_temp=args.init_temp, init_sink=0.0).to(device)
    if "gate_state" in pkg:
        gate.load_state_dict(pkg["gate_state"]); print("[warm-start] gate <- enc_ckpt")
    sem, cls = load_heads(args.joint_ckpt, args.anchors, device)
    for p in sem.parameters(): p.requires_grad = not args.freeze_heads
    for p in cls.parameters(): p.requires_grad = not args.freeze_heads

    pgroups = [{"params": enc.parameters(), "lr": args.lr},
               {"params": gate.parameters(), "lr": args.gate_lr}]
    if not args.freeze_heads:
        pgroups.append({"params": list(sem.parameters()) + list(cls.parameters()), "lr": args.head_lr})
    opt = torch.optim.AdamW(pgroups, weight_decay=1e-2)

    def evaluate(tag):
        bank = encode_bank(enc, gate, sem, cls, g_tr, hs_tr, device)
        H_va = encode_bank(enc, gate, sem, cls, g_va, hs_va, device)
        m = retrieval_eval(bank.to(device), H_va, train_cases, valid_cases,
                           cause_embs, device, top_k_cases=args.top_k_cases)
        acc = symptom_top1(sem, hs_va, mq_va, mc_va, anchors, device)
        print(f"  [{tag}] sem R@1={m['R@1']:.4f} R@10={m['R@10']:.4f} MRR={m['MRR']:.4f}  sym_top1={acc:.4f}")
        return m, acc, bank

    print("[ep0] warm-start (heads live, untrained):")
    evaluate("ep0")

    log = []
    best = -1.0
    for ep in range(1, args.epochs + 1):
        enc.train(); sem.train(); cls.train(); t0 = time.time(); perm = torch.randperm(N); losses = []
        for bs in range(0, N - args.batch_size + 1, args.batch_size):
            idx = perm[bs:bs + args.batch_size]; b = idx.size(0)
            z, o = heads_fwd(sem, cls, hs_tr[idx].float().to(device))
            w = gate(o)
            lens = torch.full((b,), Q, device=device, dtype=torch.long)
            h = enc(g_tr[idx].float().to(device), z, lens, lesion_weights=w)
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
            if args.lambda_sym > 0 and not args.freeze_heads:
                loss = loss + args.lambda_sym * sym_ce_loss(z, idx, mq_tr, mc_tr, anchors, device)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(gate.parameters())
                + list(sem.parameters()) + list(cls.parameters()), 1.0)
            opt.step(); losses.append(loss.item())
        m, acc, bank = evaluate(f"ep{ep}")
        print(f"    loss={np.mean(losses):.4f} τ={gate.temp.item():.4f} dur={time.time()-t0:.0f}s")
        log.append({"epoch": ep, "loss": float(np.mean(losses)), "sym_top1": acc, **{f"sem_{k}": v for k, v in m.items()}})
        if m["MRR"] > best:
            best = m["MRR"]
            torch.save({"encoder_state": {k: v.cpu() for k, v in enc.state_dict().items()},
                        "encoder_config": vars(enc_cfg),
                        "gate_state": {k: v.cpu() for k, v in gate.state_dict().items()},
                        "gate_init_temp": args.init_temp,
                        "semantic_embed_state": {k: v.cpu() for k, v in sem.state_dict().items()},
                        "class_embed_state": {k: v.cpu() for k, v in cls.state_dict().items()}},
                       out / "best.pt")
    json.dump(log, open(out / "train_log.json", "w"), indent=2)
    print(f"[done] best sem MRR={best:.4f} -> {out/'best.pt'}")


if __name__ == "__main__":
    main()
