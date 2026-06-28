"""End-to-end (L2) — gradient into the GROD objectness + semantic heads.

L1 (finetune_e2e_grace.py) trained only the gate + aggregator on cached head
OUTPUTS. L2 makes the two GROD heads themselves trainable, so the downstream
case-encoding loss back-props all the way into the detector's objectness head
(class_embed) and semantic head (semantic_embed) — the "整個模型端到端" the
professor asked for, with the Region Gate (∅-sink softmax_τ) sitting in the
live graph between objectness and aggregation.

Cheap, because we DON'T re-forward RF-DETR: both heads read the last decoder
layer hs[-1] (lwdetr.py), cached once by extract_hs_soft.py. We copy the two
Linear heads (weights init = current joint model) and train them on cached hs:

    hs[300,256] ─► class_embed   ─► obj_logits ─► RegionGate ─► w ─┐
              └──► semantic_embed ─► z (L2-norm) ──────────────────┴─► Aggregator ─► case vec

global g stays frozen (cached). Loss = listwise-KL teacher + case-cause InfoNCE.
Heads use a small LR (they also drive detection in deployment — retrieval loss
must not wreck boxes; this tension is the L2 ablation point).

Run (smoke): $PY -m diagnosis_model.grod.finetune_e2e_grace_l2 \
                 --hs_dir /tmp/hs_soft_smoke --limit 64 --epochs 3
Run (full):  $PY -m diagnosis_model.grod.finetune_e2e_grace_l2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diagnosis_model.cause_inference.models.case_encoder import (
    EncoderConfig, build_encoder, listwise_kl_loss, case_cause_infonce_loss,
)
from diagnosis_model.grod.train_case_encoder_soft import load_soft
from diagnosis_model.grod.region_gate import RegionGate
from diagnosis_model.grod.finetune_e2e_grace import retrieval_eval


def clone_heads(joint_ckpt, global_sd, anchors, device):
    """Load the joint model once; return fresh trainable copies of class_embed
    (objectness) and semantic_embed, weights initialised from the joint model."""
    import os
    os.environ["RFDETR_SEMANTIC_DIM"] = "768"
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = str(Path(anchors).resolve())
    os.environ["RFDETR_GLOBAL_DIM"] = "768"
    from diagnosis_model.grod.build import load_oavle
    net, _, _, _ = load_oavle(joint_ckpt, eval_mode=False)
    ce_src = None
    for name, mod in net.named_modules():
        if name.split(".")[-1] == "class_embed" and isinstance(mod, nn.Linear):
            ce_src = mod; break
    if ce_src is None:
        ce_src = net.class_embed
    se_src = net.semantic_embed
    obj_head = nn.Linear(ce_src.in_features, ce_src.out_features,
                         bias=ce_src.bias is not None).to(device)
    obj_head.load_state_dict(ce_src.state_dict())
    sem_head = nn.Linear(se_src.in_features, se_src.out_features,
                         bias=se_src.bias is not None).to(device)
    sem_head.load_state_dict(se_src.state_dict())
    del net
    return obj_head, sem_head


def heads_forward(obj_head, sem_head, gate, hs, device, use_gate=True):
    """hs[B,Q,256] -> (g-free) w[B,Q], z[B,Q,768]."""
    hs = hs.float().to(device)
    obj_logits = obj_head(hs)[..., 0]                         # [B,Q]
    z = F.normalize(sem_head(hs), dim=-1)                     # [B,Q,768]
    w = gate(obj_logits) if use_gate else obj_logits.sigmoid()
    return w, z


def encode_bank(enc, obj_head, sem_head, gate, g, hs, device, bs=128, use_gate=True):
    enc.eval(); out = []
    with torch.no_grad():
        for s in range(0, g.size(0), bs):
            e = min(s + bs, g.size(0))
            w, z = heads_forward(obj_head, sem_head, gate, hs[s:e], device, use_gate)
            lens = torch.full((e - s,), hs.size(1), device=device, dtype=torch.long)
            h = enc(g[s:e].float().to(device), z, lens, lesion_weights=w)
            out.append(h.float().cpu())
    enc.train()
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--hs_dir", default=f"{ART}/db/hs_soft")
    ap.add_argument("--soft_dir", default=f"{ART}/db/soft_inputs")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--teacher_path", default=f"{ART}/db/case_db_jointDistRawP/teacher_train_train.pt")
    ap.add_argument("--enc_ckpt", default=f"{ART}/models/encoder_grod_soft/best_encoder.pt")
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--global_sd", default=f"{ART}/models/distilled_global_rawP/global_embed_state_dict.pt")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--output_dir", default=f"{ART}/models/grace_e2e_l2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)        # aggregator
    ap.add_argument("--gate_lr", type=float, default=1e-3)
    ap.add_argument("--head_lr", type=float, default=1e-5)   # heads: small, also drive detection
    ap.add_argument("--init_temp", type=float, default=0.3)
    ap.add_argument("--infonce_weight", type=float, default=0.5)
    ap.add_argument("--infonce_temp", type=float, default=0.07)
    ap.add_argument("--top_k_cases", type=int, default=20)
    ap.add_argument("--freeze_heads", action="store_true",
                    help="L2 ablation -> reduces to L1 (heads frozen)")
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

    g_tr, _, _, cidx_tr = load_soft(Path(args.soft_dir) / "train.pt")
    g_va, _, _, _ = load_soft(Path(args.soft_dir) / "valid.pt")
    hs_tr = torch.load(Path(args.hs_dir) / "hs_train.pt", weights_only=False)["hs"]
    hs_va = torch.load(Path(args.hs_dir) / "hs_valid.pt", weights_only=False)["hs"]
    teacher = torch.load(args.teacher_path, weights_only=False, map_location="cpu")["scores"]
    if args.limit:
        n = args.limit
        train_cases, valid_cases = train_cases[:n], valid_cases[:n]
        g_tr, cidx_tr, hs_tr = g_tr[:n], cidx_tr[:n], hs_tr[:n]
        g_va, hs_va = g_va[:n], hs_va[:n]
    assert hs_tr.size(0) == g_tr.size(0), "hs and soft g misaligned"
    teacher = teacher[:g_tr.size(0), :g_tr.size(0)]
    N, Q, _ = hs_tr.shape
    print(f"[data] train={N} valid={g_va.size(0)} Q={Q}")

    pkg = torch.load(args.enc_ckpt, weights_only=False, map_location="cpu")
    enc_cfg = EncoderConfig(**pkg["encoder_config"])
    enc = build_encoder(enc_cfg).to(device); enc.load_state_dict(pkg["encoder_state"])
    gate = RegionGate(init_temp=args.init_temp, init_sink=0.0).to(device)
    obj_head, sem_head = clone_heads(args.joint_ckpt, args.global_sd, args.anchors, device)
    if args.freeze_heads:
        for p in (*obj_head.parameters(), *sem_head.parameters()):
            p.requires_grad_(False)

    param_groups = [
        {"params": enc.parameters(), "lr": args.lr},
        {"params": gate.parameters(), "lr": args.gate_lr},
    ]
    if not args.freeze_heads:
        param_groups.append({"params": [*obj_head.parameters(), *sem_head.parameters()],
                             "lr": args.head_lr})
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-2)

    def evaluate(tag, use_gate=True):
        bank = encode_bank(enc, obj_head, sem_head, gate, g_tr, hs_tr, device, use_gate=use_gate)
        H_va = encode_bank(enc, obj_head, sem_head, gate, g_va, hs_va, device, use_gate=use_gate)
        m = retrieval_eval(bank.to(device), H_va, train_cases, valid_cases,
                           cause_embs, device, top_k_cases=args.top_k_cases)
        print(f"  [{tag}] sem R@1={m['R@1']:.4f} R@10={m['R@10']:.4f} MRR={m['MRR']:.4f}"
              + ("" if use_gate else "  (baseline sigmoid w, init heads)"))
        return m

    print("[baseline] init heads + frozen sigmoid gate:")
    evaluate("sigmoid", use_gate=False)
    print("[L2 ep0] init heads + untrained Region Gate:")
    evaluate("ep0", use_gate=True)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        enc.train(); t0 = time.time(); perm = torch.randperm(N); losses = []
        ghn = ohn = 0.0
        for bs in range(0, N - args.batch_size + 1, args.batch_size):
            idx = perm[bs:bs + args.batch_size]; b = idx.size(0)
            w, z = heads_forward(obj_head, sem_head, gate, hs_tr[idx], device)
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
            opt.zero_grad(); loss.backward()
            ghn = float(torch.cat([p.grad.flatten() for p in gate.parameters()]).norm())
            if not args.freeze_heads and obj_head.weight.grad is not None:
                ohn = float(obj_head.weight.grad.norm())
            torch.nn.utils.clip_grad_norm_(
                [p for grp in param_groups for p in grp["params"]], 1.0)
            opt.step(); losses.append(loss.item())
        m = evaluate(f"ep{ep}")
        print(f"    loss={np.mean(losses):.4f} τ={gate.temp.item():.4f} ∅={gate.sink.item():+.4f} "
              f"grad‖gate‖={ghn:.4f} grad‖obj_head‖={ohn:.4f} dur={time.time()-t0:.0f}s")
        if m["MRR"] > best:
            best = m["MRR"]
            torch.save({"encoder_state": {k: v.cpu() for k, v in enc.state_dict().items()},
                        "encoder_config": vars(enc_cfg),
                        "gate_state": {k: v.cpu() for k, v in gate.state_dict().items()},
                        "obj_head_state": {k: v.cpu() for k, v in obj_head.state_dict().items()},
                        "sem_head_state": {k: v.cpu() for k, v in sem_head.state_dict().items()}},
                       out / "best.pt")
    print(f"[done] best sem MRR={best:.4f} -> {out/'best.pt'}")


if __name__ == "__main__":
    main()
