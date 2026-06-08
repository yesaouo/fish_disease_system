"""Soft-pipeline retrain — step #2: train the Aggregator on soft GROD inputs.

Same supervision as the hard encoder (reuses the GT-based teacher table from
build_teacher_table.py + the case→cause InfoNCE); only the encoder INPUT changes
from clean GT lesions to the soft inference distribution produced by
extract_soft_inputs.py (all 300 queries + sigmoid objectness weights). The
encoder learns to reproduce the GT case-similarity ranking from the soft
detector outputs it actually sees at inference.

DeepSetsEncoder.forward now takes an optional ``lesion_weights`` (added,
backward-compatible) that drives soft mean/max/sum pooling.

Output: outputs/encoder_grod_soft/best_encoder.pt

Run from repo root (SDM env):
    $PY -m diagnosis_model.grod.train_case_encoder_soft
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.models.case_encoder import (
    EncoderConfig, build_encoder, listwise_kl_loss, case_cause_infonce_loss,
)
from diagnosis_model.cause_inference.train_case_encoder import retrieval_metrics


def load_soft(path):
    d = torch.load(path, weights_only=False)
    return d["g"], d["z_all"], d["w"], d["cause_emb_indices"]


@torch.no_grad()
def encode_all_soft(encoder, g_all, z_all, w_all, device, batch_size=256):
    encoder.eval()
    N = g_all.size(0)
    out = []
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        b = e - s
        lens = torch.full((b,), z_all.size(1), device=device, dtype=torch.long)
        h = encoder(g_all[s:e].float().to(device),
                    z_all[s:e].float().to(device),
                    lens,
                    lesion_weights=w_all[s:e].float().to(device))
        out.append(h.float().cpu())
    encoder.train()
    return torch.cat(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--soft_dir", default=f"{ART}/db/soft_inputs")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--teacher_path", default=f"{ART}/db/case_db_jointDistRawP/teacher_train_train.pt")
    ap.add_argument("--output_dir", default=f"{ART}/models/encoder_grod_soft")
    # mirror encoder_grod canonical config
    ap.add_argument("--head_hidden", type=int, default=768)
    ap.add_argument("--temp_target", type=float, default=0.1)
    ap.add_argument("--temp_pred", type=float, default=0.1)
    ap.add_argument("--use_infonce", action="store_true", default=True)
    ap.add_argument("--infonce_weight", type=float, default=0.5)
    ap.add_argument("--infonce_temp", type=float, default=0.07)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--early_stop_patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap cases per split")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out_dir / "config.json", "w"), indent=2)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    suffix = "_smoke" if args.limit else ""
    g_tr, z_tr, w_tr, cidx_tr = load_soft(Path(args.soft_dir) / f"train{suffix}.pt")
    g_va, z_va, w_va, _ = load_soft(Path(args.soft_dir) / f"valid{suffix}.pt")
    if args.limit:
        g_tr, z_tr, w_tr, cidx_tr = g_tr[:args.limit], z_tr[:args.limit], w_tr[:args.limit], cidx_tr[:args.limit]
        g_va, z_va, w_va = g_va[:args.limit], z_va[:args.limit], w_va[:args.limit]
    Nt = g_tr.size(0)
    print(f"soft train={Nt} valid={g_va.size(0)} Q={z_tr.size(1)} D={z_tr.size(2)}")

    # GT case metadata for retrieval_metrics (causes) — same order as soft inputs
    train_cases = torch.load(Path(args.case_db_dir) / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(Path(args.case_db_dir) / "valid_cases.pt", weights_only=False)
    if args.limit:
        train_cases, valid_cases = train_cases[:args.limit], valid_cases[:args.limit]

    # Teacher (GT-based, reused). Sliced to Nt.
    teacher_full = torch.load(args.teacher_path, weights_only=False, map_location="cpu")["scores"]
    teacher_full = teacher_full[:Nt, :Nt]
    print(f"teacher: {tuple(teacher_full.shape)}")

    cause_text_embs = None
    if args.use_infonce:
        cte = torch.load(Path(args.case_db_dir) / "cause_text_embs.pt", weights_only=False, map_location="cpu")
        cause_text_embs = F.normalize(cte["embeddings"].float(), dim=-1).to(device)
        print(f"cause_text_embs: {tuple(cause_text_embs.shape)} (InfoNCE w={args.infonce_weight})")

    cfg = EncoderConfig(encoder_type="deepsets", d_model=z_tr.size(2),
                        head_hidden=args.head_hidden)
    encoder = build_encoder(cfg).to(device)
    print(f"encoder=deepsets params={sum(p.numel() for p in encoder.parameters())/1e6:.2f}M")

    optim = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = Nt // args.batch_size
    total_steps = max(1, args.epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    log_f = open(out_dir / "train_log.jsonl", "w")
    best_metric, patience, gstep = -1.0, 0, 0
    for epoch in range(args.epochs):
        encoder.train()
        perm = torch.randperm(Nt)
        ep_loss = ep_d = ep_i = 0.0; nb = 0; t0 = time.time()
        for bs in range(0, Nt - args.batch_size + 1, args.batch_size):
            idx = perm[bs:bs + args.batch_size]
            b = idx.size(0)
            lens = torch.full((b,), z_tr.size(1), device=device, dtype=torch.long)
            h = encoder(g_tr[idx].float().to(device),
                        z_tr[idx].float().to(device),
                        lens,
                        lesion_weights=w_tr[idx].float().to(device))    # [b, D]
            teacher_block = teacher_full[idx][:, idx].to(device).float()
            loss_d = listwise_kl_loss(h, teacher_block,
                                      temp_target=args.temp_target, temp_pred=args.temp_pred)
            if args.use_infonce:
                V = cause_text_embs.size(0)
                pos = torch.zeros(b, V, dtype=torch.bool, device=device)
                for i, ci in enumerate(idx.tolist()):
                    cidxs = cidx_tr[ci]
                    if cidxs:
                        pos[i, torch.tensor(cidxs, dtype=torch.long, device=device)] = True
                loss_i = case_cause_infonce_loss(h, cause_text_embs, pos, temp=args.infonce_temp)
                loss = loss_d + args.infonce_weight * loss_i
            else:
                loss_i = torch.tensor(0.0); loss = loss_d
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optim.step(); sched.step()
            ep_loss += loss.item(); ep_d += loss_d.item(); ep_i += float(loss_i); nb += 1; gstep += 1

        H_va = encode_all_soft(encoder, g_va, z_va, w_va, device)
        H_tr = encode_all_soft(encoder, g_tr, z_tr, w_tr, device)
        m = retrieval_metrics(H_va, H_tr, valid_cases, train_cases)
        rec = {"epoch": epoch, "loss": ep_loss / max(1, nb), "loss_distill": ep_d / max(1, nb),
               "loss_infonce": ep_i / max(1, nb), "dt": round(time.time() - t0, 1), **m}
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if epoch % 2 == 0 or epoch == args.epochs - 1:
            print(f"[ep {epoch:>2}] loss={rec['loss']:.4f} sem_R@10={m['sem_R@10']:.4f} "
                  f"R@1={m['sem_R@1']:.4f} MRR={m['sem_MRR']:.4f}")
        if m["sem_R@10"] > best_metric:
            best_metric = m["sem_R@10"]; patience = 0
            torch.save({"encoder_state": {k: v.cpu() for k, v in encoder.state_dict().items()},
                        "encoder_config": vars(cfg), "metrics": m, "epoch": epoch},
                       out_dir / "best_encoder.pt")
        else:
            patience += 1
            if args.early_stop_patience > 0 and patience >= args.early_stop_patience:
                print(f"early stop @ epoch {epoch} (best sem_R@10={best_metric:.4f})"); break

    print(f"[done] best sem_R@10={best_metric:.4f} -> {out_dir/'best_encoder.pt'}")


if __name__ == "__main__":
    main()
