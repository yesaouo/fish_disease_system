"""Train the master-slave Mamba case encoder via Phase 1 distillation.

Listwise-KL distillation against the precomputed Phase 1 hungarian teacher
score table (teacher_train_train.pt). For each batch of B train cases, the
student must reproduce per-anchor row-softmax of teacher case-similarities.

The encoder consumes (global_emb, lesion_embs sorted by area DESC) and emits
one L2-normed h_final ∈ R^768 per case so that retrieval becomes a single
case-to-case cosine.

CLI quickstart from repo root:
    CC=/usr/bin/gcc-12 \
    /home/lab603/anaconda3/envs/mamba3/bin/python \
        -m diagnosis_model.cause_inference.train_mamba_encoder \
        --output_dir diagnosis_model/cause_inference/outputs/mamba_encoder/mamba_v1

Notes:
  - mamba_ssm requires CC env var pointing to gcc on this host (gcc-12).
  - Encoder type is selectable via --encoder_type ∈ {mamba, mean, deepsets}.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from diagnosis_model.cause_inference.models.mamba_encoder import (
    EncoderConfig,
    build_encoder,
    listwise_kl_loss,
    pairwise_mse_loss,
)
from diagnosis_model.cause_inference.phase1_baseline import load_case_db


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _lesion_areas(boxes_xywh: torch.Tensor) -> torch.Tensor:
    """boxes_xywh: [N, 4] -> [N] areas (w * h, in pixel^2)."""
    if boxes_xywh.numel() == 0:
        return torch.zeros(0, dtype=torch.float32)
    return (boxes_xywh[:, 2].float() * boxes_xywh[:, 3].float())


class CaseEncoderDataset(Dataset):
    """One sample = one train case. Lesions are pre-sorted by area DESC.

    Per-case dict keys returned:
        case_id     : int (= index into teacher table)
        global_emb  : [D] L2-normed
        lesion_embs : [N, D] L2-normed, sorted by area DESC (largest first)
    """

    def __init__(self, cases: list):
        self.cases = cases
        # Pre-sort lesions by area DESC and L2-normalize once.
        self.records = []
        for ci, c in enumerate(cases):
            g = c["global_emb"]
            g = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            L = c["lesion_embs"]
            if L.size(0) > 0:
                L = L / L.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                areas = _lesion_areas(c["lesion_boxes_xywh"])
                order = torch.argsort(areas, descending=True)
                L = L[order]
            self.records.append(dict(case_id=c["case_id"], global_emb=g, lesion_embs=L))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def make_collate(D: int):
    def collate(batch: List[dict]) -> dict:
        B = len(batch)
        max_N = max(b["lesion_embs"].size(0) for b in batch)
        max_N = max(max_N, 0)
        global_embs = torch.stack([b["global_emb"] for b in batch])      # [B, D]
        lesion_pad = torch.zeros(B, max(max_N, 1), D)                    # [B, ≥1, D]
        lesion_lens = torch.zeros(B, dtype=torch.long)
        for i, b in enumerate(batch):
            n = b["lesion_embs"].size(0)
            if n > 0:
                lesion_pad[i, :n] = b["lesion_embs"]
            lesion_lens[i] = n
        return {
            "case_ids": torch.tensor([b["case_id"] for b in batch], dtype=torch.long),
            "global_emb": global_embs,
            "lesion_pad": lesion_pad,
            "lesion_lens": lesion_lens,
        }
    return collate


# ---------------------------------------------------------------------------
# Quick retrieval eval against valid set
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_all(encoder, cases, device, batch_size=256) -> torch.Tensor:
    encoder.eval()
    ds = CaseEncoderDataset(cases)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate(D=ds.records[0]["global_emb"].size(0)),
                        num_workers=0)
    out = []
    for batch in loader:
        z = encoder(
            batch["global_emb"].to(device),
            batch["lesion_pad"].to(device),
            batch["lesion_lens"].to(device),
        )
        out.append(z.float().cpu())
    encoder.train()
    return torch.cat(out, dim=0)


@torch.no_grad()
def retrieval_metrics(
    H_valid: torch.Tensor,                  # [Nv, D]
    H_train: torch.Tensor,                  # [Nt, D]
    valid_cases: list,
    train_cases: list,
    Ks: List[int] = (1, 5, 10, 20, 100),
) -> dict:
    """Semantic exact-match recall @ K. A retrieved case is "correct" if any of
    its GT causes is also a GT cause of the valid query (string match).
    """
    sims = H_valid @ H_train.T                                    # [Nv, Nt]
    ranks = sims.argsort(dim=1, descending=True)                  # [Nv, Nt]
    Nv = H_valid.size(0)
    train_causes = [set(c["causes"]) for c in train_cases]

    recalls = {k: 0 for k in Ks}
    mrr_total = 0.0
    Ks_max = max(Ks)
    for vi in range(Nv):
        gt = set(valid_cases[vi]["causes"])
        if not gt:
            continue
        for r, ti in enumerate(ranks[vi, :Ks_max].tolist()):
            if train_causes[ti] & gt:
                mrr_total += 1.0 / (r + 1)
                for k in Ks:
                    if r < k:
                        recalls[k] += 1
                break

    out = {f"sem_R@{k}": recalls[k] / Nv for k in Ks}
    out["sem_MRR"] = mrr_total / Nv
    return out


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db")
    ap.add_argument("--teacher_path", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db/"
                            "teacher_train_train.pt")
    ap.add_argument("--output_dir", type=str, required=True)
    # Encoder config
    ap.add_argument("--encoder_type", type=str, default="mamba",
                    choices=["mamba", "mean", "deepsets"])
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--d_state", type=int, default=128)
    ap.add_argument("--headdim", type=int, default=64)
    ap.add_argument("--chunk_size", type=int, default=16)
    ap.add_argument("--head_hidden", type=int, default=768)
    ap.add_argument("--no_projection_head", action="store_true")
    ap.add_argument("--no_role_emb", action="store_true")
    ap.add_argument("--no_input_proj", action="store_true")
    # Loss config
    ap.add_argument("--loss_type", type=str, default="listwise_kl",
                    choices=["listwise_kl", "pairwise_mse"])
    ap.add_argument("--temp_target", type=float, default=0.1)
    ap.add_argument("--temp_pred", type=float, default=0.1)
    # Training
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--early_stop_patience", type=int, default=5)
    ap.add_argument("--eval_every_epochs", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_train_cases", type=int, default=-1,
                    help="Cap train set (for sanity / overfit tests).")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out_dir / "config.json", "w"), indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Data
    train_cases, valid_cases, _, meta = load_case_db(Path(args.case_db_dir))
    if args.max_train_cases > 0:
        train_cases = train_cases[:args.max_train_cases]
    D = meta["global_dim"]
    print(f"train={len(train_cases)} valid={len(valid_cases)} D={D}")

    # Teacher
    teacher_pkg = torch.load(args.teacher_path, weights_only=False, map_location="cpu")
    teacher_full = teacher_pkg["scores"]                              # fp16 [Nt, Nt]
    teacher_full = teacher_full[:len(train_cases), :len(train_cases)]
    print(f"teacher table: {tuple(teacher_full.shape)}  config={teacher_pkg['config']}")

    train_ds = CaseEncoderDataset(train_cases)
    collate = make_collate(D)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # Encoder
    cfg = EncoderConfig(
        encoder_type=args.encoder_type,
        d_model=D,
        n_layers=args.n_layers,
        d_state=args.d_state,
        headdim=args.headdim,
        chunk_size=args.chunk_size,
        head_hidden=args.head_hidden,
        use_projection_head=not args.no_projection_head,
        use_role_embeddings=not args.no_role_emb,
        use_input_projection=not args.no_input_proj,
        is_mimo=False,                        # see encoder default note
    )
    encoder = build_encoder(cfg).to(device)
    n_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"encoder={cfg.encoder_type}  params={n_params/1e6:.2f}M")

    optim = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(loader))

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "w")

    best_metric = -1.0
    patience = 0
    global_step = 0
    for epoch in range(args.epochs):
        encoder.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for batch in loader:
            case_ids = batch["case_ids"]
            g = batch["global_emb"].to(device)
            L = batch["lesion_pad"].to(device)
            lens = batch["lesion_lens"].to(device)

            z = encoder(g, L, lens)                                   # [B, D]
            teacher_block = teacher_full[case_ids][:, case_ids].to(device).float()

            if args.loss_type == "listwise_kl":
                loss = listwise_kl_loss(z, teacher_block,
                                        temp_target=args.temp_target,
                                        temp_pred=args.temp_pred)
            else:
                loss = pairwise_mse_loss(z, teacher_block)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optim.step()
            sched.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        train_loss = epoch_loss / max(1, n_batches)
        epoch_dt = time.time() - t0

        log_row = {"epoch": epoch + 1, "train_loss": train_loss,
                   "lr": optim.param_groups[0]["lr"], "time_s": epoch_dt}

        if (epoch + 1) % args.eval_every_epochs == 0:
            t1 = time.time()
            H_train = encode_all(encoder, train_cases, device)
            H_valid = encode_all(encoder, valid_cases, device)
            metrics = retrieval_metrics(H_valid, H_train, valid_cases, train_cases)
            log_row.update(metrics)
            log_row["eval_time_s"] = time.time() - t1

            metric = metrics["sem_R@10"]
            if metric > best_metric:
                best_metric = metric
                patience = 0
                torch.save({
                    "encoder_state": encoder.state_dict(),
                    "encoder_config": vars(cfg),
                    "metrics": metrics,
                    "epoch": epoch + 1,
                }, out_dir / "best_encoder.pt")
            else:
                patience += 1

        print(json.dumps(log_row))
        log_f.write(json.dumps(log_row) + "\n")
        log_f.flush()

        if patience >= args.early_stop_patience:
            print(f"Early stop at epoch {epoch + 1} (patience hit).")
            break

    torch.save({"encoder_state": encoder.state_dict(),
                "encoder_config": vars(cfg)},
               out_dir / "last_encoder.pt")
    log_f.close()
    print(f"Done. best sem_R@10 = {best_metric:.4f}")


if __name__ == "__main__":
    main()
