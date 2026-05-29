"""Train the Phase 3 case encoder via Phase 1 distillation.

Listwise-KL distillation against a Phase 1 teacher case-similarity table.
For each batch of B train cases, the student must reproduce per-anchor
row-softmax of teacher case-similarities.

Two teacher modes:

- ``--teacher_mode precomputed`` (default, fish): loads the N×N
  ``teacher_train_train.pt`` produced by ``build_teacher_table.py`` and
  slices per batch. Storage = N² × 2 bytes (fp16) — 311 MiB at fish 12,780
  cases; infeasible at DDXPlus 200k (~80 GB).
- ``--teacher_mode on_the_fly`` (DDXPlus): caches normalized global +
  lesion stacks on GPU in ``--bank_dtype`` and computes the BxB intra-batch
  teacher block per step via vectorized scatter_reduce. Only ``max_mean``
  / ``max_mean_normalized`` lesion-match are supported on this path
  (hungarian's scipy LP loop has no batched equivalent and is anyway
  infeasible at the case bank sizes that motivate this mode).

The encoder consumes (global_emb, lesion_embs sorted by area DESC) and emits
one L2-normed h_final ∈ R^768 per case so that retrieval becomes a single
case-to-case cosine.

Production choice is `--encoder_type deepsets` (see README Phase 3). The
'mean' baseline is also pure PyTorch and runs in the SDM env. The 'mamba'
choice lives under diagnosis_model.cause_inference.mamba_ablation and
requires the mamba3 conda env + CC=/usr/bin/gcc-12; build_encoder() lazy-
imports it only when requested.

CLI quickstart from repo root (SDM env, default deepsets, fish):
    /home/lab603/anaconda3/envs/SDM/bin/python \
        -m diagnosis_model.cause_inference.train_case_encoder \
        --case_db_dir diagnosis_model/cause_inference/outputs/case_db \
        --teacher_path diagnosis_model/cause_inference/outputs/case_db/teacher_train_train.pt \
        --output_dir diagnosis_model/cause_inference/outputs/encoder_final \
        --encoder_type deepsets \
        --batch_size 256 --epochs 50 \
        --use_infonce --infonce_weight 0.5 --infonce_temp 0.07

DDXPlus 200k subsample (on-the-fly teacher, bf16 bank):
    /home/lab603/anaconda3/envs/SDM/bin/python \
        -m diagnosis_model.cause_inference.train_case_encoder \
        --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
        --output_dir diagnosis_model/cause_inference/outputs/ddxplus_encoder \
        --encoder_type deepsets \
        --teacher_mode on_the_fly --bank_dtype bf16 \
        --max_train_cases 200000 --max_valid_cases 5000 --sample_seed 42 \
        --batch_size 256 --epochs 30 \
        --use_infonce --infonce_weight 0.5 --infonce_temp 0.07
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from diagnosis_model.cause_inference.models.case_encoder import (
    EncoderConfig,
    build_encoder,
    listwise_kl_loss,
    pairwise_mse_loss,
    case_cause_infonce_loss,
)
from diagnosis_model.cause_inference.phase1_baseline import (
    load_case_db,
    load_cases,
    offsets_to_case_ids,
)


_BANK_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


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
        case_id            : int (= positional index in ``cases`` list,
                                  used to slice the teacher block)
        global_emb         : [D] L2-normed
        lesion_embs        : [N, D] L2-normed, sorted by area DESC (largest first)
        cause_emb_indices  : list[int] indices into cause_text_embs (for dual-target loss)

    Note: positional rather than ``c["case_id"]`` so subsample (DDXPlus path)
    indexing into the same ordered teacher table / on-the-fly bank works.
    For fish (no subsample) ``c["case_id"]`` equals the position anyway.

    ``free_source``: at DDXPlus scale, the source ``train_cases`` already holds
    ~8-18 GB of CPU tensors (200k cases × ~40 KB fp32). Copying normalized
    versions into ``self.records`` doubles this. With ``free_source=True``
    each source dict is stripped of its embedding fields immediately after
    the normalized copy is appended, so peak CPU stays at one-copy. Caller
    must have already snapshotted whatever it needs from the source first
    (e.g. constructed the on-the-fly teacher bank). ``causes`` and
    ``cause_emb_indices`` are preserved — ``retrieval_metrics`` still
    needs them.
    """

    _DROP_KEYS = (
        "global_emb", "lesion_embs",
        "text_colloquial_emb", "text_medical_emb",
        "lesion_boxes_xywh",
    )

    def __init__(self, cases: list, free_source: bool = False):
        self.cases = cases
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
            pathology_idx = c.get("pathology_emb_idx")
            self.records.append(dict(
                case_id=int(ci),
                global_emb=g,
                lesion_embs=L,
                cause_emb_indices=list(c.get("cause_emb_indices", [])),
                # DDXPlus v2 schema only: single strict GT for stricter
                # InfoNCE positives. None on fish (no field).
                pathology_emb_idx=(int(pathology_idx) if pathology_idx is not None else None),
            ))
            if free_source:
                for k in self._DROP_KEYS:
                    c.pop(k, None)

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
            "cause_indices": [b["cause_emb_indices"] for b in batch],
            "pathology_indices": [b.get("pathology_emb_idx") for b in batch],
        }
    return collate


# ---------------------------------------------------------------------------
# On-the-fly teacher (DDXPlus scale: precomputed N×N table doesn't fit)
# ---------------------------------------------------------------------------

class OnTheFlyTeacher:
    """Computes Phase 1 case-similarity teacher blocks per training batch.

    Caches normalized global + lesion stacks on GPU in ``bank_dtype`` (bf16 by
    default for DDXPlus, fp32 for fish). Per training step extracts the BxB
    intra-batch block via fully vectorized scatter_reduce, matching
    ``phase1_baseline.compute_case_similarities``'s ``max_mean`` /
    ``max_mean_normalized`` ranking up to bf16 cosine precision.

    Only the symmetric max_mean family is supported; hungarian's per-pair
    scipy LP loop has no batched equivalent (and the dataset sizes that
    motivate this teacher mode are well past hungarian-feasible anyway).
    """

    def __init__(
        self,
        train_cases: List[dict],
        device: torch.device,
        bank_dtype: torch.dtype = torch.bfloat16,
        alpha: float = 0.25,
        beta: float = 0.75,
        lesion_match: str = "max_mean",
        case_chunk_size: int = 16384,
    ):
        if lesion_match not in ("max_mean", "max_mean_normalized"):
            raise ValueError(
                f"on_the_fly teacher supports only max_mean / "
                f"max_mean_normalized; got {lesion_match!r}."
            )
        self.device = device
        self.bank_dtype = bank_dtype
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.normalize_by_max = (lesion_match == "max_mean_normalized")
        N = len(train_cases)
        D = int(train_cases[0]["global_emb"].size(-1))

        # Build banks chunk-by-chunk so the CPU `torch.cat` temp tensor never
        # exceeds ``case_chunk_size`` cases at once. At 200k DDXPlus cases
        # ×~10 lesions × 768 fp32, a single naive cat is ~6 GB CPU; chunking
        # keeps the transient under ~500 MB per step.
        D_ = D
        # Pre-pass: total lesion count to size self.L exactly.
        offsets: List[int] = [0]
        for c in train_cases:
            offsets.append(offsets[-1] + int(c["lesion_embs"].size(0)))
        total_lesions = offsets[-1]

        self.G = torch.empty(N, D_, device=device, dtype=bank_dtype)
        self.L = torch.empty(total_lesions, D_, device=device, dtype=bank_dtype)

        for start in range(0, N, case_chunk_size):
            end = min(start + case_chunk_size, N)
            G_chunk = torch.stack(
                [train_cases[i]["global_emb"] for i in range(start, end)]
            )
            self.G[start:end] = G_chunk.to(device=device, dtype=bank_dtype)
            del G_chunk
            if total_lesions > 0:
                lesion_chunk = [
                    train_cases[i]["lesion_embs"]
                    for i in range(start, end)
                    if train_cases[i]["lesion_embs"].size(0) > 0
                ]
                if lesion_chunk:
                    L_cat = torch.cat(lesion_chunk, dim=0)
                    self.L[offsets[start]:offsets[end]] = L_cat.to(
                        device=device, dtype=bank_dtype,
                    )
                    del L_cat, lesion_chunk

        self.G = F.normalize(self.G, dim=-1)
        if total_lesions > 0:
            self.L = F.normalize(self.L, dim=-1)
        self.offsets = offsets
        self.case_ids_full = offsets_to_case_ids(offsets, device)  # [ΣN]
        self.lesion_counts_full = torch.zeros(N, device=device, dtype=torch.float32)
        if self.case_ids_full.numel() > 0:
            self.lesion_counts_full.scatter_add_(
                0, self.case_ids_full,
                torch.ones_like(self.case_ids_full, dtype=torch.float32),
            )

    @torch.no_grad()
    def batch_block(self, case_ids: torch.Tensor) -> torch.Tensor:
        """Return BxB teacher score block for the given positional case_ids.

        Diagonal set to NaN (matches build_teacher_table.py's convention so the
        existing listwise-KL row-softmax masks self consistently).
        """
        B = case_ids.size(0)
        device = self.device
        # Globals.
        G_batch = self.G[case_ids]                                  # [B, D]
        g_sim = (G_batch @ G_batch.T).float()                       # [B, B]
        out = self.alpha * g_sim

        # Lesions: pull each batch case's lesion rows into a contiguous stack,
        # remapping their original case_id (positional in the full bank) to
        # batch-local id in [0, B).
        if self.L.size(0) == 0:
            out.fill_diagonal_(float("nan"))
            return out

        sub_pieces: List[torch.Tensor] = []
        sub_case_ids: List[torch.Tensor] = []
        for bi, ci in enumerate(case_ids.tolist()):
            s = self.offsets[int(ci)]
            e = self.offsets[int(ci) + 1]
            if e > s:
                sub_pieces.append(self.L[s:e])
                sub_case_ids.append(torch.full(
                    (e - s,), bi, device=device, dtype=torch.long,
                ))
        if not sub_pieces:
            out.fill_diagonal_(float("nan"))
            return out

        L_batch = torch.cat(sub_pieces, dim=0)                      # [ΣN_batch, D]
        case_ids_local = torch.cat(sub_case_ids, dim=0)             # [ΣN_batch]
        L_total = L_batch.size(0)

        # All-pairs cosine within the batch's lesion subset.
        les_sim = L_batch @ L_batch.T                               # [ΣN_b, ΣN_b]

        # Forward: for each row p (lesion of anchor i) and candidate case j,
        #   t_max[p, j] = max_{q in L_j} les_sim[p, q]
        neg_inf = float("-inf")
        t_max = torch.full(
            (L_total, B), neg_inf, device=device, dtype=les_sim.dtype,
        )
        case_ids_local_exp = case_ids_local.unsqueeze(0).expand(L_total, -1)
        t_max.scatter_reduce_(
            1, case_ids_local_exp, les_sim,
            reduce="amax", include_self=False,
        )
        t_max = torch.where(
            t_max == neg_inf, torch.zeros_like(t_max), t_max,
        )

        # Step 2: mean over rows p grouped by anchor i.
        forward_sum = torch.zeros(B, B, device=device, dtype=torch.float32)
        case_ids_local_T = case_ids_local.unsqueeze(-1).expand(-1, B)
        forward_sum.scatter_add_(0, case_ids_local_T, t_max.float())
        counts = torch.zeros(B, device=device, dtype=torch.float32)
        if case_ids_local.numel() > 0:
            counts.scatter_add_(
                0, case_ids_local,
                torch.ones_like(case_ids_local, dtype=torch.float32),
            )

        if self.normalize_by_max:
            # Symmetric max-mean normalized: 0.5 * (forward_sum + backward_sum)
            # / max(N_i, N_j); used only when lesion_match == max_mean_normalized.
            denom = torch.maximum(
                counts.unsqueeze(-1), counts.unsqueeze(0),
            ).clamp_min(1.0)
            backward_sum = forward_sum.T
            lesion_score = 0.5 * (forward_sum + backward_sum) / denom
        else:
            forward_mean = forward_sum / counts.clamp_min(1.0).unsqueeze(-1)
            # backward[i,j] = forward[j,i] by symmetry of the inner construction.
            backward_mean = forward_mean.T
            lesion_score = 0.5 * (forward_mean + backward_mean)

        valid = (counts > 0).unsqueeze(-1) & (counts > 0).unsqueeze(0)
        lesion_score = torch.where(
            valid, lesion_score, torch.zeros_like(lesion_score),
        )

        out = out + self.beta * lesion_score
        out.fill_diagonal_(float("nan"))
        return out


# ---------------------------------------------------------------------------
# Quick retrieval eval against valid set
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_all(encoder, cases_or_dataset, device, batch_size=256) -> torch.Tensor:
    """Encode every case via the encoder; returns ``[N, D]`` fp32 CPU.

    Accepts either a list of case dicts (builds a fresh
    :class:`CaseEncoderDataset` internally — used by the small valid path
    and by fish workflows) or an already-constructed dataset (reused
    across eval epochs so we don't re-allocate the 8-18 GB ``records``
    list every time on DDXPlus 200k).
    """
    encoder.eval()
    ds = (cases_or_dataset if isinstance(cases_or_dataset, CaseEncoderDataset)
          else CaseEncoderDataset(cases_or_dataset))
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
    query_batch: int = 512,
) -> dict:
    """Semantic exact-match recall @ K. A retrieved case is "correct" if any of
    its GT causes is also a GT cause of the valid query (string match).

    Batched over queries: only the top-``max(Ks)`` rows are kept per chunk,
    so peak memory is ``query_batch × Nt`` (not ``Nv × Nt``). Required at
    DDXPlus scale where ``Nv × Nt`` exceeds VRAM (130k × 200k fp32 = 100 GB).
    """
    Nv = H_valid.size(0)
    Ks_max = max(Ks)
    train_causes = [set(c["causes"]) for c in train_cases]

    recalls = {k: 0 for k in Ks}
    mrr_total = 0.0
    H_train_T = H_train.T
    for vs in range(0, Nv, query_batch):
        ve = min(vs + query_batch, Nv)
        sims_b = H_valid[vs:ve] @ H_train_T                       # [bv, Nt]
        top_idx_b = sims_b.topk(Ks_max, dim=1).indices.cpu()      # [bv, K]
        for bi in range(top_idx_b.size(0)):
            vi = vs + bi
            gt = set(valid_cases[vi]["causes"])
            if not gt:
                continue
            for r, ti in enumerate(top_idx_b[bi].tolist()):
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
    ap.add_argument("--teacher_mode", type=str,
                    choices=["precomputed", "on_the_fly"],
                    default="precomputed",
                    help="precomputed (default, fish): load N×N table from "
                         "--teacher_path. on_the_fly (DDXPlus): compute the "
                         "BxB intra-batch block per step from cached "
                         "globals/lesions stacks on GPU.")
    ap.add_argument("--teacher_alpha", type=float, default=0.25,
                    help="α for on-the-fly teacher case-similarity "
                         "(matches build_teacher_table.py --alpha_global).")
    ap.add_argument("--teacher_beta", type=float, default=0.75,
                    help="β for on-the-fly teacher case-similarity.")
    ap.add_argument("--teacher_lesion_match", type=str,
                    choices=["max_mean", "max_mean_normalized"],
                    default="max_mean",
                    help="Lesion-match used by the on-the-fly teacher. "
                         "hungarian unsupported — no batched equivalent and "
                         "infeasible at the bank sizes this mode targets.")
    ap.add_argument("--bank_dtype", type=str,
                    choices=["fp32", "fp16", "bf16"], default="fp32",
                    help="Storage dtype for the on-the-fly teacher's "
                         "pre-stacked globals/lesions. fp32 = fish default; "
                         "bf16 = DDXPlus (halves VRAM, matches case_db "
                         "storage). Ignored when teacher_mode=precomputed.")
    ap.add_argument("--max_train_cases", type=int, default=-1,
                    help="Cap retained train cases via uniform per-shard "
                         "subsampling. -1 (default) loads all. Required for "
                         "DDXPlus (200000 mirrors Phase 1/2 convention).")
    ap.add_argument("--max_valid_cases", type=int, default=-1,
                    help="Cap retained valid cases via per-shard subsampling "
                         "for the early-stop sem R@K eval. -1 loads all.")
    ap.add_argument("--sample_seed", type=int, default=42,
                    help="Seed for the per-shard subsample RNG. Must match "
                         "the seed used elsewhere (Phase 2 train_pool, etc.) "
                         "if you want aligned indices.")
    ap.add_argument("--eval_query_batch", type=int, default=512,
                    help="Query batch size for retrieval_metrics; controls "
                         "peak query×bank sim memory.")
    ap.add_argument("--output_dir", type=str, required=True)
    # Encoder config
    ap.add_argument("--encoder_type", type=str, default="deepsets",
                    choices=["deepsets", "mean", "mamba"],
                    help="'deepsets' (default, production) and 'mean' are pure "
                         "PyTorch. 'mamba' is an architecture ablation; needs "
                         "mamba3 conda env and CC=/usr/bin/gcc-12.")
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
    # Dual-target (case -> cause text) InfoNCE
    ap.add_argument("--use_infonce", action="store_true",
                    help="Add SupCon-style InfoNCE between h_final and GT cause text embs.")
    ap.add_argument("--infonce_weight", type=float, default=0.5)
    ap.add_argument("--infonce_temp", type=float, default=0.07)
    ap.add_argument("--infonce_positives", type=str,
                    choices=["cause_emb_indices", "pathology"],
                    default="cause_emb_indices",
                    help="Which cause-table rows are positives for the "
                         "dual-target InfoNCE. Default 'cause_emb_indices' "
                         "treats every entry of the per-case GT list as a "
                         "positive (fish: 1 cause; DDXPlus v2: pathology + "
                         "5 DDX = 6 positives — relatively easy). "
                         "'pathology' restricts positives to a single "
                         "strict pathology cause-table index "
                         "(``pathology_emb_idx``, DDXPlus v2 only) for a "
                         "harder distillation target. Errors if any case "
                         "is missing the field.")
    # Hard-case mining (option b)
    ap.add_argument("--miss_weight", type=float, default=1.0,
                    help="Upweight hard cases (teacher miss >=1 GT) in sampler. "
                         "weight(case) = 1 + miss_weight * miss_count. Default=1.0 disables.")
    ap.add_argument("--train_pool_path", type=str,
                    default="diagnosis_model/cause_inference/outputs/case_db/"
                            "train_candidate_pool.pt")
    # Training
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--early_stop_patience", type=int, default=5,
                    help="Stop if sem_R@10 doesn't improve for N consecutive evals. "
                         "Set <=0 to disable and let training run the full --epochs.")
    ap.add_argument("--eval_every_epochs", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out_dir / "config.json", "w"), indent=2)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Data — load_cases routes through the shared sharded loader so DDXPlus
    # and fish case_dbs both work; per-shard subsample fires when
    # max_{train,valid}_cases is set.
    case_db_dir = Path(args.case_db_dir)
    max_train = args.max_train_cases if args.max_train_cases > 0 else None
    max_valid = args.max_valid_cases if args.max_valid_cases > 0 else None
    train_cases = load_cases(
        case_db_dir, "train",
        max_cases=max_train, sample_seed=args.sample_seed,
    )
    valid_cases = load_cases(
        case_db_dir, "valid",
        max_cases=max_valid, sample_seed=args.sample_seed,
    )
    meta = json.load((case_db_dir / "meta.json").open())
    D = meta["global_dim"]
    print(f"train={len(train_cases)} valid={len(valid_cases)} D={D}")

    # Teacher
    teacher_full = None
    teacher_on_the_fly = None
    if args.teacher_mode == "precomputed":
        teacher_pkg = torch.load(
            args.teacher_path, weights_only=False, map_location="cpu",
        )
        teacher_full = teacher_pkg["scores"]                          # fp16 [Nt, Nt]
        if teacher_full.size(0) < len(train_cases):
            raise ValueError(
                f"precomputed teacher table is {teacher_full.size(0)}×"
                f"{teacher_full.size(0)} but train has {len(train_cases)} "
                f"cases — rebuild the teacher or use --teacher_mode on_the_fly."
            )
        teacher_full = teacher_full[:len(train_cases), :len(train_cases)]
        print(
            f"teacher table: {tuple(teacher_full.shape)}  "
            f"config={teacher_pkg['config']}"
        )
    else:
        bank_dtype = _BANK_DTYPES[args.bank_dtype]
        teacher_on_the_fly = OnTheFlyTeacher(
            train_cases, device=device, bank_dtype=bank_dtype,
            alpha=args.teacher_alpha, beta=args.teacher_beta,
            lesion_match=args.teacher_lesion_match,
        )
        print(
            f"teacher: on_the_fly  bank_dtype={args.bank_dtype}  "
            f"α={args.teacher_alpha} β={args.teacher_beta}  "
            f"lesion_match={args.teacher_lesion_match}  "
            f"G={tuple(teacher_on_the_fly.G.shape)} "
            f"L={tuple(teacher_on_the_fly.L.shape)}"
        )

    # Cause text embeddings for InfoNCE (frozen, L2-normed, kept on GPU)
    cause_text_embs = None
    if args.use_infonce:
        cause_pkg = torch.load(Path(args.case_db_dir) / "cause_text_embs.pt",
                               weights_only=False, map_location="cpu")
        cause_text_embs = cause_pkg["embeddings"].to(device)
        cause_text_embs = F.normalize(cause_text_embs.float(), dim=-1)
        V = cause_text_embs.size(0)
        print(f"cause_text_embs: [{V}, {D}]  (InfoNCE on, weight={args.infonce_weight}, "
              f"T={args.infonce_temp})")

    # Build the training Dataset with ``free_source=True`` so per-case
    # embedding tensors are dropped from ``train_cases`` immediately after
    # being normalized into ``train_ds.records``. Keeping both lists costs
    # 8-18 GB CPU on DDXPlus 200k. Source teacher bank (on GPU) and
    # downstream metadata (``causes`` / ``cause_emb_indices``) are
    # unaffected — only the embedding tensors are freed.
    train_ds = CaseEncoderDataset(train_cases, free_source=True)
    gc.collect()
    collate = make_collate(D)

    sampler = None
    if args.miss_weight > 1.0:
        # Hard-case mining: compute leave-one-out teacher miss count per train case
        # using the precomputed candidate pool, then build a weighted sampler.
        pool = torch.load(args.train_pool_path, weights_only=False,
                          map_location="cpu")["case_pool"]
        cause_embs_norm = F.normalize(
            torch.load(Path(args.case_db_dir) / "cause_text_embs.pt",
                       weights_only=False, map_location="cpu")["embeddings"].float(),
            dim=-1,
        ).to(device)

        weights = torch.ones(len(train_ds), dtype=torch.float32)
        n_hard = 0
        for i in range(len(train_ds)):
            gt_idx = train_cases[i]["cause_emb_indices"]
            pool_idx = pool[i]["candidate_cause_indices"].tolist()
            if not pool_idx or not gt_idx:
                miss = len(gt_idx)
            else:
                gt_e = cause_embs_norm[
                    torch.tensor(gt_idx, dtype=torch.long, device=device)]
                pool_e = cause_embs_norm[
                    torch.tensor(pool_idx, dtype=torch.long, device=device)]
                cos = gt_e @ pool_e.T
                miss = int((~(cos >= 0.95).any(dim=1)).sum().item())
            if miss > 0:
                n_hard += 1
            weights[i] = 1.0 + args.miss_weight * miss
        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(train_ds), replacement=True,
        )
        print(f"[miss-weight] hard cases (miss>=1): {n_hard}/{len(train_ds)}  "
              f"weight factor={args.miss_weight}  "
              f"max_weight={weights.max():.1f}")

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
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
        epoch_loss_distill = 0.0
        epoch_loss_infonce = 0.0
        n_batches = 0
        t0 = time.time()
        for batch in loader:
            case_ids = batch["case_ids"]
            g = batch["global_emb"].to(device)
            L = batch["lesion_pad"].to(device)
            lens = batch["lesion_lens"].to(device)

            z = encoder(g, L, lens)                                   # [B, D]
            if teacher_on_the_fly is not None:
                teacher_block = teacher_on_the_fly.batch_block(
                    case_ids.to(device),
                )
            else:
                teacher_block = teacher_full[case_ids][:, case_ids].to(device).float()

            if args.loss_type == "listwise_kl":
                loss_distill = listwise_kl_loss(z, teacher_block,
                                                temp_target=args.temp_target,
                                                temp_pred=args.temp_pred)
            else:
                loss_distill = pairwise_mse_loss(z, teacher_block)

            if args.use_infonce:
                B = z.size(0)
                V = cause_text_embs.size(0)
                pos_mask = torch.zeros(B, V, dtype=torch.bool, device=device)
                if args.infonce_positives == "pathology":
                    for i, pidx in enumerate(batch["pathology_indices"]):
                        if pidx is None:
                            raise ValueError(
                                "--infonce_positives pathology requires "
                                "pathology_emb_idx on every case "
                                "(DDXPlus v2 schema). Case at batch index "
                                f"{i} (case_id={int(batch['case_ids'][i])}) "
                                "is missing it — rebuild the case_db with "
                                "the v2 schema or use "
                                "--infonce_positives cause_emb_indices."
                            )
                        pos_mask[i, int(pidx)] = True
                else:
                    for i, cidxs in enumerate(batch["cause_indices"]):
                        if cidxs:
                            pos_mask[i, torch.tensor(cidxs, dtype=torch.long,
                                                     device=device)] = True
                loss_infonce = case_cause_infonce_loss(
                    z, cause_text_embs, pos_mask, temp=args.infonce_temp,
                )
                loss = loss_distill + args.infonce_weight * loss_infonce
            else:
                loss_infonce = torch.tensor(0.0, device=device)
                loss = loss_distill

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optim.step()
            sched.step()

            epoch_loss += loss.item()
            epoch_loss_distill += loss_distill.item()
            epoch_loss_infonce += loss_infonce.item()
            n_batches += 1
            global_step += 1

        train_loss = epoch_loss / max(1, n_batches)
        epoch_dt = time.time() - t0

        log_row = {"epoch": epoch + 1, "train_loss": train_loss,
                   "loss_distill": epoch_loss_distill / max(1, n_batches),
                   "loss_infonce": epoch_loss_infonce / max(1, n_batches),
                   "lr": optim.param_groups[0]["lr"], "time_s": epoch_dt}

        if (epoch + 1) % args.eval_every_epochs == 0:
            t1 = time.time()
            # Reuse train_ds (records list lives in RAM with normalized
            # embeddings) instead of rebuilding the dataset from
            # train_cases — train_cases no longer holds the embedding
            # tensors after free_source=True.
            H_train = encode_all(encoder, train_ds, device)
            H_valid = encode_all(encoder, valid_cases, device)
            metrics = retrieval_metrics(
                H_valid.to(device), H_train.to(device),
                valid_cases, train_cases,
                query_batch=args.eval_query_batch,
            )
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

        if args.early_stop_patience > 0 and patience >= args.early_stop_patience:
            print(f"Early stop at epoch {epoch + 1} (patience hit).")
            break

    torch.save({"encoder_state": encoder.state_dict(),
                "encoder_config": vars(cfg)},
               out_dir / "last_encoder.pt")
    log_f.close()
    print(f"Done. best sem_R@10 = {best_metric:.4f}")


if __name__ == "__main__":
    main()
