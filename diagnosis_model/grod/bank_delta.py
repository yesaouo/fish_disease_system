"""Mutable *delta* layer over the frozen production case bank.

The production retrieval bank (`bank_z_soft` + `case_db_jointDistRawP`) is a static
offline artifact: the 12780 training cases from the 4 locked datasets. This module
adds a small, in-memory *delta* of expert-submitted cases (from annotation_web's
`created_via:diagnosis` datasets) so the HITL loop closes — new cases become
retrievable / CEAH candidates immediately, with zero model retraining.

Design (see plan): frozen base ⊕ mutable delta, keyed by
`case_id = (source_dataset, source_task_id)`.

  bank_z_full     = cat([ base_z (never mutated) , stack(delta.z) ])
  cause_embs_full = cat([ base_cause_embs , cat(delta.cause_embs) ])   # delta memb offset
  memb/mlen/file_names/case_meta = base ⊕ delta

Delta is small, so delete = drop from the dict + full rebuild of the small delta
tensors (base is never touched → no tombstones, no compaction, no risk to the
paper-versioned numbers). Edit / re-submit = upsert (same case_id, in place).

No disk persistence: delta is rebuilt on serve restart via annotation_web resync.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class DeltaCase:
    case_id: str
    source_dataset: str
    source_task_id: str
    image_path: str
    file_name: str
    z: torch.Tensor            # [768]  case vector (soft-aggregated, same as bank_z_soft)
    cause_texts: List[str]     # this case's cause strings
    cause_embs: torch.Tensor   # [C,768] L2-normalized, same space as base cause_embs


@dataclass
class BankView:
    """Merged base ⊕ delta bank, ready to drop into GpuPipelineSoft's hot fields."""
    bank_z: torch.Tensor        # [N,768]
    memb: torch.Tensor          # [N,max_c] long, -1 pad
    mlen: torch.Tensor          # [N] long
    cause_embs: torch.Tensor    # [Ncause,768]
    cause_texts: List[str]
    file_names: List[str]
    case_meta: List[dict]       # per case: {image_path, source_dataset, source_task_id}


def _pad_memb(memb: torch.Tensor, width: int) -> torch.Tensor:
    """Pad a [N,c] memb tensor to [N,width] with -1 (no-op when already wide enough)."""
    if memb.size(1) >= width:
        return memb
    pad = torch.full((memb.size(0), width - memb.size(1)), -1,
                     dtype=memb.dtype, device=memb.device)
    return torch.cat([memb, pad], dim=1)


class BankDelta:
    """In-memory registry of expert-submitted delta cases, keyed by case_id."""

    def __init__(self) -> None:
        self._records: "OrderedDict[str, DeltaCase]" = OrderedDict()

    @staticmethod
    def make_case_id(source_dataset: str, source_task_id: str) -> str:
        return f"{source_dataset}::{source_task_id}"

    def list_case_ids(self) -> List[str]:
        return list(self._records.keys())

    def __len__(self) -> int:
        return len(self._records)

    def upsert(self, rec: DeltaCase) -> None:
        # In-place replace preserves position; new key appends. Either way order
        # is irrelevant to retrieval (it's a set of bank rows).
        self._records[rec.case_id] = rec

    def delete(self, case_id: str) -> bool:
        return self._records.pop(case_id, None) is not None

    def combined(
        self,
        base_bank_z: torch.Tensor,
        base_memb: torch.Tensor,
        base_mlen: torch.Tensor,
        base_cause_embs: torch.Tensor,
        base_cause_texts: List[str],
        base_file_names: List[str],
        base_case_meta: List[dict],
    ) -> BankView:
        """Merge base tensors with the current delta. Returns base unchanged (same
        objects) when delta is empty → byte-identical to the pre-delta pipeline."""
        if not self._records:
            return BankView(base_bank_z, base_memb, base_mlen, base_cause_embs,
                            base_cause_texts, base_file_names, base_case_meta)

        recs = list(self._records.values())
        device = base_bank_z.device

        # --- case vectors ---
        delta_z = torch.stack([r.z.to(device) for r in recs], dim=0)          # [Nd,768]
        bank_z = torch.cat([base_bank_z, delta_z], dim=0)

        # --- cause embeddings (append delta causes after base ones) ---
        n_base_causes = len(base_cause_texts)
        delta_cause_embs = torch.cat([r.cause_embs.to(device) for r in recs], dim=0)
        cause_embs = torch.cat([base_cause_embs, delta_cause_embs], dim=0)
        cause_texts = list(base_cause_texts)
        for r in recs:
            cause_texts.extend(r.cause_texts)

        # --- memb / mlen for delta rows (contiguous blocks into the appended region) ---
        delta_mlen = torch.tensor([len(r.cause_texts) for r in recs],
                                  dtype=base_mlen.dtype, device=device)
        max_c = int(max(base_memb.size(1), int(delta_mlen.max().item())))
        delta_memb = torch.full((len(recs), max_c), -1,
                                dtype=base_memb.dtype, device=device)
        offset = n_base_causes
        for i, r in enumerate(recs):
            c = len(r.cause_texts)
            delta_memb[i, :c] = torch.arange(offset, offset + c, device=device)
            offset += c

        memb = torch.cat([_pad_memb(base_memb, max_c), delta_memb], dim=0)
        mlen = torch.cat([base_mlen, delta_mlen], dim=0)

        # --- provenance / display metadata ---
        file_names = list(base_file_names) + [r.file_name for r in recs]
        case_meta = list(base_case_meta) + [
            {"image_path": r.image_path,
             "source_dataset": r.source_dataset,
             "source_task_id": r.source_task_id}
            for r in recs
        ]

        return BankView(bank_z, memb, mlen, cause_embs, cause_texts, file_names, case_meta)
