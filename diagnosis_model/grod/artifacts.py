"""Shared artifact loaders for the OAVLE / CEAM eval + calibration scripts.

Load a CEAH, an Aggregator, a bank, a case_db or a cluster-id array from here
rather than open-coding it — a mistyped `common_dim` silently loads the wrong
weights. `CEAH_ARCH` is the canonical CEAH architecture; pass overrides only
when deliberately loading a non-canonical checkpoint.

`ART` is the current-tree artifact root, the one paper evals must use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from diagnosis_model.cause_inference.models import CEAH
from diagnosis_model.cause_inference.models.case_encoder import EncoderConfig, build_encoder

ART = Path("data/processed/current/artifacts")

#: Canonical CEAH architecture (see cause_inference/README — sigmoid α and
#: additive scoring both collapse; do not change without a strong reason).
CEAH_ARCH = dict(common_dim=256, hidden_dim=512, dropout=0.0,
                 attribution_mode="softmax", scoring_mode="multiplicative")


def load_ceah(ckpt, in_dim: int, device, **arch_overrides) -> CEAH:
    """Build a CEAH at the canonical architecture and load `ckpt` into it."""
    arch = {**CEAH_ARCH, **arch_overrides}
    ceah = CEAH(global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim,
                cause_dim=in_dim, **arch).to(device).eval()
    ceah.load_state_dict(torch.load(ckpt, map_location=device))
    return ceah


def load_encoder(ckpt, device):
    """Build the Aggregator from the config stored in its own checkpoint."""
    pkg = torch.load(ckpt, weights_only=False, map_location="cpu")
    enc = build_encoder(EncoderConfig(**pkg["encoder_config"])).to(device).eval()
    enc.load_state_dict(pkg["encoder_state"])
    return enc


def load_bank(path, device) -> torch.Tensor:
    """Precomputed case-vector bank (`bank_z_soft.pt`)."""
    return torch.load(path, weights_only=False)["bank_z"].to(device)


@dataclass
class CaseDB:
    """The three tensors every CEAM eval needs out of a case_db directory."""
    train_cases: list
    query_cases: list
    cause_texts: List[str]
    cause_embs: torch.Tensor          # [V, D] float, on device

    @property
    def in_dim(self) -> int:
        return self.cause_embs.size(-1)


def load_case_db(case_db_dir, split: str, device) -> CaseDB:
    d = Path(case_db_dir)
    pack = torch.load(d / "cause_text_embs.pt", weights_only=False)
    return CaseDB(
        train_cases=torch.load(d / "train_cases.pt", weights_only=False),
        query_cases=torch.load(d / f"{split}_cases.pt", weights_only=False),
        cause_texts=pack["texts"],
        cause_embs=pack["embeddings"].float().to(device),
    )


def load_cluster_ids(cluster_json, cause_texts: List[str]) -> Optional[np.ndarray]:
    """-> [V] cluster id per cause text, or None when `cluster_json` is falsy."""
    if not cluster_json:
        return None
    o2c = json.load(open(cluster_json, encoding="utf-8"))["original_to_cause_id"]
    return np.array([int(o2c[t]) for t in cause_texts], dtype=np.int64)
