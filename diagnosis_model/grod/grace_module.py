"""GraceModule — single differentiable forward for GRACE (part 1 / Case Encoding).

Packages the three trainable pieces that today live behind a disk dump
(extract_soft_inputs.py -> *.pt -> train_case_encoder_soft / train_ceah_soft)
into ONE nn.Module so the gate sits in the live graph and end-to-end gradient
reaches the GROD objectness head ([[project_region_gate_e2e]],
[[project_grod_motivation_arc]] step ③):

    image ─► GROD ─► {g[768], z_all[Q,768], obj_logits[Q]}
                          │
                          ├─► RegionGate softmax_τ(·,∅) ─► w[Q] ─┐
                          │                                      ├─► Aggregator ─► case vec
                          └─► logsumexp ─► evidence_magnitude ──► (abstain branch)

What this module deliberately does NOT fold in (keeps the paper's two-model +
non-parametric-bank framing intact):
  * case-bank top-k retrieval — non-differentiable lookup, a stop-grad boundary
    between GRACE and CEAH. Train CEAH on the refreshed candidate pool; gradient
    does not flow through retrieval (see finetune_e2e_soft.py rationale).
  * CEAH (part 2 / Cause Inference) — a separate model.

GROD trainability is the caller's choice: set ``requires_grad`` on the GROD
objectness/semantic heads (or pass a frozen net) — GraceModule does not wrap the
GROD forward in no_grad, so gradient flows back if the params allow it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from diagnosis_model.cause_inference.models.case_encoder import (
    EncoderConfig, build_encoder,
)
from diagnosis_model.grod.region_gate import RegionGate


class GraceModule(nn.Module):
    def __init__(
        self,
        grod_net: nn.Module,                 # RF-DETR joint model (box/obj/sem z/global)
        encoder_cfg: EncoderConfig | None = None,
        gate: RegionGate | None = None,
    ):
        super().__init__()
        self.grod = grod_net
        self.gate = gate if gate is not None else RegionGate()
        self.aggregator = build_encoder(encoder_cfg or EncoderConfig())

    # -- GROD front-end: pixels -> raw structured findings -------------------
    def run_grod(self, pixel_values: torch.Tensor) -> dict:
        """One GROD forward. Differentiable iff grod params require grad."""
        out = self.grod(pixel_values)
        return {
            "g": out["pred_global"].float(),                       # [B, 768]
            "z_all": F.normalize(out["pred_semantic"].float(), dim=-1),  # [B, Q, 768]
            "obj_logits": out["pred_logits"][..., 0].float(),      # [B, Q] col 0 = ABNORMAL
        }

    # -- aggregation back-end: findings -> single case vector ----------------
    def aggregate(self, g, z_all, obj_logits, valid_mask=None):
        """g[B,768], z_all[B,Q,768], obj_logits[B,Q] -> case vector + w + abstain."""
        B, Q, _ = z_all.shape
        w = self.gate(obj_logits, valid_mask)                      # [B, Q]
        lens = torch.full((B,), Q, device=z_all.device, dtype=torch.long)
        case_vec = self.aggregator(g, z_all, lens, lesion_weights=w)  # [B,768] L2-normed
        evidence = self.gate.evidence_magnitude(obj_logits, valid_mask)  # [B]
        return {"case_vec": case_vec, "w": w, "z_all": z_all,
                "evidence_magnitude": evidence}

    def forward(self, pixel_values: torch.Tensor, valid_mask=None) -> dict:
        f = self.run_grod(pixel_values)
        return self.aggregate(f["g"], f["z_all"], f["obj_logits"], valid_mask)
