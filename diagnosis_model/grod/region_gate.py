"""Region Gate — ∅-sink temperature softmax over per-region objectness logits.

Replaces the production ``w_i = sigmoid(objectness_i)`` soft gate (see
extract_soft_inputs.py) for end-to-end training. Motivation
([[project_region_gate_e2e]]):

  * sigmoid two-side saturates -> ∂L/∂o_i = (∂L/∂w_i)·w_i(1-w_i) ≈ 0 once the
    detector is confident, so the objectness head receives ~no gradient from the
    downstream CEAH loss. End-to-end training stalls exactly where it matters.
  * plain softmax fixes the gradient (competition) but forces Σw = 1, so a
    healthy fish (all objectness low) still gets total weight 1 forced onto real
    boxes — wrong.

Region Gate = softmax good-gradient + sigmoid "healthy can be all-zero":

        w = softmax([o_1/τ, …, o_Q/τ, o_∅/τ])[:Q]      # ∅ column dropped

A learnable no-object (∅) logit competes with every region; when all regions
score below ∅ the mass flows to ∅ and real-box weights collapse to ≈0. ∅ is a
learnable, *relative* threshold (sigmoid's implicit threshold is the fixed 0).
This is the DETR no-object mechanism reused as the gate — not a bolt-on.

The absolute "how much evidence" magnitude that softmax normalization discards
is exposed separately via ``evidence_magnitude`` (smooth-max logsumexp) for the
abstain / disease head, which currently reads [max σ, Σ σ].

Run a shape + healthy-collapse self-test:
    $PY -m diagnosis_model.grod.region_gate
"""

from __future__ import annotations

import math

import torch
from torch import nn


class RegionGate(nn.Module):
    def __init__(
        self,
        init_temp: float = 0.07,
        init_sink: float = 0.0,
        learn_temp: bool = True,
        learn_sink: bool = True,
    ):
        super().__init__()
        log_temp = torch.tensor(math.log(init_temp), dtype=torch.float32)
        sink = torch.tensor(float(init_sink), dtype=torch.float32)
        if learn_temp:
            self.log_temp = nn.Parameter(log_temp)
        else:
            self.register_buffer("log_temp", log_temp, persistent=True)
        if learn_sink:
            self.sink = nn.Parameter(sink)
        else:
            self.register_buffer("sink", sink, persistent=True)

    @property
    def temp(self) -> torch.Tensor:
        return self.log_temp.exp()

    def _scaled(self, obj_logits: torch.Tensor, valid_mask: torch.Tensor | None):
        """obj_logits/τ with padded queries masked to -inf. [B, Q]."""
        o = obj_logits / self.temp
        if valid_mask is not None:
            o = o.masked_fill(~valid_mask, float("-inf"))
        return o

    def forward(
        self,
        obj_logits: torch.Tensor,            # [B, Q] raw objectness logits
        valid_mask: torch.Tensor | None = None,  # [B, Q] bool, True = real query
    ) -> torch.Tensor:
        """-> w [B, Q] in [0,1], Σ_i w_i + w_∅ = 1 (∅ dropped from the return)."""
        o = self._scaled(obj_logits, valid_mask)             # [B, Q]
        sink = (self.sink / self.temp).expand(o.size(0), 1)  # [B, 1]
        w_full = torch.softmax(torch.cat([o, sink], dim=1), dim=1)
        return w_full[:, :-1]                                # drop ∅ column

    def evidence_magnitude(
        self,
        obj_logits: torch.Tensor,            # [B, Q]
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Smooth-max total evidence τ·logsumexp(o/τ) [B] for the abstain head.

        Unbounded, gradient everywhere, recovers the magnitude that the softmax
        gate normalizes away. Sink is intentionally excluded so an empty/healthy
        image yields a low (not pinned) magnitude.
        """
        o = self._scaled(obj_logits, valid_mask)
        return self.temp * torch.logsumexp(o, dim=1)


def _selftest():
    torch.manual_seed(0)
    B, Q = 4, 300
    # row 0: healthy (all low); row 1: ONE dominant lesion (degenerate one-hot);
    # rows 2-3: a few competing moderate lesions (the typical case).
    logits = torch.full((B, Q), -3.0)
    logits[1, 7] = 5.0
    logits[2, [3, 11, 40]] = torch.tensor([1.0, 0.8, 0.6])
    logits[3, [5, 9]] = torch.tensor([1.2, 1.0])

    gate = RegionGate(init_temp=0.07, init_sink=0.0)
    w = gate(logits)
    print("w.shape", tuple(w.shape), "Σw per row:", [round(x, 4) for x in w.sum(1).tolist()])
    print("healthy row max w:", f"{w[0].max():.2e}", "(want ≈0 — ∅ absorbs)")
    print("lesion row argmax:", w[1].argmax().item(), "(want 7)")
    print("evidence magnitude:", [round(x, 3) for x in gate.evidence_magnitude(logits).tolist()])

    # Gradient probe: a downstream-like loss on the RELATIVE distribution
    # (competition), which is what aggregation actually uses. Compare against a
    # plain-sigmoid gate on the SAME logits.
    def grad_norm_per_row(make_w):
        lg = logits.clone().requires_grad_(True)
        ww = make_w(lg)
        # mock aggregation target: push weight toward the first valid lesion
        (ww[:, :12].sum()).backward()
        return lg.grad.norm(dim=1)

    g_region = grad_norm_per_row(lambda lg: gate(lg))
    g_sigmoid = grad_norm_per_row(lambda lg: lg.sigmoid())
    print("\nper-row grad‖·‖   [healthy, one-hot, 3-lesion, 2-lesion]")
    print("  RegionGate:", [round(x, 4) for x in g_region.tolist()])
    print("  sigmoid   :", [round(x, 4) for x in g_sigmoid.tolist()])
    print("note: row 1 (single dominant lesion) is degenerate — no softmax keeps "
          "gradient there; the win is rows 2-3 (competing lesions) + that sigmoid "
          "is flat ~0 on the confident logits regardless.")


if __name__ == "__main__":
    _selftest()
