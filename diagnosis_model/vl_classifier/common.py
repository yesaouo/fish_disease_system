"""Shared helpers used by train.py and eval.py."""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Prompt templates
# =========================
PROMPT_EN = "This is {cap}."
PROMPT_ZH = "{cap}。"


def format_caption(cap: str, lang: str) -> str:
    if lang == "en":
        return PROMPT_EN.format(cap=str(cap).lower())
    if lang == "zh":
        return PROMPT_ZH.format(cap=str(cap))
    return str(cap)
# def format_caption(cap: str, lang: str) -> str:
#     if lang == "en":
#         return str(cap)
#     if lang == "zh":
#         return str(cap)
#     return str(cap)


# =========================
# Encoder feature extractors
# =========================

def _pool_from_last_hidden(last_hidden: torch.Tensor) -> torch.Tensor:
    return last_hidden[:, 0]



def _unwrap_to_tensor(x) -> torch.Tensor:
    """Unwrap encoder outputs to a [B, D] embedding tensor.

    Some transformers versions return BaseModelOutput-like objects from
    get_image_features/get_text_features instead of returning the embedding
    tensor directly. Keep this compatibility logic in common.py so all callers,
    including LocalGlobalFusionWrapper, share the same behavior.
    """
    if isinstance(x, torch.Tensor):
        return x

    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(x, attr, None)
        if v is not None:
            return v

    last_hidden = getattr(x, "last_hidden_state", None)
    if last_hidden is not None:
        return _pool_from_last_hidden(last_hidden)

    raise RuntimeError(
        f"Cannot unwrap {type(x).__name__} to a feature tensor. "
        "Expected a Tensor or a model output with image_embeds, text_embeds, "
        "pooler_output, or last_hidden_state."
    )



def get_image_features(model, pixel_values: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "get_image_features"):
        return _unwrap_to_tensor(model.get_image_features(pixel_values=pixel_values))

    out = model(pixel_values=pixel_values, return_dict=True)
    try:
        return _unwrap_to_tensor(out)
    except RuntimeError:
        pass

    if hasattr(model, "vision_model"):
        vout = model.vision_model(pixel_values=pixel_values, return_dict=True)
        try:
            return _unwrap_to_tensor(vout)
        except RuntimeError:
            pass

    raise RuntimeError("Cannot extract image features from model output. Please use a dual-encoder model (CLIP/SigLIP).")



def get_image_patch_tokens(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return the vision tower's per-patch tokens [B, P, H] (pre-pool).

    Used by LocalGlobalFusionWrapper's `xattn` gate, which attends over the
    whole-fish patch grid instead of a single pooled global vector.
    """
    base = model.vision_model if hasattr(model, "vision_model") else model
    vout = base(pixel_values=pixel_values, return_dict=True)
    last_hidden = getattr(vout, "last_hidden_state", None)
    if last_hidden is None:
        raise RuntimeError(
            "xattn fusion needs the vision tower's last_hidden_state (patch tokens); "
            f"{type(vout).__name__} did not expose it."
        )
    return last_hidden


def get_text_features(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if hasattr(model, "get_text_features"):
        return _unwrap_to_tensor(
            model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        )

    out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    try:
        return _unwrap_to_tensor(out)
    except RuntimeError:
        pass

    if hasattr(model, "text_model"):
        tout = model.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        try:
            return _unwrap_to_tensor(tout)
        except RuntimeError:
            pass

    raise RuntimeError("Cannot extract text features from model output. Please use a dual-encoder model (CLIP/SigLIP).")


# =========================
# Local-Global fusion wrapper
# =========================
class LocalGlobalFusionWrapper(nn.Module):
    """Local (lesion crop) ⊕ global (whole-fish) image fusion.

    `gate_mode` controls how the fusion branch is mixed back into the local
    feature (`out = local + gate * fused`):

    - "scalar": a single learnable scalar (original behavior, init 0.1).
    - "film":  an input-conditioned per-channel gate
               `g = sigmoid(gate_net([local; global]))` of shape [B, hidden].
               Lets the whole-fish context modulate the lesion feature
               channel-by-channel and per-sample. Initialized so g ≈ 0.1
               everywhere, matching the scalar warm start for a fair compare.
    - "xattn": the lesion feature (query) cross-attends over the whole-fish
               *patch tokens* (key/value) instead of a single pooled global
               vector. The attended context replaces `global` in the existing
               `fusion_linear([local; ctx])` branch, mixed back via the same
               scalar gate (init 0.1) as "scalar" for a fair warm start. Lets
               the lesion pick *which* region of the fish is relevant rather
               than seeing only a global average.
    """

    def __init__(self, base_model, hidden_size: int, dropout_prob: float = 0.1, gate_mode: str = "scalar"):
        super().__init__()
        self.base_model = base_model
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fusion_linear = nn.Linear(hidden_size * 2, hidden_size)
        nn.init.xavier_uniform_(self.fusion_linear.weight)
        nn.init.zeros_(self.fusion_linear.bias)

        self.gate_mode = gate_mode
        if gate_mode == "scalar":
            self.gate = nn.Parameter(torch.tensor(0.1))
        elif gate_mode == "film":
            self.gate_net = nn.Linear(hidden_size * 2, hidden_size)
            nn.init.zeros_(self.gate_net.weight)
            nn.init.constant_(self.gate_net.bias, math.log(0.1 / 0.9))  # logit(0.1) ≈ -2.197 → g≈0.1 at init
        elif gate_mode == "xattn":
            self.gate = nn.Parameter(torch.tensor(0.1))
            num_heads = 8 if hidden_size % 8 == 0 else 1
            self.cross_attn = nn.MultiheadAttention(
                hidden_size, num_heads, dropout=dropout_prob, batch_first=True
            )
        else:
            raise ValueError(f"unknown gate_mode={gate_mode!r}, expected 'scalar', 'film', or 'xattn'")

    def _gate(self, local_feat_normalized: torch.Tensor, global_feat_normalized: torch.Tensor):
        """Return the gate to multiply with `fused`: a scalar Parameter or a [B, hidden] tensor."""
        if self.gate_mode == "scalar":
            return self.gate
        return torch.sigmoid(self.gate_net(torch.cat([local_feat_normalized, global_feat_normalized], dim=-1)))

    def _fuse_xattn(self, local_feat_normalized: torch.Tensor, global_tokens_normalized: torch.Tensor):
        """Cross-attend local (query) over global patch tokens (key/value), then
        mix the context back through the shared fusion_linear + scalar gate.

        Returns (out, fused). `out` is NOT renormalized (caller's responsibility).
        """
        query = local_feat_normalized.unsqueeze(1)  # [B, 1, H]
        ctx, _ = self.cross_attn(
            query, global_tokens_normalized, global_tokens_normalized, need_weights=False
        )
        ctx = ctx.squeeze(1)  # [B, H]
        fused = torch.cat([local_feat_normalized, ctx], dim=-1)
        fused = self.fusion_linear(fused)
        fused = self.gelu(fused)
        fused = self.dropout(fused)
        out = local_feat_normalized + self.gate * fused
        return out, fused

    def forward_image(self, pixel_values_local, pixel_values_global, return_parts: bool = False):
        local_feat = F.normalize(get_image_features(self.base_model, pixel_values_local), dim=-1)

        if self.gate_mode == "xattn":
            # `global` is the whole-fish patch grid; cache it (not a pooled vector)
            # so fuse_local_with_global_feat can reuse it on jittered crops (LSCFT).
            global_feat = F.normalize(
                get_image_patch_tokens(self.base_model, pixel_values_global), dim=-1
            )
            out, fused = self._fuse_xattn(local_feat, global_feat)
            if return_parts:
                return out, local_feat, global_feat, fused
            return out

        global_feat = F.normalize(get_image_features(self.base_model, pixel_values_global), dim=-1)

        fused = torch.cat([local_feat, global_feat], dim=-1)
        fused = self.fusion_linear(fused)
        fused = self.gelu(fused)
        fused = self.dropout(fused)

        out = local_feat + self._gate(local_feat, global_feat) * fused

        if return_parts:
            return out, local_feat, global_feat, fused
        return out

    def fuse_local_with_global_feat(self, local_feat_normalized: torch.Tensor, global_feat_normalized: torch.Tensor) -> torch.Tensor:
        """Reuse fusion module on a new local feature with a precomputed global.

        Both inputs must already be L2-normalized (caller's responsibility).
        For "xattn", `global_feat_normalized` is the [B, P, H] patch-token grid
        cached by forward_image, not a pooled [B, H] vector.
        Output is NOT renormalized — the caller normalizes after gathering all
        branches, matching the existing forward_image -> external F.normalize
        flow in compute_custom_batch_loss.
        """
        if self.gate_mode == "xattn":
            out, _ = self._fuse_xattn(local_feat_normalized, global_feat_normalized)
            return out

        fused = torch.cat([local_feat_normalized, global_feat_normalized], dim=-1)
        fused = self.fusion_linear(fused)
        fused = self.gelu(fused)
        fused = self.dropout(fused)
        return local_feat_normalized + self._gate(local_feat_normalized, global_feat_normalized) * fused

    def get_text_features(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return get_text_features(self.base_model, input_ids, attention_mask)


# =========================
# symptoms.json caption bank
# =========================

def sort_label_ids(label_ids) -> List[str]:
    return sorted(
        set(str(x) for x in label_ids),
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )


@dataclass
class FlatCaptionBank:
    """Flat view of symptoms.json with everything callers need.

    All list-typed fields are paired by index (texts[i] is for label_ids[i] in language langs[i]).
    """
    texts: List[str] = field(default_factory=list)              # already prompt-wrapped via format_caption
    label_ids: List[str] = field(default_factory=list)          # category_id as str
    langs: List[str] = field(default_factory=list)
    id_to_zh: Dict[str, str] = field(default_factory=dict)      # cat_id -> zh display name (from label_map)
    raw_by_cat: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)  # cat_id -> [(raw_caption, lang), ...]
    sorted_label_ids: List[str] = field(default_factory=list)


def load_flat_caption_bank(
    symptoms_path: str,
    langs: Tuple[str, ...] = ("en", "zh"),
) -> FlatCaptionBank:
    """Load symptoms.json and build a flat caption bank + per-category structure.

    Bank order: categories sorted (numeric-aware), then by `langs` argument order, then caption order.
    Raises if a category has zero usable captions for the requested langs.
    """
    if not symptoms_path or not os.path.exists(symptoms_path):
        raise FileNotFoundError(f"symptoms.json not found: {symptoms_path}")

    with open(symptoms_path, "r", encoding="utf-8") as f:
        s = json.load(f)

    if "data" not in s or not isinstance(s["data"], dict):
        raise ValueError("symptoms.json format error: missing dict field 'data'")

    label_map = s.get("label_map", {}) or {}
    id_to_zh: Dict[str, str] = {}
    for k, info in label_map.items():
        if isinstance(info, dict):
            id_to_zh[str(k)] = info.get("zh", str(k))
        else:
            id_to_zh[str(k)] = str(k)

    raw_by_cat: Dict[str, List[Tuple[str, str]]] = {}
    texts: List[str] = []
    label_ids: List[str] = []
    langs_out: List[str] = []

    keys_sorted = sorted(
        s["data"].keys(),
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )

    for k in keys_sorted:
        v = s["data"][k]
        if not isinstance(v, dict):
            continue
        cat_id = str(k)
        pairs: List[Tuple[str, str]] = []
        for lang in langs:
            key = f"captions_{lang}"
            caps = v.get(key, None)
            if caps is None:
                continue
            if not isinstance(caps, list):
                raise ValueError(f"{key} for category_id={cat_id} is not a list")
            for cap in caps:
                if isinstance(cap, str) and cap.strip():
                    pairs.append((cap.strip(), lang))
        if not pairs:
            raise ValueError(f"symptoms.json category_id={cat_id} has no usable captions for langs={langs}")
        raw_by_cat[cat_id] = pairs
        for cap, lang in pairs:
            texts.append(format_caption(cap, lang))
            label_ids.append(cat_id)
            langs_out.append(lang)

    if not texts:
        raise ValueError("symptoms.json has no usable captions")

    return FlatCaptionBank(
        texts=texts,
        label_ids=label_ids,
        langs=langs_out,
        id_to_zh=id_to_zh,
        raw_by_cat=raw_by_cat,
        sorted_label_ids=sort_label_ids(label_ids),
    )
