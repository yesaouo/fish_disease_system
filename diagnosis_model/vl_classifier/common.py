"""Shared helpers used by train.py and eval.py."""

import json
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



def get_image_features(model, pixel_values: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "get_image_features"):
        return model.get_image_features(pixel_values=pixel_values)

    out = model(pixel_values=pixel_values, return_dict=True)
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return _pool_from_last_hidden(out.last_hidden_state)

    if hasattr(model, "vision_model"):
        vout = model.vision_model(pixel_values=pixel_values, return_dict=True)
        if hasattr(vout, "pooler_output") and vout.pooler_output is not None:
            return vout.pooler_output
        if hasattr(vout, "last_hidden_state") and vout.last_hidden_state is not None:
            return _pool_from_last_hidden(vout.last_hidden_state)

    raise RuntimeError("Cannot extract image features from model output. Please use a dual-encoder model (CLIP/SigLIP).")



def get_text_features(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if hasattr(model, "get_text_features"):
        return model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

    out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return _pool_from_last_hidden(out.last_hidden_state)

    if hasattr(model, "text_model"):
        tout = model.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if hasattr(tout, "pooler_output") and tout.pooler_output is not None:
            return tout.pooler_output
        if hasattr(tout, "last_hidden_state") and tout.last_hidden_state is not None:
            return _pool_from_last_hidden(tout.last_hidden_state)

    raise RuntimeError("Cannot extract text features from model output. Please use a dual-encoder model (CLIP/SigLIP).")


# =========================
# Local-Global fusion wrapper
# =========================
class LocalGlobalFusionWrapper(nn.Module):
    def __init__(self, base_model, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.base_model = base_model
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fusion_linear = nn.Linear(hidden_size * 2, hidden_size)
        nn.init.xavier_uniform_(self.fusion_linear.weight)
        nn.init.zeros_(self.fusion_linear.bias)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward_image(self, pixel_values_local, pixel_values_global, return_parts: bool = False):
        local_feat = get_image_features(self.base_model, pixel_values_local)
        global_feat = get_image_features(self.base_model, pixel_values_global)

        local_feat = F.normalize(local_feat, dim=-1)
        global_feat = F.normalize(global_feat, dim=-1)

        fused = torch.cat([local_feat, global_feat], dim=-1)
        fused = self.fusion_linear(fused)
        fused = self.gelu(fused)
        fused = self.dropout(fused)

        out = local_feat + self.gate * fused

        if return_parts:
            return out, local_feat, global_feat, fused
        return out

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
