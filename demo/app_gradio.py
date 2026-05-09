"""End-to-end Gradio demo for the FaCE-R fish disease pipeline.

Stages
  1. RF-DETR — detect lesion bboxes (single class: ABNORMAL)
  2. VLM-Lesion (fusion) — classify each lesion against symptoms.json captions
                          + gradient-based heatmap on the global image
  3. VLM-Global — encode whole-image and optional overall text description
  4. FaCE-R Phase 1 + CEAH — retrieve top-K cases, score candidate causes,
                              return top-N causes with α (global / text / per-lesion)

Run from repo root:
  /home/lab603/anaconda3/envs/SDM/bin/python demo/app_gradio.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "diagnosis_model" / "vl_classifier"))

import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

# vl_classifier helpers (added to sys.path above)
from common import (  # noqa: E402
    LocalGlobalFusionWrapper,
    get_image_features,
    get_text_features,
    format_caption,
)

# cause_inference helpers
from diagnosis_model.cause_inference.models import CEAH  # noqa: E402
from diagnosis_model.cause_inference.phase1_baseline import (  # noqa: E402
    build_candidate_pool, compute_case_similarities, diversify,
    score_candidates, select_positive_top_cases, stack_train_lesions,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RFDETR_CKPT     = REPO_ROOT / "diagnosis_model/detection/outputs/rfdetr/checkpoint_best_total.pth"
VLM_GLOBAL_DIR  = REPO_ROOT / "diagnosis_model/vl_classifier/outputs/siglip2_base_patch16_224_overall_multipos_zh"
VLM_LESION_DIR  = REPO_ROOT / "diagnosis_model/vl_classifier/outputs/siglip2_base_patch16_224_multipos_fusion_en_zh"
CEAH_CKPT       = REPO_ROOT / "diagnosis_model/cause_inference/outputs/ceah_v3/best_ceah.pt"
CASE_DB_DIR     = REPO_ROOT / "diagnosis_model/cause_inference/outputs/case_db"
CLUSTER_JSON    = REPO_ROOT / "diagnosis_model/cause_inference/outputs/cause_clusters_llm.json"
SYMPTOMS_JSON   = REPO_ROOT / "data/raw/symptoms.json"
TRAIN_IMG_ROOT  = REPO_ROOT / "data/detection/coco/_merged/train"
VALID_IMG_ROOT  = REPO_ROOT / "data/detection/coco/_merged/valid"

CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

MAX_TOPN_BUTTONS = 10


def _font(size: int = 12) -> FontProperties:
    if os.path.exists(CJK_FONT_PATH):
        return FontProperties(fname=CJK_FONT_PATH, size=size)
    return FontProperties(size=size)


# ---------------------------------------------------------------------------
# Model loading (singleton)
# ---------------------------------------------------------------------------

class State:
    rfdetr = None
    vlm_global = None
    vlm_global_proc = None
    vlm_lesion = None
    vlm_lesion_proc = None
    ceah = None
    train_cases = None
    valid_cases = None
    cause_table_embs = None
    cause_texts = None
    cluster_id_array = None
    train_global_stack = None
    train_lesion_stack = None
    train_offsets = None
    in_dim = None
    # symptom classification
    sym_text_feats = None       # [n_sym, D] L2-normalized in VLM-Lesion space
    sym_label_ids = None        # List[str]: parallel to sym_text_feats rows
    sym_langs = None            # List[str]
    sym_raw_texts = None        # List[str]: the prompt (formatted) actually fed into VLM
    sym_id_to_zh = None         # Dict[str, str]
    sym_id_to_en = None         # Dict[str, str]


def _load_vlm(path: str, force_fusion: bool):
    from transformers import AutoModel, AutoProcessor
    processor = AutoProcessor.from_pretrained(path)
    base = AutoModel.from_pretrained(path).to(DEVICE)
    wrap_path = os.path.join(path, "wrapper_state.pt")
    use_wrap = bool(force_fusion or os.path.exists(wrap_path))
    if use_wrap:
        if not os.path.exists(wrap_path):
            raise FileNotFoundError(f"force_fusion but wrapper_state.pt missing: {path}")
        dummy = Image.new("RGB", (224, 224), (0, 0, 0))
        px = processor(images=[dummy], return_tensors="pt")["pixel_values"].to(DEVICE)
        with torch.no_grad():
            d = get_image_features(base, px).shape[-1]
        m = LocalGlobalFusionWrapper(base, hidden_size=d).to(DEVICE)
        m.load_state_dict(torch.load(wrap_path, map_location=DEVICE))
        m.is_wrapper = True
    else:
        m = base
        m.is_wrapper = False
    m.eval()
    return m, processor


def _load_symptoms_bank():
    """Encode symptoms.json captions via VLM-Lesion text tower for lesion classification."""
    with open(SYMPTOMS_JSON, encoding="utf-8") as f:
        s = json.load(f)
    label_map = s.get("label_map", {})
    id_to_zh: Dict[str, str] = {}
    id_to_en: Dict[str, str] = {}
    for k, v in label_map.items():
        if isinstance(v, dict):
            id_to_zh[str(k)] = v.get("zh", str(k))
            id_to_en[str(k)] = v.get("en", str(k))
        else:
            id_to_zh[str(k)] = str(v)
            id_to_en[str(k)] = str(k)

    texts: List[str] = []
    label_ids: List[str] = []
    langs: List[str] = []
    raw_texts: List[str] = []
    keys_sorted = sorted(s["data"].keys(), key=lambda x: int(x) if x.isdigit() else x)
    for k in keys_sorted:
        v = s["data"][k]
        if not isinstance(v, dict):
            continue
        for lang in ("en", "zh"):
            for cap in v.get(f"captions_{lang}", []) or []:
                if isinstance(cap, str) and cap.strip():
                    texts.append(format_caption(cap.strip(), lang))
                    raw_texts.append(cap.strip())
                    label_ids.append(str(k))
                    langs.append(lang)

    proc = State.vlm_lesion_proc
    model = State.vlm_lesion  # fusion wrapper; .get_text_features delegates to base
    feats = []
    bs = 256
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        inp = proc(text=batch, return_tensors="pt",
                   padding="max_length", truncation=True, max_length=64)
        inp = {kk: vv.to(DEVICE) for kk, vv in inp.items()}
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=DEVICE.startswith("cuda")):
            f = model.get_text_features(inp["input_ids"], inp.get("attention_mask"))
        f = F.normalize(f.float(), dim=-1).cpu()
        feats.append(f)
    State.sym_text_feats = torch.cat(feats, dim=0).to(DEVICE)
    State.sym_label_ids = label_ids
    State.sym_langs = langs
    State.sym_raw_texts = raw_texts
    State.sym_id_to_zh = id_to_zh
    State.sym_id_to_en = id_to_en


def load_all():
    print(f"[init] device={DEVICE}")
    print(f"[load] RF-DETR: {RFDETR_CKPT}")
    from rfdetr import RFDETRMedium
    State.rfdetr = RFDETRMedium(pretrain_weights=str(RFDETR_CKPT), num_classes=1)
    State.rfdetr.optimize_for_inference(compile=False)

    print(f"[load] VLM-Global: {VLM_GLOBAL_DIR}")
    State.vlm_global, State.vlm_global_proc = _load_vlm(str(VLM_GLOBAL_DIR), force_fusion=False)
    if State.vlm_global.is_wrapper:
        raise RuntimeError("VLM-Global unexpectedly loaded as fusion wrapper")
    print(f"[load] VLM-Lesion: {VLM_LESION_DIR}")
    State.vlm_lesion, State.vlm_lesion_proc = _load_vlm(str(VLM_LESION_DIR), force_fusion=True)
    if not State.vlm_lesion.is_wrapper:
        raise RuntimeError("VLM-Lesion failed to load as fusion wrapper")

    print(f"[load] case_db: {CASE_DB_DIR}")
    State.train_cases = torch.load(CASE_DB_DIR / "train_cases.pt", weights_only=False)
    State.valid_cases = torch.load(CASE_DB_DIR / "valid_cases.pt", weights_only=False)
    pack = torch.load(CASE_DB_DIR / "cause_text_embs.pt", weights_only=False)
    # Match phase1_baseline.py: explicitly L2-normalize so dot products are cosine.
    State.cause_table_embs = F.normalize(pack["embeddings"].to(DEVICE), dim=-1)
    State.cause_texts = pack["texts"]
    State.in_dim = State.cause_table_embs.size(-1)

    if CLUSTER_JSON.exists():
        with open(CLUSTER_JSON, encoding="utf-8") as f:
            cl = json.load(f)
        o2c = cl["original_to_cause_id"]
        State.cluster_id_array = np.array(
            [int(o2c[t]) for t in State.cause_texts], dtype=np.int64,
        )
        print(f"[load] cluster: {len(set(State.cluster_id_array.tolist()))} clusters")
    else:
        print("[warn] cluster json missing; clusters disabled")

    State.train_global_stack = F.normalize(
        torch.stack([c["global_emb"] for c in State.train_cases]).to(DEVICE), dim=-1,
    )
    tls, off = stack_train_lesions(State.train_cases)
    State.train_lesion_stack = F.normalize(tls.to(DEVICE), dim=-1)
    State.train_offsets = off

    print(f"[load] CEAH: {CEAH_CKPT}")
    State.ceah = CEAH(
        global_dim=State.in_dim, text_dim=State.in_dim,
        lesion_dim=State.in_dim, cause_dim=State.in_dim,
        common_dim=256, hidden_dim=512, dropout=0.1,
        attribution_mode="softmax", scoring_mode="multiplicative",
    ).to(DEVICE)
    State.ceah.load_state_dict(torch.load(CEAH_CKPT, map_location=DEVICE))
    State.ceah.eval()

    print(f"[load] symptoms.json: {SYMPTOMS_JSON}")
    _load_symptoms_bank()
    print(f"[load] symptoms encoded: {State.sym_text_feats.size(0)} captions across "
          f"{len(set(State.sym_label_ids))} categories")
    print("[init] done")


# ---------------------------------------------------------------------------
# Stage 1: detection
# ---------------------------------------------------------------------------

@torch.no_grad()
def detect_lesions(image_pil: Image.Image, score_thresh: float) -> List[Tuple[List[float], float]]:
    """Returns list of (bbox_xywh, score) sorted by score desc."""
    pred = State.rfdetr.predict([image_pil], threshold=score_thresh)
    det = pred[0] if isinstance(pred, list) else pred
    out: List[Tuple[List[float], float]] = []
    for bbox, score in zip(det.xyxy, det.confidence):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        out.append(([x1, y1, x2 - x1, y2 - y1], float(score)))
    out.sort(key=lambda x: -x[1])
    return out


def scaled_rect_crop(img: Image.Image, bbox_xywh, scale: float = 1.0) -> Image.Image:
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = w * scale, h * scale
    W, H = img.size
    x1 = max(0, min(W - 1, int(round(cx - nw / 2.0))))
    y1 = max(0, min(H - 1, int(round(cy - nh / 2.0))))
    x2 = max(1, min(W, int(round(cx + nw / 2.0))))
    y2 = max(1, min(H, int(round(cy + nh / 2.0))))
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    return img.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Stage 2: VLM encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_global_image(image_pil: Image.Image) -> torch.Tensor:
    proc = State.vlm_global_proc
    px = proc(images=[image_pil], return_tensors="pt")["pixel_values"].to(DEVICE)
    with torch.cuda.amp.autocast(enabled=DEVICE.startswith("cuda")):
        f = get_image_features(State.vlm_global, px)
    return F.normalize(f.float(), dim=-1)[0]


@torch.no_grad()
def encode_lesion_fusion_batch(local_pils: List[Image.Image], global_pil: Image.Image) -> torch.Tensor:
    if not local_pils:
        return torch.empty(0, State.in_dim, device=DEVICE)
    proc = State.vlm_lesion_proc
    lpx = proc(images=local_pils, return_tensors="pt")["pixel_values"].to(DEVICE)
    gpx = proc(images=[global_pil] * len(local_pils),
               return_tensors="pt")["pixel_values"].to(DEVICE)
    with torch.cuda.amp.autocast(enabled=DEVICE.startswith("cuda")):
        f = State.vlm_lesion.forward_image(lpx, gpx)
    return F.normalize(f.float(), dim=-1)


@torch.no_grad()
def encode_text_global(text: str) -> torch.Tensor:
    proc = State.vlm_global_proc
    inp = proc(text=[text], return_tensors="pt",
               padding="max_length", truncation=True, max_length=64)
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.cuda.amp.autocast(enabled=DEVICE.startswith("cuda")):
        f = get_text_features(State.vlm_global, inp["input_ids"], inp.get("attention_mask"))
    return F.normalize(f.float(), dim=-1)[0]


# ---------------------------------------------------------------------------
# Lesion classification (VLM-Lesion fusion vs symptoms.json bank)
# ---------------------------------------------------------------------------

def classify_lesions(lesion_embs: torch.Tensor, top_k: int = 5) -> List[Dict]:
    """Per lesion: argmax over the sym caption bank → predicted (label_id, label_zh, label_en, score),
    plus the prompt of the top-1 caption (used for heatmap) and the top-K alternative classes.
    """
    if lesion_embs.numel() == 0:
        return []
    sims = lesion_embs @ State.sym_text_feats.T  # [N, M]
    out: List[Dict] = []
    for li in range(sims.size(0)):
        # Per-class best caption: max over captions belonging to that class
        per_class_best: Dict[str, Tuple[float, int]] = {}
        for j in range(sims.size(1)):
            cid = State.sym_label_ids[j]
            sc = float(sims[li, j].item())
            if cid not in per_class_best or sc > per_class_best[cid][0]:
                per_class_best[cid] = (sc, j)
        ranked = sorted(per_class_best.items(), key=lambda x: -x[1][0])
        top1_cid, (top1_sc, top1_capj) = ranked[0]
        out.append({
            "label_id":  top1_cid,
            "label_zh":  State.sym_id_to_zh.get(top1_cid, top1_cid),
            "label_en":  State.sym_id_to_en.get(top1_cid, top1_cid),
            "score":     float(top1_sc),
            "best_caption":     State.sym_raw_texts[top1_capj],
            "best_caption_lang": State.sym_langs[top1_capj],
            "best_prompt":      format_caption(
                State.sym_raw_texts[top1_capj], State.sym_langs[top1_capj]
            ),
            "top_k": [
                {
                    "label_id": cid, "score": float(sc),
                    "label_zh": State.sym_id_to_zh.get(cid, cid),
                    "label_en": State.sym_id_to_en.get(cid, cid),
                }
                for cid, (sc, _) in ranked[:top_k]
            ],
        })
    return out


# ---------------------------------------------------------------------------
# Gradient heatmap (ported / adapted from heatmap.ipynb)
# ---------------------------------------------------------------------------

def _denorm_pixel_values(pixel_values: torch.Tensor, processor) -> np.ndarray:
    img = pixel_values[0].detach().cpu()
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return img.permute(1, 2, 0).numpy()
    mean = torch.tensor(ip.image_mean).view(-1, 1, 1)
    std = torch.tensor(ip.image_std).view(-1, 1, 1)
    img = (img * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def _resize_2d(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    t = torch.from_numpy(arr).float()[None, None, ...]
    return F.interpolate(t, size=(out_h, out_w), mode="bilinear",
                         align_corners=False)[0, 0].numpy()


def _project_local_to_global(heatmap_local: np.ndarray, bbox_xywh,
                              global_size: Tuple[int, int]) -> np.ndarray:
    """Place the local heatmap (sized to the local crop) onto a zero canvas at bbox region."""
    GW, GH = global_size
    canvas = np.zeros((GH, GW), dtype=np.float32)
    if bbox_xywh is None:
        return _resize_2d(heatmap_local, GH, GW)
    x, y, w, h = bbox_xywh
    x1 = int(max(0, min(GW - 1, np.floor(x))))
    y1 = int(max(0, min(GH - 1, np.floor(y))))
    x2 = int(max(1, min(GW, np.ceil(x + w))))
    y2 = int(max(1, min(GH, np.ceil(y + h))))
    if x2 <= x1: x2 = min(GW, x1 + 1)
    if y2 <= y1: y2 = min(GH, y1 + 1)
    canvas[y1:y2, x1:x2] = _resize_2d(heatmap_local, y2 - y1, x2 - x1)
    return canvas


def gradient_heatmap_global(local_pil: Image.Image, global_pil: Image.Image,
                             bbox_xywh, prompt_text: str) -> np.ndarray:
    """Returns a 2D heatmap (H_global, W_global) of fusion-image vs text relevance."""
    model = State.vlm_lesion       # fusion wrapper
    target = model.base_model
    proc = State.vlm_lesion_proc

    proc_kwargs = {"return_tensors": "pt", "padding": "max_length", "do_center_crop": False}
    local_inputs = proc(text=[prompt_text], images=local_pil, **proc_kwargs)
    local_inputs = {k: v.to(DEVICE) for k, v in local_inputs.items()}
    global_inputs = proc(images=global_pil, return_tensors="pt", do_center_crop=False)
    global_inputs = {k: v.to(DEVICE) for k, v in global_inputs.items()}

    target.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)

    fwd_kwargs = {"return_dict": True, "output_hidden_states": True}
    out = target(**local_inputs, **fwd_kwargs)

    if out.vision_model_output is None:
        raise RuntimeError("vision_model_output is None")
    last = out.vision_model_output.last_hidden_state
    if last is None:
        raise RuntimeError("last_hidden_state is None")
    token_source = last  # SigLIP/SigLIP2: no CLS, all patches

    # Image feats (raw) for fused score
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        local_feat_raw = out.image_embeds
    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
        local_feat_raw = out.pooler_output
    else:
        local_feat_raw = last[:, 0]

    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        text_feat_raw = out.text_embeds
    elif hasattr(out, "text_model_output") and out.text_model_output is not None:
        tmo = out.text_model_output
        if hasattr(tmo, "pooler_output") and tmo.pooler_output is not None:
            text_feat_raw = tmo.pooler_output
        else:
            text_feat_raw = tmo.last_hidden_state[:, 0]
    else:
        # fall back: re-encode the text via the wrapper
        text_feat_raw = model.get_text_features(local_inputs["input_ids"],
                                                local_inputs.get("attention_mask"))

    global_feat_raw = get_image_features(target, global_inputs["pixel_values"])

    local_n = F.normalize(local_feat_raw, dim=-1)
    global_n = F.normalize(global_feat_raw, dim=-1)
    fused = torch.cat([local_n, global_n], dim=-1)
    fused = model.fusion_linear(fused)
    fused = model.gelu(fused)
    fused = model.dropout(fused)
    image_feat = F.normalize(local_n + model.gate * fused, dim=-1)
    text_feat = F.normalize(text_feat_raw, dim=-1)
    score = (image_feat * text_feat).sum()

    grads = torch.autograd.grad(score, token_source, retain_graph=False)[0]

    # Determine grid
    spatial = local_inputs.get("spatial_shapes", None)
    if spatial is not None:
        grid_h = int(spatial[0, 0].item())
        grid_w = int(spatial[0, 1].item())
        valid = grid_h * grid_w
        patch_tokens = token_source[:, :valid, :]
        patch_grads = grads[:, :valid, :]
    else:
        patch_size = target.config.vision_config.patch_size
        ph, pw = local_inputs["pixel_values"].shape[-2:]
        grid_h = ph // patch_size
        grid_w = pw // patch_size
        patch_tokens = token_source
        patch_grads = grads

    relevance = (patch_grads * patch_tokens).sum(dim=-1)[0].clamp(min=0)
    if relevance.numel() != grid_h * grid_w:
        # if model gives [1, T, D] with T == grid; fallback to truncate
        relevance = relevance[: grid_h * grid_w]

    token_map = relevance.reshape(grid_h, grid_w).detach()
    if (token_map.max() - token_map.min()).item() < 1e-12:
        token_map = torch.zeros_like(token_map)
    else:
        token_map = (token_map - token_map.min()) / (token_map.max() + 1e-8)

    local_w, local_h = local_pil.size
    heatmap_local = F.interpolate(
        token_map[None, None, ...].float(), size=(local_h, local_w),
        mode="bilinear", align_corners=False,
    )[0, 0].detach().cpu().numpy()

    return _project_local_to_global(heatmap_local, bbox_xywh, global_pil.size)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _put_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str,
               color=(255, 255, 255), bg=(40, 40, 40), size: int = 16):
    try:
        font = ImageFont.truetype(CJK_FONT_PATH, size)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle(
        (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
        fill=bg,
    )
    draw.text((x, y), text, font=font, fill=color)


def make_missing_case_placeholder(rank: int, sim: float) -> Image.Image:
    """Placeholder card for retrieved train cases whose image file is missing."""
    W, H = 320, 220
    img = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype(CJK_FONT_PATH, 22)
        body_font = ImageFont.truetype(CJK_FONT_PATH, 17)
        small_font = ImageFont.truetype(CJK_FONT_PATH, 15)
    except OSError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # outer border
    draw.rectangle((0, 0, W - 1, H - 1), outline=(180, 180, 180), width=2)

    # simple icon box
    draw.rectangle((24, 28, 92, 96), outline=(160, 160, 160), width=2)
    draw.line((32, 84, 50, 64, 66, 78, 84, 54), fill=(160, 160, 160), width=2)
    draw.ellipse((68, 42, 78, 52), fill=(160, 160, 160))

    draw.text((112, 34), f"Train case #{rank}", fill=(50, 50, 50), font=title_font)
    draw.text((112, 72), "圖片不存在", fill=(150, 45, 45), font=body_font)

    draw.line((24, 122, W - 24, 122), fill=(210, 210, 210), width=1)

    draw.text((24, 145), f"similarity = {sim:.3f}", fill=(70, 70, 70), font=body_font)
    draw.text((24, 178), "placeholder shown", fill=(120, 120, 120), font=small_font)

    return img


def render_detection_image(image_pil: Image.Image,
                            lesions: List[Dict]) -> Image.Image:
    img = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for li, les in enumerate(lesions):
        x, y, w, h = les["bbox_xywh"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        color = (220, 50, 50)
        for off in range(3):
            draw.rectangle((x - off, y - off, x + w + off, y + h + off), outline=color)
        label = f"L{li}: {les['cls']['label_zh']}  ({les['det_score']:.2f})"
        _put_label(draw, x + 2, max(2, y - 22), label, bg=(180, 30, 30))
    return img


def overlay_heatmap_on_image(image_pil: Image.Image,
                              heatmap_global: np.ndarray,
                              alpha: float = 0.45,
                              threshold: float = 0.15) -> Image.Image:
    """Jet-colored heatmap with alpha-mask above threshold over the global image."""
    base = np.array(image_pil.convert("RGB"), dtype=np.float32) / 255.0
    h, w = base.shape[:2]
    if heatmap_global.shape != (h, w):
        heatmap_global = _resize_2d(heatmap_global, h, w)
    cmap = matplotlib.colormaps.get_cmap("jet")
    rgba = cmap(np.clip(heatmap_global, 0, 1))[..., :3]
    mask = (heatmap_global >= threshold).astype(np.float32) * alpha
    out = base * (1.0 - mask[..., None]) + rgba * mask[..., None]
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def make_lesion_card(image_pil: Image.Image, les: Dict, idx: int) -> Image.Image:
    """Compose: [global heatmap | local crop | top-K classes text panel]."""
    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 0.6, 0.9], wspace=0.04)

    overlay = overlay_heatmap_on_image(image_pil, les["heatmap"], alpha=0.5, threshold=0.15)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(overlay)
    x, y, w, h = les["bbox_xywh"]
    rect = mpatches.Rectangle((x, y), w, h, linewidth=2.4, edgecolor="lime", facecolor="none")
    ax.add_patch(rect)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"L{idx}  {les['cls']['label_zh']}  (det={les['det_score']:.2f}, "
                 f"cls={les['cls']['score']:.2f})",
                 fontproperties=_font(13), pad=6)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(les["crop"])
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("crop", fontproperties=_font(11), pad=4)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    ax3.text(0.0, 0.98, "Top-K class similarity",
             fontproperties=_font(11), transform=ax3.transAxes, fontweight="bold")
    yp = 0.90
    for k, item in enumerate(les["cls"]["top_k"]):
        line = f"{k+1}. {item['label_zh']:<10}  {item['score']:.3f}"
        ax3.text(0.0, yp, line, fontproperties=_font(10), transform=ax3.transAxes,
                 family="monospace")
        yp -= 0.07
    yp -= 0.02
    ax3.text(0.0, yp, "Caption used for heatmap:",
             fontproperties=_font(10), transform=ax3.transAxes, fontweight="bold")
    yp -= 0.07
    cap = les["cls"]["best_caption"]
    for chunk in [cap[i:i+34] for i in range(0, len(cap), 34)] or [""]:
        ax3.text(0.0, yp, chunk, fontproperties=_font(9), transform=ax3.transAxes)
        yp -= 0.06

    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba[..., :3])


def alpha_to_rgba(a: float, base, a_max: float):
    intensity = float(np.clip(a / max(a_max, 1e-6), 0.0, 1.0))
    return (*base, 0.3 + 0.7 * intensity)


def make_alpha_attribution_image(image_pil: Image.Image, lesion_boxes: np.ndarray,
                                  alpha: List[float], n_lesions: int,
                                  cause_text: str, score: float) -> Image.Image:
    g_a = float(alpha[0])
    t_a = float(alpha[1])
    les_a = [float(a) for a in alpha[2: 2 + n_lesions]]
    a_max = max([g_a, t_a] + les_a + [1e-6])

    fig = plt.figure(figsize=(11, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image_pil)
    ax.set_xticks([]); ax.set_yticks([])
    title = f"α attribution  (cause score={score:.3f})"
    ax.set_title(title, fontproperties=_font(13), pad=8)

    for li, (bbox, a_val) in enumerate(zip(lesion_boxes, les_a)):
        x, y, w, h = [int(v) for v in bbox]
        intensity = float(np.clip(a_val / a_max, 0.0, 1.0))
        lw = 1.0 + 4.5 * intensity
        edge = alpha_to_rgba(a_val, (1.0, 0.15, 0.15), a_max)
        rect = mpatches.Rectangle((x, y), w, h, linewidth=lw,
                                  edgecolor=edge, facecolor="none")
        ax.add_patch(rect)
        ax.text(x + 3, max(y - 6, 12), f"L{li}\nα={a_val:.2f}",
                fontproperties=_font(11), color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=edge[:3], alpha=0.9, edgecolor="none"))

    g_color = alpha_to_rgba(g_a, (0.15, 0.4, 1.0), a_max)
    t_color = alpha_to_rgba(t_a, (1.0, 0.55, 0.0), a_max)
    ax.text(8, 22, f"GLOBAL  α={g_a:.2f}",
            fontproperties=_font(11), color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=g_color[:3], alpha=0.95, edgecolor="none"))
    ax.text(8, 50, f"TEXT  α={t_a:.2f}",
            fontproperties=_font(11), color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=t_color[:3], alpha=0.95, edgecolor="none"))

    fig.text(0.5, 0.02, cause_text, ha="center", va="bottom",
             fontproperties=_font(12), wrap=True)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba[..., :3])


def make_alpha_breakdown_chart(alpha: List[float], n_lesions: int) -> Image.Image:
    labels = ["global", "text"] + [f"L{i}" for i in range(n_lesions)]
    vals = [float(a) for a in alpha[: 2 + n_lesions]]
    colors = ["#2660ff", "#ff8c00"] + ["#dc1e1e"] * n_lesions

    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, max(0.6, max(vals) * 1.2))
    ax.set_ylabel("α", fontproperties=_font(11))
    ax.set_title("Per-evidence attribution (softmax α, sums to 1)",
                 fontproperties=_font(11))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontproperties=_font(10))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return Image.fromarray(rgba[..., :3])


# ---------------------------------------------------------------------------
# Stage 3: Phase 1 + CEAH
# ---------------------------------------------------------------------------

def _minmax(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


@torch.no_grad()
def retrieve_and_score(q: Dict, gamma: float, top_k_cases: int, top_n_causes: int,
                        diversify_threshold: float = 0.95) -> Dict:
    # Encoders normalize before returning, but mirror phase1_baseline.py exactly.
    q_global = F.normalize(q["global_emb"], dim=-1)
    q_lesions = F.normalize(q["lesion_embs"], dim=-1) if q["lesion_embs"].numel() \
                else q["lesion_embs"]

    sims = compute_case_similarities(
        q_global, q_lesions,
        State.train_global_stack, State.train_lesion_stack, State.train_offsets,
        alpha=0.25, beta=0.75, lesion_match="hungarian",
    )
    # Phase 1 contract: drop train cases with similarity <= 0; weights normalized to sum to 1.
    top_idx, top_w, top_raw_w = select_positive_top_cases(sims, top_k_cases)
    candidate_indices = build_candidate_pool(top_idx, State.train_cases)
    if not candidate_indices:
        return {"top_n": [], "retrieved_cases": [], "pool_size": 0}

    cand_idx_t = torch.tensor(candidate_indices, device=DEVICE, dtype=torch.long)
    cand_embs = State.cause_table_embs.index_select(0, cand_idx_t)

    s1 = score_candidates(candidate_indices, top_idx, top_w,
                          State.train_cases, State.cause_table_embs)

    P = len(candidate_indices)
    n_les = q["lesion_embs"].size(0)
    # CEAH internally calls .view(B*N, -1) on lesion_embs, which fails on non-contiguous
    # expanded tensors when N > 1. Materialize contiguous copies.
    g_emb = q["global_emb"].unsqueeze(0).expand(P, -1).contiguous()
    l_emb = q["lesion_embs"].unsqueeze(0).expand(P, -1, -1).contiguous()
    l_mask = torch.ones(P, n_les, dtype=torch.bool, device=DEVICE)

    if q.get("text_emb") is None:
        t_emb = torch.zeros(P, State.in_dim, device=DEVICE)
        t_present = torch.zeros(P, dtype=torch.bool, device=DEVICE)
    else:
        t_emb = q["text_emb"].unsqueeze(0).expand(P, -1).contiguous()
        t_present = torch.ones(P, dtype=torch.bool, device=DEVICE)

    s_ceah, alphas, _ = State.ceah(g_emb, t_emb, t_present, l_emb, l_mask, cand_embs)

    s1_n = _minmax(s1)
    sc_n = _minmax(s_ceah)
    hybrid = gamma * s1_n + (1.0 - gamma) * sc_n
    sorted_local = torch.argsort(hybrid, descending=True).cpu().numpy()
    sorted_local = diversify(sorted_local, cand_embs, diversify_threshold)
    sorted_global = np.array(candidate_indices)[sorted_local]

    top_n_count = min(top_n_causes, len(sorted_local))
    out_top = []
    for r in range(top_n_count):
        li = int(sorted_local[r])
        gi = int(sorted_global[r])
        out_top.append({
            "rank":  r + 1,
            "cause_idx":  gi,
            "text":       State.cause_texts[gi],
            "score":      float(hybrid[li].item()),
            "score_p1":   float(s1_n[li].item()),
            "score_ceah": float(sc_n[li].item()),
            "alpha":      [float(a) for a in alphas[li].cpu().tolist()],
        })

    retrieved = []
    for k in range(min(len(top_idx), 5)):
        ci = int(top_idx[k])
        case = State.train_cases[ci]
        img_path = TRAIN_IMG_ROOT / case["file_name"]
        retrieved.append({
            "case_id":   ci,
            "image_id":  int(case["image_id"]),
            "file_name": case["file_name"],
            "image_path": str(img_path) if img_path.exists() else None,
            "similarity": float(top_raw_w[k]),
            "causes":    list(case["causes"]),
        })

    return {"top_n": out_top, "retrieved_cases": retrieved, "pool_size": P}


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def build_query_from_image(image_pil: Image.Image, det_thresh: float,
                            text_desc: str) -> Dict:
    image_pil = image_pil.convert("RGB")
    detections = detect_lesions(image_pil, det_thresh)
    bboxes = [d[0] for d in detections]
    det_scores = [d[1] for d in detections]

    g_emb = encode_global_image(image_pil)

    crops = [scaled_rect_crop(image_pil, b) for b in bboxes]
    l_emb = encode_lesion_fusion_batch(crops, image_pil) if crops \
            else torch.empty(0, State.in_dim, device=DEVICE)

    text_desc = text_desc or ""
    # Text evidence is enabled automatically when the textbox is non-empty.
    # Empty text => vision-only CEAH inference (t_present=False in retrieve_and_score).
    t_emb = encode_text_global(text_desc) if text_desc.strip() else None

    return {
        "image_pil":           image_pil,
        "bboxes_xywh":         bboxes,
        "det_scores":          det_scores,
        "crops":               crops,
        "global_emb":          g_emb,
        "lesion_embs":         l_emb,
        "text_emb":            t_emb,
    }


def run_full_pipeline(image_pil: Image.Image, text_desc: str,
                      det_thresh: float, gamma: float, top_k_cases: int,
                      top_n_causes: int) -> Tuple[Dict, List[Dict], Dict]:
    q = build_query_from_image(image_pil, det_thresh, text_desc)

    if len(q["bboxes_xywh"]) == 0:
        return q, [], {
            "top_n": [],
            "retrieved_cases": [],
            "pool_size": 0,
        }

    cls = classify_lesions(q["lesion_embs"])

    lesions: List[Dict] = []
    for i, (b, sc, c, cl) in enumerate(zip(q["bboxes_xywh"], q["det_scores"], q["crops"], cls)):
        heat = gradient_heatmap_global(c, q["image_pil"], b, cl["best_prompt"])
        lesions.append({
            "idx":       i,
            "bbox_xywh": b,
            "det_score": sc,
            "crop":      c,
            "cls":       cl,
            "heatmap":   heat,
        })

    cause_results = retrieve_and_score(q, gamma, top_k_cases, top_n_causes)
    return q, lesions, cause_results


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

def _empty_button_updates():
    return [gr.update(visible=False) for _ in range(MAX_TOPN_BUTTONS)]


def handler_run(image, text_desc, det_thresh, gamma,
                top_k_cases, top_n_causes):
    if image is None:
        return (
            None, None, None,
            "請先上傳或從範例選一張圖。", [], None, None, "",
            *_empty_button_updates(),
        )
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    q, lesions, cause_results = run_full_pipeline(
        image, text_desc or "",
        float(det_thresh), float(gamma),
        int(top_k_cases), int(top_n_causes),
    )

    det_img = render_detection_image(q["image_pil"], lesions)
    gallery = [(make_lesion_card(q["image_pil"], l, l["idx"]),
                f"L{l['idx']}: {l['cls']['label_zh']}") for l in lesions]

    if not lesions:
        info_md = "**RF-DETR 在此閾值下未偵測到病灶**，請降低 detection threshold。"
    else:
        info_md = (f"**偵測到 {len(lesions)} 個病灶。**  "
                   f"候選病因池 = **{cause_results['pool_size']}**  ｜  "
                   f"列出 top-{len(cause_results['top_n'])} 病因（點擊查看 α attribution）")

    btn_updates = []
    for i in range(MAX_TOPN_BUTTONS):
        if i < len(cause_results["top_n"]):
            r = cause_results["top_n"][i]
            txt = r["text"]
            if len(txt) > 50:
                txt = txt[:48] + "…"
            label = f"#{r['rank']}  s={r['score']:.2f}  {txt}"
            btn_updates.append(gr.update(value=label, visible=True))
        else:
            btn_updates.append(gr.update(visible=False))

    retrieved_gallery = []

    for i, r in enumerate(cause_results["retrieved_cases"], start=1):
        sim = float(r["similarity"])
        caption = f"#{i}  sim={sim:.3f}"

        if r.get("image_path"):
            retrieved_gallery.append((r["image_path"], caption))
        else:
            placeholder = make_missing_case_placeholder(i, sim)
            retrieved_gallery.append((placeholder, f"{caption}  image not found"))

    state = {
        "image_pil":   q["image_pil"],
        "bboxes_xywh": q["bboxes_xywh"],
        "n_lesions":   len(lesions),
        "top_n":       cause_results["top_n"],
    }

    return (
        det_img, gallery, state,
        info_md, retrieved_gallery, None, None, "",
        *btn_updates,
    )


def handler_select_cause(idx: int, state: Optional[Dict]):
    if state is None or idx >= len(state.get("top_n", [])):
        return None, None, ""
    r = state["top_n"][idx]
    bbox_arr = np.array(state["bboxes_xywh"], dtype=np.float32)
    n = state["n_lesions"]
    alpha_img = make_alpha_attribution_image(
        state["image_pil"], bbox_arr, r["alpha"], n, r["text"], r["score"],
    )
    bar_img = make_alpha_breakdown_chart(r["alpha"], n)
    explain = (
        f"### Top-{r['rank']} 病因\n"
        f"**{r['text']}**\n\n"
        f"- Hybrid score: **{r['score']:.3f}**  "
        f"(Phase 1 = {r['score_p1']:.3f}, CEAH = {r['score_ceah']:.3f})\n"
        f"- α 加總 = 1（softmax）；數字越大代表該證據對此病因的貢獻越強。"
    )
    return alpha_img, bar_img, explain


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

EXAMPLE_FILES = [
    "fish_disease_aug_yflip_230_jpg.rf.210438e360b75ba135eff7d1a4f7ee4f.jpg",
    "fish_disease_aug_origin_397_jpg.rf.51636e3869daa468901406977de45056.jpg",
    "fish_disease_aug_xflip_372_jpg.rf.9cb2fa2b9727cc0ab993501eb3c116d8.jpg",
    "fish_disease_aug_box_yflip_359_jpg.rf.59e73c66a63578aac1f58aa75073c679.jpg",
    "fish_disease_aug_yflip_141_jpg.rf.073a9d717cadcb4e548de3bf151c694b.jpg",
]


def get_example_paths() -> List[List]:
    out = []
    for fn in EXAMPLE_FILES:
        p = VALID_IMG_ROOT / fn
        if p.exists():
            # image, text_desc, det_thresh, gamma, top_k_cases, top_n_causes
            out.append([str(p), "", 0.35, 0.75, 20, 5])
    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# 🐟 FaCE-R 魚病診斷流水線 demo

完整 pipeline：
**RF-DETR 偵測病灶** → **VLM-Lesion 分類 + 熱力圖** → **VLM-Global 整圖編碼** →
**Phase 1 案例檢索** → **CEAH 病因歸因 (α)**

上傳一張魚體圖，或從下方範例選一張。系統會：
1. 偵測病灶 bbox 並用 VLM-Lesion 分類成 symptom 類別 + 產生 grad-based 熱力圖
2. 從 12,780 個訓練 case 找 top-K 相似 → 形成候選病因池
3. CEAH 對每個候選給分數 + 對 (global / text / 每個 lesion) 的 softmax α
4. Hybrid γ 排序得到 top-N 病因，**點擊任一病因按鈕**查看 α 解釋
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="FaCE-R Fish Disease Pipeline") as demo:
        gr.Markdown(DESCRIPTION)
        state = gr.State()

        with gr.Row():
            with gr.Column(scale=1):
                inp_image = gr.Image(label="魚體輸入圖", type="pil", height=320)
                inp_text_desc = gr.Textbox(
                    label="選填：文字描述",
                    lines=3,
                    placeholder="例：魚體表有潰瘍、紅腫，疑似感染；留空則以 vision-only 模式推論",
                )
                with gr.Accordion("可調參數", open=False):
                    sld_det = gr.Slider(0.1, 0.9, value=0.5, step=0.05,
                                         label="Detection threshold")
                    sld_gamma = gr.Slider(0.0, 1.0, value=0.75, step=0.05,
                                           label="Hybrid γ  (1.0=Phase1 only, 0.0=CEAH only)")
                    sld_topk = gr.Slider(5, 50, value=20, step=1, label="top_k_cases (K)")
                    sld_topn = gr.Slider(1, MAX_TOPN_BUTTONS, value=5, step=1, label="top_n_causes (N)")
                btn_run = gr.Button("Run pipeline", variant="primary")
            with gr.Column(scale=2):
                out_det = gr.Image(label="① RF-DETR detection", type="pil", height=320)
                out_info = gr.Markdown()

        gr.Markdown("---\n## ② VLM-Lesion 分類 + 熱力圖")
        out_gallery = gr.Gallery(
            label="每個偵測到的病灶（含 grad-based heatmap）",
            columns=1, height=580, show_label=False, object_fit="contain",
        )

        gr.Markdown("---\n## ③ Top-N 病因（點擊查看歸因 α）")
        cause_buttons: List[gr.Button] = []
        with gr.Row():
            with gr.Column(scale=1):
                for i in range(MAX_TOPN_BUTTONS):
                    btn = gr.Button(value="", visible=False, size="sm")
                    cause_buttons.append(btn)

                out_retrieved = gr.Markdown()
                out_retrieved_gallery = gr.Gallery(
                    label="最相似的 train cases",
                    columns=5,
                    height=220,
                    show_label=True,
                    object_fit="contain",
                )
            with gr.Column(scale=2):
                out_explain = gr.Markdown()
                out_alpha_img = gr.Image(label="α attribution overlay", type="pil", height=420)
                out_alpha_bar = gr.Image(label="α breakdown", type="pil", height=240)

        run_outputs = [
            out_det, out_gallery, state,
            out_info, out_retrieved_gallery,
            out_alpha_img, out_alpha_bar, out_explain,
            *cause_buttons,
        ]
        btn_run.click(
            fn=handler_run,
            inputs=[inp_image, inp_text_desc, sld_det, sld_gamma,
                    sld_topk, sld_topn],
            outputs=run_outputs,
        )

        for i, btn in enumerate(cause_buttons):
            btn.click(
                fn=lambda st, idx=i: handler_select_cause(idx, st),
                inputs=[state],
                outputs=[out_alpha_img, out_alpha_bar, out_explain],
            )

        ex = get_example_paths()
        if ex:
            gr.Examples(
                examples=ex,
                inputs=[inp_image, inp_text_desc, sld_det, sld_gamma,
                        sld_topk, sld_topn],
                label="範例（valid set，論文 case study 用過的圖）",
                examples_per_page=10,
            )

    return demo


def main():
    load_all()
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
