from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import classification_report, confusion_matrix

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def load_label_bank(label_bank_path: str | Path) -> Tuple[Dict[str, List[str]], Dict[str, dict]]:
    payload = load_json(label_bank_path)
    if "data" not in payload or not isinstance(payload["data"], dict):
        raise ValueError("label_bank json 缺少 data 欄位")

    label_map = payload.get("label_map", {})
    captions_by_cat: Dict[str, List[str]] = {}
    for key, value in payload["data"].items():
        if not isinstance(value, dict):
            continue
        caps = value.get("captions_en", [])
        if not isinstance(caps, list) or len(caps) == 0:
            raise ValueError(f"label_bank 缺少 captions_en 或為空: category_id={key}")
        captions_by_cat[str(key)] = [str(x) for x in caps]
    if not captions_by_cat:
        raise ValueError("label_bank 內沒有可用的 captions_en")
    return captions_by_cat, label_map



def build_caption_bank(
    captions_by_cat: Dict[str, List[str]]
) -> Tuple[List[str], List[int], Dict[int, List[int]]]:
    bank_texts: List[str] = []
    bank_labels: List[int] = []
    cat2bank_indices: Dict[int, List[int]] = {}

    keys_sorted = sorted(captions_by_cat.keys(), key=lambda x: int(x))
    for key in keys_sorted:
        label_id = int(key)
        idxs: List[int] = []
        for cap in captions_by_cat[key]:
            idxs.append(len(bank_texts))
            bank_texts.append(cap)
            bank_labels.append(label_id)
        cat2bank_indices[label_id] = idxs
    return bank_texts, bank_labels, cat2bank_indices



def load_eval_texts(label_bank_path: str | Path) -> Tuple[List[str], List[str], Dict[str, str], List[str]]:
    payload = load_json(label_bank_path)
    label_map = payload.get("label_map", {})
    data = payload.get("data", {})

    id_to_zh: Dict[str, str] = {}
    for key, info in label_map.items():
        if isinstance(info, dict):
            id_to_zh[str(key)] = str(info.get("zh", key))
        else:
            id_to_zh[str(key)] = str(key)

    flat_texts: List[str] = []
    flat_label_ids: List[str] = []
    for key, content in data.items():
        label_id = str(key)
        captions = content.get("captions_en", []) if isinstance(content, dict) else []
        for cap in captions:
            flat_texts.append(f"This is {str(cap).lower()}.")
            flat_label_ids.append(label_id)

    all_label_ids = sorted(set(flat_label_ids), key=lambda x: int(x))
    return flat_texts, flat_label_ids, id_to_zh, all_label_ids



def clamp_bbox_xyxy(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(width - 1, math.floor(x1))))
    y1 = int(max(0, min(height - 1, math.floor(y1))))
    x2 = int(max(1, min(width, math.ceil(x2))))
    y2 = int(max(1, min(height, math.ceil(y2))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2



def crop_bbox(image: Image.Image, bbox_xywh: Tuple[float, float, float, float]) -> Image.Image:
    x, y, w, h = bbox_xywh
    width, height = image.size
    x1, y1, x2, y2 = clamp_bbox_xyxy(x, y, x + w, y + h, width, height)
    return image.crop((x1, y1, x2, y2))



def crop_square_with_black_padding(image: Image.Image, bbox_xywh: Tuple[float, float, float, float]) -> Image.Image:
    x, y, w, h = bbox_xywh
    width, height = image.size

    side = float(max(w, h))
    if side <= 1:
        side = 1.0

    cx = x + w / 2.0
    cy = y + h / 2.0
    x1 = cx - side / 2.0
    y1 = cy - side / 2.0
    x2 = x1 + side
    y2 = y1 + side

    out_side = int(max(1, round(side)))
    out = Image.new("RGB", (out_side, out_side), (0, 0, 0))
    src_x1 = int(max(0, math.floor(x1)))
    src_y1 = int(max(0, math.floor(y1)))
    src_x2 = int(min(width, math.ceil(x2)))
    src_y2 = int(min(height, math.ceil(y2)))

    patch = image.crop((src_x1, src_y1, src_x2, src_y2))
    dst_x = int(max(0, -math.floor(x1)))
    dst_y = int(max(0, -math.floor(y1)))
    out.paste(patch, (dst_x, dst_y))
    return out



def _pool_from_last_hidden(last_hidden: torch.Tensor) -> torch.Tensor:
    return last_hidden[:, 0]



def get_image_features(model, pixel_values: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "get_image_features"):
        return model.get_image_features(pixel_values=pixel_values)

    out = model(pixel_values=pixel_values, return_dict=True)
    if getattr(out, "image_embeds", None) is not None:
        return out.image_embeds
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    if getattr(out, "last_hidden_state", None) is not None:
        return _pool_from_last_hidden(out.last_hidden_state)

    if hasattr(model, "vision_model"):
        vout = model.vision_model(pixel_values=pixel_values, return_dict=True)
        if getattr(vout, "pooler_output", None) is not None:
            return vout.pooler_output
        if getattr(vout, "last_hidden_state", None) is not None:
            return _pool_from_last_hidden(vout.last_hidden_state)

    raise RuntimeError("Cannot extract image features from this model.")



def get_text_features(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if hasattr(model, "get_text_features"):
        return model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

    out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    if getattr(out, "text_embeds", None) is not None:
        return out.text_embeds
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    if getattr(out, "last_hidden_state", None) is not None:
        return _pool_from_last_hidden(out.last_hidden_state)

    if hasattr(model, "text_model"):
        tout = model.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if getattr(tout, "pooler_output", None) is not None:
            return tout.pooler_output
        if getattr(tout, "last_hidden_state", None) is not None:
            return _pool_from_last_hidden(tout.last_hidden_state)

    raise RuntimeError("Cannot extract text features from this model.")



def get_logit_scale_and_bias(model, device: str, default_temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
    target_model = model.base_model if hasattr(model, "base_model") else model
    scale = None
    bias = None

    if hasattr(target_model, "logit_scale"):
        ls = getattr(target_model, "logit_scale")
        scale = ls.to(device).exp() if isinstance(ls, torch.Tensor) else torch.tensor(float(ls), device=device).exp()

    if hasattr(target_model, "logit_bias"):
        lb = getattr(target_model, "logit_bias")
        bias = lb.to(device) if isinstance(lb, torch.Tensor) else torch.tensor(float(lb), device=device)

    if scale is None:
        scale = torch.tensor(1.0 / float(default_temperature), device=device)
    if bias is None:
        bias = torch.tensor(-10.0, device=device)

    scale = torch.clamp(scale, max=100.0)
    return scale, bias



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

    def forward_image(
        self,
        pixel_values_local: torch.Tensor,
        pixel_values_global: torch.Tensor,
        return_parts: bool = False,
    ):
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

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return get_text_features(self.base_model, input_ids, attention_mask)



def _find_wrapper_state_path(name_or_path: str | Path) -> Optional[str]:
    candidate = os.path.join(str(name_or_path), "wrapper_state.pt")
    return candidate if os.path.exists(candidate) else None



def load_model_and_processor(name_or_path: str, device: str, force_fusion: bool = False):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(name_or_path)
    base_model = AutoModel.from_pretrained(name_or_path).to(device)

    wrapper_state_path = _find_wrapper_state_path(name_or_path)
    use_wrapper = bool(force_fusion or wrapper_state_path is not None)

    if use_wrapper:
        dummy_image = Image.new("RGB", (224, 224), (0, 0, 0))
        dummy_pixel = processor(images=[dummy_image], return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            hidden_size = get_image_features(base_model, dummy_pixel).shape[-1]

        model = LocalGlobalFusionWrapper(base_model, hidden_size=hidden_size).to(device)
        if wrapper_state_path is None:
            raise FileNotFoundError(
                f"--fusion 已啟用，但在 {name_or_path} 找不到 wrapper_state.pt"
            )

        state = torch.load(wrapper_state_path, map_location=device)
        model.load_state_dict(state)
        model.is_wrapper = True
        model.wrapper_state_path = wrapper_state_path
    else:
        model = base_model
        model.is_wrapper = False
        model.wrapper_state_path = None

    model.eval()
    return model, processor



def compute_symmetric_multipos_sigmoid_loss(
    logits: torch.Tensor,
    img_labels: torch.Tensor,
    bank_labels: torch.Tensor,
    evidence_bank_idx: torch.Tensor,
    evidence_alpha: float,
    t2i_lambda: float,
    t2i_skip_no_positive: bool,
    t2i_label_level_mean: bool,
    label_smoothing: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, bank_size = logits.shape
    device = logits.device
    dtype = logits.dtype

    pos_mask = bank_labels.view(1, bank_size) == img_labels.view(batch_size, 1)
    targets = pos_mask.to(dtype) * (1.0 - float(label_smoothing)) + 0.5 * float(label_smoothing)

    weight = torch.ones((batch_size, bank_size), device=device, dtype=dtype)
    if evidence_alpha is not None and float(evidence_alpha) > 1.0:
        valid = evidence_bank_idx >= 0
        if valid.any():
            rows = torch.arange(batch_size, device=device)[valid]
            cols = torch.clamp(evidence_bank_idx[valid], 0, bank_size - 1)
            ok = pos_mask[rows, cols]
            rows = rows[ok]
            cols = cols[ok]
            weight[rows, cols] = float(evidence_alpha)

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    i2t = (bce * weight).sum(dim=1) / torch.clamp(weight.sum(dim=1), min=eps)
    i2t = i2t.mean()

    probs = torch.sigmoid(logits)
    pos_num_j = (probs * pos_mask.to(dtype)).sum(dim=0)
    pos_den_j = pos_mask.sum(dim=0).to(dtype)
    neg_num_j = ((1.0 - probs) * (~pos_mask).to(dtype)).sum(dim=0)
    neg_den_j = (~pos_mask).sum(dim=0).to(dtype)

    pos_term = -torch.log(torch.clamp(pos_num_j / torch.clamp(pos_den_j, min=1.0), min=eps, max=1.0))
    neg_term = -torch.log(torch.clamp(neg_num_j / torch.clamp(neg_den_j, min=1.0), min=eps, max=1.0))
    t2i_per_caption = 0.5 * (pos_term + neg_term)

    batch_labels = torch.unique(img_labels)
    in_batch = (bank_labels.view(1, bank_size) == batch_labels.view(-1, 1)).any(dim=0)
    if t2i_skip_no_positive:
        keep = in_batch & (pos_den_j > 0)
        t2i_per_caption = t2i_per_caption[keep]
        bank_labels_eff = bank_labels[keep]
    else:
        bank_labels_eff = bank_labels

    if t2i_per_caption.numel() == 0:
        t2i = torch.zeros((), device=device, dtype=dtype)
    else:
        if t2i_label_level_mean:
            per_label_losses: List[torch.Tensor] = []
            for y in batch_labels.tolist():
                mask_y = bank_labels_eff == int(y)
                if mask_y.any():
                    per_label_losses.append(t2i_per_caption[mask_y].mean())
            t2i = torch.stack(per_label_losses, dim=0).mean() if per_label_losses else torch.zeros((), device=device, dtype=dtype)
        else:
            t2i = t2i_per_caption.mean()

    total = i2t + float(t2i_lambda) * t2i
    return total, i2t, t2i



def compute_paired_sigmoid_loss(
    logits: torch.Tensor,
    label_smoothing: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.dim() != 2 or logits.size(0) != logits.size(1):
        raise ValueError(f"paired sigmoid loss expects square logits, got shape={tuple(logits.shape)}")

    batch_size = logits.size(0)
    dtype = logits.dtype
    device = logits.device

    pos_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
    targets = pos_mask.to(dtype) * (1.0 - float(label_smoothing)) + 0.5 * float(label_smoothing)
    neg_mask = ~pos_mask

    element = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    pos_loss_i = (element * pos_mask.to(dtype)).sum(dim=1)
    neg_loss_i = (element * neg_mask.to(dtype)).sum(dim=1) / max(1, batch_size - 1)
    i2t = (0.5 * (pos_loss_i + neg_loss_i)).mean()

    pos_loss_j = (element * pos_mask.to(dtype)).sum(dim=0)
    neg_loss_j = (element * neg_mask.to(dtype)).sum(dim=0) / max(1, batch_size - 1)
    t2i = (0.5 * (pos_loss_j + neg_loss_j)).mean()

    total = 0.5 * (i2t + t2i)
    return total, i2t, t2i


@torch.no_grad()
def encode_text_features(
    model,
    processor,
    texts: List[str],
    device: str,
    text_batch_size: int = 256,
    max_length: int = 64,
    use_amp: bool = True,
) -> torch.Tensor:
    feats = []
    amp_ok = bool(use_amp and str(device).startswith("cuda"))
    for i in range(0, len(texts), text_batch_size):
        batch_texts = texts[i:i + text_batch_size]
        inputs = processor(
            text=batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.cuda.amp.autocast(enabled=amp_ok):
            f = get_text_features(model, inputs["input_ids"], inputs.get("attention_mask", None))
        f = f.float()
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)
        feats.append(f.detach().cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_image_features(
    model,
    processor,
    images_local: List[Image.Image],
    device: str,
    img_batch_size: int = 64,
    use_amp: bool = True,
    images_global: Optional[List[Image.Image]] = None,
) -> torch.Tensor:
    feats = []
    amp_ok = bool(use_amp and str(device).startswith("cuda"))
    is_wrapper = getattr(model, "is_wrapper", False)
    for i in range(0, len(images_local), img_batch_size):
        batch_local = images_local[i:i + img_batch_size]
        inputs_local = processor(images=batch_local, return_tensors="pt")
        inputs_local = {k: v.to(device) for k, v in inputs_local.items()}

        if is_wrapper:
            if images_global is None:
                raise ValueError("Fusion model 評估時必須提供 images_global")
            batch_global = images_global[i:i + img_batch_size]
            inputs_global = processor(images=batch_global, return_tensors="pt")
            inputs_global = {k: v.to(device) for k, v in inputs_global.items()}

        with torch.cuda.amp.autocast(enabled=amp_ok):
            if is_wrapper:
                f = model.forward_image(inputs_local["pixel_values"], inputs_global["pixel_values"])
            else:
                f = get_image_features(model, inputs_local["pixel_values"])
        f = f.float()
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)
        feats.append(f.detach().cpu())
    return torch.cat(feats, dim=0)



def plot_loss_curve(train_losses: List[float], valid_losses: List[float], output_path: str | Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def save_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
    normalize: Optional[str] = None,
) -> None:
    labels = sorted(list(set(labels) | set(y_true) | set(y_pred)), key=lambda x: int(x))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def save_classification_report(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    output_path: str | Path,
) -> None:
    labels = sorted(list(set(labels) | set(y_true) | set(y_pred)), key=lambda x: int(x))
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


DEFAULT_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"



def draw_prediction_visualization(
    image: Image.Image,
    records: List[dict],
    model_tags: List[str],
    id_to_zh: Dict[str, str],
    font_path: str = DEFAULT_FONT_PATH,
) -> Image.Image:
    vis = image.copy().convert("RGB")
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype(font_path, 18)
    except OSError:
        font = ImageFont.load_default()

    for rec in records:
        x, y, w, h = rec["bbox"]
        x1 = int(round(x))
        y1 = int(round(y))
        x2 = int(round(x + w))
        y2 = int(round(y + h))
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)

        gt_id = str(rec["gt_id"])
        lines = [f"GT: {id_to_zh.get(gt_id, gt_id)}"]
        for tag in model_tags:
            pred_id = str(rec["preds"].get(tag, "?"))
            lines.append(f"{tag}: {id_to_zh.get(pred_id, pred_id)}")

        text = "\n".join(lines)
        tx = max(0, x1)
        ty = max(0, y1 - (len(lines) * 20 + 4))
        draw.multiline_text((tx, ty), text, fill=(255, 255, 255), font=font, spacing=2)

    return vis
