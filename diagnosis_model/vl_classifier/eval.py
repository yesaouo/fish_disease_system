import os
import json
import math
import argparse
import zlib
import colorsys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import cv2

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEFAULT_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

_LABEL2RGB255 = {}
_LABEL2BGR255 = {}


def _label_to_rgb255(label: str, s: float = 0.65, v: float = 0.95):
    h = (zlib.crc32(str(label).encode("utf-8")) & 0xFFFFFFFF) / 2**32
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def get_label_colors(label):
    label_key = str(label)
    if label_key not in _LABEL2RGB255:
        rgb = _label_to_rgb255(label_key)
        _LABEL2RGB255[label_key] = rgb
        _LABEL2BGR255[label_key] = (rgb[2], rgb[1], rgb[0])
    return _LABEL2RGB255[label_key], _LABEL2BGR255[label_key]


# =========================
# Drawing helpers
# =========================

def put_chinese_text(img_bgr: np.ndarray, text: str, pos, color_rgb, font_path: str, size: int = 24):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(font_path, size)
    except OSError:
        font = ImageFont.load_default()

    x, y = pos
    outline = (0, 0, 0)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=color_rgb)

    out_rgb = np.array(img_pil)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


# =========================
# Metrics
# =========================

def _sorted_labels(labels):
    return sorted(
        list(labels),
        key=lambda x: int(x) if str(x).isdigit() else str(x),
    )



def save_confusion_matrix(y_true, y_pred, labels, output_path, title="Confusion Matrix", normalize=None):
    labels = _sorted_labels(set(labels) | set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    plt.figure(figsize=(12, 10))
    fmt = ".2f" if normalize else "d"
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(output_path))
    plt.close()



def save_classification_report(y_true, y_pred, labels, output_path):
    labels = _sorted_labels(set(labels) | set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


# =========================
# Data helpers
# =========================

def load_symptoms_data(symptoms_json_path: str):
    with open(symptoms_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_map = data.get("label_map", {})
    data_content = data.get("data", {})

    id_to_zh_map = {}
    for k, info in label_map.items():
        id_to_zh_map[str(k)] = info.get("zh", str(k)) if isinstance(info, dict) else str(k)

    flat_texts: List[str] = []
    flat_label_ids: List[str] = []
    for k, content in data_content.items():
        label_id = str(k)
        captions = content.get("captions_en", []) if isinstance(content, dict) else []
        for cap in captions or []:
            flat_texts.append(f"This is {str(cap).lower()}.")
            flat_label_ids.append(label_id)

    all_label_ids = _sorted_labels(set(flat_label_ids))
    return flat_texts, flat_label_ids, id_to_zh_map, all_label_ids



def get_scaled_rect_crop(img_pil: Image.Image, coco_bbox, scale: float):
    x, y, w, h = coco_bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    new_w, new_h = w * scale, h * scale

    x1 = cx - new_w / 2.0
    y1 = cy - new_h / 2.0
    x2 = cx + new_w / 2.0
    y2 = cy + new_h / 2.0

    W, H = img_pil.size
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(1, min(W, int(round(x2))))
    y2 = max(1, min(H, int(round(y2))))

    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)

    return img_pil.crop((x1, y1, x2, y2))


# =========================
# Model helpers
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

    raise RuntimeError("Cannot extract image features from model output.")



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

    raise RuntimeError("Cannot extract text features from model output.")


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



def parse_model_specs(model_specs: List[str]):
    out = []
    seen = set()
    for spec in model_specs:
        if "=" not in spec:
            raise ValueError(f"model spec 格式錯誤：{spec}，需為 tag=path_or_repo")
        tag, name_or_path = spec.split("=", 1)
        tag = tag.strip()
        name_or_path = name_or_path.strip()
        if not tag or not name_or_path:
            raise ValueError(f"model spec 格式錯誤：{spec}")
        if tag in seen:
            raise ValueError(f"model tag 重複：{tag}")
        seen.add(tag)
        out.append((tag, name_or_path))
    return out



def _find_wrapper_state_path(name_or_path: str) -> Optional[str]:
    candidate = os.path.join(name_or_path, "wrapper_state.pt")
    return candidate if os.path.exists(candidate) else None



def load_model_and_processor(name_or_path: str, device: str, force_fusion: bool = False):
    from transformers import AutoProcessor, AutoModel

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


@torch.no_grad()
def encode_text_features(model, processor, texts, device, text_batch_size=256, max_length=64, use_amp=True):
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
            f = get_text_features(
                model,
                inputs["input_ids"],
                inputs.get("attention_mask", None),
            )

        f = f.float()
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)
        feats.append(f.detach().cpu())

    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_image_features(model, processor, images_local, images_global, device, img_batch_size=64, use_amp=True):
    feats = []
    amp_ok = bool(use_amp and str(device).startswith("cuda"))

    for i in range(0, len(images_local), img_batch_size):
        batch_local = images_local[i:i + img_batch_size]
        local_inputs = processor(images=batch_local, return_tensors="pt")
        local_inputs = {k: v.to(device) for k, v in local_inputs.items()}

        with torch.cuda.amp.autocast(enabled=amp_ok):
            if getattr(model, "is_wrapper", False):
                batch_global = images_global[i:i + img_batch_size]
                global_inputs = processor(images=batch_global, return_tensors="pt")
                global_inputs = {k: v.to(device) for k, v in global_inputs.items()}
                f = model.forward_image(local_inputs["pixel_values"], global_inputs["pixel_values"])
            else:
                f = get_image_features(model, local_inputs["pixel_values"])

        f = f.float()
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)
        feats.append(f.detach().cpu())

    return torch.cat(feats, dim=0)


# =========================
# Main evaluation
# =========================

def process_gt_dataset(
    data_dir: str,
    symptoms_json_path: str,
    output_dir: str,
    model_specs,
    scale: float = 1.0,
    save_vis: bool = False,
    font_path: str = DEFAULT_FONT_PATH,
    device: str = "cuda",
    text_batch_size: int = 256,
    img_batch_size: int = 64,
    max_length: int = 64,
    use_amp: bool = True,
    force_fusion: bool = False,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coco_json_path = data_dir / "_annotations.coco.json"
    if not coco_json_path.exists():
        raise FileNotFoundError(f"找不到：{coco_json_path}")

    print("Loading symptoms.json ...")
    flat_texts, flat_label_ids, id_to_zh_map, all_label_ids = load_symptoms_data(symptoms_json_path)
    if len(flat_texts) == 0:
        raise RuntimeError("symptoms.json 的 captions_en 為空，無法做 text candidates。")

    models = {}
    text_features = {}
    print("Loading models and encoding text features ...")
    for tag, name_or_path in model_specs:
        print(f"  - {tag}: {name_or_path}")
        m, p = load_model_and_processor(name_or_path, device=device, force_fusion=force_fusion)
        models[tag] = (m, p)
        tf = encode_text_features(
            m,
            p,
            flat_texts,
            device=device,
            text_batch_size=text_batch_size,
            max_length=max_length,
            use_amp=use_amp,
        )
        text_features[tag] = tf
        mode_name = "fusion" if getattr(m, "is_wrapper", False) else "baseline"
        print(f"    mode={mode_name}, text_features={tuple(tf.shape)}")

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    print(f"Found {len(images)} images, {len(annotations)} annotations.")

    img_id_to_anns: Dict[int, List[dict]] = {}
    for ann in annotations:
        img_id = ann["image_id"]
        img_id_to_anns.setdefault(img_id, []).append(ann)

    y_true = []
    y_pred = {tag: [] for tag, _ in model_specs}

    vis_dir = output_dir / "vis"
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for img_info in images:
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = data_dir / file_name
        if not img_path.exists():
            continue

        anns = img_id_to_anns.get(img_id, [])
        if not anns:
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] 讀圖失敗 {file_name}: {e}")
            continue

        crops = []
        globals_list = []
        gt_ids = []
        bboxes_xywh = []

        for ann in anns:
            bbox = ann.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue

            crop = get_scaled_rect_crop(pil_img, bbox, scale=scale)
            if crop.size[0] <= 0 or crop.size[1] <= 0:
                continue

            crops.append(crop)
            globals_list.append(pil_img)
            gt_ids.append(str(ann.get("category_id")))
            bboxes_xywh.append(bbox)

        if not crops:
            continue

        y_true.extend(gt_ids)
        preds_this_img = {}

        for tag, (m, p) in models.items():
            img_feats = encode_image_features(
                m,
                p,
                crops,
                globals_list,
                device=device,
                img_batch_size=img_batch_size,
                use_amp=use_amp,
            )
            scores = img_feats @ text_features[tag].T
            best_idx = scores.argmax(dim=-1).tolist()
            pred_ids = [flat_label_ids[i] for i in best_idx]

            y_pred[tag].extend(pred_ids)
            preds_this_img[tag] = pred_ids

        if save_vis:
            cv2_img = cv2.imread(str(img_path))
            if cv2_img is None:
                continue
            vis_img = cv2_img.copy()

            for i, (gt_id, bbox) in enumerate(zip(gt_ids, bboxes_xywh)):
                x, y, w, h = [int(round(v)) for v in bbox]
                _, color_bgr = get_label_colors(gt_id)
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), color_bgr, 2)

                lines = []
                gt_zh = id_to_zh_map.get(gt_id, "未知")
                lines.append(("GT", gt_zh, (255, 255, 255)))

                for tag, _ in model_specs:
                    pred_id = preds_this_img[tag][i]
                    pred_zh = id_to_zh_map.get(pred_id, "未知")
                    correct = pred_id == gt_id
                    color = (0, 255, 0) if correct else (255, 0, 0)
                    lines.append((tag, pred_zh, color))

                font_size = 22
                line_h = font_size + 6
                block_h = line_h * len(lines) + 4
                start_y = y - block_h - 2
                if start_y < 0:
                    start_y = y + h + 2
                start_x = max(0, x)

                for li, (prefix, text_zh, color_rgb) in enumerate(lines):
                    ty = start_y + li * line_h
                    vis_img = put_chinese_text(
                        vis_img,
                        f"{prefix}: {text_zh}",
                        (start_x, ty),
                        color_rgb=color_rgb,
                        font_path=font_path,
                        size=font_size,
                    )

            cv2.imwrite(str(vis_dir / f"pred_{file_name}"), vis_img)

    print("\n===== Results =====")
    for tag, _ in model_specs:
        pred = y_pred[tag]
        if len(pred) != len(y_true):
            raise RuntimeError(f"{tag} 的 y_pred 長度({len(pred)}) != y_true 長度({len(y_true)})，資料對齊有問題。")

        acc = (np.array(pred) == np.array(y_true)).mean() if len(y_true) else 0.0
        print(f"[{tag}] accuracy = {acc:.4f}  (n={len(y_true)})")

        save_confusion_matrix(
            y_true,
            pred,
            all_label_ids,
            output_dir / f"confusion_matrix_{tag}.png",
            title=f"Confusion Matrix ({tag})",
            normalize=None,
        )
        save_confusion_matrix(
            y_true,
            pred,
            all_label_ids,
            output_dir / f"confusion_matrix_{tag}_norm.png",
            title=f"Confusion Matrix ({tag}) - Normalized (Recall)",
            normalize="true",
        )
        save_classification_report(
            y_true,
            pred,
            all_label_ids,
            output_dir / f"report_{tag}.txt",
        )

    print(f"\nDone! Saved outputs to: {output_dir}")
    if save_vis:
        print(f"Visualization saved to: {vis_dir}")



def build_argparser():
    ap = argparse.ArgumentParser(
        description="Unified evaluator for baseline / multipos / fusion SigLIP2 checkpoints on COCO GT bboxes."
    )
    ap.add_argument("--data_dir", type=str, required=True, help="資料夾內需包含：圖片檔 + _annotations.coco.json")
    ap.add_argument("--symptoms_json", type=str, required=True, help="symptoms.json 路徑")
    ap.add_argument("--output_dir", type=str, required=True, help="輸出資料夾")
    ap.add_argument("--scale", type=float, default=1.0, help="bbox 放大倍率（矩形，w/h 同時乘）")
    ap.add_argument(
        "--model",
        type=str,
        action="append",
        required=True,
        help="可重複指定：tag=repo_or_path，例如 base=./siglip2_finetuned",
    )
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--save_vis", action="store_true", help="是否輸出可視化（預設關閉）")
    ap.add_argument("--font_path", type=str, default=DEFAULT_FONT_PATH, help="中文字型路徑")
    ap.add_argument("--text_batch_size", type=int, default=256)
    ap.add_argument("--img_batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--no_amp", action="store_true", help="關閉 AMP（預設 GPU 會開）")
    ap.add_argument(
        "--fusion",
        action="store_true",
        help="強制用 fusion pipeline 載入所有 model；若 model 目錄中沒有 wrapper_state.pt 會報錯。"
    )
    return ap



def main():
    args = build_argparser().parse_args()
    model_specs = parse_model_specs(args.model)

    process_gt_dataset(
        data_dir=args.data_dir,
        symptoms_json_path=args.symptoms_json,
        output_dir=args.output_dir,
        model_specs=model_specs,
        scale=args.scale,
        save_vis=args.save_vis,
        font_path=args.font_path,
        device=args.device,
        text_batch_size=args.text_batch_size,
        img_batch_size=args.img_batch_size,
        max_length=args.max_length,
        use_amp=(not args.no_amp),
        force_fusion=bool(args.fusion),
    )


if __name__ == "__main__":
    main()
