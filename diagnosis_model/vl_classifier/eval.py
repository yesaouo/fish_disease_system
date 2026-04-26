import os
import json
import argparse
import zlib
import colorsys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import cv2

from common import (
    format_caption,
    get_image_features,
    get_text_features,
    load_flat_caption_bank,
    LocalGlobalFusionWrapper,
    sort_label_ids,
)

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

def save_confusion_matrix(y_true, y_pred, labels, output_path, title="Confusion Matrix", normalize=None):
    labels = sort_label_ids(set(labels) | set(y_true) | set(y_pred))
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
    labels = sort_label_ids(set(labels) | set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


# =========================
# Data helpers
# =========================

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

def _argmax_over_subset(scores: torch.Tensor, mask: torch.Tensor) -> List[int]:
    """
    scores: (N, M)
    mask:   (M,) bool，True 表示允許的欄位
    回傳 argmax 的 index list。若整列都被 mask 掉會退化成整個 scores 的 argmax。
    """
    if mask is None or mask.all():
        return scores.argmax(dim=-1).tolist()
    if not mask.any():
        return scores.argmax(dim=-1).tolist()
    s = scores.clone()
    s[:, ~mask] = float("-inf")
    return s.argmax(dim=-1).tolist()



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
    eval_lang: str = "both",
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if eval_lang == "both":
        langs = ("en", "zh")
        split_views = ["all", "en", "zh"]
    else:
        langs = (eval_lang,)
        split_views = ["all"]

    coco_json_path = data_dir / "_annotations.coco.json"
    if not coco_json_path.exists():
        raise FileNotFoundError(f"找不到：{coco_json_path}")

    print(f"Loading symptoms.json (langs={langs}) ...")
    bank = load_flat_caption_bank(symptoms_json_path, langs=langs)
    flat_texts = bank.texts
    flat_label_ids = bank.label_ids
    flat_langs = bank.langs
    id_to_zh_map = bank.id_to_zh
    all_label_ids = bank.sorted_label_ids

    # 語言過濾遮罩（對應到 bank 欄位）
    en_mask = torch.tensor([l == "en" for l in flat_langs], dtype=torch.bool)
    zh_mask = torch.tensor([l == "zh" for l in flat_langs], dtype=torch.bool)

    print(f"Bank size = {len(flat_texts)} (en={int(en_mask.sum())}, zh={int(zh_mask.sum())})")

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

    y_true: List[str] = []
    # y_pred[tag][view] -> list[str]
    y_pred: Dict[str, Dict[str, List[str]]] = {
        tag: {view: [] for view in split_views} for tag, _ in model_specs
    }

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
        preds_this_img: Dict[str, Dict[str, List[str]]] = {}

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
            scores = img_feats @ text_features[tag].T  # (n_crops, bank_size)

            view_preds: Dict[str, List[str]] = {}
            # all
            idx_all = _argmax_over_subset(scores, torch.ones(scores.size(1), dtype=torch.bool))
            view_preds["all"] = [flat_label_ids[i] for i in idx_all]

            if "en" in split_views:
                idx_en = _argmax_over_subset(scores, en_mask)
                view_preds["en"] = [flat_label_ids[i] for i in idx_en]
            if "zh" in split_views:
                idx_zh = _argmax_over_subset(scores, zh_mask)
                view_preds["zh"] = [flat_label_ids[i] for i in idx_zh]

            for view in split_views:
                y_pred[tag][view].extend(view_preds[view])

            preds_this_img[tag] = view_preds

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
                    # 可視化只顯示 "all" view（完整 bank）的結果
                    pred_id = preds_this_img[tag]["all"][i]
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
    summary_rows: List[str] = []
    for tag, _ in model_specs:
        for view in split_views:
            pred = y_pred[tag][view]
            if len(pred) != len(y_true):
                raise RuntimeError(
                    f"{tag}/{view} 的 y_pred 長度({len(pred)}) != y_true 長度({len(y_true)})，資料對齊有問題。"
                )

            acc = (np.array(pred) == np.array(y_true)).mean() if len(y_true) else 0.0
            line = f"[{tag}][view={view}] accuracy = {acc:.4f}  (n={len(y_true)})"
            print(line)
            summary_rows.append(line)

            save_confusion_matrix(
                y_true,
                pred,
                all_label_ids,
                output_dir / f"confusion_matrix_{tag}_{view}.png",
                title=f"Confusion Matrix ({tag}, view={view})",
                normalize=None,
            )
            save_confusion_matrix(
                y_true,
                pred,
                all_label_ids,
                output_dir / f"confusion_matrix_{tag}_{view}_norm.png",
                title=f"Confusion Matrix ({tag}, view={view}) - Normalized (Recall)",
                normalize="true",
            )
            save_classification_report(
                y_true,
                pred,
                all_label_ids,
                output_dir / f"report_{tag}_{view}.txt",
            )

    # summary 總表
    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_rows) + "\n")

    print(f"\nDone! Saved outputs to: {output_dir}")
    if save_vis:
        print(f"Visualization saved to: {vis_dir}")



# =========================
# Overall (per-image retrieval) evaluation
# =========================

def collect_overall_samples(data_dir: Path) -> Tuple[List[Dict[str, str]], int]:
    """Walk COCO json's images[] and keep ones with overall.colloquial_zh + medical_zh."""
    coco_json_path = data_dir / "_annotations.coco.json"
    if not coco_json_path.exists():
        raise FileNotFoundError(f"找不到：{coco_json_path}")
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    samples: List[Dict[str, str]] = []
    no_overall = 0
    missing = 0
    for img in coco.get("images", []):
        if not isinstance(img, dict):
            continue
        file_name = img.get("file_name", None)
        if not file_name:
            continue
        overall = img.get("overall", None)
        if not isinstance(overall, dict):
            no_overall += 1
            continue
        coll = overall.get("colloquial_zh", None)
        med = overall.get("medical_zh", None)
        if not (isinstance(coll, str) and coll.strip()):
            no_overall += 1
            continue
        if not (isinstance(med, str) and med.strip()):
            no_overall += 1
            continue
        img_path = data_dir / file_name
        if not img_path.exists():
            missing += 1
            continue
        samples.append({
            "file_name": file_name,
            "img_path": str(img_path),
            "colloquial": coll.strip(),
            "medical": med.strip(),
        })

    print(f"Found {len(samples)} usable images; skipped no_overall={no_overall}, missing={missing}")
    return samples, no_overall + missing



def compute_retrieval_metrics(
    image_feats: torch.Tensor,
    text_feats: torch.Tensor,
    gt_text_pairs: List[Tuple[int, int]],
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Tuple[Dict[str, float], Dict[str, float], torch.Tensor]:
    """
    image_feats: (N, d) L2-normalized
    text_feats:  (M, d) L2-normalized, M = 2N
    gt_text_pairs: per image i, the (colloquial_idx, medical_idx) within text_feats
    Returns (i2t_metrics, t2i_metrics, sims).

    I→T rank = best rank among image i's two GT captions (1-indexed).
    T→I rank = rank of caption j's source image (1-indexed).
    """
    sims = image_feats @ text_feats.T  # (N, M)
    N, M = sims.shape
    if len(gt_text_pairs) != N:
        raise ValueError(f"gt_text_pairs len {len(gt_text_pairs)} != N {N}")

    # I -> T
    gt_img_mask = torch.zeros(N, M, dtype=torch.bool)
    for i, (a, b) in enumerate(gt_text_pairs):
        gt_img_mask[i, a] = True
        gt_img_mask[i, b] = True

    pos_sims_i = sims.masked_fill(~gt_img_mask, float("-inf"))
    best_pos_i = pos_sims_i.max(dim=1).values  # (N,)
    ranks_i = 1 + (sims > best_pos_i.unsqueeze(1)).sum(dim=1)  # (N,) long
    ranks_i = ranks_i.float()

    i2t = {f"R@{k}": float((ranks_i <= k).float().mean().item()) for k in ks}
    i2t["MRR"] = float((1.0 / ranks_i).mean().item())
    i2t["median_rank"] = float(ranks_i.median().item())
    i2t["mean_rank"] = float(ranks_i.mean().item())

    # T -> I: for each caption j, the source image
    text2img = torch.full((M,), -1, dtype=torch.long)
    for i, (a, b) in enumerate(gt_text_pairs):
        text2img[a] = i
        text2img[b] = i
    valid = text2img >= 0
    sims_t = sims.T  # (M, N)
    pos_sims_t = sims_t.gather(1, text2img.clamp(min=0).view(-1, 1)).squeeze(1)  # (M,)
    ranks_t = 1 + (sims_t > pos_sims_t.unsqueeze(1)).sum(dim=1)
    ranks_t = ranks_t[valid].float()

    t2i = {f"R@{k}": float((ranks_t <= k).float().mean().item()) for k in ks}
    t2i["MRR"] = float((1.0 / ranks_t).mean().item())
    t2i["median_rank"] = float(ranks_t.median().item())
    t2i["mean_rank"] = float(ranks_t.mean().item())

    return i2t, t2i, sims



def process_overall_dataset(
    data_dir: str,
    output_dir: str,
    model_specs,
    device: str = "cuda",
    text_batch_size: int = 256,
    img_batch_size: int = 64,
    max_length: int = 64,
    use_amp: bool = True,
    topk_dump: int = 5,
):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, _ = collect_overall_samples(data_dir)
    if len(samples) == 0:
        raise RuntimeError("No usable images for overall eval (no overall.colloquial_zh/medical_zh found).")

    N = len(samples)
    images = [Image.open(s["img_path"]).convert("RGB") for s in samples]

    # Interleaved bank: bank[2i] = colloquial_i, bank[2i+1] = medical_i
    bank_texts: List[str] = []
    bank_kinds: List[str] = []  # "colloquial" or "medical"
    bank_owner: List[int] = []  # image index this caption belongs to
    gt_text_pairs: List[Tuple[int, int]] = []
    for i, s in enumerate(samples):
        bank_texts.append(s["colloquial"]); bank_kinds.append("colloquial"); bank_owner.append(i)
        bank_texts.append(s["medical"]);    bank_kinds.append("medical");    bank_owner.append(i)
        gt_text_pairs.append((2 * i, 2 * i + 1))

    print(f"Overall pool: N={N} images, M={len(bank_texts)} captions (2 per image)")

    summary_rows: List[str] = []
    for tag, name_or_path in model_specs:
        print(f"\n=== model: {tag} ({name_or_path}) ===")
        m, p = load_model_and_processor(name_or_path, device=device, force_fusion=False)
        if getattr(m, "is_wrapper", False):
            raise RuntimeError(
                f"{tag}: 此 checkpoint 為 fusion wrapper，但 overall 模式不支援 fusion。"
            )

        text_feats = encode_text_features(
            m, p, bank_texts, device=device,
            text_batch_size=text_batch_size, max_length=max_length, use_amp=use_amp,
        )
        # whole-image encoding; non-fusion path ignores images_global
        image_feats = encode_image_features(
            m, p, images, images, device=device,
            img_batch_size=img_batch_size, use_amp=use_amp,
        )
        print(f"  image_feats={tuple(image_feats.shape)}, text_feats={tuple(text_feats.shape)}")

        i2t, t2i, sims = compute_retrieval_metrics(image_feats, text_feats, gt_text_pairs)

        def _fmt(d: Dict[str, float]) -> str:
            return ", ".join(f"{k}={v:.4f}" for k, v in d.items())

        line_i2t = f"[{tag}] I->T: {_fmt(i2t)}"
        line_t2i = f"[{tag}] T->I: {_fmt(t2i)}"
        print(line_i2t)
        print(line_t2i)
        summary_rows += [line_i2t, line_t2i]

        with open(output_dir / f"retrieval_metrics_{tag}.txt", "w", encoding="utf-8") as f:
            f.write(f"N (images) = {N}\nM (captions) = {len(bank_texts)}\n\n")
            f.write("I -> T:\n")
            for k, v in i2t.items():
                f.write(f"  {k} = {v:.6f}\n")
            f.write("\nT -> I:\n")
            for k, v in t2i.items():
                f.write(f"  {k} = {v:.6f}\n")

        # Top-K dump per image (one JSON line per image)
        if topk_dump and topk_dump > 0:
            K = int(min(topk_dump, sims.size(1)))
            topv, topi = sims.topk(K, dim=1)
            jsonl_path = output_dir / f"topk_{tag}.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for i in range(N):
                    row = {
                        "image": samples[i]["file_name"],
                        "gt": {
                            "colloquial": samples[i]["colloquial"],
                            "medical": samples[i]["medical"],
                        },
                        "top": [
                            {
                                "rank": int(r + 1),
                                "score": float(topv[i, r].item()),
                                "kind": bank_kinds[int(topi[i, r].item())],
                                "owner_idx": int(bank_owner[int(topi[i, r].item())]),
                                "is_gt": bool(bank_owner[int(topi[i, r].item())] == i),
                                "text": bank_texts[int(topi[i, r].item())],
                            }
                            for r in range(K)
                        ],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  topk dump -> {jsonl_path}")

    with open(output_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_rows) + "\n")

    print(f"\nDone! Saved outputs to: {output_dir}")


# =========================
# CLI
# =========================

def build_argparser():
    ap = argparse.ArgumentParser(
        description="Unified evaluator for baseline / multipos / fusion SigLIP2 checkpoints on COCO GT bboxes."
    )
    ap.add_argument(
        "--target",
        type=str,
        choices=["lesion", "overall"],
        default="lesion",
        help="lesion: 對 bbox crop 在 symptoms.json bank 上做分類（現行）。overall: 對整張圖做 image↔caption retrieval（image['overall'].colloquial_zh/medical_zh）",
    )
    ap.add_argument("--data_dir", type=str, required=True, help="資料夾內需包含：圖片檔 + _annotations.coco.json")
    ap.add_argument("--symptoms_json", type=str, default=None, help="symptoms.json 路徑（target=lesion 必填）")
    ap.add_argument("--output_dir", type=str, required=True, help="輸出資料夾")
    ap.add_argument("--scale", type=float, default=1.0, help="bbox 放大倍率（矩形，w/h 同時乘；只用於 target=lesion）")
    ap.add_argument("--topk_dump", type=int, default=5, help="overall: 每張圖在輸出寫前 K 條 retrieval 結果到 topk_{tag}.jsonl，0 表示不寫")
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
    ap.add_argument(
        "--eval_lang",
        type=str,
        choices=["en", "zh", "both"],
        default="both",
        help="評估時要啟用的語言 bank；both 會同時輸出 all/en/zh 三組指標"
    )
    return ap



def main():
    parser = build_argparser()
    args = parser.parse_args()
    model_specs = parse_model_specs(args.model)

    if args.target == "overall":
        if args.fusion:
            parser.error("--target overall is incompatible with --fusion")
        if args.save_vis:
            print("[WARN] --save_vis is not implemented for target=overall yet; ignoring.")
        process_overall_dataset(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_specs=model_specs,
            device=args.device,
            text_batch_size=args.text_batch_size,
            img_batch_size=args.img_batch_size,
            max_length=args.max_length,
            use_amp=(not args.no_amp),
            topk_dump=int(args.topk_dump),
        )
        return

    if not args.symptoms_json:
        parser.error("--symptoms_json is required for --target lesion")

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
        eval_lang=args.eval_lang,
    )


if __name__ == "__main__":
    main()