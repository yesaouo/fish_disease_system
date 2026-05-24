import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from transformers import AutoProcessor, AutoModel, get_cosine_schedule_with_warmup

from common import (
    format_caption,
    get_image_features,
    get_text_features,
    load_flat_caption_bank,
    LocalGlobalFusionWrapper,
    normalize_text_mode,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =========================
# Default Config
# =========================
DEFAULT_CONFIG = {
    # model/data
    # "model_name": "openai/clip-vit-base-patch16",
    "model_name": "google/siglip2-base-patch16-224",
    "data_root": "/mnt/ssd/YJ/fish_disease_system/data/coco/_merged",
    "symptoms_file": "/mnt/ssd/YJ/fish_disease_system/data/annotation/backup/20260207/fish_disease/symptoms.json",
    "output_dir": None,
    "crop_mode": "bbox",  # "bbox" or "square"

    # text supervision
    "langs": ["en", "zh"],     # 訓練時用到的語言；多語會 flatten 進同一個 caption bank
    "text_mode": "captions",   # captions | class_name | captions_plus_class_name
    "warmup_ratio": 0.15,       # 單語建議 0.2；雙語建議 0.25；凍結建議預設 -1

    # train hypers
    "batch_size": 128,
    "num_epochs": 8,
    "learning_rate": 1e-4,
    "weight_decay": None,  # mode-aware default
    "max_length": 64,
    "num_workers": 8,
    "pin_memory": True,
    "grad_clip_norm": 1.0,
    "use_amp": True,
    "seed": 42,
    "freeze_text_encoder": True,

    # fusion-specific
    "dropout_prob": 0.1,
    "fusion_base_lr": 5e-5,
    "fusion_head_lr": 1e-4,
    # gate mixing the fusion branch back into the local feature:
    #   "scalar" — single learnable scalar (production)
    #   "film"   — input-conditioned per-channel gate (architecture lever, off by default)
    "fusion_gate_mode": "scalar",

    # multi-positive / custom loss
    "evidence_alpha": 3.0,
    "temperature": 0.07,
    "label_smoothing": 0.05,
    "t2i_skip_no_positive": True,
    "t2i_label_level_mean": True,
    "t2i_lambda": 0.5,

    # within-image cross-lesion supervised contrastive (B-fallback prototype)
    "cross_lesion": False,
    "cross_alpha": 0.3,
    "cross_tau": 0.1,

    # Class imbalance (goal: lift macro f1 on rare lesion categories).
    # Two orthogonal, default-off levers; both require --multipos --target lesion.
    "class_balanced_loss": False,   # weight per-sample i2t term by effective-number class weights
    "cb_beta": 0.999,               # effective-number beta (Cui et al. 2019); w_c = (1-β)/(1-β^n_c)
    "balanced_sampler": False,      # WeightedRandomSampler with inverse-frequency sample weights

    # Grouped batch sampler — pairs with cross_lesion to push firing rate
    # from ~13% (random) to ~100% (every batch contains multi-category image groups).
    "grouped_sampler": False,
    "grouped_ratio": 0.5,

    # LSCFT — Lesion-Selective Counterfactual Faithfulness Training
    # Requires --multipos --fusion --target lesion.
    # Adds 3 vision-encoder forwards per batch (orig + positive + negative
    # interventions), sharing the precomputed global feature.
    "lscft": False,
    "lscft_alpha": 0.5,        # weight on LSCFT loss
    "lscft_margin": 0.1,       # ranking margin in cosine space
    "lscft_pos_size": 80,      # center mask size for positive (lesion-removed) arm
    "lscft_neg_keep": 140,     # center keep size for peripheral-mask negative arm
    "lscft_mask_jitter": 10,   # random spatial jitter (pixels) for mask center
    "lscft_jitter": 0.05,      # color-jitter strength (negative arm)

    # splits
    "train_split": "train",
    "valid_split": "valid",
    "test_split": "test",
    "eval_test_after_train": False,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# =========================
# Utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def resolve_image_base_dir(split_dir: str) -> str:
    cand = os.path.join(split_dir, "images")
    return cand if os.path.isdir(cand) else split_dir



def resolve_image_path(image_base_dir: str, file_name: str, data_root: str, split: str) -> str:
    if os.path.isabs(file_name) and os.path.exists(file_name):
        return file_name

    p1 = os.path.join(image_base_dir, file_name)
    if os.path.exists(p1):
        return p1

    p2 = os.path.join(data_root, file_name)
    if os.path.exists(p2):
        return p2

    p3 = os.path.join(data_root, split, file_name)
    if os.path.exists(p3):
        return p3

    return p1


# =========================
# Symptoms Captions (multilingual)
# =========================

def build_caption_bank(
    captions_by_cat: Dict[str, List[Tuple[str, str]]],
) -> Tuple[List[str], List[int], List[str], Dict[int, List[int]], Dict[int, Dict[int, List[int]]]]:
    """
    Returns:
        bank_texts:       已套 prompt 的完整 bank
        bank_labels:      每個 bank 位置對應的 cat_id
        bank_langs:       每個 bank 位置的語言（'en'/'zh'/...）
        cat2bank_indices: cat_id -> 本類別所有 bank 索引（跨語言）
        cat2evidence:    cat_id -> ei -> List[bank_idx]（同一 ei 對應到各語言的同位 caption）
    """
    keys_sorted = sorted(captions_by_cat.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    bank_texts: List[str] = []
    bank_labels: List[int] = []
    bank_langs: List[str] = []
    cat2bank_indices: Dict[int, List[int]] = {}
    # cat_id -> lang -> 該語言 caption 的 bank 索引
    lang_positions: Dict[int, Dict[str, List[int]]] = {}

    for k in keys_sorted:
        cat_id = int(k)
        idxs: List[int] = []
        per_lang: Dict[str, List[int]] = {}
        for cap, lang in captions_by_cat[k]:
            bi = len(bank_texts)
            bank_texts.append(format_caption(cap, lang))
            bank_labels.append(cat_id)
            bank_langs.append(lang)
            idxs.append(bi)
            per_lang.setdefault(lang, []).append(bi)
        cat2bank_indices[cat_id] = idxs
        lang_positions[cat_id] = per_lang

    cat2evidence: Dict[int, Dict[int, List[int]]] = {}
    for cat_id, per_lang in lang_positions.items():
        max_ei = max((len(lst) for lst in per_lang.values()), default=0)
        ev_map: Dict[int, List[int]] = {}
        for ei in range(max_ei):
            positions: List[int] = []
            # 以固定順序掃語言，確保每個 sample 的 evidence 位置穩定
            for lang in sorted(per_lang.keys()):
                lst = per_lang[lang]
                if ei < len(lst):
                    positions.append(lst[ei])
            ev_map[ei] = positions
        cat2evidence[cat_id] = ev_map

    if len(bank_texts) == 0:
        raise ValueError("Caption bank is empty. Check symptoms.json")

    return bank_texts, bank_labels, bank_langs, cat2bank_indices, cat2evidence


# =========================
# Crop helpers
# =========================

def clamp_bbox_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, math.floor(x1))))
    y1 = int(max(0, min(H - 1, math.floor(y1))))
    x2 = int(max(1, min(W, math.ceil(x2))))
    y2 = int(max(1, min(H, math.ceil(y2))))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return x1, y1, x2, y2



def crop_bbox(image: Image.Image, bbox_xywh: Tuple[float, float, float, float]) -> Image.Image:
    x, y, w, h = bbox_xywh
    W, H = image.size
    x1, y1, x2, y2 = clamp_bbox_xyxy(x, y, x + w, y + h, W, H)
    return image.crop((x1, y1, x2, y2))



def crop_square_with_black_padding(image: Image.Image, bbox_xywh: Tuple[float, float, float, float]) -> Image.Image:
    x, y, w, h = bbox_xywh
    W, H = image.size

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
    canvas = Image.new("RGB", (out_side, out_side), (0, 0, 0))

    src_left = int(max(0, math.floor(x1)))
    src_top = int(max(0, math.floor(y1)))
    src_right = int(min(W, math.ceil(x2)))
    src_bottom = int(min(H, math.ceil(y2)))

    if src_right <= src_left or src_bottom <= src_top:
        return canvas

    crop_region = image.crop((src_left, src_top, src_right, src_bottom))
    dst_left = int(round(src_left - x1))
    dst_top = int(round(src_top - y1))
    canvas.paste(crop_region, (dst_left, dst_top))
    return canvas


# =========================
# Dataset
# =========================
@dataclass
class CocoSample:
    img_path: str
    bbox_xywh: Tuple[float, float, float, float]
    text: Optional[str] = None
    label_id: Optional[int] = None
    evidence_index: int = -1
    image_id: int = -1


class CocoCropDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        crop_mode: str,
        captions_by_cat: Dict[str, List[Tuple[str, str]]],
        use_multipos: bool = False,
        use_fusion: bool = False,
        use_annotation_text: bool = True,
        text_mode: str = "captions",
    ):
        self.data_root = data_root
        self.split = split
        self.crop_mode = crop_mode.lower().strip()
        self.captions_by_cat = captions_by_cat
        self.use_multipos = use_multipos
        self.use_fusion = use_fusion
        self.use_annotation_text = use_annotation_text
        self.text_mode = normalize_text_mode(text_mode)

        split_dir = os.path.join(data_root, split)
        coco_json_path = os.path.join(split_dir, "_annotations.coco.json")
        if not os.path.exists(coco_json_path):
            raise FileNotFoundError(f"COCO json not found: {coco_json_path}")

        coco = load_json(coco_json_path)
        images = coco.get("images", [])
        annotations = coco.get("annotations", [])

        images_by_id = {img["id"]: img for img in images if isinstance(img, dict) and "id" in img}
        image_base_dir = resolve_image_base_dir(split_dir)

        rr_ptr: Dict[str, int] = {}
        samples: List[CocoSample] = []
        missing_images = 0
        skipped = 0

        for ann in annotations:
            if not isinstance(ann, dict):
                skipped += 1
                continue
            if ann.get("iscrowd", 0) == 1:
                skipped += 1
                continue
            if "category_id" not in ann:
                skipped += 1
                continue

            category_id = int(ann["category_id"])
            cat_key = str(category_id)

            bbox = ann.get("bbox", None)
            if not bbox or len(bbox) != 4:
                skipped += 1
                continue
            try:
                x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            except Exception:
                skipped += 1
                continue
            if w <= 0 or h <= 0:
                skipped += 1
                continue

            image_id = ann.get("image_id", None)
            if image_id is None or image_id not in images_by_id:
                skipped += 1
                continue
            file_name = images_by_id[image_id].get("file_name", None)
            if not file_name:
                skipped += 1
                continue

            img_path = resolve_image_path(image_base_dir, file_name, data_root, split)
            if not os.path.exists(img_path):
                missing_images += 1
                continue

            if cat_key not in self.captions_by_cat:
                raise KeyError(f"symptoms.json missing category_id={cat_key} (required)")

            if self.use_multipos:
                if self.text_mode == "class_name":
                    ei = 0
                else:
                    ei = int(ann.get("evidence_index", -1)) if ann.get("evidence_index") is not None else -1
                    if ei >= 0:
                        caps = self.captions_by_cat[cat_key]
                        ei = max(0, min(ei, len(caps) - 1))

                samples.append(
                    CocoSample(
                        img_path=img_path,
                        bbox_xywh=(x, y, w, h),
                        label_id=category_id,
                        evidence_index=ei,
                        image_id=int(image_id),
                    )
                )
            else:
                # baseline paired mode：每筆從多語 caption pool 選一條（支持 evidence_index 指定 或 round-robin）
                ann_text = ann.get("text", None)
                if self.use_annotation_text and isinstance(ann_text, str) and ann_text.strip():
                    # 若 annotation 直接給 text，就不做 prompt 包裝（視為使用者自訂）
                    text = ann_text.strip()
                else:
                    caps = self.captions_by_cat[cat_key]  # List[(cap, lang)]
                    if ann.get("evidence_index") is not None:
                        try:
                            ei = int(ann["evidence_index"])
                        except Exception:
                            ei = 0
                        ei = max(0, min(ei, len(caps) - 1))
                        cap, lang = caps[ei]
                    else:
                        ptr = rr_ptr.get(cat_key, 0)
                        cap, lang = caps[ptr % len(caps)]
                        rr_ptr[cat_key] = ptr + 1
                    text = format_caption(cap, lang)

                samples.append(
                    CocoSample(
                        img_path=img_path,
                        bbox_xywh=(x, y, w, h),
                        text=text,
                        image_id=int(image_id),
                    )
                )

        self.samples = samples
        self.labels = [s.label_id for s in samples if s.label_id is not None]

        print(
            f"[{split}] samples={len(self.samples)}, skipped={skipped}, missing_images={missing_images}, coco={coco_json_path}"
        )

        if self.crop_mode not in ("bbox", "square"):
            raise ValueError("crop_mode must be 'bbox' or 'square'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image_global = Image.open(s.img_path).convert("RGB")

        if self.crop_mode == "bbox":
            image_local = crop_bbox(image_global, s.bbox_xywh)
        else:
            image_local = crop_square_with_black_padding(image_global, s.bbox_xywh)

        out: Dict[str, Any] = {}
        if self.use_fusion:
            out["image_local"] = image_local
            out["image_global"] = image_global
        else:
            out["image"] = image_local

        if self.use_multipos:
            out["label_id"] = int(s.label_id)
            out["evidence_index"] = int(s.evidence_index)
        else:
            out["text"] = str(s.text)

        out["image_id"] = int(s.image_id)
        return out


class CocoOverallDataset(Dataset):
    """Per-image dataset that pairs the whole image with overall.colloquial_zh / overall.medical_zh."""

    def __init__(
        self,
        data_root: str,
        split: str,
        use_multipos: bool,
    ):
        self.data_root = data_root
        self.split = split
        self.use_multipos = use_multipos

        split_dir = os.path.join(data_root, split)
        coco_json_path = os.path.join(split_dir, "_annotations.coco.json")
        if not os.path.exists(coco_json_path):
            raise FileNotFoundError(f"COCO json not found: {coco_json_path}")

        coco = load_json(coco_json_path)
        images = coco.get("images", [])
        image_base_dir = resolve_image_base_dir(split_dir)

        samples: List[Dict[str, str]] = []
        skipped = 0
        no_overall = 0
        missing_images = 0

        for img in images:
            if not isinstance(img, dict):
                skipped += 1
                continue
            file_name = img.get("file_name", None)
            if not file_name:
                skipped += 1
                continue

            overall = img.get("overall", None)
            if not isinstance(overall, dict):
                no_overall += 1
                continue
            colloquial = overall.get("colloquial_zh", None)
            medical = overall.get("medical_zh", None)
            if not (isinstance(colloquial, str) and colloquial.strip()):
                no_overall += 1
                continue
            if not (isinstance(medical, str) and medical.strip()):
                no_overall += 1
                continue

            img_path = resolve_image_path(image_base_dir, file_name, data_root, split)
            if not os.path.exists(img_path):
                missing_images += 1
                continue

            samples.append({
                "img_path": img_path,
                "colloquial": colloquial.strip(),
                "medical": medical.strip(),
            })

        self.samples = samples
        print(
            f"[{split}] overall samples={len(self.samples)}, skipped={skipped}, "
            f"no_overall={no_overall}, missing_images={missing_images}, coco={coco_json_path}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image = Image.open(s["img_path"]).convert("RGB")

        if self.use_multipos:
            return {
                "image": image,
                "caption_colloquial": s["colloquial"],
                "caption_medical": s["medical"],
            }

        # baseline: deterministic alternation by sample index keeps valid loss reproducible
        text = s["colloquial"] if (idx % 2 == 0) else s["medical"]
        return {"image": image, "text": text}


# =========================
# Grouped batch sampler (hybrid: multi-category image groups + random singletons)
# =========================

class MultiCategoryGroupedBatchSampler(Sampler):
    """Hybrid batch sampler that guarantees each batch contains lesion crops
    from at least one multi-category image, so the cross-lesion loss fires
    every step instead of ~13% under random sampling.

    Each batch is composed of two halves:
      - grouped half (size ≈ grouped_ratio * batch_size):
          Pack lesion crops from multi-category images (≥2 distinct labels)
          group-by-group, never splitting an image across batches.
      - random half:
          Random crops from single-category images, filling the remainder.

    The two pools cycle independently when exhausted, so num_batches is
    determined upstream (defaults to ceil(total_crops / batch_size) to match
    the random-sampler training schedule).

    Yields lists of dataset indices; intended for use as `batch_sampler`.
    """

    def __init__(
        self,
        samples,
        batch_size: int,
        grouped_ratio: float = 0.5,
        num_batches: Optional[int] = None,
        seed: int = 0,
    ):
        if not (0.0 < grouped_ratio <= 1.0):
            raise ValueError("grouped_ratio must be in (0, 1]")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.batch_size = int(batch_size)
        self.grouped_ratio = float(grouped_ratio)
        self.grouped_target = max(1, int(round(self.batch_size * self.grouped_ratio)))
        self._rng = random.Random(int(seed))

        from collections import defaultdict
        img2idx: Dict[int, List[int]] = defaultdict(list)
        img2labels: Dict[int, set] = defaultdict(set)
        for i, s in enumerate(samples):
            img2idx[int(s.image_id)].append(i)
            if s.label_id is not None:
                img2labels[int(s.image_id)].add(int(s.label_id))

        self.eligible_groups: List[List[int]] = []
        self.singletons: List[int] = []
        for img_id, indices in img2idx.items():
            if len(img2labels[img_id]) >= 2:
                self.eligible_groups.append(indices)
            else:
                self.singletons.extend(indices)

        if not self.eligible_groups:
            raise ValueError(
                "grouped sampler enabled but no multi-category images found; "
                "either disable --grouped_sampler or check dataset labels."
            )

        total = len(samples)
        self._num_batches = (
            int(num_batches) if num_batches is not None else max(1, total // self.batch_size)
        )

        elig_crops = sum(len(g) for g in self.eligible_groups)
        print(
            f"[GroupedSampler] eligible_imgs={len(self.eligible_groups)} "
            f"({elig_crops} crops), singletons={len(self.singletons)}, "
            f"batches/epoch={self._num_batches}, grouped_target={self.grouped_target}/{self.batch_size}"
        )

    def __len__(self) -> int:
        return self._num_batches

    def _iter_groups(self):
        order = list(range(len(self.eligible_groups)))
        while True:
            self._rng.shuffle(order)
            for i in order:
                yield self.eligible_groups[i]

    def _iter_singletons(self):
        if not self.singletons:
            while True:
                yield None  # sentinel — should never be consumed
        pool = list(self.singletons)
        while True:
            self._rng.shuffle(pool)
            for i in pool:
                yield i

    def __iter__(self):
        group_gen = self._iter_groups()
        single_gen = self._iter_singletons()

        for _ in range(self._num_batches):
            batch: List[int] = []
            # Pack image groups until we hit (or would exceed) the grouped target.
            attempts = 0
            while len(batch) < self.grouped_target and attempts < 32:
                grp = next(group_gen)
                if len(batch) + len(grp) > self.batch_size:
                    # Single huge image group bigger than batch_size: take first batch_size crops
                    if len(batch) == 0 and len(grp) > self.batch_size:
                        batch.extend(grp[: self.batch_size])
                    attempts += 1
                    continue
                batch.extend(grp)
                attempts = 0
            # Fill remainder with singletons (with cycle).
            need = self.batch_size - len(batch)
            if need > 0:
                if self.singletons:
                    for _ in range(need):
                        batch.append(next(single_gen))
                else:
                    # No singletons available — pull extra image groups instead.
                    while len(batch) < self.batch_size:
                        grp = next(group_gen)
                        room = self.batch_size - len(batch)
                        batch.extend(grp[:room])
            yield batch[: self.batch_size]


# =========================
# Model helpers
# =========================

def get_logit_scale_and_bias(model, device: str, temperature: float = 0.07) -> Tuple[torch.Tensor, torch.Tensor]:
    target_model = model.base_model if hasattr(model, "base_model") else model

    scale: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None

    if hasattr(target_model, "logit_scale"):
        ls = getattr(target_model, "logit_scale")
        scale = ls.to(device).exp() if isinstance(ls, torch.Tensor) else torch.tensor(float(ls), device=device).exp()

    if hasattr(target_model, "logit_bias"):
        lb = getattr(target_model, "logit_bias")
        bias = lb.to(device) if isinstance(lb, torch.Tensor) else torch.tensor(float(lb), device=device)

    if scale is None:
        scale = torch.tensor(1.0 / float(temperature), device=device)
    if bias is None:
        bias = torch.tensor(-10.0, device=device)

    scale = torch.clamp(scale, max=100.0)
    return scale, bias


# =========================
# Losses
# =========================

def compute_symmetric_multipos_sigmoid_loss(
    logits: torch.Tensor,
    img_labels: torch.Tensor,
    bank_labels: torch.Tensor,
    evidence_bank_idx: torch.Tensor,     # (B, L)，-1 代表該 slot 無 evidence
    evidence_alpha: float,
    t2i_lambda: float,
    t2i_skip_no_positive: bool,
    t2i_label_level_mean: bool,
    label_smoothing: float,
    class_weights: Optional[torch.Tensor] = None,  # [num_cat] indexed by cat_id; weights i2t per-sample term
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, M = logits.shape
    device = logits.device
    dtype = logits.dtype

    pos_mask = bank_labels.view(1, M) == img_labels.view(B, 1)
    targets = pos_mask.to(dtype) * (1.0 - float(label_smoothing)) + 0.5 * float(label_smoothing)
    neg_mask = ~pos_mask

    weight = torch.ones((B, M), device=device, dtype=dtype)
    if evidence_alpha is not None and float(evidence_alpha) > 1.0:
        if evidence_bank_idx.dim() != 2:
            raise ValueError(
                f"evidence_bank_idx must be 2D (B, L), got shape={tuple(evidence_bank_idx.shape)}"
            )
        L_ev = evidence_bank_idx.size(1)
        for l in range(L_ev):
            col = evidence_bank_idx[:, l]
            valid = col >= 0
            if valid.any():
                rows = torch.arange(B, device=device)[valid]
                cols = torch.clamp(col[valid], 0, M - 1)
                ok = pos_mask[rows, cols]
                if ok.any():
                    weight[rows[ok], cols[ok]] = float(evidence_alpha)

    element = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    pos_w = weight * pos_mask.to(dtype)
    pos_den_i = pos_w.sum(dim=1).clamp_min(eps)
    pos_loss_i = (element * pos_w).sum(dim=1) / pos_den_i

    neg_den_i = neg_mask.sum(dim=1).to(dtype).clamp_min(1.0)
    neg_loss_i = (element * neg_mask.to(dtype)).sum(dim=1) / neg_den_i
    per_sample_i2t = 0.5 * (pos_loss_i + neg_loss_i)
    if class_weights is not None:
        # Class-balanced weighted mean over the image->text direction (the
        # classification direction the confusion matrix measures). t2i is left
        # unweighted: its bank is fixed per batch and t2i_label_level_mean
        # already removes per-caption-count imbalance.
        w_i = class_weights[img_labels].to(dtype)
        i2t = (per_sample_i2t * w_i).sum() / w_i.sum().clamp_min(eps)
    else:
        i2t = per_sample_i2t.mean()

    pos_den_j = pos_w.sum(dim=0)
    pos_loss_j = (element * pos_w).sum(dim=0) / pos_den_j.clamp_min(eps)
    neg_den_j = neg_mask.sum(dim=0).to(dtype).clamp_min(1.0)
    neg_loss_j = (element * neg_mask.to(dtype)).sum(dim=0) / neg_den_j
    t2i_per_caption = 0.5 * (pos_loss_j + neg_loss_j)

    batch_labels = torch.unique(img_labels)
    in_batch = (bank_labels.view(1, M) == batch_labels.view(-1, 1)).any(dim=0)

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
            if len(per_label_losses) == 0:
                t2i = torch.zeros((), device=device, dtype=dtype)
            else:
                t2i = torch.stack(per_label_losses, dim=0).mean()
        else:
            t2i = t2i_per_caption.mean()

    total = i2t + float(t2i_lambda) * t2i
    return total, i2t, t2i



def compute_effective_number_weights(
    labels: List[int],
    beta: float,
    device: str,
) -> Tuple[torch.Tensor, Dict[int, int]]:
    """Effective-number class weights (Cui et al. 2019), indexed by cat_id.

    w_c = (1 - beta) / (1 - beta^n_c), normalized so the mean weight over present
    classes is 1.0 (keeps overall i2t loss scale unchanged). Returns a tensor of
    size (max_cat_id + 1) with weight 1.0 for absent cat_ids, plus the raw counts.
    """
    from collections import Counter

    counts = Counter(int(l) for l in labels)
    if not counts:
        raise ValueError("compute_effective_number_weights: empty label list")

    eff: Dict[int, float] = {
        c: (1.0 - beta) / (1.0 - beta ** n) if n > 0 else 1.0
        for c, n in counts.items()
    }
    mean_eff = sum(eff.values()) / len(eff)

    w = torch.ones(max(counts) + 1, dtype=torch.float32)
    for c, e in eff.items():
        w[c] = e / mean_eff
    return w.to(device), dict(counts)


def compute_cross_lesion_loss(
    img_feats: torch.Tensor,
    txt_bank_feats: torch.Tensor,
    img_labels: torch.Tensor,
    image_ids: torch.Tensor,
    cat2bank_indices: Dict[int, List[int]],
    temperature: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, int]:
    """Within-image supervised contrastive loss for VLM-Lesion.

    For each image containing >=2 lesions of >=2 distinct categories, push
    each lesion's image feature toward captions of its own category and
    away from captions of other in-image categories. Per-sample text
    representation = mean of the sample's category caption-bank embeddings
    (then L2-renormalized).

    Returns (loss, num_eligible_groups). When no eligible group exists in
    the batch, returns (0-tensor, 0).
    """
    device = img_feats.device
    dtype = img_feats.dtype
    B = img_feats.size(0)
    zero = torch.zeros((), device=device, dtype=dtype)
    if B < 2:
        return zero, 0

    sample_txt = torch.zeros_like(img_feats)
    for i, y in enumerate(img_labels.tolist()):
        idxs = cat2bank_indices.get(int(y))
        if not idxs:
            continue
        sel = torch.as_tensor(idxs, device=device, dtype=torch.long)
        sample_txt[i] = txt_bank_feats.index_select(0, sel).mean(dim=0)
    sample_txt = F.normalize(sample_txt, dim=-1)

    unique_ids, inverse = torch.unique(image_ids, return_inverse=True)
    inv_temp = 1.0 / max(float(temperature), eps)
    losses: List[torch.Tensor] = []

    for g in range(int(unique_ids.size(0))):
        idx = (inverse == g).nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        y_g = img_labels[idx]
        if torch.unique(y_g).numel() < 2:
            continue

        f = img_feats[idx]
        t = sample_txt[idx]
        sim = (f @ t.t()) * inv_temp                          # [N, N]

        pos_mask = (y_g.unsqueeze(0) == y_g.unsqueeze(1)).to(dtype)
        pos_count_row = pos_mask.sum(dim=1).clamp_min(eps)
        pos_count_col = pos_mask.sum(dim=0).clamp_min(eps)

        logp_i2t = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        l_i2t = -(logp_i2t * pos_mask).sum(dim=1) / pos_count_row

        sim_T = sim.t()
        logp_t2i = sim_T - torch.logsumexp(sim_T, dim=1, keepdim=True)
        l_t2i = -(logp_t2i * pos_mask).sum(dim=1) / pos_count_col

        losses.append(0.5 * (l_i2t + l_t2i).mean())

    if not losses:
        return zero, 0
    return torch.stack(losses).mean(), len(losses)



def compute_paired_sigmoid_loss(
    logits: torch.Tensor,
    label_smoothing: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.dim() != 2 or logits.size(0) != logits.size(1):
        raise ValueError(f"paired sigmoid loss expects square logits, got shape={tuple(logits.shape)}")

    B = logits.size(0)
    dtype = logits.dtype
    device = logits.device

    pos_mask = torch.eye(B, device=device, dtype=torch.bool)
    targets = pos_mask.to(dtype) * (1.0 - float(label_smoothing)) + 0.5 * float(label_smoothing)
    neg_mask = ~pos_mask

    element = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    pos_loss_i = (element * pos_mask.to(dtype)).sum(dim=1)
    neg_loss_i = (element * neg_mask.to(dtype)).sum(dim=1) / max(1, B - 1)
    i2t = (0.5 * (pos_loss_i + neg_loss_i)).mean()

    pos_loss_j = (element * pos_mask.to(dtype)).sum(dim=0)
    neg_loss_j = (element * neg_mask.to(dtype)).sum(dim=0) / max(1, B - 1)
    t2i = (0.5 * (pos_loss_j + neg_loss_j)).mean()

    total = 0.5 * (i2t + t2i)
    return total, i2t, t2i


# =========================
# LSCFT — Lesion-Selective Counterfactual Faithfulness Training
# =========================

def _lscft_center_bbox(H: int, W: int, size: int, jitter_px: int) -> Tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) for a size×size region centered (with random jitter)."""
    cy = H // 2 + random.randint(-jitter_px, jitter_px)
    cx = W // 2 + random.randint(-jitter_px, jitter_px)
    y0 = max(0, cy - size // 2)
    y1 = min(H, y0 + size)
    y0 = max(0, y1 - size)
    x0 = max(0, cx - size // 2)
    x1 = min(W, x0 + size)
    x0 = max(0, x1 - size)
    return y0, y1, x0, x1


def lscft_mask_mean(pv: torch.Tensor, mask_size: int, jitter_px: int) -> torch.Tensor:
    """Replace center with per-image mean pixel value (lesion content removed)."""
    B, C, H, W = pv.shape
    out = pv.clone()
    mean_per_img = pv.mean(dim=(2, 3), keepdim=True)
    y0, y1, x0, x1 = _lscft_center_bbox(H, W, mask_size, jitter_px)
    out[:, :, y0:y1, x0:x1] = mean_per_img.expand(-1, -1, y1 - y0, x1 - x0)
    return out


def lscft_mask_cutmix(pv: torch.Tensor, mask_size: int, jitter_px: int) -> torch.Tensor:
    """Replace center with a periphery patch from another image in the batch.

    Source patch comes from the top-left corner of a permuted sample, which is
    typically background/healthy tissue (lesion is centered in the crop).
    """
    B, C, H, W = pv.shape
    out = pv.clone()
    y0, y1, x0, x1 = _lscft_center_bbox(H, W, mask_size, jitter_px)
    sy = min(H, y1 - y0)
    sx = min(W, x1 - x0)
    perm = torch.randperm(B, device=pv.device)
    src = pv[perm, :, :sy, :sx]
    out[:, :, y0:y1, x0:x1] = src
    return out


def lscft_mask_noise(pv: torch.Tensor, mask_size: int, jitter_px: int) -> torch.Tensor:
    """Replace center with N(0, 1) noise (matches normalized image statistics)."""
    B, C, H, W = pv.shape
    out = pv.clone()
    y0, y1, x0, x1 = _lscft_center_bbox(H, W, mask_size, jitter_px)
    noise = torch.randn(B, C, y1 - y0, x1 - x0, device=pv.device, dtype=pv.dtype)
    out[:, :, y0:y1, x0:x1] = noise
    return out


def lscft_peripheral_mask(pv: torch.Tensor, keep_size: int) -> torch.Tensor:
    """Keep center keep_size×keep_size; fill periphery with per-image mean."""
    B, C, H, W = pv.shape
    mean_per_img = pv.mean(dim=(2, 3), keepdim=True)
    out = mean_per_img.expand_as(pv).clone()
    cy, cx = H // 2, W // 2
    y0 = max(0, cy - keep_size // 2)
    y1 = min(H, y0 + keep_size)
    x0 = max(0, cx - keep_size // 2)
    x1 = min(W, x0 + keep_size)
    out[:, :, y0:y1, x0:x1] = pv[:, :, y0:y1, x0:x1]
    return out


def lscft_color_jitter(pv: torch.Tensor, strength: float) -> torch.Tensor:
    """Per-channel multiplicative jitter (approximates hue/saturation in
    normalized tensor space). Whole-image perturbation — lesion content
    structurally preserved, only color shifted."""
    B, C, _, _ = pv.shape
    scale = 1.0 + (torch.rand(B, C, 1, 1, device=pv.device, dtype=pv.dtype) - 0.5) * 2.0 * float(strength)
    return pv * scale


def lscft_hflip(pv: torch.Tensor) -> torch.Tensor:
    return torch.flip(pv, dims=[-1])


_LSCFT_POSITIVE_BANK = ("mean", "cutmix", "noise")
_LSCFT_NEGATIVE_BANK = ("peripheral", "jitter", "hflip")


def lscft_apply_positive(pv: torch.Tensor, mask_size: int, jitter_px: int) -> Tuple[torch.Tensor, str]:
    kind = random.choice(_LSCFT_POSITIVE_BANK)
    if kind == "mean":
        return lscft_mask_mean(pv, mask_size, jitter_px), kind
    if kind == "cutmix":
        return lscft_mask_cutmix(pv, mask_size, jitter_px), kind
    return lscft_mask_noise(pv, mask_size, jitter_px), kind


def lscft_apply_negative(pv: torch.Tensor, keep_size: int, jitter_strength: float) -> Tuple[torch.Tensor, str]:
    kind = random.choice(_LSCFT_NEGATIVE_BANK)
    if kind == "peripheral":
        return lscft_peripheral_mask(pv, keep_size), kind
    if kind == "jitter":
        return lscft_color_jitter(pv, jitter_strength), kind
    return lscft_hflip(pv), kind


def compute_lscft_loss(
    feat_orig: torch.Tensor,
    feat_pos: torch.Tensor,
    feat_neg: torch.Tensor,
    txt_feat: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Ranking softplus loss.

    All inputs must be L2-normalized rows. Computes:

        drop_pos = cos(orig, t) - cos(pos, t)   (should be LARGE — lesion absent)
        drop_neg = cos(orig, t) - cos(neg, t)   (should be SMALL — only periphery)
        L = softplus(margin - (drop_pos - drop_neg))

    Margin in cosine space (e.g., 0.1). softplus over hinge avoids dead
    gradients near the margin.
    """
    sim_orig = (feat_orig * txt_feat).sum(dim=-1)
    sim_pos = (feat_pos * txt_feat).sum(dim=-1)
    sim_neg = (feat_neg * txt_feat).sum(dim=-1)
    drop_pos = sim_orig - sim_pos
    drop_neg = sim_orig - sim_neg
    return F.softplus(float(margin) - (drop_pos - drop_neg)).mean()


# =========================
# Runtime helpers
# =========================

def infer_output_dir(config: Dict[str, Any]) -> str:
    model_stub = str(config["model_name"]).split("/")[-1].replace("-", "_")
    if config["multipos"] and config["fusion"]:
        suffix = "multipos_fusion"
    elif config["multipos"]:
        suffix = "multipos"
    elif config["fusion"]:
        suffix = "fusion"
    else:
        suffix = "finetuned"

    target = config.get("target", "lesion")
    if target != "lesion":
        suffix = f"{target}_{suffix}"

    text_mode = normalize_text_mode(config.get("text_mode", "captions"))
    if target == "lesion" and text_mode != "captions":
        suffix = f"{suffix}_{text_mode}"

    langs = config.get("langs", ["en"])
    if isinstance(langs, (list, tuple)) and len(langs) > 0:
        lang_tag = "_".join(sorted(langs))
        suffix = f"{suffix}_{lang_tag}"

    return f"./outputs/{model_stub}_{suffix}"



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified training entry for baseline / multi-positive / fusion modes")
    parser.add_argument(
        "--target",
        type=str,
        choices=["lesion", "overall"],
        default="lesion",
        help="lesion: bbox crop + symptoms.json captions (existing). overall: whole image + image['overall'].colloquial_zh/medical_zh",
    )
    parser.add_argument("--multipos", action="store_true", help="use multi-positive caption-bank training")
    parser.add_argument("--fusion", action="store_true", help="use local-global image fusion")
    parser.add_argument("--freeze_text_encoder", action="store_true", help="freeze the text encoder/text projection during training")
    parser.add_argument(
        "--cross_lesion",
        action="store_true",
        help="enable within-image cross-lesion supervised contrastive aux loss (multipos+target=lesion)",
    )
    parser.add_argument("--cross_alpha", type=float, default=None, help="weight on cross-lesion aux loss")
    parser.add_argument("--cross_tau", type=float, default=None, help="softmax temperature for cross-lesion loss")
    parser.add_argument(
        "--class_balanced_loss",
        action="store_true",
        help="weight the i2t loss term by effective-number class weights (lifts macro f1 on rare lesion categories)",
    )
    parser.add_argument("--cb_beta", type=float, default=None, help="effective-number beta for --class_balanced_loss (default 0.999)")
    parser.add_argument(
        "--balanced_sampler",
        action="store_true",
        help="oversample rare lesion categories via inverse-frequency WeightedRandomSampler (mutually exclusive with --grouped_sampler)",
    )
    parser.add_argument(
        "--grouped_sampler",
        action="store_true",
        help="enable hybrid grouped batch sampler (raises cross-lesion firing rate to ~100%)",
    )
    parser.add_argument(
        "--grouped_ratio",
        type=float,
        default=None,
        help="fraction of each batch filled by multi-category image groups (default 0.5)",
    )
    parser.add_argument(
        "--lscft",
        action="store_true",
        help="enable Lesion-Selective Counterfactual Faithfulness Training (multipos+fusion+target=lesion)",
    )
    parser.add_argument("--lscft_alpha", type=float, default=None, help="weight on LSCFT ranking loss")
    parser.add_argument("--lscft_margin", type=float, default=None, help="LSCFT ranking margin (cosine space)")
    parser.add_argument("--lscft_pos_size", type=int, default=None, help="positive-arm center mask size (pixels)")
    parser.add_argument("--lscft_neg_keep", type=int, default=None, help="negative-arm center-keep size (pixels)")
    parser.add_argument("--lscft_mask_jitter", type=int, default=None, help="spatial jitter for mask center (pixels)")
    parser.add_argument("--lscft_jitter", type=float, default=None, help="negative-arm color jitter strength")

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--symptoms_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--crop_mode", type=str, choices=["bbox", "square"], default=None)

    parser.add_argument("--lang", type=str, choices=["en", "zh", "both"], default=None)
    parser.add_argument(
        "--text_mode",
        type=str,
        choices=["captions", "class_name", "captions_plus_class_name"],
        default=None,
        help="lesion text source: generated captions/descriptions, direct label names, or both",
    )
    parser.add_argument("--warmup_ratio", type=float, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--grad_clip_norm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dropout_prob", type=float, default=None)
    parser.add_argument("--fusion_base_lr", type=float, default=None)
    parser.add_argument("--fusion_head_lr", type=float, default=None)
    parser.add_argument("--fusion_gate_mode", type=str, choices=["scalar", "film", "xattn"], default=None,
                        help="fusion gate: scalar (production), film (input-conditioned per-channel gate), "
                             "or xattn (lesion cross-attends over whole-fish patch tokens)")

    parser.add_argument("--evidence_alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--t2i_lambda", type=float, default=None)

    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--valid_split", type=str, default=None)
    parser.add_argument("--test_split", type=str, default=None)

    parser.add_argument("--no_amp", action="store_true", help="disable AMP")
    parser.add_argument("--no_pin_memory", action="store_true", help="disable pin_memory")
    parser.add_argument("--eval_test_after_train", action="store_true")
    return parser



def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config["target"] = str(getattr(args, "target", "lesion") or "lesion")
    config["multipos"] = bool(args.multipos)
    config["fusion"] = bool(args.fusion)
    if getattr(args, "text_mode", None) is not None:
        config["text_mode"] = normalize_text_mode(args.text_mode)
    else:
        config["text_mode"] = normalize_text_mode(config.get("text_mode", "captions"))

    if config["target"] == "overall" and config["fusion"]:
        raise ValueError("--target overall is incompatible with --fusion (no local/global crops in overall mode)")
    if config["target"] == "overall" and config["text_mode"] != "captions":
        raise ValueError("--text_mode is only supported for --target lesion")

    for key in [
        "model_name",
        "data_root",
        "symptoms_file",
        "output_dir",
        "crop_mode",
        "batch_size",
        "num_epochs",
        "learning_rate",
        "weight_decay",
        "max_length",
        "num_workers",
        "grad_clip_norm",
        "seed",
        "dropout_prob",
        "fusion_base_lr",
        "fusion_head_lr",
        "evidence_alpha",
        "temperature",
        "label_smoothing",
        "t2i_lambda",
        "train_split",
        "valid_split",
        "test_split",
        "warmup_ratio",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    if args.lang is not None:
        if args.lang == "both":
            config["langs"] = ["en", "zh"]
        else:
            config["langs"] = [args.lang]

    if config["target"] == "overall":
        # overall captions are zh-only; lock langs for output-dir naming and downstream prints
        config["langs"] = ["zh"]

    if args.no_amp:
        config["use_amp"] = False
    if args.no_pin_memory:
        config["pin_memory"] = False
    if args.eval_test_after_train:
        config["eval_test_after_train"] = True
    if args.freeze_text_encoder:
        config["freeze_text_encoder"] = True

    if args.cross_lesion:
        config["cross_lesion"] = True
    for key in ("cross_alpha", "cross_tau"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    if config.get("cross_lesion", False):
        if config["target"] != "lesion" or not config["multipos"]:
            raise ValueError("--cross_lesion requires --target lesion and --multipos")

    if args.class_balanced_loss:
        config["class_balanced_loss"] = True
    if getattr(args, "cb_beta", None) is not None:
        config["cb_beta"] = args.cb_beta
    if args.balanced_sampler:
        config["balanced_sampler"] = True

    for key in ("class_balanced_loss", "balanced_sampler"):
        if config.get(key, False) and (config["target"] != "lesion" or not config["multipos"]):
            raise ValueError(f"--{key} requires --target lesion and --multipos")

    if args.grouped_sampler:
        config["grouped_sampler"] = True
    if getattr(args, "grouped_ratio", None) is not None:
        config["grouped_ratio"] = args.grouped_ratio

    if config.get("grouped_sampler", False):
        if config["target"] != "lesion" or not config["multipos"]:
            raise ValueError("--grouped_sampler requires --target lesion and --multipos")
        if config.get("balanced_sampler", False):
            raise ValueError("--balanced_sampler and --grouped_sampler are mutually exclusive (both override the batch composition)")

    if args.lscft:
        config["lscft"] = True
    for key in (
        "lscft_alpha",
        "lscft_margin",
        "lscft_pos_size",
        "lscft_neg_keep",
        "lscft_mask_jitter",
        "lscft_jitter",
    ):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    if config.get("lscft", False):
        if config["target"] != "lesion" or not config["multipos"] or not config["fusion"]:
            raise ValueError("--lscft requires --target lesion, --multipos, and --fusion")

    if getattr(args, "fusion_gate_mode", None) is not None:
        config["fusion_gate_mode"] = args.fusion_gate_mode
    if config.get("fusion_gate_mode", "scalar") in ("film", "xattn") and not config["fusion"]:
        raise ValueError(f"--fusion_gate_mode {config['fusion_gate_mode']} requires --fusion")

    if config["weight_decay"] is None:
        config["weight_decay"] = 0.01 if (config["multipos"] or config["fusion"]) else 0.0

    if not config["output_dir"]:
        config["output_dir"] = infer_output_dir(config)

    return config



def create_processor_and_model(config: Dict[str, Any]):
    device = config["device"]
    processor = AutoProcessor.from_pretrained(config["model_name"])
    base_model = AutoModel.from_pretrained(config["model_name"]).to(device)

    if not config["fusion"]:
        return processor, base_model

    dummy_image = Image.new("RGB", (224, 224), (0, 0, 0))
    dummy_pixel = processor(images=[dummy_image], return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        hidden_size = get_image_features(base_model, dummy_pixel).shape[-1]
    print(f"Detected Model Hidden Size: {hidden_size}")

    model = LocalGlobalFusionWrapper(
        base_model=base_model,
        hidden_size=hidden_size,
        dropout_prob=float(config["dropout_prob"]),
        gate_mode=str(config.get("fusion_gate_mode", "scalar")),
    ).to(device)
    return processor, model


def _unwrap_trainable_base_model(model):
    """Return the underlying dual-encoder model.

    LocalGlobalFusionWrapper owns a Hugging Face dual-encoder under `base_model`.
    For plain Hugging Face CLIP/SigLIP/SigLIP2 models, keep the model itself.
    """
    if isinstance(model, LocalGlobalFusionWrapper):
        return model.base_model
    return model


def _set_requires_grad_unique(params, requires_grad: bool) -> int:
    seen = set()
    n_params = 0
    for p in params:
        if p is None or id(p) in seen:
            continue
        seen.add(id(p))
        p.requires_grad = requires_grad
        n_params += int(p.numel())
    return n_params


def _collect_attr_parameters(obj, attr_name: str) -> List[torch.nn.Parameter]:
    if not hasattr(obj, attr_name):
        return []

    attr = getattr(obj, attr_name)
    if attr is None:
        return []
    if isinstance(attr, nn.Module):
        return list(attr.parameters())
    if isinstance(attr, nn.Parameter):
        return [attr]
    if isinstance(attr, torch.Tensor) and getattr(attr, "requires_grad", False):
        return [attr]
    return []


def freeze_text_encoder(model) -> int:
    """Freeze the text side of a CLIP/SigLIP-style dual encoder.

    This freezes common text-module names and text projection parameters.
    `logit_scale` / `logit_bias` are intentionally left trainable because they are
    global calibration parameters rather than text-encoder weights.
    """
    base = _unwrap_trainable_base_model(model)

    text_param_candidates: List[torch.nn.Parameter] = []
    for attr_name in [
        "text_model",
        "text_encoder",
        "language_model",
        "text_projection",
        "text_proj",
        "text_embedder",
    ]:
        text_param_candidates.extend(_collect_attr_parameters(base, attr_name))

    # Fallback: catch named parameters that clearly belong to the text branch.
    # This helps if a future model uses a slightly different module layout.
    for name, p in base.named_parameters():
        name_l = name.lower()
        if (
            name_l.startswith("text_")
            or ".text_" in name_l
            or name_l.startswith("text.")
            or ".text." in name_l
            or "text_model" in name_l
            or "text_encoder" in name_l
        ):
            text_param_candidates.append(p)

    return _set_requires_grad_unique(text_param_candidates, False)


def count_trainable_parameters(model) -> Tuple[int, int]:
    total = sum(int(p.numel()) for p in model.parameters())
    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    return trainable, total


def apply_freeze_config(model, config: Dict[str, Any]):
    if bool(config.get("freeze_text_encoder", False)):
        frozen = freeze_text_encoder(model)
        if frozen == 0:
            print("[Freeze] freeze_text_encoder=True, but no text-encoder parameters were matched.")
        else:
            print(f"[Freeze] text encoder/text projection frozen: {frozen:,} parameters")

    trainable, total = count_trainable_parameters(model)
    pct = 100.0 * trainable / max(1, total)
    print(f"[Params] trainable={trainable:,} / total={total:,} ({pct:.2f}%)")


def trainable_parameters(module) -> List[torch.nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]



def build_collate_fn(
    processor,
    config: Dict[str, Any],
    use_multipos: bool,
    use_fusion: bool,
    cat2bank_indices: Optional[Dict[int, List[int]]] = None,
    cat2evidence: Optional[Dict[int, Dict[int, List[int]]]] = None,
    evidence_slots: int = 2,
    target: str = "lesion",
):
    def collate_fn(items):
        if target == "overall":
            images = [x["image"] for x in items]
            img_batch = processor(images=images, return_tensors="pt")
            if "pixel_values" not in img_batch:
                raise RuntimeError("Processor did not return 'pixel_values' for images")

            if use_multipos:
                B = len(items)
                # interleaved bank: [colloquial_0, medical_0, colloquial_1, medical_1, ...]
                texts: List[str] = []
                bank_labels: List[int] = []
                for i, x in enumerate(items):
                    texts.append(str(x["caption_colloquial"]))
                    bank_labels.append(i)
                    texts.append(str(x["caption_medical"]))
                    bank_labels.append(i)

                txt_batch = processor(
                    text=texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=config["max_length"],
                )
                out: Dict[str, torch.Tensor] = {
                    "pixel_values": img_batch["pixel_values"],
                    "input_ids": txt_batch["input_ids"],
                    "labels": torch.arange(B, dtype=torch.long),
                    "bank_labels": torch.tensor(bank_labels, dtype=torch.long),
                    # evidence_alpha is disabled for overall, but the loss expects a 2D tensor
                    "evidence_bank_idx": torch.full((B, 1), -1, dtype=torch.long),
                }
                if "attention_mask" in txt_batch:
                    out["attention_mask"] = txt_batch["attention_mask"]
                return out

            # baseline overall: paired (image, one caption)
            texts = [str(x["text"]) for x in items]
            return processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=config["max_length"],
            )

        if use_multipos:
            if cat2evidence is None:
                raise ValueError("multipos collate 需要 cat2evidence（由 build_caption_bank 產生）")

            labels = torch.tensor([int(x["label_id"]) for x in items], dtype=torch.long)
            evidence_idx = [int(x["evidence_index"]) for x in items]

            L = int(max(1, evidence_slots))
            evidence_bank_idx = torch.full((len(items), L), -1, dtype=torch.long)

            for i, (y, ei) in enumerate(zip(labels.tolist(), evidence_idx)):
                if ei is None or int(ei) < 0:
                    continue
                ev_map = cat2evidence.get(int(y))
                if not ev_map:
                    continue
                positions = ev_map.get(int(ei))
                if not positions:
                    continue
                for j, pos in enumerate(positions[:L]):
                    evidence_bank_idx[i, j] = int(pos)

            image_ids = torch.tensor(
                [int(x.get("image_id", -1)) for x in items], dtype=torch.long
            )

            if use_fusion:
                images_local = [x["image_local"] for x in items]
                images_global = [x["image_global"] for x in items]
                batch_local = processor(images=images_local, return_tensors="pt")
                batch_global = processor(images=images_global, return_tensors="pt")
                return {
                    "pixel_values_local": batch_local["pixel_values"],
                    "pixel_values_global": batch_global["pixel_values"],
                    "labels": labels,
                    "evidence_bank_idx": evidence_bank_idx,
                    "image_ids": image_ids,
                }
            else:
                images = [x["image"] for x in items]
                img_batch = processor(images=images, return_tensors="pt")
                if "pixel_values" not in img_batch:
                    raise RuntimeError("Processor did not return 'pixel_values' for images")
                return {
                    "pixel_values": img_batch["pixel_values"],
                    "labels": labels,
                    "evidence_bank_idx": evidence_bank_idx,
                    "image_ids": image_ids,
                }

        # baseline paired text supervision
        texts = [x["text"] for x in items]
        if use_fusion:
            images_local = [x["image_local"] for x in items]
            images_global = [x["image_global"] for x in items]
            batch_local = processor(images=images_local, return_tensors="pt")
            batch_global = processor(images=images_global, return_tensors="pt")
            text_batch = processor(
                text=texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=config["max_length"],
            )
            out = {
                "pixel_values_local": batch_local["pixel_values"],
                "pixel_values_global": batch_global["pixel_values"],
                "input_ids": text_batch["input_ids"],
            }
            if "attention_mask" in text_batch:
                out["attention_mask"] = text_batch["attention_mask"]
            return out

        images = [x["image"] for x in items]
        return processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=config["max_length"],
        )

    return collate_fn



def build_optimizer(model, config: Dict[str, Any]):
    if config["fusion"]:
        param_groups = []

        base_params = trainable_parameters(model.base_model)
        if base_params:
            param_groups.append({"params": base_params, "lr": float(config["fusion_base_lr"])})

        fusion_head_params = trainable_parameters(model.fusion_linear)
        if hasattr(model, "cross_attn"):
            fusion_head_params += trainable_parameters(model.cross_attn)
        if fusion_head_params:
            param_groups.append({"params": fusion_head_params, "lr": float(config["fusion_head_lr"])})

        gate_params: List[torch.nn.Parameter] = []
        if hasattr(model, "gate") and model.gate.requires_grad:
            gate_params.append(model.gate)
        if hasattr(model, "gate_net"):
            gate_params += [p for p in model.gate_net.parameters() if p.requires_grad]
        if gate_params:
            param_groups.append({"params": gate_params, "lr": float(config["fusion_head_lr"])})

        if not param_groups:
            raise RuntimeError("No trainable parameters found. Check freeze settings.")

        return torch.optim.AdamW(
            param_groups,
            weight_decay=float(config["weight_decay"]),
        )

    params = trainable_parameters(model)
    if not params:
        raise RuntimeError("No trainable parameters found. Check freeze settings.")

    return torch.optim.AdamW(
        params,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )



def save_best_checkpoint(model, processor, config: Dict[str, Any], best_metric: float):
    os.makedirs(config["output_dir"], exist_ok=True)

    if config["fusion"]:
        model.base_model.save_pretrained(config["output_dir"])
        wrapper_state_path = os.path.join(config["output_dir"], "wrapper_state.pt")
        torch.save(model.state_dict(), wrapper_state_path)
    else:
        model.save_pretrained(config["output_dir"])

    processor.save_pretrained(config["output_dir"])

    with open(os.path.join(config["output_dir"], "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"  -> Saved best to {config['output_dir']} (valid_loss={best_metric:.4f})")



def plot_losses(train_losses: List[float], valid_losses: List[float], output_dir: str):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="train")
    plt.plot(valid_losses, label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()
    print("Done. loss_curve.png saved.")



def inspect_fusion_state(model, valid_loader, device: str):
    if not hasattr(model, "forward_image"):
        return
    sample_batch = next(iter(valid_loader), None)
    if sample_batch is None:
        return

    pixel_values_local = sample_batch["pixel_values_local"].to(device, non_blocking=True)
    pixel_values_global = sample_batch["pixel_values_global"].to(device, non_blocking=True)

    model.eval()
    with torch.no_grad():
        _, local_feat, global_feat, fused = model.forward_image(
            pixel_values_local,
            pixel_values_global,
            return_parts=True,
        )
        local_norm = float(local_feat.norm(dim=-1).mean().item())
        global_norm = float(global_feat.norm(dim=-1).mean().item())
        fused_norm = float(fused.norm(dim=-1).mean().item())
        if getattr(model, "gate_mode", "scalar") == "film":
            g = torch.sigmoid(model.gate_net(torch.cat([local_feat, global_feat], dim=-1)))  # [B, H]
            gate_value = float(g.mean().item())  # mean per-channel gate
            effective_ratio = float((g * fused).norm(dim=-1).mean().item() / (local_norm + 1e-8))
        else:
            gate_value = float(model.gate.detach().cpu().item())
            effective_ratio = abs(gate_value) * fused_norm / (local_norm + 1e-8)

    print(
        f"[Inspect] gate={gate_value:.4f}, "
        f"local_norm={local_norm:.4f}, "
        f"global_norm={global_norm:.4f}, "
        f"fused_norm={fused_norm:.4f}, "
        f"effective_ratio={effective_ratio:.4f}"
    )


# =========================
# Baseline train / eval
# =========================

def run_eval_baseline(model, dataloader, device: str, use_amp: bool) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                try:
                    out = model(**batch, return_loss=True)
                except TypeError:
                    out = model(**batch)
                if not hasattr(out, "loss") or out.loss is None:
                    raise RuntimeError("Model did not return loss.")
                loss = out.loss
            bs = batch["pixel_values"].size(0)
            total += float(loss.item()) * bs
            n += bs
    return total / max(1, n)



def train_baseline(config: Dict[str, Any]):
    set_seed(int(config["seed"]))
    device = config["device"]
    os.makedirs(config["output_dir"], exist_ok=True)

    target = config.get("target", "lesion")

    if target == "overall":
        captions_by_cat = None
        train_ds = CocoOverallDataset(
            data_root=config["data_root"],
            split=config["train_split"],
            use_multipos=False,
        )
        valid_ds = CocoOverallDataset(
            data_root=config["data_root"],
            split=config["valid_split"],
            use_multipos=False,
        )
    else:
        langs = tuple(config.get("langs", ["en"]))
        captions_by_cat = load_flat_caption_bank(
            config["symptoms_file"],
            langs=langs,
            text_mode=config["text_mode"],
        ).raw_by_cat
        train_ds = CocoCropDataset(
            data_root=config["data_root"],
            split=config["train_split"],
            crop_mode=config["crop_mode"],
            captions_by_cat=captions_by_cat,
            use_multipos=False,
            use_fusion=False,
            use_annotation_text=(config["text_mode"] == "captions"),
            text_mode=config["text_mode"],
        )
        valid_ds = CocoCropDataset(
            data_root=config["data_root"],
            split=config["valid_split"],
            crop_mode=config["crop_mode"],
            captions_by_cat=captions_by_cat,
            use_multipos=False,
            use_fusion=False,
            use_annotation_text=(config["text_mode"] == "captions"),
            text_mode=config["text_mode"],
        )

    processor, model = create_processor_and_model(config)
    apply_freeze_config(model, config)
    collate_fn = build_collate_fn(processor, config, use_multipos=False, use_fusion=False, target=target)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"]) and device == "cuda",
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"]) and device == "cuda",
        collate_fn=collate_fn,
    )

    optimizer = build_optimizer(model, config)
    total_steps = len(train_loader) * int(config["num_epochs"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(config["warmup_ratio"])),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(bool(config["use_amp"]) and device == "cuda"))

    train_losses: List[float] = []
    valid_losses: List[float] = []
    best_val = float("inf")

    for epoch in range(int(config["num_epochs"])):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(bool(config["use_amp"]) and device == "cuda")):
                try:
                    out = model(**batch, return_loss=True)
                except TypeError:
                    out = model(**batch)
                if not hasattr(out, "loss") or out.loss is None:
                    raise RuntimeError("Model did not return loss.")
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if config["grad_clip_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip_norm"]))

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += float(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running / max(1, len(train_loader))
        val_loss = run_eval_baseline(model, valid_loader, device, bool(config["use_amp"]))

        train_losses.append(train_loss)
        valid_losses.append(val_loss)

        print(f"Epoch {epoch+1}: train={train_loss:.4f}, valid={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_best_checkpoint(model, processor, config, best_val)

    plot_losses(train_losses, valid_losses, config["output_dir"])

    if config.get("eval_test_after_train", False):
        if target == "overall":
            test_ds = CocoOverallDataset(
                data_root=config["data_root"],
                split=config["test_split"],
                use_multipos=False,
            )
        else:
            test_ds = CocoCropDataset(
                data_root=config["data_root"],
                split=config["test_split"],
                crop_mode=config["crop_mode"],
                captions_by_cat=captions_by_cat,
                use_multipos=False,
                use_fusion=False,
                use_annotation_text=(config["text_mode"] == "captions"),
                text_mode=config["text_mode"],
            )
        test_loader = DataLoader(
            test_ds,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            pin_memory=bool(config["pin_memory"]) and device == "cuda",
            collate_fn=collate_fn,
        )
        test_loss = run_eval_baseline(model, test_loader, device, bool(config["use_amp"]))
        print(f"Test loss: {test_loss:.4f}")


# =========================
# Custom train / eval (multipos and/or fusion)
# =========================

def prepare_custom_supervision(config: Dict[str, Any], processor):
    langs = tuple(config.get("langs", ["en"]))
    captions_by_cat = load_flat_caption_bank(
        config["symptoms_file"],
        langs=langs,
        text_mode=config["text_mode"],
    ).raw_by_cat

    bank_tokens_device = None
    bank_labels_t = None
    cat2bank_indices = None
    cat2evidence = None
    evidence_slots = 1

    if config["multipos"]:
        bank_texts, bank_labels, bank_langs, cat2bank_indices, cat2evidence = build_caption_bank(captions_by_cat)
        bank_labels_t = torch.tensor(bank_labels, dtype=torch.long, device=config["device"])
        bank_tokens = processor(
            text=bank_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=config["max_length"],
        )
        bank_tokens_device = {
            k: v.to(config["device"], non_blocking=True)
            for k, v in bank_tokens.items()
            if isinstance(v, torch.Tensor)
        }

        evidence_slots = max(
            (max((len(pos) for pos in ev_map.values()), default=1)
             for ev_map in cat2evidence.values()),
            default=1,
        )

        print(
            f"[CaptionBank] size={len(bank_texts)}, langs={langs}, "
            f"text_mode={config['text_mode']}, evidence_slots={evidence_slots}"
        )

    return captions_by_cat, bank_tokens_device, bank_labels_t, cat2bank_indices, cat2evidence, evidence_slots



def compute_custom_batch_loss(
    model,
    batch,
    config: Dict[str, Any],
    bank_tokens_device: Optional[Dict[str, torch.Tensor]],
    bank_labels_t: Optional[torch.Tensor],
    cat2bank_indices: Optional[Dict[int, List[int]]] = None,
    class_weights: Optional[torch.Tensor] = None,
):
    device = config["device"]
    zero_cross = torch.zeros((), device=device, dtype=torch.float32)
    zero_lscft = torch.zeros((), device=device, dtype=torch.float32)

    target = config.get("target", "lesion")
    lscft_enabled = bool(config.get("lscft", False)) and target == "lesion" and config["fusion"]
    global_feat_cached: Optional[torch.Tensor] = None
    pixel_values_local: Optional[torch.Tensor] = None

    if config["fusion"]:
        pixel_values_local = batch["pixel_values_local"].to(device, non_blocking=True)
        pixel_values_global = batch["pixel_values_global"].to(device, non_blocking=True)
        if lscft_enabled:
            img_feats, _local_feat_cached, global_feat_cached, _fused_cached = model.forward_image(
                pixel_values_local, pixel_values_global, return_parts=True
            )
        else:
            img_feats = model.forward_image(pixel_values_local, pixel_values_global)
        batch_size = int(pixel_values_local.size(0))
    else:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        img_feats = get_image_features(model, pixel_values)
        batch_size = int(pixel_values.size(0))

    if config["multipos"]:
        img_labels = batch["labels"].to(device, non_blocking=True)
        evidence_bank_idx = batch["evidence_bank_idx"].to(device, non_blocking=True)

        if target == "overall":
            # in-batch bank: text features come from this batch's input_ids
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            if config["fusion"]:
                txt_feats = model.get_text_features(input_ids, attention_mask)
            else:
                txt_feats = get_text_features(model, input_ids, attention_mask)
            bank_lbls = batch["bank_labels"].to(device, non_blocking=True)
            evidence_alpha_eff = 1.0  # disabled for overall
        else:
            if config["fusion"]:
                txt_feats = model.get_text_features(
                    bank_tokens_device["input_ids"],
                    bank_tokens_device.get("attention_mask", None),
                )
            else:
                txt_feats = get_text_features(
                    model,
                    bank_tokens_device["input_ids"],
                    bank_tokens_device.get("attention_mask", None),
                )
            bank_lbls = bank_labels_t
            evidence_alpha_eff = float(config["evidence_alpha"])

        img_feats = F.normalize(img_feats, dim=-1)
        txt_feats = F.normalize(txt_feats, dim=-1)
        scale, bias = get_logit_scale_and_bias(model, device, float(config["temperature"]))
        logits = scale * (img_feats @ txt_feats.t()) + bias

        loss, li2t, lt2i = compute_symmetric_multipos_sigmoid_loss(
            logits=logits,
            img_labels=img_labels,
            bank_labels=bank_lbls,
            evidence_bank_idx=evidence_bank_idx,
            evidence_alpha=evidence_alpha_eff,
            t2i_lambda=float(config["t2i_lambda"]),
            t2i_skip_no_positive=bool(config["t2i_skip_no_positive"]),
            t2i_label_level_mean=bool(config["t2i_label_level_mean"]),
            label_smoothing=float(config["label_smoothing"]),
            class_weights=class_weights if target != "overall" else None,
        )

        lcross = zero_cross.to(loss.dtype)
        if (
            bool(config.get("cross_lesion", False))
            and target == "lesion"
            and cat2bank_indices is not None
            and "image_ids" in batch
        ):
            image_ids = batch["image_ids"].to(device, non_blocking=True)
            lcross_val, _ = compute_cross_lesion_loss(
                img_feats=img_feats,
                txt_bank_feats=txt_feats,
                img_labels=img_labels,
                image_ids=image_ids,
                cat2bank_indices=cat2bank_indices,
                temperature=float(config.get("cross_tau", 0.1)),
            )
            lcross = lcross_val
            loss = loss + float(config.get("cross_alpha", 0.3)) * lcross

        llscft = zero_lscft.to(loss.dtype)
        if (
            lscft_enabled
            and cat2bank_indices is not None
            and global_feat_cached is not None
            and pixel_values_local is not None
        ):
            sample_txt = torch.zeros_like(img_feats)
            for i, y in enumerate(img_labels.tolist()):
                idxs = cat2bank_indices.get(int(y))
                if not idxs:
                    continue
                sel = torch.as_tensor(idxs, device=device, dtype=torch.long)
                sample_txt[i] = txt_feats.index_select(0, sel).mean(dim=0)
            sample_txt = F.normalize(sample_txt, dim=-1)

            pos_size = int(config.get("lscft_pos_size", 80))
            neg_keep = int(config.get("lscft_neg_keep", 140))
            jitter_px = int(config.get("lscft_mask_jitter", 10))
            jitter_str = float(config.get("lscft_jitter", 0.05))
            margin = float(config.get("lscft_margin", 0.1))

            pv_pos, _pos_kind = lscft_apply_positive(pixel_values_local, pos_size, jitter_px)
            pv_neg, _neg_kind = lscft_apply_negative(pixel_values_local, neg_keep, jitter_str)

            local_pos = F.normalize(get_image_features(model.base_model, pv_pos), dim=-1)
            local_neg = F.normalize(get_image_features(model.base_model, pv_neg), dim=-1)
            feat_pos = F.normalize(
                model.fuse_local_with_global_feat(local_pos, global_feat_cached), dim=-1
            )
            feat_neg = F.normalize(
                model.fuse_local_with_global_feat(local_neg, global_feat_cached), dim=-1
            )

            llscft = compute_lscft_loss(
                feat_orig=img_feats,
                feat_pos=feat_pos,
                feat_neg=feat_neg,
                txt_feat=sample_txt,
                margin=margin,
            )
            loss = loss + float(config.get("lscft_alpha", 0.5)) * llscft

        return loss, li2t, lt2i, lcross, llscft, batch_size

    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device, non_blocking=True)

    if config["fusion"]:
        txt_feats = model.get_text_features(input_ids, attention_mask)
    else:
        txt_feats = get_text_features(model, input_ids, attention_mask)

    img_feats = F.normalize(img_feats, dim=-1)
    txt_feats = F.normalize(txt_feats, dim=-1)
    scale, bias = get_logit_scale_and_bias(model, device, float(config["temperature"]))
    logits = scale * (img_feats @ txt_feats.t()) + bias

    loss, li2t, lt2i = compute_paired_sigmoid_loss(
        logits=logits,
        label_smoothing=float(config["label_smoothing"]),
    )
    return (
        loss,
        li2t,
        lt2i,
        zero_cross.to(loss.dtype),
        zero_lscft.to(loss.dtype),
        batch_size,
    )



def run_eval_custom(
    model,
    dataloader,
    config: Dict[str, Any],
    bank_tokens_device: Optional[Dict[str, torch.Tensor]],
    bank_labels_t: Optional[torch.Tensor],
    cat2bank_indices: Optional[Dict[int, List[int]]] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_i2t = 0.0
    total_t2i = 0.0
    total_cross = 0.0
    total_lscft = 0.0
    n = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            with torch.cuda.amp.autocast(enabled=(bool(config["use_amp"]) and config["device"] == "cuda")):
                loss, li2t, lt2i, lcross, llscft, bs = compute_custom_batch_loss(
                    model=model,
                    batch=batch,
                    config=config,
                    bank_tokens_device=bank_tokens_device,
                    bank_labels_t=bank_labels_t,
                    cat2bank_indices=cat2bank_indices,
                )
            total_loss += float(loss.item()) * bs
            total_i2t += float(li2t.item()) * bs
            total_t2i += float(lt2i.item()) * bs
            total_cross += float(lcross.item()) * bs
            total_lscft += float(llscft.item()) * bs
            n += bs

    denom = max(1, n)
    return {
        "loss": total_loss / denom,
        "i2t": total_i2t / denom,
        "t2i": total_t2i / denom,
        "cross": total_cross / denom,
        "lscft": total_lscft / denom,
    }



def train_custom(config: Dict[str, Any]):
    set_seed(int(config["seed"]))
    device = config["device"]
    os.makedirs(config["output_dir"], exist_ok=True)

    processor, model = create_processor_and_model(config)
    apply_freeze_config(model, config)

    target = config.get("target", "lesion")

    if target == "overall":
        captions_by_cat = None
        bank_tokens_device = None
        bank_labels_t = None
        cat2bank_indices = None
        cat2evidence = None
        evidence_slots = 1

        train_ds = CocoOverallDataset(
            data_root=config["data_root"],
            split=config["train_split"],
            use_multipos=bool(config["multipos"]),
        )
        valid_ds = CocoOverallDataset(
            data_root=config["data_root"],
            split=config["valid_split"],
            use_multipos=bool(config["multipos"]),
        )
    else:
        (
            captions_by_cat,
            bank_tokens_device,
            bank_labels_t,
            cat2bank_indices,
            cat2evidence,
            evidence_slots,
        ) = prepare_custom_supervision(config, processor)

        train_ds = CocoCropDataset(
            data_root=config["data_root"],
            split=config["train_split"],
            crop_mode=config["crop_mode"],
            captions_by_cat=captions_by_cat,
            use_multipos=bool(config["multipos"]),
            use_fusion=bool(config["fusion"]),
            use_annotation_text=(config["text_mode"] == "captions"),
            text_mode=config["text_mode"],
        )
        valid_ds = CocoCropDataset(
            data_root=config["data_root"],
            split=config["valid_split"],
            crop_mode=config["crop_mode"],
            captions_by_cat=captions_by_cat,
            use_multipos=bool(config["multipos"]),
            use_fusion=bool(config["fusion"]),
            use_annotation_text=(config["text_mode"] == "captions"),
            text_mode=config["text_mode"],
        )

    collate_fn = build_collate_fn(
        processor=processor,
        config=config,
        use_multipos=bool(config["multipos"]),
        use_fusion=bool(config["fusion"]),
        cat2bank_indices=cat2bank_indices,
        cat2evidence=cat2evidence,
        evidence_slots=int(evidence_slots),
        target=target,
    )

    class_weights = None
    if config.get("class_balanced_loss", False):
        class_weights, counts = compute_effective_number_weights(
            train_ds.labels, beta=float(config["cb_beta"]), device=device
        )
        print(f"[ClassBalanced] beta={config['cb_beta']} (cat_id: n -> weight)")
        for c in sorted(counts):
            print(f"  {c:>3}: n={counts[c]:<5d} w={class_weights[c].item():.3f}")

    use_grouped = (
        bool(config.get("grouped_sampler", False))
        and target == "lesion"
        and bool(config["multipos"])
    )
    if use_grouped:
        if not isinstance(train_ds, CocoCropDataset):
            raise ValueError("--grouped_sampler requires CocoCropDataset (lesion target).")
        train_batch_sampler = MultiCategoryGroupedBatchSampler(
            samples=train_ds.samples,
            batch_size=int(config["batch_size"]),
            grouped_ratio=float(config.get("grouped_ratio", 0.5)),
            seed=int(config["seed"]),
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=int(config["num_workers"]),
            pin_memory=bool(config["pin_memory"]) and device == "cuda",
            collate_fn=collate_fn,
        )
    elif bool(config.get("balanced_sampler", False)):
        from collections import Counter
        cnt = Counter(int(l) for l in train_ds.labels)
        sample_weights = [1.0 / cnt[int(l)] for l in train_ds.labels]
        balanced = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_ds), replacement=True
        )
        print(f"[BalancedSampler] inverse-frequency oversampling over {len(cnt)} categories")
        train_loader = DataLoader(
            train_ds,
            batch_size=int(config["batch_size"]),
            sampler=balanced,
            num_workers=int(config["num_workers"]),
            pin_memory=bool(config["pin_memory"]) and device == "cuda",
            collate_fn=collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config["num_workers"]),
            pin_memory=bool(config["pin_memory"]) and device == "cuda",
            collate_fn=collate_fn,
        )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"]) and device == "cuda",
        collate_fn=collate_fn,
    )

    optimizer = build_optimizer(model, config)
    total_steps = len(train_loader) * int(config["num_epochs"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(config["warmup_ratio"])),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(bool(config["use_amp"]) and device == "cuda"))

    train_losses: List[float] = []
    valid_losses: List[float] = []
    best_val = float("inf")

    for epoch in range(int(config["num_epochs"])):
        model.train()
        running = 0.0
        running_i2t = 0.0
        running_t2i = 0.0
        running_cross = 0.0
        running_lscft = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(bool(config["use_amp"]) and device == "cuda")):
                loss, li2t, lt2i, lcross, llscft, _ = compute_custom_batch_loss(
                    model=model,
                    batch=batch,
                    config=config,
                    bank_tokens_device=bank_tokens_device,
                    bank_labels_t=bank_labels_t,
                    cat2bank_indices=cat2bank_indices,
                    class_weights=class_weights,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if config["grad_clip_norm"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip_norm"]))

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += float(loss.item())
            running_i2t += float(li2t.item())
            running_t2i += float(lt2i.item())
            running_cross += float(lcross.item())
            running_lscft += float(llscft.item())
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "i2t": f"{li2t.item():.4f}",
                "t2i": f"{lt2i.item():.4f}",
                "cross": f"{lcross.item():.4f}",
                "lscft": f"{llscft.item():.4f}",
            })

        train_loss = running / max(1, len(train_loader))
        train_i2t = running_i2t / max(1, len(train_loader))
        train_t2i = running_t2i / max(1, len(train_loader))
        train_cross = running_cross / max(1, len(train_loader))
        train_lscft = running_lscft / max(1, len(train_loader))

        val = run_eval_custom(
            model=model,
            dataloader=valid_loader,
            config=config,
            bank_tokens_device=bank_tokens_device,
            bank_labels_t=bank_labels_t,
            cat2bank_indices=cat2bank_indices,
        )

        train_losses.append(train_loss)
        valid_losses.append(val["loss"])

        if config["fusion"]:
            inspect_fusion_state(model, valid_loader, device)

        print(
            f"Epoch {epoch+1}: train(loss={train_loss:.4f}, i2t={train_i2t:.4f}, t2i={train_t2i:.4f}, "
            f"cross={train_cross:.4f}, lscft={train_lscft:.4f}) | "
            f"valid(loss={val['loss']:.4f}, i2t={val['i2t']:.4f}, t2i={val['t2i']:.4f}, "
            f"cross={val['cross']:.4f}, lscft={val['lscft']:.4f})"
        )

        if val["loss"] < best_val:
            best_val = val["loss"]
            save_best_checkpoint(model, processor, config, best_val)

    plot_losses(train_losses, valid_losses, config["output_dir"])

    if config.get("eval_test_after_train", False):
        if target == "overall":
            test_ds = CocoOverallDataset(
                data_root=config["data_root"],
                split=config["test_split"],
                use_multipos=bool(config["multipos"]),
            )
        else:
            test_ds = CocoCropDataset(
                data_root=config["data_root"],
                split=config["test_split"],
                crop_mode=config["crop_mode"],
                captions_by_cat=captions_by_cat,
                use_multipos=bool(config["multipos"]),
                use_fusion=bool(config["fusion"]),
                use_annotation_text=(config["text_mode"] == "captions"),
                text_mode=config["text_mode"],
            )
        test_loader = DataLoader(
            test_ds,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            pin_memory=bool(config["pin_memory"]) and device == "cuda",
            collate_fn=collate_fn,
        )
        test = run_eval_custom(
            model=model,
            dataloader=test_loader,
            config=config,
            bank_tokens_device=bank_tokens_device,
            bank_labels_t=bank_labels_t,
            cat2bank_indices=cat2bank_indices,
        )
        print(
            f"Test: loss={test['loss']:.4f}, i2t={test['i2t']:.4f}, t2i={test['t2i']:.4f}, "
            f"cross={test['cross']:.4f}, lscft={test['lscft']:.4f}"
        )


# =========================
# Main
# =========================

def main():
    parser = build_parser()
    args = parser.parse_args()
    config = resolve_config(args)

    print(
        f"Mode: target={config['target']}, multipos={config['multipos']}, fusion={config['fusion']}, "
        f"freeze_text_encoder={config['freeze_text_encoder']} | "
        f"langs={config['langs']} | text_mode={config['text_mode']} | "
        f"warmup_ratio={config['warmup_ratio']} | output_dir={config['output_dir']}"
    )

    if not config["multipos"] and not config["fusion"]:
        train_baseline(config)
    else:
        train_custom(config)


if __name__ == "__main__":
    main()