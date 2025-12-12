from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import torch.multiprocessing as mp

from .config import DIM, ANNOTATIONS_PATH, IMAGE_ROOT, CROSS_ALIGN_CHECKPOINT_DIR
from .cross_align import CrossAlignFormer
from .text_encoder import EmbeddingGemma
from .vision import VisionBackbone


class FishAlignDataset(Dataset):
    """Read annotations from JSONL and return raw PIL image + metadata."""

    def __init__(self, ann_path: Path, image_root: Path) -> None:
        super().__init__()
        self.ann_path = ann_path
        self.image_root = image_root

        text = ann_path.read_text(encoding="utf-8")
        self.items: List[Dict[str, Any]] = [json.loads(line) for line in text.splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.items[idx]
        img_path = self.image_root / rec["dataset"].replace("_", " ") / "images" / rec["image_name"]
        image = Image.open(img_path).convert("RGB")

        boxes = torch.tensor([det["box"] for det in rec.get("detections", [])], dtype=torch.float32)
        labels = torch.tensor([det["label"] for det in rec.get("detections", [])], dtype=torch.long)
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        w = float(rec.get("image_width", image.width))
        h = float(rec.get("image_height", image.height))
        boxes_norm = boxes.clone()
        if boxes_norm.numel() > 0:
            boxes_norm[:, [0, 2]] /= w
            boxes_norm[:, [1, 3]] /= h
            boxes_norm = boxes_norm.clamp(0.0, 1.0)

        return {
            "text": rec.get("text", ""),
            "causes": rec.get("causes", []),
            "treats": rec.get("treats", []),
            "image": image,
            "boxes": boxes,
            "boxes_norm": boxes_norm,
            "labels": labels,
        }


class Collator:
    def __init__(self, vision: VisionBackbone, text_enc: EmbeddingGemma):
        self.vision = vision
        self.text_enc = text_enc

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # K = 每張圖的 ROI 數，取 batch 內最大值
        max_k = max(item["boxes"].shape[0] for item in batch)

        global_tokens: List[torch.Tensor] = []
        roi_feats_list: List[torch.Tensor] = []
        boxes_norm_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        roi_masks: List[torch.Tensor] = []
        text_emb_list: List[torch.Tensor] = []
        cause_emb_list: List[torch.Tensor] = []
        treat_emb_list: List[torch.Tensor] = []
        has_cause_list: List[torch.Tensor] = []
        has_treat_list: List[torch.Tensor] = []

        for item in batch:
            img = item["image"]
            boxes = item["boxes"]          # (k, 4)
            boxes_norm = item["boxes_norm"]  # (k, 4)
            labels = item["labels"]        # (k,)

            # --- global token: 整張圖 ---
            global_tok = torch.from_numpy(self.vision.extract(img))

            # --- ROI features: 只算框 ---
            roi_feats: List[torch.Tensor] = []
            for box in boxes.tolist():
                x1, y1, x2, y2 = box
                crop = img.crop((x1, y1, x2, y2))
                roi_feats.append(torch.from_numpy(self.vision.extract(crop)))

            if roi_feats:
                roi_feats_t = torch.stack(roi_feats, dim=0)   # (k, dim)
            else:
                roi_feats_t = torch.zeros((0, DIM), dtype=torch.float32)

            n_roi = roi_feats_t.shape[0]
            pad_k = max_k - n_roi
            if pad_k > 0:
                roi_feats_t = torch.cat(
                    [roi_feats_t, torch.zeros((pad_k, DIM), dtype=torch.float32)],
                    dim=0,
                )
                boxes_norm = torch.cat(
                    [boxes_norm, torch.zeros((pad_k, 4), dtype=torch.float32)],
                    dim=0,
                )
                labels = torch.cat(
                    [labels, torch.zeros((pad_k,), dtype=torch.long)],
                    dim=0,
                )

            # True 的部分是「真的有 ROI」的位置
            roi_mask = torch.zeros((max_k,), dtype=torch.bool)
            roi_mask[:n_roi] = True

            # --- text / cause / treat ----
            text_emb = torch.from_numpy(self.text_enc.encode(item["text"] or ""))

            if item["causes"]:
                cause_emb = torch.from_numpy(
                    self.text_enc.encode_document(item["causes"]).mean(axis=0)
                )
            else:
                cause_emb = torch.zeros((DIM,), dtype=torch.float32)

            if item["treats"]:
                treat_emb = torch.from_numpy(
                    self.text_enc.encode_document(item["treats"]).mean(axis=0)
                )
            else:
                treat_emb = torch.zeros((DIM,), dtype=torch.float32)

            global_tokens.append(global_tok)
            roi_feats_list.append(roi_feats_t)
            boxes_norm_list.append(boxes_norm)
            labels_list.append(labels)
            roi_masks.append(roi_mask)
            text_emb_list.append(text_emb)
            cause_emb_list.append(cause_emb)
            treat_emb_list.append(treat_emb)
            has_cause_list.append(torch.tensor(bool(item["causes"]), dtype=torch.bool))
            has_treat_list.append(torch.tensor(bool(item["treats"]), dtype=torch.bool))

        batch_dict = {
            "global_token": torch.stack(global_tokens, dim=0),   # (B, dim)
            "roi_feats": torch.stack(roi_feats_list, dim=0),     # (B, K, dim)
            "boxes_norm": torch.stack(boxes_norm_list, dim=0),   # (B, K, 4)
            "labels": torch.stack(labels_list, dim=0),           # (B, K)
            "roi_mask": torch.stack(roi_masks, dim=0),           # (B, K)
            "text_emb": torch.stack(text_emb_list, dim=0),       # (B, dim)
            "cause_emb": torch.stack(cause_emb_list, dim=0),     # (B, dim)
            "treat_emb": torch.stack(treat_emb_list, dim=0),     # (B, dim)
            "has_cause": torch.stack(has_cause_list, dim=0),     # (B,)
            "has_treat": torch.stack(has_treat_list, dim=0),     # (B,)
        }
        return batch_dict


def build_train_val_dataloaders(
    ann_path: Path,
    image_root: Path,
    batch_size: int,
    num_workers: int,
    vision: VisionBackbone,
    text_enc: EmbeddingGemma,
    val_ratio: float,
    seed: int,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    ds = FishAlignDataset(ann_path, image_root)
    collate_fn = Collator(vision, text_enc)

    # 如果 val_ratio 不合理，就只回傳 train_loader，不做驗證
    if val_ratio <= 0.0 or val_ratio >= 1.0:
        train_loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        return train_loader, None

    n_val = int(len(ds) * val_ratio)
    # 確保 train / val 都至少有一筆
    n_val = max(1, min(len(ds) - 1, n_val))
    n_train = len(ds) - n_val

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return train_loader, val_loader


def contrastive_loss(
    q: torch.Tensor,  # (B, dim)
    e: torch.Tensor,  # (B, dim)
    temperature: float = 0.07,
) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    e = F.normalize(e, dim=-1)
    logits = torch.matmul(q, e.t()) / temperature  # (B, B)
    target = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) * 0.5


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    vision = VisionBackbone(device=args.vision_device or None)
    text_enc = EmbeddingGemma(device=args.text_device or None)
    model = CrossAlignFormer().to(device)

    train_loader, val_loader = build_train_val_dataloaders(
        ann_path=Path(args.annotations),
        image_root=Path(args.image_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        vision=vision,
        text_enc=text_enc,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # === 新增：用來存每個 epoch 的平均 loss（train / val 各一組） ===
    train_epoch_losses: List[float] = []
    train_epoch_cause_losses: List[float] = []
    train_epoch_treat_losses: List[float] = []

    val_epoch_losses: List[float] = []
    val_epoch_cause_losses: List[float] = []
    val_epoch_treat_losses: List[float] = []

    for epoch in range(1, args.epochs + 1):

        # ======== Training Phase ========
        model.train()

        running_loss = 0.0
        running_cause = 0.0
        running_treat = 0.0
        num_steps = 0

        for step, batch in enumerate(train_loader, start=1):
            global_token = batch["global_token"].to(device, non_blocking=True)
            roi_feats = batch["roi_feats"].to(device, non_blocking=True)
            boxes_norm = batch["boxes_norm"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            roi_mask = batch["roi_mask"].to(device, non_blocking=True)
            text_emb = batch["text_emb"].to(device, non_blocking=True)
            cause_emb = batch["cause_emb"].to(device, non_blocking=True)
            treat_emb = batch["treat_emb"].to(device, non_blocking=True)
            has_cause = batch["has_cause"].to(device, non_blocking=True)
            has_treat = batch["has_treat"].to(device, non_blocking=True)

            _, q_cause, q_treat = model(
                global_token=global_token,
                roi_feats=roi_feats,
                boxes_norm=boxes_norm,
                class_ids=labels,
                text_emb=text_emb,
                roi_mask=roi_mask,
            )

            q_cause_pool = q_cause.mean(dim=1)
            q_treat_pool = q_treat.mean(dim=1)

            loss_cause = torch.tensor(0.0, device=device)
            loss_treat = torch.tensor(0.0, device=device)

            if has_cause.any():
                loss_cause = contrastive_loss(
                    q_cause_pool[has_cause], cause_emb[has_cause], temperature=args.temperature
                )
            if has_treat.any():
                loss_treat = contrastive_loss(
                    q_treat_pool[has_treat], treat_emb[has_treat], temperature=args.temperature
                )

            loss = loss_cause + loss_treat

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            running_cause += loss_cause.item()
            running_treat += loss_treat.item()
            num_steps += 1

            if step % args.log_every == 0:
                print(
                    f"epoch {epoch} step {step} "
                    f"train_loss {loss.item():.4f} "
                    f"train_cause {loss_cause.item():.4f} train_treat {loss_treat.item():.4f}"
                )

        avg_train_loss = running_loss / max(num_steps, 1)
        avg_train_cause = running_cause / max(num_steps, 1)
        avg_train_treat = running_treat / max(num_steps, 1)

        train_epoch_losses.append(avg_train_loss)
        train_epoch_cause_losses.append(avg_train_cause)
        train_epoch_treat_losses.append(avg_train_treat)

        # ======== Validation Phase ========
        avg_val_loss = 0.0
        avg_val_cause = 0.0
        avg_val_treat = 0.0

        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            val_running_cause = 0.0
            val_running_treat = 0.0
            val_num_steps = 0

            with torch.no_grad():
                for batch in val_loader:
                    global_token = batch["global_token"].to(device, non_blocking=True)
                    roi_feats = batch["roi_feats"].to(device, non_blocking=True)
                    boxes_norm = batch["boxes_norm"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    roi_mask = batch["roi_mask"].to(device, non_blocking=True)
                    text_emb = batch["text_emb"].to(device, non_blocking=True)
                    cause_emb = batch["cause_emb"].to(device, non_blocking=True)
                    treat_emb = batch["treat_emb"].to(device, non_blocking=True)
                    has_cause = batch["has_cause"].to(device, non_blocking=True)
                    has_treat = batch["has_treat"].to(device, non_blocking=True)

                    _, q_cause, q_treat = model(
                        global_token=global_token,
                        roi_feats=roi_feats,
                        boxes_norm=boxes_norm,
                        class_ids=labels,
                        text_emb=text_emb,
                        roi_mask=roi_mask,
                    )

                    q_cause_pool = q_cause.mean(dim=1)
                    q_treat_pool = q_treat.mean(dim=1)

                    loss_cause = torch.tensor(0.0, device=device)
                    loss_treat = torch.tensor(0.0, device=device)

                    if has_cause.any():
                        loss_cause = contrastive_loss(
                            q_cause_pool[has_cause], cause_emb[has_cause], temperature=args.temperature
                        )
                    if has_treat.any():
                        loss_treat = contrastive_loss(
                            q_treat_pool[has_treat], treat_emb[has_treat], temperature=args.temperature
                        )

                    loss = loss_cause + loss_treat

                    val_running_loss += loss.item()
                    val_running_cause += loss_cause.item()
                    val_running_treat += loss_treat.item()
                    val_num_steps += 1

            avg_val_loss = val_running_loss / max(val_num_steps, 1)
            avg_val_cause = val_running_cause / max(val_num_steps, 1)
            avg_val_treat = val_running_treat / max(val_num_steps, 1)

            val_epoch_losses.append(avg_val_loss)
            val_epoch_cause_losses.append(avg_val_cause)
            val_epoch_treat_losses.append(avg_val_treat)

        # ======== log 當前 epoch 的結果 ========
        log_msg = (
            f"[epoch {epoch}] "
            f"train_loss={avg_train_loss:.4f} train_cause={avg_train_cause:.4f} train_treat={avg_train_treat:.4f}"
        )
        if val_loader is not None:
            log_msg += (
                f" | val_loss={avg_val_loss:.4f} val_cause={avg_val_cause:.4f} val_treat={avg_val_treat:.4f}"
            )
        print(log_msg)

        if epoch % args.save_every == 0:
            ckpt_path = save_dir / f"epoch_{epoch}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"saved checkpoint: {ckpt_path}")

    # === 全部訓練完之後畫 loss 曲線圖（train / val） ===
    epochs = list(range(1, args.epochs + 1))

    # (1) total loss
    plt.figure()
    plt.plot(epochs, train_epoch_losses, label="train total loss")
    if len(val_epoch_losses) > 0:
        plt.plot(epochs, val_epoch_losses, label="val total loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss (Train vs Val)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = save_dir / "loss_total_curve.png"
    plt.savefig(curve_path)
    print(f"saved total loss curve: {curve_path}")

    # (2) cause loss
    plt.figure()
    plt.plot(epochs, train_epoch_cause_losses, label="train cause loss")
    if len(val_epoch_cause_losses) > 0:
        plt.plot(epochs, val_epoch_cause_losses, label="val cause loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cause Loss (Train vs Val)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    cause_curve_path = save_dir / "loss_cause_curve.png"
    plt.savefig(cause_curve_path)
    print(f"saved cause loss curve: {cause_curve_path}")

    # (3) treat loss
    plt.figure()
    plt.plot(epochs, train_epoch_treat_losses, label="train treat loss")
    if len(val_epoch_treat_losses) > 0:
        plt.plot(epochs, val_epoch_treat_losses, label="val treat loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Treat Loss (Train vs Val)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    treat_curve_path = save_dir / "loss_treat_curve.png"
    plt.savefig(treat_curve_path)
    print(f"saved treat loss curve: {treat_curve_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CrossAlignFormer on fish disease annotations.")

    p.add_argument("--annotations", type=str, default=ANNOTATIONS_PATH)
    p.add_argument("--image-root", type=str, default=IMAGE_ROOT)
    p.add_argument("--save-dir", type=str, default=CROSS_ALIGN_CHECKPOINT_DIR)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--device", type=str, default=None, help="torch device for the model")
    p.add_argument("--vision-device", type=str, default=None, help="device override for DINOv3")
    p.add_argument("--text-device", type=str, default=None, help="device override for EmbeddingGemma")
    p.add_argument("--save-every", type=int, default=1, help="save every N epochs")
    p.add_argument("--log-every", type=int, default=10, help="log every N steps")
    p.add_argument("--val-ratio", type=float, default=0.2, help="ratio of data used for validation (0 to disable)")
    p.add_argument("--seed", type=int, default=42, help="random seed for train/val split")

    return p.parse_args()


if __name__ == "__main__":
    # 統一用 spawn，避免 Linux 預設 fork + CUDA 出錯
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # 已經設定過 start method 的情況（例如被外部環境先設過），就忽略
        pass

    args = parse_args()
    train(args)
