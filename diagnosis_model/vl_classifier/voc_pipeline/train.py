from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, get_cosine_schedule_with_warmup

from common import (
    LocalGlobalFusionWrapper,
    build_caption_bank,
    compute_paired_sigmoid_loss,
    compute_symmetric_multipos_sigmoid_loss,
    get_image_features,
    get_logit_scale_and_bias,
    get_text_features,
    load_label_bank,
    plot_loss_curve,
    save_json,
    set_seed,
)
from voc_dataset import VocRegionDataset
from voc_labels import save_default_voc_label_bank

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


DEFAULT_CONFIG = {
    "model_name": "google/siglip2-base-patch16-224",
    "voc_root": "./data",
    "year": "2007",
    "train_image_set": "train",
    "valid_image_set": "val",
    "test_image_set": None,
    "label_bank_json": None,
    "output_dir": None,
    "crop_mode": "bbox",
    "batch_size": 128,
    "num_epochs": 10,
    "learning_rate": 1e-4,
    "weight_decay": None,
    "max_length": 64,
    "num_workers": 8,
    "pin_memory": True,
    "grad_clip_norm": 1.0,
    "use_amp": True,
    "seed": 42,
    "dropout_prob": 0.1,
    "fusion_base_lr": 3e-5,
    "fusion_head_lr": 1e-4,
    "evidence_alpha": 3.0,
    "temperature": 0.07,
    "label_smoothing": 0.05,
    "t2i_skip_no_positive": True,
    "t2i_label_level_mean": True,
    "t2i_lambda": 0.5,
    "download": False,
    "skip_difficult_train": False,
    "skip_difficult_valid": False,
    "skip_difficult_test": False,
    "eval_test_after_train": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}



def infer_output_dir(config: Dict[str, Any]) -> str:
    model_stub = str(config["model_name"]).split("/")[-1].replace("-", "_")
    if config["multipos"] and config["fusion"]:
        suffix = "multipos_fusion"
    elif config["multipos"]:
        suffix = "multipos"
    elif config["fusion"]:
        suffix = "fusion"
    else:
        suffix = "baseline"
    return f"./{model_stub}_voc{config['year']}_{suffix}"



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified VOC training entry for baseline / multi-positive / fusion modes")
    parser.add_argument("--multipos", action="store_true", help="use multi-positive caption-bank training")
    parser.add_argument("--fusion", action="store_true", help="use local-global image fusion")

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--voc_root", type=str, default=None)
    parser.add_argument("--year", type=str, choices=["2007", "2012"], default=None)
    parser.add_argument("--train_image_set", type=str, default=None)
    parser.add_argument("--valid_image_set", type=str, default=None)
    parser.add_argument("--test_image_set", type=str, default=None)
    parser.add_argument("--label_bank_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--crop_mode", type=str, choices=["bbox", "square"], default=None)

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
    parser.add_argument("--base_learning_rate", type=float, default=None)
    parser.add_argument("--fusion_learning_rate", type=float, default=None)

    parser.add_argument("--evidence_alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--t2i_lambda", type=float, default=None)

    parser.add_argument("--download", action="store_true")
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--skip_difficult_train", action="store_true")
    parser.add_argument("--skip_difficult_valid", action="store_true")
    parser.add_argument("--skip_difficult_test", action="store_true")

    parser.add_argument("--no_amp", action="store_true", help="disable AMP")
    parser.add_argument("--no_pin_memory", action="store_true", help="disable pin_memory")
    parser.add_argument("--eval_test_after_train", action="store_true")
    parser.add_argument("--t2i_skip_no_positive", action="store_true")
    parser.add_argument("--t2i_label_level_mean", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    return parser



def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config["multipos"] = bool(args.multipos)
    config["fusion"] = bool(args.fusion)

    for key in [
        "model_name",
        "voc_root",
        "year",
        "train_image_set",
        "valid_image_set",
        "test_image_set",
        "label_bank_json",
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
        "base_learning_rate",
        "fusion_learning_rate",
        "evidence_alpha",
        "temperature",
        "label_smoothing",
        "t2i_lambda",
        "device",
    ]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    if args.base_learning_rate is not None:
        config["fusion_base_lr"] = args.base_learning_rate
    if args.fusion_learning_rate is not None:
        config["fusion_head_lr"] = args.fusion_learning_rate

    if args.pin_memory:
        config["pin_memory"] = True
    if args.t2i_skip_no_positive:
        config["t2i_skip_no_positive"] = True
    if args.t2i_label_level_mean:
        config["t2i_label_level_mean"] = True

    if args.no_amp:
        config["use_amp"] = False
    if args.no_pin_memory:
        config["pin_memory"] = False
    if args.eval_test_after_train:
        config["eval_test_after_train"] = True
    if args.download:
        config["download"] = True
    if args.skip_difficult_train:
        config["skip_difficult_train"] = True
    if args.skip_difficult_valid:
        config["skip_difficult_valid"] = True
    if args.skip_difficult_test:
        config["skip_difficult_test"] = True

    if config["weight_decay"] is None:
        config["weight_decay"] = 0.01 if (config["multipos"] or config["fusion"]) else 0.0

    if not config["output_dir"]:
        config["output_dir"] = infer_output_dir(config)

    if not config["test_image_set"]:
        config["test_image_set"] = "test" if config["year"] == "2007" else "val"

    return config



def ensure_label_bank(config: Dict[str, Any]) -> str:
    if config["label_bank_json"]:
        path = Path(config["label_bank_json"])
    else:
        path = Path(config["output_dir"]) / "voc_label_bank.json"
    if not path.exists():
        save_default_voc_label_bank(path)
    return str(path)



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
    ).to(device)
    return processor, model



def build_collate_fn(
    processor,
    config: Dict[str, Any],
    use_multipos: bool,
    use_fusion: bool,
    cat2bank_indices: Optional[Dict[int, List[int]]] = None,
):
    def collate_fn(items):
        if use_multipos:
            labels = torch.tensor([int(x["label_id"]) for x in items], dtype=torch.long)
            evidence_idx = [int(x["evidence_index"]) for x in items]
            evidence_bank_idx = torch.full((len(items),), -1, dtype=torch.long)

            for i, (y, ei) in enumerate(zip(labels.tolist(), evidence_idx)):
                if ei is None or int(ei) < 0:
                    continue
                idxs = cat2bank_indices.get(int(y), None) if cat2bank_indices is not None else None
                if not idxs:
                    continue
                ei2 = max(0, min(int(ei), len(idxs) - 1))
                evidence_bank_idx[i] = int(idxs[ei2])

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
                }

            images = [x["image"] for x in items]
            img_batch = processor(images=images, return_tensors="pt")
            return {
                "pixel_values": img_batch["pixel_values"],
                "labels": labels,
                "evidence_bank_idx": evidence_bank_idx,
            }

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
        return torch.optim.AdamW(
            [
                {"params": model.base_model.parameters(), "lr": float(config["fusion_base_lr"])},
                {"params": model.fusion_linear.parameters(), "lr": float(config["fusion_head_lr"])},
                {"params": [model.gate], "lr": float(config["fusion_head_lr"])},
            ],
            weight_decay=float(config["weight_decay"]),
        )

    return torch.optim.AdamW(
        model.parameters(),
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
        saveable_config = dict(config)
        json_like = {k: v for k, v in saveable_config.items()}
        import json
        json.dump(json_like, f, ensure_ascii=False, indent=2)

    print(f"  -> Saved best to {config['output_dir']} (valid_loss={best_metric:.4f})")



def prepare_custom_supervision(config: Dict[str, Any], processor):
    captions_by_cat, _ = load_label_bank(config["label_bank_json"])

    bank_texts = None
    bank_labels_t = None
    cat2bank_indices = None
    bank_tokens_device = None

    if config["multipos"]:
        bank_texts, bank_labels, cat2bank_indices = build_caption_bank(captions_by_cat)
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

    return captions_by_cat, bank_tokens_device, bank_labels_t, cat2bank_indices



def compute_custom_batch_loss(
    model,
    batch,
    config: Dict[str, Any],
    bank_tokens_device: Optional[Dict[str, torch.Tensor]],
    bank_labels_t: Optional[torch.Tensor],
):
    device = config["device"]

    if config["fusion"]:
        pixel_values_local = batch["pixel_values_local"].to(device, non_blocking=True)
        pixel_values_global = batch["pixel_values_global"].to(device, non_blocking=True)
        img_feats = model.forward_image(pixel_values_local, pixel_values_global)
        batch_size = int(pixel_values_local.size(0))
    else:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        img_feats = get_image_features(model, pixel_values)
        batch_size = int(pixel_values.size(0))

    if config["multipos"]:
        img_labels = batch["labels"].to(device, non_blocking=True)
        evidence_bank_idx = batch["evidence_bank_idx"].to(device, non_blocking=True)

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

        img_feats = F.normalize(img_feats, dim=-1)
        txt_feats = F.normalize(txt_feats, dim=-1)
        scale, bias = get_logit_scale_and_bias(model, device, float(config["temperature"]))
        logits = scale * (img_feats @ txt_feats.t()) + bias

        loss, li2t, lt2i = compute_symmetric_multipos_sigmoid_loss(
            logits=logits,
            img_labels=img_labels,
            bank_labels=bank_labels_t,
            evidence_bank_idx=evidence_bank_idx,
            evidence_alpha=float(config["evidence_alpha"]),
            t2i_lambda=float(config["t2i_lambda"]),
            t2i_skip_no_positive=bool(config["t2i_skip_no_positive"]),
            t2i_label_level_mean=bool(config["t2i_label_level_mean"]),
            label_smoothing=float(config["label_smoothing"]),
        )
        return loss, li2t, lt2i, batch_size

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
    return loss, li2t, lt2i, batch_size



def run_eval_baseline(model, dataloader, device: str, use_amp: bool) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(use_amp and str(device).startswith("cuda"))):
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



def run_eval_custom(
    model,
    dataloader,
    config: Dict[str, Any],
    bank_tokens_device: Optional[Dict[str, torch.Tensor]],
    bank_labels_t: Optional[torch.Tensor],
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_i2t = 0.0
    total_t2i = 0.0
    n = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            with torch.cuda.amp.autocast(enabled=(bool(config["use_amp"]) and str(config["device"]).startswith("cuda"))):
                loss, li2t, lt2i, bs = compute_custom_batch_loss(
                    model=model,
                    batch=batch,
                    config=config,
                    bank_tokens_device=bank_tokens_device,
                    bank_labels_t=bank_labels_t,
                )
            total_loss += float(loss.item()) * bs
            total_i2t += float(li2t.item()) * bs
            total_t2i += float(lt2i.item()) * bs
            n += bs

    denom = max(1, n)
    return {
        "loss": total_loss / denom,
        "i2t": total_i2t / denom,
        "t2i": total_t2i / denom,
    }



def train_baseline(config: Dict[str, Any]):
    set_seed(int(config["seed"]))
    device = config["device"]
    os.makedirs(config["output_dir"], exist_ok=True)

    load_label_bank(config["label_bank_json"])
    train_ds = VocRegionDataset(
        root=config["voc_root"],
        year=config["year"],
        image_set=config["train_image_set"],
        label_bank_json=config["label_bank_json"],
        crop_mode=config["crop_mode"],
        use_multipos=False,
        use_fusion=False,
        download=bool(config["download"]),
        skip_difficult=bool(config["skip_difficult_train"]),
    )
    valid_ds = VocRegionDataset(
        root=config["voc_root"],
        year=config["year"],
        image_set=config["valid_image_set"],
        label_bank_json=config["label_bank_json"],
        crop_mode=config["crop_mode"],
        use_multipos=False,
        use_fusion=False,
        download=bool(config["download"]),
        skip_difficult=bool(config["skip_difficult_valid"]),
    )

    processor, model = create_processor_and_model(config)
    collate_fn = build_collate_fn(processor, config, use_multipos=False, use_fusion=False)

    pin_memory = bool(config["pin_memory"]) and str(device).startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    optimizer = build_optimizer(model, config)
    total_steps = len(train_loader) * int(config["num_epochs"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(bool(config["use_amp"]) and str(device).startswith("cuda")))

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

            with torch.cuda.amp.autocast(enabled=(bool(config["use_amp"]) and str(device).startswith("cuda"))):
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

    plot_loss_curve(train_losses, valid_losses, Path(config["output_dir"]) / "loss_curve.png")

    if config.get("eval_test_after_train", False):
        test_ds = VocRegionDataset(
            root=config["voc_root"],
            year=config["year"],
            image_set=config["test_image_set"],
            label_bank_json=config["label_bank_json"],
            crop_mode=config["crop_mode"],
            use_multipos=False,
            use_fusion=False,
            download=bool(config["download"]),
            skip_difficult=bool(config["skip_difficult_test"]),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        test_loss = run_eval_baseline(model, test_loader, device, bool(config["use_amp"]))
        print(f"Test loss: {test_loss:.4f}")



def train_custom(config: Dict[str, Any]):
    set_seed(int(config["seed"]))
    device = config["device"]
    os.makedirs(config["output_dir"], exist_ok=True)

    processor, model = create_processor_and_model(config)
    captions_by_cat, bank_tokens_device, bank_labels_t, cat2bank_indices = prepare_custom_supervision(config, processor)

    train_ds = VocRegionDataset(
        root=config["voc_root"],
        year=config["year"],
        image_set=config["train_image_set"],
        label_bank_json=config["label_bank_json"],
        crop_mode=config["crop_mode"],
        use_multipos=bool(config["multipos"]),
        use_fusion=bool(config["fusion"]),
        download=bool(config["download"]),
        skip_difficult=bool(config["skip_difficult_train"]),
    )
    valid_ds = VocRegionDataset(
        root=config["voc_root"],
        year=config["year"],
        image_set=config["valid_image_set"],
        label_bank_json=config["label_bank_json"],
        crop_mode=config["crop_mode"],
        use_multipos=bool(config["multipos"]),
        use_fusion=bool(config["fusion"]),
        download=bool(config["download"]),
        skip_difficult=bool(config["skip_difficult_valid"]),
    )

    collate_fn = build_collate_fn(
        processor=processor,
        config=config,
        use_multipos=bool(config["multipos"]),
        use_fusion=bool(config["fusion"]),
        cat2bank_indices=cat2bank_indices,
    )

    pin_memory = bool(config["pin_memory"]) and str(device).startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    optimizer = build_optimizer(model, config)
    total_steps = len(train_loader) * int(config["num_epochs"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(bool(config["use_amp"]) and str(device).startswith("cuda")))

    train_losses: List[float] = []
    valid_losses: List[float] = []
    best_val = float("inf")

    for epoch in range(int(config["num_epochs"])):
        model.train()
        running = 0.0
        running_i2t = 0.0
        running_t2i = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(bool(config["use_amp"]) and str(device).startswith("cuda"))):
                loss, li2t, lt2i, _ = compute_custom_batch_loss(
                    model=model,
                    batch=batch,
                    config=config,
                    bank_tokens_device=bank_tokens_device,
                    bank_labels_t=bank_labels_t,
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
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "i2t": f"{li2t.item():.4f}",
                "t2i": f"{lt2i.item():.4f}",
            })

        train_loss = running / max(1, len(train_loader))
        train_i2t = running_i2t / max(1, len(train_loader))
        train_t2i = running_t2i / max(1, len(train_loader))
        val = run_eval_custom(
            model=model,
            dataloader=valid_loader,
            config=config,
            bank_tokens_device=bank_tokens_device,
            bank_labels_t=bank_labels_t,
        )

        train_losses.append(train_loss)
        valid_losses.append(val["loss"])

        print(
            f"Epoch {epoch+1}: train(loss={train_loss:.4f}, i2t={train_i2t:.4f}, t2i={train_t2i:.4f}) | "
            f"valid(loss={val['loss']:.4f}, i2t={val['i2t']:.4f}, t2i={val['t2i']:.4f})"
        )

        if val["loss"] < best_val:
            best_val = val["loss"]
            save_best_checkpoint(model, processor, config, best_val)

    plot_loss_curve(train_losses, valid_losses, Path(config["output_dir"]) / "loss_curve.png")

    if config.get("eval_test_after_train", False):
        test_ds = VocRegionDataset(
            root=config["voc_root"],
            year=config["year"],
            image_set=config["test_image_set"],
            label_bank_json=config["label_bank_json"],
            crop_mode=config["crop_mode"],
            use_multipos=bool(config["multipos"]),
            use_fusion=bool(config["fusion"]),
            download=bool(config["download"]),
            skip_difficult=bool(config["skip_difficult_test"]),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        test = run_eval_custom(
            model=model,
            dataloader=test_loader,
            config=config,
            bank_tokens_device=bank_tokens_device,
            bank_labels_t=bank_labels_t,
        )
        print(f"Test: loss={test['loss']:.4f}, i2t={test['i2t']:.4f}, t2i={test['t2i']:.4f}")



def main():
    parser = build_parser()
    args = parser.parse_args()
    config = resolve_config(args)
    config["label_bank_json"] = ensure_label_bank(config)

    print(
        f"Mode: multipos={config['multipos']}, fusion={config['fusion']} | "
        f"output_dir={config['output_dir']}"
    )

    if not config["multipos"] and not config["fusion"]:
        train_baseline(config)
    else:
        train_custom(config)


if __name__ == "__main__":
    main()
