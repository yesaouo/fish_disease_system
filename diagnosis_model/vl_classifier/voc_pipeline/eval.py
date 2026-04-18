from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from common import (
    DEFAULT_FONT_PATH,
    draw_prediction_visualization,
    encode_image_features,
    encode_text_features,
    load_eval_texts,
    load_model_and_processor,
    save_classification_report,
    save_confusion_matrix,
)
from voc_dataset import VocRegionDataset
from voc_labels import save_default_voc_label_bank



def parse_model_specs(model_specs: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
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



def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Unified VOC evaluator for baseline / multipos / fusion checkpoints.")
    ap.add_argument("--voc_root", type=str, required=True)
    ap.add_argument("--year", type=str, choices=["2007", "2012"], default="2007")
    ap.add_argument("--image_set", type=str, default="test")
    ap.add_argument("--label_bank_json", type=str, default="")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--crop_mode", type=str, choices=["bbox", "square"], default="bbox")
    ap.add_argument("--model", type=str, action="append", required=True, help="可重複指定：tag=repo_or_path")
    ap.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--text_batch_size", type=int, default=256)
    ap.add_argument("--img_batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--save_vis", action="store_true")
    ap.add_argument("--font_path", type=str, default=DEFAULT_FONT_PATH)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--skip_difficult", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument(
        "--fusion",
        action="store_true",
        help="強制用 fusion pipeline 載入所有 model；若 model 目錄中沒有 wrapper_state.pt 會報錯。",
    )
    return ap



def ensure_label_bank(label_bank_json: str, output_dir: Path) -> str:
    if label_bank_json:
        path = Path(label_bank_json)
    else:
        path = output_dir / "voc_label_bank.json"
    if not path.exists():
        save_default_voc_label_bank(path)
    return str(path)



def process_voc_dataset(
    voc_root: str,
    year: str,
    image_set: str,
    label_bank_json: str,
    output_dir: str,
    model_specs,
    crop_mode: str = "bbox",
    save_vis: bool = False,
    font_path: str = DEFAULT_FONT_PATH,
    device: str = "cuda",
    text_batch_size: int = 256,
    img_batch_size: int = 64,
    max_length: int = 64,
    use_amp: bool = True,
    force_fusion: bool = False,
    download: bool = False,
    skip_difficult: bool = False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading label bank ...")
    flat_texts, flat_label_ids, id_to_zh_map, all_label_ids = load_eval_texts(label_bank_json)
    if len(flat_texts) == 0:
        raise RuntimeError("label_bank_json 的 captions_en 為空，無法做 text candidates。")

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

    dataset = VocRegionDataset(
        root=voc_root,
        year=year,
        image_set=image_set,
        label_bank_json=label_bank_json,
        crop_mode=crop_mode,
        use_multipos=False,
        use_fusion=True,
        return_meta=True,
        download=download,
        skip_difficult=skip_difficult,
    )

    image_to_samples: Dict[str, List] = defaultdict(list)
    for sample in dataset.samples:
        image_to_samples[sample.image_path].append(sample)

    y_true: List[str] = []
    y_pred = {tag: [] for tag, _ in model_specs}

    vis_dir = output_dir / "vis"
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    total_images = len(image_to_samples)
    for image_idx, (image_path, samples) in enumerate(image_to_samples.items(), start=1):
        if image_idx % 100 == 0 or image_idx == total_images:
            print(f"Processing images: {image_idx}/{total_images}")

        pil_img = Image.open(image_path).convert("RGB")
        crops = [dataset._crop(pil_img, s.bbox_xywh) for s in samples]
        globals_list = [pil_img] * len(samples)
        gt_ids = [str(s.label_id) for s in samples]
        bboxes_xywh = [s.bbox_xywh for s in samples]

        y_true.extend(gt_ids)
        preds_this_img = {}

        for tag, (m, p) in models.items():
            img_feats = encode_image_features(
                m,
                p,
                crops,
                device=device,
                img_batch_size=img_batch_size,
                use_amp=use_amp,
                images_global=globals_list,
            )
            scores = img_feats @ text_features[tag].T
            best_idx = scores.argmax(dim=-1).tolist()
            pred_ids = [flat_label_ids[i] for i in best_idx]

            y_pred[tag].extend(pred_ids)
            preds_this_img[tag] = pred_ids

        if save_vis:
            records = []
            for i, (gt_id, bbox) in enumerate(zip(gt_ids, bboxes_xywh)):
                records.append(
                    {
                        "bbox": bbox,
                        "gt_id": gt_id,
                        "preds": {tag: preds_this_img[tag][i] for tag, _ in model_specs},
                    }
                )
            vis = draw_prediction_visualization(pil_img, records, [tag for tag, _ in model_specs], id_to_zh_map, font_path)
            vis.save(vis_dir / f"pred_{Path(image_path).name}")

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



def main():
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_bank_json = ensure_label_bank(args.label_bank_json, output_dir)
    model_specs = parse_model_specs(args.model)

    process_voc_dataset(
        voc_root=args.voc_root,
        year=args.year,
        image_set=args.image_set,
        label_bank_json=label_bank_json,
        output_dir=str(output_dir),
        model_specs=model_specs,
        crop_mode=args.crop_mode,
        save_vis=args.save_vis,
        font_path=args.font_path,
        device=args.device,
        text_batch_size=args.text_batch_size,
        img_batch_size=args.img_batch_size,
        max_length=args.max_length,
        use_amp=(not args.no_amp),
        force_fusion=bool(args.fusion),
        download=bool(args.download),
        skip_difficult=bool(args.skip_difficult),
    )


if __name__ == "__main__":
    main()
