"""Build the FaCE-R case database.

For each non-healthy COCO image with non-empty `global_causes_zh`, extract:
  - global vision embedding from VLM-Global (whole image)
  - lesion fusion embeddings from VLM-Lesion (per-bbox crop + full image)
  - text embeddings (colloquial / medical) from VLM-Global
  - cause text embeddings from VLM-Global (deduped across splits)

Outputs (under --output_dir):
  - train_cases.pt          list[dict], one per non-healthy training image
  - valid_cases.pt          list[dict], one per non-healthy valid image
  - cause_text_embs.pt      {"texts": List[str], "embeddings": Tensor[U, D]}
  - meta.json               config, model paths, dims, counts

Per-case dict schema:
  {
    "case_id":              int,                      # within-split running id
    "image_id":             int,                      # COCO image id
    "split":                "train" | "valid",
    "file_name":            str,
    "global_emb":           Tensor[D]                 # L2-normalized
    "text_colloquial_emb":  Optional[Tensor[D]]       # None if missing
    "text_medical_emb":     Optional[Tensor[D]]
    "lesion_embs":          Tensor[N, D]              # L2-normalized
    "lesion_boxes_xywh":    Tensor[N, 4]              # int
    "causes":               List[str]                 # raw global_causes_zh
    "cause_emb_indices":    List[int]                 # indices into cause_text_embs
  }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

VL_CLASSIFIER_DIR = Path(__file__).resolve().parents[2] / "vl_classifier"
if str(VL_CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(VL_CLASSIFIER_DIR))

from common import LocalGlobalFusionWrapper, get_image_features, get_text_features  # noqa: E402


# ---------------------------------------------------------------------------
# Model loading (avoids importing eval.py since it pulls cv2)
# ---------------------------------------------------------------------------

def load_vlm(path: str, device: str, force_fusion: bool = False):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(path)
    base = AutoModel.from_pretrained(path).to(device)
    wrap_path = os.path.join(path, "wrapper_state.pt")
    use_wrap = bool(force_fusion or os.path.exists(wrap_path))

    if use_wrap:
        if not os.path.exists(wrap_path):
            raise FileNotFoundError(
                f"force_fusion=True but wrapper_state.pt missing under {path}"
            )
        dummy = Image.new("RGB", (224, 224), (0, 0, 0))
        px = processor(images=[dummy], return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            d = get_image_features(base, px).shape[-1]
        m = LocalGlobalFusionWrapper(base, hidden_size=d).to(device)
        m.load_state_dict(torch.load(wrap_path, map_location=device))
        m.is_wrapper = True
    else:
        m = base
        m.is_wrapper = False
    m.eval()
    return m, processor


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_global_vision(
    model, processor, pil_images: Sequence[Image.Image],
    device: str, img_batch_size: int = 64, use_amp: bool = True,
) -> torch.Tensor:
    feats = []
    amp = bool(use_amp and str(device).startswith("cuda"))
    for i in range(0, len(pil_images), img_batch_size):
        batch = pil_images[i : i + img_batch_size]
        px = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.cuda.amp.autocast(enabled=amp):
            f = get_image_features(model, px)
        feats.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats, dim=0) if feats else torch.empty(0)


@torch.no_grad()
def encode_lesion_fusion(
    model, processor,
    local_pil: Sequence[Image.Image], global_pil: Sequence[Image.Image],
    device: str, img_batch_size: int = 64, use_amp: bool = True,
) -> torch.Tensor:
    assert len(local_pil) == len(global_pil)
    feats = []
    amp = bool(use_amp and str(device).startswith("cuda"))
    for i in range(0, len(local_pil), img_batch_size):
        lb = local_pil[i : i + img_batch_size]
        gb = global_pil[i : i + img_batch_size]
        lpx = processor(images=lb, return_tensors="pt")["pixel_values"].to(device)
        gpx = processor(images=gb, return_tensors="pt")["pixel_values"].to(device)
        with torch.cuda.amp.autocast(enabled=amp):
            f = model.forward_image(lpx, gpx)
        feats.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats, dim=0) if feats else torch.empty(0)


@torch.no_grad()
def encode_text(
    model, processor, texts: Sequence[str],
    device: str, text_batch_size: int = 256, max_length: int = 64, use_amp: bool = True,
) -> torch.Tensor:
    feats = []
    amp = bool(use_amp and str(device).startswith("cuda"))
    for i in range(0, len(texts), text_batch_size):
        batch = texts[i : i + text_batch_size]
        ti = processor(
            text=batch, return_tensors="pt",
            padding="max_length", truncation=True, max_length=max_length,
        )
        ti = {k: v.to(device) for k, v in ti.items()}
        with torch.cuda.amp.autocast(enabled=amp):
            f = get_text_features(model, ti["input_ids"], ti.get("attention_mask"))
        feats.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats, dim=0) if feats else torch.empty(0)


# ---------------------------------------------------------------------------
# Crop helper (mirrors vl_classifier/eval.py:get_scaled_rect_crop)
# ---------------------------------------------------------------------------

def scaled_rect_crop(img: Image.Image, bbox_xywh, scale: float = 1.0) -> Image.Image:
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = w * scale, h * scale
    W, H = img.size
    x1 = max(0, min(W - 1, int(round(cx - nw / 2.0))))
    y1 = max(0, min(H - 1, int(round(cy - nh / 2.0))))
    x2 = max(1, min(W, int(round(cx + nw / 2.0))))
    y2 = max(1, min(H, int(round(cy + nh / 2.0))))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return img.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# COCO scanning
# ---------------------------------------------------------------------------

def _safe_str(x) -> str:
    if isinstance(x, str):
        return x.strip()
    return ""


def collect_split_cases(coco_path: Path, image_root: Path) -> List[Dict]:
    """Scan one COCO file, return per-image case info (filtered)."""
    with coco_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    img_id_to_anns: Dict[int, List[dict]] = {}
    for ann in coco.get("annotations", []):
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    out: List[Dict] = []
    n_skip_healthy = 0
    n_skip_no_cause = 0
    n_skip_no_lesion = 0
    n_skip_missing_img = 0

    for img in coco.get("images", []):
        if bool(img.get("isHealthy", False)):
            n_skip_healthy += 1
            continue

        causes_raw = img.get("global_causes_zh") or []
        causes = [_safe_str(c) for c in causes_raw if isinstance(c, str)]
        causes = [c for c in causes if c]
        if not causes:
            n_skip_no_cause += 1
            continue

        anns = img_id_to_anns.get(img["id"], [])
        valid_anns = [a for a in anns if a.get("bbox") and len(a["bbox"]) == 4]
        if not valid_anns:
            n_skip_no_lesion += 1
            continue

        file_name = img.get("file_name", "")
        img_path = image_root / file_name
        if not img_path.exists():
            n_skip_missing_img += 1
            continue

        overall = img.get("overall") or {}
        out.append({
            "image_id": int(img["id"]),
            "file_name": file_name,
            "img_path": str(img_path),
            "causes": causes,
            "colloquial": _safe_str(overall.get("colloquial_zh")),
            "medical": _safe_str(overall.get("medical_zh")),
            "lesion_boxes_xywh": [list(map(int, a["bbox"])) for a in valid_anns],
        })

    print(
        f"[collect] {coco_path.name}: kept={len(out)} "
        f"skip(healthy={n_skip_healthy}, no_cause={n_skip_no_cause}, "
        f"no_lesion={n_skip_no_lesion}, missing_img={n_skip_missing_img})"
    )
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_cause_text_embeddings(
    all_unique_causes: List[str], vlm_global, processor_global,
    device: str, text_batch_size: int, max_length: int, use_amp: bool,
) -> Tuple[List[str], torch.Tensor]:
    print(f"[cause-text] encoding {len(all_unique_causes)} unique cause strings via VLM-Global ...")
    embs = encode_text(
        vlm_global, processor_global, all_unique_causes,
        device=device, text_batch_size=text_batch_size,
        max_length=max_length, use_amp=use_amp,
    )
    return all_unique_causes, embs


def process_split(
    cases: List[Dict], split_name: str,
    vlm_global, processor_global,
    vlm_lesion, processor_lesion,
    cause_text_to_idx: Dict[str, int],
    device: str, img_batch_size: int, text_batch_size: int,
    max_length: int, use_amp: bool,
    chunk_size: int = 64, lesion_scale: float = 1.0,
    progress_every: int = 200,
) -> List[Dict]:
    """Encode all per-case embeddings in chunks of `chunk_size` cases.

    Within each chunk:
      1. Load all PIL images
      2. Batched VLM-Global vision over chunk
      3. Flatten lesions across chunk → single VLM-Lesion fusion batch
      4. Batched VLM-Global text over chunk's colloquial+medical
    """
    out_cases: List[Dict] = []
    n = len(cases)
    t0 = time.time()

    for chunk_start in range(0, n, chunk_size):
        chunk = cases[chunk_start : chunk_start + chunk_size]
        pil_images = [Image.open(c["img_path"]).convert("RGB") for c in chunk]

        # 1) global vision
        g_embs = encode_global_vision(
            vlm_global, processor_global, pil_images,
            device=device, img_batch_size=img_batch_size, use_amp=use_amp,
        )  # [B, D]

        # 2) lesion fusion (flatten across chunk)
        all_local: List[Image.Image] = []
        all_global: List[Image.Image] = []
        per_case_n: List[int] = []
        for c, pil in zip(chunk, pil_images):
            crops = [scaled_rect_crop(pil, b, lesion_scale) for b in c["lesion_boxes_xywh"]]
            all_local.extend(crops)
            all_global.extend([pil] * len(crops))
            per_case_n.append(len(crops))

        if all_local:
            l_embs_flat = encode_lesion_fusion(
                vlm_lesion, processor_lesion, all_local, all_global,
                device=device, img_batch_size=img_batch_size, use_amp=use_amp,
            )  # [sum(N_i), D]
        else:
            l_embs_flat = torch.empty(0, g_embs.size(-1)) if g_embs.numel() else torch.empty(0)

        # 3) text (colloquial + medical) — encode together, slot None for empties
        text_inputs: List[Tuple[int, str, str]] = []  # (case_idx, kind, text)
        for ci, c in enumerate(chunk):
            if c["colloquial"]:
                text_inputs.append((ci, "colloquial", c["colloquial"]))
            if c["medical"]:
                text_inputs.append((ci, "medical", c["medical"]))

        if text_inputs:
            t_embs = encode_text(
                vlm_global, processor_global,
                [t[2] for t in text_inputs],
                device=device, text_batch_size=text_batch_size,
                max_length=max_length, use_amp=use_amp,
            )
        else:
            t_embs = torch.empty(0)

        # Distribute text embeddings back per case
        text_by_case: Dict[int, Dict[str, torch.Tensor]] = {ci: {} for ci in range(len(chunk))}
        for k, (ci, kind, _) in enumerate(text_inputs):
            text_by_case[ci][kind] = t_embs[k]

        # 4) Assemble case dicts
        cursor = 0
        for ci, (c, n_les) in enumerate(zip(chunk, per_case_n)):
            lesion_embs = l_embs_flat[cursor : cursor + n_les].clone()
            cursor += n_les
            case_dict = {
                "case_id": chunk_start + ci,
                "image_id": c["image_id"],
                "split": split_name,
                "file_name": c["file_name"],
                "global_emb": g_embs[ci].clone(),
                "text_colloquial_emb": text_by_case[ci].get("colloquial"),
                "text_medical_emb": text_by_case[ci].get("medical"),
                "lesion_embs": lesion_embs,
                "lesion_boxes_xywh": torch.tensor(c["lesion_boxes_xywh"], dtype=torch.long),
                "causes": list(c["causes"]),
                "cause_emb_indices": [cause_text_to_idx[s] for s in c["causes"]],
            }
            out_cases.append(case_dict)

        # release PIL handles for this chunk
        for pil in pil_images:
            pil.close()

        if (chunk_start // chunk_size) % max(1, progress_every // chunk_size) == 0:
            elapsed = time.time() - t0
            done = min(chunk_start + len(chunk), n)
            rate = done / max(elapsed, 1e-9)
            eta = (n - done) / max(rate, 1e-9)
            print(
                f"[{split_name}] {done}/{n}  "
                f"({100.0 * done / n:.1f}%)  rate={rate:.1f} cases/s  ETA={eta/60:.1f} min"
            )

    return out_cases


def main():
    ap = argparse.ArgumentParser(description="Build FaCE-R case database.")
    ap.add_argument("--coco_train", type=str, required=True)
    ap.add_argument("--coco_valid", type=str, default=None)
    ap.add_argument("--image_root_train", type=str, required=True)
    ap.add_argument("--image_root_valid", type=str, default=None)
    ap.add_argument("--vlm_global", type=str, required=True)
    ap.add_argument("--vlm_lesion", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--img_batch_size", type=int, default=64)
    ap.add_argument("--text_batch_size", type=int, default=256)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--chunk_size", type=int, default=64,
                    help="cases per chunk for vision encoding")
    ap.add_argument("--lesion_scale", type=float, default=1.0)
    ap.add_argument("--max_cases", type=int, default=-1,
                    help="cap cases per split (smoke testing); -1 = all")
    args = ap.parse_args()

    device = args.device
    use_amp = not args.no_amp
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Collect raw cases per split
    train_cases_raw = collect_split_cases(Path(args.coco_train), Path(args.image_root_train))
    if args.max_cases > 0:
        train_cases_raw = train_cases_raw[: args.max_cases]

    if args.coco_valid:
        if not args.image_root_valid:
            raise ValueError("--image_root_valid required when --coco_valid is given")
        valid_cases_raw = collect_split_cases(
            Path(args.coco_valid), Path(args.image_root_valid),
        )
        if args.max_cases > 0:
            valid_cases_raw = valid_cases_raw[: args.max_cases]
    else:
        valid_cases_raw = []

    # 2) Build deduplicated unique-cause string list (across all splits)
    all_causes_seen: Dict[str, int] = {}
    for c in train_cases_raw + valid_cases_raw:
        for s in c["causes"]:
            if s not in all_causes_seen:
                all_causes_seen[s] = len(all_causes_seen)

    unique_causes = list(all_causes_seen.keys())
    print(f"[unique-cause] {len(unique_causes)} unique cause strings across splits")

    # 3) Load VLMs
    print(f"[load] VLM-Global: {args.vlm_global}")
    vlm_global, proc_global = load_vlm(args.vlm_global, device=device, force_fusion=False)
    if vlm_global.is_wrapper:
        raise RuntimeError("VLM-Global unexpectedly loaded as fusion wrapper")
    print(f"[load] VLM-Lesion: {args.vlm_lesion}")
    vlm_lesion, proc_lesion = load_vlm(args.vlm_lesion, device=device, force_fusion=True)
    if not vlm_lesion.is_wrapper:
        raise RuntimeError("VLM-Lesion failed to load as fusion wrapper")

    # 4) Encode unique cause text
    cause_texts, cause_embs = build_cause_text_embeddings(
        unique_causes, vlm_global, proc_global,
        device=device, text_batch_size=args.text_batch_size,
        max_length=args.max_length, use_amp=use_amp,
    )
    cause_text_to_idx = {s: i for i, s in enumerate(cause_texts)}

    # 5) Process each split
    train_cases = process_split(
        train_cases_raw, "train",
        vlm_global, proc_global, vlm_lesion, proc_lesion,
        cause_text_to_idx,
        device=device, img_batch_size=args.img_batch_size,
        text_batch_size=args.text_batch_size, max_length=args.max_length,
        use_amp=use_amp, chunk_size=args.chunk_size,
        lesion_scale=args.lesion_scale,
    )

    if valid_cases_raw:
        valid_cases = process_split(
            valid_cases_raw, "valid",
            vlm_global, proc_global, vlm_lesion, proc_lesion,
            cause_text_to_idx,
            device=device, img_batch_size=args.img_batch_size,
            text_batch_size=args.text_batch_size, max_length=args.max_length,
            use_amp=use_amp, chunk_size=args.chunk_size,
            lesion_scale=args.lesion_scale,
        )
    else:
        valid_cases = []

    # 6) Save
    torch.save(train_cases, out_dir / "train_cases.pt")
    print(f"[save] train_cases.pt  n={len(train_cases)}  -> {out_dir}")

    if valid_cases:
        torch.save(valid_cases, out_dir / "valid_cases.pt")
        print(f"[save] valid_cases.pt  n={len(valid_cases)}  -> {out_dir}")

    torch.save(
        {"texts": cause_texts, "embeddings": cause_embs},
        out_dir / "cause_text_embs.pt",
    )
    print(f"[save] cause_text_embs.pt  n={len(cause_texts)}  dim={cause_embs.size(-1)}")

    meta = {
        "vlm_global": str(args.vlm_global),
        "vlm_lesion": str(args.vlm_lesion),
        "global_dim": int(cause_embs.size(-1)),
        "lesion_dim": int(train_cases[0]["lesion_embs"].size(-1)) if train_cases else None,
        "n_train_cases": len(train_cases),
        "n_valid_cases": len(valid_cases),
        "n_unique_causes": len(cause_texts),
        "lesion_scale": args.lesion_scale,
        "max_length": args.max_length,
        "img_batch_size": args.img_batch_size,
        "text_batch_size": args.text_batch_size,
        "amp": use_amp,
        "max_cases": args.max_cases,
        "coco_train": str(args.coco_train),
        "coco_valid": str(args.coco_valid) if args.coco_valid else None,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[save] meta.json  -> {out_dir}")
    print("[done]")


if __name__ == "__main__":
    main()
