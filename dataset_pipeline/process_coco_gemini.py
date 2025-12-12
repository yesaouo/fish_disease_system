import os
import sys
import json
import argparse
import time
import random
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageOps
import yaml

DEFAULT_OTHER_LABEL = "healthy_region"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------- I/O ---------------- #

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


# --------------- Gemini --------------- #

def _retry(func, times: int = 3, base: float = 1.0):
    for i in range(times):
        try:
            return func()
        except Exception:
            if i == times - 1:
                raise
            time.sleep(base * (2 ** i) + random.random() * 0.2)


def call_gemini(image_path: str, prompt: str, model: str) -> str:
    from google import genai
    from google.genai import types as genai_types

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("請設定 GEMINI_API_KEY 或 GOOGLE_API_KEY")

    client = genai.Client(api_key=api_key)

    with Image.open(image_path) as raw_im:
        im = ImageOps.exif_transpose(raw_im).copy()
    im.thumbnail([1280, 1280])

    def _do_req():
        resp = client.models.generate_content(
            model=model,
            contents=[prompt, im],
            config=genai_types.GenerateContentConfig(
                system_instruction="你是專精觀賞魚疾病的視覺助理，請輸出嚴格 JSON。",
                temperature=0.2,
            ),
        )
        out = getattr(resp, "text", None) or getattr(resp, "output_text", None)
        if not out:
            raise RuntimeError("模型沒有回傳文字內容")
        return out

    return _retry(_do_req, times=3, base=1.0)


def parse_json_strict(text: str) -> Dict[str, Any]:
    import re

    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I | re.M)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        raise ValueError("回應缺少 JSON 物件：\n" + text)
    return json.loads(m.group(0))


def validate_gemini_payload(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Gemini 回傳不是 JSON 物件")

    overall = data.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("缺少 overall")
    for k in ("colloquial_zh", "medical_zh"):
        v = overall.get(k)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"overall.{k} 需為非空字串")

    for key in ("global_causes_zh", "global_treatments_zh"):
        arr = data.get(key)
        if not (isinstance(arr, list) and all(isinstance(x, str) for x in arr)):
            raise ValueError(f"{key} 需為 List[str]")

    if not isinstance(data.get("generated_by"), str):
        raise ValueError("缺少 generated_by")

    return data


# -------------- COCO + YAML -------------- #

def load_coco(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_coco(coco: dict, json_path: Path) -> None:
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)


def load_yaml_map(yaml_path: Path) -> Dict[int, str]:
    """
    讀 classes.yaml：
      返回 (category_id -> label)
    """
    with yaml_path.open("r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f) or {}

    yaml_map: Dict[int, str] = {}

    for k, v in mapping.items():
        try:
            cid = int(k)
        except Exception:
            cid = int(str(k))
        label = str(v)
        yaml_map[cid] = label

    return yaml_map


def segmentation_to_bbox(segmentation):
    """
    segmentation 可能是：
      - list[float]
      - list[list[float]]
    """
    if not segmentation:
        raise ValueError("Empty segmentation")

    xs: List[float] = []
    ys: List[float] = []

    if isinstance(segmentation, list):
        if segmentation and isinstance(segmentation[0], list):
            for poly in segmentation:
                xs.extend(poly[0::2])
                ys.extend(poly[1::2])
        else:
            xs.extend(segmentation[0::2])
            ys.extend(segmentation[1::2])
    else:
        raise ValueError(f"Unsupported segmentation type: {type(segmentation)}")

    if not xs or not ys:
        raise ValueError("Invalid segmentation coordinates")

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def process_segmentation(coco: dict) -> None:
    """
    對整個 COCO：
      - 有 segmentation 的 annotation 嘗試轉成 bbox
      - 無論是否成功，一律移除 segmentation 欄位
    """
    anns = coco.get("annotations", [])
    seg_count = 0
    conv_count = 0

    for ann in anns:
        if "segmentation" in ann and ann["segmentation"]:
            seg_count += 1
            try:
                bbox = segmentation_to_bbox(ann["segmentation"])
                ann["bbox"] = bbox
                conv_count += 1
            except Exception as e:
                print(f"[WARN] annotation id={ann.get('id')} segmentation 轉 bbox 失敗: {e}")
        if "segmentation" in ann:
            del ann["segmentation"]

    print(f"[INFO] 有 {seg_count} 個 annotation 含 segmentation，其中 {conv_count} 個成功轉成 bbox。")


def bbox_contains(inner, outer) -> bool:
    """
    檢查 inner bbox 是否完全在 outer bbox 之內
    bbox 格式為 [x, y, w, h]
    """
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer

    ix1, iy1, ix2, iy2 = ix, iy, ix + iw, iy + ih
    ox1, oy1, ox2, oy2 = ox, oy, ox + ow, oy + oh

    return (ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2)


def remove_healthy_region_if_other_inside(
    coco: dict,
    yaml_map: Dict[int, str],
    healthy_label: str = DEFAULT_OTHER_LABEL,
) -> None:
    """
    對每張圖：
      - 找出 YAML 映射為 healthy_label (預設 healthy_region) 的框
      - 找出 YAML 映射為其他 label 的框
      - 若某個 healthy_region bbox 內含任一其他類別 bbox，則移除該 healthy_region annotation
    """
    anns = coco.get("annotations", [])
    if not anns:
        print("[INFO] 沒有任何 annotation，不需移除 healthy_region。")
        return

    anns_by_image: Dict[int, List[dict]] = defaultdict(list)
    for ann in anns:
        anns_by_image[ann["image_id"]].append(ann)

    total_healthy = sum(
        1 for ann in anns
        if yaml_map.get(ann.get("category_id")) == healthy_label
    )
    removed_ids: set[int] = set()

    for image_id, image_anns in anns_by_image.items():
        healthy_list: List[dict] = []
        other_list: List[dict] = []

        for ann in image_anns:
            cid = ann.get("category_id")
            label = yaml_map.get(cid)
            if label is None:
                # YAML 沒寫到的 id 不參與 healthy/other 判斷
                continue
            if label == healthy_label:
                healthy_list.append(ann)
            else:
                other_list.append(ann)

        if not healthy_list or not other_list:
            continue

        removed_here = 0

        for h in healthy_list:
            hbbox = h.get("bbox")
            if not (isinstance(hbbox, (list, tuple)) and len(hbbox) == 4):
                continue
            for o in other_list:
                obbox = o.get("bbox")
                if not (isinstance(obbox, (list, tuple)) and len(obbox) == 4):
                    continue
                try:
                    if bbox_contains(obbox, hbbox):
                        removed_ids.add(h["id"])
                        removed_here += 1
                        break
                except Exception:
                    continue

        if removed_here > 0:
            print(f"[INFO] image_id={image_id} 移除 {removed_here} 個 {healthy_label} 框（內含其他類別框）。")

    if not removed_ids:
        print(f"[INFO] 沒有任何 {healthy_label} 框被移除。")
        return

    before = len(anns)
    coco["annotations"] = [a for a in anns if a.get("id") not in removed_ids]
    after = len(coco["annotations"])
    removed_count = len(removed_ids)
    remain_healthy = total_healthy - removed_count

    print(f"[INFO] {healthy_label} 總數：{total_healthy}，被移除：{removed_count}，剩下：{remain_healthy}")
    print(f"[INFO] annotations 總數：{before} -> {after}")


def build_coco_index(coco: dict) -> Dict[str, Any]:
    images = coco.get("images", [])
    anns = coco.get("annotations", [])

    images_by_id: Dict[int, dict] = {img["id"]: img for img in images}
    anns_by_image_id: Dict[int, List[dict]] = defaultdict(list)
    for ann in anns:
        iid = ann.get("image_id")
        if iid is None:
            continue
        anns_by_image_id[iid].append(ann)

    return {"images_by_id": images_by_id, "anns_by_image_id": anns_by_image_id}


def ann_to_detection(
    ann: dict,
    yaml_map: Dict[int, str],
    W: int,
    H: int,
) -> Dict[str, Any] | None:
    """
    只有 category_id 有出現在 yaml_map 的才會變成 detection。
    這裡輸出 box_xyxy，為原始像素座標 [x1, y1, x2, y2]。
    """
    cid = ann.get("category_id")
    if cid not in yaml_map:
        return None

    label = yaml_map[cid]

    bbox = ann.get("bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        seg = ann.get("segmentation")
        if seg:
            try:
                bbox = segmentation_to_bbox(seg)
            except Exception as e:
                print(f"[WARN] ann id={ann.get('id')} segmentation 轉 bbox 失敗: {e}")
                return None
        else:
            return None

    x, y, w, h = map(float, bbox)
    if w <= 0 or h <= 0 or W <= 0 or H <= 0:
        return None

    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    box_xyxy = [x1, y1, x2, y2]

    return {
        "label": label,
        "confidence": 1.0,
        "evidence_zh": "",
        "box_xyxy": box_xyxy,
    }


# -------------- 視覺化（可選） -------------- #

def _clip_and_fix_xyxy(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int
) -> Tuple[int, int, int, int]:
    lx, rx = sorted(
        [
            max(0, min(int(round(x1)), W - 1)),
            max(0, min(int(round(x2)), W - 1)),
        ]
    )
    ty, by = sorted(
        [
            max(0, min(int(round(y1)), H - 1)),
            max(0, min(int(round(y2)), H - 1)),
        ]
    )

    if rx <= lx:
        rx = min(lx + 1, W - 1)
    if by <= ty:
        by = min(ty + 1, H - 1)

    return lx, ty, rx, by


def draw_boxes(im: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    im = im.convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for det in detections:
        box_xyxy = det.get("box_xyxy")
        if not (isinstance(box_xyxy, (list, tuple)) and len(box_xyxy) == 4):
            continue

        x1, y1, x2, y2 = map(float, box_xyxy)
        lx, ty, rx, by = _clip_and_fix_xyxy(x1, y1, x2, y2, W, H)

        draw.rectangle([(lx, ty), (rx, by)], outline="red", width=3)
        label = det.get("label", "lesion")
        conf = det.get("confidence")
        text = f"{label}" + (f" ({conf:.2f})" if isinstance(conf, (float, int)) else "")
        bbox = draw.textbbox((lx, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        by1 = max(ty - th - 4, 0)
        by2 = max(ty - 2, 0)
        draw.rectangle([(lx, by1), (lx + tw + 8, by2)], fill="red")
        draw.text((lx + 4, max(ty - th - 2, 0)), text, fill="white", font=font)
    return im


# -------------- 主流程 -------------- #

def _is_valid_box_xyxy(box: Any) -> bool:
    if not (isinstance(box, list) and len(box) == 4):
        return False
    x1, y1, x2, y2 = box
    return all(isinstance(v, (int, float)) for v in (x1, y1, x2, y2))


def fetch_or_generate_json(
    image_path: str,
    prompt: str,
    model: str,
    json_path: Path,
    cache_only: bool,
    overwrite_cache: bool,
    reuse_annotations_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    先試著從 reuse_annotations_dir 讀舊 JSON（如果有給）：
      - 若同名 JSON 有 overall / global_causes_zh / global_treatments_zh 就直接沿用
    否則走原本快取 + Gemini 流程
    """
    stem = Path(image_path).stem

    # ==== 暫時復用舊 annotations（之後可以整段移除） ====
    if reuse_annotations_dir is not None:
        legacy_path = reuse_annotations_dir / f"{stem}.json"
        if legacy_path.is_file():
            try:
                legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
                if all(k in legacy for k in ("overall", "global_causes_zh", "global_treatments_zh")):
                    data = {
                        "overall": legacy["overall"],
                        "global_causes_zh": legacy["global_causes_zh"],
                        "global_treatments_zh": legacy["global_treatments_zh"],
                        "generated_by": legacy.get("generated_by", "legacy_import"),
                    }
                    return validate_gemini_payload(data)
            except Exception as e:
                print(f"[WARN] 復用舊 annotations 失敗 ({legacy_path}): {e}", file=sys.stderr)
    # ==== 暫時復用邏輯結束 ====

    if overwrite_cache:
        text = call_gemini(image_path, prompt, model=model)
        data = parse_json_strict(text)
        data.setdefault("generated_by", model)
        return validate_gemini_payload(data)

    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if "generated_by" not in data:
                data["generated_by"] = model
            return validate_gemini_payload(data)
        except Exception:
            if cache_only:
                raise

    if cache_only:
        raise FileNotFoundError(f"找不到快取 JSON：{json_path}")

    text = call_gemini(image_path, prompt, model=model)
    data = parse_json_strict(text)
    data.setdefault("generated_by", model)
    return validate_gemini_payload(data)


def process_one_image(
    image_path: str,
    prompt: str,
    model: str,
    detections: List[Dict[str, Any]],
    json_dir: Path,
    results_dir: Path | None,
    cache_only: bool = False,
    overwrite_cache: bool = False,
    reuse_annotations_dir: Path | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> None:
    ensure_dir(json_dir)
    stem = Path(image_path).stem
    json_path = json_dir / f"{stem}.json"

    data = fetch_or_generate_json(
        image_path=image_path,
        prompt=prompt,
        model=model,
        json_path=json_path,
        cache_only=cache_only,
        overwrite_cache=overwrite_cache,
        reuse_annotations_dir=reuse_annotations_dir,
    )

    out = dict(data)

    # === 在每張圖的 JSON 最外層加三個欄位 ===
    out["image_filename"] = Path(image_path).name
    if image_width is not None:
        out["image_width"] = int(image_width)
    if image_height is not None:
        out["image_height"] = int(image_height)

    items: List[Dict[str, Any]] = []
    for det in detections:
        bxy = det.get("box_xyxy")
        if not _is_valid_box_xyxy(bxy):
            continue
        item = {
            "label": det.get("label", DEFAULT_OTHER_LABEL),
            "box_xyxy": list(bxy),  # 直接存原始座標
        }
        if isinstance(det.get("confidence"), (float, int)):
            item["confidence"] = float(det["confidence"])
        if isinstance(det.get("evidence_zh"), str):
            item["evidence_zh"] = det["evidence_zh"]
        items.append(item)
    out["detections"] = items

    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if results_dir is not None and detections:
        with Image.open(image_path) as im:
            viz = draw_boxes(im, detections)
            out_path = results_dir / f"{stem}_overlay{Path(image_path).suffix or '.jpg'}"
            ensure_dir(out_path.parent)
            viz.save(out_path)


def save_healthy_only_json(
    image_path: str | Path,
    json_dir: Path,
    detections: List[Dict[str, Any]],
    image_width: int | None = None,
    image_height: int | None = None,
) -> None:
    """
    healthy_only 的圖片不呼叫 Gemini，
    但要保留 bbox detections，並在最外層標記 isHealthy=true：

      {
        "image_filename": "...",
        "image_width": ...,
        "image_height": ...,
        "isHealthy": true,
        "detections": [ ... ]
      }
    """
    ensure_dir(json_dir)
    image_path = Path(image_path)
    stem = image_path.stem
    json_path = json_dir / f"{stem}.json"

    out: Dict[str, Any] = {
        "image_filename": image_path.name,
        "isHealthy": True,
    }
    if image_width is not None:
        out["image_width"] = int(image_width)
    if image_height is not None:
        out["image_height"] = int(image_height)

    items: List[Dict[str, Any]] = []
    for det in detections:
        box = det.get("box_xyxy")
        if not _is_valid_box_xyxy(box):
            continue

        item: Dict[str, Any] = {
            "label": det.get("label", DEFAULT_OTHER_LABEL),
            "box_xyxy": list(box),
        }
        conf = det.get("confidence")
        if isinstance(conf, (int, float)):
            item["confidence"] = float(conf)
        items.append(item)

    out["detections"] = items

    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="魚病偵測（Gemini 全域敘述 + COCO bbox/seg→bbox）"
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="根目錄，底下包含 images/、prompt_zh.txt、classes.yaml；COCO 在 images/_annotations.coco.json",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="輸出給 annotation_web 用的根目錄（會產生 images / annotations / viz_results 等）",
    )

    # 這兩個參數拿掉
    # parser.add_argument("--prompt", required=True, help="Gemini 用的中文 prompt 檔案路徑")
    # parser.add_argument("--yaml_map", required=True, help="類別 id → 名稱 的 YAML（classes.yaml）")

    parser.add_argument("--model", default="gemini-2.5-pro")

    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")

    parser.add_argument(
        "--reuse_annotations_dir",
        help="暫時功能：指定舊 annotations 目錄，若同名 JSON 有 overall/global_causes_zh/global_treatments_zh 就直接沿用，不呼叫 Gemini",
    )

    parser.add_argument(
        "--keep_healthy_only",
        action="store_true",
        help="保留只含 healthy_region 的圖片（不呼叫 Gemini，另存 isHealthy JSON + healthy_viz_results）",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # ---- 新增：固定路徑 ---- #
    images_src_dir = input_dir / "images"
    prompt_path = input_dir / "prompt_zh.txt"
    yaml_path = input_dir / "classes.yaml"
    coco_json_path = images_src_dir / "_annotations.coco.json"

    if not images_src_dir.is_dir():
        raise FileNotFoundError(f"找不到圖片資料夾：{images_src_dir}")
    if not coco_json_path.is_file():
        raise FileNotFoundError(f"找不到 COCO 標註檔：{coco_json_path}")
    if not prompt_path.is_file():
        raise FileNotFoundError(f"找不到 prompt 檔案：{prompt_path}")
    if not yaml_path.is_file():
        raise FileNotFoundError(f"找不到 classes.yaml：{yaml_path}")

    prompt = read_text(prompt_path)
    coco = load_coco(coco_json_path)
    yaml_map = load_yaml_map(yaml_path)

    # 2. segmentation → bbox 並移除 segmentation 欄位
    process_segmentation(coco)

    # 4. healthy_region 裡面如果包著其他類別 bbox，先把該 healthy_region bbox 移除
    remove_healthy_region_if_other_inside(coco, yaml_map, healthy_label=DEFAULT_OTHER_LABEL)

    index = build_coco_index(coco)
    images_by_id: Dict[int, dict] = index["images_by_id"]
    anns_by_image_id: Dict[int, List[dict]] = index["anns_by_image_id"]

    images_out_dir = ensure_dir(output_dir / "images")
    json_dir = ensure_dir(output_dir / "annotations")
    results_dir = ensure_dir(output_dir / "viz_results") if args.save_viz else None

    healthy_images_dir: Path | None = None
    healthy_json_dir: Path | None = None
    healthy_results_dir: Path | None = None
    if args.keep_healthy_only:
        healthy_images_dir = ensure_dir(output_dir / "healthy_images")
        healthy_json_dir = ensure_dir(output_dir / "healthy_annotations")
        if args.save_viz:
            healthy_results_dir = ensure_dir(output_dir / "healthy_viz_results")

    reuse_annotations_dir = Path(args.reuse_annotations_dir) if args.reuse_annotations_dir else None

    images = list(images_by_id.values())

    success = fail = 0
    healthy_only = 0
    skipped_unmapped = 0  # 例如只含跳過類別、或完全沒被 yaml_map cover 的情況

    # 為了最後輸出 subset COCO
    used_images: List[dict] = []
    used_annotations: List[dict] = []

    for idx, img in enumerate(images, 1):
        iid = img["id"]
        file_name = img["file_name"]

        src = images_src_dir / file_name
        if not src.is_file():
            print(f"[WARN] 找不到圖片檔案: {src}", file=sys.stderr)
            fail += 1
            continue

        try:
            with Image.open(src) as _im_:
                W, H = _im_.size

            anns = anns_by_image_id.get(iid, [])
            detections: List[Dict[str, Any]] = []
            filtered_anns: List[dict] = []

            for ann in anns:
                det = ann_to_detection(ann, yaml_map, W, H)
                if det is not None:
                    detections.append(det)
                    filtered_anns.append(ann)  # 只把 yaml_map 有 cover 的 ann 放進 subset COCO

            has_det = len(detections) > 0
            has_non_healthy = any(det["label"] != DEFAULT_OTHER_LABEL for det in detections)

            # 1) 完全沒有有效標籤（例如只含跳過類別或完全沒標）：整張略過
            if not has_det:
                skipped_unmapped += 1
                print(f"[略過(無有效標籤，可能都是被跳過的類別)] {file_name}")
                continue

            # 2) 只含 healthy_region
            if has_det and not has_non_healthy:
                healthy_only += 1
                if not args.keep_healthy_only:
                    # 預設：整張略過
                    print(f"[略過(只有 {DEFAULT_OTHER_LABEL})] {file_name}")
                    continue
                else:
                    # 不呼叫 Gemini，單純標記為 isHealthy=True，存在獨立路徑，並輸出 healthy_viz_results
                    if healthy_images_dir is None or healthy_json_dir is None:
                        print(f"[WARN] --keep_healthy_only 啟用但 healthy 目錄未建立，略過 {file_name}")
                        continue

                    dst = healthy_images_dir / file_name
                    ensure_dir(dst.parent)
                    shutil.copy2(src, dst)

                    save_healthy_only_json(
                        image_path=dst,
                        json_dir=healthy_json_dir,
                        detections=detections,
                        image_width=W,
                        image_height=H,
                    )

                    # healthy_viz_results
                    if healthy_results_dir is not None and args.save_viz and detections:
                        with Image.open(dst) as im:
                            viz = draw_boxes(im, detections)
                            out_path = healthy_results_dir / f"{Path(file_name).stem}_overlay{Path(file_name).suffix or '.jpg'}"
                            ensure_dir(out_path.parent)
                            viz.save(out_path)

                    print(f"[HEALTHY {idx}/{len(images)}] {dst}")
                    success += 1
                    continue

            # 能走到這裡代表：
            #   - 至少有 1 個 yaml_map cover 的類別（不會是被跳過的類別）
            #   - 且不會只有 healthy_region
            dst = images_out_dir / file_name
            ensure_dir(dst.parent)
            shutil.copy2(src, dst)

            print(f"[{idx}/{len(images)}] {dst}")

            process_one_image(
                image_path=str(dst),
                prompt=prompt,
                model=args.model,
                detections=detections,
                json_dir=json_dir,
                results_dir=results_dir,
                cache_only=args.cache_only,
                overwrite_cache=args.overwrite_cache,
                reuse_annotations_dir=reuse_annotations_dir,
                image_width=W,
                image_height=H,
            )

            used_images.append(img)
            used_annotations.extend(filtered_anns)
            success += 1

        except Exception:
            fail += 1
            print(f"[錯誤] {src}", file=sys.stderr)

    # 產生 subset _annotations.coco.json 到 images 資料夾
    used_image_ids = {img["id"] for img in used_images}
    used_cat_ids = {ann["category_id"] for ann in used_annotations}

    subset_images = [img for img in used_images]
    subset_annotations = [ann for ann in used_annotations if ann["image_id"] in used_image_ids]

    subset_categories = []
    for cid, name in yaml_map.items():
        if cid in used_cat_ids:
            subset_categories.append(
                {
                    "id": cid,
                    "name": name,
                    "supercategory": name,
                }
            )

    subset_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": subset_images,
        "annotations": subset_annotations,
        "categories": subset_categories,
    }

    save_coco(subset_coco, images_out_dir / "_annotations.coco.json")

    if args.keep_healthy_only:
        print(
            f"完成：成功 {success}，失敗 {fail}，"
            f"僅 {DEFAULT_OTHER_LABEL} 的圖片已另外輸出 {healthy_only} 張，"
            f"無有效標籤(可能都是被跳過的類別或沒標) 的圖片被略過 {skipped_unmapped} 張"
        )
    else:
        print(
            f"完成：成功 {success}，失敗 {fail}，"
            f"僅 {DEFAULT_OTHER_LABEL} 的圖片被略過 {healthy_only} 張，"
            f"無有效標籤(可能都是被跳過的類別或沒標) 的圖片被略過 {skipped_unmapped} 張"
        )


if __name__ == "__main__":
    main()
