import os
import sys
import json
import argparse
import time
import random
import shutil
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from PIL import Image, ImageOps

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
    # 去掉 ```json ... ``` 包裝
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
    - 若 reuse_annotations_dir 有同名 JSON，且有需要的 key，就直接用舊的。
    - 否則看 output 的 json_path 是否已存在（快取）。
    - 最後才呼叫 Gemini。
    """
    stem = Path(image_path).stem

    # 復用舊 annotations（可選）
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

    # 讀快取（若存在且不強制 overwrite）
    if json_path.exists() and not overwrite_cache:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if "generated_by" not in data:
                data["generated_by"] = model
            # 這裡允許快取裡包含其他欄位，我們只驗需要的
            core = {
                "overall": data.get("overall"),
                "global_causes_zh": data.get("global_causes_zh"),
                "global_treatments_zh": data.get("global_treatments_zh"),
                "generated_by": data.get("generated_by", model),
            }
            return validate_gemini_payload(core)
        except Exception as e:
            print(f"[WARN] 讀取快取失敗，將重新呼叫 Gemini: {json_path} ({e})", file=sys.stderr)
            if cache_only:
                raise

    if cache_only:
        raise FileNotFoundError(f"找不到快取 JSON：{json_path}")

    # 真的呼叫 Gemini
    text = call_gemini(image_path, prompt, model=model)
    data = parse_json_strict(text)
    data.setdefault("generated_by", model)
    return validate_gemini_payload(data)


# -------------- 主處理：有病圖片 -------------- #

def process_sick_image(
    image_path: Path,
    prompt: str,
    model: str,
    json_dir: Path,
    cache_only: bool = False,
    overwrite_cache: bool = False,
    reuse_annotations_dir: Path | None = None,
) -> None:
    """
    有病圖片：呼叫 Gemini 產生 overall / global_causes_zh / global_treatments_zh，
    並在最外層加：
      - image_filename
      - image_width / image_height
      - isHealthy = False
      - detections = []   <-- 依需求固定空陣列
    """
    ensure_dir(json_dir)
    image_path = Path(image_path)
    json_path = json_dir / f"{image_path.stem}.json"

    with Image.open(image_path) as im:
        W, H = im.size

    data = fetch_or_generate_json(
        image_path=str(image_path),
        prompt=prompt,
        model=model,
        json_path=json_path,
        cache_only=cache_only,
        overwrite_cache=overwrite_cache,
        reuse_annotations_dir=reuse_annotations_dir,
    )

    out = dict(data)
    out["image_filename"] = image_path.name
    out["image_width"] = int(W)
    out["image_height"] = int(H)
    out["isHealthy"] = False
    out["detections"] = []  # 這裡固定空陣列

    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------- 主處理：沒病圖片 -------------- #

def process_healthy_image(
    image_path: Path,
    json_dir: Path,
) -> None:
    """
    沒病圖片：不呼叫 Gemini，只簡單標記 isHealthy = True，
    並在最外層加：
      - image_filename
      - image_width / image_height
      - detections = []   <-- 同樣固定空陣列
    """
    ensure_dir(json_dir)
    image_path = Path(image_path)
    json_path = json_dir / f"{image_path.stem}.json"

    with Image.open(image_path) as im:
        W, H = im.size

    out: Dict[str, Any] = {
        "image_filename": image_path.name,
        "image_width": int(W),
        "image_height": int(H),
        "isHealthy": True,
        "detections": [],  # 固定空陣列
    }

    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


# -------------- 工具：列出資料夾裡的圖片 -------------- #

def list_images(dir_path: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return files


# -------------- CLI 主程式 -------------- #

def main():
    parser = argparse.ArgumentParser(
        description="魚病偵測（資料夾版：有病 / 沒病 分資料夾，無 bbox；detections 固定空陣列）"
    )

    parser.add_argument(
        "--sick_dir",
        required=True,
        help="有病圖片的資料夾路徑",
    )
    parser.add_argument(
        "--healthy_dir",
        help="沒病圖片的資料夾路徑（可選；若未提供則只處理有病資料夾）",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Gemini 用的中文 prompt 檔案路徑（例如 prompt_zh.txt）",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="輸出根目錄，底下會產生 images/、annotations/、healthy_images/、healthy_annotations/",
    )

    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument(
        "--reuse_annotations_dir",
        help="暫時功能：指定舊 annotations 目錄，若同名 JSON 有 overall/global_causes_zh/global_treatments_zh 就直接沿用，不呼叫 Gemini",
    )

    args = parser.parse_args()

    sick_dir = Path(args.sick_dir)
    if not sick_dir.is_dir():
        raise FileNotFoundError(f"找不到有病圖片資料夾：{sick_dir}")

    healthy_dir: Path | None = None
    if args.healthy_dir:
        healthy_dir = Path(args.healthy_dir)
        if not healthy_dir.is_dir():
            raise FileNotFoundError(f"找不到沒病圖片資料夾：{healthy_dir}")

    prompt = read_text(args.prompt)

    output_root = Path(args.output_dir)
    images_out_dir = ensure_dir(output_root / "images")
    ann_out_dir = ensure_dir(output_root / "annotations")

    healthy_images_out_dir: Path | None = None
    healthy_ann_out_dir: Path | None = None
    if healthy_dir is not None:
        healthy_images_out_dir = ensure_dir(output_root / "healthy_images")
        healthy_ann_out_dir = ensure_dir(output_root / "healthy_annotations")

    reuse_annotations_dir = Path(args.reuse_annotations_dir) if args.reuse_annotations_dir else None

    # 處理有病圖片（會改名成 sick_1, sick_2, ...）
    sick_files = list_images(sick_dir)
    print(f"[INFO] 有病圖片數量：{len(sick_files)}")

    sick_ok = sick_fail = 0
    for idx, src in enumerate(sick_files, 1):
        # 檔名改成 sick_1.jpg / sick_2.png ...
        new_name = f"sick_{idx}{src.suffix.lower()}"
        dst = images_out_dir / new_name
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)

        print(f"[SICK {idx}/{len(sick_files)}] {dst}")
        try:
            process_sick_image(
                image_path=dst,            # 用改名後的檔案
                prompt=prompt,
                model=args.model,
                json_dir=ann_out_dir,
                cache_only=args.cache_only,
                overwrite_cache=args.overwrite_cache,
                reuse_annotations_dir=reuse_annotations_dir,
            )
            sick_ok += 1
        except Exception as e:
            sick_fail += 1
            print(f"[錯誤] 處理有病圖片失敗：{src} ({e})", file=sys.stderr)

    # 處理沒病圖片（若有提供，改名成 healthy_1, healthy_2, ...）
    healthy_ok = healthy_fail = 0
    if healthy_dir is not None and healthy_images_out_dir is not None and healthy_ann_out_dir is not None:
        healthy_files = list_images(healthy_dir)
        print(f"[INFO] 沒病圖片數量：{len(healthy_files)}")

        for idx, src in enumerate(healthy_files, 1):
            # 檔名改成 healthy_1.jpg / healthy_2.png ...
            new_name = f"healthy_{idx}{src.suffix.lower()}"
            dst = healthy_images_out_dir / new_name
            ensure_dir(dst.parent)
            shutil.copy2(src, dst)

            print(f"[HEALTHY {idx}/{len(healthy_files)}] {dst}")
            try:
                process_healthy_image(
                    image_path=dst,          # 用改名後的檔案
                    json_dir=healthy_ann_out_dir,
                )
                healthy_ok += 1
            except Exception as e:
                healthy_fail += 1
                print(f"[錯誤] 處理沒病圖片失敗：{src} ({e})", file=sys.stderr)

    print(
        f"完成：有病圖片成功 {sick_ok}，失敗 {sick_fail}"
        + (
            f"；沒病圖片成功 {healthy_ok}，失敗 {healthy_fail}"
            if healthy_dir is not None
            else ""
        )
    )


if __name__ == "__main__":
    main()
