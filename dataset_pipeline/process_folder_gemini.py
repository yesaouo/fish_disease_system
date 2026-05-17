import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gemini import fetch_or_generate_json  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------- I/O ---------------- #

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


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
