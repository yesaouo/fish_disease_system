from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


HEALTHY_TEXT = "魚體外觀正常，無異常表徵。"
SICK_FALLBACK_TEXT = "魚體外觀異常，似乎生病了。"


def _iter_datasets(input_dir: Path, dataset: Optional[str]) -> Iterable[Tuple[str, Path]]:
    """
    Yield (dataset_name, dataset_dir) pairs.

    If `dataset` is provided, only that one is returned.
    Otherwise, every subdirectory of `input_dir` that contains an `annotations` directory
    is considered a dataset.
    """
    if dataset:
        ds_dir = input_dir / dataset
        if ds_dir.is_dir() and (ds_dir / "annotations").is_dir():
            yield dataset, ds_dir
        return

    if not input_dir.is_dir():
        return

    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        if (child / "annotations").is_dir():
            yield child.name, child


def _normalize_str_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    out: List[str] = []
    for item in raw:
        if item is None:
            continue
        s = str(item).replace("\n", " ").strip()
        if s:
            out.append(s)
    return out


def _build_text_and_reasons(
    data: Dict[str, Any],
    style: str,
) -> Optional[Tuple[str, List[str], List[str]]]:
    """
    Return (text, causes, treats) or None if this example should be skipped
    based on isHealthy / causes / treatments rules.
    """
    is_healthy = bool(data.get("isHealthy"))

    if is_healthy:
        # Healthy case: fixed text, no causes/treatments
        return HEALTHY_TEXT, [], []

    # Diseased case: require non-empty causes and treatments
    causes = _normalize_str_list(data.get("global_causes_zh"))
    treats = _normalize_str_list(data.get("global_treatments_zh"))
    if not causes or not treats:
        return None

    overall = data.get("overall") or {}
    key = "colloquial_zh" if style == "colloquial" else "medical_zh"
    raw_text = str(overall.get(key, "") or "").replace("\n", " ").strip()
    text = raw_text if raw_text else SICK_FALLBACK_TEXT
    return text, causes, treats


def _build_detections(raw_dets: Any) -> Optional[List[Dict[str, Any]]]:
    """
    From raw detections, build the simplified list of detections.

    - Keep only label (mapped to 0/1) and box.
    - Label: "healthy_region" -> 0, others -> 1.
    """
    if not isinstance(raw_dets, list) or not raw_dets:
        return None

    out: List[Dict[str, Any]] = []
    for det in raw_dets:
        if not isinstance(det, dict):
            continue
        label_raw = str(det.get("label", "") or "").strip()
        label = 0 if label_raw == "healthy_region" else 1

        box = det.get("box_xyxy")
        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
        ):
            continue
        try:
            box_ints = [int(round(float(v))) for v in box]
        except Exception:
            continue

        out.append({"label": label, "box": box_ints})

    if not out:
        return None
    return out


def process_dataset(
    dataset_name: str,
    dataset_dir: Path,
    style: str,
    output_path: Path,
) -> int:
    """
    Process one dataset directory and append lines to the JSONL file.

    Returns the number of written lines.
    """
    annotations_dir = dataset_dir / "annotations"
    healthy_annotations_dir = dataset_dir / "healthy_annotations"

    if not annotations_dir.is_dir() and not healthy_annotations_dir.is_dir():
        # 兩個都沒有，就當這個 dataset 不存在
        return 0

    written = 0

    json_files: List[Path] = []
    if annotations_dir.is_dir():
        json_files.extend(sorted(annotations_dir.glob("*.json")))
    if healthy_annotations_dir.is_dir():
        json_files.extend(sorted(healthy_annotations_dir.glob("*.json")))

    if not json_files:
        return 0

    # Open once per dataset in append mode
    with output_path.open("a", encoding="utf-8") as f_out:
        for json_path in json_files:
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            # Skip if detections missing/empty, or comments exist
            raw_dets = data.get("detections") or []
            comments = data.get("comments") or []
            if not raw_dets:
                continue
            if isinstance(comments, list) and len(comments) > 0:
                continue

            dets = _build_detections(raw_dets)
            if not dets:
                continue

            text_causes_treats = _build_text_and_reasons(data, style)
            if text_causes_treats is None:
                continue
            text, causes, treats = text_causes_treats

            image_name = data.get("image_filename") or data.get("image_name")
            if not image_name:
                continue
            image_name = str(image_name)

            image_width = int(data.get("image_width", 0) or 0)
            image_height = int(data.get("image_height", 0) or 0)

            if image_width <= 0 or image_height <= 0:
                # If dimensions are missing/invalid, skip this record
                continue

            out_obj = {
                "dataset": str(data.get("dataset") or dataset_name),
                "image_name": image_name,
                "image_width": image_width,
                "image_height": image_height,
                "text": text,
                "causes": causes,
                "treats": treats,
                "detections": dets,
            }

            f_out.write(json.dumps(out_obj, ensure_ascii=False))
            f_out.write("\n")
            written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export annotation JSON files under input_dir into a single JSONL file."
        )
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=Path,
        required=True,
        help="Root directory that contains dataset subdirectories (e.g. data).",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default=None,
        help=(
            "Name of a single dataset to process (subdirectory under input_dir). "
            "If omitted, all datasets that contain an 'annotations' folder are processed."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Output JSONL file path. If not set, defaults to "
            "<input_dir>/annotations.jsonl."
        ),
    )
    parser.add_argument(
        "--style",
        choices=["colloquial", "medical"],
        default="colloquial",
        help="Which overall text to use: colloquial_zh or medical_zh (default: colloquial).",
    )

    args = parser.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        raise SystemExit(f"input_dir does not exist or is not a directory: {input_dir}")

    output_path: Path
    if args.output is not None:
        output_path = args.output
    else:
        output_path = input_dir / "annotations.jsonl"

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for dataset_name, dataset_dir in _iter_datasets(input_dir, args.dataset):
        written = process_dataset(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            style=args.style,
            output_path=output_path,
        )
        total_written += written

    print(f"Total lines written to {output_path}: {total_written}")


if __name__ == "__main__":
    main()

