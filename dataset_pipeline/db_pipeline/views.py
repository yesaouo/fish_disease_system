from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image

from .sources import TaskRecord


def _coco_skeleton(categories: list[dict], description: str) -> dict:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "info": {
            "year": int(now.split("-")[0]),
            "version": "1.0",
            "description": description,
            "contributor": "db_pipeline",
            "url": "",
            "date_created": now,
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/",
            }
        ],
        "categories": list(categories),
        "images": [],
        "annotations": [],
    }


def _image_entry(task: TaskRecord, width: int, height: int) -> dict:
    doc = task.doc
    return {
        "id": task.image_id,
        "file_name": f"{task.image_id}.jpg",
        "width": width,
        "height": height,
        "date_captured": doc.get("last_modified_at", ""),
        "license": 1,
        "isHealthy": bool(task.is_healthy),
        "source_dataset": task.dataset,
        "source_filename": task.image_filename,
        "source_task_id": task.task_id,
        "generated_by": doc.get("generated_by"),
        "general_editor": doc.get("general_editor") or [],
        "expert_editor": doc.get("expert_editor") or [],
        "overall": doc.get("overall") or {},
        "global_causes_zh": doc.get("global_causes_zh") or [],
        "global_treatments_zh": doc.get("global_treatments_zh") or [],
    }


def _annotation_entry(ann_id: int, image_id: int, det: dict, cat_id: int) -> dict:
    x1, y1, x2, y2 = det["box_xyxy"]
    w = float(x2) - float(x1)
    h = float(y2) - float(y1)
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": int(cat_id),
        "bbox": [float(x1), float(y1), w, h],
        "area": w * h,
        "segmentation": [],
        "iscrowd": 0,
        "evidence_zh": det.get("evidence_zh", ""),
        "evidence_index": det.get("evidence_index"),
    }


def write_views(
    version_dir: Path,
    tasks_per_split: dict[str, list[TaskRecord]],
    name_to_cat_id: dict[str, int],
    full_categories: list[dict],
    labels_id_map: dict[int, int],
    detection_categories: list[dict],
    progress=None,
) -> None:
    full_root = version_dir / "full"
    det_root = version_dir / "detection"
    healthy_root = version_dir / "healthy_images"

    for split, tasks in tasks_per_split.items():
        full_split = full_root / split
        det_split = det_root / split
        full_split.mkdir(parents=True, exist_ok=True)
        det_split.mkdir(parents=True, exist_ok=True)

        full_coco = _coco_skeleton(full_categories, f"Full view ({split})")
        det_coco = _coco_skeleton(detection_categories, f"Detection view ({split})")
        full_ann_id = 1
        det_ann_id = 1

        iterator = progress(tasks, desc=f"write:{split}", leave=False) if progress else tasks
        for task in iterator:
            if task.from_healthy_folder:
                # Negative sample: single physical copy lives under healthy_images/<split>/,
                # full/ and detection/ both symlink to it.
                healthy_split = healthy_root / split
                healthy_split.mkdir(parents=True, exist_ok=True)
                dst = healthy_split / f"{task.image_id}.jpg"
                shutil.copy2(task.image_path, dst)
                with Image.open(dst) as im:
                    width, height = im.size
                for split_dir in (full_split, det_split):
                    link = split_dir / f"{task.image_id}.jpg"
                    if link.is_symlink() or link.exists():
                        link.unlink()
                    link.symlink_to(os.path.relpath(dst, split_dir))
            else:
                dst_full = full_split / f"{task.image_id}.jpg"
                shutil.copy2(task.image_path, dst_full)
                with Image.open(dst_full) as im:
                    width, height = im.size

                link = det_split / f"{task.image_id}.jpg"
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(os.path.relpath(dst_full, det_split))

            img_entry = _image_entry(task, width, height)
            full_coco["images"].append(img_entry)
            det_coco["images"].append(dict(img_entry))

            for det in task.doc["detections"]:
                orig_cat_id = name_to_cat_id[det["label"]]
                full_coco["annotations"].append(
                    _annotation_entry(full_ann_id, task.image_id, det, orig_cat_id)
                )
                full_ann_id += 1
                new_cat_id = labels_id_map.get(orig_cat_id)
                if new_cat_id is not None:
                    det_ann = _annotation_entry(det_ann_id, task.image_id, det, new_cat_id)
                    # Per-box symptom category for GROD joint training's loss_semantic
                    # (the fork's coco loader reads this field, falls back to -1 when
                    # absent). category_id stays class-agnostic (ABNORMAL); this extra
                    # field lets train_joint consume the detection view directly, so no
                    # separate build_merged_coco / _merged_semantic join is needed.
                    det_ann["symptom_category_id"] = orig_cat_id
                    det_coco["annotations"].append(det_ann)
                    det_ann_id += 1

        (full_split / "_annotations.coco.json").write_text(
            json.dumps(full_coco, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (det_split / "_annotations.coco.json").write_text(
            json.dumps(det_coco, ensure_ascii=False, indent=2), encoding="utf-8"
        )
