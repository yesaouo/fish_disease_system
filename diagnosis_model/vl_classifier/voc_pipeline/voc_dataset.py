from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection

from common import crop_bbox, crop_square_with_black_padding, load_label_bank
from voc_labels import VOC_CLASS_TO_ID, VOC_CLASSES_EN


@dataclass
class VocRegionSample:
    image_path: str
    file_name: str
    image_id: str
    bbox_xywh: Tuple[float, float, float, float]
    label_id: int
    label_name: str
    difficult: int
    object_index: int
    text: Optional[str] = None
    evidence_index: int = -1


class VocRegionDataset(Dataset):
    def __init__(
        self,
        root: str,
        year: str,
        image_set: str,
        label_bank_json: str,
        crop_mode: str = "bbox",
        use_multipos: bool = False,
        use_fusion: bool = False,
        return_meta: bool = False,
        download: bool = False,
        skip_difficult: bool = False,
        mode: Optional[str] = None,
    ):
        self.root = root
        self.year = str(year)
        self.image_set = image_set
        self.crop_mode = crop_mode.lower().strip()
        self.skip_difficult = skip_difficult
        self.return_meta = return_meta

        if mode is not None:
            mode = str(mode).lower().strip()
            if mode == "baseline":
                use_multipos = False
                use_fusion = False
                return_meta = False
            elif mode == "ours":
                use_multipos = True
                use_fusion = False
                return_meta = False
            elif mode == "fusion":
                use_multipos = True
                use_fusion = True
                return_meta = False
            elif mode == "eval_local":
                use_multipos = False
                use_fusion = False
                return_meta = True
            elif mode == "eval_fusion":
                use_multipos = False
                use_fusion = True
                return_meta = True
            else:
                raise ValueError(f"Unknown mode: {mode}")
            self.return_meta = return_meta

        self.use_multipos = bool(use_multipos)
        self.use_fusion = bool(use_fusion)

        if self.crop_mode not in ("bbox", "square"):
            raise ValueError("crop_mode must be 'bbox' or 'square'")
        if self.year not in ("2007", "2012"):
            raise ValueError("year must be '2007' or '2012'")
        if self.year == "2012" and self.image_set == "test":
            raise ValueError("VOC2012 的 test annotations 不公開，請改用 val 或 trainval")

        self.captions_by_cat, self.label_map = load_label_bank(label_bank_json)
        for idx, class_name in enumerate(VOC_CLASSES_EN):
            if str(idx) not in self.captions_by_cat:
                raise KeyError(f"label_bank 缺少 VOC 類別 {idx} / {class_name}")

        base = VOCDetection(root=root, year=self.year, image_set=self.image_set, download=download)
        self.image_paths = [str(p) for p in base.images]
        self.ann_paths = [str(p) for p in base.annotations]

        rr_ptr: Dict[int, int] = {}
        samples: List[VocRegionSample] = []
        skipped = 0
        for image_path, ann_path in zip(self.image_paths, self.ann_paths):
            tree = ET.parse(ann_path)
            root_node = tree.getroot()
            filename = (root_node.findtext("filename") or Path(image_path).name).strip()
            image_id = Path(filename).stem
            objects = root_node.findall("object")
            for object_index, obj in enumerate(objects):
                name = (obj.findtext("name") or "").strip()
                if name not in VOC_CLASS_TO_ID:
                    skipped += 1
                    continue
                difficult = int(obj.findtext("difficult") or 0)
                if self.skip_difficult and difficult == 1:
                    skipped += 1
                    continue
                bbox_node = obj.find("bndbox")
                if bbox_node is None:
                    skipped += 1
                    continue
                try:
                    xmin = float(bbox_node.findtext("xmin")) - 1.0
                    ymin = float(bbox_node.findtext("ymin")) - 1.0
                    xmax = float(bbox_node.findtext("xmax"))
                    ymax = float(bbox_node.findtext("ymax"))
                except (TypeError, ValueError):
                    skipped += 1
                    continue
                width = max(1.0, xmax - xmin)
                height = max(1.0, ymax - ymin)
                label_id = VOC_CLASS_TO_ID[name]

                text = None
                evidence_index = -1
                if not self.use_multipos:
                    caps = self.captions_by_cat[str(label_id)]
                    ptr = rr_ptr.get(label_id, 0)
                    text = caps[ptr % len(caps)]
                    rr_ptr[label_id] = ptr + 1

                samples.append(
                    VocRegionSample(
                        image_path=image_path,
                        file_name=filename,
                        image_id=image_id,
                        bbox_xywh=(xmin, ymin, width, height),
                        label_id=label_id,
                        label_name=name,
                        difficult=difficult,
                        object_index=object_index,
                        text=text,
                        evidence_index=evidence_index,
                    )
                )

        self.samples = samples
        self.labels = [s.label_id for s in samples]
        print(
            f"[VOC {self.year} {self.image_set}] samples={len(self.samples)} "
            f"skipped={skipped} crop_mode={self.crop_mode} "
            f"multipos={self.use_multipos} fusion={self.use_fusion} return_meta={self.return_meta}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _crop(self, image: Image.Image, bbox_xywh: Tuple[float, float, float, float]) -> Image.Image:
        if self.crop_mode == "bbox":
            return crop_bbox(image, bbox_xywh)
        return crop_square_with_black_padding(image, bbox_xywh)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_global = Image.open(sample.image_path).convert("RGB")
        image_local = self._crop(image_global, sample.bbox_xywh)

        out = {}
        if self.use_fusion:
            out["image_local"] = image_local
            out["image_global"] = image_global
        else:
            out["image"] = image_local

        if self.use_multipos:
            out["label_id"] = int(sample.label_id)
            out["evidence_index"] = int(sample.evidence_index)
        else:
            out["text"] = str(sample.text)

        if self.return_meta:
            out.update(
                {
                    "bbox_xywh": sample.bbox_xywh,
                    "image_path": sample.image_path,
                    "file_name": sample.file_name,
                    "image_id": sample.image_id,
                    "object_index": sample.object_index,
                    "difficult": sample.difficult,
                    "label_id": int(sample.label_id),
                    "label_name": sample.label_name,
                }
            )

        return out
