from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SPLIT_NAMES = ("train", "valid", "test")


@dataclass
class ImageEntry:
    id: int
    dataset: str
    original_filename: str
    task_id: str
    split: str
    first_seen: str


def _stable_split(filename: str, ratios: tuple[float, ...]) -> str:
    total = sum(ratios)
    normalized = [r / total for r in ratios]
    h = hashlib.md5(filename.encode("utf-8")).hexdigest()
    bucket = (int(h[:8], 16) % 10_000) / 10_000.0
    cum = 0.0
    for name, r in zip(SPLIT_NAMES, normalized):
        cum += r
        if bucket < cum:
            return name
    return SPLIT_NAMES[-1]


class ImageRegistry:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path):
        self.path = path
        self.next_id: int = 1
        self.images: dict[int, ImageEntry] = {}
        self.index: dict[tuple[str, str], int] = {}

    @classmethod
    def load(cls, path: Path) -> "ImageRegistry":
        reg = cls(path)
        if not path.exists():
            return reg
        data = json.loads(path.read_text(encoding="utf-8"))
        reg.next_id = int(data.get("next_id", 1))
        for id_str, entry in data.get("images", {}).items():
            img_id = int(id_str)
            ie = ImageEntry(
                id=img_id,
                dataset=entry["dataset"],
                original_filename=entry["original_filename"],
                task_id=entry["task_id"],
                split=entry["split"],
                first_seen=entry["first_seen"],
            )
            reg.images[img_id] = ie
            reg.index[(ie.dataset, ie.original_filename)] = img_id
        return reg

    def get_or_create(
        self,
        dataset: str,
        original_filename: str,
        task_id: str,
        split_ratios: tuple[float, ...],
    ) -> tuple[int, bool]:
        """Return (image_id, is_new). Assigns next_id + stable split if new."""
        key = (dataset, original_filename)
        existing = self.index.get(key)
        if existing is not None:
            return existing, False

        img_id = self.next_id
        self.next_id += 1
        entry = ImageEntry(
            id=img_id,
            dataset=dataset,
            original_filename=original_filename,
            task_id=task_id,
            split=_stable_split(original_filename, split_ratios),
            first_seen=datetime.now(timezone.utc).isoformat(),
        )
        self.images[img_id] = entry
        self.index[key] = img_id
        return img_id, True

    def split_of(self, image_id: int) -> str:
        return self.images[image_id].split

    def save(self) -> None:
        data = {
            "schema_version": self.SCHEMA_VERSION,
            "next_id": self.next_id,
            "images": {
                str(e.id): {
                    "dataset": e.dataset,
                    "original_filename": e.original_filename,
                    "task_id": e.task_id,
                    "split": e.split,
                    "first_seen": e.first_seen,
                }
                for e in sorted(self.images.values(), key=lambda x: x.id)
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
