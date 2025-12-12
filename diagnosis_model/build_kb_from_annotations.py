from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .config import KB_VERSION, ANNOTATIONS_PATH, KNOWLEDGE_BASE_DIR


def load_annotation_counts(ann_path: Path) -> tuple[Counter[str], Counter[str]]:
    """Scan JSONL annotations and count distinct cause/treat texts."""
    causes = Counter()
    treats = Counter()

    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for text in rec.get("causes", []):
                t = text.strip()
                if t:
                    causes[t] += 1
            for text in rec.get("treats", []):
                t = text.strip()
                if t:
                    treats[t] += 1
    return causes, treats


def write_jsonl(counter: Counter[str], prefix: str, version: str, out_path: Path) -> None:
    """Write a JSONL file sorted by frequency then text for determinism."""
    version_tag = version.replace("-", "_")
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, (desc, _) in enumerate(items, start=1):
            obj = {"id": f"{prefix}_{idx:04d}_{version_tag}", "desc": desc}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_kb(ann_path: Path, kb_dir: Path, version: str) -> None:
    causes, treats = load_annotation_counts(ann_path)

    write_jsonl(causes, "CAUSE", version, kb_dir / "causes.jsonl")
    write_jsonl(treats, "TREAT", version, kb_dir / "actions.jsonl")

    print(f"Wrote {len(causes)} causes -> {kb_dir / 'causes.jsonl'}")
    print(f"Wrote {len(treats)} actions -> {kb_dir / 'actions.jsonl'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build knowledge_base JSONL files from annotations.jsonl.")

    p.add_argument("--annotations", type=Path, default=Path(ANNOTATIONS_PATH), help="Path to annotations.jsonl")
    p.add_argument(
        "--kb-dir",
        type=Path,
        default=Path(KNOWLEDGE_BASE_DIR),
        help="Output directory containing causes.jsonl and actions.jsonl",
    )
    p.add_argument(
        "--version",
        type=str,
        default=KB_VERSION,
        help="Version tag appended to generated IDs (default: config.KB_VERSION)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_kb(args.annotations, args.kb_dir, args.version)


if __name__ == "__main__":
    main()
