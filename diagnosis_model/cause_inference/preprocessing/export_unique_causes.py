from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .cause_texts import collect_cause_strings_from_coco
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from diagnosis_model.cause_inference.preprocessing.cause_texts import (  # type: ignore
        collect_cause_strings_from_coco,
    )


DEFAULT_COCO_FILES = [
    "data/detection/coco/_merged/train/_annotations.coco.json",
    "data/detection/coco/_merged/valid/_annotations.coco.json",
]
DEFAULT_OUTPUT = "diagnosis_model/cause_inference/outputs/cause.txt"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export deduplicated fish disease cause strings, one cause per line.",
    )
    parser.add_argument(
        "--coco_files",
        nargs="+",
        default=DEFAULT_COCO_FILES,
        help="COCO annotation JSON files to read.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output txt path. Each line contains one deduplicated cause string.",
    )
    parser.add_argument(
        "--include_healthy",
        action="store_true",
        help="Also collect causes from images marked isHealthy=true.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    causes = collect_cause_strings_from_coco(
        args.coco_files,
        skip_healthy=not args.include_healthy,
        verbose=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(causes) + ("\n" if causes else ""), encoding="utf-8")

    print(f"[export] wrote {len(causes)} unique cause strings to {output_path}")


if __name__ == "__main__":
    main()
