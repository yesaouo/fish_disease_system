from __future__ import annotations

import argparse
from pathlib import Path

from .config import CROSS_ALIGN_CHECKPOINT
from .inference import FishDiseaseSystem, report_to_json


def main() -> None:
    p = argparse.ArgumentParser(description="Fish disease multimodal diagnosis CLI")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--text", default=None, help="Optional text description; defaults to config.DEFAULT_TEXT")
    p.add_argument("--out", default=None, help="Output JSON path; print to stdout if omitted")
    p.add_argument(
        "--align-ckpt",
        default=None,
        help=f"Path to CrossAlignFormer checkpoint (default: {CROSS_ALIGN_CHECKPOINT})",
    )
    args = p.parse_args()

    system = FishDiseaseSystem(align_ckpt=args.align_ckpt)
    report = system.infer(args.image, args.text)
    js = report_to_json(report)

    if args.out:
        Path(args.out).write_text(js, encoding="utf-8")
        print(f"Saved: {args.out}")
    else:
        print(js)


if __name__ == "__main__":
    main()
