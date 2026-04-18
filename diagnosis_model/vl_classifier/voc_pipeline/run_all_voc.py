from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from voc_labels import save_default_voc_label_bank



def quote(parts):
    return " ".join(shlex.quote(str(p)) for p in parts)



def run(cmd, cwd: Path):
    print("\n>>>", quote(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)



def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run all unified VOC training/evaluation pipelines.")
    ap.add_argument("--voc_root", type=str, required=True)
    ap.add_argument("--year", type=str, choices=["2007", "2012"], default="2007")
    ap.add_argument("--label_bank_json", type=str, default="")
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="google/siglip2-base-patch16-224")
    ap.add_argument("--crop_mode", type=str, choices=["bbox", "square"], default="bbox")
    ap.add_argument("--train_image_set", type=str, default="train")
    ap.add_argument("--valid_image_set", type=str, default="val")
    ap.add_argument("--eval_image_set", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--fusion_batch_size", type=int, default=64)
    ap.add_argument("--num_epochs", type=int, default=10)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--base_learning_rate", type=float, default=3e-5)
    ap.add_argument("--fusion_learning_rate", type=float, default=1e-4)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--skip_difficult_train", action="store_true")
    ap.add_argument("--skip_difficult_eval", action="store_true")
    ap.add_argument("--save_vis", action="store_true")
    ap.add_argument("--no_amp", action="store_true")
    return ap



def main() -> None:
    args = build_argparser().parse_args()
    here = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    label_bank_json = Path(args.label_bank_json) if args.label_bank_json else output_root / "voc_label_bank.json"
    if not label_bank_json.exists():
        save_default_voc_label_bank(label_bank_json)

    eval_image_set = args.eval_image_set.strip() or ("test" if args.year == "2007" else "val")
    common = [
        "--voc_root", args.voc_root,
        "--year", args.year,
        "--crop_mode", args.crop_mode,
        "--label_bank_json", str(label_bank_json),
        "--max_length", str(args.max_length),
        "--num_workers", str(args.num_workers),
        "--device", args.device,
    ]
    if args.download:
        common.append("--download")
    if args.no_amp:
        common.append("--no_amp")

    baseline_dir = output_root / f"voc{args.year}_baseline"
    multipos_dir = output_root / f"voc{args.year}_multipos"
    fusion_dir = output_root / f"voc{args.year}_fusion"
    multipos_fusion_dir = output_root / f"voc{args.year}_multipos_fusion"
    eval_dir = output_root / f"voc{args.year}_eval"

    def train_cmd(extra_flags, out_dir, batch_size, lr=None, fusion_base_lr=None, fusion_head_lr=None):
        cmd = [
            sys.executable,
            str(here / "train.py"),
            "--model_name", args.model_name,
            "--train_image_set", args.train_image_set,
            "--valid_image_set", args.valid_image_set,
            "--output_dir", str(out_dir),
            "--batch_size", str(batch_size),
            "--num_epochs", str(args.num_epochs),
            *common,
            *extra_flags,
        ]
        if lr is not None:
            cmd += ["--learning_rate", str(lr)]
        if fusion_base_lr is not None:
            cmd += ["--fusion_base_lr", str(fusion_base_lr)]
        if fusion_head_lr is not None:
            cmd += ["--fusion_head_lr", str(fusion_head_lr)]
        if args.skip_difficult_train:
            cmd.append("--skip_difficult_train")
        if args.skip_difficult_eval:
            cmd.append("--skip_difficult_valid")
        return cmd

    run(train_cmd([], baseline_dir, args.batch_size, lr=args.learning_rate), cwd=here)
    run(train_cmd(["--multipos"], multipos_dir, args.batch_size, lr=args.learning_rate), cwd=here)
    run(train_cmd(["--fusion"], fusion_dir, args.fusion_batch_size, fusion_base_lr=args.base_learning_rate, fusion_head_lr=args.fusion_learning_rate), cwd=here)
    run(train_cmd(["--multipos", "--fusion"], multipos_fusion_dir, args.fusion_batch_size, fusion_base_lr=args.base_learning_rate, fusion_head_lr=args.fusion_learning_rate), cwd=here)

    eval_common = [
        sys.executable,
        str(here / "eval.py"),
        "--voc_root", args.voc_root,
        "--year", args.year,
        "--image_set", eval_image_set,
        "--label_bank_json", str(label_bank_json),
        "--output_dir", str(eval_dir),
        "--crop_mode", args.crop_mode,
        "--device", args.device,
        "--max_length", str(args.max_length),
        "--img_batch_size", str(args.fusion_batch_size),
        "--text_batch_size", "256",
        "--model", f"zeroshot={args.model_name}",
        "--model", f"baseline={baseline_dir}",
        "--model", f"multipos={multipos_dir}",
        "--model", f"fusion={fusion_dir}",
        "--model", f"multipos_fusion={multipos_fusion_dir}",
    ]
    if args.download:
        eval_common.append("--download")
    if args.skip_difficult_eval:
        eval_common.append("--skip_difficult")
    if args.no_amp:
        eval_common.append("--no_amp")
    if args.save_vis:
        eval_common.append("--save_vis")

    run(eval_common, cwd=here)

    print("\nAll done.")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
