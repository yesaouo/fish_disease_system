from __future__ import annotations
import argparse
import os

from rfdetr import RFDETRMedium


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RF-DETR with rfdetr")
    p.add_argument("--dataset_dir", type=str, required=True,
                   help="包含 train/、valid/（與可選 test/）且每個 split 具有 _annotations.coco.json 與影像檔")
    p.add_argument("--output_dir", type=str, default=None,
                   help="訓練輸出資料夾（預設為 dataset_dir/outputs/rfdetr）")

    # 僅保留常用的幾個關鍵超參；其餘均交由 rfdetr 預設
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum_steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # 若未指定 output_dir，則自動用 dataset_dir/outputs/rfdetr
    if args.output_dir is None:
        args.output_dir = os.path.join(args.dataset_dir, "outputs", "rfdetr")
    os.makedirs(args.output_dir, exist_ok=True)
    
    model = RFDETRMedium(num_classes=2)
    
    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        output_dir=args.output_dir,
        early_stopping=True,
    )

    print("訓練完成，檢查輸出：", args.output_dir)


if __name__ == "__main__":
    main()
