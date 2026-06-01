"""Joint detection + semantic-head training (step 2).

Resumes the fish-finetuned RF-DETR and trains the decoder + new semantic head
(backbone frozen) so the decoder query features (hs) learn to project into the
frozen SigLIP2 text space — pushing the by-construction faithfulness signal from
the weak frozen-probe regime (+0.0063) up to a usable strength.

Design (locked):
  - backbone FROZEN (freeze_encoder=True): protects detection, lets decoder+head move
  - semantic NOT in Hungarian matching cost: pure aux loss on matched queries
  - merged COCO dataset with per-box symptom_category_id feeds loss_semantic

The semantic head is enabled via env vars consumed by the fork's build_namespace:
  RFDETR_SEMANTIC_DIM / RFDETR_SEMANTIC_ANCHORS / RFDETR_SEMANTIC_LOSS_COEF / RFDETR_SEMANTIC_TEMP

Run from repo root:
  PY=/home/lab603/anaconda3/envs/SDM/bin/python
  $PY -m diagnosis_model.grod.train_joint \
      --dataset_dir data/detection/coco/_merged_semantic \
      --pretrain_weights diagnosis_model/detection/outputs/rfdetr/checkpoint_best_total.pth \
      --anchors diagnosis_model/grod/outputs/text_anchors.pt \
      --output_dir diagnosis_model/grod/outputs/joint_rfdetr \
      --epochs 30 --semantic_loss_coef 2.0
"""

from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser(description="Joint detection+semantic RF-DETR training.")
    ap.add_argument("--dataset_dir", type=str, required=True,
                    help="merged COCO dir (train/valid with symptom_category_id)")
    ap.add_argument("--pretrain_weights", type=str, required=True)
    ap.add_argument("--anchors", type=str, required=True,
                    help="text_anchors.pt ([C, D] frozen SigLIP2 anchors)")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--semantic_dim", type=int, default=768)
    ap.add_argument("--semantic_layers", type=int, default=1,
                    help="# of last decoder layers mean-pooled into the (still-Linear) semantic head; "
                         "1 = byte-identical current behavior. Ablation knob — sweep {1,2,3}.")
    ap.add_argument("--semantic_loss_coef", type=float, default=2.0)
    ap.add_argument("--semantic_temp", type=float, default=0.07)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    # Enable the semantic head through the fork's env-var hook (build_namespace).
    os.environ["RFDETR_SEMANTIC_DIM"] = str(args.semantic_dim)
    os.environ["RFDETR_SEMANTIC_LAYERS"] = str(args.semantic_layers)
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(args.anchors)
    os.environ["RFDETR_SEMANTIC_LOSS_COEF"] = str(args.semantic_loss_coef)
    os.environ["RFDETR_SEMANTIC_TEMP"] = str(args.semantic_temp)

    from rfdetr import RFDETRMedium

    os.makedirs(args.output_dir, exist_ok=True)

    # freeze_encoder=True -> backbone frozen; decoder + semantic head stay trainable.
    model = RFDETRMedium(
        pretrain_weights=args.pretrain_weights,
        num_classes=1,            # class-agnostic ABNORMAL detector (unchanged)
        freeze_encoder=True,
    )

    model.train(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        output_dir=args.output_dir,
        device=args.device,
        early_stopping=False,
    )
    print(f"[done] joint training -> {args.output_dir}")


if __name__ == "__main__":
    main()
