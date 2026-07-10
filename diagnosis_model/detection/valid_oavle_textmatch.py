"""OAVLE 文字語意匹配偵測評估（A1）。

與 valid_oavle.py（封閉頭 COCOeval）不同：此檔評估「class-agnostic 偵測 + 文字語意匹配
分類」的完整 OAVLE。單次 joint forward 取每個 query 的 box / objectness / 語意 z，
類別由 argmax(z · 文字錨) 指派、score 取 objectness，再丟 detection_eval_common.run_eval
算多類別 COCO mAP。用於論文表 10 的 OAVLE 列。

  RFDETR_BACKBONE 由 joint checkpoint 自動判定（load_oavle）。

  ⚠️ 用 `checkpoint_best_regular.pth`，勿用 `best_ema`/`best_total`：語意頭 semantic_embed
     是 train_joint 才新加的頭，EMA 追蹤不到 → best_ema 的語意頭是 stale/隨機（分類≈隨機、
     mAP≈0）。best_regular 才是實際訓好的語意頭。

  $PY diagnosis_model/detection/valid_oavle_textmatch.py \
      --dataset_dir data/detection_voc_val \
      --joint_ckpt diagnosis_model/grod/outputs/joint_voc_ca/checkpoint_best_regular.pth \
      --anchors diagnosis_model/grod/outputs/voc_text_anchors.pt \
      --output_dir diagnosis_model/detection/outputs/oavle_voc_textmatch_eval
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)   # detection_eval_common
sys.path.insert(0, _ROOT)   # diagnosis_model.*

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
import detection_eval_common as dec  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OAVLE text-matching detection eval (shared engine)")
    p.add_argument("--dataset_dir", required=True, type=Path,
                   help="多類別 COCO 資料集（GT 需含各 VOC 類 category_id，非 class-agnostic）")
    p.add_argument("--joint_ckpt", required=True, type=str, help="joint 模型（含語意頭）checkpoint")
    p.add_argument("--anchors", required=True, type=str, help="文字錨 .pt（anchor_embs[C,768]）")
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--model_name", type=str, default="OAVLE")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def cxcywh_norm_to_xyxy_abs(boxes: torch.Tensor, W: int, H: int) -> np.ndarray:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    return torch.stack([x1, y1, x2, y2], dim=-1).numpy()


def main() -> None:
    args = parse_args()

    # 開啟語意頭（RFDETRMedium 由 ckpt rebuild 時讀這些 env）
    anc = torch.load(args.anchors, weights_only=False)
    anchor_embs = anc["anchor_embs"] if isinstance(anc, dict) else anc
    os.environ["RFDETR_SEMANTIC_DIM"] = str(anchor_embs.shape[-1])
    os.environ["RFDETR_SEMANTIC_LAYERS"] = "1"
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(args.anchors)

    import diagnosis_model.grod  # noqa: F401  monkeypatch
    from diagnosis_model.grod.build import load_oavle

    device = args.device
    net, resolution, means, stds = load_oavle(args.joint_ckpt, device=device)
    assert hasattr(net, "semantic_embed"), "joint ckpt 未載入語意頭"
    # centering：文字錨於圖文空間高度各向異性（VOC 類名錨 pairwise cos ~0.88），
    # 減均錨後重正規化可去除共享方向、讓 argmax 依判別性成分決定（+~1.4pp）。
    A0 = F.normalize(anchor_embs.float(), dim=-1)
    anchors = F.normalize(A0 - A0.mean(0, keepdim=True), dim=-1).to(device)   # [C, 768] centered

    cfg = dec.EvalConfig(dataset_dir=args.dataset_dir, output_dir=args.output_dir, model_name=args.model_name)
    cats = dec.get_categories(cfg)
    assert len(cats) == anchors.shape[0], f"類別數 {len(cats)} != 錨數 {anchors.shape[0]}"

    @torch.no_grad()
    def predict_fn(image, score_thresh):
        W, H = image.size
        t = TF.to_tensor(image)
        t = TF.resize(t, [resolution, resolution])
        t = TF.normalize(t, means, stds).unsqueeze(0).to(device)
        out = net(t)
        z = F.normalize(out["pred_semantic"][0].float(), dim=-1)         # [Q, 768]
        obj = out["pred_logits"][0].float().sigmoid()[:, 0]              # [Q] class-agnostic objectness
        boxes = out["pred_boxes"][0].detach().float().cpu()             # [Q, 4] cxcywh norm
        xyxy = cxcywh_norm_to_xyxy_abs(boxes, W, H)                     # [Q, 4] abs
        sims = z @ anchors.t()                                          # [Q, C] 對 centered 錨
        cls = sims.argmax(dim=1).cpu().numpy()                          # [Q] 0..C-1（= 文字語意匹配）
        msim = sims.max(dim=1).values.clamp(min=0)                      # [Q] 類別相似度
        sc = (obj * msim).cpu().numpy()                                 # 偵測信心 = objectness × 類別相似度（+~2.2pp vs 只用 obj）
        keep = sc >= score_thresh
        return dec.Detections(xyxy[keep], sc[keep], cls[keep])

    dec.run_eval(cfg, predict_fn, cats)


if __name__ == "__main__":
    main()
