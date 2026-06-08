"""Joint gate eval: abstain (reject healthy) + lesion selection, end-to-end.

One gate does both jobs in deployment, so evaluate them together over healthy +
diseased val. Three gates compared:

  A  const         : select {obj ≥ τ*}; image=diseased iff any selected (abstain = none)
  B  disease-head  : τ(g) selection + learned abstain p(g,max_w) < disease_tau  (current production)
  C  decouple      : const τ* selection + disease-head abstain p<disease_tau    (the recommendation)

Metrics (correct GT→selected box matching; healthy boxes are all FP):
  box  P/R/F1   — unified lesion-detection quality over ALL images
  image sens    — diseased images flagged diseased (≥1 box kept)
  image spec    — healthy images rejected (0 boxes kept)

Run from repo root (SDM env):
    $PY -m diagnosis_model.grod.eval_gate_joint
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from diagnosis_model.grod.extract_hs import (
    xywh_to_xyxy, iou_matrix, greedy_match, cxcywh_norm_to_xyxy_abs,
)
from diagnosis_model.grod.disease_head import load_disease_head

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def f1(TP, FP, FN):
    P = TP / (TP + FP + 1e-9); R = TP / (TP + FN + 1e-9)
    return P, R, 2 * P * R / (P + R + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ART = "data/processed/current/artifacts"
    ap.add_argument("--joint_ckpt", default=f"{ART}/models/joint_rfdetr/checkpoint_best_regular.pth")
    ap.add_argument("--global_sd", default=f"{ART}/models/distilled_global_rawP/global_embed_state_dict.pt")
    ap.add_argument("--anchors", default=f"{ART}/models/text_anchors.pt")
    ap.add_argument("--disease_ckpt", default=f"{ART}/models/disease_head/disease_head.pt")
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--img_root", default="data/processed/current/detection")
    ap.add_argument("--healthy_dir", default="data/healthy_images")
    ap.add_argument("--tau_const", type=float, default=0.5)
    ap.add_argument("--iou_gt", type=float, default=0.5)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    os.environ["RFDETR_SEMANTIC_DIM"] = "768"
    os.environ["RFDETR_SEMANTIC_ANCHORS"] = os.path.abspath(args.anchors)
    os.environ["RFDETR_GLOBAL_DIM"] = "768"
    from rfdetr import RFDETRMedium
    rf = RFDETRMedium(pretrain_weights=args.joint_ckpt, num_classes=1)
    net = rf.model.model.to(dev).eval()
    net.global_embed.load_state_dict(torch.load(args.global_sd, map_location=dev))
    res = int(rf.model.resolution); means, stds = list(rf.means), list(rf.stds)
    head, disease_tau = load_disease_head(args.disease_ckpt, dev)

    # val images: diseased (valid_cases) + healthy (val split of healthy_images)
    dis = [(c["file_name"], c["lesion_boxes_xywh"], True)
           for c in torch.load(Path(args.case_db_dir) / "valid_cases.pt", weights_only=False)]
    healthy = sorted(p for p in Path(args.healthy_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    random.Random(args.seed).shuffle(healthy)
    hv = healthy[:int(len(healthy) * args.val_frac)]
    print(f"[joint] diseased={len(dis)} healthy={len(hv)}")

    # accumulators per gate: [TP,FP,FN], image sens (dis flagged), spec (healthy rejected)
    G = {k: {"box": [0, 0, 0], "dis_flag": 0, "h_rej": 0} for k in ("A_const", "B_head", "C_decouple")}

    @torch.no_grad()
    def fwd(image):
        px = TF.normalize(TF.resize(TF.to_tensor(image), [res, res]), means, stds).unsqueeze(0).to(dev)
        out = net(px)
        obj = out["pred_logits"][0, :, 0].sigmoid()
        g = out["pred_global"][0].float()
        boxes = out["pred_boxes"][0].cpu()
        return obj.cpu(), g, boxes

    def selected_indices(obj, g, gate):
        max_w = obj.max().unsqueeze(0).to(dev)
        p, tau = head(g.unsqueeze(0), max_w)
        abstain_head = p.item() < disease_tau
        if gate == "A_const":
            keep = obj >= args.tau_const
        elif gate == "B_head":
            if abstain_head:
                return torch.zeros(0, dtype=torch.long)
            keep = obj >= tau.item()
        else:  # C_decouple
            if abstain_head:
                return torch.zeros(0, dtype=torch.long)
            keep = obj >= args.tau_const
        return torch.where(keep)[0]

    def box_score(keep, pred_xyxy, gt_xyxy):
        if keep.numel() == 0:
            return 0, 0, gt_xyxy.size(0)
        assign = greedy_match(iou_matrix(gt_xyxy, pred_xyxy[keep]), args.iou_gt)
        m = sum(1 for q in assign if q >= 0)
        return m, keep.numel() - m, gt_xyxy.size(0) - m   # TP, FP, FN

    for fn, gt, is_d in dis:
        image = Image.open(Path(args.img_root) / "valid" / fn).convert("RGB")
        W, H = image.size
        obj, g, boxes = fwd(image); image.close()
        pred_xyxy = cxcywh_norm_to_xyxy_abs(boxes, W, H)
        gt = gt.tolist() if torch.is_tensor(gt) else gt
        gt_xyxy = torch.tensor([xywh_to_xyxy(tuple(b)) for b in gt], dtype=torch.float32) if gt else torch.zeros(0, 4)
        for gate in G:
            keep = selected_indices(obj, g, gate)
            tp, fp, fn_ = box_score(keep, pred_xyxy, gt_xyxy)
            G[gate]["box"][0] += tp; G[gate]["box"][1] += fp; G[gate]["box"][2] += fn_
            G[gate]["dis_flag"] += int(keep.numel() > 0)

    for p in hv:
        image = Image.open(p).convert("RGB")
        obj, g, boxes = fwd(image); image.close()
        for gate in G:
            keep = selected_indices(obj, g, gate)
            G[gate]["box"][1] += keep.numel()                # healthy: every kept box is FP
            G[gate]["h_rej"] += int(keep.numel() == 0)

    nd, nh = len(dis), len(hv)
    print(f"\n{'gate':<12}{'box_F1':>8}{'box_P':>8}{'box_R':>8}{'img_sens':>10}{'img_spec':>10}")
    for gate, lab in [("A_const", "A const"), ("B_head", "B disease-head"), ("C_decouple", "C decouple")]:
        P, R, Fb = f1(*G[gate]["box"])
        sens = G[gate]["dis_flag"] / nd; spec = G[gate]["h_rej"] / nh
        print(f"{lab:<14}{Fb:>8.3f}{P:>8.3f}{R:>8.3f}{sens:>10.3f}{spec:>10.3f}")
    print(f"\n  τ_const={args.tau_const}  disease_tau(abstain)={disease_tau:.3f}  "
          f"(A: no learned abstain; B: τ(g) both; C: const select + learned abstain)")


if __name__ == "__main__":
    main()
