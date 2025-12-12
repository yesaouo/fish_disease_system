from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rfdetr import RFDETRMedium

from .config import CHECKPOINT_PATH, LABELS


@dataclass
class RFDetrOutput:
    boxes: np.ndarray         # (K,4) pixel coords [x1,y1,x2,y2]
    boxes_norm: np.ndarray    # (K,4) in [0,1] normalized to image size
    labels: np.ndarray        # (K,)
    scores: np.ndarray        # (K,)


class RFDETR:
    def __init__(self, top_k: int = 10, score_thresh: float = 0.5) -> None:
        self.top_k = top_k
        self.score_thresh = score_thresh
        self.model = RFDETRMedium(
            pretrain_weights=CHECKPOINT_PATH,
            num_classes=len(LABELS),
        )
        self.model.optimize_for_inference(compile=False)

    def infer(self, img_rgb: np.ndarray) -> RFDetrOutput:
        detections = self.model.predict(img_rgb, threshold=self.score_thresh)

        boxes = np.asarray(detections.xyxy, dtype=np.float32)        # (N, 4)
        labels = np.asarray(detections.class_id, dtype=np.int64)     # (N,)
        scores = np.asarray(detections.confidence, dtype=np.float32) # (N,)

        h, w = img_rgb.shape[:2]
        boxes_norm = boxes.copy()
        boxes_norm[:, [0, 2]] /= float(w)
        boxes_norm[:, [1, 3]] /= float(h)
        boxes_norm = np.clip(boxes_norm, 0.0, 1.0)

        keep = scores >= self.score_thresh
        boxes = boxes[keep]
        boxes_norm = boxes_norm[keep]
        labels = labels[keep]
        scores = scores[keep]

        if boxes.shape[0] > self.top_k:
            order = np.argsort(scores)[::-1][: self.top_k]
            boxes = boxes[order]
            boxes_norm = boxes_norm[order]
            labels = labels[order]
            scores = scores[order]

        return RFDetrOutput(
            boxes=boxes,
            boxes_norm=boxes_norm,
            labels=labels,
            scores=scores,
        )
