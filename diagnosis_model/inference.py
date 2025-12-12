from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .config import CROSS_ALIGN_CHECKPOINT, DEFAULT_TEXT, DISCLAIMER, LABELS
from .cross_align import CrossAlignFormer
from .detection import RFDETR
from .kb import KnowledgeBase
from .text_encoder import EmbeddingGemma
from .types import ActionCandidate, CauseCandidate, Detection, Report
from .vision import VisionBackbone


class FishDiseaseSystem:
    """
    影像為主、文字為輔的魚病多模態診斷系統推論流程：
      1) 影像全域特徵 + 偵測框
      2) 無 ABNORMAL 則健康早退
      3) 文本編碼（句向量）
      4) CrossAlignFormer 對齊後產生 cause/treat 檢索 query
      5) 檢索 KB 並去重
      6) 整理 detection + 檢索輸出
    """

    def __init__(self, align_ckpt: Optional[str] = None) -> None:
        self.vision = VisionBackbone()
        self.detector = RFDETR()
        self.text_enc = EmbeddingGemma()
        self.align = CrossAlignFormer()
        self._load_align_weights(align_ckpt or CROSS_ALIGN_CHECKPOINT)
        self.kb = KnowledgeBase(encoder=self.text_enc)
        self._disclaimer = DISCLAIMER

    def _load_align_weights(self, ckpt_path: Optional[str]) -> None:
        if not ckpt_path:
            return
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"CrossAlign checkpoint not found: {ckpt_path}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.align.load_state_dict(state, strict=True)
        self.align.eval()

    def _to_image(self, image: Image.Image | str | Path) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def _extract_roi_feats(self, im: Image.Image, boxes: np.ndarray) -> np.ndarray:
        """Crop per-box regions and run DINOv3 to obtain ROI tokens."""
        feats = []
        W, H = im.size
        for b in boxes:
            x1 = float(max(0.0, min(W - 1, b[0])))
            y1 = float(max(0.0, min(H - 1, b[1])))
            x2 = float(max(x1 + 1.0, min(W, b[2])))
            y2 = float(max(y1 + 1.0, min(H, b[3])))
            crop = im.crop((x1, y1, x2, y2))
            feats.append(self.vision.extract(crop))
        if not feats:
            return np.zeros((0, self.vision.dim), dtype=np.float32)
        return np.stack(feats, axis=0).astype(np.float32, copy=False)

    def _format_detections(self, boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> list[Detection]:
        out: list[Detection] = []
        for box, lbl, score in zip(boxes, labels, scores):
            lbl_idx = int(lbl)
            lbl_str = LABELS[lbl_idx] if 0 <= lbl_idx < len(LABELS) else str(lbl_idx)
            out.append(
                Detection(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    label=lbl_str,
                    score=float(score),
                )
            )
        return out

    def _dedup_causes(self, q_cause: np.ndarray) -> list[CauseCandidate]:
        raw = self.kb.search_causes(q_cause, topk=1)  # [(id, desc, sim)]
        best: dict[str, tuple[str, float]] = {}
        for cid, desc, sim in raw:
            if (cid not in best) or (sim > best[cid][1]):
                best[cid] = (desc, sim)
        items = [CauseCandidate(cause_id=k, desc=v[0], similarity=float(v[1])) for k, v in best.items()]
        items.sort(key=lambda x: x.similarity, reverse=True)
        return items

    def _dedup_actions(self, q_treat: np.ndarray) -> list[ActionCandidate]:
        raw = self.kb.search_actions(q_treat, topk=1)  # [(id, desc, sim, note)]
        best: dict[str, tuple[str, float, Optional[str]]] = {}
        for aid, desc, sim, note in raw:
            if (aid not in best) or (sim > best[aid][1]):
                best[aid] = (desc, sim, note)
        items = [
            ActionCandidate(action_id=k, desc=v[0], similarity=float(v[1]), safety_note=v[2]) for k, v in best.items()
        ]
        items.sort(key=lambda x: x.similarity, reverse=True)
        return items

    def infer(self, image: Image.Image | str | Path, text: Optional[str]) -> Report:
        # 1) 影像特徵與偵測
        im = self._to_image(image)
        np_im = np.array(im)
        global_token = self.vision.extract(im)
        det = self.detector.infer(np_im)

        # det.labels is converted to 0-based in RFDETR (0=HEALTHY, 1=ABNORMAL)
        detections = self._format_detections(det.boxes, det.labels, det.scores)
        has_abnormal = bool(np.any(det.labels == 1))

        # 1.1 無 ABNORMAL：健康路徑早退
        if not has_abnormal:
            summary = "魚體外觀正常，無明顯異常表徵"
            return Report(
                detections=detections,
                causes=[],
                actions=[],
                knowledge_base_version=self.kb.version,
                disclaimer=self._disclaimer,
                summary=summary,
            )

        # 2) ROI 特徵（以偵測框裁切、縮放後送入 DINOv3 backbone）
        roi_feats = self._extract_roi_feats(im, det.boxes)

        # 3) 文本編碼（句向量）
        text_value = text.strip() if text and text.strip() else DEFAULT_TEXT
        text_emb = self.text_enc.encode(text_value)

        # 4) CrossAlignFormer 推出 cause / treat 槽位 query
        device = next(self.align.parameters()).device

        def _to_batch(x: np.ndarray, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
            t = torch.from_numpy(x)
            if dtype:
                t = t.to(dtype=dtype)
            return t.unsqueeze(0).to(device)

        global_token_t = _to_batch(global_token, dtype=torch.float32)  # (1, dim)
        roi_feats_t = _to_batch(roi_feats, dtype=torch.float32)        # (1, K, dim)
        boxes_norm_t = _to_batch(det.boxes_norm, dtype=torch.float32)  # (1, K, 4)
        class_ids_t = _to_batch(det.labels, dtype=torch.long)          # (1, K)
        text_emb_t = _to_batch(text_emb, dtype=torch.float32)          # (1, dim)

        with torch.no_grad():
            _, q_cause, q_treat = self.align.forward(
                global_token=global_token_t,
                roi_feats=roi_feats_t,
                boxes_norm=boxes_norm_t,
                class_ids=class_ids_t,
                text_emb=text_emb_t,
            )

        q_cause = q_cause.squeeze(0).cpu().numpy()
        q_treat = q_treat.squeeze(0).cpu().numpy()

        # 5) 檢索與去重
        causes = self._dedup_causes(q_cause)
        actions = self._dedup_actions(q_treat)

        # 6) 檢索空槽 fallback（僅在已確認存在 ABNORMAL 的情境）
        if len(causes) == 0:
            causes = [CauseCandidate(cause_id="CAUSE_pending_investigation", desc="異常尚待釐清", similarity=0.0)]
        if len(actions) == 0:
            actions = [ActionCandidate(action_id="TREAT_observe_followup", desc="持續觀察追蹤", similarity=0.0)]

        return Report(
            detections=detections,
            causes=causes,
            actions=actions,
            knowledge_base_version=self.kb.version,
            disclaimer=self._disclaimer,
        )


def report_to_json(report: Report) -> str:
    """
    轉成外部友善的 JSON（欄位名對照架構文件）。
    """
    dets = []
    for d in report.detections:
        dets.append({"bbox": [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]], "class": d.label, "score": d.score})

    causes = []
    for c in report.causes:
        causes.append({"cause_id": c.cause_id, "desc": c.desc, "similarity": c.similarity})

    acts = []
    for a in report.actions:
        item = {"action_id": a.action_id, "desc": a.desc, "similarity": a.similarity}
        if a.safety_note:
            item["safety_note"] = a.safety_note
        acts.append(item)

    obj = {
        "detections": dets,
        "causes": causes,
        "actions": acts,
        "knowledge_base_version": report.knowledge_base_version,
        "disclaimer": report.disclaimer,
    }
    if report.summary:
        obj["summary"] = report.summary
    return json.dumps(obj, ensure_ascii=False, indent=2)
