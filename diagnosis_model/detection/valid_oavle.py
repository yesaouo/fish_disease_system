"""OAVLE 偵測評估（DINOv3 backbone）。

`valid_rfdetr.py` 專供 stock RF-DETR baseline、`valid_yolo.py` 專供 YOLO；OAVLE 為
DINOv3-backbone 的整合偵測器，須先套 grod monkeypatch 才能正確建出 backbone，故另開
此檔。COCOeval 評估邏輯（含 AP@50 / AP@50:95 / 各尺度 AP、圖表、混淆矩陣）完全複用
`valid_rfdetr`；此檔僅負責：(1) 於載入 rfdetr 前套上 grod monkeypatch，(2) 由
checkpoint 權重維度自動判定 `RFDETR_BACKBONE`，再委派給 `valid_rfdetr` 的 main。

用法與 valid_rfdetr 完全相同（backbone 可不設，會從 checkpoint 自動判定）：

    PY=/home/lab603/anaconda3/envs/SDM/bin/python
    $PY diagnosis_model/detection/valid_oavle.py \
        --dataset_dir data/detection_voc \
        --checkpoint_path diagnosis_model/detection/outputs/rfdetr_voc_small_ft/checkpoint_best_ema.pth \
        --output_dir diagnosis_model/detection/outputs/rfdetr_voc_small_ft/eval_val
"""
from __future__ import annotations

import os
import sys

# repo root 上 sys.path，再套 monkeypatch —— 必須早於任何 rfdetr import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import diagnosis_model.grod  # noqa: F401,E402  applies rfdetr monkeypatch (backbone swap)
from diagnosis_model.grod.build import _backbone_from_ckpt  # noqa: E402


def _autoset_backbone(argv: list[str]) -> None:
    """未指定 RFDETR_BACKBONE 時，由 --checkpoint_path 權重維度自動判定（同 build.py 慣例）。

    DINOv3 的 mask_token 維度 384/768/1024 → dinov3_small/base/large；缺此設定會靜默
    退回 stock DINOv2、丟掉 DINOv3 權重而產生垃圾特徵，故在此強制對齊。
    """
    if os.environ.get("RFDETR_BACKBONE"):
        return
    if "--checkpoint_path" in argv:
        ckpt = argv[argv.index("--checkpoint_path") + 1]
        bb = _backbone_from_ckpt(ckpt)
        if bb:
            os.environ["RFDETR_BACKBONE"] = bb
            print(f"[valid_oavle] auto-set RFDETR_BACKBONE={bb} (from checkpoint)")


if __name__ == "__main__":
    _autoset_backbone(sys.argv)
    import runpy

    # 委派給 valid_rfdetr 的 __main__（rfdetr 已被 monkeypatch、backbone env 已設）
    runpy.run_path(
        os.path.join(os.path.dirname(__file__), "valid_rfdetr.py"),
        run_name="__main__",
    )
