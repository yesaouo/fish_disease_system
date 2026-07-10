# EVAL.md — 偵測效能比較 runbook

一份指令、兩張表:**表 9**（病灶偵測，`tab:lesion_detection_results`）與 **表 10**（公開資料集
PASCAL VOC，`tab:voc_detection`）。兩者只差**資料路徑**，故合併為單一參數化流程。

比較三個模型，**同一 valid set、同一套 pycocotools `COCOeval`**（bbox mAP、`score_thresh=0.001`、
AP@50 / AP@50:95 / AP_S/M/L），輸出同格式 `metrics_all_splits.json`、數字直接併排：

| 模型 | 角色 | 類別判定 | eval 入口 | train |
|---|---|---|---|---|
| **OAVLE** | 本研究整合模型 | 文字語意匹配（z·文字錨） | `valid_oavle.py` | **不在此**（見下方說明） |
| **RF-DETR Medium** | baseline（stock DINOv2） | 封閉類別分類頭 | `valid_rfdetr.py` | 在此 |
| **YOLO11m** | baseline（Ultralytics） | 封閉類別分類頭 | `valid_yolo.py` | 在此 |

- **表 9 為 class-agnostic 單類（`ABNORMAL`）偵測——只判異常、不分類**，故上表「類別判定」欄
  **僅適用於表 10**（VOC 20 類）；魚病症狀分類（14 類）為另一獨立任務（§5.4，語意頭 z·症狀錨），
  不在本偵測 runbook。
- 三支 eval 共用 `detection_eval_common.py` 引擎；各司其職、**勿混用**（valid_rfdetr 只服務 stock
  RF-DETR、valid_oavle 才認 DINOv3 backbone 並自 checkpoint 判定）。
- **OAVLE 的訓練屬 `BUILD_PIPELINE.md`（grod 流程），EVAL.md 只評估、不訓練。** OAVLE ≠ RF-DETR
  baseline（不同 backbone、不同類別判定），兩者各自獨立評估。

## 環境

```bash
PY=/home/lab603/anaconda3/envs/SDM/bin/python
cd /mnt/ssd/YJ/fish_disease_system     # 一律 repo root
# ultralytics 已於 SDM env；若換機器 $PY -m pip install ultralytics
```

## 選資料集（設路徑變數，下面所有指令共用）

```bash
# --- 表 9 病灶偵測（current 樹）---
DET=data/processed/current/detection          # COCO 格式（RF-DETR/OAVLE/YOLO-eval 都吃這個）
YOLO=data/detection/yolo/_current             # YOLO 訓練格式
OAVLE_CKPT=diagnosis_model/grod/outputs/joint_rfdetr/checkpoint_best_ema.pth

# --- 表 10 公開 VOC（先跑 V0 資料準備，見下）---
DET=data/detection_voc
YOLO=data/detection_voc_yolo
OAVLE_CKPT=diagnosis_model/detection/outputs/rfdetr_voc_small_ft/checkpoint_best_ema.pth
```

> 表 10 專屬前置（VOC 不在 current 樹）：
> ```bash
> $PY dataset_pipeline/voc2coco.py                       # VOC→COCO：train=07+12 trainval / valid=07 test（跳過 0-byte 壞圖）→ data/detection_voc
> $PY dataset_pipeline/coco2yolo.py --data_root $DET --output_dir $YOLO   # COCO→YOLO（同 split）
> ```
> 表 9 的 `$DET`/`$YOLO` 由既有 pipeline 產生，見 dataset_pipeline / BUILD_PIPELINE。

---

## OAVLE（僅評估）

```bash
$PY diagnosis_model/detection/valid_oavle.py \
  --dataset_dir $DET \
  --checkpoint_path $OAVLE_CKPT \
  --output_dir diagnosis_model/detection/outputs/oavle_eval
```

- backbone 由 checkpoint 權重維度自動判定（384/768/1024 → dinov3_small/base/large），免設 env。
- OAVLE checkpoint 來源:表 9 = BUILD_PIPELINE 的 `joint_rfdetr`；表 10 = VOC 版 OAVLE（見
  BUILD_PIPELINE 之 VOC 變體）。

---

## RF-DETR Medium baseline（訓練 + 評估）

```bash
# 訓練:不加 --freeze_encoder = 微調整個 backbone（stock DINOv2）
$PY diagnosis_model/detection/train_rfdetr.py \
  --dataset_dir $DET \
  --output_dir diagnosis_model/detection/outputs/rfdetr_baseline
# 評估
$PY diagnosis_model/detection/valid_rfdetr.py \
  --dataset_dir $DET \
  --checkpoint_path diagnosis_model/detection/outputs/rfdetr_baseline/checkpoint_best_ema.pth \
  --output_dir diagnosis_model/detection/outputs/rfdetr_baseline/eval
```

---

## YOLO11m baseline（訓練 + 評估）

```bash
# 訓練吃 YOLO 格式;腳本固定 epochs=100, batch=32, imgsz=640;首次自動下載 yolo11m.pt
$PY diagnosis_model/detection/train_yolo.py --dataset_dir $YOLO --model yolo11m
# 評估吃 COCO 格式（valid_yolo 走同一套 COCOeval;--checkpoint_path 必填,權重在 YOLO 樹下）
$PY diagnosis_model/detection/valid_yolo.py \
  --dataset_dir $DET \
  --model yolo11m \
  --checkpoint_path $YOLO/outputs/yolo11m/train/weights/best.pt \
  --output_dir diagnosis_model/detection/outputs/yolo11m_eval
```

---

## 取數字 / 填表

各 eval 輸出 `metrics_all_splits.json`,取 **`valid`** split（兩表皆 valid set）。JSON key → 表欄位:

| 表欄位 | JSON key |
|---|---|
| mAP@50 | `AP_50` |
| mAP@50:95 | `AP_50_95` |
| AP_S / AP_M / AP_L | `AP_small` / `AP_medium` / `AP_large` |

（JSON 為 0–1 小數,表用百分比 ×100。）

- **表 9**（`tab:lesion_detection_results`）:填 OAVLE、RF-DETR Medium、YOLO11m 三列。
- **表 10**（`tab:voc_detection`）:OAVLE / RF-DETR-M / YOLO11m 為**我們跑的 COCO 協定 AP@50**;
  經典三列（Faster R-CNN / SSD / YOLOv2）為**文獻 VOC 協定**、直接引用不重跑。**表下須加協定腳註**
  （現代列 = COCO AP@50、經典列 = VOC 協定,兩者近似非同一內插）。

## 注意

- **資料集一改動,YOLO 格式必重轉、baseline 需重訓。**
- 三模型各需近 ~23GB 顯存,**逐一序列跑、勿併跑**（否則 OOM）。
