# Dataset Pipeline 操作手冊

把四個原始影像來源 → 專家標註平台 → 三階段 ML 模型訓練資料 的所有指令、產出位置，以及每個產出被誰使用，全部列在這裡。所有指令都從 repo root（`/mnt/ssd/YJ/fish_disease_system/`）執行。

---

## 0. 影像來源

| 子資料夾 | 來源 | 原始格式 |
| --- | --- | --- |
| `data/raw/fish disease/` | Roboflow Universe — **Fish Disease** | COCO（`images/` + `images/_annotations.coco.json` + `classes.yaml`） |
| `data/raw/Fish disease 9911/` | Roboflow Universe — **Fish Disease 9911** | COCO（同上） |
| `data/raw/fish_disease_detection/` | Roboflow Universe — **Fish Disease Detection** | COCO（同上） |
| `data/raw/tilapia/` | 國立屏東科技大學 — **Tilapia** | 純資料夾（`sick/` + `healthy/` 子目錄，**無 bbox**） |

共用設定檔（`data/raw/` 底下）：

- `symptoms.json` — 病灶類別 + 中英文 caption。多處共用：
  - `annotation_web` 前端 / 後端的類別清單
  - `anotation2coco.py` 寫 COCO categories
  - `vl_classifier/train.py` + `vl_classifier/eval.py` 的 caption bank
  - `demo/app_gradio.py` 推論時的 VLM-Lesion caption bank
- `labels.txt` — 偵測類別 ID/名稱表（給 `process_coco.py` 用，會做 ID 重編號）
- `fish_disease_des.txt` — 中文整體描述模板
- 每個 Roboflow 子資料集還會額外放 `prompt_zh.txt` + `classes.yaml`

---

## 1. Raw → 專家標註平台（Gemini 自動初稿）

兩個入口腳本：Roboflow 資料用 `process_coco_gemini.py`（有 bbox），Tilapia 用 `process_folder_gemini.py`（無 bbox）。兩者都會呼叫 Gemini 2.5 Pro 產出 `overall`（口語 + 醫學）/ `global_causes_zh` / `global_treatments_zh`，並寫成 [annotation_web](../annotation_web/) 預期的目錄結構。需要 `.env` 設定 `GEMINI_API_KEY`。

### 1a. Roboflow COCO 資料（Fish Disease / Fish Disease 9911 / Fish Disease Detection）

```bash
$PY dataset_pipeline/process_coco_gemini.py \
    --input_dir  "data/raw/fish disease" \
    --output_dir "annotation_web/data/fish_disease" \
    --keep_healthy_only --save_viz

$PY dataset_pipeline/process_coco_gemini.py \
    --input_dir  "data/raw/Fish disease 9911" \
    --output_dir "annotation_web/data/Fish_disease_9911" \
    --keep_healthy_only --save_viz

$PY dataset_pipeline/process_coco_gemini.py \
    --input_dir  "data/raw/fish_disease_detection" \
    --output_dir "annotation_web/data/fish_disease_detection" \
    --keep_healthy_only --save_viz
```

- 讀取：`<input_dir>/images/*.jpg` + `<input_dir>/images/_annotations.coco.json` + `<input_dir>/classes.yaml` + `<input_dir>/prompt_zh.txt`
- 產出：
  - `<output_dir>/images/`、`<output_dir>/annotations/*.json`（病魚）
  - `<output_dir>/healthy_images/`、`<output_dir>/healthy_annotations/*.json`（健康，當 `--keep_healthy_only` 時保留）
  - `<output_dir>/viz_results/`（`--save_viz` 時的 bbox 視覺化）
- 其他旗標：`--cache_only`（只用快取不打 API）、`--overwrite_cache`、`--reuse_annotations_dir <舊資料夾>`（沿用既有結果，省 token）
- **下游**：[annotation_web 前端](../annotation_web/) — 專家在 UI 校正 bbox/類別/敘述

### 1b. Tilapia 純資料夾資料

```bash
$PY dataset_pipeline/process_folder_gemini.py \
    --sick_dir    "data/raw/tilapia/sick" \
    --healthy_dir "data/raw/tilapia/healthy" \
    --prompt      "data/raw/tilapia/prompt_zh.txt" \
    --output_dir  "annotation_web/data/tilapia"
```

- 讀取：兩個影像資料夾 + 一個 prompt 檔
- 產出：與 1a 相同的 `images/ annotations/ healthy_images/ healthy_annotations/` 結構，但所有 JSON 的 `detections` 固定為空陣列（沒有 bbox 來源）
- **下游**：[annotation_web 前端](../annotation_web/) — 專家在 UI **新增** bbox（從零開始標）

---

## 2. 專家標註平台 → 標註版本歷史

`annotation_web` 的後端會把每一輪人工修改寫進 `data/annotation/<dataset>/annotations_versions/<stem>_v{N}.json`，原版檔名沒有 `_vX` 後綴。這一步沒有腳本要跑，但接下來的 QA 與 COCO 轉換步驟全部以此為輸入。

---

## 3. 標註品質檢查（QA / 報表）

純報表類，**不影響任何模型訓練**，只用來人工檢查每個資料集的標註異動量。

```bash
$PY dataset_pipeline/analyze_annotation.py \
    --dir "data/annotation/fish_disease/annotations_versions" \
    --out "data/annotation/results/fish_disease"

$PY dataset_pipeline/analyze_annotation.py \
    --dir "data/annotation/Fish_disease_9911/annotations_versions" \
    --out "data/annotation/results/Fish_disease_9911"

$PY dataset_pipeline/analyze_annotation.py \
    --dir "data/annotation/fish_disease_detection/annotations_versions" \
    --out "data/annotation/results/fish_disease_detection"

$PY dataset_pipeline/analyze_annotation.py \
    --dir "data/annotation/tilapia/annotations_versions" \
    --out "data/annotation/results/tilapia"
```

每個 dataset 會吐 `files_summary.csv` + `label_delta.csv`。然後對合併後的總覽再跑一次 summary：

```bash
$PY dataset_pipeline/analyze_files_summary.py \
    --input  "data/annotation/results/files_summary_merged.csv" \
    --output "data/annotation/results" \
    --top_n  50
```

- **下游**：人工 QA 報表（給研究團隊看，不進模型）

---

## 4. 標註版本 → COCO

把人工校正完的最終版（每個 stem 的最大 `_vN`）轉成標準 COCO（`train/valid/test` 三切分，切分由 `md5(file_name)` 穩定決定，新增影像不會打亂舊切分）。

```bash
$PY dataset_pipeline/anotation2coco.py \
    --input  "data/annotation/fish_disease" \
    --output "data/coco/fish_disease" --split 8 1 1

$PY dataset_pipeline/anotation2coco.py \
    --input  "data/annotation/Fish_disease_9911" \
    --output "data/coco/Fish_disease_9911" --split 8 1 1

$PY dataset_pipeline/anotation2coco.py \
    --input  "data/annotation/fish_disease_detection" \
    --output "data/coco/fish_disease_detection" --split 8 1 1

$PY dataset_pipeline/anotation2coco.py \
    --input  "data/annotation/tilapia" \
    --output "data/coco/tilapia" --split 8 1 1
```

- 讀類別表：`<input>/symptoms.json`
- 產出：`data/coco/<dataset>/{train,valid,test}/_annotations.coco.json` + 對應影像
- **下游**：`merge_coco.py`

---

## 5. 合併四個 COCO 子資料集

```bash
$PY dataset_pipeline/merge_coco.py --root_dir data/coco
```

- 自動掃描 `data/coco/` 底下每個含 `train/` 或 `valid/` 的子資料夾，合併到 `data/coco/_merged/`
- **注意**：`.gitignore` 已用 `**/data/` 把整個 `data/` 樹排除版控，所以 `_merged/` 自然不會被 commit
- **下游**：
  - **直接消費**：[diagnosis_model/vl_classifier/train.py](../diagnosis_model/vl_classifier/train.py)（`--data_root data/coco/_merged`；訓練時在記憶體即時 crop bbox、對 `data/raw/symptoms.json` 組 caption）
    - VLM-Lesion 變體：`--multipos --fusion --freeze_text_encoder`（per-bbox crop ↔ symptom caption）
    - VLM-Global 變體：`--target overall --multipos`（整魚圖 ↔ 整體敘述）
  - **再加工**：§6 的 `process_coco.py`

---

## 6. `_merged` → Detection 訓練資料

### 6a. → Detection COCO（RF-DETR 生產 + cause_inference）

```bash
$PY dataset_pipeline/process_coco.py \
    --data_root  data/coco/_merged \
    --label_file data/raw/labels.txt \
    --output_dir data/detection/coco/_merged
```

- 行為：根據 `labels.txt` 重新編號類別（從 0 開始）、把同名類別合併、過濾未列出的類別
- **下游**：
  - [diagnosis_model/detection/train_rfdetr.py](../diagnosis_model/detection/) — Stage 1 病灶 bbox 偵測（**生產用**）
  - `diagnosis_model/cause_inference/` 全程都吃這份 COCO（`build_case_database.py` / `phase1_baseline.py` / `cause_cluster_llm.py` 等都用 `data/detection/coco/_merged/{train,valid}/_annotations.coco.json` 當 image-level + per-bbox 標註來源）

### 6b. → Detection YOLO（餵 Ultralytics YOLO，比較用）

```bash
$PY dataset_pipeline/coco2yolo.py \
    --data_root  data/detection/coco/_merged \
    --output_dir data/detection/yolo/_merged
```

- 行為：COCO bbox 轉 YOLO normalized `xywh`、產 `data.yaml`
- **下游**：`diagnosis_model/detection/train_yolo.py` — 與 RF-DETR 做對比實驗

### 6c. → Mosaic 增強（給 RF-DETR 對比實驗）

```bash
$PY dataset_pipeline/mosaic_coco.py \
    --data_root data/detection/coco/_merged_mosaic/train \
    --img_size  640 --ratio 0.5
```

- 行為：4 圖 mosaic、寫進新的 COCO JSON
- 需要先 `cp -r data/detection/coco/_merged data/detection/coco/_merged_mosaic` 再對 `train/` 跑增強
- **下游**：`diagnosis_model/detection/train_rfdetr.py --dataset_dir data/detection/coco/_merged_mosaic`（mosaic vs 原始的對照組）

---

## 7. 共用工具模組（不需要直接執行）

| 檔案 | 用途 |
| --- | --- |
| [_bbox.py](_bbox.py) | `bbox_contains(inner, outer, fmt=...)` — `process_coco_gemini.py` 與 `anotation2coco.py` 共用的 bbox 包含判定（支援 `xywh` / `xyxy`） |
| [_gemini.py](_gemini.py) | Gemini 2.5 Pro client：重試、strict-JSON parse、`overall/global_causes_zh/global_treatments_zh` schema 驗證、快取 / `--reuse_annotations_dir` 邏輯，被兩個 `process_*_gemini.py` 共用 |

---

## 8. 全流程總覽圖

```
┌─────────────────────────────────────────────────┐
│  data/raw/                                       │
│   ├─ fish disease/         (Roboflow, COCO)      │
│   ├─ Fish disease 9911/    (Roboflow, COCO)      │
│   ├─ fish_disease_detection (Roboflow, COCO)     │
│   └─ tilapia/              (NPUST, 純資料夾)     │
└──────────────┬──────────────────────────────────┘
               │  process_coco_gemini.py (1a)
               │  process_folder_gemini.py (1b)
               ▼
┌─────────────────────────────────────────────────┐
│  annotation_web/data/<dataset>/                  │
│   ├─ images/  annotations/   (病魚)              │
│   └─ healthy_images/ healthy_annotations/        │
└──────────────┬──────────────────────────────────┘
               │  專家於 annotation_web 前端校正
               ▼
┌─────────────────────────────────────────────────┐
│  data/annotation/<dataset>/annotations_versions/ │
│   └─ <stem>_v{N}.json   (人工修正版本歷史)       │
└──────┬────────────────────────┬──────────────────┘
       │ analyze_*.py (§3)      │ anotation2coco.py (§4)
       │ → QA CSV               ▼
       │                ┌──────────────────┐
       │                │ data/coco/<ds>/  │
       │                └────────┬─────────┘
       │                         │ merge_coco.py (§5)
       │                         ▼
       │                ┌──────────────────────┐
       │                │ data/coco/_merged/   │
       │                └─┬───────────────┬────┘
       │                  │ (直接讀)      │ process_coco (§6a)
       │                  ▼               ▼
       │       diagnosis_model/      data/detection/coco/_merged/
       │        vl_classifier/             │
       │        train.py                   │ coco2yolo (§6b)
       │        (SigLIP2-Global /          │ mosaic_coco (§6c)
       │         SigLIP2-Lesion;           ▼
       │         crop 於記憶體即時   data/detection/yolo/_merged/
       │         做，吃 symptoms.json) data/detection/coco/_merged_mosaic/
       │              │                    │
       │              │             diagnosis_model/detection/
       │              │              (RF-DETR 生產 / YOLO 比較)
       │              │                    │
       │              └──────────┬─────────┘
       │                         ▼
       │              diagnosis_model/cause_inference/
       │              (FaCE-R：build_case_database.py 吃
       │               data/detection/coco/_merged 的 COCO + VLM ckpt)
       │
       └─→ data/annotation/results/  (人工 QA 報表，不進模型)
```

---

## 9. 各產出 → 模型對照速查

| 產出位置 | 被誰使用 | 階段 |
| --- | --- | --- |
| `annotation_web/data/<dataset>/` | annotation_web 前端 / 後端 | 專家標註 UI |
| `data/annotation/<dataset>/` | `analyze_*.py`、`anotation2coco.py` | QA + 轉 COCO |
| `data/annotation/results/` | 人工檢查 | QA 報表 |
| `data/coco/<dataset>/` | `merge_coco.py` | 中介 |
| `data/coco/_merged/` | `diagnosis_model/vl_classifier/train.py`（直接讀，訓練時即時 crop）、`process_coco.py` | **Stage 2 來源** + Detection 中介 |
| `data/detection/coco/_merged/` | `diagnosis_model/detection/train_rfdetr.py`、`diagnosis_model/cause_inference/`（`build_case_database.py` / `phase1_baseline.py` / `cause_cluster_llm.py` 等） | **Stage 1（生產）+ Stage 3 標註來源** |
| `data/detection/coco/_merged_mosaic/` | `train_rfdetr.py`（mosaic 對照組） | Stage 1 ablation |
| `data/detection/yolo/_merged/` | `diagnosis_model/detection/train_yolo.py` | Stage 1（YOLO 對照） |
| `data/raw/symptoms.json` | `annotation_web`、`anotation2coco.py`、`vl_classifier/train.py`、`vl_classifier/eval.py`、`demo/app_gradio.py` | 全程共用類別表 / caption |
| `data/raw/labels.txt` | `process_coco.py` | Stage 1 類別表 |

**Stage 3（`cause_inference`）的資料來源**：image-level + per-bbox 標註直接讀 `data/detection/coco/_merged/{train,valid}/_annotations.coco.json`；視覺特徵則消費 Stage 2 的 SigLIP2 checkpoint，透過 `build_case_database.py` 建立 `outputs/case_db{,_raw}/`。詳見 [diagnosis_model/cause_inference/README.md](../diagnosis_model/cause_inference/README.md)。
