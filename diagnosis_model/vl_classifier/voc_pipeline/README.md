# VOC SigLIP Pipeline

以 SigLIP / SigLIP2 為基礎的 VOC 物件區域分類流程，支援四種訓練模式：

- baseline
- multipos
- fusion
- multipos + fusion

並提供：

- 單一訓練入口：`train.py`
- 單一評估入口：`eval.py`
- 一鍵跑完整訓練與評估：`run_all_voc.py`
- Heatmap 與 VOC2012 semantic segmentation 差異可視化：`voc_heatmap.ipynb`

## 檔案結構

- `train.py`：VOC 訓練主程式
- `eval.py`：VOC 評估主程式
- `run_all_voc.py`：依序訓練四種模式並統一評估
- `voc_dataset.py`：VOC region dataset 與 bbox / square crop 邏輯
- `common.py`：共用 loss、feature extraction、fusion wrapper、視覺化工具
- `voc_labels.py`：VOC 類別定義與預設 label bank 產生器
- `voc_heatmap.ipynb`：heatmap 與 segmentation 差異分析 notebook

## 安裝

```bash
pip install torch torchvision transformers pillow tqdm matplotlib scikit-learn opencv-python seaborn
```

## 資料集

本專案使用 `torchvision.datasets.VOCDetection` 讀取 VOC，因此 `--voc_root` 應指向 VOC 的根目錄，例如：

```text
/path/to/data/
└── VOCdevkit/
    ├── VOC2007/
    └── VOC2012/
```

若本機尚未下載資料，可在訓練或評估時加上 `--download`。

### split 建議

#### VOC 2007

- train：`train`
- valid：`val`
- eval：`test`

#### VOC 2012

- train：`train`
- valid：`val`
- eval：`val`

> VOC2012 的 `test` annotations 不公開，因此通常使用 `val` 做驗證與評估。

## 訓練模式

`train.py` 使用兩個旗標控制模式：

- baseline：不加旗標
- multipos：`--multipos`
- fusion：`--fusion`
- multipos + fusion：`--multipos --fusion`

### 1. baseline

```bash
python train.py \
  --voc_root /path/to/data \
  --year 2007 \
  --train_image_set train \
  --valid_image_set val \
  --output_dir ./voc2007_baseline
```

### 2. multipos

```bash
python train.py \
  --multipos \
  --voc_root /path/to/data \
  --year 2007 \
  --train_image_set train \
  --valid_image_set val \
  --output_dir ./voc2007_multipos
```

### 3. fusion

```bash
python train.py \
  --fusion \
  --voc_root /path/to/data \
  --year 2007 \
  --train_image_set train \
  --valid_image_set val \
  --output_dir ./voc2007_fusion
```

### 4. multipos + fusion

```bash
python train.py \
  --multipos --fusion \
  --voc_root /path/to/data \
  --year 2007 \
  --train_image_set train \
  --valid_image_set val \
  --output_dir ./voc2007_multipos_fusion
```

## 常用訓練參數

```bash
--model_name google/siglip2-base-patch16-224
--crop_mode bbox            # bbox 或 square
--batch_size 128
--num_epochs 10
--learning_rate 1e-4
--fusion_base_lr 3e-5
--fusion_head_lr 1e-4
--max_length 64
--num_workers 8
--skip_difficult_train
--skip_difficult_valid
--eval_test_after_train
--download
```

### 裁切模式

- `bbox`：直接使用 VOC bbox 裁切
- `square`：以 bbox 為中心補成正方形，超出部分以黑底補齊

## 評估

`eval.py` 支援同時比較多個模型，`--model` 可重複指定，格式為：

```bash
--model tag=path_or_repo
```

例如：

```bash
python eval.py \
  --voc_root /path/to/data \
  --year 2007 \
  --image_set test \
  --output_dir ./eval_voc2007 \
  --model zeroshot=google/siglip2-base-patch16-224 \
  --model baseline=./voc2007_baseline \
  --model multipos=./voc2007_multipos \
  --model fusion=./voc2007_fusion \
  --model multipos_fusion=./voc2007_multipos_fusion \
  --save_vis
```

### 常用評估參數

```bash
--crop_mode bbox
--text_batch_size 256
--img_batch_size 64
--max_length 64
--skip_difficult
--save_vis
--download
```

### 評估輸出

`eval.py` 會在 `--output_dir` 下輸出：

- `confusion_matrix_<tag>.png`
- `confusion_matrix_<tag>_norm.png`
- `report_<tag>.txt`
- `vis/`：每張圖的預測可視化結果（使用 `--save_vis` 時）

若 checkpoint 目錄內存在 `wrapper_state.pt`，評估時會自動以 fusion wrapper 載入；否則以一般 dual encoder 載入。

## 一鍵執行完整流程

```bash
python run_all_voc.py \
  --voc_root /path/to/data \
  --year 2007 \
  --output_root ./outputs_voc2007 \
  --download
```

這會依序執行：

- baseline 訓練
- multipos 訓練
- fusion 訓練
- multipos + fusion 訓練
- 統一評估

### 常用參數

```bash
--model_name google/siglip2-base-patch16-224
--crop_mode bbox
--train_image_set train
--valid_image_set val
--eval_image_set test
--batch_size 128
--fusion_batch_size 64
--num_epochs 10
--learning_rate 1e-4
--base_learning_rate 3e-5
--fusion_learning_rate 1e-4
--save_vis
--skip_difficult_train
--skip_difficult_eval
--download
```

## Label Bank

若未指定 `--label_bank_json`，程式會自動建立預設的 VOC label bank。

格式如下：

```json
{
  "label_map": {
    "0": {"en": "aeroplane", "zh": "飛機"}
  },
  "data": {
    "0": {
      "captions_en": [
        "aeroplane",
        "an aeroplane",
        "a photo of an aeroplane"
      ]
    }
  }
}
```

## voc_heatmap.ipynb

`voc_heatmap.ipynb` 使用 VOC2012 的物件分割資料做 heatmap 檢查，流程包含：

1. 讀取 VOC2012 detection annotation 與 segmentation mask
2. 針對指定物件 bbox 產生 local crop heatmap
3. 將 heatmap 映射回原圖座標
4. 與對應類別的 semantic segmentation mask 比較
5. 顯示下列指標與視覺化差異：
   - IoU
   - Precision
   - Recall
   - F1
   - TP / FP / FN overlay

注意事項：

- VOC2012 提供的是 **semantic segmentation**，不是 instance segmentation
- 因此 notebook 比較的是「該類別在 bbox 範圍內的語意遮罩」，不是單一 instance mask
- 建議使用 `year=2012` 並搭配 `val` split 進行測試

## 建議指令

### VOC2007 訓練 + 測試集評估

```bash
python run_all_voc.py \
  --voc_root /path/to/data \
  --year 2007 \
  --train_image_set train \
  --valid_image_set val \
  --eval_image_set test \
  --output_root ./outputs_voc2007
```

### VOC2012 訓練 + 驗證集評估

```bash
python run_all_voc.py \
  --voc_root /path/to/data \
  --year 2012 \
  --train_image_set train \
  --valid_image_set val \
  --eval_image_set val \
  --output_root ./outputs_voc2012
```
