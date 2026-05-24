# VOC Region Benchmark for VLM Lesion Classifier

這個資料夾把 PASCAL VOC 轉成 region-level vision-language benchmark。它不是魚病任務；用途是測試同一套 crop / text / fusion 流程，能不能在固定類別的物件區域資料集上運作。

pipeline 會把每個 VOC object bbox 視為一筆 region sample：

- local image：object bbox crop
- global image：完整 VOC 影像，只在 fusion checkpoint 使用
- text candidates：來自 `voc_label_bank.json` 的 caption bank
- prediction：在 VLM embedding space 中找最接近的 caption / label

## 這個 benchmark 的角色

論文敘事裡，VOC 是語意分類階段的 sanity benchmark：

```text
RF-DETR abnormal-region proposal -> VLM semantic region classifier/retriever
```

VOC 不能證明魚病診斷能力。它的作用是顯示這個分類器本質上是 general region-text classifier，並且可以拿來比較 closed fixed-class recognition 和 semantic VLM recognition。

## 檔案

- `voc_dataset.py`：把 VOC XML annotation 轉成 region samples 和 crops。
- `common.py`：VOC 專用的 label bank、crop、metric、visualization helper。VLM feature extraction 和 fusion 直接復用 `diagnosis_model.vl_classifier.common`。
- `train.py`：訓練 paired、multipos、fusion、multipos+fusion 的 SigLIP/SigLIP2 checkpoint。
- `eval.py`：評估 zero-shot model 或 finetuned checkpoint 的 VOC region classification。
- `run_all_voc.py`：一次跑標準比較組。

### Text Mode

`--text_mode captions` 維持原本 VOC caption bank；`--text_mode class_name` 只用 `label_map.en` 的直接類別名稱；`--text_mode captions_plus_class_name` 會合併兩者並去除重複文字。這可用來把描述式 prompts 與常見 class-name baseline 分開比較。
- `voc_heatmap_fixed.ipynb`：薄 notebook wrapper，只用來跑 eval、讀 summary、看 visualization。

## 建議指令

請從 repo root 執行。

### 評估 Zero-Shot SigLIP2

```bash
python -m diagnosis_model.vl_classifier.voc_pipeline.eval \
  --voc_root /path/to/data \
  --year 2007 \
  --image_set test \
  --output_dir diagnosis_model/vl_classifier/voc_pipeline/outputs/eval_zeroshot \
  --model zeroshot=google/siglip2-base-patch16-224 \
  --download
```

### 訓練單一模式

```bash
python -m diagnosis_model.vl_classifier.voc_pipeline.train \
  --voc_root /path/to/data \
  --year 2007 \
  --train_image_set train \
  --valid_image_set val \
  --multipos --fusion \
  --freeze_text_encoder \
  --fusion_gate_mode scalar \
  --output_dir diagnosis_model/vl_classifier/voc_pipeline/outputs/voc2007_multipos_fusion
```

支援模式：

- paired baseline：不加 `--multipos`、不加 `--fusion`
- multipos：`--multipos`
- fusion：`--fusion`
- multipos + fusion：`--multipos --fusion`

fusion gate mode 與主線 lesion classifier 一致：

- `scalar`：production-style scalar context gate
- `film`：input-conditioned per-channel gate
- `xattn`：local query 對 whole-image patch tokens 做 cross-attention

### 一次跑完整 VOC 比較

```bash
python -m diagnosis_model.vl_classifier.voc_pipeline.run_all_voc \
  --voc_root diagnosis_model/vl_classifier/voc_pipeline/data \
  --year 2007 \
  --label_bank_json diagnosis_model/vl_classifier/voc_pipeline/voc_label_bank.json \
  --output_root diagnosis_model/vl_classifier/voc_pipeline/outputs/voc2007_suite \
  --freeze_text_encoder

python -m diagnosis_model.vl_classifier.voc_pipeline.run_all_voc \
  --voc_root diagnosis_model/vl_classifier/voc_pipeline/data \
  --year 2007 \
  --label_bank_json diagnosis_model/vl_classifier/voc_pipeline/voc_label_bank.json \
  --output_root diagnosis_model/vl_classifier/voc_pipeline/outputs/voc2007_suite_class \
  --freeze_text_encoder \
  --text_mode class_name
```

完整 suite 會評估：

- `zeroshot=google/siglip2-base-patch16-224`
- paired finetune
- multipos finetune
- fusion finetune
- multipos + fusion finetune

### VOC2007 Suite Results

以下結果使用 VOC2007 test split，`--freeze_text_encoder`，`n=14976`。

`--text_mode captions`，輸出到 `outputs/voc2007_suite`：

| setting | acc | macro-F1 | weighted-F1 | balanced acc |
| --- | ---: | ---: | ---: | ---: |
| zeroshot | 0.7595 | 0.7777 | 0.7623 | 0.8450 |
| baseline | 0.8816 | 0.8504 | 0.8838 | 0.8682 |
| multipos | 0.9146 | 0.8873 | 0.9147 | 0.8856 |
| fusion | 0.9328 | 0.9167 | 0.9335 | 0.9220 |
| multipos_fusion | 0.9403 | 0.9217 | 0.9405 | 0.9199 |

`--text_mode class_name`，輸出到 `outputs/voc2007_suite_class`：

| setting | acc | macro-F1 | weighted-F1 | balanced acc |
| --- | ---: | ---: | ---: | ---: |
| zeroshot | 0.7772 | 0.7882 | 0.7771 | 0.8806 |
| baseline | 0.9209 | 0.8944 | 0.9213 | 0.8972 |
| multipos | 0.9107 | 0.8803 | 0.9107 | 0.8737 |
| fusion | 0.9429 | 0.9227 | 0.9433 | 0.9263 |
| multipos_fusion | 0.9447 | 0.9282 | 0.9445 | 0.9220 |

## 輸出

evaluation 會輸出：

- `summary_metrics.json`：accuracy、macro-F1、weighted-F1、balanced accuracy
- `report_<tag>.txt` 和 `report_<tag>.json`
- `confusion_matrix_<tag>.png`
- `confusion_matrix_<tag>_norm.png`
- `vis/`：若啟用 `--save_vis`

## 解讀方式

VOC 要被當成固定詞彙的 controlled benchmark。closed-set classifier 在 VOC 上比 semantic VLM 更強是合理的，這不代表魚病方法失敗。

魚病 pipeline 的核心主張不同：RF-DETR 先提出異常區域後，VLM stage 可以對可擴充的症狀 / 疾病文字描述打分，也可以在不替換 final classification head 的情況下評估 unseen labels。
