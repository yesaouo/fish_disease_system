# VALID_PIPELINE：Ch5 評估與論文表圖產出

用途：在 `BUILD_PIPELINE.md` 已完成 artifacts 後，於 **current 樹生產 artifacts** 上執行第五章所有評估，輸出論文表格與圖所需數字。

生產設定固定為：

- 模型：Region Gate / `grod_soft`
  - `encoder_grod_soft`，含 `gate_state`
  - `ceah_grod_soft`
  - gated bank
- 操作點：`top_k_cases=3`
- 評估原則：本文件只讀既有 artifacts，僅輸出 eval JSON / log，不覆蓋生產檔。

與 `BUILD_PIPELINE.md` 分工：

- BUILD：建立 artifacts。重跑時需避免覆蓋生產版本。
- VALID：只讀 artifacts 跑 eval，產出 Ch5 表圖數字。

---

## 1. Quickstart

```bash
PY=/home/lab603/anaconda3/envs/SDM/bin/python
cd /mnt/ssd/YJ/fish_disease_system

DET=data/processed/current/detection
SYM=data/processed/current/symptoms.json
ART=data/processed/current/artifacts
CLU=$ART/cause_clusters_llm.json
K=3

export RFDETR_BACKBONE=dinov3_small
export RFDETR_SEMANTIC_DIM=768
export RFDETR_GLOBAL_DIM=768
export RFDETR_SEMANTIC_ANCHORS=$(realpath $ART/models/text_anchors.pt)

COMMON_BASE="--case_db_dir $ART/db/case_db_jointDistRawP \
  --soft_dir $ART/db/soft_inputs_gated \
  --encoder_ckpt $ART/models/encoder_grod_soft/best_encoder.pt \
  --bank_path $ART/models/encoder_grod_soft/bank_z_soft.pt \
  --ceah_ckpt $ART/models/ceah_grod_soft/best_ceah.pt \
  --cluster_json $CLU \
  --text medical"

COMMON_K3="$COMMON_BASE --top_k_cases $K"
```

### 必要環境提醒

`RFDETR_BACKBONE=dinov3_small` 必須 export。凡是會重建偵測器的步驟都會讀它，例如：

- 症狀辨識
- 偵測 mAP
- live inference

current 樹的 joint checkpoint 的影像特徵編碼器是 DINOv3；若漏設，monkeypatch 可能靜默丟掉不匹配的 backbone tensor，改用預設 DINOv2，導致症狀 z 錯誤。只讀 bank 的檢索、排序、faithfulness 不受影響，因 bank 已在 BUILD 階段用正確 backbone 編好。

---

## 2. 論文表圖 ↔ 評估腳本

| 論文表圖 label | 章節 | 數字來源 | 操作點 |
|---|---|---|---|
| `tab:dataset_statistics` | §5.1 | case DB / taxonomy 統計 | — |
| `tab:lesion_detection_results` | 病灶偵測 | `detection.valid_rfdetr` | — |
| 多病灶症狀辨識表 | 症狀辨識 | `grod.eval_lesion_symptom_cls` | — |
| `tab:retrieval_results` | 相似案例檢索 | γ-sweep 的 `γ=1.00` 端點 | k=3 |
| `tab:topk_ablation` | top-k 操作點 | `grod.eval_ceah_soft_paper` top-k sweep | k∈{1,3,5,10,20} |
| `tab:cause_ranking_results` | 多病因排序 | `grod.eval_ceah_soft_paper --gammas 0.0` | k=3 |
| `fig:gamma_ablation` | γ 消融 | `grod.eval_ceah_soft_paper --gammas ...` | k=3 |
| `tab:integration_ablation` | 整合架構 + 區域門控 | soft / hard / base 三路 eval + pipeline timing | k=3 |
| `tab:evidence_removal` | 證據歸因 | `grod.faithfulness_eval_soft` | k=3 |
| §sec:efficiency | 系統效率 | pipeline wall-clock timing / params | k=3 |
| `tab:gemini_expert_analysis` | Gemini vs 專家 | `dataset_pipeline` 分析腳本 | — |

圖檔一律由 `paper/make_figures.py` 產生；eval 只負責輸出數字。將數字填入對應的 `build_<name>()`，再輸出 `figures/<name>.png`。

---

## 3. 病灶偵測：`tab:lesion_detection_results`

目的：比較 RF-DETR baseline 與 OAVLE joint 的 class-agnostic lesion box mAP。軟 / 硬門控不改變 lesion box，因此偵測效能相同，可合併呈現。

`valid_rfdetr` 使用 stock RFDETRMedium；對 DINOv3 checkpoint 需先 import `diagnosis_model.grod` 套 monkeypatch。

```bash
RUN='import diagnosis_model.grod, runpy, sys; runpy.run_module("diagnosis_model.detection.valid_rfdetr", run_name="__main__")'

# RF-DETR baseline
$PY -c "$RUN" \
  --dataset_dir $DET \
  --checkpoint_path $ART/models/rfdetr/checkpoint_best_total.pth

# OAVLE joint
$PY -c "$RUN" \
  --dataset_dir $DET \
  --checkpoint_path $ART/models/joint_rfdetr/checkpoint_best_regular.pth
```

填表欄位：

- mAP@50
- mAP@50:95
- AP_S
- AP_M
- AP_L

---

## 4. 多病灶症狀辨識

目的：使用 GT lesion boxes，評估 OAVLE 語意 z 是否能分到正確症狀類別。流程為 IoU matching 後計算 `argmax cos(z, anchor)`。

```bash
$PY -m diagnosis_model.grod.eval_lesion_symptom_cls \
  --joint_ckpt $ART/models/joint_rfdetr/checkpoint_best_regular.pth \
  --anchors $ART/models/text_anchors.pt \
  --coco $DET/valid/_annotations.coco.json \
  --image_root $DET/valid \
  --symptoms $SYM
```

填表欄位：

- `acc_matched`
- `acc_all`
- top-3 accuracy
- macro-F1

---

## 5. 相似案例檢索：`tab:retrieval_results`

目的：評估 OAVLE 軟性聚合單向量的純檢索先驗排序。

檢索表不另跑獨立腳本，直接使用 γ-sweep 的 `γ=1.00` 端點：

- `γ=1.00`：純檢索 prior 排序
- `γ=0.00`：純 CEAM evidence 排序

填表單列：

- 模型：OAVLE 軟性聚合單向量
- 指標：Recall@1 / Recall@5 / Recall@10 / MRR / Cluster R@10

不要使用 `phase1_baseline` 的 global / lesion / combined 多向量流程；該流程是為 raw SigLIP2 case DB 設計，與 `case_db_jointDistRawP` 不相容，會得到接近 0 的假結果。

---

## 6. top_k_cases 操作點：`tab:topk_ablation`

目的：說明 `top_k_cases=3` 是生產操作點。候選池太小會覆蓋不足，太大則引入 distractor 並稀釋排名；因此需要用 sweep 呈現非單調趨勢。

```bash
for K in 1 3 5 10 20; do
  $PY -m diagnosis_model.grod.eval_ceah_soft_paper \
    $COMMON_BASE \
    --top_k_cases $K \
    --gammas 0.0 \
    --output_dir $ART/models/ceah_grod_soft/ch5/ksweep_k$K
done
```

填表 / 圖：

- sem R@10
- Cluster R@10
- 標出 `k=3` 為內部最佳或生產操作點

若改用圖呈現，於 `paper/make_figures.py` 新增 `build_topk_operating_point()`。

---

## 7. 多病因排序與 γ 消融

### 7.1 Full model：`tab:cause_ranking_results`

目的：在固定 k=3、γ=0 的設定下，呈現全模態 OAVLE 的病因排序涵蓋率。論文表保留 Full model 單列；Image+Lesion / Image+Text 模態消融因差異落於雜訊內，不列入主表，模態貢獻改由證據歸因呈現。

```bash
$PY -m diagnosis_model.grod.eval_ceah_soft_paper \
  $COMMON_K3 \
  --gammas 0.0 \
  --output_dir $ART/models/ceah_grod_soft/ch5/full
```

填表欄位：

- sem R@1
- sem R@5
- sem R@10
- MRR
- NDCG@5
- Cluster R@10

### 7.2 γ 消融：`fig:gamma_ablation`

目的：比較檢索 prior 與 CEAM evidence 的權重。預期 `γ=0` 最佳，表示魚病任務中 retrieval prior 較弱。

```bash
$PY -m diagnosis_model.grod.eval_ceah_soft_paper \
  $COMMON_K3 \
  --gammas 0.0 0.25 0.5 0.75 1.0 \
  --output_dir $ART/models/ceah_grod_soft/ch5/gamma
```

填圖：

- x-axis：γ
- y-axis：sem R@10 / Cluster R@10
- 數字填入 `paper/make_figures.py::build_gamma_ablation()`

---

## 8. 整合架構與區域門控消融：`tab:integration_ablation`

目的：在相同 valid split 上比較三種設定：

1. base：分離式 fine-tuned SigLIP2
2. grod：硬閘
3. grod_soft：Region Gate，生產設定

比較欄位：

- params
- end-to-end latency
- sem R@10
- Cluster R@10

不要使用 `eval_case_retrieval_modes` 填此表；它輸出 case-level hit / pool coverage，不是 cause Recall@10 / Cluster R@10。

### 8.1 grod_soft：Region Gate

直接使用 §7.1 Full model 的 `γ=0` 結果，無需重跑。

定版參考值：

- sem R@10：0.692
- Cluster R@10：0.547

### 8.2 grod：hard gate

```bash
$PY -m diagnosis_model.grod.eval_ceah_soft_paper \
  --case_db_dir $ART/db/case_db_jointDistRawP \
  --soft_dir $ART/db/soft_inputs_hard \
  --encoder_ckpt $ART/models/encoder_grod_soft/best_encoder.pt \
  --bank_path $ART/models/encoder_grod_soft/bank_z_soft.pt \
  --ceah_ckpt $ART/models/ceah_grod_soft/best_ceah.pt \
  --cluster_json $CLU \
  --text medical \
  --top_k_cases 3 \
  --gammas 0.0 \
  --output_dir $ART/models/ceah_grod_soft/ch5/hard
```

`soft_inputs_hard` 是 `apply_gate` 的 hard 變體，`w=objectness` 二值化。

### 8.3 base：分離式 SigLIP2

```bash
$PY -m diagnosis_model.cause_inference.eval_ceah \
  --case_db_dir $ART/db/case_db_base \
  --ceah_ckpt $ART/models/ceah_base/best_ceah.pt \
  --cluster_json $CLU \
  --top_k_cases 3 \
  --gammas 0.0 \
  --attribution_mode softmax \
  --scoring_mode multiplicative \
  --output_dir $ART/models/ceah_base/ch5_k3
```

`ceah_base` 是 multiplicative 雙頭；必須傳：

- `--scoring_mode multiplicative`
- `--attribution_mode softmax`

否則會發生 state_dict 不相容。

### 8.4 params / latency

端到端延遲使用 wall-clock：

- 暖機後，在整個 `infer_rich` 外圍計時
- 前後各一次 `torch.cuda.synchronize()`
- 不使用 `infer_rich` 回傳的 per-stage timings 加總

per-stage timings 只用於分析各階段比例，不作端到端總延遲。

定版值：

| mode | params | latency |
|---|---:|---:|
| base | 225M | 30.4 ms |
| OAVLE hard | 40.8M | 15.4 ms |
| OAVLE soft | 40.8M | 12.2 ms |

base 參數只計 SigLIP2 vision-only；文字塔不參與逐圖推論，不納入推論參數量。

---

## 9. 證據歸因：`tab:evidence_removal`

目的：以反事實遮罩評估各類 evidence 對最終排序的貢獻。

```bash
$PY -m diagnosis_model.grod.faithfulness_eval_soft \
  --case_db_dir $ART/db/case_db_jointDistRawP \
  --soft_dir $ART/db/soft_inputs_gated \
  --encoder_ckpt $ART/models/encoder_grod_soft/best_encoder.pt \
  --bank_path $ART/models/encoder_grod_soft/bank_z_soft.pt \
  --ceah_ckpt $ART/models/ceah_grod_soft/best_ceah.pt \
  --output_dir $ART/models/ceah_grod_soft/ch5/faithfulness
```

填表設定：

- `no_global`
- `no_lesion`
- `no_top_α`
- `no_random`

填表分組：

- all
- 病灶型
- 全域型

current 樹中 `no_global` 可能為負，敘述上應改成「全域 evidence 主要影響檢索，不是最終 CEAM 歸因中的主要 load-bearing evidence」。

---

## 10. 非模型評估：資料層

### 10.1 `tab:dataset_statistics`

由 `case_db_jointDistRawP` 的 cases 與 `cause_clusters_llm.json` 直接統計：

- train / valid / test 影像數
- 症狀類別數
- 相異病因數
- singleton 比例
- 平均每案例病因數
- cluster 數

current 樹資料常數：

- cluster：471
- valid：1,584

### 10.2 `tab:gemini_expert_analysis`

由 `dataset_pipeline` 的標註分析腳本產出：

- Precision
- Recall
- F1
- 修改率

詳見 `dataset_pipeline/README.md`。

---

## 11. Gotchas

- 操作點固定為 `top_k_cases=3`。舊論文表的 k=20 數字作廢。
- `Cluster R@K` 必須傳 `--cluster_json $CLU`；語意指標不需要 taxonomy。
- 生產 Region Gate eval 必須使用 `soft_inputs_gated`，不是舊的 `soft_inputs`。
- current 樹為 471 clusters / 1,584 valid，表格與 caption 要同步。
- `valid_rfdetr` 跑 DINOv3 checkpoint 時，必須先 import `diagnosis_model.grod` 套 monkeypatch。
- base 與 OAVLE 的 cause vocabulary / cluster 數可能略有差異；整合消融表需註明比較僅限三設定內。
- eval 輸出集中於：
  - `$ART/models/ceah_grod_soft/ch5/`
  - `$ART/eval/`
