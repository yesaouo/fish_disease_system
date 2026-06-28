# BUILD_PIPELINE：資料版本切換後重建 artifacts

用途：資料版本切換後，依序重建 `data/processed/current/artifacts/` 下的 OAVLE / CEAM / base baseline 產物。  
`current` 是 symlink，artifacts 會跟著目前資料版本走。

本文件分工：

- **主線 Step 1–6**：建立 raw DB、RF-DETR / joint detector、feature cache、distilled global、case DB、soft inputs、gating artifacts。
- **OAVLE Step 7–11**：訓練 OAVLE Aggregator、Region Gate、CEAM，並校準 demo 門檻。
- **base 分支 B0–B5**：建立分離式 SigLIP2 baseline。
- **Gotchas**：集中放所有高風險設定與命名規則。

---

## 1. Quickstart

```bash
PY=/home/lab603/anaconda3/envs/SDM/bin/python
cd /mnt/ssd/YJ/fish_disease_system

DET=data/processed/current/detection
FULL=data/processed/current/full
SYM=data/processed/current/symptoms.json
ART=data/processed/current/artifacts

mkdir -p $ART/models $ART/db

# current production backbone
export RFDETR_BACKBONE=dinov3_small
```

若切換成其他影像特徵編碼器（Image Feature Encoder），需確保所有 build / eval / demo 指向同一組 checkpoint 與同一組 env。

---

## 2. Pipeline map

### 2.1 輸入資料

| 名稱 | 路徑 | 用途 |
|---|---|---|
| symptoms | `$SYM` | symptom 類別空間，含 `healthy_region=0` |
| detection view | `$DET` | 偵測 view；box 全併 `ABNORMAL`，排除 healthy；image 保留 `global_causes_zh` / `overall`；box 保留 `symptom_category_id` |
| full view | `$FULL` | 完整 symptom 類別；供 `vl_classifier` / base SigLIP2 finetune 使用 |

### 2.2 輸出目錄

```text
data/processed/current/artifacts/
├── models/
│   ├── text_anchors.pt
│   ├── rfdetr/
│   ├── joint_rfdetr/
│   ├── distilled_global_rawP/
│   ├── encoder_grod_soft/
│   │   └── bank_z_soft.pt
│   ├── ceah_grod_soft/
│   ├── disease_head/
│   │   └── lesion_threshold.json
│   ├── siglip2_base_finetuned/
│   ├── encoder_base/
│   └── ceah_base/
└── db/
    ├── case_db_raw/
    ├── z_joint/
    ├── dino_global/
    ├── soft_inputs/
    ├── soft_inputs_gated/
    ├── case_db_jointDistRaw/
    ├── case_db_jointDistRawP/
    └── case_db_base/
```

---

## 3. 主線：Step 1–6

### Step 1. 建 raw artifacts：`text_anchors.pt` + `case_db_raw`

Depends on：

- `$SYM`
- `$DET`

Produces：

- `$ART/models/text_anchors.pt`
- `$ART/db/case_db_raw`

Recommended command：

```bash
$PY -m diagnosis_model.build_pipeline.build_raw
```

Expanded commands：

```bash
# 1a. text anchors
$PY -m diagnosis_model.grod.build_text_anchors \
    --symptoms $SYM \
    --out $ART/models/text_anchors.pt

# 1b. raw SigLIP2 case DB
$PY -m diagnosis_model.cause_inference.preprocessing.build_case_database \
    --coco_train  $DET/train/_annotations.coco.json --image_root_train  $DET/train \
    --coco_valid  $DET/valid/_annotations.coco.json --image_root_valid  $DET/valid \
    --vlm_global google/siglip2-base-patch16-224 \
    --vlm_lesion google/siglip2-base-patch16-224 --raw_lesion \
    --output_dir $ART/db/case_db_raw \
    --chunk_size 64 --img_batch_size 64 --text_batch_size 256
```

---

### Step 2. 訓練 RF-DETR detector

Purpose：訓練 OAVLE 偵測前端的 pretrain weights；也供分離式 baseline 與 demo detector 使用。

Depends on：

- `$DET`

Produces：

- `$ART/models/rfdetr/checkpoint_best_total.pth`

Command：

```bash
$PY -m diagnosis_model.detection.train_rfdetr \
    --dataset_dir $DET \
    --output_dir  $ART/models/rfdetr
```

Notes：

- 約數小時 / 100 epochs。
- `--freeze_encoder` 代表凍結 backbone，只訓練 projector / decoder / heads。預設不加＝微調整個 backbone。

---

### Step 3. 訓練 joint detector + semantic head

Purpose：在 Step 2 checkpoint 上加 region heads，訓練 box / objectness / semantic z。這是 OAVLE 一次 forward 產生 box、objectness、lesion semantic z 的前端。

Depends on：

- Step 1：`$ART/models/text_anchors.pt`
- Step 2：`$ART/models/rfdetr/checkpoint_best_total.pth`

Produces：

- `$ART/models/joint_rfdetr/checkpoint_best_regular.pth`

Command：

```bash
$PY -m diagnosis_model.grod.train_joint \
    --dataset_dir $DET \
    --pretrain_weights $ART/models/rfdetr/checkpoint_best_total.pth \
    --anchors $ART/models/text_anchors.pt \
    --output_dir $ART/models/joint_rfdetr \
    --epochs 30 --semantic_loss_coef 2.0
```

Gotchas：

- downstream semantic z 一律使用 `checkpoint_best_regular.pth`。
- 不要用 `checkpoint_best_total.pth` 當 semantic head 來源；它是 detection mAP 選出的 checkpoint，semantic head 較弱。
- `semantic_loss_coef=2.0`、`batch_size=16` 是 sweep 最佳點。
- 約 3.5 小時 / 30 epochs。

---

### Step 4. Stage1：抽 `z_joint` + backbone global cache

Purpose：建立 Step 5 distillation 與 Step 6 case DB swap 所需 feature cache。

Depends on：

- Step 1：`$ART/db/case_db_raw` + `$ART/models/text_anchors.pt`
- Step 2：`$ART/models/rfdetr/checkpoint_best_total.pth`
- Step 3：`$ART/models/joint_rfdetr/checkpoint_best_regular.pth`

Produces：

- `$ART/db/z_joint`
- `$ART/db/dino_global`

Recommended command：

```bash
$PY -m diagnosis_model.build_pipeline.build_case_dbs --to stage1
```

Expanded commands：

```bash
# 4a. joint semantic z
$PY -m diagnosis_model.grod.extract_z_joint \
    --case_db_dir $ART/db/case_db_raw \
    --joint_ckpt $ART/models/joint_rfdetr/checkpoint_best_regular.pth \
    --anchors $ART/models/text_anchors.pt \
    --image_root $DET \
    --output_dir $ART/db/z_joint --splits train valid

# 4b. 影像特徵編碼器 global cache
for SP in train valid; do
  $PY -m diagnosis_model.grod.extract_dino_global \
      --case_db_dir $ART/db/case_db_raw \
      --split $SP \
      --det_ckpt $ART/models/rfdetr/checkpoint_best_total.pth \
      --image_root $DET/$SP \
      --output_dir $ART/db/dino_global
done
```

Gotchas：

- `extract_z_joint` 必須使用 `checkpoint_best_regular.pth`。
- `dino_global` 是 deterministic cache，但仍依賴 Step 2 detector checkpoint。
- 若 production 已切到 DINOv3，不建議在說明文字中固定寫「DINOv2 / 1536-d」，除非該維度確定與目前 backbone 相符。

---

### Step 5. 訓練 distilled global MLP

Purpose：將影像特徵編碼器的 global feature 蒸餾到 raw SigLIP2 whole-image global space。

Depends on：

- Step 1：`$ART/db/case_db_raw`
- Step 4：`$ART/db/dino_global`

Produces：

- `$ART/models/distilled_global_rawP/global_embed_state_dict.pt`

Command：

```bash
$PY -m diagnosis_model.grod.distill_global_mlp \
    --dino_dir   $ART/db/dino_global \
    --target_db  $ART/db/case_db_raw \
    --out_dir    $ART/models/distilled_global_rawP
```

---

### Step 6. Stage2：組 `case_db_jointDistRawP` + `soft_inputs` + gating artifacts

Purpose：建立 OAVLE Aggregator / CEAM 訓練前的 deterministic artifacts。

Depends on：

- Step 1：`$ART/db/case_db_raw`
- Step 3：`$ART/models/joint_rfdetr/checkpoint_best_regular.pth`
- Step 4：`$ART/db/z_joint`
- Step 5：`$ART/models/distilled_global_rawP/global_embed_state_dict.pt`

Produces：

- `$ART/db/case_db_jointDistRaw`
- `$ART/db/case_db_jointDistRawP`
- `$ART/db/case_db_jointDistRawP/train_candidate_pool.pt`
- `$ART/db/case_db_jointDistRawP/teacher_train_train.pt`
- `$ART/db/soft_inputs`
- `$ART/db/disease_perquery/{train,val}.pt`
- `$ART/models/disease_head/lesion_threshold.json`
- `data/processed/current/thresholds.json`

Recommended command：

```bash
$PY -m diagnosis_model.build_pipeline.build_case_dbs --from stage2 --to stage2
```

Expanded commands：

```bash
# 6a. lesion_embs <- joint semantic z
$PY -m diagnosis_model.grod.rebuild_case_db \
    --src_case_db $ART/db/case_db_raw \
    --hs_dir $ART/db/z_joint --from_joint \
    --out_case_db $ART/db/case_db_jointDistRaw

# 6b. global_emb <- distilled global
$PY -m diagnosis_model.grod.build_case_db_swap_global \
    --src_db $ART/db/case_db_jointDistRaw \
    --global_dir $ART/models/distilled_global_rawP \
    --global_prefix distilled_global \
    --dst_db $ART/db/case_db_jointDistRawP

# 6c. candidate pool
$PY -m diagnosis_model.cause_inference.preprocessing.build_train_candidate_pool \
    --case_db_dir $ART/db/case_db_jointDistRawP \
    --output_path $ART/db/case_db_jointDistRawP/train_candidate_pool.pt \
    --top_k_cases 20 --alpha_global 0.25 --beta_lesion 0.75 \
    --lesion_match max_mean --semantic_threshold 0.95

# 6d. teacher score table
$PY -m diagnosis_model.cause_inference.preprocessing.build_teacher_table \
    --case_db_dir $ART/db/case_db_jointDistRawP \
    --output_path $ART/db/case_db_jointDistRawP/teacher_train_train.pt \
    --alpha_global 0.25 --beta_lesion 0.75 --lesion_match max_mean

# 6e. soft_inputs
$PY -m diagnosis_model.grod.extract_soft_inputs \
    --joint_ckpt $ART/models/joint_rfdetr/checkpoint_best_regular.pth \
    --global_sd  $ART/models/distilled_global_rawP/global_embed_state_dict.pt \
    --anchors    $ART/models/text_anchors.pt \
    --case_db_dir $ART/db/case_db_jointDistRawP \
    --img_root   $DET \
    --out_dir    $ART/db/soft_inputs

# 6f. gating artifacts
$PY -m diagnosis_model.grod.extract_disease_perquery
$PY -m diagnosis_model.grod.compute_lesion_threshold
$PY -m diagnosis_model.grod.calibrate_thresholds
```

Gotchas：

- 6c / 6d 約各 3 分鐘。
- `build_case_db_swap_global` 預設會從 source DB symlink 共享檔。
- 6c / 6d 會在 `case_db_jointDistRawP` 內把 pool / teacher symlink 覆蓋成真檔，屬正常行為。
- gating 依賴 joint + distilled，不依賴 OAVLE Aggregator / CEAM，所以不應排在 Step 7 之後。

---

## 4. OAVLE / CEAM：Step 7–11

論文命名：

- part1 encoder：**OAVLE**
- part2 attribution module：**CEAM**

code / artifact 命名沿用舊名：

- module path：`diagnosis_model.grod.*`
- artifacts：`encoder_grod_soft` / `ceah_grod_soft` / `grace_e2e`
- demo mode：`grod` / `grod_soft`

推論保留全部 300 queries；聚合 / 歸因權重由 Region Gate（∅-sink softmax）產生；篩選 / abstain 使用原始 objectness。

### Step 7. 訓練 OAVLE Aggregator

Purpose：用 Step 6 的 `soft_inputs` 訓練 case encoder。

Depends on：

- Step 6：`$ART/db/soft_inputs`
- Step 6：`$ART/db/case_db_jointDistRawP`
- Step 6：`$ART/db/case_db_jointDistRawP/teacher_train_train.pt`

Produces：

- `$ART/models/encoder_grod_soft/best_encoder.pt`

Command：

```bash
$PY -m diagnosis_model.grod.train_case_encoder_soft \
    --soft_dir   $ART/db/soft_inputs \
    --case_db_dir $ART/db/case_db_jointDistRawP \
    --teacher_path $ART/db/case_db_jointDistRawP/teacher_train_train.pt \
    --output_dir $ART/models/encoder_grod_soft
```

---

### Step 8. 訓練 Region Gate + 建 bank

Purpose：以 Step 7 encoder 為起點，聯合訓練 Region Gate 與 aggregator，並套用 gate 產生 gated soft inputs 與 retrieval bank。

Depends on：

- Step 6：`$ART/db/soft_inputs`
- Step 6：`$ART/db/case_db_jointDistRawP`，含 teacher
- Step 7：`$ART/models/encoder_grod_soft/best_encoder.pt`

Produces：

- `$ART/models/encoder_grod_soft/best_encoder.pt`，含 `gate_state`
- `$ART/models/encoder_grod_soft/bank_z_soft.pt`
- `$ART/db/soft_inputs_gated/{train,valid}.pt`

Command：

```bash
# 8a. joint train Region Gate + encoder；偵測器凍結
$PY -m diagnosis_model.grod.finetune_e2e_grace \
    --enc_ckpt $ART/models/encoder_grod_soft/best_encoder.pt \
    --output_dir $ART/models/grace_e2e

# 8b. apply gate -> gated soft_inputs + bank
$PY -m diagnosis_model.grod.apply_gate_to_soft \
    --gate_ckpt $ART/models/grace_e2e/best.pt \
    --soft_dir  $ART/db/soft_inputs \
    --out_dir   $ART/db/soft_inputs_gated

# 8c. copy to production paths
cp $ART/models/grace_e2e/best.pt            $ART/models/encoder_grod_soft/best_encoder.pt
cp $ART/db/soft_inputs_gated/bank_z_soft.pt $ART/models/encoder_grod_soft/bank_z_soft.pt
```

Gotchas：

- `finetune_e2e_grace` 訓練時印出的內部 300-q 協定數值不作正式數字。
- 正式數字一律用 `eval_ceah_soft_paper`。
- Step 9 需要本步的 `bank_z_soft.pt`，因此必須先完成本步。
- demo 的 hard gate 模式不另訓練；共用此套 artifacts，只在推論時將 objectness 依 τ 二值化。

---

### Step 9. 訓練 CEAM

Purpose：訓練病因-證據歸因模組 CEAM。

Depends on：

- Step 6：`$ART/db/case_db_jointDistRawP`
- Step 6：`$ART/db/case_db_jointDistRawP/train_candidate_pool.pt`
- Step 8：`$ART/db/soft_inputs_gated`
- Step 8：`$ART/models/encoder_grod_soft/best_encoder.pt`
- Step 8：`$ART/models/encoder_grod_soft/bank_z_soft.pt`

Produces：

- `$ART/models/ceah_grod_soft/best_ceah.pt`

Command：

```bash
$PY -m diagnosis_model.grod.train_ceah_soft \
    --case_db_dir $ART/db/case_db_jointDistRawP \
    --train_pool_path $ART/db/case_db_jointDistRawP/train_candidate_pool.pt \
    --soft_dir    $ART/db/soft_inputs_gated \
    --encoder_ckpt $ART/models/encoder_grod_soft/best_encoder.pt \
    --bank_path   $ART/models/encoder_grod_soft/bank_z_soft.pt \
    --output_dir  $ART/models/ceah_grod_soft \
    --top_k_lesions 32
```

Gotchas：

- `ceah_grod_soft` 不能在 Step 8 之前訓練，因為它需要 `bank_z_soft.pt`。
- `--soft_dir` 必須使用 `soft_inputs_gated`，CEAM 要配合 production gate 的權重分佈。

---

### Step 10. 校準 demo 病因聚合門檻：`fold_thresh`

Purpose：demo 對每個 query 候選池做 agglomerative folding。`fold_thresh` 由離線 LLM taxonomy ARI 最大化選出，推論時不呼叫 LLM。

Depends on：

- Step 6：`$ART/db/case_db_jointDistRawP`
- Step 8：`$ART/models/encoder_grod_soft/bank_z_soft.pt`
- `$ART/cause_clusters_llm.json`

Produces：

- `data/processed/current/thresholds.json` 新增或更新 `fold_thresh`

Command：

```bash
$PY -m diagnosis_model.grod.calibrate_fold_threshold
```

Notes：

- 預設網格：`0.30:1.01:0.05`。
- 若換資料集後最佳點卡在邊界，再放寬，例如 `--cut_grid 0.30:1.21:0.05`。
- 本步只需 bank，不依賴 Step 9 CEAM。
- 寫入是 read-modify-write，會保留既有 objectness keys。
- 若之後重跑 `calibrate_thresholds`，因該步會整檔覆寫，需再跑一次本步補回 `fold_thresh`。
- 缺鍵時 demo fallback 0.75。

---

### Optional. 重建 LLM-judge taxonomy

只在 taxonomy 不存在、病因字串大改、或需要 100% coverage 時重建。一般資料版本切換若 `$ART/cause_clusters_llm.json` 已可用，不需要重跑。

Command：

```bash
ollama serve
ollama pull <model>     # e.g. gemma4:26b

$PY -m diagnosis_model.cause_inference.preprocessing.cause_resolve_llm \
    --case_db $ART/db/case_db_jointDistRawP \
    --judge ollama --model gemma4:26b
```

Cosine-only baseline：

```bash
$PY -m diagnosis_model.cause_inference.preprocessing.cause_resolve_llm \
    --case_db $ART/db/case_db_jointDistRawP \
    --judge cosine --judge_cosine_tau 0.95
```

保留的核心設定：

- 預設 `tau=0.90 / k=32 / tau_merge=0.94`。
- embedding 只做 blocking，LLM 才做 same / different 判決。
- 合併階段使用單輪 medoid-anchored merge。
- 不建議 `--merge_rounds >1`，可能因 intransitive / bridge-sentence 造成跨類過併。

更詳細的 entity-resolution 理由與失敗案例，建議放到 `docs/LLM_TAXONOMY.md`，不要塞在 BUILD 主流程。

---

### Step 11. 校準 demo 病因顯示門檻：`cause_score_thresh` + `cause_max_n`

Purpose：demo 顯示折疊後 min-max CEAM score ≥ τ 的病因，最多顯示 `cause_max_n`，至少顯示 1 個。

Depends on：

- Step 6：`$ART/db/case_db_jointDistRawP`
- Step 8：`$ART/db/soft_inputs_gated`
- Step 8：`$ART/models/encoder_grod_soft/bank_z_soft.pt`
- Step 9：`$ART/models/ceah_grod_soft/best_ceah.pt`
- Step 10：`fold_thresh`

Produces：

- `data/processed/current/thresholds.json` 新增或更新：
  - `cause_score_thresh`
  - `cause_max_n`

Command：

```bash
$PY -m diagnosis_model.grod.calibrate_cause_threshold
```

Notes：

- 預設：production `k=3`、`n_max=6`、`target_avg=5`、τ grid `0.10:0.96:0.05`。
- 更精簡可用 `--target_avg 4`；更寬上限可用 `--n_max 8`。
- 本步需要 CEAM，且需要 Step 10 的 `fold_thresh`。
- 寫入是 read-modify-write，會保留 objectness / fold keys。
- 若重跑 `calibrate_thresholds`，需重跑 Step 10 與 Step 11 補回 demo calibration keys。
- `top_n_causes` 在 serve / demo 中語意已改為顯示上限，不是固定顯示數。

---

### Step 12. OAVLE smoke test

```bash
$PY -m diagnosis_model.grod.gpu_infer_soft \
    --image data/processed/current/full/test/<某張>.jpg --verify
```

---

## 5. base 分支：B0–B5

base 是常規分離式 baseline：

- 不使用 joint DETR-routed z。
- 使用標準微調 SigLIP2 編 lesion crop。
- global 使用 raw SigLIP2。
- 不從 `case_db_raw` swap，而是自己建 `case_db_base`。
- demo 的 base mode 使用 Step 2 RF-DETR 作 detector。

### B0. 微調 SigLIP2

Depends on：

- `$FULL`
- `$SYM`

Produces：

- `$ART/models/siglip2_base_finetuned`

Command：

```bash
$PY diagnosis_model/vl_classifier/train.py \
    --data_root data/processed/current/full \
    --symptoms_file $SYM \
    --output_dir $PWD/$ART/models/siglip2_base_finetuned
```

Notes：

- 一傷口一 caption。
- 無 multipos / fusion。
- 約 6 分鐘。

---

### B1. 建 `case_db_base`

Depends on：

- B0
- `$DET`

Produces：

- `$ART/db/case_db_base`

Command：

```bash
$PY -m diagnosis_model.cause_inference.preprocessing.build_case_database \
    --coco_train $DET/train/_annotations.coco.json --image_root_train $DET/train \
    --coco_valid $DET/valid/_annotations.coco.json --image_root_valid $DET/valid \
    --vlm_global google/siglip2-base-patch16-224 \
    --vlm_lesion $ART/models/siglip2_base_finetuned --raw_lesion \
    --output_dir $ART/db/case_db_base \
    --chunk_size 64 --img_batch_size 64 --text_batch_size 256
```

---

### B2. 建 base candidate pool

Depends on：

- B1：`$ART/db/case_db_base`

Produces：

- `$ART/db/case_db_base/train_candidate_pool.pt`

Command：

```bash
$PY -m diagnosis_model.cause_inference.preprocessing.build_train_candidate_pool \
    --case_db_dir $ART/db/case_db_base \
    --output_path $ART/db/case_db_base/train_candidate_pool.pt \
    --top_k_cases 20 --alpha_global 0.25 --beta_lesion 0.75 \
    --lesion_match max_mean --semantic_threshold 0.95
```

---

### B3. 建 base teacher table

Depends on：

- B1：`$ART/db/case_db_base`

Produces：

- `$ART/db/case_db_base/teacher_train_train.pt`

Command：

```bash
$PY -m diagnosis_model.cause_inference.preprocessing.build_teacher_table \
    --case_db_dir $ART/db/case_db_base \
    --output_path $ART/db/case_db_base/teacher_train_train.pt \
    --alpha_global 0.25 --beta_lesion 0.75 --lesion_match max_mean
```

---

### B4. 訓練 `encoder_base`

Depends on：

- B2：`$ART/db/case_db_base/train_candidate_pool.pt`
- B3：`$ART/db/case_db_base/teacher_train_train.pt`

Produces：

- `$ART/models/encoder_base/best_encoder.pt`

Command：

```bash
$PY -m diagnosis_model.cause_inference.train_case_encoder \
    --case_db_dir $ART/db/case_db_base \
    --teacher_path $ART/db/case_db_base/teacher_train_train.pt \
    --train_pool_path $ART/db/case_db_base/train_candidate_pool.pt \
    --output_dir $ART/models/encoder_base \
    --encoder_type deepsets --batch_size 256 --epochs 50 \
    --temp_target 0.1 --temp_pred 0.1 \
    --use_infonce --infonce_weight 0.5 --infonce_temp 0.07
```

---

### B5. 訓練 `ceah_base`

Depends on：

- B2：`$ART/db/case_db_base/train_candidate_pool.pt`

Produces：

- `$ART/models/ceah_base/best_ceah.pt`

Command：

```bash
$PY -m diagnosis_model.cause_inference.train_ceah \
    --case_db_dir $ART/db/case_db_base \
    --train_pool_path $ART/db/case_db_base/train_candidate_pool.pt \
    --output_dir $ART/models/ceah_base \
    --attribution_mode softmax --scoring_mode multiplicative \
    --lambda_sparsity 0.0 --text_dropout 0.5
```

---

## 6. Gotchas

### 6.1 `best_regular` vs `best_total`

Semantic z 相關流程一律用：

```text
$ART/models/joint_rfdetr/checkpoint_best_regular.pth
```

影響範圍：

- Step 4 `extract_z_joint`
- Step 6 `extract_soft_inputs`
- `gpu_infer_soft.py`

原因：

- `best_total` 是 detection mAP 選出，semantic head 可能較弱。
- `best_regular` 的 semantic head 訓練較完整。

### 6.2 `lesion_threshold.json` 不是 CLI runtime config

CLI soft inference 仍使用程式內常數：

```text
DEFAULT_LESION_THRESH=0.5
```

`lesion_threshold.json` 是離線推導與紀錄，不是 `gpu_infer_soft.py` runtime 讀取的檔案。

### 6.3 base 不套用 OAVLE τ

base 使用 Step 2 RF-DETR detector score；其 score scale 與 OAVLE joint objectness 不同，因此不共用 OAVLE τ。

### 6.4 `case_db_swap_global` symlink 行為

`build_case_db_swap_global` 會把 meta / cause text embeddings / teacher / pool 等共享檔從 source DB symlink。後續重建 pool / teacher 會把 symlink 覆蓋成真檔，這是預期行為。

### 6.5 命名對照

| 名稱 | 意義 |
|---|---|
| `case_db_raw` | global = raw SigLIP2；lesion = raw SigLIP2 |
| `case_db_jointDistRaw` | lesion = joint semantic z；global 仍沿用 raw SigLIP2 |
| `case_db_jointDistRawP` | lesion = joint semantic z；global = distilled raw global |
| `distilled_global_rawP` | 影像特徵編碼器 global 蒸餾到 raw SigLIP2 global space |
| `soft_inputs` | ungated OAVLE training inputs |
| `soft_inputs_gated` | Region Gate 後的 production training / eval inputs |

### 6.6 重跑順序提醒

- 重跑 `calibrate_thresholds` 後，需補跑 Step 10 / Step 11，否則 `fold_thresh`、`cause_score_thresh`、`cause_max_n` 可能消失。
- 重跑 Step 8 後，需確認 production paths 已更新：
  - `$ART/models/encoder_grod_soft/best_encoder.pt`
  - `$ART/models/encoder_grod_soft/bank_z_soft.pt`
  - `$ART/db/soft_inputs_gated`
- 重跑 Step 3 後，下游 Step 4–11 原則上都應視為 stale。
