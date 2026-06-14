# BUILD_PIPELINE：資料版本切換後重建 artifacts

用途：`dataset_pipeline/db_pipeline` 產生新資料版本到 `data/processed/current/` 後，用本文件重建 `GROD`、`base` 與 `demo/app_gradio.py` 需要的 artifacts。

> 命名：GROD 只有一條訓練線（hard 已退役）。demo 仍有 `grod`（硬閘）/ `grod_soft` 兩種**推論模式**，但共用同一套 GROD artifacts，差別只在推論時 per-query 權重是否在 τ 二值化。on-disk 的 encoder / CEAH / bank / 腳本沿用歷史的 `_soft` 後綴（`encoder_grod_soft` / `ceah_grod_soft` / `bank_z_soft.pt` / `gpu_infer_soft.py`），prose 一律稱 GROD。

---

## 1. Quickstart

### 1.1 環境變數

```bash
PY=/home/lab603/anaconda3/envs/SDM/bin/python
cd /mnt/ssd/YJ/fish_disease_system

DET=data/processed/current/detection
FULL=data/processed/current/full
SYM=data/processed/current/symptoms.json
ART=data/processed/current/artifacts

mkdir -p $ART/models $ART/db
```

### 1.2 你要重建哪條？

| 目標 | 最短路徑 | 需要先完成 |
|---|---|---|
| 只重建 raw artifacts | Step 1 | db_pipeline 已完成 |
| 重建 GROD | Step 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 | 每個訓練 gate 依序完成 |
| 重建 base | B0 → B1 → B2/B3 → B4/B5 | `$FULL` / `$DET` / `$SYM` |
| demo 三模式都可切換 | GROD + base + gating（demo 的 `grod` 硬閘＝GROD 模型 + τ 二值化，無獨立 artifacts） | 全部完成 |

---

## 2. Orchestrators：照這個順序跑

`diagnosis_model/build_pipeline/` 有兩支 orchestrator：

| Orchestrator | 做什麼 | 是否訓練 |
|---|---|---:|
| `build_raw` | 建 `text_anchors.pt` + `case_db_raw` | 否 |
| `build_case_dbs` | 串接非訓練的 feature / case DB / gating / bank 組裝 | 否 |

> 訓練步驟仍手動跑，因為需要 GPU、epoch、超參與實驗控制。Orchestrator 只負責 deterministic glue。

### 2.1 GROD 主執行表

照順序跑。每一列完成後，才進下一列。

| 順序 | 執行內容 | 指令 / 區段 | 必須先有 | 產出後解鎖 |
|---:|---|---|---|---|
| 0 | db_pipeline 建好新資料版本 | 外部流程 | 原始資料 | `$DET` / `$FULL` / `$SYM` |
| 1 | 建 raw artifacts：anchors + raw case DB | Step 1 / `build_raw` | `$DET` + `$SYM` | Step 2、Step 3、Step 4 |
| 2 | 訓 base RF-DETR detector | Step 2 | Step 1 可已完成；實際只吃 `$DET` | Step 3、Step 4 的 `dino_global`、demo base detector |
| 3 | 訓 joint RF-DETR | Step 3 | Step 1 anchors + Step 2 rfdetr | Step 4 的 `z_joint`、Step 6 `soft_inputs`、GROD inference |
| 4 | 抽 `z_joint` + `dino_global` | Step 4 / `build_case_dbs --to stage1` | Step 1 + Step 2 + Step 3 | Step 5 |
| 5 | 訓 distilled global MLP | Step 5 | Step 1 `case_db_raw` + Step 4 `dino_global` | Step 6 |
| 6 | 組 `case_db_jointDistRawP` + pool/teacher + `soft_inputs` + gating | Step 6 / `build_case_dbs --from stage2 --to stage2` | Step 3 joint + Step 5 distilled | 7、demo gating |
| 7 | 訓 GROD encoder | Step 7 | Step 6 `soft_inputs` + teacher | Step 8 |
| 8 | 建 GROD bank `bank_z_soft.pt` | Step 8 / `build_case_dbs --from stage3` | Step 7 encoder | Step 9 |
| 9 | 訓 GROD CEAH | Step 9 | Step 8 bank | `gpu_infer_soft.py` attribution |

重要順序：

```text
build_raw
→ train rfdetr
→ train joint
→ stage1: z_joint + dino_global
→ train distilled_global
→ stage2: case_db_jointDistRawP + pool/teacher + soft_inputs + gating
→ GROD encoder
→ stage3: GROD bank
→ GROD CEAH
```

### 2.2 對應指令骨架

```bash
# Step 1. raw deterministic artifacts
$PY -m diagnosis_model.build_pipeline.build_raw

# 可選：換 VLM 做消融，輸出會加 __<tag>，避免互蓋
$PY -m diagnosis_model.build_pipeline.build_raw --vlm openai/clip-vit-base-patch32

# Step 2. 手動訓 base RF-DETR detector：見 Step 2

# Step 3. 手動訓 joint RF-DETR：見 Step 3

# Step 4. stage1：extract_z_joint + extract_dino_global
$PY -m diagnosis_model.build_pipeline.build_case_dbs --to stage1

# Step 5. 手動訓 distilled_global_rawP：見 Step 5

# Step 6. stage2：case_db_jointDistRawP + pool/teacher + soft_inputs + gating
$PY -m diagnosis_model.build_pipeline.build_case_dbs --from stage2 --to stage2

# Step 7. 手動訓 GROD encoder：見 Step 7

# Step 8. stage3：build_soft_bank
$PY -m diagnosis_model.build_pipeline.build_case_dbs --from stage3

# Step 9. 手動訓 GROD CEAH：見 Step 9
```

### 2.3 base 分支順序

base 是獨立 baseline，不依賴 `joint_rfdetr`、`distilled_global_rawP` 或 `case_db_jointDistRawP`。但 demo 的 base detector 會用 Step 2 的 `rfdetr`。

| 順序 | 執行內容 | 區段 | 必須先有 |
|---:|---|---|---|
| B0 | 訓 SigLIP2 baseline lesion encoder | B0 | `$FULL` + `$SYM` |
| B1 | 建 `case_db_base` | B1 | B0 + `$DET` |
| B2 | 建 candidate pool | B2 | B1 |
| B3 | 建 teacher table | B3 | B1 |
| B4 | 訓 `encoder_base` | B4 | B2 + B3 |
| B5 | 訓 `ceah_base` | B5 | B2 |

可併行規則：

- B0–B5 可以和 GROD 主線分開排程。
- 若要 demo 三模式都可用，最後需要同時完成：Step 7/8/9、B0–B5，以及 Step 6 內的 gating（demo 的 `grod` 硬閘模式不需額外訓練，共用 GROD artifacts）。

---

## 3. Pipeline map

### 3.1 輸入資料路徑

| 名稱 | 路徑 | 用途 |
|---|---|---|
| symptoms | `$SYM` | symptom 類別空間，本版 15 類，含 `healthy_region=0` |
| detection view | `$DET` | 偵測 view；box 全併 `ABNORMAL`，排除 healthy；image 仍帶 `global_causes_zh` / `overall`；每 box 內建 `symptom_category_id` |
| full view | `$FULL` | 完整 symptom 類別；供 vl_classifier / base SigLIP2 finetune 使用 |

舊路徑對照：

| 舊路徑 | 新路徑 |
|---|---|
| `data/raw/symptoms.json` | `data/processed/current/symptoms.json` |
| `data/detection/coco/_merged` | `data/processed/current/detection` |
| `data/coco/_merged` | `data/processed/current/full` |

### 3.2 輸出根目錄

所有產物統一寫到：

```bash
data/processed/current/artifacts/
```

`current` 是 symlink，指向目前資料版本目錄。因此 artifacts 會跟著其訓練資料版本走。

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
    ├── case_db_jointDistRaw/
    ├── case_db_jointDistRawP/
    └── case_db_base/
```

### 3.3 Artifact layers

| 層級 | 內容 | 資料變動後是否重建 | 是否訓練 |
|---|---|---:|---:|
| A. 資料衍生 / 凍結層 | `text_anchors.pt`、`case_db_raw`、`dino_global` | 必須重建 | 否 |
| B. 訓練層 | `rfdetr`、`joint_rfdetr`、`distilled_global_rawP`、`encoder_*`、`ceah_*`、base SigLIP2 finetune、`case_db_base` | 必須重訓 | 是 |

原則：**資料變 → A 層重建 → B 層重訓**。

### 3.4 Branch 共用關係

| Artifact / Step | GROD | base | demo |
|---|---:|---:|---:|
| `text_anchors.pt` | ✅ | ✅ 分類用 | ✅ |
| `rfdetr` | ✅ 間接依賴 | ✅ detector | ✅ |
| `joint_rfdetr` | ✅ | ❌ | grod / grod_soft |
| `distilled_global_rawP` | ✅ | ❌ | grod / grod_soft |
| `case_db_jointDistRawP` | ✅ | ❌ | grod / grod_soft |
| `encoder_grod_soft` | ✅ | ❌ | grod / grod_soft |
| `ceah_grod_soft` | ✅ | ❌ | grod / grod_soft |
| `case_db_base` | ❌ | ✅ | base |

> demo 的 `grod`（硬閘）與 `grod_soft` 模式共用同一套 `encoder_grod_soft` / `ceah_grod_soft`；硬閘只是在推論時把 per-query 權重在 τ 二值化成 {0,1}，不需要獨立訓練的 encoder / CEAH。

### 3.5 DAG：依賴關係，不代表文件 step 編號

```text
Step 1 anchors ───────┐
Step 2 rfdetr ──► Step 3 joint ──► Step 4 z_joint ─┐
       │                                             │
       └────────────────────────► Step 4 dino_global ─► Step 5 distilled_global ─┐
Step 1 case_db_raw ───────────────────────────────────────────────────────────────┘

Step 4 z_joint + Step 5 distilled_global + Step 1 case_db_raw
  └─► Step 6 case_db_jointDistRawP + pool/teacher + soft_inputs + gating
        └─► Step 7 encoder_grod_soft ──► Step 8 bank_z_soft ──► Step 9 ceah_grod_soft
              (demo 的 grod 硬閘與 grod_soft 模式共用此 GROD encoder / CEAH)
```

---

## 4. Gating rule

Gating 規則只在此處定義，後續分支不再重複。

| 使用端 | 病灶門檻 τ |
|---|---|
| `gpu_infer_soft.py` | 寫死 `DEFAULT_LESION_THRESH=0.5` |
| `demo/app_gradio.py` | 讀 `thresholds.json` 作 slider 預設 |

補充：

- `lesion_threshold.json` 是離線校準紀錄，不是 CLI runtime config。
- soft（`gpu_infer_soft.py` / demo `grod_soft`）：300 queries 全留，以 `sigmoid(objectness)` 當 soft weight；`max_w < τ` 則 abstain。
- hard 閘（demo `grod` 模式）：同一顆 GROD 模型，僅把 per-query 權重在 τ（`display_thresh`）二值化成 {0,1}（`objectness > τ` 留、其餘 0）；DeepSets/CEAH 在 {0,1} 權重下 bytes-exactly 退化成硬路徑。全 0（無 lesion 過關）則 abstain。
- base 使用自己的 RF-DETR detector score，分數尺度不同，不套用 GROD 的 τ。
- 詳細設計理由見 `diagnosis_model/grod/LESION_GATE.md`。

---

## 5. 主線 steps：按實際執行順序

### Step 1. 建 raw artifacts：`text_anchors.pt` + `case_db_raw`

Purpose：建立資料版本切換後最早可建的 deterministic artifacts。這一步不需要訓練，不依賴 RF-DETR / joint。

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

Gotcha：

- 類別數若改變，必須使用新 `$SYM` 重建 anchors。
- `case_db_raw` 是所有 grod swap 的底，不要等到 joint 後才建。

---

### Step 2. 訓練 base RF-DETR detector

Purpose：訓練 detector，供 joint 的 `--pretrain_weights` 使用，也供 base / demo detector 使用。

Depends on：

- `$DET`

Produces：

- `$ART/models/rfdetr/checkpoint_best_total.pth`

Command：

```bash
cd diagnosis_model/detection

$PY train_rfdetr.py \
    --dataset_dir /mnt/ssd/YJ/fish_disease_system/$DET \
    --output_dir  /mnt/ssd/YJ/fish_disease_system/$ART/models/rfdetr

cd ../..
```

Notes：

- 直接執行 `train_rfdetr.py`，不是 `python -m`。
- 約數小時 / 100 epochs。

---

### Step 3. 訓練 joint detector + semantic head

Purpose：凍 backbone，訓 decoder 上的 region heads：box / objectness / semantic z。

Depends on：

- Step 1: `$ART/models/text_anchors.pt`
- Step 2: `$ART/models/rfdetr/checkpoint_best_total.pth`

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

Gotcha：

- downstream semantic z 一律用 `checkpoint_best_regular.pth`。
- 不要用 `checkpoint_best_total.pth` 作 semantic head 來源；`best_total` 是 detection mAP 選出的 checkpoint，semantic head 較弱。
- `semantic_loss_coef=2.0`、`batch_size=16` 是 sweep 最佳點。
- 約 3.5 小時 / 30 epochs。

---

### Step 4. Stage1：抽 `z_joint` + `dino_global`

Purpose：建立 Step 5 distillation 與 Step 6 case DB swap 需要的 feature cache。

Depends on：

- Step 1: `$ART/db/case_db_raw` + `$ART/models/text_anchors.pt`
- Step 2: `$ART/models/rfdetr/checkpoint_best_total.pth`
- Step 3: `$ART/models/joint_rfdetr/checkpoint_best_regular.pth`

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

# 4b. RF-DETR backbone global, 1536-d DINOv2 neck feature
for SP in train valid; do
  $PY -m diagnosis_model.grod.extract_dino_global \
      --case_db_dir $ART/db/case_db_raw \
      --split $SP \
      --det_ckpt $ART/models/rfdetr/checkpoint_best_total.pth \
      --image_root $DET/$SP \
      --output_dir $ART/db/dino_global
done
```

Gotcha：

- `extract_z_joint` 必須使用 `checkpoint_best_regular.pth`。
- `dino_global` 是 deterministic cache，但依賴已訓好的 Step 2 detector checkpoint。

---

### Step 5. 訓練 distilled global MLP

Purpose：將 DINO neck 1536-d 蒸餾到 raw SigLIP2 whole-image global space。

Depends on：

- Step 1: `$ART/db/case_db_raw`
- Step 4: `$ART/db/dino_global`

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

### Step 6. Stage2：組 `case_db_jointDistRawP` + `soft_inputs` + gating

Purpose：把 GROD 各步共用的 case DB、teacher/pool、`soft_inputs` 與 demo gating artifacts 一次建齊。這是 GROD encoder / CEAH 訓練前的最後 deterministic gate。

Depends on：

- Step 1: `$ART/db/case_db_raw`
- Step 3: `$ART/models/joint_rfdetr/checkpoint_best_regular.pth`
- Step 4: `$ART/db/z_joint`
- Step 5: `$ART/models/distilled_global_rawP/global_embed_state_dict.pt`

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
# 6a. lesion_embs <- joint z
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

# 6e. soft_inputs（GROD encoder / CEAH 訓練輸入）
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

Gotcha：

- 6c / 6d 約各 3 分鐘。
- `build_case_db_swap_global` 預設會把共享檔從 `src_db` symlink。
- 6c / 6d 會在 `case_db_jointDistRawP` 內寫入真檔，覆蓋 pool / teacher symlink，這是正常行為。
- gating 依賴 joint + distilled，不依賴 GROD encoder / CEAH，所以不應排在 Step 7/8/9 後面。

---

## 6. GROD encoder / bank / CEAH（Step 7–9）

GROD 推論保留全部 300 queries，以 `sigmoid(objectness)` 作連續權重（不做 hard lesion 篩選）。這三步共用主線的：

- Step 1: `text_anchors.pt` + `case_db_raw`
- Step 3: `joint_rfdetr/checkpoint_best_regular.pth`
- Step 5: `distilled_global_rawP`
- Step 6: `case_db_jointDistRawP` + `soft_inputs` + gating artifacts

產出 `encoder_grod_soft` / `ceah_grod_soft`（on-disk 沿用 `_soft` 後綴）。demo 的 `grod`（硬閘）模式不另外訓練，直接共用這兩個 artifacts，只在推論時把 per-query 權重在 τ 二值化成 {0,1}。

### Step 7. 訓練 GROD Aggregator

Purpose：用 Step 6 的 `soft_inputs` 訓練 GROD case encoder。

Depends on：

- Step 6: `$ART/db/soft_inputs`
- Step 6: `$ART/db/case_db_jointDistRawP`
- Step 6: `$ART/db/case_db_jointDistRawP/teacher_train_train.pt`

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

### Step 8. 建 GROD bank

Purpose：用 GROD encoder 建 retrieval bank。`ceah_grod_soft` 必須等這一步完成後才能訓。

Depends on：

- Step 6: `$ART/db/soft_inputs`
- Step 7: `$ART/models/encoder_grod_soft/best_encoder.pt`

Produces：

- `$ART/models/encoder_grod_soft/bank_z_soft.pt`

Recommended command：

```bash
$PY -m diagnosis_model.build_pipeline.build_case_dbs --from stage3
```

Expanded command：

```bash
$PY -m diagnosis_model.grod.build_soft_bank \
    --soft_dir    $ART/db/soft_inputs \
    --encoder_ckpt $ART/models/encoder_grod_soft/best_encoder.pt \
    --out         $ART/models/encoder_grod_soft/bank_z_soft.pt
```

---

### Step 9. 訓練 GROD CEAH

Purpose：訓練 GROD 的 case evidence attribution head。

Depends on：

- Step 6: `$ART/db/case_db_jointDistRawP`
- Step 6: `$ART/db/case_db_jointDistRawP/train_candidate_pool.pt`
- Step 6: `$ART/db/soft_inputs`
- Step 7: `$ART/models/encoder_grod_soft/best_encoder.pt`
- Step 8: `$ART/models/encoder_grod_soft/bank_z_soft.pt`

Produces：

- `$ART/models/ceah_grod_soft/best_ceah.pt`

Command：

```bash
$PY -m diagnosis_model.grod.train_ceah_soft \
    --case_db_dir $ART/db/case_db_jointDistRawP \
    --train_pool_path $ART/db/case_db_jointDistRawP/train_candidate_pool.pt \
    --soft_dir    $ART/db/soft_inputs \
    --encoder_ckpt $ART/models/encoder_grod_soft/best_encoder.pt \
    --bank_path   $ART/models/encoder_grod_soft/bank_z_soft.pt \
    --output_dir  $ART/models/ceah_grod_soft \
    --top_k_lesions 32
```

Gotcha：

- `ceah_grod_soft` 不能在 Step 8 之前訓，因為它需要 `bank_z_soft.pt`。

---

### Step 10. 校準 demo 病因聚合門檻 `fold_thresh`

Purpose：demo 在每個 query 的候選池上以 agglomerative（pool-centered cosine、average
linkage）聚合近義病因，切點 `fold_thresh` 用**離線對齊 LLM taxonomy 的 ARI 最大化**來定，
凍結成常數；推論時不碰 LLM。與 `calibrate_thresholds`（objectness）同性質，都是寫進
`thresholds.json` 的 demo 校準常數。

Depends on：

- Step 6: `$ART/db/case_db_jointDistRawP`（`cause_text_embs.pt` + `train_cases.pt`）
- Step 8: `$ART/models/encoder_grod_soft/bank_z_soft.pt`
- `$ART/cause_clusters_llm.json`（LLM-judge taxonomy，只離線用；一次性建好通常不必重建。
  如需重建見下方「建 LLM-judge taxonomy」）

Produces：

- `data/processed/current/thresholds.json` 內新增 `fold_thresh` 鍵（保留既有 objectness 鍵）

Command：

```bash
$PY -m diagnosis_model.grod.calibrate_fold_threshold
# 預設網格 0.30:1.01:0.05（夠寬，fish 的 ARI 峰 ≈0.75 落在內部）。
# 換資料集後若最佳點仍卡在網格邊界，再放寬，例如 --cut_grid 0.30:1.21:0.05
```

Gotcha：

- 只需 Step 8 的 bank，不依賴 Step 9 的 CEAH（聚合只在 cause embedding 上做）。
- 寫入是 read-modify-write，**保留** `calibrate_thresholds` 寫的 objectness 鍵；但若之後重跑
  `calibrate_thresholds`（它整檔覆寫），需再跑一次本步補回 `fold_thresh`。
- taxonomy 為主因層級（同主因合併、跨類別不混），ARI 最佳點偏向**較激進合併**（fish 上 ≈0.75，
  0.70–0.80 為平台）；demo `_load_fold_thresh()` 直接讀此值，缺鍵則 fallback 0.75。

建 LLM-judge taxonomy（Phase 0.5，只在 taxonomy 不存在或病因字串大改時才需重建；用 Ollama、可斷點續跑）：

由 [`cause_resolve_llm.py`](cause_inference/preprocessing/cause_resolve_llm.py) 以 **entity-resolution（LLM 當裁判）**
建：embedding kNN **blocking**（高召回產候選）→ **pairwise LLM 判 same/different**（高精度、附理由）→
**medoid-anchored leader clustering**（Round 1）+ leader 間 looser-threshold **補併**（Round 2）。代表句取群內
medoid（真實字串、不改寫），每筆歸屬都有一筆 pairwise 判決背書、可逐筆稽核（`<output>.judgments.jsonl`），
判決快取於 `<output>.judge_cache.json`、含 prompt 指紋（改 prompt 自動失效重判），可斷點續跑。recall 與
precision 解耦：embedding 只負責產候選不做決定，LLM 做語意決定——這正是純嵌入門檻聚類做不到的（這些 cause
向量高度 anisotropic，random-pair cosine 中位數 ≈0.90，cos≥0.94 的近重複對仍有近三成其實是不同主因）。
取代舊的 greedy-incremental `cause_cluster_llm.py`（在成長 prompt 中漏比同義 → under-merge；保留作 baseline/ablation）。

```bash
ollama serve            # 另一個 terminal
ollama pull <model>     # e.g. gemma4:26b

# 輸出固定為 $ART/cause_clusters_llm.json（DEFAULT_OUTPUT）
$PY -m diagnosis_model.cause_inference.preprocessing.cause_resolve_llm \
    --case_db $ART/db/case_db_jointDistRawP \
    --judge ollama --model gemma4:26b
# 純嵌入 baseline（免 Ollama，同時就是「傳統門檻聚類」對照組）：
$PY -m diagnosis_model.cause_inference.preprocessing.cause_resolve_llm \
    --case_db $ART/db/case_db_jointDistRawP \
    --judge cosine --judge_cosine_tau 0.95
```

預設 `tau=0.90 / k=32 / tau_merge=0.94`（這些 cause 向量 anisotropic，blocking floor 須高、由 `k` 上限兜成本）。
合併階段是**單輪 medoid-anchored 合併**（`--merge_rounds` 預設 1）：重算每群 medoid 當錨、在 `tau_merge`
內取**全部** medoid 候選（不設 top-k 窗）做一次 leader-clustering。兩個召回關鍵：(1) 錨在 medoid 而非
round-1 leader——leader 可能偏離群中心，使同主因群（medoid cos >0.96）從沒被比到；(2) 不設 top-k 窗——
同主因對 cos 多 ≥0.96、跨主因多 <0.94，floor 太低（如 0.88）會把跨主因全塞進候選、擠出真 anchor。
**不要 `--merge_rounds >1`**：迭代會在每輪重算 medoid，把 LLM 的 intransitive / bridge-sentence 判決
（如「外傷後二次細菌感染」橫跨物理↔細菌）雪球成跨類過併（實測 4 輪→細菌群吃進物理、寄生蟲吃進病毒，
最大群衝到 20k）。單輪類別乾淨、最大群純淨；殘留同主因碎裂（如寄生蟲分數群）源於 LLM 對近同句的本質
不一致，多迭代救不了。要更激進/保守合併改 judge prompt 的主因粒度，不靠超參。

不重建也能用：Step 10 校準已把**沒對到 taxonomy 的 cause 排除在 ARI 之外**（不塞進
單一 -1 桶汙染門檻），所以覆蓋不足也夠校準 scalar；只有想要 100% 覆蓋才需重建。

### GROD 驗證

```bash
$PY -m diagnosis_model.grod.gpu_infer_soft \
    --image data/processed/current/full/test/<某張>.jpg --verify
```

---

## 7. base 分支

base 是常規分離式 baseline：

- 不使用 joint 的 DETR-routed z。
- 使用一顆標準微調 SigLIP2 編 lesion crop。
- global 使用 raw SigLIP2。
- 不從 `case_db_raw` swap，而是自己建 `case_db_base`。
- demo 的 base 模式會使用 Step 2 的 `rfdetr` 作 detector。

### B0. 微調 SigLIP2

Purpose：用 full view 訓練標準 lesion classifier。

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

Purpose：global 使用 raw SigLIP2，lesion 使用 base 微調 SigLIP2。

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

- B1: `$ART/db/case_db_base`

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

- B1: `$ART/db/case_db_base`

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

- B2: `$ART/db/case_db_base/train_candidate_pool.pt`
- B3: `$ART/db/case_db_base/teacher_train_train.pt`

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

- B2: `$ART/db/case_db_base/train_candidate_pool.pt`

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

## 8. Inference artifact mapping

### 8.1 `gpu_infer_soft.py`

| 參數 | artifact | 產生步驟 |
|---|---|---|
| `--anchors` | `$ART/models/text_anchors.pt` | Step 1 |
| `--joint_ckpt` | `$ART/models/joint_rfdetr/checkpoint_best_regular.pth` | Step 3 |
| `--global_sd` | `$ART/models/distilled_global_rawP/global_embed_state_dict.pt` | Step 5 |
| `--case_db_dir` | `$ART/db/case_db_jointDistRawP` | Step 6 |
| `--enc_ckpt` | `$ART/models/encoder_grod_soft/best_encoder.pt` | Step 7 |
| `--bank_path` | `$ART/models/encoder_grod_soft/bank_z_soft.pt` | Step 8 |
| `--ceah_ckpt` | `$ART/models/ceah_grod_soft/best_ceah.pt` | Step 9 |
| `--det_thresh` | hard-coded `DEFAULT_LESION_THRESH=0.5` | Step 6 產生離線紀錄 |

---

## 9. Gotchas

### 9.1 `best_regular` vs `best_total`

Semantic z 相關流程一律用：

```text
$ART/models/joint_rfdetr/checkpoint_best_regular.pth
```

原因：

- `best_total` 實測等於 `best_ema`。
- `best_total` 是 detection mAP 選出，EMA best 可能落在早期 epoch。
- 其 semantic head 可能是 EMA 平均 + 偵測早期狀態，語意較弱。
- `best_regular` 的 semantic head 訓練最完整。

影響範圍：

- Step 4 `extract_z_joint`
- `gpu_infer_soft.py`
- Step 6 `extract_soft_inputs`

### 9.2 `lesion_threshold.json` 不是 CLI runtime config

CLI 的 soft 推論（`gpu_infer_soft.py`）仍使用程式內常數：

```text
DEFAULT_LESION_THRESH=0.5
```

`lesion_threshold.json` 是離線推導與紀錄，不是 `gpu_infer_soft.py` runtime 讀取的檔案。

### 9.3 base 不套用 GROD τ

base 使用 Step 2 的 RF-DETR detector score；其 score scale 與 GROD 的 joint objectness 不同，因此不共用 GROD τ。

### 9.4 `case_db_swap_global` 的 symlink 行為

`build_case_db_swap_global` 會把 meta / cause text embeddings / teacher / pool 等共享檔從 source DB symlink。後續重建 pool / teacher 會把 symlink 覆蓋成真檔，這是預期行為。

### 9.5 `rawP` / `jointDistRawP` 命名

| 名稱 | 意義 |
|---|---|
| `case_db_raw` | global = raw SigLIP2；lesion = raw SigLIP2 |
| `case_db_jointDistRaw` | lesion 換成 joint semantic z；global 仍沿用 raw |
| `case_db_jointDistRawP` | lesion = joint semantic z；global = distilled raw global |
| `distilled_global_rawP` | DINO neck 蒸餾到 raw SigLIP2 global space |
