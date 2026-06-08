# 在新資料版本上重建全鏈（base / grod / grod_soft / demo）

這是**資料版本切換後的主重建流程**（canonical rebuild runbook）。`dataset_pipeline/db_pipeline`
（純資料、輕環境、無 torch）把新資料版本建到 **`data/processed/current/`**；本文件接手，在
`SDM` 環境下把該版本資料所需的全部 artifacts 建起來，服務三個推論流水線
（`base`、`grod`=`gpu_infer.py`、`grod_soft`=`gpu_infer_soft.py`）與 `demo/app_gradio.py`。

**兩層 artifacts（每次 db_pipeline 重生資料都要處理）：**

- **A. 資料衍生／凍結層**（資料 + 公開凍結模型的純函數，**無訓練**，確定性）：
  `text_anchors.pt`（步驟 1）· `case_db_raw`（步驟 4）· `dino_global`（步驟 6）。
  **資料一變，這層整個作廢、必須重建**；不放進 db_pipeline（會把純資料管線耦合到 GPU/ML 環境），
  而是 db_pipeline 跑完後在此手動跑這幾步即可。
- **B. 訓練層**（要 GPU 訓練、有超參、要實驗控制）：其餘全部
  （`rfdetr` · `joint_rfdetr` · `distilled_global_rawP` · `encoder_grod(_soft)` · `ceah_*` ·
  `disease_head` · base 的 SigLIP2 finetune + `case_db_base`）。
  **A 層重建後，B 層也要跟著重訓**（依賴關係顯式：資料變 → A 重建 → B 重訓）。

> 命名沿革：原為 `diagnosis_model/grod/RETRAIN_gpu_infer.md`（只講 grod hard 鏈的 6 個產物）；
> 現已涵蓋 base/grod/grod_soft + demo，故上移到 `diagnosis_model/` 並更名 `BUILD_PIPELINE.md`，
> 定位＝「資料版本切換後重建全部 artifacts」，不再以 gpu_infer / retrain 為框架。

## 統一輸出根目錄

所有重訓產物寫到單一根目錄 **`data/processed/current/artifacts/`**，底下分兩區：
`models/`（有學習權重的 checkpoint + 錨點）、`db/`（case 庫與特徵快取）。
`current` 是 symlink → 資料集版本目錄，所以 artifacts 跟著它訓練的那版資料走、版本對得起來。

```bash
PY=/home/lab603/anaconda3/envs/SDM/bin/python
cd /mnt/ssd/YJ/fish_disease_system
ART=data/processed/current/artifacts
mkdir -p $ART/models $ART/db
```

```
data/processed/current/artifacts/
├── models/   text_anchors.pt · rfdetr/ · joint_rfdetr/ · distilled_global_rawP/
│             encoder_grod/ · ceah_jointDistRawP/                       (grod)
│             encoder_grod_soft/ (+bank_z_soft.pt) · ceah_grod_soft/   (grod_soft)
│             disease_head/lesion_threshold.json  (病灶門檻 τ；hard+soft 共用)
│             siglip2_base_finetuned/ · encoder_base/ · ceah_base/      (base)
└── db/       case_db_raw/ · z_joint/ · dino_global/ · soft_inputs/
              case_db_jointDistRaw/ · case_db_jointDistRawP/            (grod / grod_soft 共用)
              case_db_base/                                             (base)
```

---

## 0. 新資料集路徑（db_pipeline 產出）＋ 舊→新對照

| 舊路徑 | 新路徑 | 內容 |
|---|---|---|
| `data/raw/symptoms.json` | `data/processed/current/symptoms.json` | symptom 類別空間（本版 15 類，含 `healthy_region`=0） |
| `data/detection/coco/_merged` | `data/processed/current/detection` | 偵測 view：box 全併 `ABNORMAL`、排除 healthy；image 仍帶 `global_causes_zh`/`overall`；**每 box 已內建 `symptom_category_id`** |
| `data/coco/_merged`（vl_classifier full view） | `data/processed/current/full` | 完整 symptom 類別；只服務 vl_classifier，**本鏈不需要** |

```bash
DET=data/processed/current/detection            # 偵測 view（box=ABNORMAL，含 causes/overall + 每box symptom_category_id）
SYM=data/processed/current/symptoms.json        # symptom 類別空間
```

---

## grod（`gpu_infer.py`）產物 → 由哪一步產生

> 下面步驟 1–10 是 **grod hard 鏈**（含共用的 A 層 1/4/6 + B 層）。
> **grod_soft** 接在其後（見「grod_soft 分支」），**base** 為獨立分支（見「base 分支」）。

| `gpu_infer.py` 參數 | 產生步驟 |
|---|---|
| `--anchors`  `$ART/models/text_anchors.pt` | **1** |
| `--joint_ckpt`  `$ART/models/joint_rfdetr/checkpoint_best_regular.pth` | **3**（依賴 **2**） |
| `--global_sd`  `$ART/models/distilled_global_rawP/global_embed_state_dict.pt` | **7**（依賴 **6**） |
| `--enc_ckpt`  `$ART/models/encoder_grod/best_encoder.pt` | **9**（依賴 **8**） |
| `--ceah_ckpt`  `$ART/models/ceah_jointDistRawP/best_ceah.pt` | **10** |
| `--case_db_dir`  `$ART/db/case_db_jointDistRawP` | **8**（依賴 4＋5＋7） |
| `--det_thresh`  固定常數 τ（預設 0.5） | **S5**（`compute_lesion_threshold.py`） |

> **病灶門檻 = 固定常數 τ（預設 0.5）**：hard `gpu_infer.py` keep `obj > τ`、沒框就 abstain；soft `gpu_infer_soft.py` 全留軟權重、`max_w < τ` 才 abstain。τ 由 **S5** `compute_lesion_threshold.py` 推導（train F1-最佳，存 `lesion_threshold.json`）。完整理由（為何用常數而非學習門檻、disease head 為何不進推論）見 [grod/LESION_GATE.md](grod/LESION_GATE.md)。

相依 DAG（→ 表示「被…依賴」）：

```
1 anchors ─────────────┐
2 base detector ──► 3 joint(box/obj/sem z) ──► 5 z_joint ─┐
                  └────────────────────► 6 dino_global ──► 7 distilled global ─┐
4 case_db_raw ─────────────────────────────────────┴─────────────┐             │
                                          (lesion z) 5 ───► 8a jointDistRaw ───► 8b jointDistRawP ◄┘
                                                                  8c pool + 8d teacher ─► 9 encoder_grod
                                                                                    └──► 10 ceah_jointDistRawP
```

---

## 步驟（依序執行）

### 1. 文字錨點 `text_anchors.pt`  →  `--anchors`
凍結 raw SigLIP2 對每個 symptom 類別的 caption 取平均錨點。**必須用新 `SYM`**（類別數改了）。

```bash
$PY -m diagnosis_model.grod.build_text_anchors \
    --symptoms $SYM \
    --out $ART/models/text_anchors.pt
```

### 2. Base RF-DETR 偵測器（joint 的 `--pretrain_weights`）
> `train_rfdetr.py` 直接執行（非 -m）；用絕對 `--output_dir` 把權重寫進 `$ART`。~數小時 / 100 epoch。

```bash
cd diagnosis_model/detection
$PY train_rfdetr.py \
    --dataset_dir /mnt/ssd/YJ/fish_disease_system/$DET \
    --output_dir  /mnt/ssd/YJ/fish_disease_system/$ART/models/rfdetr
cd ../..
```

### 3. Joint 偵測+語意訓練 `joint_rfdetr`  →  `--joint_ckpt`
凍 backbone，訓 decoder 上的 **region heads（box / objectness / semantic z）**。`semantic_loss_coef=2.0`、`batch_size=16` 為 sweep 最佳點。~3.5 h / 30 epoch。

```bash
$PY -m diagnosis_model.grod.train_joint \
    --dataset_dir $DET \
    --pretrain_weights $ART/models/rfdetr/checkpoint_best_total.pth \
    --anchors $ART/models/text_anchors.pt \
    --output_dir $ART/models/joint_rfdetr \
    --epochs 30 --semantic_loss_coef 2.0
```

> ⚠️ **semantic head 要用 `best_regular`，不要用 `best_total`**：`best_total` 實測 == `best_ema`、是用
> **detection mAP** 選的（EMA best 落在 epoch 0），其 semantic head 是 EMA 平均 + 偵測早期狀態、較弱；
> `best_regular` 的 semantic head 訓得最完整。**步驟 5（建 bank z）與 `gpu_infer`（query z）都用
> `best_regular`**。

### 4. `case_db_raw`（raw SigLIP2 case 庫，下游所有 swap 的底）→ `$ART/db/case_db_raw`

```bash
$PY -m diagnosis_model.cause_inference.preprocessing.build_case_database \
    --coco_train  $DET/train/_annotations.coco.json --image_root_train  $DET/train \
    --coco_valid  $DET/valid/_annotations.coco.json --image_root_valid  $DET/valid \
    --vlm_global google/siglip2-base-patch16-224 \
    --vlm_lesion google/siglip2-base-patch16-224 --raw_lesion \
    --output_dir $ART/db/case_db_raw \
    --chunk_size 64 --img_batch_size 64 --text_batch_size 256
```

### 5. 抽 joint 的 `pred_semantic z`（`z_joint`）→ `$ART/db/z_joint`
跑 joint 模型，IoU-match 每個 case_db GT lesion box → query，dump 訓練後的 768-d z。

```bash
$PY -m diagnosis_model.grod.extract_z_joint \
    --case_db_dir $ART/db/case_db_raw \
    --joint_ckpt $ART/models/joint_rfdetr/checkpoint_best_regular.pth \
    --anchors $ART/models/text_anchors.pt \
    --image_root $DET \
    --output_dir $ART/db/z_joint --splits train valid
```

### 6. 抽 RF-DETR backbone global（1536-d）→ `$ART/db/dino_global`
從最原始 DINOv2 encoder 抽 4 尺度各 pool+L2 後 concat = 1536-d，當蒸餾來源（`distill_global_mlp` 的
`d_in=1536`）。每個 split 各跑一次（`--image_root` 是 split 子目錄）。

```bash
for SP in train valid; do
  $PY -m diagnosis_model.grod.extract_dino_global \
      --case_db_dir $ART/db/case_db_raw \
      --split $SP \
      --det_ckpt $ART/models/rfdetr/checkpoint_best_total.pth \
      --image_root $DET/$SP \
      --output_dir $ART/db/dino_global
done
```

### 7. 蒸餾 global MLP `distilled_global_rawP`  →  `--global_sd`
DINO neck（1536-d）蒸餾到 raw SigLIP2 whole-image global（target=`case_db_raw`）。輸出 `global_embed_state_dict.pt`。

```bash
$PY -m diagnosis_model.grod.distill_global_mlp \
    --dino_dir   $ART/db/dino_global \
    --target_db  $ART/db/case_db_raw \
    --out_dir    $ART/models/distilled_global_rawP
```

### 8. 組 `case_db_jointDistRawP`  →  `--case_db_dir`
單變數 swap：lesion=GROD z、global=蒸餾 raw global，其餘沿用 `case_db_raw`。

```bash
# 8a. lesion_embs <- joint z  → case_db_jointDistRaw
$PY -m diagnosis_model.grod.rebuild_case_db \
    --src_case_db $ART/db/case_db_raw \
    --hs_dir $ART/db/z_joint --from_joint \
    --out_case_db $ART/db/case_db_jointDistRaw

# 8b. global_emb <- 蒸餾 global  → case_db_jointDistRawP
$PY -m diagnosis_model.grod.build_case_db_swap_global \
    --src_db $ART/db/case_db_jointDistRaw \
    --global_dir $ART/models/distilled_global_rawP \
    --global_prefix distilled_global \
    --dst_db $ART/db/case_db_jointDistRawP

# 8c. candidate pool（~3 min）
$PY -m diagnosis_model.cause_inference.preprocessing.build_train_candidate_pool \
    --case_db_dir $ART/db/case_db_jointDistRawP \
    --output_path $ART/db/case_db_jointDistRawP/train_candidate_pool.pt \
    --top_k_cases 20 --alpha_global 0.25 --beta_lesion 0.75 \
    --lesion_match max_mean --semantic_threshold 0.95

# 8d. teacher score table（~3 min）
$PY -m diagnosis_model.cause_inference.preprocessing.build_teacher_table \
    --case_db_dir $ART/db/case_db_jointDistRawP \
    --output_path $ART/db/case_db_jointDistRawP/teacher_train_train.pt \
    --alpha_global 0.25 --beta_lesion 0.75 --lesion_match max_mean
```

> `build_case_db_swap_global` 預設把共享檔（meta/cause_text_embs/teacher/pool）從 `src_db` symlink；
> 8c/8d 之後會把 pool/teacher 寫成 `case_db_jointDistRawP` 內的真檔、覆蓋 symlink，正常。

### 9. Aggregator（DeepSets）`encoder_grod`  →  `--enc_ckpt`
listwise-KL 蒸餾 + 0.5×SupCon InfoNCE。~10 min。

```bash
$PY -m diagnosis_model.cause_inference.train_case_encoder \
    --case_db_dir $ART/db/case_db_jointDistRawP \
    --teacher_path $ART/db/case_db_jointDistRawP/teacher_train_train.pt \
    --train_pool_path $ART/db/case_db_jointDistRawP/train_candidate_pool.pt \
    --output_dir $ART/models/encoder_grod \
    --encoder_type deepsets --batch_size 256 --epochs 50 \
    --temp_target 0.1 --temp_pred 0.1 \
    --use_infonce --infonce_weight 0.5 --infonce_temp 0.07
```

### 10. CEAH `ceah_jointDistRawP`  →  `--ceah_ckpt`
canonical：`softmax` × `multiplicative` × `lambda_sparsity 0.0` × `text_dropout 0.5`。

```bash
$PY -m diagnosis_model.cause_inference.train_ceah \
    --case_db_dir $ART/db/case_db_jointDistRawP \
    --train_pool_path $ART/db/case_db_jointDistRawP/train_candidate_pool.pt \
    --output_dir $ART/models/ceah_jointDistRawP \
    --attribution_mode softmax --scoring_mode multiplicative \
    --lambda_sparsity 0.0 --text_dropout 0.5
```

---

## 驗證

`gpu_infer.py` 的 default 已指向 `$ART`，跑推論只要帶 `--image`：

```bash
$PY -m diagnosis_model.grod.gpu_infer \
    --image data/processed/current/full/test/<某張>.jpg --verify
```

預期：印 `[bank] z=(Nt,768) ...`、`[verify] ... all compute tensors on CUDA ✓`，再列 Top causes。
可選 faithfulness gate（招牌 `no_lesion` 須為正）：

```bash
$PY -m diagnosis_model.cause_inference.faithfulness_eval \
    --case_db_dir $ART/db/case_db_jointDistRawP \
    --ceah_ckpt $ART/models/ceah_jointDistRawP/best_ceah.pt \
    --output_dir $ART/models/ceah_jointDistRawP_faithfulness \
    --attribution_mode softmax --scoring_mode multiplicative
```

---

## grod_soft 分支（`gpu_infer_soft.py`）

soft 版把 hard 的硬選取（`obj>τ` 選 lesion）換成**連續權重**（`w=sigmoid(objectness)`，
300 query 全留、用 w 當 soft mask）；abstain＝`max_w < τ`。它**疊在 hard 鏈之上**——直接共用
hard **步驟 1/3/7/8** 的 `text_anchors` / `joint_rfdetr` / `distilled_global_rawP` /
`case_db_jointDistRawP`（含 8c pool + 8d teacher），**不需要** hard 步驟 9/10（soft 自己訓
`encoder_grod_soft` / `ceah_grod_soft`）。

> **門檻機制 = 固定常數 τ（預設 0.5，hard + soft 共用）**：hard keep `obj > τ`、沒框 abstain；
> soft 全留軟權重、`max_w < τ` 才 abstain。τ 由 **S5** `compute_lesion_threshold.py` 推導。
> 為何用常數而非學習門檻、disease head 為何不進推論 → [grod/LESION_GATE.md](grod/LESION_GATE.md)。
> **base 用自己的 `rfdetr` 偵測器門檻**（step 2，分數尺度不同、不套此 τ）。

```bash
# 接續 hard 鏈的環境變數（$PY / $ART / $DET）

# S1. soft 輸入（跑 joint 出 300 query 的 g/z/objectness）→ $ART/db/soft_inputs
$PY -m diagnosis_model.grod.extract_soft_inputs \
    --joint_ckpt $ART/models/joint_rfdetr/checkpoint_best_regular.pth \
    --global_sd  $ART/models/distilled_global_rawP/global_embed_state_dict.pt \
    --anchors    $ART/models/text_anchors.pt \
    --case_db_dir $ART/db/case_db_jointDistRawP \
    --img_root   $DET \
    --out_dir    $ART/db/soft_inputs

# S2. soft Aggregator → $ART/models/encoder_grod_soft  (→ gpu_infer_soft --enc_ckpt)
$PY -m diagnosis_model.grod.train_case_encoder_soft \
    --soft_dir   $ART/db/soft_inputs \
    --case_db_dir $ART/db/case_db_jointDistRawP \
    --teacher_path $ART/db/case_db_jointDistRawP/teacher_train_train.pt \
    --output_dir $ART/models/encoder_grod_soft

# S3. soft bank → bank_z_soft.pt  (→ gpu_infer_soft --bank_path)
$PY -m diagnosis_model.grod.build_soft_bank \
    --soft_dir    $ART/db/soft_inputs \
    --encoder_ckpt $ART/models/encoder_grod_soft/best_encoder.pt \
    --out         $ART/models/encoder_grod_soft/bank_z_soft.pt

# S4. soft CEAH → $ART/models/ceah_grod_soft  (→ gpu_infer_soft --ceah_ckpt)
$PY -m diagnosis_model.grod.train_ceah_soft \
    --case_db_dir $ART/db/case_db_jointDistRawP \
    --train_pool_path $ART/db/case_db_jointDistRawP/train_candidate_pool.pt \
    --soft_dir    $ART/db/soft_inputs \
    --encoder_ckpt $ART/models/encoder_grod_soft/best_encoder.pt \
    --bank_path   $ART/models/encoder_grod_soft/bank_z_soft.pt \
    --output_dir  $ART/models/ceah_grod_soft \
    --top_k_lesions 32

# S5. 病灶門檻 τ（hard + soft 共用，固定常數）→ lesion_threshold.json
#     per-query GT-IoU 標籤 → 算 train F1-最佳的全域 objectness 門檻（τ*≈0.5）。
#     gpu_infer.py / gpu_infer_soft.py 的 DEFAULT_LESION_THRESH 即此值。
#     （disease head 為何不用、完整 ablation → grod/LESION_GATE.md；其訓練腳本仍在但不進推論）
$PY -m diagnosis_model.grod.extract_disease_perquery        # → $ART/db/disease_perquery/{train,val}.pt
$PY -m diagnosis_model.grod.compute_lesion_threshold        # → $ART/models/disease_head/lesion_threshold.json
```

驗證（`gpu_infer_soft.py` default 已全指向 `$ART`，帶 `--image` 即可）：

```bash
$PY -m diagnosis_model.grod.gpu_infer_soft \
    --image data/processed/current/full/test/<某張>.jpg --verify
```

---

## base 分支（GROD baseline；`demo/app_gradio.py` 的 base 模式用）

常規分離式對照組：**不**用 joint 的 DETR-routed z，而是一顆「最常規微調的 SigLIP2」編病灶
crop、global 用 raw SigLIP2。**獨立分支**——不從 `case_db_raw` swap，自己 build case_db_base
（其 lesion 來自 base 微調的 SigLIP2，故歸 B 訓練層）。grod/grod_soft 的步驟 1–10 / S1–S5 與此無關。

```bash
# B0. 標準微調 SigLIP2（一傷口一 caption，無 multipos/fusion）→ models/siglip2_base_finetuned
#     從 full view（15 類 caption）訓；output_dir 給絕對路徑。~6 min。
$PY diagnosis_model/vl_classifier/train.py \
    --data_root data/processed/current/full \
    --symptoms_file $SYM \
    --output_dir $PWD/$ART/models/siglip2_base_finetuned

# B1. case_db_base（global=raw SigLIP2、lesion=base 微調、--raw_lesion）→ db/case_db_base
$PY -m diagnosis_model.cause_inference.preprocessing.build_case_database \
    --coco_train $DET/train/_annotations.coco.json --image_root_train $DET/train \
    --coco_valid $DET/valid/_annotations.coco.json --image_root_valid $DET/valid \
    --vlm_global google/siglip2-base-patch16-224 \
    --vlm_lesion $ART/models/siglip2_base_finetuned --raw_lesion \
    --output_dir $ART/db/case_db_base --chunk_size 64 --img_batch_size 64 --text_batch_size 256

# B2. pool + B3. teacher（同 8c/8d，但 case_db_dir 指 case_db_base）
$PY -m diagnosis_model.cause_inference.preprocessing.build_train_candidate_pool \
    --case_db_dir $ART/db/case_db_base --output_path $ART/db/case_db_base/train_candidate_pool.pt \
    --top_k_cases 20 --alpha_global 0.25 --beta_lesion 0.75 --lesion_match max_mean --semantic_threshold 0.95
$PY -m diagnosis_model.cause_inference.preprocessing.build_teacher_table \
    --case_db_dir $ART/db/case_db_base --output_path $ART/db/case_db_base/teacher_train_train.pt \
    --alpha_global 0.25 --beta_lesion 0.75 --lesion_match max_mean

# B4. encoder_base（DeepSets，同步驟 9 超參）→ models/encoder_base
$PY -m diagnosis_model.cause_inference.train_case_encoder \
    --case_db_dir $ART/db/case_db_base --teacher_path $ART/db/case_db_base/teacher_train_train.pt \
    --train_pool_path $ART/db/case_db_base/train_candidate_pool.pt \
    --output_dir $ART/models/encoder_base \
    --encoder_type deepsets --batch_size 256 --epochs 50 --temp_target 0.1 --temp_pred 0.1 \
    --use_infonce --infonce_weight 0.5 --infonce_temp 0.07

# B5. ceah_base（canonical，同步驟 10）→ models/ceah_base
$PY -m diagnosis_model.cause_inference.train_ceah \
    --case_db_dir $ART/db/case_db_base --train_pool_path $ART/db/case_db_base/train_candidate_pool.pt \
    --output_dir $ART/models/ceah_base \
    --attribution_mode softmax --scoring_mode multiplicative --lambda_sparsity 0.0 --text_dropout 0.5
```

demo 載 base 時 global 用 raw SigLIP2、偵測用步驟 2 的 `rfdetr`（非 joint）、病灶分類沿用
`text_anchors.pt`。三模式（base/grod/grod_soft）建完後 `demo/app_gradio.py` 即可下拉切換。
