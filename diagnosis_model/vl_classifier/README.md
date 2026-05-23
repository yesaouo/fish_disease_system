# vl_classifier — SigLIP2 病灶分類 / 描述檢索

兩個 SigLIP2 finetune 由同一支 [train.py](train.py) 切 flag 產生：

| 產出 checkpoint | 訓練 flag | 用途 |
|---|---|---|
| `outputs/siglip2_base_patch16_224_overall_multipos_zh/` (VLM-Global) | `--target overall --multipos` | 整圖 ↔ overall 描述檢索 |
| `outputs/siglip2_base_patch16_224_multipos_fusion_en_zh/` (VLM-Lesion) | `--multipos --fusion --freeze_text_encoder` | 病灶 crop ↔ 症狀 caption；下游 `cause_inference` CEAH 依賴 |

評估用 [eval.py](eval.py)（病灶分類在 `data/coco/_merged/test` 上對 GT bbox 做 caption-bank 分類，輸出 `classification_report`）。

---

## VLM-Lesion 強化方法記錄

兩個正交的優化目標，方法與結論分開記。**Production checkpoint 一律不被下列實驗覆蓋**（各自輸出到獨立 `--output_dir`）。

### 目標 A — 病灶分類準確率（macro f1）

**現況診斷**：base VLM-Lesion 在 test（n=3785, 19 類）weighted accuracy = **0.89**、但 macro f1 只有 **0.53**。差距全來自類別極度不平衡——大類（0/5/8/12/15，n>2900）已飽和於 0.85–0.99，中間層（4/6/7/9/10/16，n=14–65）被餓死在 0.34–0.64，另有 **4 類（2/3/13/17）訓練 n≤4 根本學不動、永遠 ~0**。

| # | 方法 | flag | 結果（macro f1 / acc） | 採用？ |
|---|---|---|---|---|
| A1 | base（對照） | `--multipos --fusion --freeze_text_encoder` | 0.53 / 0.89 | production |
| A2 | Class-balanced loss（effective-number 權重,β=0.999,加權 i2t per-sample 項） | `--class_balanced_loss --cb_beta 0.999` | 0.59\* / 0.89 | **暫不使用** |
| A3 | Balanced sampler（反頻率 WeightedRandomSampler 過採樣） | `--balanced_sampler` | 0.56 / 0.88 | **暫不使用** |
| A4 | A2 + A3 同開（β=0.999） | `--class_balanced_loss --cb_beta 0.999 --balanced_sampler` | 0.55 / 0.87 | **暫不使用** |
| A5 | A4 但 β=0.99 / 0.9 | `--cb_beta 0.99` / `0.9` | 0.53 / 0.57 | **暫不使用** |

\* A2 的 0.59 被噪音類灌水（類 17 n=**1** 從 0→0.67 只是猜對 1 個樣本）。**去掉 n≤5 噪音類後**，13 個可學類的 macro f1：base 0.703 → A2 0.727 / A3 0.732 / A4 0.738。

**結論：暫不使用整條 loss/sampler 槓桿。** 去噪後真實增益僅 **+0.02~0.03**，且**所有變體都讓 6 個大類略掉（0.89→0.88）**；ablation 顯示 loss-only ≈ sampler-only ≈ +0.025、兩者疊加無加成。macro 的結構性上限被 4 個 n≤4 的死類壓在 ~0.5，**任何損失加權都救不了**——瓶頸是資料量與 taxonomy，不是損失函數。

#### 架構槓桿

| # | 方法 | flag | 結果（macro / denoised n≥14 / acc） | 採用？ |
|---|---|---|---|---|
| A6 | **FiLM / per-channel gate**：把單一純量 `gate` 換成 input-conditioned 逐通道 gate `g=σ(gate_net([local;global]))`，init g≈0.1（與 scalar 同起點）。見 [common.py](common.py) `LocalGlobalFusionWrapper(gate_mode="film")` | `--fusion_gate_mode film` | 0.51 / 0.668 / 0.88 | **暫不使用（negative result）** |
| A7 | **xattn / cross-attention**：lesion 特徵當 query,對整尾魚的 **patch tokens**（非 pooled-global）做 cross-attention（8 head），attended context 走同一條 `fusion_linear([local;ctx])` + scalar gate 殘差合回。見 [common.py](common.py) `LocalGlobalFusionWrapper(gate_mode="xattn")` | `--fusion_gate_mode xattn` | 0.55 / — / 0.888 | **暫不使用（中性結果）** |

**結論：FiLM 是負結果，且否證了「fusion 容量不足」的假設。** 訓練中 effective_ratio 從 base 的 0.06 漲到 0.15（~2.3×,fused_norm 0.63→1.16,mean gate 維持 ~0.1）——**機制按設計成功讓模型用更多全圖 context,但分類反而退步**（denoised macro 0.703→0.668,類 7 崩 −0.29、類 16 −0.10、類 9/10 −0.07）。代表**全圖 context 對病灶分類比較像干擾而非線索**,拉進更多會稀釋 crop 本身的判別訊號。base 的 scalar-gate（只用 6% context）已接近此資料集上限。`--fusion_gate_mode film` 程式碼保留當 ablation（向後相容,不影響 production scalar 路徑;eval 從 `wrapper_state.pt` 的 `gate_net.*` key 自動偵測）。

**結論：xattn 是中性結果——比 FiLM 好（FiLM 退步、xattn 持平），但對分類與 faithfulness 兩項都無增益。** 把 pooled-global 換成「lesion query 對整張 patch grid cross-attention」這個更豐富的空間 global 表徵後：分類 acc 0.888 vs scalar 0.894（−0.6pp）、macro-F1 0.55 vs 0.53（+0.02,皆 noise 級）；下游 CEAH faithfulness lesion-type no_lesion drop **+0.0383 ≈ production scalar +0.040**（差 −0.002,通過且為正,N=2 bucket 0.050≈production 0.051——完整 faithfulness 鏈見下方目標 B）。訓後 scalar gate 停在 **0.0989（≈init 0.1）**——模型沒學會比 pooled-global 更倚重 patch context。**證實 pooled global 已足夠,fusion 的 faithfulness 貢獻不依賴空間細節。** 程式碼保留當 ablation（不影響 production scalar;`common.py`/`train.py`/`eval.py`/`build_case_database.py` 從 `wrapper_state.pt` 的 `cross_attn.*` key 自動偵測 `gate_mode="xattn"`）。

**三條路（A 不平衡槓桿 + A6 FiLM + A7 xattn）一致指向：瓶頸不是訊號分配也不是 fusion 容量/空間表徵,是資料量與 taxonomy。**

**尚未嘗試的架構槓桿**（FiLM 負結果、xattn 中性後優先序已降低，塞更多 context machinery 大概率同樣無解）：
- **多尺度 crop**（不同 padding 比例的 peri-lesion 區）。

**真正能動天花板（建議優先）**：
- **資料 / taxonomy**：補標 n 小的類；或將 n≤1 的極罕見類（2/3/13/17）合併進母類或移出分類體系，直接解開 macro 上限。

### 目標 B — 下游 CEAH attribution faithfulness

VLM-Lesion 對 cause_inference Phase 2 的 faithfulness 必要（對 retrieval 無貢獻）。這些方法已記在 [`cause_inference/training.txt`](../cause_inference/training.txt) `[0-clsc]` / `[0-lscft]` 與 README，此處只列索引：

| 方法 | flag | 結論 |
|---|---|---|
| CLSC（Cross-Lesion Supervised Contrastive,同圖 sibling lesion 當 hard negative） | `--cross_lesion --cross_alpha 1.5 --cross_tau 0.1` | **採用為方法貢獻**：faithfulness lesion-type +17%、mixed +67% |
| CLSC α=3.0（過大權重） / grouped_sampler（100% firing） | `--cross_alpha 3.0` / `--grouped_sampler` | **暫不使用**：排擠主任務 / 改 batch 組成致主任務 underfit |
| LSCFT（counterfactual augmentation,中心 mask 模擬病灶移除） | `--lscft ...` | **暫不使用（negative result）**：訓練訊號無法 transfer 到 CEAH attribution faithfulness |
| xattn（A7,lesion query 對整尾魚 patch grid cross-attention） | `--fusion_gate_mode xattn` | **暫不使用（中性結果）**：跑完整鏈（case_db_xattn → CEAH → faithfulness_eval),lesion-type drop +0.0383 ≈ production scalar +0.040,無增益 → pooled global 對 faithfulness 已足夠 |

---

## 復現指令（目標 A 實驗）

```bash
cd diagnosis_model/vl_classifier
PY=/home/lab603/anaconda3/envs/SDM/bin/python

# A2 loss-only / A3 sampler-only / A4 both（各 ~50min,輸出獨立目錄）
$PY train.py --multipos --fusion --freeze_text_encoder --class_balanced_loss --cb_beta 0.999 \
  --output_dir outputs/siglip2_base_patch16_224_multipos_fusion_cbloss_en_zh
$PY train.py --multipos --fusion --freeze_text_encoder --balanced_sampler \
  --output_dir outputs/siglip2_base_patch16_224_multipos_fusion_cbsamp_en_zh
$PY train.py --multipos --fusion --freeze_text_encoder --class_balanced_loss --cb_beta 0.999 --balanced_sampler \
  --output_dir outputs/siglip2_base_patch16_224_multipos_fusion_cb_en_zh

# 一次比全部 macro f1（注意資料樹是 data/coco/_merged）
$PY eval.py --target lesion --fusion \
  --data_dir ../../data/coco/_merged/test --symptoms_json ../../data/raw/symptoms.json \
  --model base=outputs/siglip2_base_patch16_224_multipos_fusion_en_zh \
  --model lossonly=outputs/siglip2_base_patch16_224_multipos_fusion_cbloss_en_zh \
  --model samponly=outputs/siglip2_base_patch16_224_multipos_fusion_cbsamp_en_zh \
  --model both=outputs/siglip2_base_patch16_224_multipos_fusion_cb_en_zh \
  --output_dir outputs/cb_sweep_eval
```

## 復現指令（A7 xattn — classification + 完整 faithfulness 鏈）

```bash
PY=/home/lab603/anaconda3/envs/SDM/bin/python
cd /mnt/ssd/YJ/fish_disease_system   # faithfulness 鏈用 -m 從 repo root 跑

# 1) 訓練 xattn VLM-Lesion（從 vl_classifier dir,~16 min,8 epochs）
cd diagnosis_model/vl_classifier && $PY train.py \
  --multipos --fusion --freeze_text_encoder --fusion_gate_mode xattn \
  --output_dir outputs/siglip2_base_patch16_224_multipos_fusion_xattn_en_zh && cd -

# 2) lesion classification（xattn vs production scalar vs raw）
cd diagnosis_model/vl_classifier && $PY eval.py \
  --data_dir ../../data/coco/_merged/test \
  --symptoms_json ../../data/annotation/backup/20260207/fish_disease/symptoms.json \
  --output_dir outputs/eval_xattn_compare \
  --model zero=google/siglip2-base-patch16-224 \
  --model scalar=outputs/siglip2_base_patch16_224_multipos_fusion_en_zh \
  --model xattn=outputs/siglip2_base_patch16_224_multipos_fusion_xattn_en_zh && cd -

# 3) faithfulness 鏈：case_db → candidate_pool → CEAH → faithfulness_eval（gate_mode 自動偵測）
$PY -m diagnosis_model.cause_inference.preprocessing.build_case_database \
  --coco_train data/detection/coco/_merged/train/_annotations.coco.json --image_root_train data/detection/coco/_merged/train \
  --coco_valid data/detection/coco/_merged/valid/_annotations.coco.json --image_root_valid data/detection/coco/_merged/valid \
  --vlm_global diagnosis_model/vl_classifier/outputs/siglip2_base_patch16_224_overall_multipos_zh \
  --vlm_lesion diagnosis_model/vl_classifier/outputs/siglip2_base_patch16_224_multipos_fusion_xattn_en_zh \
  --output_dir diagnosis_model/cause_inference/outputs/case_db_xattn \
  --chunk_size 64 --img_batch_size 64 --text_batch_size 256
$PY -m diagnosis_model.cause_inference.preprocessing.build_train_candidate_pool \
  --case_db_dir diagnosis_model/cause_inference/outputs/case_db_xattn \
  --output_path diagnosis_model/cause_inference/outputs/case_db_xattn/train_candidate_pool.pt \
  --top_k_cases 20 --alpha_global 0.25 --beta_lesion 0.75 --lesion_match hungarian --semantic_threshold 0.95
$PY -m diagnosis_model.cause_inference.train_ceah \
  --case_db_dir diagnosis_model/cause_inference/outputs/case_db_xattn \
  --train_pool_path diagnosis_model/cause_inference/outputs/case_db_xattn/train_candidate_pool.pt \
  --output_dir diagnosis_model/cause_inference/outputs/ceah_xattn \
  --attribution_mode softmax --scoring_mode multiplicative --epochs 20 --batch_size 32 --lr 1e-4 \
  --lambda_sparsity 0.0 --text_dropout 0.5 --warmup_steps 200 --eval_every 2 --eval_max_queries 300
$PY -m diagnosis_model.cause_inference.faithfulness_eval \
  --case_db_dir diagnosis_model/cause_inference/outputs/case_db_xattn \
  --ceah_ckpt diagnosis_model/cause_inference/outputs/ceah_xattn/best_ceah.pt \
  --output_dir diagnosis_model/cause_inference/outputs/ceah_xattn_faithfulness \
  --attribution_mode softmax --scoring_mode multiplicative --gamma 0.5 --max_queries -1
```

實測結果（2026-05-21）：classification acc 0.888 / macro-F1 0.55（vs scalar 0.894 / 0.53）；CEAH lesion-type faithfulness drop **+0.0383 ≈ production +0.040** → 中性,pooled global 已足夠。
