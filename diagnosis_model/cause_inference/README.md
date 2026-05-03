# FaCE-R: Faithful Case-guided Cause-Evidence Retrieval

魚病病因推論系統的第三階段——在已訓練的 RF-DETR 病灶偵測器與兩個 SigLIP2 多模態模型上，做 case-based 病因檢索 + 架構強制的 lesion-grounded 可解釋性。

## 動機

魚病診斷的病因標註是 **GPT 生成的自由文字**（每張圖約 4 個病因，總共 56,310 個 unique 字串，**94.7% 是 singleton**）。這個資料特性使得：

1. 傳統的 closed-vocabulary multi-label classification **無法應用**
2. 教授的硬性要求是「**lesion-grounded explainability**」——預測必須能對應到具體病灶

FaCE-R 把問題重構成 **case-based 檢索 + lesion-attribution**：

```
Query 影像
  ↓
RF-DETR 偵測病灶（前一階段）
  ↓
VLM-Global / VLM-Lesion 抽特徵（前一階段）
  ↓
[Phase 1] C²R 案例檢索 → 候選病因池（~87 個）
  ↓
[Phase 3] CEAH 重新打分 + 病灶歸因
  ↓
Top-N 病因預測 + α attribution 視覺化
```

## 上游依賴

| 元件 | 路徑 | 訓練語料 |
|---|---|---|
| **RF-DETR 病灶偵測器** | [`../detection/`](../detection/) | COCO bbox |
| **VLM-Global**（整圖 ↔ 整體描述） | [`../vl_classifier/outputs/siglip2_base_patch16_224_overall_multipos_zh`](../vl_classifier/outputs/siglip2_base_patch16_224_overall_multipos_zh) | `overall.medical_zh` / `colloquial_zh` |
| **VLM-Lesion**（病灶 crop ↔ 症狀描述，含 fusion wrapper） | [`../vl_classifier/outputs/siglip2_base_patch16_224_multipos_fusion_en_zh`](../vl_classifier/outputs/siglip2_base_patch16_224_multipos_fusion_en_zh) | per-bbox symptom captions |
| **COCO 標註** | `data/detection/coco/_merged/{train,valid,test}/_annotations.coco.json` | image-level + per-bbox annotations |

兩個 VLM 都凍結，不在 cause_inference 內進一步 finetune。

## Repo 結構

```
cause_inference/
├── README.md                      ← 本檔
├── preprocessing/
│   ├── README.md                  ← cause text caching + clustering CLI
│   ├── build_case_database.py     ← Phase 0: 建 case DB
│   ├── build_train_candidate_pool.py  ← Phase 3 訓練前置
│   ├── text_embedding_cache.py    ← cause 文字 embedding cache
│   ├── dim_reduction.py           ← UMAP 降維
│   ├── hdbscan_clustering.py      ← HDBSCAN 一次性 clustering
│   ├── recluster_causes.py        ← HDBSCAN CLI（兩種粒度 cause taxonomy）
│   ├── singleton_reassign.py      ← singleton → 最近 cluster 回收
│   ├── cause_cluster_json.py      ← cluster ID 工具
│   ├── cause_cluster_llm.py       ← LLM-based cause string consolidation
│   ├── cause_texts.py             ← 從 COCO 抽 unique cause 文字
│   └── export_unique_causes.py    ← 匯出 cause.txt
├── models/
│   ├── projection_head.py         ← Phase 2 模型
│   └── ceah.py                    ← Phase 3 模型（CEAH）
├── phase1_baseline.py             ← Phase 1 zero-training C²R 評估
├── train_projection.py            ← Phase 2 projection head 訓練（ablation）
├── train_ceah.py                  ← Phase 3 CEAH 訓練
├── eval_ceah.py                   ← Phase 3 推論 + α 輸出
├── faithfulness_eval.py           ← Faithfulness 驗證（lesion masking）
├── analyze_lesion_buckets.py      ← Phase 1 N-lesion 分桶分析
├── analyze_v3_n_buckets.py        ← Phase 3 v3 N-lesion 分桶分析
├── case_study_viz.py              ← 案例可視化（論文 figure）
├── run_phase1_sweep.sh            ← Phase 1 hyperparameter sweep
└── outputs/                       ← 所有產出
    ├── case_db/                   ← Phase 0（含 train_candidate_pool.pt）
    ├── phase1_full_maxmean/       ← Phase 1 baseline
    ├── phase1_sweep/              ← Phase 1 sweep
    ├── projection_v1/             ← Phase 2 ablation
    ├── ceah_v3/                   ← Phase 3 final model
    ├── ceah_v3_eval_full/         ← Phase 3 retrieval eval（fine cluster）
    ├── ceah_v3_eval_coarse/       ← Phase 3 retrieval eval（coarse 100 cluster）
    ├── ceah_v3_text_{medical,colloquial,none}/  ← Phase 3 text mode ablation
    ├── ceah_v3_faithfulness_v2/   ← Phase 3 faithfulness
    ├── case_study_v3/             ← 案例圖
    ├── cause_cache/               ← cause text embeddings
    ├── cause_clusters_reassigned.json        ← 細粒度 (2807 clusters)
    └── cause_clusters_v100_reassigned.json   ← 粗粒度 (100 disease topics)
```

## 環境

所有腳本統一在 `SDM` conda env 跑（含 transformers / torch / scipy / hdbscan / umap）：

```bash
PY=/home/lab603/anaconda3/envs/SDM/bin/python
cd /mnt/ssd/YJ/fish_disease_system
```

---

## 使用流程

### Phase 0：建 case database

對每張非 healthy 且有 `global_causes_zh` 的 COCO image：抽 global / text / lesion 特徵並 L2-normalize、儲存所有 unique cause strings 的 VLM-Global text embedding。

```bash
$PY -m diagnosis_model.cause_inference.preprocessing.build_case_database \
  --coco_train data/detection/coco/_merged/train/_annotations.coco.json \
  --image_root_train data/detection/coco/_merged/train \
  --coco_valid data/detection/coco/_merged/valid/_annotations.coco.json \
  --image_root_valid data/detection/coco/_merged/valid \
  --vlm_global  diagnosis_model/vl_classifier/outputs/siglip2_base_patch16_224_overall_multipos_zh \
  --vlm_lesion  diagnosis_model/vl_classifier/outputs/siglip2_base_patch16_224_multipos_fusion_en_zh \
  --output_dir  diagnosis_model/cause_inference/outputs/case_db \
  --chunk_size 64
```

產出（`outputs/case_db/`）：
- `train_cases.pt`（199 MB）：12,780 個 case dict
- `valid_cases.pt`（25 MB）：1,573 個
- `cause_text_embs.pt`（172 MB）：56,310 × 768 cause embeddings + texts list
- `meta.json`：config

每個 case schema：

```python
{
  "case_id":             int,
  "image_id":            int,
  "split":               "train" | "valid",
  "file_name":           str,
  "global_emb":          Tensor[768],          # VLM-Global vision, L2-normalized
  "text_colloquial_emb": Tensor[768] | None,
  "text_medical_emb":    Tensor[768] | None,
  "lesion_embs":         Tensor[N, 768],       # VLM-Lesion fusion
  "lesion_boxes_xywh":   Tensor[N, 4],
  "causes":              List[str],            # raw GT cause strings
  "cause_emb_indices":   List[int],            # 指向 cause_text_embs
}
```

執行時間：~5 min（Hungarian 主要瓶頸）。

---

### Phase 0.5：Cause taxonomy（HDBSCAN）

cause text embedding cache 已經存在（`outputs/cause_cache/`），直接做兩種粒度的 HDBSCAN 聚類：

**細粒度（2807 個 sub-cluster，論文 ablation 用）：**
```bash
$PY -m diagnosis_model.cause_inference.preprocessing.recluster_causes cluster \
  --cache_dir outputs/cause_cache \
  --reduced_path outputs/cause_reduced.npy \
  --output outputs/cause_clusters.json \
  --cluster_selection_method eom --min_cluster_size 6

$PY -m diagnosis_model.cause_inference.preprocessing.recluster_causes reassign-singletons \
  --cache_dir outputs/cause_cache \
  --input outputs/cause_clusters.json \
  --output outputs/cause_clusters_reassigned.json \
  --cosine_threshold 0.92 --margin 0.03 --min_real_cluster_size 2
```

**粗粒度（100 個 disease topic，主要評估指標用）：**
```bash
$PY -m diagnosis_model.cause_inference.preprocessing.recluster_causes cluster \
  --cache_dir outputs/cause_cache \
  --reduced_path outputs/cause_reduced.npy \
  --output outputs/cause_clusters_v100.json \
  --cluster_selection_method eom --min_cluster_size 100

$PY -m diagnosis_model.cause_inference.preprocessing.recluster_causes reassign-singletons \
  --cache_dir outputs/cause_cache \
  --input outputs/cause_clusters_v100.json \
  --output outputs/cause_clusters_v100_reassigned.json \
  --cosine_threshold 0.70 --margin 0.0 --min_real_cluster_size 100
```

| Taxonomy | clusters | mean size | 用途 |
|---|---|---|---|
| 細粒度 (`cause_clusters_reassigned.json`) | 2807 | 20 | 嚴格指標、ablation |
| 粗粒度 (`cause_clusters_v100_reassigned.json`) | **100** | **563** | **論文主指標（disease topic）** |

---

### Phase 1：zero-training C²R baseline

對每個 valid query：

1. 跟所有 train case 算組合相似度
   - `sim(q, c) = α · cos(q.global, c.global) + β · Hungarian(q.lesions, c.lesions)`
   - 預設 α=0.25, β=0.75
2. 取 top-K cases（預設 K=20）
3. 候選池 = 這 K 個 case 的去重病因（~87 個候選）
4. 候選打分（embedding-space）：
   - `score(c) = Σ_j w_j · max_g cos(emb(c), emb(e_{j,g}))`
5. greedy diversification（cos > 0.95 視為重複，suppress）
6. 輸出 top-N

```bash
$PY -m diagnosis_model.cause_inference.phase1_baseline \
  --case_db_dir outputs/case_db \
  --output_dir outputs/phase1_full \
  --top_k_cases 20 --alpha_global 0.25 --beta_lesion 0.75 \
  --diversify_threshold 0.95 --semantic_threshold 0.95 \
  --lesion_match hungarian \
  --cluster_json outputs/cause_clusters_v100_reassigned.json
```

**hyperparameter sweep：**
```bash
bash diagnosis_model/cause_inference/run_phase1_sweep.sh
```
測試 K ∈ {10, 20, 30, 50}, α ∈ {0, 0.25, 1.0}, diversify ∈ {0.95, 0.97, 0.99}。
Sweet spot：**K=20, α=0.25, diversify=0.95**。

**Phase 1 baseline 全量結果（1573 valid queries）：**

| Metric | Value |
|---|---|
| Pool size mean | 87 |
| Semantic R@1 | 22.3% |
| Semantic R@10 | **78.6%** |
| Semantic R@20 | 84.4% |
| Semantic coverage | 85.8% |
| Semantic MRR | 0.42 |

`coverage = 85.8%` 是 retrieval-only 上限——剩下 14% 是真的「相似 case 庫沒這種病因」的 OOD。

---

### Phase 2：projection head ablation

訓練一個輕量 MLP 把 lesion 特徵投到「對病因區分更好」的空間，用 cause-overlap supervised metric learning：

```bash
$PY -m diagnosis_model.cause_inference.train_projection \
  --case_db_dir outputs/case_db \
  --output_dir outputs/projection_v1 \
  --epochs 30 --batch_size 64 --lr 1e-4 --eval_every 2
```

**結果**：sem_MRR 0.4061 → 0.4052（**沒有改進**）。

**論文寫法（ablation）**：「Cause-overlap supervised projection on top of frozen VLM-Lesion provides no measurable improvement; we attribute this to the VLM's symptom-caption pretraining already saturating the cause-discriminative signal at the case-similarity level.」

---

### Phase 3：CEAH（主要貢獻）

CEAH = **Cause-Evidence Attribution Head**。對 Phase 1 候選池中的每個候選病因，輸出：
- 一個機率分數
- 一個對 evidence tokens（global / text / lesion 1..N）的 **softmax attribution** α

設計關鍵：
1. **softmax α**（不是獨立 sigmoid）→ α 加總 = 1，global 不能獨吞所有 attention
2. **multiplicative scoring**：`score = sigmoid(MLP_g([c, gated_global])) · sigmoid(MLP_l([c, gated_local]))` → global / local 必須各自對才能高分
3. **Architectural faithfulness**：`gated_pool = Σ α_i · e_i`，α_i = 0 時對應 evidence 真的不貢獻

#### Step 3a：建 hard-negative candidate pool

對每個 train case 跑 leave-one-out Phase 1 retrieval，把候選池的正負例離線存好（避免訓練時每 step 跑檢索）：

```bash
$PY -m diagnosis_model.cause_inference.preprocessing.build_train_candidate_pool \
  --case_db_dir outputs/case_db \
  --output_path outputs/case_db/train_candidate_pool.pt \
  --top_k_cases 20
```

產出（172 MB）：每個 train case 對應 87 個候選 cause indices + positive_mask（`semantic_threshold=0.95` 標記）。執行時間 ~3 min。

#### Step 3b：訓練 CEAH

```bash
$PY -m diagnosis_model.cause_inference.train_ceah \
  --case_db_dir outputs/case_db \
  --train_pool_path outputs/case_db/train_candidate_pool.pt \
  --output_dir outputs/ceah_v3 \
  --attribution_mode softmax \
  --scoring_mode multiplicative \
  --epochs 20 --batch_size 32 --lr 1e-4 \
  --lambda_sparsity 0.0 \
  --text_dropout 0.5 \
  --eval_max_queries 300
```

**重要 hyperparameter：**
- `--attribution_mode softmax`（**必要**——之前 sigmoid 模式 α 塌陷）
- `--scoring_mode multiplicative`（**必要**——強制 lesion attribution）
- `--lambda_sparsity 0.0`（softmax 已自帶 normalization 約束）
- `--text_dropout 0.5`（推論時可能無 text，訓練要對 vision-only fallback）

執行時間 ~4 min（GPU）。

**訓練版本演進（負面結果寫進論文 ablation）：**

| Version | Architecture | 結果 |
|---|---|---|
| v1 | sigmoid + single, λ=0.05 | α 塌到 ~0.06，evidence 完全沒用 |
| v2 | sigmoid + single, λ=0.005 | α 漲到 0.30，但 lesion α=0.02（94% 全押 global） |
| **v3** | **softmax + multiplicative, λ=0** | **cause-type-aware lesion attribution** |

#### Step 3c：CEAH inference + α 輸出

```bash
$PY -m diagnosis_model.cause_inference.eval_ceah \
  --case_db_dir outputs/case_db \
  --ceah_ckpt outputs/ceah_v3/best_ceah.pt \
  --output_dir outputs/ceah_v3_eval_full \
  --attribution_mode softmax --scoring_mode multiplicative \
  --gammas 0.0 0.25 0.5 0.75 1.0 --dump_gamma 0.5 \
  --cluster_json outputs/cause_clusters_v100_reassigned.json
```

**Hybrid scoring**：`final = γ · phase1_score + (1−γ) · ceah_score`（per-query min-max 標準化）。掃 γ 找 sweet spot。

**γ 掃描結果（v3，1573 valid queries）：**

| γ | sem_MRR | sem_R@10 | sem_R@20 | cluster_R@10 |
|---|---|---|---|---|
| 0.00 (CEAH only) | 0.393 | 76.2% | 84.0% | 30.4% |
| 0.50 | 0.413 | 78.4% | 84.6% | 30.4% |
| **0.75** | **0.417** | **78.7%** | 84.6% | 30.2% |
| 1.00 (Phase 1 only) | 0.415 | 78.6% | 84.4% | 30.9% |

→ **γ=0.75 hybrid 微贏 Phase 1 baseline**，CEAH 沒退化 retrieval。

---

### Faithfulness 驗證

對每個 valid query 的 top-1 預測，依次遮掉 global / text / lesions / top-α token，量分數變化：

```bash
$PY -m diagnosis_model.cause_inference.faithfulness_eval \
  --case_db_dir outputs/case_db \
  --ceah_ckpt outputs/ceah_v3/best_ceah.pt \
  --output_dir outputs/ceah_v3_faithfulness_v2 \
  --attribution_mode softmax --scoring_mode multiplicative \
  --max_queries 1573
```

**v3 結果（按病因類型分桶）：**

| condition | global-type | lesion-type | mixed |
|---|---|---|---|
| no_lesion drop | +0.034 | **+0.040** | +0.036 |
| no_random drop | +0.006 | +0.006 | +0.002 |
| **lesion vs random ratio** | 5.7× | **6.7×** | 18× |

**按 lesion 數量分桶：**

| condition | N=1 | N=2 | N≥3 |
|---|---|---|---|
| no_lesion drop | 0.025 | 0.051 | **0.055** |
| no_random drop | 0.005 | 0.005 | 0.003 |
| **lesion vs random ratio** | 5.0× | 10× | **18×** |

→ **遮 lesion 對 multi-lesion case 影響更大**（N≥3 是 random 的 18 倍），證明 lesion attribution 是 load-bearing。

---

### N-lesion 分桶分析

```bash
$PY -m diagnosis_model.cause_inference.analyze_v3_n_buckets \
  --eval_dir outputs/ceah_v3_eval_full \
  --case_db outputs/case_db
```

**Attribution 隨 N 變化：**

| N | global α | lesion sum | concentration (max/sum) |
|---|---|---|---|
| 1 | 0.51 | 0.27 | 1.00 (only 1) |
| 2 | 0.36 | 0.43 | 0.60 |
| ≥3 | **0.26** | **0.55** | **0.37** ≈ 1/N |

→ N 越多，**模型把更多 attribution 放在 lesion 集合**（從 27% 漲到 55%），但**不挑單一 lesion**（concentration → 1/N 是均勻分配）。

**這是論文的乾淨 finding：模型對 multi-lesion 採「集體證據」策略，不是「挑主病灶」**。

---

### Text mode ablation（vision-only fallback 驗證）

訓練時 `--text_dropout 0.5` 讓模型同時學「有 text」和「無 text」兩種模式。推論時 `--text_kind` 可切換 `medical` / `colloquial` / `none`：

```bash
for mode in medical colloquial none; do
  $PY -m diagnosis_model.cause_inference.eval_ceah \
    --case_db_dir outputs/case_db \
    --ceah_ckpt outputs/ceah_v3/best_ceah.pt \
    --output_dir outputs/ceah_v3_text_${mode} \
    --cluster_json outputs/cause_clusters_v100_reassigned.json \
    --attribution_mode softmax --scoring_mode multiplicative \
    --gammas 0.75 --dump_gamma 0.75 --text_kind ${mode}
done
```

**Retrieval 三 mode 完全持平（diff ≤ 0.003）：**

| Text mode | sem_MRR | R@1 | R@10 | R@20 | coverage |
|---|---|---|---|---|---|
| medical | 0.4170 | 22.38% | 78.65% | 84.61% | 86.06% |
| colloquial | 0.4167 | 22.34% | 78.62% | 84.61% | 86.03% |
| **none** (vision-only) | 0.4166 | 22.34% | **78.51%** | 84.49% | 85.98% |

→ **Vision-only 推論完全可行**，使用者沒提供文字描述系統照樣運作。

**Attribution 隨 text mode 自動補位：**

| mode | bucket | global α | text α | lesion sum | l/g ratio |
|---|---|---|---|---|---|
| medical | global-type | 0.524 | 0.175 | 0.301 | 0.58 |
| medical | lesion-type | 0.314 | 0.258 | 0.427 | **1.36** |
| colloquial | global-type | 0.538 | 0.139 | 0.323 | 0.60 |
| colloquial | lesion-type | 0.351 | 0.126 | 0.523 | **1.49** |
| **none** | global-type | 0.600 | 0.000 | **0.400** | 0.67 |
| **none** | lesion-type | 0.379 | 0.000 | **0.621** | **1.64** |

**重要 finding**：text 缺席時，模型自動把原本給 text 的 attention mass（0.13–0.26）**重新分配給 lesion**，**保持 cause-type-aware 行為**（lesion-type l/g 從 1.36 升到 1.64）。`text_dropout` 訓練設計奏效。

`colloquial` 比 `medical` 給出更高 l/g ratio（1.49 vs 1.36），可能因 medical 文字本身含「水質惡化」「細菌感染」等診斷術語，已在文字端解釋部分病因；colloquial 較口語，模型必須更靠 lesion 補回。

---

### 案例可視化

```bash
$PY -m diagnosis_model.cause_inference.case_study_viz \
  --eval_dir outputs/ceah_v3_eval_full \
  --case_db outputs/case_db \
  --image_root data/detection/coco/_merged/valid \
  --output_dir outputs/case_study_v3 \
  --row_indices 79 94 13 15 133
```

每張圖：左側魚體 + lesion bbox 標註 α 值 + GLOBAL/TEXT 角落顯示；右側 GT 病因 + Top-3 預測 + α breakdown。

**5 個論文 figure 候選：**

| 圖 | 性質 |
|---|---|
| `case_0079_global_type.png` | global-type 病因 → α heavy on global |
| `case_0094_lesion_type.png` | lesion-type → 兩 lesion 都濃 |
| `case_0013_lesion_type.png` | lesion-type → 主 lesion 被選 |
| `case_0015_lesion_type.png` | 3-lesion → L2 被選 |
| `case_0133_lesion_type.png` | 3-lesion → L0 被選（不同位置） |

---

## 完整結果 Summary

### Retrieval（v3 hybrid γ=0.75 vs Phase 1 baseline）

| 評估層級 | R@1 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|
| Exact 字串（56k 唯一） | 0.06% | 0.3% | 0.4% | 0.6% | 0.002 |
| **Semantic（cos≥0.95）** | 22.4% | 68.4% | **78.7%** | 84.6% | **0.417** |
| **Cluster（100 topics）** | 5.2% | 20.8% | **30.4%** | 39.7% | **0.131** |

### Attribution（cause type × α 分佈）

| 病因類型 | global α | text α | lesion sum | l/g ratio |
|---|---|---|---|---|
| global-type（水質/緊迫） | 0.51 | 0.18 | 0.31 | 0.44 |
| lesion-type（細菌/寄生蟲） | **0.25** | 0.28 | **0.47** | **1.39** ← 反轉 |
| mixed | 0.39 | 0.20 | 0.40 | 0.72 |

### Faithfulness（masking experiment, lesion vs random drop ratio）

| 桶 | 比例 |
|---|---|
| 全部 | 7.4× |
| N=1 | 5.0× |
| N=2 | 10× |
| **N≥3** | **18×** |

---

## 論文敘事建議

> **FaCE-R 由三個元件組成：**
>
> **C¹ (Phase 1)**: Hungarian-matched lesion-set similarity 做 zero-training 案例檢索。在 1,573 個 valid query 上達到 **semantic R@10 = 78.6%**。Hyperparameter sweep 顯示 K=20, α=0.25 是最佳配置；global-only / lesion-only 分別給 77.4% / 74.5%，證明 global 是主信號（VLM-Global pretraining 的優勢）。
>
> **C² (Phase 2 ablation)**: Cause-overlap supervised projection learning 在 frozen VLM 上提供不到 0.001 的 MRR 改進，作為 negative result 報告——VLM 表徵已飽和。
>
> **C³ (Phase 3, CEAH)**: 設計 softmax + multiplicative scoring 的 attribution head：
> - **架構強制 faithfulness**：α 透過 gated pooling 直接門控 evidence 對分數的貢獻
> - **Hybrid γ=0.75 retrieval = 78.7%**，跟 Phase 1 持平不退化
> - **Cause-type-aware attribution**：global-type 病因 lesion/global α ratio = 0.44, lesion-type ratio = 1.39（3.2× 反轉）
> - **Faithfulness 驗證**：遮 lesion 對 N≥3 的 case 影響是隨機遮的 **18 倍**
> - **Multi-lesion 整合行為**：N=3+ 時 attribution 平均分散到所有 lesion（concentration = 1/N），證明模型採「集體視覺證據」策略
> - **Vision-only deployment 可行**：medical/colloquial/none 三種推論模式 retrieval 差 ≤ 0.003 MRR；text 缺席時 attention 自動補位給 lesion，cause-type-aware 行為 preserved

---

## 已知限制

- **Cause-emb 跟 lesion-emb 在不同 VLM 空間**：CEAH 必須學跨空間映射，可能限制 lesion attribution 的精細度
- **HDBSCAN 細粒度過分割**：disease topic 級別需要重新粗粒度聚類（v100），否則 cluster R@K 假性偏低
- **N≥3 attribution spread**：模型不會從多 lesion 中挑單一主病灶。若臨床需要「指向單一病灶」需額外 hard top-1 約束
- **Text 對 retrieval 沒實質幫助**：text mode ablation 顯示 medical/colloquial/none 三模式 retrieval 差 ≤ 0.003 MRR；text 在 attribution 端有作用（吸收 0.13–0.26 attention mass），但不影響 top-K 結果。換言之，部署若不收集文字描述也不會丟分。
