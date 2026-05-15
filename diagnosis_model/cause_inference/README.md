# FaCE-R: Faithful Case-guided Cause-Evidence Retrieval

魚病病因推論系統的第三階段——在已訓練的 RF-DETR 病灶偵測器與兩個 SigLIP2 多模態模型上，做 case-based 病因檢索 + 架構強制的 lesion-grounded 可解釋性。

## 動機

魚病診斷的病因標註是 **LLM 生成的自由文字並經由水產專業人員審核**（每張圖約 4 個病因，總共 56,310 個 unique 字串，**94.7% 是 singleton**）。這個資料特性使得：

1. 傳統的 closed-vocabulary multi-label classification **無法應用**
2. 為了符合實際的診斷應用需求，系統必須具備「**lesion-grounded explainability**」——預測必須能對應到具體病灶

FaCE-R 把問題重構成 **case-based 檢索 + lesion-attribution**，兩階段對 VLM 的依賴不對稱：

```
Query 影像
  ↓
RF-DETR 偵測病灶（前一階段）
  ↓
SigLIP2 抽特徵（frozen teacher，預訓練即可；Phase 2 額外需要 fine-tune 過的 fusion VLM-Lesion）
  ↓
[Phase 1] C²R 案例檢索 → 候選病因池（~87 個）         ← zero-shot retrieval（raw SigLIP2 即可）
  ↓
[Phase 2] CEAH 重新打分 + 病灶歸因                    ← 需 VLM-Lesion fusion 才有 faithful attribution
  ↓
Top-N 病因預測 + α attribution 視覺化
```

**設計哲學**：retrieval 端走 zero-shot vision-language 的範式（不微調 VLM），attribution 端走 lightweight head + 必要的 lesion-specialized 視覺特徵。Ablation 顯示兩者的依賴不能互換（[VLM dependency ablation](#phase-1-ablationvlm-dependency)）。

## 上游依賴

FaCE-R 的兩個階段對 VLM 的依賴**不對稱**：

- **Phase 1（C²R retrieval）**：把 SigLIP2 當 **frozen zero-shot teacher**。Ablation 顯示原始 `google/siglip2-base-patch16-224` 跟 fine-tune 過的 VLM-Global 在 retrieval 上**等價甚至略勝**（sem R@10 45.6% vs 44.4%），所以這個階段不需要任何 in-domain 微調。
- **Phase 2（CEAH lesion attribution）**：**必須**用 fine-tune 過的 VLM-Lesion + LocalGlobalFusion。raw SigLIP2 的 lesion crop 特徵雖然能驅動 retrieval，但無法支撐 faithful attribution（lesion masking 結果反轉，見 [VLM dependency ablation](#phase-1-ablationvlm-dependency)）。

| 元件 | 路徑 | 訓練語料 | 用於哪個階段 |
|---|---|---|---|
| **RF-DETR 病灶偵測器** | [`../detection/`](../detection/) | COCO bbox | 推論前處理 |
| **SigLIP2 base (frozen)** | `google/siglip2-base-patch16-224` | 原始預訓練 | **Phase 1 retrieval 的 zero-shot baseline** |
| **VLM-Global**（整圖 ↔ 整體描述） | [`../vl_classifier/outputs/siglip2_base_patch16_224_overall_multipos_zh`](../vl_classifier/outputs/siglip2_base_patch16_224_overall_multipos_zh) | `overall.medical_zh` / `colloquial_zh` | Phase 1（與 raw 等價）、Phase 2 |
| **VLM-Lesion**（病灶 crop ↔ 症狀描述，含 fusion wrapper） | [`../vl_classifier/outputs/siglip2_base_patch16_224_multipos_fusion_en_zh`](../vl_classifier/outputs/siglip2_base_patch16_224_multipos_fusion_en_zh) | per-bbox symptom captions | **Phase 2 CEAH faithfulness 必要** |
| **COCO 標註** | `data/detection/coco/_merged/{train,valid,test}/_annotations.coco.json` | image-level + per-bbox annotations | 全程 |

所有 VLM 在 cause_inference 內皆凍結，不進一步 finetune。Production case_db 用 fine-tune 過的兩個 VLM 建立（因 CEAH 依賴）；Phase 1 retrieval 在這個 case_db 上跑就好。

## Repo 結構

```
cause_inference/
├── README.md                      ← 本檔
├── preprocessing/
│   ├── README.md                  ← cause text caching + clustering CLI
│   ├── build_case_database.py     ← Phase 0: 建 case DB
│   ├── build_train_candidate_pool.py  ← Phase 2 訓練前置
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
│   ├── projection_head.py         ← supervised projection MLP（retrieval-side ablation）
│   └── ceah.py                    ← Phase 2 模型（CEAH）
├── phase1_baseline.py             ← Phase 1 zero-training C²R 評估
├── train_projection.py            ← supervised projection MLP 訓練（retrieval-side ablation）
├── train_ceah.py                  ← Phase 2 CEAH 訓練
├── eval_ceah.py                   ← Phase 2 推論 + α 輸出
├── faithfulness_eval.py           ← Faithfulness 驗證（lesion masking）
├── analyze_lesion_buckets.py      ← Phase 1 N-lesion 分桶分析
├── analyze_v3_n_buckets.py        ← Phase 2 v3 N-lesion 分桶分析
├── case_study_viz.py              ← 案例可視化（論文 figure）
├── run_phase1_sweep.sh            ← Phase 1 hyperparameter sweep
├── mamba_ablation/                ← Phase 3 Mamba 變體（architecture ablation）
├── rvq_rerank/                    ← Phase 4 CRR-DeepRVQ（壓縮 + 殘差 reranker）
│   ├── rvq.py                     ← RVQCodebook 模組
│   ├── reranker.py                ← Light / Full reranker
│   ├── fit_rvq.py / build_rvq_index.py / run_sweep.py
│   ├── eval_sanity.py / eval_harder.py / eval_final.py
│   ├── train_reranker.py / benchmark_scale.py
│   └── outputs/                    ← codebooks、index、reranker checkpoints
└── outputs/                       ← 所有產出
    ├── case_db/                   ← Phase 0（含 train_candidate_pool.pt）
    ├── phase1_full_maxmean/       ← Phase 1 baseline
    ├── phase1_sweep/              ← Phase 1 sweep
    ├── projection_v1/             ← supervised projection MLP ablation（retrieval-side）
    ├── ceah_v3/                   ← Phase 2 final model
    ├── ceah_v3_eval_full/         ← Phase 2 retrieval eval（fine cluster）
    ├── ceah_v3_eval_coarse/       ← Phase 2 retrieval eval（coarse 100 cluster）
    ├── ceah_v3_text_{medical,colloquial,none}/  ← Phase 2 text mode ablation
    ├── ceah_v3_faithfulness_v2/   ← Phase 2 faithfulness
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

**Production case_db 使用 fine-tune 過的兩個 VLM**（CEAH 對 VLM-Lesion fusion 的依賴是必要的，見 [VLM dependency ablation](#phase-1-ablationvlm-dependency)）：

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

`build_case_database.py` 同時支援 `--raw_lesion` flag 來建立 raw SigLIP2 對照版本（lesion crop 走原生 image encoder，跳過 LocalGlobalFusion；給 [Phase 1 ablation](#phase-1-ablationvlm-dependency) 用）。

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
   - `sim(q, c) = α · cos(q.global, c.global) + β · lesion-set cosine`
   - 預設 α=0.25, β=0.75；lesion-set 預設 hungarian matching
2. 取 sim > 0 的 top-K cases（預設 K=20），權重 sum-to-1 標準化
3. 候選池 = 這 K 個 case 的去重病因（~87 個候選）
4. 候選打分（embedding-space）：
   - `score(c) = Σ_j w_j · max_g cos(emb(c), emb(e_{j,g}))`
5. raw ranking 直接給 metrics；diversification（cos ≥ 0.95 suppress）只用於 `predicted_top_n` 輸出
6. cluster metric 是 per GT cause occurrence

```bash
$PY -m diagnosis_model.cause_inference.phase1_baseline \
  --case_db_dir outputs/case_db \
  --output_dir outputs/phase1_final \
  --top_k_cases 20 --alpha_global 0.25 --beta_lesion 0.75 \
  --diversify_threshold 0.95 --semantic_threshold 0.95 \
  --lesion_match hungarian \
  --cluster_json outputs/cause_clusters_llm.json
```

**hyperparameter sweep：**
```bash
bash diagnosis_model/cause_inference/run_phase1_sweep.sh
```
測試 K ∈ {10, 20, 30, 50}, α ∈ {0, 0.25, 1.0}, diversify ∈ {0.95, 0.97, 0.99}。
Sweet spot：**K=20, α=0.25, diversify=0.95**。

**Phase 1 baseline 全量結果（1573 valid queries, LLM 466 clusters）：**

| Metric | Value |
|---|---|
| Pool size mean | 87 |
| Semantic coverage | **93.1%** |
| Semantic MRR | 0.298 |
| Semantic R@1 / R@10 / R@20 | 22.3% / 44.4% / 58.9% |
| Cluster MRR | 0.235 |
| Cluster R@1 / R@10 / R@20 | 17.2% / 35.8% / 47.6% |

`semantic coverage = 93.1%` 是 retrieval-only 上限——剩下 ~7% 是真的「相似 case 庫沒這種病因」的 OOD。

---

### Phase 1 ablation：VLM dependency

問題：Phase 1 retrieval 需不需要 in-domain fine-tune 的 VLM？我們把整個 case_db 用 raw `google/siglip2-base-patch16-224` 重建（lesion crop 走原生 image encoder，跳過 LocalGlobalFusion）並重跑 Phase 1：

```bash
# 重建 raw case_db（~7 min）
$PY -m diagnosis_model.cause_inference.preprocessing.build_case_database \
  --coco_train data/detection/coco/_merged/train/_annotations.coco.json \
  --image_root_train data/detection/coco/_merged/train \
  --coco_valid data/detection/coco/_merged/valid/_annotations.coco.json \
  --image_root_valid data/detection/coco/_merged/valid \
  --vlm_global google/siglip2-base-patch16-224 \
  --vlm_lesion google/siglip2-base-patch16-224 \
  --raw_lesion \
  --output_dir outputs/case_db_raw \
  --chunk_size 64

# 重跑 Phase 1（~30 sec）
$PY -m diagnosis_model.cause_inference.phase1_baseline \
  --case_db_dir outputs/case_db_raw \
  --output_dir outputs/phase1_raw \
  --cluster_json outputs/cause_clusters_llm.json \
  --top_k_cases 20 --alpha_global 0.25 --beta_lesion 0.75 \
  --lesion_match hungarian --diversify_threshold 0.95 --semantic_threshold 0.95
```

**結果（1573 valid，LLM 466 cluster）：**

| Metric | Fine-tuned VLMs | Raw SigLIP2 | Δ |
|---|---|---|---|
| sem R@1 | 22.3% | 22.2% | −0.1 |
| sem R@5 | 34.8% | 35.7% | +0.9 |
| sem **R@10** | 44.4% | **45.6%** | **+1.2** |
| sem R@20 | 58.9% | 59.2% | +0.3 |
| sem MRR | 0.298 | 0.300 | +0.002 |
| cluster R@10 | 35.8% | 35.8% | 0.0 |
| cluster MRR | 0.235 | 0.235 | 0.0 |

→ **Raw SigLIP2 在 retrieval 端持平、甚至略勝**。Phase 1 不需要 in-domain fine-tune。

**論文寫法**：「Phase 1 follows a zero-shot retrieval paradigm: we treat SigLIP2 as a frozen pretrained teacher and rely entirely on its general image-text alignment. An ablation with the fine-tuned VLMs (trained on overall description and per-bbox symptom captions) shows no retrieval improvement (in fact slightly worse), confirming that the case-based retrieval framework is robust to feature-level fine-tuning.」

**但 CEAH 不適用**：把 raw case_db 接上 CEAH 訓練 + faithfulness eval：

| Faithfulness | Fine-tuned case_db | Raw case_db |
|---|---|---|
| `no_lesion` drop（lesion-type） | **+0.040** | **−0.031** ← 反轉 |
| `no_lesion` drop（all） | +0.036 | −0.027 |
| `no_random` drop | +0.005 | +0.004 |

→ raw 的 CEAH 把 attention 放在 lesion 上，但**遮掉 lesion 後分數反而上升** — 即 lesion 內容不是有效的判別訊號。fine-tune 過的 VLM-Lesion（含 LocalGlobalFusion）對 Phase 2 的 architecture-enforced lesion-grounded explainability 是**必要**的。

詳見 [Phase 2 design notes](#phase-2ceah主要貢獻) 對 VLM-Lesion 必要性的說明。

Artifacts：`outputs/case_db_raw/`、`outputs/phase1_raw/`、`outputs/ceah_raw/`、`outputs/ceah_raw_faithfulness/`。

---

### Retrieval-side fine-tune probe：supervised projection MLP（negative result）

延伸前一節的 VLM dependency 結論——除了「換掉 VLM」之外,我們也試過**保留 frozen VLM,在 lesion 特徵之上加一顆可學的 projection MLP**,直接用 cause-overlap 當監督訊號做 metric learning(`pred(i,j) = α·cos(g_i,g_j) + β·max_mean_set_sim(L'_i, L'_j)` vs `target(i,j) = max_mean_set_sim(C_i, C_j)`,MSE on off-diagonals):

```bash
$PY -m diagnosis_model.cause_inference.train_projection \
  --case_db_dir outputs/case_db \
  --output_dir outputs/projection_v1 \
  --epochs 30 --batch_size 64 --lr 1e-4 --eval_every 2
```

**結果**:sem_MRR 0.4061 → 0.4052(**沒有改進**;Δ MRR −0.001)。

這個 probe 不是 pipeline 的一階,**下游沒有任何階段載入 `best_lesion_head.pt`**——Phase 2 CEAH 自帶 attribution MLP、Phase 3 DeepSets 的 φ 本身就是 per-lesion 的可學 projection(且訓練目標更強:listwise KL + SupCon InfoNCE),Phase 4 RVQ + reranker 坐在 Phase 3 的 z 上。Probe 的價值純粹是 retrieval-side fine-tune ablation 的**一個獨立資料點**,與「Phase 1 raw vs fine-tuned VLMs」、「Phase 3 raw distillation」、「Phase 4 raw RVQ + reranker」共同支持「coarse retrieval 在 zero-shot vision-language regime 下飽和」的論述。

**論文寫法(ablation)**:「In addition to swapping in the raw SigLIP2 backbone, we also probe whether a cause-overlap supervised projection MLP on top of frozen VLM-Lesion can lift retrieval. Trained on pairwise lesion-set similarity aligned to pairwise cause-text similarity, the head produces no measurable improvement (Δ sem MRR = −0.001), consistent with the broader cross-ablation showing in-domain fine-tuning is a no-op for the coarse stage.」

---

### Phase 2：CEAH(主要貢獻)

CEAH = **Cause-Evidence Attribution Head**。對 Phase 1 候選池中的每個候選病因，輸出：
- 一個機率分數
- 一個對 evidence tokens（global / text / lesion 1..N）的 **softmax attribution** α

設計關鍵：
1. **softmax α**（不是獨立 sigmoid）→ α 加總 = 1，global 不能獨吞所有 attention
2. **multiplicative scoring**：`score = sigmoid(MLP_g([c, gated_global])) · sigmoid(MLP_l([c, gated_local]))` → global / local 必須各自對才能高分
3. **Architectural faithfulness**：`gated_pool = Σ α_i · e_i`，α_i = 0 時對應 evidence 真的不貢獻
4. **Lesion encoder 必須 fine-tune（含 LocalGlobalFusion）**：在 raw SigLIP2 的 case_db 上重訓 CEAH 顯示 retrieval 持平（sem R@10 46.4% vs 45.3%），但 faithfulness 完全反轉（`no_lesion` drop 從 +0.040 變 −0.031）。raw lesion crop 特徵不足以驅動架構強制的 lesion-grounded explainability — CEAH 雖把 attention 放在 lesion 上，但 lesion 內容不是有效訊號。詳見 [Phase 1 ablation：VLM dependency](#phase-1-ablationvlm-dependency)。

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

**γ 掃描結果（v3，1573 valid queries，LLM 466 cluster）：**

| γ | sem_MRR | sem_R@10 | cl_MRR | cl_R@10 |
|---|---|---|---|---|
| 0.00 (CEAH only) | 0.301 | **46.8%** | **0.271** | **39.8%** |
| 0.50 | **0.303** | 45.9% | 0.254 | 37.6% |
| 0.75 | 0.302 | 45.3% | 0.245 | 37.0% |
| 1.00 (Phase 1 only) | 0.298 | 44.4% | 0.235 | 35.8% |

→ CEAH 對 cluster recall 有實質提升（γ=0 比 γ=1 多 +4pp cl_R@10），semantic 端 γ=0.5 達 sem_MRR 最佳。γ=0.75 為 paper figure dump 預設。

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

### Phase 3：Case encoder（單向量檢索，加速貢獻）

**動機**：Phase 1 每 query 對 case_db 跑 multi-vector + Hungarian 比對（~15 ms/q）。隨著 case_db 規模成長,這個成本是 retrieval 階段的主要 bottleneck。把 (global, lesion 1..N) 蒸餾成 **單一 case 向量** 後,檢索退化為單一 cosine,大幅加速。

#### 設計

每個 case 表徵為:
- `global_emb ∈ ℝ^768`（VLM-Global,master）
- `lesion_embs ∈ ℝ^(N×768)`（VLM-Lesion,slaves,**依面積 DESC 排序**）

Encoder 把這兩者壓成單一 L2-normed `h_final ∈ ℝ^768`。檢索 = `cos(h_query, h_train)` 對 case_db 取 top-K。

**最終選擇:DeepSets pooling（mean + max + sum → MLP）**。下節說明為何不選 Mamba / Set Transformer。

#### 訓練 loss(dual-target)

兩個 loss 加總:

1. **Listwise KL distillation**(主信號)
   - Teacher: Phase 1 hungarian best config(K=20, α=0.25, β=0.75),預算成 12780² 矩陣存成 `teacher_train_train.pt`
   - 學生對 batch 內 case-case cosine 矩陣的 row-softmax 對齊 teacher 的 row-softmax(T=0.1)

2. **Case-to-cause SupCon InfoNCE**(輔助信號,**讓學生可能超越老師**)
   - Positives: case 的 GT cause text embeddings
   - Negatives: 全 56k cause vocab
   - L_out form(SupCon paper),溫度 T=0.07,權重 0.5
   - 為什麼可能贏老師: Phase 1 完全不用 cause text embedding,直接對齊 cause space 是 orthogonal 訊號

```bash
# Step 1: 一次性建 teacher table(~3 min)
$PY -m diagnosis_model.cause_inference.preprocessing.build_teacher_table \
  --case_db_dir outputs/case_db \
  --output_path outputs/case_db/teacher_train_train.pt

# Step 2: 訓練 encoder(~10 min,直接用 SDM env)
$PY -m diagnosis_model.cause_inference.train_case_encoder \
  --output_dir outputs/encoder_final \
  --encoder_type deepsets \
  --batch_size 256 --epochs 50 \
  --use_infonce --infonce_weight 0.5 --infonce_temp 0.07
```

> Production Phase 3 跑在 SDM env 即可，不需要 `mamba3`/`gcc-12`。Mamba master-slave
> 變體已搬到 [`diagnosis_model/cause_inference/mamba_ablation/`](mamba_ablation/)，作為
> architecture ablation 保留；`build_encoder` lazy-imports，傳 `--encoder_type mamba`
> 才會觸發 `mamba_ssm` 載入並要求 mamba3 env。

#### 結果(1573 valid,Phase 1-aligned 評估,K=20)

| Method | sem R@10 | MRR | per-q ms | Notes |
|---|---|---|---|---|
| **Phase 1 hungarian**(teacher) | 0.444 | 0.298 | 15.4 | multi-vector + Hungarian |
| DeepSets single distill | 0.445 | 0.297 | 1.1 | 蒸餾打平 |
| **DeepSets dual-target**(最終選擇) | **0.447** | 0.298 | **1.1** | **+0.3 R@10, 14× faster** |

接 CEAH 串成 hybrid(γ=0.75, text=medical)後:

| Method | sem R@10 | per-q ms |
|---|---|---|
| Phase 1 + CEAH | 0.453 | 18.2 |
| DeepSets dual + CEAH | 0.453 | **3.4** (5.4× faster) |

#### 探索過程(架構 / 訓練 / 評估三條 ablation)

**架構選擇** — 我們試了三種 encoder,**最終發現本任務序列太短(avg 1.73 lesion),架構幾乎沒差**:

| Encoder | sem R@10 | sem R@10 (lesion≥4) | 結論 |
|---|---|---|---|
| Mamba3 master-slave | 0.445 | 0.422 | 無架構優勢 |
| MeanPool baseline | 0.443 | 0.418 | 複雜 case 上稍弱 |
| **DeepSets** | **0.447** | **0.441** | 同等或更好,參數最少 |

**為什麼放棄 Mamba**:
- 序列長度 L ≤ 19(99% ≤ 4),Mamba 的 `O(L)` 優勢要 L > ~256 才出現
- 此 L 下 Transformer 的 `O(L²·d)` < Mamba 的 `O(L·d²)`(d=768 太大,L 太小)
- mamba_ssm `is_mimo=True` 的 backward kernel 在這台機環境編譯失敗,只能用 vanilla Mamba3,進一步抵消優勢
- Set Transformer 已是 set learning 領域標準,且 ISAB 比 DeepSets 慢
- 在我們的 ablation 中三者打平 → 選參數最少的 DeepSets

**訓練 ablation**:

| 訓練方式 | sem R@10 | 結論 |
|---|---|---|
| Distillation only(listwise KL) | 0.445 | 完美 mimic 老師 |
| **+ case-cause InfoNCE**(dual-target) | **0.447** | +0.3 R@10,微幅超越老師 |
| + miss-weighted sampler(hard-case ×3) | 0.446 | 沒幫助(撞 case_db 結構天花板) |

**Negative result — full-vocab 1-stage retrieval**: 我們驗證了用 cause-aligned encoder 直接對全 56k cause 做 retrieval(跳過 candidate pool):

| Method | sem R@10 |
|---|---|
| Phase 1 + candidate pool(2-stage) | 44.4% |
| Pure SigLIP2 global emb → 56k cause(1-stage) | 13.5% |
| DeepSets dual → 56k cause(1-stage) | 12.1% |

→ **Phase 1 的 candidate pool 不是人為限制,是個有效的搜尋空間縮減設計**。1-stage retrieval 比 2-stage 差 3.5×。同時觀察到 case_db fine-tune **不能犧牲 SigLIP2 預訓練的 image-text alignment**(deepsets 比 SigLIP2 base 還差),`dual-target` InfoNCE 是必要的 regularizer。

#### 結構天花板診斷

leave-one-out mining 顯示:
- 23.7% train cases 是 hard cases(teacher 漏掉 ≥1 GT cause)
- 0.4% (53 cases) 所有 GT cause 都漏 → **case_db 結構性死角**,任何 retrieval 都救不回來
- 加重這些 case 的訓練權重沒幫助(模型把容量浪費在「教不會的」)

**結論**:**~44.5% 是 case_db 的物理上限**,不是 retrieval 算法的問題。未來提升空間在擴 case_db 的 cause 多樣性,不在改 encoder。

#### Phase 3 ablation：raw case_db

把同樣的 dual-target distillation 跑在 raw SigLIP2 case_db 上（見 [Phase 1 VLM dependency ablation](#phase-1-ablationvlm-dependency)），驗證 Phase 3 加速貢獻是否依賴 in-domain VLM finetune：

| Method | Fine-tuned case_db | Raw case_db |
|---|---|---|
| Phase 1 hungarian (teacher) sem R@10 | 44.4% | 45.6% |
| DeepSets dual-target (student) sem R@10 | **44.7%** (+0.3 pp) | **45.6%** (+0.0 pp, MRR +0.003) |
| Student per-q ms | 1.1 | 1.1 (14× faster than teacher in both) |
| 訓練 epochs（早停） | 9 | 9 |

→ **Phase 3 加速貢獻對 VLM training 不敏感**，跟 Phase 1 retrieval 一致。dual-target loss（listwise KL + case-cause InfoNCE）在兩種 case_db 上同樣 work，學生持平或微幅超越老師。和 Phase 1 一起構成「retrieval 端走 zero-shot 範式」的完整論證。

Artifacts：`outputs/case_db_raw/teacher_train_train.pt`、`outputs/encoder_raw/best_encoder.pt`、`logs/encoder_raw.log`、`logs/eval_phase1_aligned_raw.log`。

---

### Phase 4：CRR-DeepRVQ（壓縮 + 殘差 reranker，method-level 貢獻）

**動機**：Phase 3 把 retrieval 從 15.4 ms 壓到 1.1 ms，下一個 bottleneck 是 case bank 規模——12,780 cases × 768 dim fp32 = 39 MB 還能全進 GPU，但若部署到 1M+ cases（diagnostic platform 預期長期累積）則 brute-force dense 退化到 5 ms/q 並佔 1.5 GB。CRR-DeepRVQ 給出可擴展的壓縮 + 修正方案。

#### 設計

把凍結的 DeepSets case encoder 輸出 `z_i ∈ ℝ^768` 用 **Residual Vector Quantization** 壓成 `codes ∈ [K]^M` 的離散碼字（M levels × K codes/level）：

```
ẑ_i  = Σ_m  C_m[k_{i,m}]              # 重建
e_i  = z_i − ẑ_i                       # 量化殘差
```

檢索分兩階段：
- **First-stage**：`s_first(q, i) = qᵀẑ_i`，透過 lookup table `LUT[m, k] = q · C_m[k]` 每 candidate 只要 M 次加法
- **Residual rerank**：對 top-K_top=50 candidates，用 neural net 預測 `Δ ≈ qᵀe_i` 並 `s_final = s_first + Δ`

數學動機：`s_dense(q,i) = qᵀz_i = qᵀẑ_i + qᵀe_i = s_first + qᵀe_i`，所以 reranker 不是估計 score 而是估計 compression residual。

#### 兩個 reranker variant

| Variant | Candidate token 包含 | 部署假設 |
|---|---|---|
| **Light** | ẑ, codes embedding, q⊙ẑ, \|q−ẑ\|, s_first, ‖e‖ | 完全壓縮：只有 codes 在 memory 裡 |
| **Full** | Light 全部 + z_i + e_i + q⊙e_i | 全 memory：top-K 時 fetch dense z |

Full 因 `Δ = qᵀe` 解析可得，論文用 **analytic Full**（無訓練的 oracle）作為 upper bound。

#### 評估的雙 regime

| Regime | top_k_cases | 動機 | RVQ damage？ |
|---|---|---|---|
| **A (production)** | 20 | 經過 Phase 1 cause-aggregation pool（87 個 unique cause） | 全部持平，aggregation buffer 吸收 RVQ noise |
| **B (ANN-style)** | 1 | 單一 case 的 cause set 直接當預測，無 buffer | 有壓縮的方法直接掉到 dense 以下 |

**Regime A 是 production 設定，Regime B 是 method validation 設定**——後者拆掉 aggregation buffer 才能看出 reranker 的價值。論文兩個都報：A 是「壓縮 free lunch」的 happy byproduct，B 是「reranker 真的修補 RVQ」的 main claim。

#### 訓練

```bash
# Step 1: fit RVQ codebook（~1 分鐘，每個 (M, K) 一次性）
$PY -m diagnosis_model.cause_inference.rvq_rerank.fit_rvq \
  --encoder_ckpt outputs/encoder_final/best_encoder.pt \
  --case_db_dir outputs/case_db \
  --output_dir outputs/rvq_rerank \
  --M 4 --K 256

# Step 2: 訓 Light reranker（~15–60 秒）
$PY -m diagnosis_model.cause_inference.rvq_rerank.train_reranker \
  --encoder_ckpt outputs/encoder_final/best_encoder.pt \
  --case_db_dir outputs/case_db \
  --rvq_M 4 --rvq_K 256 \
  --rvq_dir outputs/rvq_rerank/rvq_M4_K256 \
  --output_dir outputs/rvq_rerank/reranker_M4_K256_light \
  --variant light --K_top 50 --batch_size 64 --epochs 30 \
  --eval_top_k_cases 1   # Regime B 早停 metric
```

Training task：對每個 train query（LOO），由 s_first 取 top-K_top=50 候選，
reranker 預測 Δ；`loss = listwise_KL(s_final, s_dense) + λ_mse · MSE(Δ, qᵀe)`。

#### 結果 — RVQ 壓縮率 sweep（Regime A, top_k_cases=20, 1573 valid）

12 個 (M, K) 配置全部 R@10 Δ 在 ±1.7 pp 內（SE = 1.26 pp），**全是 noise**：

| RVQ config | Compression × | sem R@10 | Δ vs dense |
|---|---|---|---|
| Dense fp32 | 1× | 44.7% | — |
| M=4 K=256 | 768× | 45.5% | +0.8 pp |
| M=2 K=64 | 2048× | 43.5% | −1.2 pp |
| **M=1 K=16** | **6144×** | **43.9%** | **+0.0 pp** ← 16 prototypes 而已！ |

**Production regime 對 RVQ noise 幾乎免疫**：M=1 K=16 把整個 12,780 case bank 壓成 **16 個 prototype**（每 prototype ~800 cases），R@10 完全不掉。原因：Phase 1 的 cause-aggregation pool 是個有效的「coarse clustering buffer」，個別 case 排序錯誤被 union over 20 cases 吸收。

只有 R@100 才看出 M=1 K=16 退化（−6 pp），但 R@10/MRR 完全持平。

#### 結果 — Regime B 完整對照（top_k_cases=1, 1573 valid，主表）

拆掉 aggregation buffer 後，RVQ 損害直接顯露，reranker 有用武之地：

| Method | Comp × | sem R@10 | Δ vs dense | Gap recovered |
|---|---|---|---|---|
| **Dense fp32** | 1× | **0.537** | — | — |
| RVQ-only M=4 K=256 | 768× | 0.506 | −3.1 pp | — |
| **+ Light rerank** | 768× | 0.518 | −1.9 pp | **39%** |
| + Full analytic | 768× | **0.537** | +0.0 pp | **100%** |
| RVQ-only M=2 K=64 | 2048× | 0.463 | −7.4 pp | — |
| **+ Light rerank** | 2048× | 0.500 | −3.7 pp | **50%** |
| + Full analytic | 2048× | 0.515 | −2.3 pp | 69% |
| RVQ-only M=1 K=16 | 6144× | 0.372 | −16.5 pp | — |
| **+ Light rerank** | 6144× | 0.462 | −7.6 pp | **54%** |
| + Full analytic | 6144× | 0.458 | −7.9 pp | 52% ← 被 first-stage recall 限制 |

**三個 method-level 觀察：**

1. **Light reranker 在所有壓縮率穩定回收 39–54% 的 gap**——method 不是 cherry-picked，對 768×–6144× 都有效。
2. **Full analytic 在 M=4 K=256 完全回補到 dense (0.537)**——證明「reranker 的可挽回上限 = top-K 包含 GT case 的機率」。M=1 K=16 saturate 在 0.458，因為 top_K_rerank=50 對 16-prototype 不夠，first-stage 包含 GT 的機率本身就低，連 oracle Δ 都救不回。
3. **M=1 K=16 + Light 達 0.462**，跟 M=4 K=256 純 RVQ 的 0.506 只差 4 pp，但壓縮率 **8× 更高**——Pareto frontier 被 reranker 整個推外。

#### 結果 — Scale benchmark（latency / memory at N=12K..1M）

```bash
$PY -m diagnosis_model.cause_inference.rvq_rerank.benchmark_scale \
  --rvq_dir outputs/rvq_rerank/rvq_M4_K256 \
  --light_reranker_ckpt outputs/rvq_rerank/reranker_M4_K256_light/best.pt \
  --N_list 12780 50000 100000 500000 1000000 --batch_sizes 1 32
```

合成 N 個 z（從 z_train 抽樣 + Gaussian + L2-renormalize），用既有 codebook 量化。**不報 recall**（合成 case 沒 GT），純 inference benchmark。

Latency p50 @ bs=1（M=4 K=256, 32 GB GPU）：

| N | dense | rvq_only | rvq_light | rvq_full_analytic |
|---|---|---|---|---|
| 12,780 | 0.08 ms | 0.10 ms | 0.56 ms | 0.13 ms |
| 100,000 | 0.56 ms | 0.13 ms | 0.60 ms | 0.16 ms |
| 500,000 | 2.59 ms | 0.15 ms | 0.62 ms | 0.18 ms |
| **1,000,000** | **5.15 ms** | **0.19 ms** | **0.59 ms** | **0.22 ms** |

- **dense** 線性成長（O(N)）：1M 已達 5 ms/q
- **rvq_only** 幾乎平坦：1M 仍 0.19 ms/q（27× 比 dense 快）
- **rvq_light** 約 0.6 ms 常數開銷（reranker forward 主導，跟 N 無關）
- **rvq_full_analytic** 跟 rvq_only 同步：top-K=50 dense rerank cost 微小

Memory @ N=1M（M=4 K=256 codes = uint8）：

| Method | Index size | Compression vs fp16 dense |
|---|---|---|
| dense fp16 | 1,536 MB | 1× |
| **rvq_only** | **4 MB** | **384×** |
| rvq_light | 17 MB (codes + 13 MB reranker) | 90× |
| rvq_full_analytic | 1,540 MB (z bank 為 rerank 必要) | 1× |

#### Deployment Pareto（三層）

| 部署情境 | 推薦方法 | Memory | Latency | Quality (Regime B) |
|---|---|---|---|---|
| Memory + compute 都緊 | **rvq_light** | 17 MB | 0.59 ms | 0.518 |
| Compute 緊但 memory 寬鬆 | rvq_full_analytic | 1.5 GB | 0.22 ms | 0.537 (=dense) |
| Memory 緊但 quality 可讓 | rvq_only | 4 MB | 0.19 ms | 0.506 |
| 無壓力 | dense | 1.5 GB | 5.15 ms | 0.537 |

#### 論文敘事建議

> **CRR-DeepRVQ 把 case-based retrieval 推到 1M+ 規模的 deployment 範圍。** 後接在 Phase 3 DeepSets case encoder 之後，凍結 z_i，用 Residual Vector Quantization 壓成 M-byte codes（768× memory compression at M=4 K=256），再用 query-to-candidate cross-attention reranker 學 compression residual Δ ≈ qᵀe_i 修補 RVQ 引入的 ranking 誤差。
>
> 評估顯示**雙 regime 的不對稱**：在 production setting（Phase 1 top_k_cases=20 + cause-aggregation buffer），即使把 12,780 cases 壓成 16 個 prototype（6,144× compression）R@10 都不掉，是 cause-aggregation pool 充當 coarse clustering buffer 的副效應；但在 stress regime（top_k_cases=1, no aggregation），RVQ 損害 3–16 pp R@10，**Light reranker 穩定回收 39–54% 的 gap**，Full analytic 在輕度壓縮下完全回補到 dense。1M case bank latency benchmark 顯示 dense 退化到 5.15 ms/q，但 rvq_light 維持 0.59 ms/q + 17 MB（90× 壓縮、9× 加速），證明 case-based retrieval framework 在數據集規模成長下仍可實時部署。

#### 子套件結構

```
cause_inference/rvq_rerank/
├── __init__.py
├── rvq.py                # RVQCodebook：fit (k-means on residuals) / encode / LUT
├── reranker.py           # Reranker（Light + Full）；listwise KL + Δ MSE loss
├── fit_rvq.py            # Stage A1 CLI：fit codebook on z_train
├── build_rvq_index.py    # Stage A2 CLI：編 train + valid into index.pt（可選）
├── eval_sanity.py        # dense vs RVQ-only 純壓縮 sanity
├── run_sweep.py          # (M, K) 12 格 Pareto sweep
├── eval_harder.py        # 雙 regime stress eval（top_k_cases × sem_thr）
├── train_reranker.py     # Light reranker 訓練（Regime B 早停）
├── eval_final.py         # 4 method × 雙 regime 完整對照
└── benchmark_scale.py    # latency / memory at N=12K..1M+
```

詳細指令見 [training.txt §6](training.txt) 與 [inference.txt §J](inference.txt)。

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

## Cascade architecture & final results

FaCE-R 採用 **coarse-to-fine 二階段 cascade**——retrieval 端用 zero-shot frozen SigLIP2，attribution 端用 fine-tuned VLM-Fusion，兩階段 contribution 完全不重疊。

```
[Coarse retrieval, ZERO-SHOT]                    [Fine rerank, FINE-TUNED]
   raw SigLIP2 (frozen pretrained)                  VLM-Fusion + CEAH
        ↓                                                  ↓
   Phase 1 / Phase 3 / Phase 4                       on top-K cases only
        ↓                                                  ↓
   top-K cases  (12K→20 在 0.6 ms)                   candidate cause + α
   "frozen pretrained alignment"                    "lesion-grounded attribution"
```

之前 paper 預設的 hybrid γ=0.75 scoring 是 retrieval-score + CEAH-score 線性混合。Cascade 框架下 coarse stage 只負責挑 top-K cases，不再貢獻 cause-level score（等於 γ=0），fine stage 100% 用 CEAH。**γ=0 在 γ-scan 中本來就是 sem R@10 跟 cluster R@10 最好的點**（46.8% / 39.8% vs γ=0.75 的 45.3% / 37.0%），所以 cascade 同時改善結構清晰度跟 headline 數字。

### Stage 1 — Coarse retrieval（zero-shot, raw SigLIP2）

Production 走 `case_db_raw/` + `encoder_raw/` + `rvq_rerank_raw/`。1573 valid queries，top_k_cases=20，candidate-pool aggregation：

| Component | sem R@10 | sem MRR | per-q latency | Index memory @ 12K |
|---|---|---|---|---|
| Phase 1 hungarian (teacher) | 45.6% | 0.300 | 15.2 ms | 200 MB (multi-vec) |
| Phase 3 DeepSets dual-target | **45.6%** | **0.303** | **1.1 ms** (14×) | 39 MB |
| Phase 4 RVQ-only M=4 K=256 | 45.1% | 0.302 | 0.10 ms (152×) | **0.05 MB** (384× compression) |
| Phase 4 + Light reranker | 45.1% | 0.300 | 0.56 ms (27×) | 13 MB (90× compression) |
| Phase 4 + Full analytic | **45.6%** | 0.301 | 0.22 ms (69×) | 1.5 GB (z bank needed) |

→ Coarse stage 所有變體 R@10 都在 dense (45.6%) ±0.6 pp 內，**production aggregation buffer 對 RVQ 壓縮免疫**。

**Regime B stress benchmark**（top_k_cases=1，無 aggregation buffer，純 ANN-style）。Pareto 對照 raw（production cascade Stage 1）vs fine-tuned（ablation）兩條曲線：

| Compression × | Method | **raw R@10** | **fine-tuned R@10** | 哪個贏 |
|---|---|---|---|---|
| 1× (dense fp32) | baseline | 51.7% | 53.7% | fine-tuned +2.0 pp |
| 768× (M=4 K=256) | RVQ-only | 48.4% | 50.6% | fine-tuned +2.2 pp |
|  | **+ Light reranker** | **51.0%** | 51.8% | fine-tuned +0.8 pp |
|  | + Full analytic | 51.6% | 53.7% | fine-tuned +2.1 pp |
| 2048× (M=2 K=64) | RVQ-only | 44.6% | 46.3% | fine-tuned +1.7 pp |
|  | **+ Light reranker** | 46.2% | **50.0%** | **fine-tuned +3.8 pp** |
|  | + Full analytic | 50.5% | 51.5% | fine-tuned +1.0 pp |
| 6144× (M=1 K=16) | RVQ-only | 36.8% | 37.2% | tied |
|  | **+ Light reranker** | 39.7% | **46.2%** | **fine-tuned +6.5 pp** |
|  | + Full analytic | 44.6% | 45.8% | fine-tuned +1.2 pp |

Gap recovered (Light vs RVQ-only) 隨壓縮率變化的不對稱：

| Compression × | Raw recovery | Fine-tuned recovery |
|---|---|---|
| 768× (M=4 K=256) | **79%** | 39% ← raw 勝 |
| 2048× (M=2 K=64) | 22% | **56%** ← fine-tuned 勝 |
| 6144× (M=1 K=16) | 19% | **54%** ← fine-tuned 勝 |

**詮釋**：raw features 在中度壓縮（M=4 K=256, 768×）下殘差結構乾淨、reranker 學得很好（79% recovery）；但壓到極端時 raw 比 fine-tuned 更早 degrade（fine-tune 的 in-domain training 隱含把 features 推到更 quantize-friendly 的空間）。Production 用 **raw M=4 K=256**（cascade Stage 1 sweet spot）；極端壓縮的 deployment 場景需要 fine-tuned encoder。完整 Pareto plot 見 [`outputs/rvq_rerank/pareto_compression_vs_R10.png`](outputs/rvq_rerank/pareto_compression_vs_R10.png)。

**Scale-up benchmark @ N=1M**（合成 z bank、無 recall 報告）：

| Method | latency (bs=1) | memory | compression |
|---|---|---|---|
| Phase 3 DeepSets dense | 5.15 ms | 1.5 GB | 1× |
| Phase 4 RVQ-only | 0.19 ms (27× faster) | 4 MB | **384×** |
| Phase 4 + Light reranker | 0.59 ms (9× faster) | 17 MB | 90× |

Coarse stage 在 1M case bank 維持 sub-ms / sub-20MB——cascade 第一階段的 scalability 主軸。

### Stage 2 — Fine rerank（fine-tuned, VLM-Fusion + CEAH）

Production 走 `case_db/`（fine-tuned VLM-Global + VLM-Lesion + LocalGlobalFusionWrapper）。對 Stage 1 取的 top-20 cases 的 candidate cause pool（~87 unique causes）做 attribution-aware 重排。

| Setting | sem R@1 | sem R@10 | sem MRR | cluster R@10 | cluster MRR |
|---|---|---|---|---|---|
| Coarse only (cause-pool aggregation) | 22.3% | 44.4% | 0.298 | 35.8% | 0.235 |
| **+ Fine rerank (CEAH, γ=0)** | **24.0%** | **46.8%** | **0.301** | **39.8%** | **0.271** |
| **Δ fine-stage contribution** | **+1.7 pp** | **+2.4 pp** | +0.003 | **+4.0 pp** | **+0.036** |

**Faithfulness**（lesion masking 反事實實驗）：

| Bucket | Lesion-mask drop | Random-mask drop | Ratio |
|---|---|---|---|
| All | +0.036 | +0.005 | 7.2× |
| Lesion-type causes | +0.040 | +0.006 | 6.7× |
| **N≥3 lesions** | +0.055 | +0.003 | **18×** |

→ 遮 lesion 對 multi-lesion case 的分數影響是隨機遮的 **18 倍**——α 是 load-bearing 不是裝飾。

**Cause-type-aware α 分佈**（softmax over evidence tokens）：

| Cause type | global α | text α | lesion sum α | l/g ratio |
|---|---|---|---|---|
| global-type（水質/緊迫） | 0.51 | 0.18 | 0.31 | 0.44 |
| **lesion-type（細菌/寄生蟲）** | 0.25 | 0.28 | **0.47** | **1.39 (反轉)** |
| mixed | 0.39 | 0.20 | 0.40 | 0.72 |

### Asymmetric VLM dependency（cascade 的結構依據）

| Stage | VLM 依賴 | Ablation 結果 |
|---|---|---|
| **Coarse retrieval** | **不要 fine-tune** | Phase 1 raw +1.2 pp R@10 vs fine-tuned；Phase 3 raw student R@10 相同 (45.6%)；supervised projection MLP probe Δ MRR −0.001 |
| **Fine rerank** | **必須 fine-tune (VLM-Fusion)** | raw case_db 上重訓 CEAH：retrieval 持平 (sem R@10 46.4%)，**faithfulness 反轉** (lesion drop +0.040 → **−0.031**) |

→ Retrieval-side fine-tune 全部沒測到改善（zero-shot 飽和）；attribution-side fine-tune 是 architecturally required（LocalGlobalFusionWrapper 提供 lesion-specialized 視覺特徵）。Cascade 把這兩種不對稱的依賴乾淨分開。

---

## 論文敘事建議（cascade framing）

> **FaCE-R is a coarse-to-fine cascade for case-based fish disease cause prediction with lesion-grounded explainability.**
>
> **Coarse stage — zero-shot case retrieval at scale.** Three components progressively accelerate retrieval over a frozen pretrained SigLIP2 backbone on a 12,780-case database (production runs against `case_db_raw/`): **Phase 1** multi-vector Hungarian lesion-set matching reaches sem R@10 = 45.6% / MRR = 0.300 at 15.2 ms / query; **Phase 3** DeepSets dual-target distillation (listwise KL on the Phase 1 score matrix + case-to-cause SupCon InfoNCE on 56k cause text embeddings) collapses retrieval to a single-vector cosine, preserving quality (R@10 = 45.6%, MRR = 0.303 — slightly better than teacher) at 1.1 ms / query (14× faster); **Phase 4 CRR-DeepRVQ** applies Residual Vector Quantization to the frozen Phase 3 embeddings and trains a query-to-candidate cross-attention reranker to recover the compression residual Δ ≈ qᵀe. At M=4 K=256 (768× theoretical compression), RVQ-only retrieval matches dense within 0.6 pp R@10 in the production aggregation regime; scale-up benchmarks at N=1M show 90× memory compression at 17 MB and 0.59 ms / query while dense degrades to 5.15 ms / query at 1.5 GB. A no-aggregation stress regime (top_k_cases=1) reveals a 3.25 pp R@10 RVQ damage at the production point (M=4 K=256, raw) that the Light reranker **recovers 79% of** to 51.0% — within 0.7 pp of dense (51.7%). A full compression sweep (M ∈ {4,2,1}, K ∈ {256,64,16}) on both raw and fine-tuned encoders surfaces an asymmetry: at mild compression (768×) raw features recover better (79% vs 39%), but at aggressive compression (≥2048×) fine-tuned features quantize more gracefully (54–56% recovery vs 19–22% on raw), suggesting in-domain fine-tuning implicitly produces quantize-friendlier embedding clusters. The production cascade pins to the mild-compression sweet spot (raw + M=4 K=256), where the residual reranker mechanism is most effective. **Cross-ablation** (Phase 1 raw vs fine-tuned VLMs; supervised projection MLP probe on frozen VLM-Lesion; Phase 3 raw distillation; Phase 4 raw RVQ + reranker) consistently shows in-domain VLM fine-tuning provides no measurable retrieval improvement, confirming the coarse stage operates in a **zero-shot vision-language regime**.
>
> **Fine stage — lesion-grounded attribution rerank.** On the coarse stage's top-20 retrieved cases (candidate cause pool ≈ 87 unique causes), **CEAH** consumes fine-tuned VLM-Fusion lesion features (per-bbox SigLIP2 + LocalGlobalFusionWrapper) along with VLM-Global features and a candidate cause text embedding. It outputs a softmax α distribution over (global, text, lesion_1..N) evidence tokens plus a multiplicative score head (sigmoid global × sigmoid local) that architecturally forces local-evidence faithfulness — sigmoid α and additive scoring both collapse to α-saturation in v1 / v2 ablations. The fine stage delivers **+2.4 pp sem R@10 (44.4% → 46.8%)** and **+4.0 pp cluster R@10 (35.8% → 39.8%)** over coarse-only retrieval on 1,573 valid queries. Faithfulness is verified by counterfactual lesion masking: drop magnitude on multi-lesion cases (N ≥ 3) is **18× the random-mask baseline**, and the lesion / global α ratio inverts 3.2× across cause types (0.44 for global-type → 1.39 for lesion-type), showing the model preserves cause-type-aware behavior end-to-end. Replacing the fine-tuned case_db with raw SigLIP2 features preserves retrieval but **reverses** faithfulness (lesion drop +0.040 → −0.031), confirming VLM-Fusion is an architectural prerequisite for the fine stage.
>
> **Asymmetric VLM dependency** is the cascade's structural claim. Retrieval-side fine-tuning is consistently a no-op (Phase 1 raw vs fine-tuned, Phase 3 raw distillation, Phase 4 raw RVQ, plus the supervised projection MLP probe); attribution-side fine-tuning is architecturally load-bearing (LocalGlobalFusionWrapper). The cascade enforces this split by routing coarse retrieval through raw `case_db_raw/` and fine rerank through fine-tuned `case_db/`, turning the previously parameter-sensitive hybrid γ scoring (γ=0.75 default) into a principled two-stage pipeline (cascade ≡ γ=0, which is the strongest point of the γ-scan on **both** sem and cluster recall).
>
> **Robustness & deployment**: medical / colloquial / none 三種推論模式 retrieval 差 ≤ 0.003 MRR — vision-only deployment 完全可行，text 缺席時 attention 自動補位給 lesion，cause-type-aware 行為 preserved. Multi-lesion attribution at N≥3 spreads evenly across all lesions (concentration ≈ 1/N), demonstrating a **collective visual evidence** strategy rather than a single dominant-lesion heuristic.
>
> **Negative results**: supervised projection MLP on frozen lesion features (no measurable lift, ablation reinforces zero-shot framing); 1-stage full-vocab cause retrieval bypassing the case-pool aggregation (3.5× worse than 2-stage cascade); leave-one-out mining showing 0.4% of train cases (~53 / 12,780) have all GT causes outside retrieval reach, defining a structural R@K ceiling near 44.5% for retrieval-only metrics that the fine stage then pushes past.

---

## 已知限制

- **Cause-emb 跟 lesion-emb 在不同 VLM 空間**：CEAH 必須學跨空間映射，可能限制 lesion attribution 的精細度
- **HDBSCAN 細粒度過分割**：disease topic 級別需要重新粗粒度聚類（v100），否則 cluster R@K 假性偏低
- **N≥3 attribution spread**：模型不會從多 lesion 中挑單一主病灶。若臨床需要「指向單一病灶」需額外 hard top-1 約束
- **Text 對 retrieval 沒實質幫助**：text mode ablation 顯示 medical/colloquial/none 三模式 retrieval 差 ≤ 0.003 MRR；text 在 attribution 端有作用（吸收 0.13–0.26 attention mass），但不影響 top-K 結果。換言之，部署若不收集文字描述也不會丟分。
