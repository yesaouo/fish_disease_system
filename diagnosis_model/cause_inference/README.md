# FaCE-R: Faithful Case-guided Cause-Evidence Reasoning

魚病病因推論系統的第三階段——在已訓練的 RF-DETR 病灶偵測器與兩個 SigLIP2 多模態模型上，做 case-based 病因檢索 + 架構強制的 lesion-grounded 可解釋性。

## 動機

魚病診斷的病因標註是 **LLM 生成的自由文字並經由水產專業人員審核**（每張圖約 4 個病因，總共 56,310 個 unique 字串，**94.7% 是 singleton**）。這個資料特性使得：

1. 傳統的 closed-vocabulary multi-label classification **無法應用**
2. 為了符合實際的診斷應用需求，系統必須具備「**lesion-grounded explainability**」——預測必須能對應到具體病灶

FaCE-R 把問題重構成 **case-based 檢索 + lesion-attribution**。論文層級上，整個系統 = **兩個 model** + 中間一層**非參數 case bank**，流程 **GRACE（Case Encoding）→ case bank 檢索 → CEAM（Cause Inference）**：

- **GRACE**（*Grounded Region Aggregated Case Encoder*，part 1）：image(+text) → 單一 **Matryoshka case 向量** + 結構化 findings（box / anomaly / lesion）。一個 model、一次 forward（子模組分階段訓練、SigLIP2 凍結）。內含兩個貢獻——**GROD**（grounding 前端：偵測 + semantic head，faithfulness-by-construction）與 **Aggregator**（pooling 後端：set→case 向量，對外稱 Aggregator，DeepSets 實作）。
- **case bank 檢索**（兩 model 之間，**非參數**＝case-based 記憶，不是第三個 model）：case 向量 → top-k cases → 候選病因池 + 檢索 prior。
- **CEAM**（part 2）：step 2 唯一的學習元件，與檢索 prior 經 γ 融合 → faithful 病因排序 + α。（γ 隨資料集翻、非調參：fish γ=0 純 CEAM、DDXPlus γ=0.75 prior 扛大半，別讓「model 2 = CEAM」蓋掉 prior 份量。）

**三個掛牌貢獻 = GROD / ABQ / CEAM**。其中 **ABQ** 已從「模組」改述為「Aggregator 輸出向量的壓縮-robustness 性質」：該向量經 Matryoshka 維度截斷或 RVQ 量化都不掉，因為檢索時的 aggregation buffer 吸收損害（scaling law `D≈c·ε/K^{0.84}`）——是跨介面 claim，不是架構圖上的 box；RVQ 是部署期壓縮實作。Matryoshka 同樣是輸出向量的*性質*（nested 前綴 + 各自 L2-norm），不是一層。

> 程式仍以 Phase 0–4 + [`../grod/`](../grod/) 組織：GRACE = `grod/`(GROD) + Phase 3 Aggregator（`train_case_encoder.py`, DeepSets）的論文級合稱，CEAM = Phase 2；**Matryoshka 輸出尚未實作（目標）**，目前 Phase 3 仍吐 dense z。

下面的 Phase 流程是 offline/online 的實作對照，兩階段對 VLM 的依賴不對稱：

```
Query 影像
  ↓
RF-DETR 偵測病灶（前一階段）
  ↓
SigLIP2 抽特徵（frozen teacher，預訓練即可；Phase 2 額外需要 fine-tune 過的 fusion VLM-Lesion）
  ↓
[Phase 1] zero-shot case-based cause retrieval（案例式病因檢索）→ 候選病因池（~87 個）
  ↓
[Phase 2] CEAM 重新打分 + 病灶歸因                    ← 需 VLM-Lesion fusion 才有 faithful attribution
  ↓
Top-N 病因預測 + α attribution 視覺化
```

**設計哲學**：retrieval 端走 zero-shot vision-language 的範式（不微調 VLM），attribution 端走 lightweight head + 必要的 lesion-specialized 視覺特徵。Ablation 顯示兩者的依賴不能互換（[VLM dependency ablation](#phase-1-ablationvlm-dependency)）。

## Offline 訓練 vs Online 推論

整個系統分兩個時期。**Offline** 訓模型 + 建索引（各跑一次）；**Online** 對單張新影像做 coarse-to-fine cascade 推論。最常見的誤解：**Phase 3 / Phase 4 是 offline 訓練步驟**，online 只是*載入*它們產出的 encoder / RVQ codebook 來檢索。

**OFFLINE（訓練 + 建索引）**
1. 訓 RF-DETR 病灶偵測器（[`../detection/`](../detection/)）
2. `vl_classifier` 訓 VLM-Global 與 VLM-Lesion（病灶 crop encoder + LocalGlobalFusionWrapper），語料是 overall 描述 / per-bbox 症狀 caption，**與檢索分數無關**
3. 建兩份索引：`case_db_raw/`（raw SigLIP2，coarse 用）+ `case_db/`（fine-tuned，CEAM 用）
4. raw SigLIP2 的 Phase 1 檢索分數＝**teacher** → 蒸餾出 **Phase 3** case encoder（student，`encoder_raw/`）
5. **Phase 4**：在 Phase 3 凍結的 z 上 fit RVQ codebook，並可選訓練 compressed residual reranker（Light implementation；raw 路）
6. 在 fine `case_db/` 上訓 **CEAM**（用 Phase 1 candidate pool 的 hard negatives）

**ONLINE（單張影像推論，見 [inference.txt](inference.txt) `[H]`）**
1. RF-DETR 偵測病灶 bbox
2. raw SigLIP2 編碼 → **Stage 1 coarse**：Phase 3 `encoder_raw` + 選配 Phase 4 RVQ → top-K cases
3. union top-K 的病因 → candidate cause pool（~87 個）
4. VLM-Global + VLM-Lesion 編碼 → **Stage 2 fine**：CEAM 重排 → cause + α
5. 輸出：VLM-Lesion 病灶分類 + CEAM 病因排序與 α attribution

兩條支路是**平行的非對稱依賴**——coarse retrieval（Phase 1/3/4）走 raw SigLIP2、不微調（teacher / student 都是 raw）；fine attribution（Phase 2 CEAM）**必須**用 fine-tuned VLM-Lesion（raw 上 faithfulness 反轉）。詳見 [Cascade architecture & final results](#cascade-architecture--final-results)。

## 上游依賴

FaCE-R 的兩個階段對 VLM 的依賴**不對稱**：

- **Phase 1（zero-shot case-based cause retrieval）**：把 SigLIP2 當 **frozen pretrained teacher**，只用其原始 image-text alignment 產生候選病因池。Ablation 顯示原始 `google/siglip2-base-patch16-224` 跟 fine-tune 過的 VLM-Global 在 retrieval 上**等價甚至略勝**（sem R@10 45.6% vs 44.4%），所以這個階段不需要任何 in-domain 微調。
- **Phase 2（CEAM lesion attribution）**：**必須**用 fine-tune 過的 VLM-Lesion + LocalGlobalFusion。raw SigLIP2 的 lesion crop 特徵雖然能驅動 retrieval，但無法支撐 faithful attribution（lesion masking 結果反轉，見 [VLM dependency ablation](#phase-1-ablationvlm-dependency)）。

| 元件 | 路徑 | 訓練語料 | 用於哪個階段 |
|---|---|---|---|
| **RF-DETR 病灶偵測器** | [`../detection/`](../detection/) | COCO bbox | 推論前處理 |
| **SigLIP2 base (frozen)** | `google/siglip2-base-patch16-224` | 原始預訓練 | **Phase 1 retrieval 的 zero-shot baseline** |
| **VLM-Global**（整圖 ↔ 整體描述） | [`../vl_classifier/outputs/siglip2_base_patch16_224_overall_multipos_zh`](../vl_classifier/outputs/siglip2_base_patch16_224_overall_multipos_zh) | `overall.medical_zh` / `colloquial_zh` | Phase 1（與 raw 等價）、Phase 2 |
| **VLM-Lesion**（病灶 crop ↔ 症狀描述，含 fusion wrapper） | [`../vl_classifier/outputs/siglip2_base_patch16_224_multipos_fusion_en_zh`](../vl_classifier/outputs/siglip2_base_patch16_224_multipos_fusion_en_zh) | per-bbox symptom captions | **Phase 2 CEAM faithfulness 必要** |
| **COCO 標註** | `data/detection/coco/_merged/{train,valid,test}/_annotations.coco.json` | image-level + per-bbox annotations | 全程 |

所有 VLM 在 cause_inference 內皆凍結，不進一步 finetune。Production 是 coarse-to-fine cascade，兩個階段走**不同**索引：**coarse 檢索（Phase 1/3/4）走 raw SigLIP2 建的 `case_db_raw/`**，**fine attribution（Phase 2 CEAM）走 fine-tune 過的兩個 VLM 建的 `case_db/`**。兩份索引共享 case / cause 對應，差別只在 image feature 是否經過 fine-tune（詳見 [Cascade architecture & final results](#cascade-architecture--final-results)）。

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
│   └── ceah.py                    ← Phase 2 模型（CEAM）
├── phase1_baseline.py             ← Phase 1 zero-shot case-based cause retrieval 評估
├── train_projection.py            ← supervised projection MLP 訓練（retrieval-side ablation）
├── train_ceah.py                  ← Phase 2 CEAM 訓練
├── eval_ceah.py                   ← Phase 2 推論 + α 輸出
├── eval_ceah_compressed.py        ← 端到端：壓縮 coarse → CEAM（compressed residual reranker 多餘性驗證）
├── faithfulness_eval.py           ← Faithfulness 驗證（lesion masking）
├── analyze_lesion_buckets.py      ← Phase 1 N-lesion 分桶分析
├── analyze_v3_n_buckets.py        ← Phase 2 v3 N-lesion 分桶分析
├── case_study_viz.py              ← 案例可視化（論文 figure）
├── run_phase1_sweep.sh            ← Phase 1 hyperparameter sweep
├── mamba_ablation/                ← Phase 3 Mamba 變體（architecture ablation）
├── rvq_rerank/                    ← Phase 4 case-bank 壓縮（codebook/RVQ + residual correction 對照）
│   ├── rvq.py                     ← RVQCodebook 模組
│   ├── reranker.py                ← compressed residual reranker / dense-residual oracle variants
│   ├── fit_rvq.py / build_rvq_index.py / run_sweep.py
│   ├── eval_sanity.py / eval_harder.py / eval_final.py
│   ├── train_reranker.py / benchmark_scale.py
│   ├── weighted_rvq.py            ← WeightedRVQCodebook（importance-weighted k-means）
│   ├── compute_rvq_weights.py     ← per-case w_ranking / w_agg / w_agg_inv
│   ├── fit_rvq_variants.py        ← fit 4 variants × 多 (M,K) 的 codebook
│   ├── eval_absorption_surface.py ← top_k × RVQ-config 吸收曲面 sweep
│   └── outputs/                    ← codebooks、index、residual-correction checkpoints
└── outputs/                       ← 所有產出
    ├── case_db/                   ← Phase 0 fine（fine-tuned VLM；Stage 2 CEAM。含 train_candidate_pool.pt）
    ├── case_db_raw/               ← Phase 0 raw（raw SigLIP2；★ Stage 1 coarse retrieval，production）
    ├── encoder_raw/               ← Phase 3 case encoder（蒸餾自 raw teacher；★ production coarse）
    ├── encoder_final/             ← Phase 3 case encoder（fine case_db；ablation 對照）
    ├── rvq_rerank_raw/            ← Phase 4 codebook + optional compressed residual reranker（raw；★ production coarse）
    ├── rvq_rerank/                ← Phase 4 codebook + optional compressed residual reranker（fine；ablation 對照）
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
    ├── cause_clusters_llm.json               ← 466 LLM topics（★ 論文主指標）
    ├── cause_clusters_hdbscan.json           ← 100 HDBSCAN clusters（論文 LLM-free baseline）
    └── _archive_clusters/                    ← 中間/歷史 cluster JSON（不在 paper pipeline）

★ = production cascade 走 raw 路：coarse retrieval 用 `case_db_raw` / `encoder_raw` /
`rvq_rerank_raw`，fine attribution 才用 `case_db`。raw 不是 ablation，是 production 索引。
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

**Stage-2 / CEAM case_db 使用 fine-tune 過的兩個 VLM**（CEAM 對 VLM-Lesion fusion 的依賴是必要的，見 [VLM dependency ablation](#phase-1-ablationvlm-dependency)）：

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

`build_case_database.py` 同時支援 `--raw_lesion` flag 來建立 **Stage-1 production coarse index**（lesion crop 走原生 image encoder，跳過 LocalGlobalFusion）。同一份 raw index 也用於 [Phase 1 ablation](#phase-1-ablationvlm-dependency)，因此 `case_db_raw/` 不是單純對照版本，而是 production coarse path 的主要索引。

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

### Phase 0.5：Cause taxonomy

病因字串聚成 disease topic 是 **evaluation 用 lenient recall taxonomy**(cluster R@K / cluster MRR), **不進任何訓練 loss**(`train_ceah.py` / `train_case_encoder.py` 內 0 引用)。論文兩條 taxonomy:

| Taxonomy | clusters | 產生方式 | 用途 |
|---|---|---|---|
| **`cause_clusters_llm.json`** | **466** | LLM (Ollama gemma4:26b) | **論文主**(disease topic) |
| `cause_clusters_hdbscan.json` | 100 | HDBSCAN(EOM, mcs=100) + reassign-singletons | **論文 LLM-free baseline** |

eval 指令(`[A]` / `[B]` / `[E]` / `[G]`)的 `--cluster_json` 預設指向 LLM main。完整 baseline 對照與 cluster R@K 數字見 [paper_tables.md Table E1](paper_tables.md#e-cause-taxonomy-對照)(LLM 39.8% vs HDBSCAN 19.7% cl R@10)。

兩條 taxonomy 與其建立步驟集中在 [`preprocessing/README.md`](preprocessing/README.md):

```bash
# Paper main(LLM, 數小時, 需 Ollama)
$PY -m diagnosis_model.cause_inference.preprocessing.cause_cluster_llm \
  --cache_dir outputs/cause_cache \
  --output outputs/cause_clusters_llm.json \
  --batch_size 10 --shard_size 100 --merge_batch_size 20

# Paper baseline(HDBSCAN, 秒級, 固定論文參數)
$PY -m diagnosis_model.cause_inference.preprocessing.recluster_causes baseline \
  --cache_dir outputs/cause_cache
  # 輸出: outputs/cause_clusters_hdbscan.json
```

`baseline` 子命令 = `reduce → cluster(eom, mcs=100) → reassign-singletons(0.70)`,單一封裝、無可調參數。`recluster_causes.py` 另有 `cluster` / `sweep` / `merge-similar-clusters` 等 dev tools,但**不出現在論文 pipeline**。

---

### Phase 1：zero-shot case-based cause retrieval（candidate generation baseline）

> **這是 coarse 階段的檢索原理**：檢索相似 case → 取其病因併成候選池。它建立在標準想法上（case-based retrieval、multi-vector matching），論文以描述性的「case-based cause retrieval」稱呼、**不另立縮寫**。整個系統在框架 **FaCE-R**（Faithful Case-guided Cause-Evidence Reasoning）之下有**兩個掛牌貢獻**：coarse 階段的 **[ABQ](#phase-4-abq-aggregation-buffered-quantization)**（Aggregation-Buffered Quantization，Phase 3+4：可部署 case 檢索 + aggregation 吸收壓縮的 scaling law）與 fine 階段的 **[CEAM](#phase-2ceam)**（lesion-grounded attribution）。Phase 1 檢索原理是兩者共同的 substrate，描述性帶過。

對每個 valid query：

1. 跟所有 train case 算組合相似度
   - `sim(q, c) = α · cos(q.global, c.global) + β · lesion-set cosine`
   - 預設 α=0.25, β=0.75；lesion-set 預設 **max_mean** matching（2026-05-26 從 hungarian 翻成 max_mean，向量化 GPU `scatter_reduce`、~90× 快；fish 上 R@K/MRR 全部 |Δ| ≤ 0.5pp 且 max_mean 微勝，見 [ablations/lesion_match_ranking_equiv](ablations/lesion_match_ranking_equiv.py)；hungarian 只在重現 paper 舊數字或 train bank ≤ ~10k 時用）
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
  --lesion_match max_mean \
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
  --lesion_match max_mean --diversify_threshold 0.95 --semantic_threshold 0.95
```

> 註：下方表中 hungarian 標的數字是 paper 報的原值（4.4% sem R@10）；2026-05-26 之後 default 為 max_mean，max_mean 在同樣 case_db_raw 上 sem R@10 = **46.05%**（hungarian = 45.62%），全 metric |Δ| ≤ 0.5pp 且 max_mean 微勝（見 [ablations/lesion_match_ranking_equiv/metrics.json](outputs/ablations/lesion_match_ranking_equiv/metrics.json)）。

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

**但 CEAM 不適用**：把 raw case_db 接上 CEAM 訓練 + faithfulness eval：

| Faithfulness | Fine-tuned case_db | Raw case_db |
|---|---|---|
| `no_lesion` drop（lesion-type） | **+0.038** | **−0.031** ← 反轉 |
| `no_lesion` drop（all） | +0.035 | −0.024 |
| `no_random` drop | +0.003 | +0.004 |

→ raw 的 CEAM 把 attention 放在 lesion 上，但**遮掉 lesion 後分數反而上升** — 即 lesion 內容不是有效的判別訊號。fine-tune 過的 VLM-Lesion（含 LocalGlobalFusion）對 Phase 2 的 architecture-enforced lesion-grounded explainability 是**必要**的。

詳見 [Phase 2 design notes](#phase-2ceam) 對 VLM-Lesion 必要性的說明。

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

這個 probe 不是 pipeline 的一階,**下游沒有任何階段載入 `best_lesion_head.pt`**——Phase 2 CEAM 自帶 attribution MLP、Phase 3 DeepSets 的 φ 本身就是 per-lesion 的可學 projection(且訓練目標更強:listwise KL + SupCon InfoNCE),Phase 4 RVQ + reranker 坐在 Phase 3 的 z 上。Probe 的價值純粹是 retrieval-side fine-tune ablation 的**一個獨立資料點**,與「Phase 1 raw vs fine-tuned VLMs」、「Phase 3 raw distillation」、「Phase 4 raw RVQ + compressed residual reranker」共同支持「coarse retrieval 在 zero-shot vision-language regime 下飽和」的論述。

**論文寫法(ablation)**:「In addition to swapping in the raw SigLIP2 backbone, we also probe whether a cause-overlap supervised projection MLP on top of frozen VLM-Lesion can lift retrieval. Trained on pairwise lesion-set similarity aligned to pairwise cause-text similarity, the head produces no measurable improvement (Δ sem MRR = −0.001), consistent with the broader cross-ablation showing in-domain fine-tuning is a no-op for the coarse stage.」

---

### Phase 2：CEAM

CEAM = **Cause-Evidence Attribution Module**，是 fine 階段的掛牌貢獻，跟 coarse 階段的 [ABQ](#phase-4-abq-aggregation-buffered-quantization) 並列為框架 FaCE-R 下的**兩個掛牌貢獻之一**。對 Phase 1 候選池中的每個候選病因，輸出：
- 一個機率分數
- 一個對 evidence tokens（global / text / lesion 1..N）的 **softmax attribution** α

設計關鍵：
1. **softmax α**（不是獨立 sigmoid）→ α 加總 = 1，global 不能獨吞所有 attention
2. **multiplicative scoring**：`score = sigmoid(MLP_g([c, gated_global])) · sigmoid(MLP_l([c, gated_local]))` → global / local 必須各自對才能高分
3. **Architectural faithfulness**：`gated_pool = Σ α_i · e_i`，α_i = 0 時對應 evidence 真的不貢獻
4. **Lesion encoder 必須 fine-tune（含 LocalGlobalFusion）**：在 raw SigLIP2 的 case_db 上重訓 CEAM 顯示 retrieval 持平（sem R@10 46.4% vs 45.3%），但 faithfulness 完全反轉（`no_lesion` drop 從 +0.038 變 −0.031）。raw lesion crop 特徵不足以驅動架構強制的 lesion-grounded explainability — CEAM 雖把 attention 放在 lesion 上，但 lesion 內容不是有效訊號。詳見 [Phase 1 ablation：VLM dependency](#phase-1-ablationvlm-dependency)。

#### Phase 2a：建 hard-negative candidate pool

對每個 train case 跑 leave-one-out Phase 1 retrieval，把候選池的正負例離線存好（避免訓練時每 step 跑檢索）：

```bash
$PY -m diagnosis_model.cause_inference.preprocessing.build_train_candidate_pool \
  --case_db_dir outputs/case_db \
  --output_path outputs/case_db/train_candidate_pool.pt \
  --top_k_cases 20
```

產出（172 MB）：每個 train case 對應 87 個候選 cause indices + positive_mask（`semantic_threshold=0.95` 標記）。執行時間 ~3 min。

#### Phase 2b：訓練 CEAM

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

#### Phase 2c：CEAM inference + α 輸出

```bash
$PY -m diagnosis_model.cause_inference.eval_ceah \
  --case_db_dir outputs/case_db \
  --ceah_ckpt outputs/ceah_v3/best_ceah.pt \
  --output_dir outputs/ceah_v3_eval_full \
  --attribution_mode softmax --scoring_mode multiplicative \
  --gammas 0.0 --dump_gamma 0.0 \
  --cluster_json outputs/cause_clusters_llm.json
```

Cascade 下 coarse stage 只負責挑 top-K cases，cause-level 打分**完全由 CEAM 負責**——CEAM 是 fine 階段的方法，不是拿來跟檢索分數線性混合的修正項。最終 production 數字見 [Stage 2 — Fine rerank](#stage-2--fine-rerankfine-tuned-vlm-lesion--ceam)。（早期曾試 `γ·phase1 + (1−γ)·ceah` 的 hybrid 混合，contribution-breakdown ablation 顯示純 CEAM 最強，該 γ-scan 留在 [inference.txt](inference.txt) `[B]` 當佐證。）

---

### Faithfulness 驗證

對每個 valid query 的 top-1 預測，依次遮掉 global / text / lesions / top-α token，量分數變化：

```bash
$PY -m diagnosis_model.cause_inference.faithfulness_eval \
  --case_db_dir outputs/case_db \
  --ceah_ckpt outputs/ceah_v3/best_ceah.pt \
  --output_dir outputs/ceah_v3_faithfulness_pureceah \
  --attribution_mode softmax --scoring_mode multiplicative \
  --gamma 0.0 --max_queries -1
```

**v3 結果（按病因類型分桶）：**

| condition | global-type | lesion-type | mixed |
|---|---|---|---|
| no_lesion drop | +0.029 | **+0.038** | +0.041 |
| no_random drop | +0.004 | +0.004 | −0.001 |
| **lesion vs random ratio** | 7.4× | **9.7×** | random≈0 |

**按 lesion 數量分桶：**

| condition | N=1 | N=2 | N≥3 |
|---|---|---|---|
| no_lesion drop | 0.024 | 0.047 | **0.053** |
| no_random drop | 0.002 | 0.004 | 0.004 |
| **lesion vs random ratio** | 12× | 13× | **15×** |

→ **遮 lesion 對 multi-lesion case 影響更大**（N≥3 是 random 的 15 倍），證明 lesion attribution 是 load-bearing。

---

### D-i ablation：faithfulness–ranking 張力（為何不直接靠 listwise 修 R@1/R@20）

Table C1 的 paired bootstrap（B=10000, query-level paired）顯示 v3 對 Phase 1 有顯著的 sem R@1 −1.16pp（p=2e-4）/ R@20 −3.41pp（p<1e-4）退化，sem MRR +0.003 則**不顯著**（CI [−0.002, +0.009], p=0.20）；cluster 線全部 p<1e-4 顯著（cl R@10 +4.02pp、cl MRR +0.0355）。一個自然的問題：能不能加 listwise / soft target loss 把 R@1/R@20 拉回來？

我們做了兩條軸、共 6 個訓練點的 ablation（架構不動，softmax α + multiplicative scoring 保留）：

- **Rung 0（graded soft labels）**：`positive_mask` 換成 `clamp((maxcos−0.90)/0.10, 0, 1)` ramp（候選的 max-cosine-to-GT）。
- **Rung 1（listwise multi-positive CE）**：`loss = BCE + λ·CE(softmax(score/T), uniform-over-positives)`，T=0.1，λ ∈ {0.05, 0.1, 0.3, 0.5, 1.0}，binary 正例集合。

**Rung 0**：sem R@1 −2.61pp***（更糟）、retrieval 全線退、faithfulness 未測。**Rung 1 λ-scan**：retrieval Δ 對 λ 單調上升（λ=1.0 達 sem R@10 +4.82pp***、R@20 +5.20pp*** vs v3），**但任何 λ≥0.05 皆把 `no_lesion`(lesion-type) drop 從 +0.038 翻成 −0.004~−0.085**（負＝遮病灶反而加分），`no_global` 從 v3 的 −0.030 翻成 +0.046~+0.124——listwise 將 attribution mass 整體路由到 global，繞過 multiplicative 的病灶 gate，與 raw-`case_db` ablation 同一失敗簽名。`no_random` 全程 ≈ 0（為**選擇性反轉**，非普遍敏感性退化）。

**Faithful 區（λ=0）與 retrieval-improving 區（λ≥0.3）零重疊**：λ=0.05/0.1 已破壞 faithfulness 但 retrieval Δ ≈ 0 或退（雙輸），λ≥0.3 才有 retrieval 收益但 faithfulness 早崩。R@1 退化亦與 λ 無關（5 個 λ 的 sem R@1 Δ vs v3 全 ns），是結構性極限。

**結論：v3 (λ=0, BCE only) 是 CEAM 在 faithfulness–ranking 空間中的奇異 Pareto-faithful 操作點；Table C1 的 sem R@1/R@20 退化是 architecture-enforced lesion grounding 的結構性代價。** 完整數字 + 取得指令見 [paper_tables.md](paper_tables.md) Table C7。

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
    --cluster_json outputs/cause_clusters_llm.json \
    --attribution_mode softmax --scoring_mode multiplicative \
    --gammas 0.0 --dump_gamma 0.0 --text_kind ${mode}
done
```

**Retrieval 三 mode 完全持平（sem_MRR diff ≤ 0.0002）：**

| Text mode | sem_MRR | R@10 | lesion-type l/g (N=2) |
|---|---|---|---|
| medical | 0.3009 | 46.83% | 1.50 |
| colloquial | 0.3010 | 46.86% | 1.63 |
| **none** (vision-only) | 0.3008 | 46.76% | 1.87 |

→ **Vision-only 推論完全可行**，使用者沒提供文字描述系統照樣運作。

> 純 CEAM 實測（2026-05-22）。下方 α 分佈表是 attribution breakdown（α 是 CEAM 自身輸出）；趨勢一致：text 缺席時 lesion-type l/g 上升。

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

### Faithfulness 強化：CLSC（cross-lesion supervised contrastive，VLM-Lesion 訓練 auxiliary；非 FaCE-R 主流程）

> 本節屬於 VLM-Lesion 訓練側的 representation-learning ablation。CLSC 能改善下游 CEAM lesion-masking faithfulness，但目前在上游 VLM-Lesion 分類任務上沒有穩定全面優於 baseline，因此**不作為 FaCE-R 主流程預設 encoder**；在 FaCE-R 章節中只作為 lesion representation quality 會影響 CEAM faithfulness 的 downstream validation。

CEAM 的 architectural faithfulness（softmax α、multiplicative scoring）保證**結構上**評分必須走 lesion evidence，但**實質 faithfulness** 還受限於 lesion features 本身的品質。若 VLM-Lesion 訓練時可以靠 background context shortcut 分對 caption，CEAM 的 α 就會落在 features 不可靠的維度上。CLSC 是針對這個 shortcut 開的 auxiliary loss。

#### 動機：消除 background shortcut

baseline VLM-Lesion 用 SigLIP-style sigmoid loss + 全 batch caption bank（B=128 對 190 條 captions）訓練，negatives 都來自跨 image：

```
positive caption ←→ this lesion crop (cat A)
negative captions ←→ other batch images (cat B/C/D...) — 全部來自不同魚、不同光線、不同背景
```

模型可以靠 background features（魚體姿勢、底色、拍攝距離）分對 captions，**完全不用看 lesion content**。對 Phase 1 retrieval 無傷（目標只是分得開），對 CEAM attribution 是致命傷 — features 不 lesion-grounded → α 歸因失去意義。LocalGlobalFusion 雖把 global context concat 進 lesion features 作 representational regularization，但**沒提供「不能 shortcut」的明確訓練訊號**。

#### Formulation

只在 batch 內**同圖 ≥2 個不同類別 lesion** 時觸發。對每張這樣的圖，取它所有 lesion features $\{f_1, ..., f_N\}$（Path A fused output）跟對應 caption embeddings（從 bank 拉出每樣本所屬類別的 caption 平均後 L2-norm）$\{t_1, ..., t_N\}$，做雙向 supervised contrastive softmax：

$$
\mathcal{L}^{i2t}_i = -\log \frac{\sum_{k: y_k = y_i} \exp(\tau f_i^\top t_k)}{\sum_{k=1}^{N} \exp(\tau f_i^\top t_k)},\quad
\mathcal{L}^{t2i}_j = -\log \frac{\sum_{k: y_k = y_j} \exp(\tau t_j^\top f_k)}{\sum_{k=1}^{N} \exp(\tau t_j^\top f_k)}
$$

對稱兩方向平均並對 batch 內合格圖取 mean，加進主 multipos sigmoid loss：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{multipos sigmoid}} + \alpha_{\text{clsc}} \cdot \mathcal{L}_{\text{clsc}}
$$

**設計關鍵**：negatives 來自**同一張圖** — 背景、光照、魚體、距離完全一樣。模型唯一能用來區辨同類 vs 不同類 lesion 的訊號就是 **lesion 像素本身**。把「不能 shortcut」的訊號**結構性地塞進訓練資料**而不是靠額外 augmentation。

#### Trigger 條件

|  | count | ratio |
|---|---|---|
| 訓練圖總數 | 18077 | 100% |
| 多類別合格圖（≥2 lesion 類別） | **3814** | **21.1%** |
| → 內含 lesion crops（合格 pool）| 9849 | 31.5% of crops |

隨機抽 B=128 batches 下，每 batch 平均 0.57 個同圖 dup pair，**約 13% 的 batch** 真正命中合格條件。觸發稀疏，但**每次 fire 時 loss 量級 ~0.86**（≈ ln 2，2-way 區辨的 chance loss 之上）— 訊號強度足以推動 features，但不會排擠主 sigmoid loss 在其它 87% batches 的訓練。

#### 訓練命令

```bash
cd diagnosis_model/vl_classifier
python train.py \
    --multipos --fusion --freeze_text_encoder \
    --cross_lesion --cross_alpha 1.5 --cross_tau 0.1 \
    --output_dir outputs/siglip2_base_patch16_224_multipos_fusion_xL1.5_en_zh \
    --eval_test_after_train
```

CLSC ckpt 用於 `build_case_database.py` 的 `--vlm_lesion` 重建 `case_db_xL1.5/`，下游 CEAM / faithfulness eval 流程不變。執行時間 ~50 min（B=128, 8 epochs）。

#### 結果（vs baseline VLM-Lesion，1573 valid queries）

**Faithfulness 按 cause-type 分桶：**

| 條件 | baseline | **+CLSC** | Δ | 相對改善 |
|---|---|---|---|---|
| no_lesion · global-type | +0.0289 | +0.0312 | +0.0023 | +8% |
| no_lesion · **lesion-type** | +0.0378 | **+0.0444** | +0.0066 | **+17%** |
| no_lesion · **mixed** | +0.0413 | **+0.0694** | +0.0281 | **+68%** |
| no_random（噪音 sanity） | +0.0027 | +0.0022 | −0.0005 | 維持 ≈ 0 ✓ |

**N-lesion 分桶：**

| Bucket | n | baseline | **+CLSC** | Δ | 相對改善 |
|---|---|---|---|---|---|
| N=1 | 886 | 0.0239 | 0.0326 | +0.0087 | +36% |
| **N=2** | 456 | 0.0467 | **0.0632** | +0.0165 | **+35%** |
| **N≥3** | 231 | 0.0531 | 0.0587 | +0.0056 | +11% |

**Retrieval 基本不變**：CEAM retrieval baseline sem R@10 46.8% → +CLSC 47.2%（+0.4pp，noise 內）。CLSC 的價值不是提升 cause retrieval headline，而是把 lesion features 往 lesion-grounded evidence 方向推；是否採用為預設 encoder 仍需回到 VLM-Lesion 分類主指標決定。

→ **CLSC 結果讀法**：retrieval 無感、no_random 維持 ≈ 0（沒對任意擾動變敏感），但 lesion-type / mixed / N=2+ 的 attribution drop magnitude 全面上升。這代表 CLSC 是 **faithfulness-oriented auxiliary**，不是 final encoder selection 的充分理由；主流程仍以 baseline VLM-Lesion 為準。

#### Sparse firing 是 design feature，不是 limitation

提高 cross loss 的「訊號強度」有兩條軸：(a) 加大 α 權重；(b) 加大 firing rate（換 grouped sampler）。我們驗證兩條軸**都不會線性放大效益**，反而推過甜蜜點都會回退。

| 設定 | firing rate | α | val main loss (i2t+0.5·t2i, ep 6) | lesion-type drop | mixed drop |
|---|---|---|---|---|---|
| baseline (no CLSC) | — | — | 0.298 | 0.0378 | 0.0413 |
| α=0.3 random (smoke) | 13% | 0.3 | 0.298 | ≈ baseline | ≈ baseline |
| **α=1.5 random (CLSC)** | **13%** | **1.5** | **0.298** | **0.0444** | **0.0694** |
| α=3.0 random | 13% | 3.0 | 0.304 | 0.0336 (back to baseline) | 0.0387 (back to baseline) |
| α=1.5 + grouped sampler (50/50) | ~100% | 1.5 | 0.425 (退化 +43%) | 0.0479 (winner!) | 0.0346 (loser) |

**讀法**：
- 軸 (a) α scaling：α=0.3 訊號太弱、α=1.5 甜蜜點、α=3.0 cross loss 在 13% 觸發 batches 上 gradient 量級壓過主 sigmoid loss，features 被拉去 in-image 區辨方向但離 caption alignment 太遠，下游 CEAM 拿到的 features 回退到 baseline 水準
- 軸 (b) firing-rate scaling：grouped sampler 100% 觸發 + α=1.5 → batch 內 ~30 個同圖 dup pairs，主 sigmoid loss 的 negatives 變得太難（同 image 共享 background）→ main task 在 8 epochs 內沒收斂到 baseline 水準 → CEAM features 雖然 lesion-type 端更銳化（+0.0035 vs xL1.5）但 mixed 端因主任務 underfitting 退化（−0.035 vs xL1.5）

**Design principle**：multi-task fine-tune 上的 auxiliary loss，**保持 sparse triggering + 中等 weight** 比 dense triggering 或大 weight 更穩。auxiliary 只在「主任務無法處理的 hard case」（這裡是「同圖多 lesion 區辨」）發力，**不搶主任務的 gradient budget**。CLSC 的 13% firing + α=1.5 是這條 principle 在 fish disease 場景的具體實作點。

論文 ablation 用上面這張表呈現，**α-scan 跟 firing-rate ablation 一起講**，故事比單軸 α-scan 強。

---

### CLSC 對照組：LSCFT（lesion-selective counterfactual faithfulness training，negative result）

CLSC 用「**自然存在於 dataset 中的多 lesion 同圖**」當 hard negatives。一個合理的對照組問法：**如果用 augmentation 製造 counterfactual（mask 掉 lesion 中心區域），訓練模型對「lesion 拿掉」變敏感，是不是也可以做到 CEAM attribution faithfulness？**

LSCFT 是我們做的這個對照組，**negative result**：訓練目標達成了但**不 transfer 到 CEAM attribution**。

#### LSCFT 設計（雙臂 ranking）

每個 lesion crop 並行做兩個干預，比較 caption similarity drop：

- **Positive arm**（拿掉 lesion 內容）：中心 P×P 區域置換為 mean / cutmix / noise 之一（隨機抽 multi-intervention bank）
- **Negative arm**（保留 lesion）：peripheral mask / color jitter / hflip 之一（隨機抽）

Loss：
$$
\mathcal{L}_{\text{lscft}} = \text{softplus}(m - (\text{drop}_+ - \text{drop}_-))
$$

其中 $\text{drop}_+ = \cos(f, t) - \cos(f^+, t)$（positive 應該大）、$\text{drop}_- = \cos(f, t) - \cos(f^-, t)$（negative 應該小）。Ranking 公式抵銷 $f$ 對 $t$ 的絕對 alignment，只要求**相對排序** — 訓練不跟主 sigmoid loss 搶 orig feature 的監督。

訓練 trick：global feature 算一次共用，positive / negative 各做一次 local encoder forward（總 3 次 local + 1 次 global）。記憶體上需要 B=96。

#### 結果：訓練目標達成、faithfulness 沒 transfer

| 設定 | val lscft (ep 4 best) | lesion-type drop | mixed drop | N=2 drop | N≥3 drop |
|---|---|---|---|---|---|
| baseline | — | 0.0378 | 0.0413 | 0.0467 | 0.0531 |
| **+CLSC (xL1.5)** | — | **0.0444** | **0.0694** | **0.0632** | **0.0587** |
| LSCFT P=80 | 0.243（從 0.500 降）| 0.0369 | 0.0403 | 0.0459 | 0.0548 |
| LSCFT P=180（對齊 eval intervention magnitude）| 0.241 | **0.0067 ↓↓** | 0.0708 | 0.0336 ↓ | 0.0296 ↓ |

- **LSCFT P=80**：valid lscft 收斂（0.50 → 0.24，訓練目標達成）但下游 faithfulness 跟 baseline 持平、**比 CLSC 明顯弱**
- **LSCFT P=180**（加大 mask 對齊 eval-time 完整 bbox masking）：**反而把 lesion-type drop 砍到 0.0067**（baseline 的 18%）— 強行對齊 mask magnitude 反而把 features 拉去錯方向

#### 為什麼不 work：訓練目標跟 CEAM evaluation 不在同一 vector space

- **LSCFT 訓練的是**：image-text cosine similarity 對 lesion 像素遮蔽的敏感度
- **CEAM faithfulness 量的是**：attribution α 引導下的最終 ranking score 對 lesion 像素遮蔽的敏感度
- 兩個 quantity **經過 CEAM MLP 後不是同一個量**。LSCFT 把 features 推到「caption-similarity 對 lesion 敏感」的方向，但 CEAM 的 α 機制需要的是「lesion features 之間相互區辨」的方向（這正是 CLSC 直接訓的）

**CLSC vs LSCFT 的對比**告訴我們：在 fine 階段的 attribution module 上，**discriminative signal**（lesion 之間相互區辨）比 **counterfactual signal**（lesion 拿掉變不像 caption）更直接、更可 transfer。論文 narrative 上可以用這個對比強化 method 章節的合理性 — 我們不是只試了一個 idea，我們試了顯而易見的 counterfactual augmentation 對照組並證明它不 work。

---

### Phase 3：Case encoder（ABQ representation stage）

Phase 3 是 ABQ 的 **representation stage**：它本身不做 quantization，而是先把 multi-vector case evidence（global + variable-length lesions）蒸餾成單一 L2-normalized case embedding。這一步把原本昂貴的 Hungarian / multi-vector matching 轉成 single-vector cosine retrieval，為 Phase 4 的 quantized retrieval 提供可壓縮的 case representation。

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

接 CEAM 重排後(text=medical):

| Method | sem R@10 | per-q ms |
|---|---|---|
| Phase 1 + CEAM | 0.453 | 18.2 |
| DeepSets dual + CEAM | 0.453 | **3.4** (5.4× faster) |

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

### Phase 4 ABQ Aggregation-Buffered Quantization

> **ABQ = Aggregation-Buffered Quantization** 是 coarse 階段的部署框架，由兩個層次組成：
> **Phase 3 representation stage** 先將 multi-vector case evidence 蒸餾成單一 case embedding；
> **Phase 4 quantization stage** 再對這些 case embeddings 做 codebook compression，並分析 downstream cause aggregation 如何吸收量化誤差。
> 因此，ABQ 的核心不是「發明 RVQ」，也不是單純把 case vectors 壓小，而是刻畫並利用 **aggregation buffer absorbs quantization noise** 這個現象。
>
> ABQ 的四條 contribution（不是「發明 RVQ」，是不同層級的貢獻）：
> 1. **可部署的 case-based 病因檢索（系統）**：Phase 3 把 multi-vector Hungarian retrieval 轉成 single-vector cosine retrieval（15.4 ms → 1.1 ms），Phase 4 再用 codebook quantization 壓縮 case bank，在 1M 規模維持 sub-ms / sub-20MB。
> 2. **Aggregation-buffer 吸收 scaling law（科學發現，最 novel）**：`D ≈ c·ε/K^{0.84}`，指數 encoder-invariant，附 operating-point selector `K* = (c·ε/D_target)^{1/0.84}`。
> 3. **被發現驅動的設計（方法）**：importance-weighted codebook（K\* 降 25–33%）；DeepSets dual-target 蒸餾（InfoNCE 防 alignment 崩塌、student 微幅超越 teacher）。
> 4. **負結果（刻畫）**：compressed residual reranker 在 production 多餘（端到端驗證）、1-stage full-vocab 差 3.5×、isolation/density 加權不穩。
>
> 本階段以 Residual Vector Quantization (RVQ) 為量化器實作；以下小節依「壓縮做法 → 端到端證據 → scaling law → weighted codebook」展開。
>
> 壓縮會引入 case 排序誤差，可選掛一個 **compressed residual reranker（Light implementation）** 學殘差 Δ ≈ qᵀe 修正。但**下面的端到端實驗（[結果 — 端到端](#結果--端到端-ceam-主張)）顯示：production cascade（top_k=20 + CEAM）的 aggregation buffer 已把壓縮損害吃光，residual correction 在 production 多餘**——它只在 buffer-free 部署（single-case ANN）才賺到價值。所以 reranker 在本文是**設計空間的對照點**，不是 production 元件。

**動機**：Phase 3 把 retrieval 從 15.4 ms 壓到 1.1 ms，下一個 bottleneck 是 case bank 規模——12,780 cases × 768 dim fp32 = 39 MB 還能全進 GPU，但若部署到 1M+ cases（diagnostic platform 預期長期累積）則 brute-force dense 退化到 5 ms/q 並佔約 3.0 GB（fp32）。Codebook 壓縮把 index 縮到 4–17 MB、retrieval 維持 sub-ms。

#### 設計

把凍結的 DeepSets case encoder 輸出 `z_i ∈ ℝ^768` 用 **Residual Vector Quantization** 壓成 `codes ∈ [K]^M` 的離散碼字（M levels × K codes/level）：

```
ẑ_i  = Σ_m  C_m[k_{i,m}]              # 重建
e_i  = z_i − ẑ_i                       # 量化殘差
```

檢索分兩階段：
- **First-stage**：`s_first(q, i) = qᵀẑ_i`，透過 lookup table `LUT[m, k] = q · C_m[k]` 每 candidate 只要 M 次加法
- **Optional residual correction**：對 top-K_top=50 candidates，用 compressed residual reranker 預測 `Δ ≈ qᵀe_i` 並 `s_final = s_first + Δ`

數學動機：`s_dense(q,i) = qᵀz_i = qᵀẑ_i + qᵀe_i = s_first + qᵀe_i`，所以 residual correction 不是重新估計整個 similarity score，而是估計被 RVQ 壓縮掉的 residual term。

#### Residual correction variants

壓縮後的 first-stage score 可寫成：

`s_dense(q,i) = qᵀz_i = qᵀẑ_i + qᵀe_i`

因此 reranker 的角色不是重新估計整個 similarity score，而是估計被 RVQ 壓縮掉的 residual term `qᵀe_i`。我們比較兩種 residual correction 設定：

| Variant | Candidate token 包含 | 部署假設 | 論文定位 |
|---|---|---|---|
| **Compressed residual reranker（Light implementation）** | ẑ, codes embedding, q⊙ẑ, \|q−ẑ\|, s_first, ‖e‖ | 完全壓縮：memory 中只保留 codes / compact metadata | 實際可部署的速度–記憶體權衡 |
| **Dense-residual oracle（dense-residual oracle（Full analytic））** | Light features + z_i + e_i + q⊙e_i | top-K 時可讀取 dense z / residual e | 可恢復量化誤差的 upper bound |

Dense-residual oracle 可直接取得或解析 `qᵀe_i`，因此不是 production setting，而是用來估計「若 residual correction 完美，最多能補回多少壓縮損害」的理論上界。Compressed residual reranker 則只使用壓縮索引中可取得的資訊，是實際部署可用的折衷做法。

#### 評估的雙 regime

| Regime | top_k_cases | 動機 | 壓縮 damage？ |
|---|---|---|---|
| **A (production)** | 20 | 經過 cause-aggregation pool（~87 個 unique cause） | 全部持平，aggregation buffer 吸收量化 noise |
| **B (buffer-free)** | 1 | 單一 case 的 cause set 直接當預測，無 buffer | 壓縮損害直接顯露 |

兩個 regime 拆的是「aggregation buffer 在不在」。**主結果是 Regime A**：production 的 buffer 讓壓縮幾乎免費（free lunch），這由下面的端到端實驗（經 CEAM）直接證實。Regime B 是 buffer-free 部署（single-case ANN）的對照，是唯一能看出 reranker 價值的設定——所以 reranker 是對照點不是 production 元件。下面先給最終、最 production-relevant 的端到端結果，再給 retrieval-side proxy（壓縮率 sweep）。

#### 訓練

```bash
# Step 1: fit RVQ codebook（~1 分鐘，每個 (M, K) 一次性）
$PY -m diagnosis_model.cause_inference.rvq_rerank.fit_rvq \
  --encoder_ckpt outputs/encoder_final/best_encoder.pt \
  --case_db_dir outputs/case_db \
  --output_dir outputs/rvq_rerank \
  --M 4 --K 256

# Step 2: 訓 compressed residual reranker（Light implementation；~15–60 秒）
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
compressed residual reranker 預測 Δ；`loss = listwise_KL(s_final, s_dense) + λ_mse · MSE(Δ, qᵀe)`。

#### 結果 — 端到端 CEAM 主張

前面的 Regime A/B 量的是**中間 case-retrieval 代理指標**（壓縮後的 case 排序 → cause 聚合 R@10），沒接 CEAM。這裡量**真正 production cascade 的最終病因 metric**：壓縮後的 coarse case 排序 → top_k=20 candidate pool → **CEAM 重排**。1573 valid，LLM 466 cluster，coarse 走 production raw 路（encoder_raw + rvq_rerank_raw），CEAM 走 fine `case_db`。腳本 [eval_ceah_compressed.py](eval_ceah_compressed.py)。

**Regime A（top_k=20，production，有 buffer）：**

| coarse 來源 | 壓縮× | sem R@10 | cl R@10 | sem MRR |
|---|---|---|---|---|
| dense（無壓縮，reference） | 1× | 0.476 | 0.396 | 0.303 |
| RVQ-only M=4 K=256 | 768× | 0.472 | 0.398 | 0.302 |
| + compressed residual rerank（Light） | 768× | 0.478 | 0.398 | 0.304 |
| RVQ-only M=2 K=64 | 2048× | 0.475 | 0.402 | 0.308 |
| + compressed residual rerank（Light） | 2048× | 0.472 | 0.401 | 0.306 |
| **RVQ-only M=1 K=16** | **6144×** | **0.470** | **0.396** | 0.303 |
| + compressed residual rerank（Light） | 6144× | 0.464 | 0.385 | 0.288 |

→ **壓縮損害穿不過 CEAM**：dense → 6144× RVQ-only，最終 sem R@10 只掉 **0.6 pp**、cluster R@10 **完全不動**。用 16-entry codebook 加上 per-case code assignments 表示 12,780 cases，最終病因排序幾乎零損失。**compressed residual reranker 在 production 沒貢獻甚至微害**——每個壓縮率 +Light residual correction vs RVQ-only 全在 ±0.5 pp noise 內，M1K16 甚至 −0.5 pp sem / −1.1 pp cl / −1.4 pp MRR。

**Regime B（top_k=1，buffer-free，single-case ANN-style，同樣經 CEAM）：**

| coarse 來源 | 壓縮× | sem R@10 | cl R@10 |
|---|---|---|---|
| dense | 1× | 0.517 | 0.469 |
| RVQ-only M=4 K=256 | 768× | 0.484 | 0.454 |
| **+ compressed residual rerank（Light）** | 768× | **0.510** | **0.467** |
| RVQ-only M=1 K=16 | 6144× | 0.368 | 0.405 |
| **+ compressed residual rerank（Light）** | 6144× | **0.397** | **0.428** |

→ 拆掉 buffer 後壓縮**真傷**（6144× dense → RVQ-only −14.8 pp sem R@10），compressed residual reranker **真救**（M4K256 +2.6 pp 回到接近 dense、回收率 ~79% 與 retrieval-only proxy 吻合；M1K16 +2.9 pp sem / +2.3 pp cl）。**這就是 compressed residual reranker 唯一的 niche：buffer-free 部署，不是 FaCE-R 的 production cascade。**

> **讀法**：production（Regime A）壓縮免費、residual correction 多餘；buffer-free（Regime B）壓縮傷、compressed residual reranker 有效。reranker 的價值**完全條件於 aggregation buffer 缺席**。本文據此把 reranker 定位成 buffer-free 部署的 conditional fallback，不是 production 方法。caveat：Regime A 的 <1 pp 差異都在 noise（SE ~1.26 pp）內——「在 noise 內」正是 free-lunch 的證據；Regime B 的 15 pp 損害與 2–3 pp 回收都在 noise 之上。

#### 結果 — 端到端 CEAM 主張（cross-dataset：DDXPlus）

fish 的端到端在 production（γ=0，純 CEAM）下 reranker 多餘。DDXPlus 是 **coarse-dominant 的 γ=0.75** production 點，所以我們掃 γ 並經 CEAM 重打分，看 reranker 在「CEAM 丟掉 coarse 順序」與「CEAM 吃進 coarse 順序」兩種 regime 各自的端到端影響。腳本 [ddxplus/eval_ceah_compressed.py](ddxplus/eval_ceah_compressed.py)（fish `eval_ceah_compressed.py` 的 DDXPlus 版），M4K256 768×、5000 valid、`text_kind=none`。

| γ | dense | rvq_only | **light** | full（oracle） | 讀法 |
|---|---|---|---|---|---|
| 0.00（純 CEAM） | 0.878 | 0.876 | 0.877 | 0.876 | **壓縮全隱形**，dense/rvq/light/oracle 全塌成同值（±0.16 pp noise）——CEAM 重打分把 coarse 順序丟掉，reranker 修的東西不存在 |
| 0.50 | 0.954 | 0.971 | 0.966 | 0.966 | |
| **0.75（production）** | 0.954 | **0.981** | 0.974 | 0.973 | rvq_only **贏 dense +2.7 pp**；light **拖回 −0.7 pp**；**oracle 也 −0.9 pp** |
| 1.00（純 Phase1） | 0.525 | 0.666 | 0.604 | 0.627 | rvq_only +14 pp；light −6.2 pp |

（pathology R@1。soft DDX NDCG@5 在 γ=0.75 reranker 給 +0.5 pp（0.826→0.831），量級小、嚴格 pathology metric 主導。）

→ **DDXPlus 端到端兩個 γ regime 各自獨立地殺掉 reranker，機制不同**：(A) γ=0 純 CEAM 把 coarse 順序丟掉，壓縮與 reranker 全隱形；(B) γ=0.75 coarse 主導時，RVQ 是 **implicit regularizer**（rvq_only 贏 dense），reranker 學 `Δ≈qᵀe` 把分數**拉回較差的 dense**，所以有害——**連 full analytic oracle（完美 Δ）都 −0.9 pp**，徹底排除架構問題。

> **Cross-dataset 收斂**：reranker 修的量 = 「回補 dense case 排序」。fish production（γ=0）下被 aggregation buffer + CEAM 丟順序吃光 → 多餘；DDXPlus production（γ=0.75）下 dense 本身比 RVQ 差 → 回補 dense 即有害。**兩個資料集、兩種機制、同一結論：reranker 在 production cascade 永遠不幫忙，唯一 niche 是 buffer-free single-case 部署。** 架構非瓶頸（oracle 印證），故定位為設計空間對照點，production 不掛。

#### 結果 — RVQ 壓縮率 sweep（retrieval-side proxy，Regime A, top_k_cases=20, 1573 valid）

> 以下兩張表是**未接 CEAM 的 retrieval-side proxy**（case-retrieval R@10），數字與上面端到端表不同量、僅供對照壓縮對 coarse 檢索本身的影響。

12 個 (M, K) 配置全部 R@10 Δ 在 ±1.7 pp 內（SE = 1.26 pp），**全是 noise**：

| RVQ config | Compression × | sem R@10 | Δ vs dense |
|---|---|---|---|
| Dense fp32 | 1× | 44.7% | — |
| M=4 K=256 | 768× | 45.5% | +0.8 pp |
| M=2 K=64 | 2048× | 43.5% | −1.2 pp |
| **M=1 K=16** | **6144×** | **43.9%** | **+0.0 pp** ← 16 prototypes 而已！ |

**Production regime 對 RVQ noise 幾乎免疫**：M=1 K=16 用 **16-entry codebook + per-case assignments** 表示整個 12,780 case bank，R@10 完全不掉。原因：Phase 1 的 cause-aggregation pool 是個有效的「coarse clustering buffer」，個別 case 排序錯誤被 union over 20 cases 吸收。

只有 R@100 才看出 M=1 K=16 退化（−6 pp），但 R@10/MRR 完全持平。

#### 結果 — Regime B 完整對照（retrieval-side proxy，top_k_cases=1, 1573 valid）

> 同樣是**未接 CEAM 的 retrieval-side proxy**。它是 4-method × 壓縮率的完整 Pareto，端到端版見上面 [結果 — 端到端](#結果--端到端-ceam-主張)。

拆掉 aggregation buffer 後，RVQ 損害直接顯露，compressed residual reranker 才有用武之地：

| Method | Comp × | sem R@10 | Δ vs dense | Gap recovered |
|---|---|---|---|---|
| **Dense fp32** | 1× | **0.537** | — | — |
| RVQ-only M=4 K=256 | 768× | 0.506 | −3.1 pp | — |
| **+ compressed residual rerank（Light）** | 768× | 0.518 | −1.9 pp | **39%** |
| + dense-residual oracle（Full analytic） | 768× | **0.537** | +0.0 pp | **100%** |
| RVQ-only M=2 K=64 | 2048× | 0.463 | −7.4 pp | — |
| **+ compressed residual rerank（Light）** | 2048× | 0.500 | −3.7 pp | **50%** |
| + dense-residual oracle（Full analytic） | 2048× | 0.515 | −2.3 pp | 69% |
| RVQ-only M=1 K=16 | 6144× | 0.372 | −16.5 pp | — |
| **+ compressed residual rerank（Light）** | 6144× | 0.462 | −7.6 pp | **54%** |
| + dense-residual oracle（Full analytic） | 6144× | 0.458 | −7.9 pp | 52% ← 被 first-stage recall 限制 |

**三個 method-level 觀察：**

1. **Compressed residual reranker（Light）在所有壓縮率穩定回收 39–54% 的 gap**——method 不是 cherry-picked，對 768×–6144× 都有效。
2. **Dense-residual oracle（dense-residual oracle（Full analytic））在 M=4 K=256 完全回補到 dense (0.537)**——證明「reranker 的可挽回上限 = top-K 包含 GT case 的機率」。M=1 K=16 saturate 在 0.458，因為 top_K_rerank=50 對 16-prototype 不夠，first-stage 包含 GT 的機率本身就低，連 oracle Δ 都救不回。
3. **M=1 K=16 + compressed residual reranker（Light）達 0.462**，跟 M=4 K=256 純 RVQ 的 0.506 只差 4 pp，但壓縮率 **8× 更高**——Pareto frontier 被 residual correction 整個推外。

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
- **rvq_light** 約 0.6 ms 常數開銷（compressed residual reranker forward 主導，跟 N 無關）
- **rvq_full_analytic** 跟 rvq_only 同步：top-K=50 dense-residual oracle correction cost 微小

Memory @ N=1M（M=4 K=256 codes = uint8）：

| Method | Index size | Compression vs fp32 dense |
|---|---|---|
| dense fp32 | 3,072 MB | 1× |
| **rvq_only** | **4 MB** | **768×** |
| rvq_light | 17 MB (codes + 13 MB compressed residual reranker) | 181× |
| rvq_full_analytic | 3,076 MB (z bank / residual e 為 oracle correction 必要) | 1× |

#### Deployment Pareto（三層）

| 部署情境 | 推薦方法 | Memory | Latency | Quality (Regime B) |
|---|---|---|---|---|
| Memory + compute 都緊 | **rvq_light（compressed residual reranker）** | 17 MB | 0.59 ms | 0.518 |
| Compute 緊但 memory 寬鬆 | rvq_full_analytic（dense-residual oracle） | 3.0 GB | 0.22 ms | 0.537 (=dense) |
| Memory 緊但 quality 可讓 | rvq_only | 4 MB | 0.19 ms | 0.506 |
| 無壓力 | dense | 3.0 GB | 5.15 ms | 0.537 |

#### 論文敘事建議

> **ABQ（Aggregation-Buffered Quantization）把 case-based retrieval 推到 1M+ 規模的 deployment 範圍。** 後接在 Phase 3 DeepSets case encoder 之後，凍結 z_i，用 codebook 量化（RVQ 為實作）壓成 M-byte codes（768× memory compression at M=4 K=256）。1M case bank latency benchmark 顯示 dense 退化到 5.15 ms/q + 3.0 GB，壓縮後維持 0.19–0.59 ms/q + 4–17 MB，證明 case-based retrieval framework 在數據集規模成長下仍可實時部署。
>
> **本階段的 method-level 貢獻不是壓縮本身，而是刻畫壓縮損害如何被 aggregation buffer 吸收。** 端到端實驗（壓縮 coarse → CEAM）顯示：production cascade（top_k_cases=20）即使把 12,780 cases 壓成 16 個 prototype（6,144× compression），最終病因 sem R@10 只掉 0.6 pp、cluster R@10 不動——cause-aggregation pool 充當 coarse clustering buffer 把個別 case 排序誤差 union 掉。我們把這個現象量化成 scaling law `D ≈ c·ε/K^{0.84}`，並據此做 importance-weighted codebook（K\* 降 25–33%）。
>
> **Compressed residual reranker 是設計空間的對照點，不是 production 元件。** 它學 Δ ≈ qᵀe 修補 RVQ 排序誤差，但端到端實驗顯示它在 production（有 buffer）多餘甚至微害；只在 buffer-free 部署（top_k_cases=1, single-case ANN）才回收 39–79% 的 gap；dense-residual oracle 則作為可恢復量化誤差的 upper bound。換言之，免費的結構性 buffer 在 production 支配了學習式修正——這個對比本身就是 scaling-law finding 的最佳襯托。

#### Aggregation-buffer 吸收 scaling law + importance-weighted codebook（method-level 擴展）

前面「雙 regime」框架是**定性**觀察（aggregation buffer 吸收 RVQ noise）。這個 subsection 把它**定量化**，並基於 scaling law 推導出一個 drop-in codebook 改造，把 K* 降低 25–33% 而不動 RVQ 數據結構。

##### Scaling law

對 5 個 RVQ config（壓縮率 384× → 24576×，relMSE ε 從 0.025 到 0.50）× 10 個 `top_k_cases` (1→50) 做雙 sweep，發現 dense vs RVQ-only 的 R@10 damage `D` 服從 closed-form：

> **D ≈ c · ε^p / K^q**，其中 **p ≈ 1.00, q ≈ 0.84**

擬合證據（[eval_absorption_surface.py](rvq_rerank/eval_absorption_surface.py)）：

| Encoder | c | p (ε exp) | q (K exp) | R² | jackknife q std |
|---|---|---|---|---|---|
| Fine-tuned (case_db) | 59.9 | 0.93 | 0.88 | 0.90 | 0.088 |
| Raw SigLIP2 (case_db_raw) | 156.0 | 1.11 | 0.78 | 0.79 | 0.080 |

**Encoder-invariance 測試**（fixed-effect：兩個 encoder 共用 (p, q)，各自有 intercept c）：

| Model | params | pooled R² |
|---|---|---|
| 3-param fully shared (c, p, q) | 3 | 0.757 |
| **4-param 共用 exponents** | 4 | **0.828** |
| 6-param fully separate | 6 | 0.834 |

從 6-param → 4-param 只損失 ΔR²=0.006，意思是**指數 (p, q) 在兩個 encoder 之間無實質差異**——absorption 的*形狀*是 representation-agnostic，**只有 intercept c 隨 encoder 變**。

**Encoder-specific intercept 的物理解讀**：c_raw / c_fine ≈ 2.06——同樣的 ε 下，raw embeddings 的 ranking damage 是 fine-tuned 的兩倍。fine-tuned encoder 把 ranking 訊號塞進更窄 subspace，量化噪聲只要不正好打在那個 subspace 就影響有限；raw 的 ranking 訊號分散在 768 維，對 isotropic noise 更敏感。**fine-tuning 不只改善 retrieval，也提供 2× quantization robustness**。

##### Operating-point selector

從擬合反推 damage ≤ 1pp 所需的最小 `top_k_cases`（用 fine-tuned encoder fit）：

| ε (relMSE) | 對應 (M,K) | top_k* （預測） | top_k* 觀察 |
|---|---|---|---|
| 0.027 | M=8 K=256 | 2 | 2 |
| 0.046 | M=4 K=256 | 4 | 5 |
| 0.131 | M=2 K=64 | 12 | 15 |
| 0.273 | M=1 K=16 | 26 | 20–30 |
| 0.496 | M=1 K=4 | 49 | 50 |

> **K\* = (c · ε / D_target)^{1/q}**

直接告訴部署：「想要 6144× compression？aggregation top-K 要設 ≥26 才 free lunch」。

##### Importance-weighted RVQ（method-level 演算法貢獻）

Scaling law 告訴我們 damage 跟 ε 線性。但**ε 是 12780 個 case 的 mean**——如果重新分配 codebook capacity，讓「常被 retrieve 的 case」 ε 變小、「從來不被 retrieve 的 case」 ε 變大，**mean ε 可以不變但 downstream 看到的 effective ε 變小**。

具體：把 vanilla `_kmeans` 換成 weighted k-means（[weighted_rvq.py](rvq_rerank/weighted_rvq.py)）：

```
標準: c_k = mean of cases assigned to k
加權: c_k = weighted_mean of cases assigned to k,  weights = w_i
```

其中 `w_i = retrieval_frequency(i)` = case `i` 在 train query top-20 中出現次數（[compute_rvq_weights.py](rvq_rerank/compute_rvq_weights.py)）。Assignment step 不變、encode/decode/LUT 完全不變——**只是 codebook tensor 那 `[M, K, D]` 個數字算法不同**。

**結果（K\* shift：damage ≤ 1pp 所需的最小 `top_k_cases`，越小越好）：**

| Config | Compression × | vanilla K* | **ranking K*** | Δ |
|---|---|---|---|---|
| M=8 K=256 | 384× | 2 | 2 | 0 |
| M=4 K=256 | 768× | 8 | **3** | **−5** |
| M=2 K=64 | 2048× | 15 | **10** | **−5** |
| M=1 K=16 | 6144× | 20 | **15** | **−5** |
| M=1 K=4 | 24576× | 20 | **15** | **−5** |

4/5 configs K* 下降 5 個位置（25–33% relative reduction）。Production regime（top_k=15–30）平均 R@10 lift **+0.2 到 +1.2 pp**。

##### Aggregation-aware (isolation / density)：negative result

進一步試 `w_agg = retrieval_frequency × isolation`（isolation = 1 − mean cosine to 50 NN，up-weight 孤立 case 因為它們沒 peer 提供 cause backup）以及反向 `w_agg_inv = retrieval_frequency × density`：

| Config | vanilla K* | ranking K* | agg (iso) K* | agg_inv (density) K* |
|---|---|---|---|---|
| M=4 K=256 | 8 | 3 | 3 | 3 |
| **M=2 K=64** | 15 | **10** | **20 ⚠️** | **20 ⚠️** |
| M=1 K=16 | 20 | 15 | 15 | 20 ⚠️ |
| M=1 K=4 | 20 | 15 | 15 | 15 |

兩個方向都在中度壓縮（M=2 K=64）退步到 K\*=20，比 vanilla 還差。Isolation/density 是 **first-order scalar 鄰域統計**，把 co-retrieval graph structure 壓縮成單一數字——丟掉太多訊息。**未來方向**：co-retrieval spectral clustering / 學習 gradient-influence weight 等 graph-structured aggregation-aware codebook 信號。

##### Method 總結

ABQ 的兩條核心 method-level claim（搭配 Phase 4 開頭四條 contribution 全貌）：

1. **Scaling law `D ≈ c · ε / K^{0.84}`**（R² = 0.83 pooled，encoder-invariant exponents），fine encoder 抗量化 2× raw encoder
2. **Importance-weighted codebook**（drop-in 替換 vanilla codebook 的 fit 程序，runtime path 完全不變），K\* 在 4/5 configs 降 ≈33%

這兩條把 ABQ 從「ColBERTv2 + light reranker 的應用」拉到「characterizing & exploiting structured-prediction retrieval under quantization」的 method-level contribution——名字掛在這個原理上，而非掛在現成的 RVQ。

#### 子套件結構

```
cause_inference/rvq_rerank/
├── __init__.py
├── rvq.py                # RVQCodebook：fit (k-means on residuals) / encode / LUT
├── reranker.py           # Reranker（compressed residual reranker + dense-residual oracle）；listwise KL + Δ MSE loss
├── fit_rvq.py            # Stage A1 CLI：fit codebook on z_train
├── build_rvq_index.py    # Stage A2 CLI：編 train + valid into index.pt（可選）
├── eval_sanity.py        # dense vs RVQ-only 純壓縮 sanity
├── run_sweep.py          # (M, K) 12 格 Pareto sweep
├── eval_harder.py        # 雙 regime stress eval（top_k_cases × sem_thr）
├── train_reranker.py     # compressed residual reranker 訓練（Regime B 早停）
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

FaCE-R 採用 **coarse-to-fine 二階段 cascade**——retrieval 端用 zero-shot frozen SigLIP2，attribution 端用 fine-tuned VLM-Lesion，兩階段 contribution 完全不重疊。

```
[Coarse retrieval, ZERO-SHOT]                    [Fine rerank, FINE-TUNED]
   raw SigLIP2 (frozen pretrained)                  VLM-Lesion + CEAM
        ↓                                                  ↓
   Phase 1 / Phase 3 / Phase 4                       on top-K cases only
        ↓                                                  ↓
   top-K cases  (12K→20 在 0.6 ms)                   candidate cause + α
   "frozen pretrained alignment"                    "lesion-grounded attribution"
```

Cascade 框架下 coarse stage 只負責挑 top-K cases，cause-level 打分**完全由 CEAM 負責**——CEAM 是 fine 階段的方法本體，不是跟檢索分數線性混合的修正項。end-to-end（raw coarse pool → CEAM）在 1573 valid 上 sem R@10 = **47.6%**、cluster R@10 = **39.6%**（[eval_ceah_compressed.py](eval_ceah_compressed.py) dense row），比純檢索 baseline（45.6% / 35.8%）**更準**（cluster +3.8pp）。（系統的「更快」是 coarse 端 ABQ／Phase 3 的功勞，CEAM 本身是 rerank 成本——兩個貢獻分開算。）（早期 hybrid 混合的 contribution-breakdown ablation 顯示純 CEAM 最強，留在 [inference.txt](inference.txt) `[B]`。）

### Stage 1 — Coarse retrieval（zero-shot, raw SigLIP2）

Production 走 `case_db_raw/` + `encoder_raw/` + `rvq_rerank_raw/`。1573 valid queries，top_k_cases=20，candidate-pool aggregation：

| Component | sem R@10 | sem MRR | per-q latency | Index memory @ 12K |
|---|---|---|---|---|
| Phase 1 hungarian (teacher) | 45.6% | 0.300 | 15.4 ms | 200 MB (multi-vec) |
| Phase 3 DeepSets dual-target | **45.6%** | **0.303** | **1.1 ms** (14×) | 39 MB |
| Phase 4 RVQ-only M=4 K=256 | 45.1% | 0.302 | 0.10 ms (152×) | **0.05 MB** (768× compression vs fp32 dense) |
| Phase 4 + compressed residual reranker（Light） | 45.1% | 0.300 | 0.56 ms (27×) | 13 MB (reranker ckpt dominated) |
| Phase 4 + dense-residual oracle（Full analytic） | **45.6%** | 0.301 | 0.22 ms (69×) | 3.0 GB @ 1M scale (z bank needed) |

→ Coarse stage 所有變體 R@10 都在 dense (45.6%) ±0.6 pp 內，**production aggregation buffer 對 RVQ 壓縮免疫**。

**Regime B stress benchmark**（top_k_cases=1，無 aggregation buffer，純 ANN-style；retrieval-side proxy，未接 CEAM）。這是**唯一能看出 compressed residual reranker 價值的設定**——production cascade（top_k=20 buffer + 下游 CEAM）不依賴它（見 Phase 4 [結果 — 端到端](#結果--端到端-ceam-主張)）。此表單純做 raw（production Stage 1 encoder）vs fine-tuned（ablation）的 quantization-robustness Pareto 對照：

| Compression × | Method | **raw R@10** | **fine-tuned R@10** | 哪個贏 |
|---|---|---|---|---|
| 1× (dense fp32) | baseline | 51.7% | 53.7% | fine-tuned +2.0 pp |
| 768× (M=4 K=256) | RVQ-only | 48.4% | 50.6% | fine-tuned +2.2 pp |
|  | **+ compressed residual reranker（Light）** | **51.0%** | 51.8% | fine-tuned +0.8 pp |
|  | + dense-residual oracle（Full analytic） | 51.6% | 53.7% | fine-tuned +2.1 pp |
| 2048× (M=2 K=64) | RVQ-only | 44.6% | 46.3% | fine-tuned +1.7 pp |
|  | **+ compressed residual reranker（Light）** | 46.2% | **50.0%** | **fine-tuned +3.8 pp** |
|  | + dense-residual oracle（Full analytic） | 50.5% | 51.5% | fine-tuned +1.0 pp |
| 6144× (M=1 K=16) | RVQ-only | 36.8% | 37.2% | tied |
|  | **+ compressed residual reranker（Light）** | 39.7% | **46.2%** | **fine-tuned +6.5 pp** |
|  | + dense-residual oracle（Full analytic） | 44.6% | 45.8% | fine-tuned +1.2 pp |

Gap recovered (compressed residual reranker vs RVQ-only) 隨壓縮率變化的不對稱：

| Compression × | Raw recovery | Fine-tuned recovery |
|---|---|---|
| 768× (M=4 K=256) | **79%** | 39% ← raw 勝 |
| 2048× (M=2 K=64) | 22% | **56%** ← fine-tuned 勝 |
| 6144× (M=1 K=16) | 19% | **54%** ← fine-tuned 勝 |

**詮釋**：raw features 在中度壓縮（M=4 K=256, 768×）下殘差結構乾淨、compressed residual reranker 學得很好（79% recovery）；但壓到極端時 raw 比 fine-tuned 更早 degrade（fine-tune 的 in-domain training 隱含把 features 推到更 quantize-friendly 的空間）。Production Stage 1 用 **raw + M=4 K=256**（encoder / 壓縮率的選點）；極端壓縮的 buffer-free deployment 場景才需要 fine-tuned encoder + compressed residual reranker。注意這整張表是 buffer-free proxy——production 真正的 top_k=20 + CEAM 路徑下，壓縮已經免費、compressed residual reranker 不參與（見上方端到端表）。完整 Pareto plot 見 [`outputs/rvq_rerank/pareto_compression_vs_R10.png`](outputs/rvq_rerank/pareto_compression_vs_R10.png)。

**Scale-up benchmark @ N=1M**（合成 z bank、無 recall 報告）：

| Method | latency (bs=1) | memory | compression |
|---|---|---|---|
| Phase 3 DeepSets dense | 5.15 ms | 3.0 GB | 1× |
| Phase 4 RVQ-only | 0.19 ms (27× faster) | 4 MB | **768×** |
| Phase 4 + compressed residual reranker（Light） | 0.59 ms (9× faster) | 17 MB | 181× |

Coarse stage 在 1M case bank 維持 sub-ms / sub-20MB——cascade 第一階段的 scalability 主軸。

### Stage 2 — Fine rerank（fine-tuned, VLM-Lesion + CEAM）

Production 走 `case_db/`（fine-tuned VLM-Global + VLM-Lesion + LocalGlobalFusionWrapper）。對 Stage 1 取的 top-20 cases 的 candidate cause pool（~87 unique causes）做 attribution-aware 重排。

#### 為什麼不直接用 Phase 1 的相似度排病因？（CEAM 的貢獻所在）

Phase 1 確實能輸出病因排序，但它是一個 **相似度黑箱**——`Σ_j w_j · max_g cos(cause, evidence)` 給出排名，卻**無法說明「這個病因是因為哪個病灶」**，而且對所有病因用同一組固定 α/β 權重。CEAM 提供 Phase 1 **結構上做不到**的三件事：

1. **架構強制的 faithful 歸因（主貢獻，教授原始要求）**：每個病因輸出一個 softmax α over (global, text, lesion₁..ₙ)，明確指出證據來源；並用反事實遮罩**驗證**這個歸因真的 load-bearing（遮 lesion 掉分 15× random；見下）。Phase 1 沒有可驗證的 per-evidence 歸因。
2. **病因條件式證據路由**：CEAM 學到 lesion-type 病因該看 lesion（l/g 1.39）、global-type 該看 global（0.44）；Phase 1 對每個病因都同一組權重。
3. **歸因不是免費的，Phase 1 接不上**：faithfulness 反轉 ablation 證明，即使把 CEAM 接到 raw 特徵 faithfulness 就反轉（+0.038→−0.031）——faithful 歸因需要 CEAM 架構 + fine-tuned lesion 特徵兩者，Phase 1 的 max-cosine 兩者都沒有。

**在同一候選池上 CEAM 還順帶更準（by-product，不是主張）：**

| Setting | sem R@1 | sem R@10 | sem MRR | cluster R@10 | cluster MRR |
|---|---|---|---|---|---|
| Coarse only (raw retrieval, Phase 1/3 on `case_db_raw`) | 22.2% | 45.6% | 0.300 | 35.8% | 0.235 |
| **+ CEAM 歸因重排 (raw pool → `case_db`)** | 21.0% | **47.6%** | **0.303** | **39.6%** | **0.269** |
| **Δ** | −1.2 pp | **+2.0 pp** | +0.003 | **+3.8 pp** | **+0.034** |

→ 增益集中在 **cluster (disease-topic) recall +3.8 pp**；R@1 略降、MRR 持平——CEAM 把更多 cluster-correct 病因拉進 top-10。**這個準確度增益是佐證（證明歸因不是犧牲準確度換來的解釋），主貢獻是上面的可驗證歸因本身。** 數字量自 [eval_ceah_compressed.py](eval_ceah_compressed.py) 的 dense row（1573 valid，LLM 466 cluster）；coarse-only baseline 取 [Phase 1 ablation](#phase-1-ablationvlm-dependency) 的 raw retrieval。

**Faithfulness**（lesion masking 反事實實驗）：

| Bucket | Lesion-mask drop | Random-mask drop | Ratio |
|---|---|---|---|
| All | +0.035 | +0.003 | 12× |
| Lesion-type causes | +0.038 | +0.004 | 9.7× |
| **N≥3 lesions** | +0.053 | +0.004 | **15×** |

→ 遮 lesion 對 multi-lesion case 的分數影響是隨機遮的 **15 倍**——α 是 load-bearing 不是裝飾。

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
| **Fine rerank** | **必須 fine-tune (VLM-Lesion)** | raw case_db 上重訓 CEAM：retrieval 持平 (sem R@10 46.4%)，**faithfulness 反轉** (lesion drop +0.038 → **−0.031**) |

→ Retrieval-side fine-tune 全部沒測到改善（zero-shot 飽和）；attribution-side fine-tune 是 architecturally required（LocalGlobalFusionWrapper 提供 lesion-specialized 視覺特徵）。Cascade 把這兩種不對稱的依賴乾淨分開。

---

## 論文敘事建議（cascade framing）

> **FaCE-R is a coarse-to-fine cascade for case-based fish disease cause prediction with lesion-grounded explainability.**
>
> **Coarse stage — zero-shot case retrieval at scale.** Three components progressively accelerate retrieval over a frozen pretrained SigLIP2 backbone on a 12,780-case database (production runs against `case_db_raw/`): **Phase 1** multi-vector Hungarian lesion-set matching reaches sem R@10 = 45.6% / MRR = 0.300 at 15.4 ms / query; **Phase 3** DeepSets dual-target distillation (listwise KL on the Phase 1 score matrix + case-to-cause SupCon InfoNCE on 56k cause text embeddings) collapses retrieval to a single-vector cosine, preserving quality (R@10 = 45.6%, MRR = 0.303 — slightly better than teacher) at 1.1 ms / query (14× faster); **Phase 4 — ABQ (Aggregation-Buffered Quantization)** compresses the frozen Phase 3 embeddings with codebook quantization (instantiated with Residual Vector Quantization). At M=4 K=256 (768× compression), scale-up benchmarks at N=1M show 181× memory compression at 17 MB and 0.19–0.59 ms / query while dense degrades to 5.15 ms / query at 3.0 GB. The method-level contribution is **not the compression itself but a scaling law** `D ≈ c·ε/K^{0.84}` (R²=0.83 pooled, encoder-invariant exponents) characterizing how the cause-aggregation buffer absorbs quantization damage, yielding an operating-point selector and an **importance-weighted codebook** that cuts the required aggregation depth K\* by 25–33% as a drop-in codebook refit. End-to-end through the CEAM fine stage, the production cascade (top_k_cases=20) is essentially lossless under aggressive compression: even 6,144× (16 prototypes) costs only 0.6 pp final sem R@10 and 0 pp cluster R@10. A compressed residual reranker (Light implementation, Δ ≈ qᵀe) is kept as a **design-space probe, not a production component** — end-to-end it is redundant (even marginally harmful) in the production regime, and only recovers 39–79% of the RVQ gap in a buffer-free regime (top_k_cases=1, single-case ANN lookup) the production cascade never enters; the free structural buffer dominates the learned correction. A full compression sweep on raw vs fine-tuned encoders further shows fine-tuning yields ~2× quantization robustness (intercept c_raw/c_fine ≈ 2.06). **Cross-ablation** (Phase 1 raw vs fine-tuned VLMs; supervised projection MLP probe on frozen VLM-Lesion; Phase 3 raw distillation; Phase 4 raw RVQ + compressed residual reranker) consistently shows in-domain VLM fine-tuning provides no measurable retrieval improvement, confirming the coarse stage operates in a **zero-shot vision-language regime**.
>
> **Fine stage — CEAM, faithful lesion-grounded attribution (the core contribution).** Phase 1 retrieval already ranks causes by similarity aggregation, but it is a black box: it cannot say *which lesion* drives a cause, and it weights every cause with the same fixed α/β. **CEAM provides what Phase 1 structurally cannot:** for each candidate cause it outputs a softmax α distribution over (global, text, lesion_1..N) evidence tokens via a multiplicative score head (sigmoid global × sigmoid local) that **architecturally forces** local-evidence faithfulness (sigmoid α and additive scoring both collapse to α-saturation — v1/v2 ablations). This attribution is *verified*, not decorative: counterfactual lesion masking drops the score by **15× the random-mask baseline** on multi-lesion cases (N ≥ 3), and the lesion/global α ratio inverts 3.2× across cause types (0.44 global-type → 1.39 lesion-type), i.e. CEAM learns *cause-conditional* evidence routing. Crucially the attribution is **not obtainable for free**: replacing the fine-tuned case_db with raw SigLIP2 features preserves retrieval but **reverses** faithfulness (lesion drop +0.038 → −0.031) — faithful attribution requires both the CEAM head and fine-tuned lesion features, neither of which Phase 1's max-cosine has. As a *by-product* (not the claim), CEAM also reranks more accurately than Phase 1 on the same candidate pool: **+2.0 pp sem R@10 (45.6 → 47.6)** and **+3.8 pp cluster R@10 (35.8 → 39.6)** on 1,573 valid queries, confirming the explainability does not cost accuracy.
>
> **Asymmetric VLM dependency** is the cascade's structural claim. Retrieval-side fine-tuning is consistently a no-op (Phase 1 raw vs fine-tuned, Phase 3 raw distillation, Phase 4 raw RVQ, plus the supervised projection MLP probe); attribution-side fine-tuning is architecturally load-bearing (LocalGlobalFusionWrapper). The cascade enforces this split by routing coarse retrieval through raw `case_db_raw/` and fine rerank through fine-tuned `case_db/`: a fast zero-shot retriever generates candidates, and CEAM is the sole cause-level scorer (no score mixing). An earlier hybrid that linearly combined retrieval and CEAM scores was dropped — the contribution-breakdown ablation showing CEAM alone is strongest is kept in inference.txt `[B]`.
>
> **Robustness & deployment**: medical / colloquial / none 三種推論模式 retrieval 差 ≤ 0.003 MRR — vision-only deployment 完全可行，text 缺席時 attention 自動補位給 lesion，cause-type-aware 行為 preserved. Multi-lesion attribution at N≥3 spreads evenly across all lesions (concentration ≈ 1/N), demonstrating a **collective visual evidence** strategy rather than a single dominant-lesion heuristic.
>
> **Negative results**: supervised projection MLP on frozen lesion features (no measurable lift, ablation reinforces zero-shot framing); 1-stage full-vocab cause retrieval bypassing the case-pool aggregation (3.5× worse than 2-stage cascade); leave-one-out mining showing 0.4% of train cases (~53 / 12,780) have all GT causes outside retrieval reach, defining a structural R@K ceiling near 44.5% for retrieval-only metrics that the fine stage then pushes past.

---

## 已知限制

- **Cause-emb 跟 lesion-emb 在不同 VLM 空間**：CEAM 必須學跨空間映射，可能限制 lesion attribution 的精細度
- **HDBSCAN 細粒度過分割**：disease topic 級別需要重新粗粒度聚類（v100），否則 cluster R@K 假性偏低
- **N≥3 attribution spread**：模型不會從多 lesion 中挑單一主病灶。若臨床需要「指向單一病灶」需額外 hard top-1 約束
- **Text 對 retrieval 沒實質幫助**：text mode ablation 顯示 medical/colloquial/none 三模式 retrieval 差 ≤ 0.003 MRR；text 在 attribution 端有作用（吸收 0.13–0.26 attention mass），但不影響 top-K 結果。換言之，部署若不收集文字描述也不會丟分。
