# Cause Clustering Usage

這份說明是給目前這版
[`recluster_causes.py`](./recluster_causes.py)
用的。

建議從 repo root 執行：

```bash
cd /mnt/ssd/YJ/fish_disease_system
```

可以用兩種方式執行，擇一即可：

```bash
python diagnosis_model/cause_inference/data/recluster_causes.py --help
python -m diagnosis_model.cause_inference.data.recluster_causes --help
```

目前 CLI 子命令有：

- `cache`
- `reduce`
- `cluster`
- `sweep`
- `reassign-singletons`

## 1. 先做文字 embedding cache

### 用 VLM encoder

```bash
python diagnosis_model/cause_inference/data/recluster_causes.py cache \
  --coco_files data/detection/coco/_merged/train/_annotations.coco.json \
               data/detection/coco/_merged/valid/_annotations.coco.json \
  --cache_dir diagnosis_model/cause_inference/data/cause_cache \
  --encoder_backend vlm \
  --vlm_path diagnosis_model/vl_classifier/outputs/siglip2_base_patch16_224_multipos_fusion_en_zh \
  --text_batch_size 1024 \
  --max_length 64
```

### 用 Hugging Face model

```bash
python diagnosis_model/cause_inference/data/recluster_causes.py cache \
  --coco_files data/detection/coco/_merged/train/_annotations.coco.json \
               data/detection/coco/_merged/valid/_annotations.coco.json \
  --cache_dir diagnosis_model/cause_inference/data/cause_cache_hf \
  --encoder_backend hf \
  --hf_model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  --hf_pooling mean \
  --text_template "{cap}"
```

### 用 sentence-transformers backend

```bash
python diagnosis_model/cause_inference/data/recluster_causes.py cache \
  --coco_files data/detection/coco/_merged/train/_annotations.coco.json \
               data/detection/coco/_merged/valid/_annotations.coco.json \
  --cache_dir diagnosis_model/cause_inference/data/cause_cache_st \
  --encoder_backend sentence-transformers \
  --hf_model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  --text_template "{cap}"
```

輸出會寫到 `--cache_dir`：

- `embeddings.pt`
- `texts.json`
- `cache_meta.json`

## 2. 先單獨做 PCA + UMAP

如果你想把 reduce 結果存下來，避免每次 clustering 都重跑：

```bash
python diagnosis_model/cause_inference/data/recluster_causes.py reduce \
  --cache_dir diagnosis_model/cause_inference/data/cause_cache \
  --output temp/cause_reduced.npy \
  --pca_components 50 \
  --umap_n_neighbors 15 \
  --umap_min_dist 0.0 \
  --umap_n_components 5 \
  --umap_metric cosine \
  --random_state 42
```

這會產生：

- `temp/cause_reduced.npy`
- `temp/cause_reduced.npy.meta.json`

## 3. 直接做 clustering 輸出 JSON

### 方式 A: 讓 `cluster` 自己跑 reduce

```bash
python diagnosis_model/cause_inference/data/recluster_causes.py cluster \
  --cache_dir diagnosis_model/cause_inference/data/cause_cache \
  --output temp/cause_clusters.json \
  --pca_components 50 \
  --umap_n_neighbors 15 \
  --umap_min_dist 0.0 \
  --umap_n_components 5 \
  --umap_metric cosine \
  --random_state 42 \
  --cluster_selection_method eom \
  --min_cluster_size 6 \
  --report_top_n 20
```

### 方式 B: 使用已經存好的 reduced embedding

```bash
python diagnosis_model/cause_inference/data/recluster_causes.py cluster \
  --cache_dir diagnosis_model/cause_inference/data/cause_cache \
  --reduced_path temp/cause_reduced.npy \
  --output temp/cause_clusters.json \
  --cluster_selection_method eom \
  --min_cluster_size 6 \
  --report_top_n 20
```

輸出 JSON schema 會是：

- `cause_id_to_canonical`
- `original_to_cause_id`
- `cluster_meta`

## 4. 做 sweep 找參數

```bash
python diagnosis_model/cause_inference/data/recluster_causes.py sweep \
  --cache_dir diagnosis_model/cause_inference/data/cause_cache \
  --output temp/cause_clusters.json \
  --pca_components 50 \
  --umap_min_dist 0.0 \
  --umap_n_components 5 \
  --umap_metric cosine \
  --random_state 42 \
  --cluster_selection_method eom \
  --sweep_umap_n_neighbors 3 5 8 12 \
  --sweep_umap_n_components 3 5 8 \
  --sweep_min_cluster_size 3 6 9 12 \
  --sweep_save_summary_only
```

如果沒有給 `--sweep_umap_n_components`，會沿用 `--umap_n_components` 的單一值。
如果拿掉 `--sweep_save_summary_only`，每個組合都會另外輸出一份 JSON。

summary 會寫成：

```text
temp/cause_clusters.json.sweep.json
```

## 5. 把 singleton 回掛到 real cluster

這一步是用 embedding cosine similarity，把 size=1 的 cluster 掛回既有 real cluster。

```bash
python diagnosis_model/cause_inference/data/recluster_causes.py reassign-singletons \
  --cache_dir diagnosis_model/cause_inference/data/cause_cache \
  --input temp/cause_clusters.json \
  --output temp/cause_clusters_reassigned.json \
  --cosine_threshold 0.92 \
  --margin 0.03 \
  --min_real_cluster_size 2 \
  --report_top_n 20
```

常用參數：

- `--cosine_threshold`: 最低相似度門檻
- `--margin`: best 和 second-best 的差距門檻
- `--min_real_cluster_size`: 至少多大的 cluster 才能當 anchor

## 6. 常見流程

### 最常見：一次建 cache，之後反覆 cluster

```bash
python diagnosis_model/cause_inference/data/recluster_causes.py cache ...
python diagnosis_model/cause_inference/data/recluster_causes.py cluster ...
python diagnosis_model/cause_inference/data/recluster_causes.py sweep ...
python diagnosis_model/cause_inference/data/recluster_causes.py reassign-singletons ...
```

### 如果你要一直試 HDBSCAN 參數

先把 reduce 存起來：

```bash
python diagnosis_model/cause_inference/data/recluster_causes.py reduce ...
python diagnosis_model/cause_inference/data/recluster_causes.py cluster --reduced_path ...
```

## 7. 相關檔案

- `cause_texts.py`: 從 COCO 收集 cause strings
- `text_embedding_cache.py`: 建立 / 載入文字 embedding cache
- `dim_reduction.py`: PCA + UMAP
- `hdbscan_clustering.py`: HDBSCAN
- `cause_cluster_json.py`: labels 轉 cause JSON、quality report
- `singleton_reassign.py`: singleton 回掛邏輯
- `recluster_causes_old.py`: 舊版參考
- `recluster_causes_oldv2.py`: 舊版參考

## 8. 小提醒

- `cluster` 如果沒有給 `--reduced_path`，會自己重跑 PCA + UMAP。
- `cache` 是最花時間的步驟，通常只需要跑一次。
- `sweep` 主要拿來找 `umap_n_neighbors` / `umap_n_components` / `min_cluster_size`。
- singleton 回掛是後處理，不會直接改 HDBSCAN 本身的結果生成方式。
