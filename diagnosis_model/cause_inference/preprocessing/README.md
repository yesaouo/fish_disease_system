# Cause Clustering

Cause taxonomy 是 cause_inference 的 **evaluation 用 lenient recall 標準**(disease-topic R@K),不進任何訓練 loss。論文只用兩條 taxonomy:

| 角色 | 檔名 | 方法 | clusters |
|---|---|---|---|
| **Paper main** | `outputs/cause_clusters_llm.json` | Ollama LLM (`gemma4:26b`) | 466 |
| **Paper baseline (LLM-free)** | `outputs/cause_clusters_hdbscan.json` | PCA+UMAP+HDBSCAN+reassign-singletons | 100 |

兩條都跑一次即可,不需要重建除非 cause 字串集變更。Cmd 都從 repo root 執行:

```bash
cd /mnt/ssd/YJ/fish_disease_system
PY=/home/lab603/anaconda3/envs/SDM/bin/python
```

## 0. 前置: text embedding cache

兩條 taxonomy 都需要 `outputs/cause_cache/`(unique cause strings + 對應 VLM-Lesion text embedding)。建一次:

```bash
$PY -m diagnosis_model.cause_inference.preprocessing.recluster_causes cache \
  --coco_files data/detection/coco/_merged/train/_annotations.coco.json \
               data/detection/coco/_merged/valid/_annotations.coco.json \
               data/detection/coco/_merged/test/_annotations.coco.json \
  --cache_dir diagnosis_model/cause_inference/outputs/cause_cache \
  --encoder_backend vlm \
  --vlm_path diagnosis_model/vl_classifier/outputs/siglip2_base_patch16_224_multipos_fusion_en_zh
```

Cache 收集 train+valid+test 三 split 的 unique cause(test 一定要納入,否則 test query 的 cause 沒有 cluster id)。Cluster taxonomy 是 IR-style topic-relevance label,不流入模型訓練;跨 split 收 cause 字串只是 metric 定義,不是 leakage。

## 1. Paper main — LLM taxonomy (`cause_clusters_llm.json`)

需要本機 Ollama 服務:

```bash
ollama serve
ollama pull gemma4:26b
```

```bash
$PY -m diagnosis_model.cause_inference.preprocessing.cause_cluster_llm \
  --cache_dir diagnosis_model/cause_inference/outputs/cause_cache \
  --output diagnosis_model/cause_inference/outputs/cause_clusters_llm.json \
  --batch_size 10 \
  --shard_size 100 \
  --merge_batch_size 20
```

- 慢(數小時),每批 checkpoint;`Ctrl+C` 中斷後重跑同 command 會接續。
- 若 schema mode 不穩,加 `--json_mode`;若 chat 介面回應品質差,加 `--use_generate`。
- 從頭重跑加 `--overwrite`。

## 2. Paper baseline — HDBSCAN taxonomy (`cause_clusters_hdbscan.json`)

一行封裝(固定論文參數,無 hyperparameter):

```bash
$PY -m diagnosis_model.cause_inference.preprocessing.recluster_causes baseline \
  --cache_dir diagnosis_model/cause_inference/outputs/cause_cache
```

等價於 `reduce → cluster → reassign-singletons` 三步,固定參數為:

| 步驟 | 參數 |
|---|---|
| PCA | `n=50` |
| UMAP | `n_neighbors=15, n_components=5, metric=cosine, random_state=42` |
| HDBSCAN | `cluster_selection_method=eom, min_cluster_size=100` |
| Reassign singletons | `cosine_threshold=0.70, margin=0.0, min_real_cluster_size=100` |

幾秒到一分鐘內完成。Output schema 與 LLM 版相同:`cause_id_to_canonical` / `original_to_cause_id` / `cluster_meta`。

## 3. 在 eval 中使用

`phase1_baseline.py` / `eval_ceah.py` / `eval_ceah_compressed.py` / `train_projection.py` 的 `--cluster_json` 預設都指向 LLM main:

```bash
$PY -m diagnosis_model.cause_inference.phase1_baseline ...       # 自動用 cause_clusters_llm.json
$PY -m diagnosis_model.cause_inference.eval_ceah ...             # 同上
```

跑 baseline 對照:

```bash
$PY -m diagnosis_model.cause_inference.eval_ceah \
  --cluster_json diagnosis_model/cause_inference/outputs/cause_clusters_hdbscan.json \
  ...
```

## 4. Dev / 探索性 sub-commands

`recluster_causes.py` 還有 `reduce` / `cluster` / `sweep` / `reassign-singletons` / `merge-similar-clusters`,用於 HDBSCAN 參數探索或客製 taxonomy。**這些不出現在論文 pipeline**,只是建 `baseline` 之前的開發歷史與 ablation 工具。`--help` 看用法。

## 5. Schema reference

```json
{
  "cause_id_to_canonical": {"<cluster_id>": "<canonical cause string>"},
  "original_to_cause_id": {"<raw cause string>": <cluster_id>},
  "cluster_meta":         {"<cluster_id>": {"size": N, "members": [...]}}
}
```

Loader 與 normalizer 在 `cause_cluster_json.py`。
