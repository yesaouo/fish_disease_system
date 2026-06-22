# DDXPlus Integration Plan

This note defines the DDXPlus adaptation contract for `cause_inference`.
The goal is to validate FaCE-R outside fish images by replacing visual lesions
with clinical evidence tokens while preserving the existing retrieval and CEAM
interfaces.

## Dataset Role

DDXPlus provides synthetic patients for automatic diagnosis:

- `PATHOLOGY`: strict ground-truth disease label.
- `DIFFERENTIAL_DIAGNOSIS`: ranked disease candidates with probabilities.
- `EVIDENCES`: symptoms and antecedents observed for the patient.
- `AGE` and `SEX`: patient-level demographics.

For this repository, DDXPlus is a text-only analogue of the fish disease setup:

```text
fish image case:
  global image embedding + lesion embeddings -> cause labels

DDXPlus case:
  patient summary embedding + evidence embeddings -> pathology labels
```

## Case DB Contract

DDXPlus builds emit a **sharded** layout to keep peak RAM bounded during
encoding (full release is ~1.03M patients — `1,025,602` train per build log):

```text
train_cases_00000.pt
train_cases_00001.pt
...
valid_cases_00000.pt
test_cases_00000.pt
cause_text_embs.pt
meta.json   # lists train_shards / valid_shards / test_shards
```

Consumers do **not** read `{split}_cases.pt` directly. The shared
`diagnosis_model.cause_inference.phase1_baseline.load_cases(case_db_dir, split)`
helper reads `meta.json` and transparently concatenates the shards; if a single
`{split}_cases.pt` is present (legacy fish case_db layout) it falls back to
that. Every Phase 1 / Phase 2 entry point (`phase1_baseline.py`,
`build_train_candidate_pool.py`, `train_ceah.py`, `eval_ceah.py`, and the four
DDXPlus scripts) goes through this helper, so the fish pipeline is unaffected.

> Caveat: shards bound build-time peak RAM. For eval-time, the full 1M
> bank's lesion stack is ~30 GB even in bf16 and won't fit on a single 32 GB
> GPU as one monolithic tensor. The three DDXPlus eval scripts therefore
> support two modes (default + `--stream`) — see [Memory modes](#memory-modes-default-subsample-vs---stream).

Per-case keys are unchanged:

```python
{
    "case_id": int,
    "image_id": int,                  # reused as patient row id
    "split": "train" | "valid" | "test",
    "file_name": str,                 # synthetic stable id, e.g. ddxplus/train/123
    "global_emb": Tensor[D],          # full patient summary (Age + Sex + Evidence concatenated)
    "text_colloquial_emb": Tensor[D], # duplicate of global_emb on DDXPlus — no
                                       # image/text modality split exists in this
                                       # domain; train/eval should disable text via
                                       # --text_dropout 1.0 / --text_kind none
    "text_medical_emb": Tensor[D],    # duplicate (same reason)
    "lesion_embs": Tensor[N, D],      # atomic evidence tokens, ORDER:
                                       #   [Age: X], [Sex: Y], evidence_1, evidence_2, ...
                                       # Age/Sex prepended (since 2026-05-28 schema)
                                       # so CEAM can attribute α to each demographic
                                       # / symptom separately rather than collapsing
                                       # them inside global.
    "lesion_boxes_xywh": Tensor[N, 4],# dummy zeros for compatibility
    "causes": list[str],              # [PATHOLOGY] — strict GT pathology name(s)
    "pathology_emb_idx": int,         # ★ single strict pathology cause-table index
                                       # — used for pathology R@K, CEAM positive_mask,
                                       # soft-label GT
    "cause_emb_indices": list[int],   # ★ EXPANDED: [pathology_idx, ddx1_idx, ddx2_idx, ...]
                                       # — used by build_candidate_pool so DDX
                                       # alternatives land in the pool and give
                                       # NDCG@K real ranking space
    "ddx": list[dict],                # original DDX names + probs (valid/test only,
                                       # for soft-label NDCG eval)
    "evidence_texts": list[str],      # decoded EVIDENCES, prepended with
                                       #   ["Age: X", "Sex: Y"]
                                       # (valid/test shards only — train shards
                                       # drop string fields)
}
```

`lesion_embs` is intentionally kept as the tensor key even though the semantic
meaning is evidence embeddings. This keeps the current CEAM code path usable:
global/text/evidence tokens are still projected as evidence and scored against
candidate causes.

### Why `pathology_emb_idx` is separate from `cause_emb_indices`

DDXPlus has only 1 pathology per case, so the old `cause_emb_indices = [pathology]`
gave `build_candidate_pool` a single-cause pool (mean=1.08) that flattened
NDCG@5+ (nothing to rank past position 1). The 2026-05-28 schema splits the
two roles:

| Field | Type | Used by | Why |
|---|---|---|---|
| `pathology_emb_idx` | int | pathology R@K, CEAM positive_mask, soft-label GT | strict 1-of-49 GT — DDX alternatives must NOT count as pathology hits |
| `cause_emb_indices` | list[int] (pathology + DDX) | `build_candidate_pool` for retrieved cases | pool diversity — DDX alternatives become hard negatives in pool, expanding from mean=1.08 to ~10-30 unique causes per query |

Fish builds (pre-2026-05-28) don't write `pathology_emb_idx`; downstream
consumers fall back to `cause_emb_indices` (fish's multi-cause GT semantics).

## Text Construction

`global_text` (still full descriptor — Phase 1 retrieval needs a rich
case-level vector):

```text
Age: {AGE}. Sex: {SEX}. Evidence: {decoded evidence 1}; ...
```

Evidence tokens (now include Age/Sex as the first two atomic tokens):

```text
lesion_emb_1 = encode("Age: {AGE}")
lesion_emb_2 = encode("Sex: {SEX}")
lesion_emb_3 = encode("{question_en}: yes")            # binary evidence
lesion_emb_4 = encode("{question_en}: {value_meaning.en}")  # categorical
lesion_emb_5 = encode("{question_en}: {value_meaning.en}")  # multi-choice (one token per selected value)
...
```

Use `release_evidences.json` to decode evidence IDs and values. Keep raw IDs in
metadata for debugging.

### Why CEAM on DDXPlus needs `--text_dropout 1.0` / `--text_kind none`

`text_colloquial_emb` and `text_medical_emb` are populated as duplicates of
`global_emb`. They exist for backward compatibility with the fish 3-token CEAM
architecture (image global + textual notes + lesion crops), where text is a
separately-encoded modality. DDXPlus has no such modality split — the patient
summary is the only available case-level descriptor. Training with
`--text_dropout 1.0` and evaluating with `--text_kind none` masks the text
token to zero throughout, which avoids feeding CEAM a redundant copy of the
global token. Faithfulness on the previous (v1) schema confirmed `no_text`
drop = -0.0002 (essentially noise), validating that text adds no information
on DDXPlus.

## Cause Table

`cause_text_embs.pt` should include all disease names observed in:

- train/valid/test `PATHOLOGY`
- optionally every disease listed in `DIFFERENTIAL_DIAGNOSIS`

Recommended default:

```text
causes = union(PATHOLOGY, DIFFERENTIAL_DIAGNOSIS disease names)
```

This allows evaluation against both strict pathology and soft differential
diagnosis targets.

## Phase 1 Metrics

Reuse current semantic retrieval metrics for strict `PATHOLOGY` first:

- R@1/R@5/R@10/R@20
- MRR
- candidate pool coverage

Add DDXPlus-specific metrics in a later pass:

- top-k hit of strict `PATHOLOGY`
- NDCG@K against `DIFFERENTIAL_DIAGNOSIS` probabilities
- coverage of differential diagnosis entries in candidate pool

## CEAM / Faithfulness Mapping

CEAM can be reused directly:

```text
global token  = patient summary embedding
text token    = optional duplicated summary embedding or disabled
lesion tokens = evidence embeddings
cause token   = disease embedding
```

Faithfulness becomes evidence masking:

- `no_lesion` in existing code maps to `no_evidence`.
- `no_random` remains a random evidence masking baseline.
- Top-alpha evidence can be compared against the disease's known symptoms and
  antecedents in `release_conditions.json`.

## Commands

Set the project Python once:

```bash
PY=/home/lab603/anaconda3/envs/SDM/bin/python
cd /mnt/ssd/YJ/fish_disease_system
```

### 1. Build DDXPlus Case DB

Expected input files are the official DDXPlus release files:

```text
release_train_patients.zip
release_validate_patients.zip
release_test_patients.zip
release_evidences.json
release_conditions.json
```

Build a full case DB:

```bash
$PY -m diagnosis_model.cause_inference.ddxplus.build_case_database \
  --train_csv data/ddxplus/release_train_patients.zip \
  --valid_csv data/ddxplus/release_validate_patients.zip \
  --test_csv data/ddxplus/release_test_patients.zip \
  --evidences_json data/ddxplus/release_evidences.json \
  --conditions_json data/ddxplus/release_conditions.json \
  --output_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
  --embedding_backend transformer \
  --text_encoder google/siglip2-base-patch16-224 \
  --include_ddx_causes \
  --chunk_size 8192 \
  --shard_size 65536 \
  --embedding_dtype bf16
```

### `--embedding_dtype`: storage dtype for shards / cause_text_embs

Use `--embedding_dtype bf16` by default for DDXPlus shards and `cause_text_embs.pt`.
Downstream loaders upcast embeddings to fp32 before CEAM, avoiding mixed-dtype errors.
Use `fp32` only on older GPUs without native bf16 support.

### `--lesion_match max_mean` (now the global default)

2026-05-26 起 `phase1_baseline.compute_case_similarities` 的 default 從
`hungarian` 翻成 `max_mean`（fish 和 DDXPlus 都適用）。前者在 1M scale 不可行；
fish ablation 上 max_mean 在每個 metric 都微幅勝出（|Δ| ≤ 0.5 pp R@K / MRR /
coverage）—— 詳見 [`ablations/lesion_match_ranking_equiv`](../ablations/lesion_match_ranking_equiv.py)。
下方命令明寫 `--lesion_match max_mean` 是冗餘的，但保留以便文件自我說明。

| Stage | hungarian (per-pair LP, Python loop) | max_mean ✅ (vectorized GPU `scatter_reduce`) |
|---|---|---|
| per-pair cost | 1.7 µs LP solve | embed in 1 kernel |
| per-query @ 1M train | **~6 sec** | **~63 ms** (~90×) |
| `eval_retrieval` 全 valid (134k × 1M) | ~9 天 (infeasible) | **~2.3 hr** |
| `build_train_candidate_pool` (1M × 1M) | ~70 days (infeasible) | **~17 hr** |
| `train_ceah` validation (300 q subset/epoch) | ~30 min/epoch | **~20 s/epoch** |
| `eval_ceah` (134k × 1M + CEAM) | infeasible | **~3 hr** |
| `faithfulness_eval` (134k × 1M, 6 masks) | infeasible | **~8 hr** |

> 注:DDXPlus 標準切分 1.3M 約 80/10/10 → train 1.03M / valid 134k / test 134k。
> 上方表用 valid 134k 估算;若你只想跑 paper-headline 數字可用 `--max_queries 2000`
> 取代表式 sample,~2 分鐘完成。

要重現 fish 論文舊數字（Table B1、B2、E1 等用 hungarian 的歷史值）顯式傳
`--lesion_match hungarian` 即可——只要 train bank ≤ ~10k 都跑得動。

### Memory modes: default subsample vs `--stream`

DDXPlus 1M bank 在 bf16 是 ~31 GB，加上 query / working buffer 後直接超出 32 GB
GPU。三個 eval script（`eval_retrieval` / `eval_ceah` / `faithfulness_eval`）都
支援兩種記憶體模式：

| 模式 | bank | 結果保證 | 何時用 |
|---|---|---|---|
| **預設**：`--max_train_cases 200000 --bank_dtype bf16` | 200K 子採樣 (49 conditions × ~4k cases/condition) | 5× 子集近似；變異與 paper sweep 無關 | 快速 sweep / debug / hyperparam tuning |
| **`--stream`** | 完整 1M | bf16 精度內等價（fish 12k 等價測試 19/20 query 100% top-K overlap） | paper-grade eval、跟其他 DDXPlus baseline 比 |

`--stream` 模式用 shard-streaming：外層 loop 過 16 shards（每 shard 載入一次到
GPU bf16，~2 GB），內層 batch 處理所有 queries 對該 shard 的 partial 分數，跨
shard 用 top-K-merge 累積。每 shard 載入後 inner compute 用 vectorized
`scatter_reduce_amax` 算 max_mean。Numerically 等價於 monolithic-bank 計算，
但只占用單張 shard 的記憶體量。

子採樣模式選 200K 是因為 DDXPlus 49 個 condition × ~20k cases/condition 在
~4k cases/condition 已遠大於 `top_k_cases=20`；R@K 變異 < 1 pp。要 robustness
ablation 可以加跑 `--max_train_cases 500000`。

`--stream_query_batch` (預設 64) 控制 inner batch 大小，主要 GPU 記憶體峰值由
`les_sim = q_batch_lesions @ shard.lesion.T` 的 `[B × avg_lesions, m_shard]`
bf16 tensor 決定。50-series / Blackwell 大卡可調 `--stream_query_batch 128` 翻倍
速度。

### 2. Phase 1 Retrieval

Run DDXPlus-specific retrieval metrics. Default uses 200K subsample for fast
iteration; add `--stream` for full 1M-bank paper-grade eval.

```bash
# Default: 200K subsample, ~30 min on full valid (132K queries)
$PY -m diagnosis_model.cause_inference.ddxplus.eval_retrieval \
  --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
  --output_dir diagnosis_model/cause_inference/outputs/ddxplus_phase1 \
  --query_split valid \
  --top_k_cases 20 \
  --top_n_causes 20 \
  --alpha_global 0.25 \
  --beta_lesion 0.75 \
  --lesion_match max_mean \
  --ks 1 5 10 20 50

# --stream: full 1M bank, ~30-50 min on full valid, shard-streamed retrieval
$PY -m diagnosis_model.cause_inference.ddxplus.eval_retrieval \
  --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
  --output_dir diagnosis_model/cause_inference/outputs/ddxplus_phase1_stream \
  --query_split valid \
  --top_k_cases 20 --top_n_causes 20 \
  --alpha_global 0.25 --beta_lesion 0.75 \
  --lesion_match max_mean \
  --ks 1 5 10 20 50 \
  --stream --stream_query_batch 64
```

Outputs:

```text
metrics.json
per_query_results.jsonl
```

Primary metrics:

- `pathology_exact`: strict `PATHOLOGY` coverage, MRR, R@K.
- `differential_diagnosis.NDCG`: NDCG@K over DDXPlus differential probabilities.
- `differential_diagnosis.pool_relevance_mass_coverage`: how much DDX relevance appears in the retrieved candidate pool before reranking.

### 3. Train Candidate Pool

Reuse the existing Phase 2 candidate-pool builder. Same memory constraints
as the eval scripts: the full 1M bank's lesion stack (~60 GB fp32 / ~30 GB
bf16) blows past 32 GB VRAM, AND the inline bank build's fp32 upcast of all
per-case fields (global + 2× text + lesion embs) blows past 62 GB CPU RAM.
DDXPlus must explicitly pass `--max_train_cases 200000 --bank_dtype bf16` —
applied to BOTH the bank and the leave-one-out queries (LOO requires
query[i] == bank[i], so the same subsample serves both via shared
`sample_seed`). 200k × top_k=20 = 4M (q, retrieved_case) pairs for CEAM
training, well above what 49-condition retrieval needs.

```bash
$PY -m diagnosis_model.cause_inference.preprocessing.build_train_candidate_pool \
  --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
  --output_path diagnosis_model/cause_inference/outputs/ddxplus_case_db/train_candidate_pool.pt \
  --top_k_cases 20 \
  --alpha_global 0.25 \
  --beta_lesion 0.75 \
  --lesion_match max_mean \
  --semantic_threshold 0.95 \
  --bank_dtype bf16 \
  --max_train_cases 200000 \
  --sample_seed 42
```

For DDXPlus, exact pathology labels are included in the cause table. The `semantic_threshold` path still works because exact duplicate disease strings share the same embedding row. ~7 min on the 200k subsample (10k smoke test ran at ~500 q/s).

> The script's CLI defaults (`--bank_dtype fp32 --max_train_cases 0`) preserve
> the historic fish workflow. DDXPlus must explicitly opt into the bf16 +
> subsample combination above.

### 4. Train CEAM

Train CEAM on DDXPlus evidence tokens. Pass `--max_train_cases` / `--sample_seed`
matching the values used in step 3 so `CEAMDataset[idx]` lines up with
`train_pool[idx]` (sanity check enforced at startup). `--bank_dtype bf16` keeps
the validation Phase 1 bank in half precision (200k cases ~6 GB GPU bf16 vs
~12 GB fp32, leaves headroom for CEAM forward). **`--text_dropout 1.0` is
required for DDXPlus** — masks the redundant text token (= global duplicate)
to zero so CEAM attribution can't collapse onto it; **`--eval_text_kind none`**
matches that at validation time.

```bash
$PY -m diagnosis_model.cause_inference.train_ceah \
  --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
  --train_pool_path diagnosis_model/cause_inference/outputs/ddxplus_case_db/train_candidate_pool.pt \
  --output_dir diagnosis_model/cause_inference/outputs/ddxplus_ceah \
  --attribution_mode softmax \
  --scoring_mode multiplicative \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4 \
  --lambda_sparsity 0.0 \
  --text_dropout 1.0 \
  --eval_top_k_cases 20 \
  --eval_max_queries 300 \
  --eval_text_kind none \
  --eval_lesion_match max_mean \
  --bank_dtype bf16 \
  --max_train_cases 200000 \
  --sample_seed 42
```

The full 1M train via `load_cases` blows past 62 GB CPU RAM during the
bf16→fp32 upcast of all per-case fields (global + 2× text + lesion embs) —
hence the 200k subsample, which mirrors step 3. The existing trainer prints
semantic retrieval metrics during validation. Use the DDXPlus evaluators
below for final DDXPlus-specific metrics.

> The script's CLI defaults (`--bank_dtype fp32 --max_train_cases 0`) preserve
> the historic fish 12,780-case workflow (full train, fp32 valid bank).

### 5. CEAM Rerank Evaluation

Evaluate the trained CEAM checkpoint. Add `--stream` for full 1M-bank
Phase 1 retrieval before CEAM reranking.

```bash
# Default: 200K subsample bank
$PY -m diagnosis_model.cause_inference.ddxplus.eval_ceah \
  --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
  --ceah_ckpt diagnosis_model/cause_inference/outputs/ddxplus_ceah/best_ceah.pt \
  --output_dir diagnosis_model/cause_inference/outputs/ddxplus_ceah_eval \
  --query_split valid \
  --top_k_cases 20 \
  --top_n_causes 20 \
  --gammas 0.0 0.25 0.5 0.75 1.0 \
  --dump_gamma 0.0 \
  --alpha_global 0.25 \
  --beta_lesion 0.75 \
  --lesion_match max_mean \
  --text_kind none \
  --attribution_mode softmax \
  --scoring_mode multiplicative \
  --ks 1 5 10 20 50

# --stream: full 1M bank for Phase 1 retrieval stage of the cascade
$PY -m diagnosis_model.cause_inference.ddxplus.eval_ceah \
  --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
  --ceah_ckpt diagnosis_model/cause_inference/outputs/ddxplus_ceah/best_ceah.pt \
  --output_dir diagnosis_model/cause_inference/outputs/ddxplus_ceah_eval_stream \
  --query_split valid \
  --top_k_cases 20 --top_n_causes 20 \
  --gammas 0.0 0.25 0.5 0.75 1.0 --dump_gamma 0.0 \
  --alpha_global 0.25 --beta_lesion 0.75 \
  --lesion_match max_mean \
  --text_kind none \
  --attribution_mode softmax --scoring_mode multiplicative \
  --ks 1 5 10 20 50 \
  --stream --stream_query_batch 64
```

`gamma=0.0` is pure CEAM reranking. `gamma=1.0` is Phase 1 scoring only.

Outputs:

```text
metrics_gammas.json
per_query.jsonl
```

`per_query.jsonl` includes evidence alpha values for each dumped top prediction.

### 6. Evidence Faithfulness

Run evidence masking. Add `--stream` for full 1M-bank Phase 1 retrieval.

```bash
# Default: 200K subsample bank
$PY -m diagnosis_model.cause_inference.ddxplus.faithfulness_eval \
  --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
  --ceah_ckpt diagnosis_model/cause_inference/outputs/ddxplus_ceah/best_ceah.pt \
  --output_dir diagnosis_model/cause_inference/outputs/ddxplus_faithfulness \
  --query_split valid \
  --gamma 0.0 \
  --top_k_cases 20 \
  --alpha_global 0.25 \
  --beta_lesion 0.75 \
  --lesion_match max_mean \
  --text_kind none \
  --attribution_mode softmax \
  --scoring_mode multiplicative

# --stream: full 1M bank
$PY -m diagnosis_model.cause_inference.ddxplus.faithfulness_eval \
  --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
  --ceah_ckpt diagnosis_model/cause_inference/outputs/ddxplus_ceah/best_ceah.pt \
  --output_dir diagnosis_model/cause_inference/outputs/ddxplus_faithfulness_stream \
  --query_split valid \
  --gamma 0.0 --top_k_cases 20 \
  --alpha_global 0.25 --beta_lesion 0.75 \
  --lesion_match max_mean \
  --text_kind none \
  --attribution_mode softmax --scoring_mode multiplicative \
  --stream --stream_query_batch 64
```

Outputs:

```text
faithfulness.json
per_query.jsonl
```

Main conditions:

- `no_evidence`: masks all symptom/antecedent tokens.
- `no_top_evidence`: masks the highest-alpha symptom/antecedent token.
- `no_random_evidence`: random evidence-token baseline.
- `no_top_alpha`: masks the highest-alpha token across global/text/evidence.

Positive `full_score - masked_score` means the masked token group supported the prediction.

## Results: schema v1 vs v2 (2026-05-28 rebuild)

Same 200k-subsample bank, same hyperparameters, full 132,448 valid queries.
v1 = single-pathology `cause_emb_indices`, Age/Sex inside global only, text token
identical to global, `--text_dropout 0.5 / --text_kind medical`. v2 = the
schema documented above (`pathology_emb_idx` split, expanded `cause_emb_indices`,
Age/Sex as atomic lesion tokens, `--text_dropout 1.0 / --text_kind none`).

### Phase 1 retrieval

| Metric                       | v1     | v2         | Δ          |
|------------------------------|--------|------------|------------|
| Pool size mean               | 1.08   | **16.34**  | +15×       |
| Pool size max                | 8      | 42         |            |
| pathology R@1                | 98.87% | 53.83%     | −45.04 pp  |
| pathology R@5                | 99.96% | 93.92%     | −6.04 pp   |
| pathology R@10               | 99.96% | 98.52%     | −1.44 pp   |
| pathology R@20               | 99.96% | **100.0%** | +0.04 pp   |
| **DDX NDCG@5**               | 0.485  | **0.824**  | **+0.339** |
| **DDX NDCG@10**              | 0.434  | **0.860**  | **+0.426** |
| **DDX NDCG@20**              | 0.422  | **0.898**  | **+0.476** |
| **DDX pool_mass_coverage**   | 29.7%  | **99.9%**  | **+70.2 pp** |

Pathology R@1 drops because the expanded pool now has DDX alternatives
competing for rank 1 under raw Phase 1 scoring (cosine doesn't know to
prefer pathology over DDX). R@20=100% confirms pathology is still **in**
the pool — CEAM at γ=0.75 recovers it (next table). DDX NDCG@K and pool
mass coverage are paper-grade leaderboard numbers now.

### CEAM training (200k cases, 20 epochs)

| Metric             | v1                    | v2                       |
|--------------------|-----------------------|--------------------------|
| neg_score (ep 0)   | 0.666                 | 0.036                    |
| neg_score (ep 19)  | 0.073                 | 0.003                    |
| val sem_R@1 (300 q)| 1.000 (trivial)       | 0.128 (real ranking work)|

v1 saturated trivially (pool size 1 = R@1 is just coverage). v2 has a real
ranking task and the loss/score curves are healthy.

### CEAM γ-scan (full 132k valid)

| γ              | v1 path R@1 | v2 path R@1     | v1 NDCG@5 | v2 NDCG@5     |
|----------------|-------------|------------------|-----------|----------------|
| 0.00 pure CEAM | 97.97%      | 87.63%           | 0.485     | 0.699          |
| 0.25           | 98.04%      | 93.56%           | 0.485     | 0.841          |
| 0.50           | 98.16%      | 94.96%           | 0.485     | 0.848          |
| **0.75**       | 98.87%      | **95.64% ★**     | 0.485     | **0.853 ★**    |
| 1.00 pure Φ1   | **98.88% ★**| 53.85%           | 0.485     | 0.824          |

**Best-γ flip**: v1 γ=1.0 (CEAM harmful) → v2 **γ=0.75** (CEAM essential).
γ=0.75 over γ=1.0 on v2 gives +41.79 pp pathology R@1 — direct evidence
that CEAM does meaningful reranking on the expanded pool. On v1 the
ranking task didn't exist so CEAM couldn't show value.

### Faithfulness (full 132k, mean score drop per masked condition)

| Condition         | v1     | v2        | Reading                                  |
|-------------------|--------|-----------|------------------------------------------|
| no_global         | 0.638  | 0.432 ↓   | global still important but no longer dominant |
| no_text           |−0.0002 | 0.0000    | text disabled (correct)                  |
| **no_evidence**   | 0.062  | **0.217** | evidence collective impact +3.5×         |
| **no_top_alpha**  | 0.627  | 0.572     | top-α token still primary                |
| **no_top_evidence**| 0.0018| **0.142** | **single top-α evidence impact +80×**    |
| no_random_evidence|−0.0001 | 0.006     | random baseline (correctly small)        |

v1: `no_global ≈ no_top_alpha` indicated top-α was always global (no
lesion-grounding). v2: `no_top_evidence` jumped 80× — for many queries
the highest-α token is now an evidence, not global. This is the
"lesion-grounded attribution" CEAM was designed to provide.

### Per-query α (v2 sample over 1000 queries)

| Statistic                | Value       |
|--------------------------|-------------|
| global α mean            | 0.48        |
| global α median          | 0.31        |
| top evidence α mean      | 0.45        |
| Age token α mean         | 0.0004      |
| Age token median rank    | 11/26       |
| Sex token α mean         | 0.0007      |
| Sex token median rank    | 10/26       |

CEAM learned a **bimodal** attribution: most queries put α ≈ 1 on a
single evidence token (global ≈ 0); a minority concentrate α on global
(e.g. Boerhaave queries with α(global) = 0.93). Age/Sex are atomic
tokens with separable α but CEAM consistently finds them
non-diagnostic for the 49-condition task — a "negative finding" that
validates the design: CEAM correctly identifies which atomic tokens
carry diagnostic signal rather than diluting attribution across all
inputs.

### Trade-off owned in the paper

Pathology R@1 at γ=0.75 dropped from v1 98.87% (trivial pool=1) to v2
95.64% (in-pool top-1 of 16 candidates). This is the cost of pool
expansion. R@5 = 99.96% on v2 stays at v1 levels — within a few ranks,
pathology is always found. Frame as: "We expand the candidate pool
from a degenerate single-pathology case to a 16-cause pool to make
DDX NDCG evaluable; pathology Top-1 within the expanded pool is 95.6%
at γ=0.75."

## Phase 3 / 4 on DDXPlus

The Phase 3 case encoder (`train_case_encoder.py`) and Phase 4 RVQ + Light
reranker (`rvq_rerank/`) now accept the DDXPlus subsample convention:
`--max_train_cases 200000 --sample_seed 42 --bank_dtype bf16`. Two DDXPlus
constraints drove the changes:

1. The precomputed `teacher_train_train.pt` is 80 GB at 200k cases (vs 311 MB
   for fish 12,780) — infeasible to store and ~14 hr to compute. A new
   `--teacher_mode on_the_fly` mode caches normalized globals + lesion
   stacks on GPU in `--bank_dtype` and computes the BxB intra-batch teacher
   block per step via vectorized `scatter_reduce`. Numerically equivalent to
   the precomputed table (verified `|Δ| ≤ 6e-8` on a fp32 synthetic test
   against `compute_case_similarities`), only `max_mean` /
   `max_mean_normalized` lesion-match (hungarian has no batched equivalent).
2. The `[Nv, Nt]` validation sim matrix is 100 GB at 130k × 200k fp32;
   `retrieval_metrics` is now query-chunked (`--eval_query_batch 512`) and
   `train_reranker.py` / `eval_final.py` early-stop on a `--max_valid_cases`
   subsample (5 000 keeps the matrix < 4 GB).

`CaseEncoderDataset.case_id` was switched from `c["case_id"]` to its
positional index so subsample (DDXPlus path) lines up with both the
on-the-fly teacher and any subsample-built precomputed table.

### Phase 3 training rationale: harder teacher target (v1 → v2)

The default training command in §7 below uses the **v2** hyperparameters
(`--temp_target 0.05 --temp_pred 0.05 --infonce_temp 0.05
--infonce_positives pathology`). This subsection records the v1→v2
ablation that motivated those defaults — keep the v2 command and skip
this block unless you're reproducing the negative result.

v1 used the fish-style defaults (`--temp_target 0.1 --temp_pred 0.1
--infonce_temp 0.07 --infonce_positives cause_emb_indices`). On DDXPlus
v2 schema the dual-target InfoNCE positives are
`[pathology_emb_idx, *DDX_5]` = 6 positives, and the listwise-KL
target's row-softmax is flat because most within-batch pairs share
condition. Training signal saturated at epoch 1 (sem R@10 = 99.92%) and
the encoder learned a single-vec representation that under-discriminates
between cases sharing evidence signatures.

Effect on pathology eval (200k bank × 5,000 valid, top_k_cases=20):

| Setting | temp_target / pred | InfoNCE temp | InfoNCE positives | path R@1 | path R@5 | pool mean | median rank |
|---|---|---|---|---:|---:|---:|---:|
| **v1** (fish defaults) | 0.10 | 0.07 | cause_emb_indices (6) | 28.12% | 77.32% | 12.14 | 3 |
| **v2** (DDXPlus prod)  | **0.05** | **0.05** | **pathology (1)** | **52.54%** | **94.16%** | **16.00** | **1** |
| Phase 1 reference | — | — | — | 53.83% | 93.92% | 16.34 | — |

> v2 recovers ~25 pp R@1 — single-vec encoder at 768-dim has the
> capacity, but needs a peakier teacher distribution + a strictly
> single-positive InfoNCE to learn within-condition discrimination.
> v2 matches Phase 1's hand-crafted α·cos(g) + β·max_mean(L) at
> roughly 100× per-query speed (single-vec cosine over the bank vs
> Hungarian/max_mean over the lesion stack).

> **Fish does not exhibit this**: the per-case GT has a single cause,
> so `cause_emb_indices` and `pathology_emb_idx` would coincide.
> `--infonce_positives pathology` errors on fish (no field) — the
> v1 defaults remain correct for fish.

### 7. Phase 3 — train the case encoder

```bash
$PY -m diagnosis_model.cause_inference.train_case_encoder \
  --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
  --output_dir diagnosis_model/cause_inference/outputs/ddxplus_encoder \
  --teacher_mode on_the_fly --bank_dtype bf16 \
  --teacher_alpha 0.25 --teacher_beta 0.75 --teacher_lesion_match max_mean \
  --max_train_cases 200000 --max_valid_cases 5000 --sample_seed 42 \
  --encoder_type deepsets \
  --batch_size 256 --epochs 30 --early_stop_patience 5 \
  --temp_target 0.05 --temp_pred 0.05 \
  --use_infonce --infonce_weight 0.5 --infonce_temp 0.05 \
  --infonce_positives pathology
```

Early-stop signal: `sem_R@10` over `--max_valid_cases` queries (any
DDX or pathology cause within cos≥0.95 of a retrieved case's GT). This
is a training proxy — final paper numbers come from
`ddxplus/eval_phase3.py` (pathology R@K + DDX NDCG).

> Fish workflow (precomputed teacher) is unchanged: omit
> `--teacher_mode on_the_fly`, `--max_*_cases`, `--temp_*`, and
> `--infonce_positives pathology` to get the legacy path.

### 8. Phase 3 — DDXPlus retrieval eval

```bash
# Full 132k valid (paper-grade); drop --max_query_cases for full split.
$PY -m diagnosis_model.cause_inference.ddxplus.eval_phase3 \
  --encoder_ckpt diagnosis_model/cause_inference/outputs/ddxplus_encoder/best_encoder.pt \
  --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
  --output_dir diagnosis_model/cause_inference/outputs/ddxplus_phase3_eval_full \
  --query_split valid \
  --max_train_cases 200000 --sample_seed 42 \
  --max_query_cases -1 \
  --top_k_cases 20 --top_n_causes 20 \
  --ks 1 5 10 20 50
```

Outputs the same `pathology_exact` + `differential_diagnosis` blocks as
`ddxplus/eval_retrieval.py`, so the Phase 3 row can sit next to the Phase 1
row in paper tables.

### 9. Phase 4 — fit RVQ + train Light reranker

The absorption sweep (G2 in paper_tables.md) fits three (M, K) codebooks.
Light reranker training is **optional**: on DDXPlus the reranker pulls
`rvq_only` ranking back toward dense, which is the SUBOPTIMAL operating
point (see §"Cross-domain ABQ" below), so production = `rvq_only`. We
keep the M=4 K=256 Light reranker for the paper's "Light row" in the
Phase 4 table as an explicit ablation that the reranker hurts here.

```bash
# 1. Fit 3 RVQ codebooks on z_train from the Phase 3 v2 encoder. Bank
#    must match Phase 3 (--max_train_cases + --sample_seed identical).
for MK in "4 256" "2 64" "1 16"; do
  M=$(echo $MK | cut -d' ' -f1); K=$(echo $MK | cut -d' ' -f2)
  $PY -m diagnosis_model.cause_inference.rvq_rerank.fit_rvq \
    --encoder_ckpt diagnosis_model/cause_inference/outputs/ddxplus_encoder/best_encoder.pt \
    --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
    --output_dir diagnosis_model/cause_inference/outputs/ddxplus_rvq \
    --M $M --K $K --kmeans_iters 25 \
    --max_train_cases 200000 --sample_seed 42
done

# 2. (Optional) Train Light reranker on M=4 K=256 for ablation.
#    Regime B (top_k_cases=1) early-stop, same as fish.
$PY -m diagnosis_model.cause_inference.rvq_rerank.train_reranker \
  --encoder_ckpt diagnosis_model/cause_inference/outputs/ddxplus_encoder/best_encoder.pt \
  --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
  --rvq_M 4 --rvq_K 256 \
  --rvq_dir diagnosis_model/cause_inference/outputs/ddxplus_rvq/rvq_M4_K256 \
  --output_dir diagnosis_model/cause_inference/outputs/ddxplus_rvq/reranker_M4_K256_light \
  --variant light --K_top 50 \
  --batch_size 64 --epochs 30 \
  --max_train_cases 200000 --max_valid_cases 5000 --sample_seed 42 \
  --eval_top_k_cases 1
```

### 10. Phase 4 — DDXPlus retrieval eval (absorption sweep, full 132k valid)

The absorption sweep is the paper headline (Table G2). Three (M, K)
configurations, each evaluated with `dense / rvq_only / full_analytic`.
`light` is reported only for M=4 K=256 since the reranker is an explicit
ablation that the v1 result already characterized (Light pulls
`rvq_only` toward `dense`, which on DDXPlus is the SUBOPTIMAL operating
point — see §"Cross-domain ABQ" / paper_tables.md Table G3).

```bash
# Absorption sweep (3 codebooks × {dense, rvq_only, full_analytic})
for MK_DIR in "M4_K256" "M2_K64" "M1_K16"; do
  $PY -m diagnosis_model.cause_inference.ddxplus.eval_phase4 \
    --encoder_ckpt diagnosis_model/cause_inference/outputs/ddxplus_encoder/best_encoder.pt \
    --rvq_dir diagnosis_model/cause_inference/outputs/ddxplus_rvq/rvq_$MK_DIR \
    --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
    --output_dir diagnosis_model/cause_inference/outputs/ddxplus_phase4_eval_full_$MK_DIR \
    --methods dense rvq_only full_analytic \
    --max_train_cases 200000 --sample_seed 42 --max_query_cases -1 \
    --top_k_cases 20 --top_n_causes 20 --K_top 50 \
    --ks 1 5 10 20 50
done

# Optional Light reranker row (M=4 K=256 only)
$PY -m diagnosis_model.cause_inference.ddxplus.eval_phase4 \
  --encoder_ckpt diagnosis_model/cause_inference/outputs/ddxplus_encoder/best_encoder.pt \
  --rvq_dir diagnosis_model/cause_inference/outputs/ddxplus_rvq/rvq_M4_K256 \
  --reranker_ckpt diagnosis_model/cause_inference/outputs/ddxplus_rvq/reranker_M4_K256_light/best.pt \
  --case_db_dir diagnosis_model/cause_inference/outputs/ddxplus_case_db \
  --output_dir diagnosis_model/cause_inference/outputs/ddxplus_phase4_eval_full_M4_K256_with_light \
  --methods dense rvq_only light full_analytic \
  --max_train_cases 200000 --sample_seed 42 --max_query_cases -1 \
  --top_k_cases 20 --top_n_causes 20 --K_top 50 \
  --ks 1 5 10 20 50
```

> Cross-domain framing: on **fish**, `rvq_only` ≈ `dense` (lossless
> regime) and `light` recovers any RVQ gap in `top_k=1` Regime B —
> reranker is the buffer-free deployment fallback. On **DDXPlus**,
> `rvq_only` BEATS `dense` (regularization regime, +23 pp R@1 at
> M=2 K=64); `light` predicts Δ ≈ q·e to recover dense, but dense
> itself is suboptimal here, so `light` regresses toward dense and
> hurts. Both observations follow from the same ABQ framework —
> see paper_tables.md Table G4. The note above is the *retrieval-side
> proxy*; step 11 confirms it survives end-to-end through CEAM.

### 11. Phase 4 — end-to-end CEAM compression eval (γ-swept)

The retrieval proxy (step 10) is pre-CEAM. This step runs the compressed
coarse ranking *through CEAM* and sweeps γ, the DDXPlus analogue of fish
`eval_ceah_compressed.py`. DDXPlus production is the **coarse-dominant
γ=0.75** point (not fish's γ=0), so the reranker's end-to-end effect must
be read across the curve.

```bash
$PY -m diagnosis_model.cause_inference.ddxplus.eval_ceah_compressed \
  --encoder_ckpt diagnosis_model/cause_inference/outputs/ddxplus_encoder/best_encoder.pt \
  --rvq_dir      diagnosis_model/cause_inference/outputs/ddxplus_rvq/rvq_M4_K256 \
  --reranker_ckpt diagnosis_model/cause_inference/outputs/ddxplus_rvq/reranker_M4_K256_light/best.pt \
  --case_db_dir  diagnosis_model/cause_inference/outputs/ddxplus_case_db \
  --ceah_ckpt    diagnosis_model/cause_inference/outputs/ddxplus_ceah/best_ceah.pt \
  --output_dir   diagnosis_model/cause_inference/outputs/ddxplus_ceah_compressed_eval \
  --methods dense rvq_only light full_analytic \
  --max_query_cases 5000
```

Result (M4K256 768×, 5000 valid, pathology R@1):

| γ | dense | rvq_only | light | full(oracle) |
|---|---|---|---|---|
| 0.00 (pure CEAM) | 0.878 | 0.876 | 0.877 | 0.876 |
| 0.75 (production) | 0.954 | **0.981** | 0.974 | 0.973 |
| 1.00 (pure Phase1) | 0.525 | 0.666 | 0.604 | 0.627 |

> **Two regimes, both kill the reranker end-to-end.** (A) γ=0: CEAM
> re-scores from scratch and discards coarse order — compression and
> reranker both vanish into noise (±0.16 pp). (B) γ=0.75: coarse order
> matters but `rvq_only` beats `dense` (+2.7 pp); `light` drags it back
> toward dense (−0.7 pp) and **so does the analytic oracle (−0.9 pp)** —
> ruling out an architecture limit. The reranker recovers the dense
> ranking, which is exactly the wrong target on DDXPlus. Matches the fish
> conclusion (reranker redundant in production) via a different mechanism.

## Smoke Test

Validated with a synthetic DDXPlus-like CSV using `--embedding_backend hash`.
Covered: case DB build, retrieval eval, CEAM eval, faithfulness eval, and bf16 load/upcast path.

Phase 3 / 4 changes additionally smoke-tested on fish `case_db` (12,780
cases): precomputed-teacher path unchanged; `--teacher_mode on_the_fly` on
fp32 reproduces the precomputed table to `|Δ| ≤ 6e-8`; `ddxplus/eval_phase3`
and `ddxplus/eval_phase4` both run end-to-end on fish case_dbs (pathology
metrics degenerate as expected since fish has no `pathology_emb_idx`, but
the cascade plumbing is exercised).