# FaCE-R 論文比較表 + 取得指令

> **FaCE-R: Faithful Case-guided Cause-Evidence Reasoning** — coarse-to-fine cascade，
> 兩掛牌貢獻：coarse 的 **ABQ**（Aggregation-Buffered Quantization）+ fine 的 **CEAH**
> （Cause-Evidence Attribution Head）。
>
> 本檔集中所有論文比較表，每表下附「取得指令」。所有數據 cell 已以現行 metric 填入；
> 殘留的 `—` 是 reference / baseline / oracle 欄位（無對應數值），`⚠` 僅標 Table D8 中
> 中度壓縮不穩的 weighted-RVQ 變體（negative result，非待跑）。所有指令從 repo root 執行：
>
> ```bash
> PY=/home/lab603/anaconda3/envs/SDM/bin/python
> cd /mnt/ssd/YJ/fish_disease_system
> ```
>
> 詳細逐節說明見 [README.md](README.md)；逐指令記錄見 [inference.txt](inference.txt) / [training.txt](training.txt)。
> 全部 1573 valid queries、cluster 用 `cause_clusters_llm.json`（466 topics）除非另註。

---

## A. 主結果 — production cascade 端到端

### Table A1 — FaCE-R vs Phase 1 baseline

> **CEAH 的主貢獻是 Phase 1 結構上做不到的 faithful lesion-grounded 歸因**（per-evidence
> softmax α + 反事實驗證 15×，見 Table B2/C3/C4）+ 病因條件式證據路由（Table C5）。
> 這張表是佐證——證明加上可解釋歸因**沒犧牲準確度（反而更準）**——不是主張本身。
> 「為何不直接用 Phase 1 排病因」的完整 rebuttal 見 README §Stage 2 開頭。

| Method | Pool / Coarse index | Fine scorer | sem R@1 | sem R@5 | sem R@10 | sem R@20 | sem MRR | cl R@1 | cl R@10 | cl R@20 | cl MRR | per-q† |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 1 baseline（case 檢索，無 CEAH） | raw `case_db_raw` | none | 22.2% | 35.7% | 45.6% | 59.2% | 0.300 | 17.2% | 35.8% | 47.0% | 0.235 | 15.4 ms |
| **FaCE-R（檢索 → CEAH 歸因重排）** | raw coarse → fine `case_db` | CEAH | 21.0% | 39.4% | **47.6%** | 56.2% | **0.303** | 19.6% | **39.6%** | 44.0% | **0.269** | ~3.4 ms |
| **Δ** | — | — | −1.2 | +3.7 | **+2.0** | −3.0 | +0.003 | +2.4 | **+3.8** | −3.0 | **+0.034** | — |

> 準確度增益集中在淺排名與 cluster (disease-topic) recall：sem R@5 +3.7pp、R@10 +2.0pp、
> cluster R@10 +3.8pp、cluster MRR +0.034。R@1 略降（−1.2pp），且**深排名 R@20 反降**
> （sem −3.0pp、cluster −3.0pp）——CEAH 在固定候選池上把 cluster-correct 病因往 top-10 拉，
> 代價是把部分原本落在 11–20 名的 GT 擠出 top-20；production 取 top-N≤10 故以淺排名增益為準。
> **† 速度不是 CEAH 的功勞**——CEAH 本身是 rerank 成本（+~2.3 ms）。整體 FaCE-R 3.4 ms 仍 < Phase 1
> baseline 15.4 ms，但那個「快」來自 **coarse 端 ABQ**（Phase 3 單向量檢索取代 Phase 1 Hungarian，
> 15.4→1.1 ms），是另一個貢獻（見 Table D1/D6），不該算進這張 CEAH 表。
> 數字＝`eval_ceah_compressed.py` 的 **dense row**；Phase 1 baseline 取 Table B1 raw 欄。
>
> **Paired bootstrap CI 說明（A1 path 本身的 paired bootstrap 待補；以下 CI 引用 Table C1 同池 controlled path）：** Δ sem MRR 在 C1 path 為 +0.003 [−0.002, +0.009] p=0.20 → **statistically indistinguishable from 0**（不要把它當增益寫進 headline）；R@1 −1.16 *** / R@10 +2.46 *** / R@20 −3.41 *** / cl R@10 +4.02 *** / cl MRR +0.0355 *** 均顯著。詳見 Table C1 註腳與 Table C7（D-i ablation）。

**取得指令：**
```bash
# FaCE-R（檢索 → CEAH）那列 = dense row
$PY -m diagnosis_model.cause_inference.eval_ceah_compressed \
  --coarse_case_db_dir diagnosis_model/cause_inference/outputs/case_db_raw \
  --encoder_ckpt       diagnosis_model/cause_inference/outputs/encoder_raw/best_encoder.pt \
  --rvq_root           diagnosis_model/cause_inference/outputs/rvq_rerank_raw \
  --fine_case_db_dir   diagnosis_model/cause_inference/outputs/case_db \
  --ceah_ckpt          diagnosis_model/cause_inference/outputs/ceah_v3/best_ceah.pt \
  --cluster_json       diagnosis_model/cause_inference/outputs/cause_clusters_llm.json \
  --attribution_mode softmax --scoring_mode multiplicative \
  --top_k_cases 20 --semantic_threshold 0.95 --text_kind medical
# coarse-only baseline 那列見 Table B1 raw 欄（phase1_baseline on case_db_raw）
```

---

## B. 核心主張 — Asymmetric VLM dependency（檢索零樣本、歸因須微調）

### Table B1 — Phase 1 retrieval：raw SigLIP2 vs fine-tuned VLM

| Metric | Fine-tuned VLMs | Raw SigLIP2 | Δ |
|---|---|---|---|
| sem R@1 | 22.3% | 22.2% | −0.1 |
| sem R@5 | 34.8% | 35.7% | +0.9 |
| **sem R@10** | 44.4% | **45.6%** | **+1.2** |
| sem R@20 | 58.9% | 59.2% | +0.3 |
| sem MRR | 0.298 | 0.300 | +0.002 |
| cl R@10 | 35.8% | 35.8% | 0.0 |
| cl MRR | 0.235 | 0.235 | 0.0 |

> Raw 持平甚至略勝 → coarse 檢索零樣本飽和，不需 in-domain 微調。

**取得指令：**
```bash
for db in case_db case_db_raw; do
  $PY -m diagnosis_model.cause_inference.phase1_baseline \
    --case_db_dir diagnosis_model/cause_inference/outputs/$db \
    --output_dir  diagnosis_model/cause_inference/outputs/phase1_${db} \
    --cluster_json diagnosis_model/cause_inference/outputs/cause_clusters_llm.json \
    --top_k_cases 20 --alpha_global 0.25 --beta_lesion 0.75 \
    --lesion_match hungarian --diversify_threshold 0.95 --semantic_threshold 0.95
done
```

### Table B2 — Phase 2 CEAH faithfulness：fine-tuned vs raw case_db（反轉）

純 CEAH 實測（2026-05-22，1573 valid）：

| Faithfulness（CEAH score drop） | Fine-tuned case_db | Raw case_db |
|---|---|---|
| `no_lesion`（lesion-type） | **+0.0378** | **−0.0309** ← 反轉 |
| `no_lesion`（all） | +0.0348 | −0.0244 |
| `no_global`（all） | −0.0297 | +0.0468 ← raw 改靠 global |
| `no_random`（sanity） | +0.0027 | +0.0041 |

（檢索側 raw≈fine 的佐證見 Table B1；本表專看歸因。）

> 檢索在 raw 上持平，但遮 lesion 後分數**反而上升**（負 drop）、改由 global 載重 → 歸因
> faithfulness 反轉，fine stage 須 fine-tuned VLM-Lesion + fusion。`no_random` 兩邊 ≈0（sanity 過）。

**取得指令：**
```bash
# raw case_db 需先建（見 training.txt [0-raw]）：build_case_database --raw_lesion
# 在 raw case_db 上重訓 CEAH + faithfulness
$PY -m diagnosis_model.cause_inference.preprocessing.build_train_candidate_pool \
  --case_db_dir diagnosis_model/cause_inference/outputs/case_db_raw \
  --output_path diagnosis_model/cause_inference/outputs/case_db_raw/train_candidate_pool.pt \
  --top_k_cases 20
$PY -m diagnosis_model.cause_inference.train_ceah \
  --case_db_dir diagnosis_model/cause_inference/outputs/case_db_raw \
  --train_pool_path diagnosis_model/cause_inference/outputs/case_db_raw/train_candidate_pool.pt \
  --output_dir diagnosis_model/cause_inference/outputs/ceah_raw \
  --attribution_mode softmax --scoring_mode multiplicative \
  --epochs 20 --batch_size 32 --lr 1e-4 --lambda_sparsity 0.0 --text_dropout 0.5
for db in case_db ceah_raw; do :; done   # 對 case_db (fine) / case_db_raw (raw) 各跑 faithfulness：
$PY -m diagnosis_model.cause_inference.faithfulness_eval \
  --case_db_dir diagnosis_model/cause_inference/outputs/case_db \
  --ceah_ckpt diagnosis_model/cause_inference/outputs/ceah_v3/best_ceah.pt \
  --attribution_mode softmax --scoring_mode multiplicative --max_queries -1
$PY -m diagnosis_model.cause_inference.faithfulness_eval \
  --case_db_dir diagnosis_model/cause_inference/outputs/case_db_raw \
  --ceah_ckpt diagnosis_model/cause_inference/outputs/ceah_raw/best_ceah.pt \
  --attribution_mode softmax --scoring_mode multiplicative --max_queries -1
```

### Table B3 — Phase 3 distillation：raw vs fine-tuned（student 持平、加速一致）

| Method | Fine-tuned case_db | Raw case_db |
|---|---|---|
| Phase 1 hungarian（teacher）sem R@10 | 44.4% | 45.6% |
| DeepSets dual-target（student）sem R@10 | 44.7% | 45.6% |
| student per-q latency | 1.1 ms | 1.1 ms（皆 14× faster than teacher） |

**取得指令：**
```bash
for db in case_db case_db_raw; do
  enc=$([ $db = case_db ] && echo encoder_final || echo encoder_raw)
  $PY -m diagnosis_model.cause_inference.eval_phase1_aligned \
    --case_db_dir diagnosis_model/cause_inference/outputs/$db --include_phase1 \
    --checkpoints deepsets=diagnosis_model/cause_inference/outputs/$enc/best_encoder.pt \
    --top_k_cases 20 --semantic_threshold 0.95
done
```

### Table B4 — Retrieval-side fine-tune probe：supervised projection MLP（negative）

| Method | sem MRR |
|---|---|
| frozen VLM-Lesion（baseline） | 0.4061 |
| + supervised projection MLP | 0.4052（Δ −0.001，無改進） |

**取得指令：**
```bash
$PY -m diagnosis_model.cause_inference.train_projection \
  --case_db_dir diagnosis_model/cause_inference/outputs/case_db \
  --output_dir diagnosis_model/cause_inference/outputs/projection_v1 \
  --epochs 30 --batch_size 64 --lr 1e-4 --eval_every 2
```

> B1+B2+B3+B4 合起來＝asymmetric dependency 的完整論證（檢索側 4 個 fine-tune 槓桿全 no-op，歸因側必須 fine-tune）。

---

## C. Fine stage — CEAH

### Table C1 — Contribution breakdown：CEAH vs 純檢索（fine-tuned pool controlled ablation）

> 本表不是 production headline，而是固定在同一個 fine-tuned candidate pool 上，隔離 CEAH 對 cause scoring / attribution 的貢獻。production headline 見 Table A1（raw coarse → fine CEAH）。

| Method | Pool / Coarse index | Fine scorer | sem R@1 | sem R@5 | sem R@10 | sem R@20 | sem MRR | cl R@1 | cl R@10 | cl R@20 | cl MRR | per-q |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 1 only | fine `case_db` | none | 22.3% | 34.8% | 44.4% | 58.9% | 0.298 | 17.2% | 35.8% | 47.6% | 0.235 | 15.4 ms |
| **CEAH** | fine `case_db` | CEAH | 21.1% | 38.3% | **46.8%** | 55.5% | **0.301** | 20.1% | **39.8%** | 44.1% | **0.271** | — |
| **Δ** | — | — | −1.16 *** | +3.47 *** | **+2.46 *** | −3.41 *** | +0.003 **ns** | +2.96 *** | **+4.02 *** | −3.51 *** | **+0.036 *** | — |

> **Paired bootstrap CI**（query-level, B=10000, 同池 controlled, 2026-05-22 實測）— Δ in pp / MRR raw, 95% CI, two-sided p（`***` p<0.001, `**` <0.01, `*` <0.05, `ns` CI 跨 0）：
> sem R@1 [−1.76, −0.57] *** ・ R@5 [+2.53, +4.40] *** ・ R@10 [+1.56, +3.35] *** ・ R@20 [−4.18, −2.67] *** ・ **sem MRR [−0.002, +0.009] p=0.20 ns** ・ cl R@1 [+2.30, +3.61] *** ・ cl R@10 [+3.20, +4.83] *** ・ cl R@20 [−4.24, −2.78] *** ・ cl MRR [+0.0299, +0.0411] ***
>
> 在同一 fine-tuned candidate pool 上，CEAH **顯著**強化 disease-topic recall（cl R@10 +4.02***、cl MRR +0.0355***、cl R@1 +2.96***）與淺排名語意（sem R@5/R@10），**代價是顯著的 R@1 −1.16*** 與 R@20 −3.41*** 退化**。sem MRR 在統計上與 0 無異（**不可作為增益寫進 headline**）。CEAH 的主貢獻是 Phase 1 結構上做不到的 faithful lesion-grounded attribution（Table C3/C4），cluster 線增益是附加收益，sem 兩端退化的成因見 **Table C7 D-i ablation**（faithfulness–ranking 張力）。
> 中間「hybrid 線性混合」的完整 γ-scan 屬舊框架實驗，留在 [inference.txt](inference.txt) `[B]` 當
> 「為何不用 hybrid、直接用 CEAH」的 ablation 佐證。

**取得指令：**
```bash
$PY -m diagnosis_model.cause_inference.eval_ceah \
  --case_db_dir diagnosis_model/cause_inference/outputs/case_db \
  --ceah_ckpt diagnosis_model/cause_inference/outputs/ceah_v3/best_ceah.pt \
  --output_dir diagnosis_model/cause_inference/outputs/ceah_v3_final \
  --cluster_json diagnosis_model/cause_inference/outputs/cause_clusters_llm.json \
  --attribution_mode softmax --scoring_mode multiplicative \
  --gammas 0.0 1.0 --dump_gamma 0.0 \
  --top_k_cases 20 --diversify_threshold 0.95 --semantic_threshold 0.95 --text_kind medical
```

### Table C2 — CEAH 架構版本 ablation（為何 softmax + multiplicative）

| Version | Architecture | 結果 |
|---|---|---|
| v1 | sigmoid + single, λ=0.05 | α 塌到 ~0.06，evidence 完全沒用 |
| v2 | sigmoid + single, λ=0.005 | α 漲到 0.30，但 lesion α=0.02（94% 押 global） |
| **v3** | **softmax + multiplicative, λ=0** | **cause-type-aware lesion attribution** |

**取得指令：**（改 `train_ceah` 的 `--attribution_mode` / `--scoring_mode` / `--lambda_sparsity`）
```bash
$PY -m diagnosis_model.cause_inference.train_ceah \
  --case_db_dir diagnosis_model/cause_inference/outputs/case_db \
  --train_pool_path diagnosis_model/cause_inference/outputs/case_db/train_candidate_pool.pt \
  --output_dir outputs/ceah_v1 --attribution_mode sigmoid --scoring_mode single --lambda_sparsity 0.05 ...
# v2: sigmoid/single/λ=0.005；v3(production): softmax/multiplicative/λ=0
```

### Table C3 — Faithfulness（lesion masking）by cause-type（純 CEAH）

| condition | global-type | lesion-type | mixed |
|---|---|---|---|
| no_lesion drop | +0.0289 | **+0.0378** | +0.0413 |
| no_random drop | +0.0039 | +0.0039 | −0.0009 |
| **lesion vs random ratio** | 7.4× | **9.7×** | random≈0（drop 純由 lesion 驅動） |

### Table C4 — Faithfulness by N-lesion bucket（純 CEAH）

| condition | N=1 | N=2 | N≥3 |
|---|---|---|---|
| no_lesion drop | 0.0239 | 0.0467 | **0.0531** |
| no_random drop | 0.0020 | 0.0036 | 0.0036 |
| **lesion vs random ratio** | 12× | 13× | **15×** |

> no_lesion drop 隨 lesion 數遞增（N≥3 最大），random ≈ 0 → 遮 lesion 的傷害是 selective、
> mechanism-aligned，不是模型對任意擾動變敏感。

**C3 + C4 取得指令：**
```bash
$PY -m diagnosis_model.cause_inference.faithfulness_eval \
  --case_db_dir diagnosis_model/cause_inference/outputs/case_db \
  --ceah_ckpt diagnosis_model/cause_inference/outputs/ceah_v3/best_ceah.pt \
  --output_dir diagnosis_model/cause_inference/outputs/ceah_v3_faithfulness_pureceah \
  --attribution_mode softmax --scoring_mode multiplicative --gamma 0.0 --max_queries -1
```

### Table C5 — Attribution 隨 N 變化（集體證據 vs 挑主病灶）

| N | global α | lesion sum | concentration (max/sum) |
|---|---|---|---|
| 1 | 0.51 | 0.27 | 1.00 |
| 2 | 0.36 | 0.43 | 0.60 |
| ≥3 | **0.26** | **0.55** | **0.37** ≈ 1/N |

**取得指令：**
```bash
$PY -m diagnosis_model.cause_inference.analyze_v3_n_buckets \
  --eval_dir diagnosis_model/cause_inference/outputs/ceah_v3_final \
  --case_db diagnosis_model/cause_inference/outputs/case_db --gamma_tag g=0.00
```
> `--gamma_tag g=0.00`（兩位小數）須對應 C1 的 `--dump_gamma 0.0`。值對 γ 不敏感（α 是 CEAH 自身輸出）。

### Table C6 — Text mode ablation（vision-only fallback）

| Text mode | sem MRR | sem R@10 | lesion-type l/g (N=2) |
|---|---|---|---|
| medical | 0.3009 | 46.83% | 1.50 |
| colloquial | 0.3010 | 46.86% | 1.63 |
| none（vision-only） | 0.3008 | 46.76% | 1.87 |

> 純 CEAH 實測（2026-05-22）。retrieval 三模式持平（MRR diff ≤ 0.0002、R@10 ≤ 0.1pp）；
> text 缺席時 lesion-type l/g 上升（1.50→1.87）＝attention 自動補位給 lesion。vision-only 可行。

**取得指令：**
```bash
for mode in medical colloquial none; do
  $PY -m diagnosis_model.cause_inference.eval_ceah \
    --case_db_dir diagnosis_model/cause_inference/outputs/case_db \
    --ceah_ckpt diagnosis_model/cause_inference/outputs/ceah_v3/best_ceah.pt \
    --output_dir diagnosis_model/cause_inference/outputs/ceah_v3_text_${mode} \
    --cluster_json diagnosis_model/cause_inference/outputs/cause_clusters_llm.json \
    --attribution_mode softmax --scoring_mode multiplicative \
    --gammas 0.0 --dump_gamma 0.0 --text_kind ${mode}
done
```

### Table C7 — D-i ablation：faithfulness–ranking 張力（為何 v3 是奇異 Pareto 點）

> Table C1 的 paired bootstrap 確認 v3 對 Phase 1 有顯著 sem R@1（−1.16***）/ R@20（−3.41***）退化。本表測試**能否在不犧牲 lesion-grounded faithfulness 的前提下修復這兩個退化**。
>
> 兩條 loss 軸（架構不動，仍 softmax α + multiplicative scoring，BCE 與 candidate pool 不變）：
> - **Rung 0（graded soft labels）**：把硬 `positive_mask`（cos≥0.95）換成 ramp `clamp((maxcos−0.90)/0.10, 0, 1)`。
> - **Rung 1（listwise multi-positive CE，binary 監督）**：`loss = BCE + λ·CE(softmax(score/T), uniform-over-positives)`，T=0.1，掃 λ ∈ {0.05, 0.1, 0.3, 0.5, 1.0}。
>
> Retrieval Δ vs v3（paired bootstrap，B=10000，同 fine pool）；**Faithfulness 欄＝`no_lesion`(lesion-type) drop**（正＝遮病灶會掉分＝病灶載重；v3 baseline +0.0378，`no_random` ≈ 0 sanity）。

| Variant | sem R@1 Δ | sem R@10 Δ | sem R@20 Δ | sem MRR Δ | cl R@10 Δ | cl MRR Δ | no_lesion (lesion-type) | Faithful? |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| **v3 (BCE only)** | — | — | — | — | — | — | **+0.0378** | ✅ |
| Rung 0（graded labels）| −2.61 *** | −0.87 * | +0.16 ns | −0.0226 *** | +0.52 * | −0.0025 ns | (未測；retrieval 已退無上行空間) | — |
| Rung 1 λ=0.05 | +0.16 ns | −0.74 ** | +0.09 ns | −0.0000 ns | −1.14 *** | −0.0096 *** | **−0.0039** | ❌ |
| Rung 1 λ=0.1 | −0.04 ns | +0.07 ns | −0.60 ** | +0.0022 ns | −0.01 ns | −0.0028 ns | **−0.0085** | ❌ |
| Rung 1 λ=0.3 | +0.41 ns | +0.90 ** | +2.07 *** | +0.0063 ** | −0.01 ns | −0.0042 * | **−0.0851** | ❌ |
| Rung 1 λ=0.5 | +0.42 ns | +1.09 *** | +2.34 *** | +0.0075 ** | +0.12 ns | −0.0041 * | (預期 ❌, 未測) | ❌ |
| Rung 1 λ=1.0 | +0.32 ns | **+4.82 ***** | **+5.20 ***** | **+0.0216 ***** | +1.81 *** | +0.0029 ns | **−0.0373** | ❌ |

> **讀法 1：sem R@1 退化是結構性極限，與 λ 無關。** Rung 1 五個 λ（0.05→1.0）的 R@1 Δ 全部 ns（CI 都跨 0），cluster 偏移亦相同方向（cl R@1 全為負）。listwise 怎麼加強都推不動 R@1。
>
> **讀法 2：faithfulness 在 λ=0 之外為奇異點。** 任何 λ ≥ 0.05 即把 `no_lesion`(lesion-type) drop 由 +0.038 翻成負（−0.004~−0.085），且 `no_global` 從 v3 的 −0.030 翻成 +0.046~+0.124——listwise 將 attribution mass **整體路由到 global**，繞過 multiplicative 的病灶 gate；`no_random` 全程 ≈ 0（為選擇性反轉，非普遍敏感性增強）。與 Rung 0 / Table B2 raw-`case_db` 同一失敗簽名。
>
> **讀法 3：faithful 區與 retrieval-improving 區零重疊。** retrieval Δ 對 λ 單調上升（R@10/R@20/MRR 在 λ=0.3 起轉正、λ=1.0 達 +4.82/+5.20/+0.022），但 faithfulness 在 λ=0.05 就已反轉。λ=0.05/0.1 處於「faithfulness 已崩、retrieval 又零收益（甚至 cl MRR −0.0096***）」的雙輸區，這把張力證為**硬約束**而非可調曲面。
>
> **結論：v3（λ=0, BCE only）是 CEAH 在 faithfulness–ranking 空間中的奇異 Pareto-faithful 操作點；Table C1 的 sem R@1/R@20 退化是 architecture-enforced lesion grounding 的結構性代價，非可調瑕疵。** 這同時也是為什麼 CEAH 的賣點是 **cluster recall + faithful attribution**（Tables C1/C3/C4），而不是 sem R@K 全面 SOTA。

**取得指令：**
```bash
# Rung 0
$PY -m diagnosis_model.cause_inference.train_ceah \
  --case_db_dir diagnosis_model/cause_inference/outputs/case_db \
  --train_pool_path diagnosis_model/cause_inference/outputs/case_db/train_candidate_pool.pt \
  --output_dir diagnosis_model/cause_inference/outputs/ceah_rung0 \
  --attribution_mode softmax --scoring_mode multiplicative \
  --epochs 20 --batch_size 32 --lr 1e-4 --lambda_sparsity 0.0 --text_dropout 0.5 \
  --eval_max_queries 300 --soft_labels --soft_lo 0.90 --soft_hi 1.0

# Rung 1 λ-scan（per λ）
for L in 0.05 0.1 0.3 0.5 1.0; do
  $PY -m diagnosis_model.cause_inference.train_ceah \
    --case_db_dir diagnosis_model/cause_inference/outputs/case_db \
    --train_pool_path diagnosis_model/cause_inference/outputs/case_db/train_candidate_pool.pt \
    --output_dir diagnosis_model/cause_inference/outputs/ceah_rung1_l${L} \
    --attribution_mode softmax --scoring_mode multiplicative \
    --epochs 20 --batch_size 32 --lr 1e-4 --lambda_sparsity 0.0 --text_dropout 0.5 \
    --eval_max_queries 300 --lambda_rank $L --rank_temp 0.1
done

# Faithfulness gate
$PY -m diagnosis_model.cause_inference.faithfulness_eval \
  --case_db_dir diagnosis_model/cause_inference/outputs/case_db \
  --ceah_ckpt <ckpt> --attribution_mode softmax --scoring_mode multiplicative \
  --gamma 0.0 --max_queries -1

# Paired query-level bootstrap CI（vs v3 或 vs Phase1）
$PY <bootstrap_ci.py> --a <per_query_A> --a_kind {ceah,phase1} \
                     --b <per_query_B> --b_kind {ceah,phase1}
```

---

## D. Coarse stage — ABQ（Phase 3 加速 + Phase 4 壓縮）

### Table D1 — Phase 3 case encoder：品質 / 速度（K=20）

| Method | sem R@10 | MRR | per-q ms | Notes |
|---|---|---|---|---|
| Phase 1 hungarian（teacher） | 44.4% | 0.298 | 15.4 | multi-vector + Hungarian |
| DeepSets single distill | 44.5% | 0.297 | 1.1 | 蒸餾打平 |
| **DeepSets dual-target** | **44.7%** | 0.298 | **1.1** | +0.3 R@10, 14× faster |

接 CEAH 重排後（text=medical）：

| Method | sem R@10 | per-q ms |
|---|---|---|
| Phase 1 + CEAH | 45.3% | 18.2 |
| DeepSets dual + CEAH | 45.3% | **3.4**（5.4× faster） |

**取得指令：**
```bash
$PY -m diagnosis_model.cause_inference.eval_phase1_aligned --include_phase1 \
  --checkpoints deepsets_dual=diagnosis_model/cause_inference/outputs/encoder_final/best_encoder.pt \
  --top_k_cases 20 --semantic_threshold 0.95
$PY -m diagnosis_model.cause_inference.eval_hybrid_aligned --include_phase1 \
  --ceah_ckpt diagnosis_model/cause_inference/outputs/ceah_v3/best_ceah.pt \
  --checkpoints deepsets_dual=diagnosis_model/cause_inference/outputs/encoder_final/best_encoder.pt \
  --gammas 0.0 0.5 0.75 1.0 --text_kind medical
```

### Table D2 — Phase 3 架構 / 訓練 ablation

| Encoder | sem R@10 | sem R@10 (lesion≥4) |
|---|---|---|
| Mamba3 master-slave | 44.5% | 42.2% |
| MeanPool | 44.3% | 41.8% |
| **DeepSets** | **44.7%** | **44.1%** |

| 訓練方式 | sem R@10 |
|---|---|
| Distillation only（listwise KL） | 44.5% |
| **+ case-cause InfoNCE（dual）** | **44.7%** |
| + miss-weighted sampler | 44.6% |

| 1-stage full-vocab（negative result） | sem R@10 |
|---|---|
| Phase 1 + candidate pool（2-stage） | 44.4% |
| SigLIP2 global → 56k cause（1-stage） | 13.5% |
| DeepSets dual → 56k cause（1-stage） | 12.1% |

**取得指令：**架構/訓練變體改 `train_case_encoder --encoder_type {deepsets,mamba,mean}` /
`--use_infonce`；full-vocab 用 `eval_full_vocab.py`。
```bash
$PY -m diagnosis_model.cause_inference.eval_full_vocab \
  --checkpoints deepsets_dual=diagnosis_model/cause_inference/outputs/encoder_final/best_encoder.pt
```

### Table D3 — Phase 4 RVQ 壓縮 sweep（retrieval-side proxy, Regime A, top_k=20）

> 本表是**未接 CEAH 的 retrieval-side proxy**，用來看 RVQ 對 coarse retrieval 本身的影響。為避免 raw production path 與 fine-tuned ablation 混淆，raw / fine 拆欄。Dense Δ 欄為 reference（`—`）。

| RVQ config | Compression×（vs fp32 dense） | raw sem R@10 | raw Δ vs dense | fine sem R@10 | fine Δ vs dense |
|---|---:|---:|---:|---:|---:|
| Dense fp32 | 1× | 45.6% | — | 44.7% | — |
| M=4 K=256 | 768× | 45.1% | −0.5 pp | 45.5% | +0.8 pp |
| M=2 K=64 | 2048× | 45.0% | −0.6 pp | 43.5% | −1.2 pp |
| M=1 K=16 | 6144× | 45.6% | −0.1 pp | 43.9% | +0.0 pp |

> fine sweep 12 個 config Δ 在 ±1.7pp（SE=1.26pp）內＝noise；raw production path 三個壓縮率全落在 dense −0.6pp 內（M=4 K=256 −0.5、M=2 K=64 −0.6、M=1 K=16 −0.1）。aggregation buffer 吸收量化噪聲，raw / fine 兩條 path 結論一致。

**取得指令：**
```bash
# fine-tuned path（run_sweep 預設 top_k=20）
$PY -m diagnosis_model.cause_inference.rvq_rerank.run_sweep --M_list 1 2 4 8 --K_list 16 64 256

# raw production path（eval_harder 指向 raw encoder/case_db，top_k=20 即 Regime A）
$PY -m diagnosis_model.cause_inference.rvq_rerank.eval_harder \
  --encoder_ckpt diagnosis_model/cause_inference/outputs/encoder_raw/best_encoder.pt \
  --case_db_dir  diagnosis_model/cause_inference/outputs/case_db_raw \
  --rvq_configs 4,256 2,64 1,16 --eval_configs 20,0.95
```

### Table D4 — **Phase 4 端到端 through CEAH（最 production-relevant）**

**Regime A（top_k=20, production, 有 buffer）— 本 session 2026-05-22 實測：**

| coarse 來源 | 壓縮× | sem R@10 | cl R@10 | sem MRR |
|---|---|---|---|---|
| dense（reference） | 1× | 47.6% | 39.6% | 0.303 |
| RVQ-only M=4 K=256 | 768× | 47.2% | 39.8% | 0.302 |
| + Light rerank | 768× | 47.8% | 39.8% | 0.304 |
| RVQ-only M=2 K=64 | 2048× | 47.5% | 40.2% | 0.308 |
| + Light rerank | 2048× | 47.2% | 40.1% | 0.306 |
| **RVQ-only M=1 K=16** | **6144×** | **47.0%** | **39.6%** | 0.303 |
| + Light rerank | 6144× | 46.4% | 38.5% | 0.288 |

> 壓縮損害穿不過 CEAH：6144× 只掉 0.6pp sem、cluster 不動。reranker 多餘甚至微害。

**Regime B（top_k=1, buffer-free, single-case ANN）— 2026-05-22 同 session 實測：**

| coarse 來源 | 壓縮× | sem R@10 | cl R@10 | sem MRR |
|---|---|---|---|---|
| dense | 1× | 51.7% | 46.9% | 0.313 |
| RVQ-only M=4 K=256 | 768× | 48.4% | 45.4% | 0.298 |
| **+ Light rerank** | 768× | **51.0%** | **46.7%** | 0.310 |
| RVQ-only M=2 K=64 | 2048× | 44.6% | 44.8% | 0.280 |
| + Light rerank | 2048× | 46.2% | 44.3% | 0.287 |
| RVQ-only M=1 K=16 | 6144× | 36.8% | 40.5% | 0.240 |
| **+ Light rerank** | 6144× | **39.7%** | **42.8%** | 0.251 |

> pool ≈ 4.4 unique cause/query（buffer 消失）。壓縮真傷（6144× −14.8pp sem），reranker
> 真救（M4K256 +2.6pp sem 回到接近 dense）→ reranker 唯一 niche 是 buffer-free 部署。

**取得指令：**（Regime A 換 `--top_k_cases 1` 即得 Regime B）
```bash
$PY -m diagnosis_model.cause_inference.eval_ceah_compressed \
  --coarse_case_db_dir diagnosis_model/cause_inference/outputs/case_db_raw \
  --encoder_ckpt diagnosis_model/cause_inference/outputs/encoder_raw/best_encoder.pt \
  --rvq_root diagnosis_model/cause_inference/outputs/rvq_rerank_raw \
  --fine_case_db_dir diagnosis_model/cause_inference/outputs/case_db \
  --ceah_ckpt diagnosis_model/cause_inference/outputs/ceah_v3/best_ceah.pt \
  --cluster_json diagnosis_model/cause_inference/outputs/cause_clusters_llm.json \
  --attribution_mode softmax --scoring_mode multiplicative \
  --top_k_cases 20   # Regime B: --top_k_cases 1 (--output_dir ...eval_k1)
```
> A/B 皆 2026-05-22 同 session 實測（A→`ceah_compressed_eval/`，B→`ceah_compressed_eval_k1/`）。

### Table D5 — Phase 4 Regime B 壓縮壓力測試（retrieval-side proxy, top_k=1）

> 本表是**未接 CEAH 的 buffer-free retrieval-side proxy**，不是 production headline。它用來比較 raw production encoder 與 fine-tuned encoder 在 top_k=1、無 aggregation buffer 時的 quantization robustness；D4 才是 through-CEAH 的端到端結果。

| Method | Comp×（vs fp32 dense） | raw sem R@10 | fine sem R@10 | raw gap recovered | fine gap recovered |
|---|---:|---:|---:|---:|---:|
| Dense fp32 | 1× | 51.7% | 53.7% | — | — |
| RVQ-only M=4 K=256 | 768× | 48.4% | 50.6% | — | — |
| + Light rerank | 768× | 51.0% | 51.8% | 79% | 39% |
| + Full analytic | 768× | 51.6% | 53.7% | — | 100% |
| RVQ-only M=2 K=64 | 2048× | 44.6% | 46.3% | — | — |
| + Light rerank | 2048× | 46.2% | 50.0% | 22% | 56% |
| + Full analytic | 2048× | 50.5% | 51.5% | — | 69% |
| RVQ-only M=1 K=16 | 6144× | 36.8% | 37.2% | — | — |
| + Light rerank | 6144× | 39.7% | 46.2% | 19% | 54% |
| + Full analytic | 6144× | 44.6% | 45.8% | — | 52% |

> 這張表解釋了為什麼 D4 / D5 的 dense 分數會不同：D4 是 raw coarse → CEAH 的端到端 path；D5 是 retrieval-side proxy，且同時列 raw / fine encoder。production 真正使用的是 top_k=20 aggregation buffer，因此 reranker 只作為 buffer-free deployment fallback，不是 FaCE-R production 元件。

**取得指令：**
```bash
$PY -m diagnosis_model.cause_inference.rvq_rerank.eval_final \
  --configs 4,256,reranker_M4_K256_light 2,64,reranker_M2_K64_light 1,16,reranker_M1_K16_light \
  --eval_configs 20,0.95 1,0.95 --K_top 50
```

### Table D6 — Scale benchmark（latency / memory, N=12K..1M, M=4 K=256）

Latency p50 @ bs=1：

| N | dense | rvq_only | rvq_light | rvq_full_analytic |
|---|---|---|---|---|
| 12,780 | 0.08 ms | 0.10 ms | 0.56 ms | 0.13 ms |
| 100,000 | 0.56 ms | 0.13 ms | 0.60 ms | 0.16 ms |
| 500,000 | 2.59 ms | 0.15 ms | 0.62 ms | 0.18 ms |
| 1,000,000 | 5.15 ms | 0.19 ms | 0.59 ms | 0.22 ms |

Memory @ N=1M（全部 compression 皆以 fp32 dense 為 reference）：

| Method | Index size | Compression |
|---|---:|---:|
| dense fp32 | 3,072 MB | 1× |
| rvq_only (M=4) | 4 MB | 768× |
| rvq_light | 17 MB | 181× |
| rvq_full_analytic | 3,076 MB | 1× |

**取得指令：**
```bash
$PY -m diagnosis_model.cause_inference.rvq_rerank.benchmark_scale \
  --rvq_dir diagnosis_model/cause_inference/outputs/rvq_rerank/rvq_M4_K256 \
  --light_reranker_ckpt diagnosis_model/cause_inference/outputs/rvq_rerank/reranker_M4_K256_light/best.pt \
  --N_list 12780 50000 100000 500000 1000000 --batch_sizes 1 32
```

### Table D7 — Aggregation-buffer 吸收 scaling law `D ≈ c·ε^p/K^q`

| Encoder | c | p (ε exp) | q (K exp) | R² |
|---|---|---|---|---|
| Fine-tuned (case_db) | 59.9 | 0.93 | 0.88 | 0.90 |
| Raw SigLIP2 (case_db_raw) | 156.0 | 1.11 | 0.78 | 0.79 |

Encoder-invariance（共用 exponents 只損 ΔR²=0.006）：

| Model | params | pooled R² |
|---|---|---|
| 3-param fully shared | 3 | 0.757 |
| **4-param 共用 exponents** | 4 | **0.828** |
| 6-param fully separate | 6 | 0.834 |

> c_raw/c_fine ≈ 2.06 → fine-tuning 給 2× quantization robustness。
> selector：`K* = (c·ε/D_target)^{1/q}`，q≈0.84。

**取得指令：**
```bash
$PY -m diagnosis_model.cause_inference.rvq_rerank.eval_absorption_surface \
  --encoder_ckpt diagnosis_model/cause_inference/outputs/encoder_final/best_encoder.pt \
  --case_db_dir diagnosis_model/cause_inference/outputs/case_db \
  --rvq_root diagnosis_model/cause_inference/outputs/rvq_rerank
# raw encoder 換 encoder_raw / case_db_raw / rvq_rerank_raw 再跑一次，兩條曲線一起 fit
```

### Table D8 — Importance-weighted RVQ：K* shift（damage≤1pp 最小 top_k，越小越好）

| Config | Comp× | vanilla K* | ranking K* | agg(iso) K* | agg_inv K* |
|---|---|---|---|---|---|
| M=8 K=256 | 384× | 2 | 2 | 2 | 2 |
| M=4 K=256 | 768× | 8 | **3** | 3 | 3 |
| M=2 K=64 | 2048× | 15 | **10** | 20 ⚠ | 20 ⚠ |
| M=1 K=16 | 6144× | 20 | **15** | 15 | 20 ⚠ |
| M=1 K=4 | 24576× | 20 | **15** | 15 | 15 |

> ranking variant 4/5 configs K* 降 5（25–33%），drop-in。agg/agg_inv 中度壓縮退步（negative）。

**取得指令：**
```bash
# fit 4 variants 後對每個跑 absorption surface
for v in vanilla ranking agg agg_inv; do
  $PY -m diagnosis_model.cause_inference.rvq_rerank.eval_absorption_surface \
    --rvq_root diagnosis_model/cause_inference/outputs/rvq_rerank/variants/$v \
    --out diagnosis_model/cause_inference/outputs/rvq_rerank/variants/$v/absorption_surface.json
done
```

---

## E. Cause taxonomy 對照

### Table E1 — LLM vs HDBSCAN cluster（cluster R@K）

現行 metric、純 CEAH v3（2026-05-22 實測）。論文只報這兩條 taxonomy：LLM 為主、HDBSCAN 為 LLM-free baseline。HDBSCAN baseline 走固定 pipeline（PCA(50)+UMAP(15,5,cosine)+HDBSCAN(eom, mcs=100)+reassign-singletons(0.70)），無 hyperparameter 旋鈕。

| Taxonomy | clusters | cl R@1 | cl R@10 | cl MRR |
|---|---|---|---|---|
| HDBSCAN baseline（LLM-free） | 100 | 4.7% | 19.7% | 0.103 |
| **LLM topic（論文主）** | 466 | **20.1%** | **39.8%** | **0.271** |

> LLM ≫ HDBSCAN baseline by ~2× across all three metrics。LLM 在 absolute 與粒度合理性上都最強。
> 註：[inference.txt G] 舊記 LLM 57.9% / coarse 30.2% 是 metric 語意改寫前的數字，已被本表取代。

**Supplementary — HDBSCAN 粒度 ablation（不在 main table，僅佐證選 mcs=100 而非更細）：**

| Taxonomy | clusters | cl R@1 | cl R@10 | cl MRR |
|---|---|---|---|---|
| HDBSCAN fine (mcs=6) | 2807 | 0.5% | 3.0% | 0.015 |

> HDBSCAN fine 的 cluster R@K 退化到接近 strict semantic match 水準（cl R@10 3% vs sem R@10 47.6%），證明 disease-topic recall 在 ~3000 cluster 粒度下已失去 leniency 意義，故 baseline 採 mcs=100 的 coarse 設定。

**取得指令：**
```bash
# Main table
for cj in cause_clusters_llm cause_clusters_hdbscan; do
  $PY -m diagnosis_model.cause_inference.eval_ceah \
    --case_db_dir diagnosis_model/cause_inference/outputs/case_db \
    --ceah_ckpt diagnosis_model/cause_inference/outputs/ceah_v3/best_ceah.pt \
    --output_dir diagnosis_model/cause_inference/outputs/ceah_v3_cluster_${cj} \
    --cluster_json diagnosis_model/cause_inference/outputs/${cj}.json \
    --attribution_mode softmax --scoring_mode multiplicative --gammas 0.0 --dump_gamma 0.0
done

# Supplementary（HDBSCAN fine 2807；需從 outputs/_archive_clusters/ 取回，或重新跑
# recluster_causes.py cluster --min_cluster_size 6 + reassign-singletons）
```

---

## F. Faithfulness 強化 ablation（VLM-Lesion auxiliary；非 FaCE-R 主流程）

> CLSC / LSCFT 屬於 **VLM-Lesion 訓練章節** 的 representation-learning ablation，不是 cause_inference 主流程元件。它們在這裡只作為 downstream validation：觀察不同 lesion encoder 對 CEAH faithfulness 的影響。主流程仍使用 baseline VLM-Lesion，因為 CLSC 雖改善 CEAH lesion-masking faithfulness，但在上游 VLM-Lesion 分類任務上沒有穩定全面優於 baseline。

### Table F1 — CLSC（cross-lesion supervised contrastive）vs baseline VLM-Lesion

| 條件 | baseline | +CLSC | Δ | 相對改善 |
|---|---|---|---|---|
| no_lesion · global-type | +0.0289 | +0.0312 | +0.0023 | +8% |
| no_lesion · **lesion-type** | +0.0378 | **+0.0444** | +0.0066 | **+17%** |
| no_lesion · **mixed** | +0.0413 | **+0.0694** | +0.0281 | **+68%** |
| no_random（sanity） | +0.0027 | +0.0022 | −0.0005 | 維持 ≈0 ✓ |

retrieval 無感佐證：CEAH retrieval baseline 46.8% → +CLSC 47.2%（+0.4pp，noise 內）。CLSC 的價值是讓 lesion feature 更 lesion-grounded，而不是提升 cause retrieval headline；是否採用為預設 encoder 需回到 VLM-Lesion 分類主指標決定。

### Table F2 — LSCFT（counterfactual masking, negative result）

| 設定 | lesion-type drop | mixed drop | N=2 | N≥3 |
|---|---|---|---|---|
| baseline | 0.0378 | 0.0413 | 0.0467 | 0.0531 |
| **+CLSC (xL1.5)** | **0.0444** | **0.0694** | **0.0632** | **0.0587** |
| LSCFT P=80 | 0.0369 | 0.0403 | 0.0459 | 0.0548 |
| LSCFT P=180 | 0.0067 ↓↓ | 0.0708 | 0.0336 ↓ | 0.0296 ↓ |

> CLSC（discriminative signal）transfer 到 CEAH attribution；LSCFT（counterfactual signal）
> 訓練目標達成但不 transfer。論文中建議把此結果放在 VLM-Lesion 訓練章節，FaCE-R 章節只引用為「lesion representation quality 會影響 CEAH faithfulness」的下游證據。

**F1 + F2 取得指令：**
```bash
# CLSC ckpt 訓練（vl_classifier，~50 min）
cd diagnosis_model/vl_classifier && python train.py \
  --multipos --fusion --freeze_text_encoder --cross_lesion --cross_alpha 1.5 --cross_tau 0.1 \
  --output_dir outputs/siglip2_base_patch16_224_multipos_fusion_xL1.5_en_zh
cd /mnt/ssd/YJ/fish_disease_system
# 用 CLSC/LSCFT ckpt 各自 build_case_database --vlm_lesion <ckpt> → case_db_<tag>/
# 再對每個 case_db 跑 faithfulness_eval（同 Table C3/C4 指令，改 --case_db_dir）
```
