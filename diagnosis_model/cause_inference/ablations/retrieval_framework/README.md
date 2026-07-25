# 檢索框架消融：案例式 vs 直接圖文匹配

論文檢索圖（`paper/make_figures.py` 之 `build_retrieval_framework_rpf1`）的資料產生腳本。
故事軸 = **案例式框架（OAVLE+CEAM）>> 直接圖文匹配（CLIP/SigLIP2）**，同任務比框架。

從 repo root 執行：`$PY -m diagnosis_model.cause_inference.ablations.retrieval_framework.<module>`

| 腳本 | 做什麼 | 輸出 |
|---|---|---|
| `baseline_imgtext_vs_ours.py` | baseline = 影像↔影像檢索 + **影像↔病因文字對齊**（CLIP/SigLIP2 之圖文對比本命）；ours = 生產 DeepSets 檢索 + score_candidates。R/P/F1 @ n，k∈{3,5} | `rpf1_imgtext.json` |
| `ours_ceam_text_ablation.py` | ours 病因排序改用**生產 CEAM**（有文字槽），比較吃文字 vs 不吃 vs score_candidates，k=3 | `rpf1_text.json` |
| `topk_cases_sweep_ceam.py` | CEAM 排序下掃 top_k_cases，看最佳相似案例數 | `rpf1_k_ceam.json` |

## 定案結論（valid 1583, k=3, 群集 relevance, @ n=5）

- **Ours（生產 OAVLE+CEAM）群集 F1 = 0.467 / 語意 0.585**；SigLIP2 0.325/0.419、CLIP 0.341/0.453 → **ours 領先 ~14–16pp，贏在 precision**（把對的病因排最前）。
- **baseline 病因排序必用 query 影像↔病因文字對齊**（`cos(query_global, cause_text)`），**不可用 score_candidates**（那是案例式、會假打平）。
- **零訓練案例式**（score_candidates）群集 F1 0.417 也贏直接匹配 → 贏的是框架、非僅 CEAM 訓練。
- **文字增益中性**（CEAM+文字 0.465 ≈ 不吃 0.467）→ 視覺已足夠。
- **CEAM 排序最佳 top_k_cases = 2–3**（非 score_candidates 的 5）→ 坐實論文 k=3。

論點：CLIP/SigLIP2 把影像對齊**視覺描述**，但病因是**抽象因果推論**，魚照片對不上因果文字；案例式（影像→相似病例→專家標的病因）繞過此牆。
