# GROD — Grounded Region Open-vocabulary Detector

GROD is a single-stage open-vocabulary lesion detector. A small **semantic head**
on RF-DETR's decoder query features projects each detected region into the frozen
SigLIP2 text space, so lesion crops are **never re-encoded** — one forward pass
yields box + objectness + a semantic vector `z` that retrieves directly against a
symptom-caption bank.

It is a standalone paper contribution (own acronym, alongside ABQ / CEAM). The
full narrative, reviewer-defense, and ablation matrix live in
[`note/merge_narrative.md`](../../note/merge_narrative.md).

> Run everything from the **repo root** via `python -m diagnosis_model.grod.<module>`,
> never by `cd`-ing into this folder (paths are repo-root-relative).

## vs. the conventional two-stage pipeline

The conventional flow detects boxes, **crops** each lesion, and **re-encodes**
every crop through a separate SigLIP2 vision tower. GROD folds the region
encoding into the detector's own decoder, so the SigLIP2 vision tower is dropped
at inference entirely (only its frozen *text* anchors remain, precomputed once).

| | conventional two-stage | **GROD (single-stage)** |
|---|---|---|
| inference path | detect → crop ×N → SigLIP2 re-encode ×N → retrieve | detect → semantic head → retrieve |
| inference params | RF-DETR 33.6 M **+ SigLIP2 vision 92.9 M** | RF-DETR 33.6 M only (semantic head +0.2 M) |
| latency (bs=1, ~2 lesions) | 7.7 ms | **5.3 ms** (−2.4 ms; grows with #lesions) |
| region feature | isolated crop (no global context) | decoder query (global context built in) |
| faithfulness (lesion-mask drop) | −0.0314 (reversed) or needs fusion bolt-on | **+0.0141** by construction |

Latency measured on one 32 GB GPU, RF-DETR-Medium @576, SigLIP2-base @224. The
two-stage re-encode cost scales with the number of lesions (one SigLIP2 forward
per crop); GROD is constant (one detector forward regardless of #lesions).

---

## Headline — faithfulness-by-construction

The controlled ablation fixes the semantic anchor (frozen SigLIP2 text) and
varies **only the routing** of the lesion's visual feature:

| routing | lesion-mask drop (lesion-type) | faithful? |
|---|---|---|
| isolated crop → re-encode (≈ vl_classifier old path / non-DETR RoI) | **−0.0314** | reversed ❌ |
| RF-DETR decoder query (`hs`) → semantic head | **+0.0141** (joint) | yes ✅ |
| fused VLM-Lesion (old production, needs `LocalGlobalFusionWrapper` bolt-on) | +0.0620 | yes, but bolt-on |

Because the DETR query is born with global context via cross-attention,
faithfulness comes from the **architecture**, not from any fusion module. Three
rungs of evidence:

1. **frozen probe** (zero training): −0.0314 → **+0.0063** (sign flips)
2. **joint training**: +0.0063 → **+0.0141**, retrieval held (sem R@10 0.442)
3. **coef sweep**: `no_lesion` all-positive / `no_top_α` all-negative across
   coef 0.5/1.0/2.0 → the faithful↔token-localization trade-off is **structural,
   not a hyperparameter** (a clean negative result).

---

## Benchmark tables (two-contribution positioning)

GROD is positioned as a **dual contribution**: an open-vocabulary lesion
detector (Table A) *and* a faithfulness-by-construction region encoder (Table B).
The two axes are evaluated independently, but **share the same baselines** — each
competing method is run once and scored on both tables.

**"Open-vocabulary" here = open *vocabulary* (retrieve against arbitrary
free-text symptom descriptions), not open *category* (detect unseen classes).**
The 19 symptom categories are long-tailed and the rare tail (7 classes, 193 anns
total, only 2 with ≥5 valid samples) is too small for a statistically meaningful
novel-category zero-shot split — a small rare-class probe is reported in the
appendix only, never as a headline claim.

All baselines are run **on our dataset** (faithfulness has ground truth only
here); we do not chase COCO/LVIS box AP (different task, not our contribution).
Numbers left blank — fill after running.

### Table A — open-vocabulary symptom retrieval axis

GROD ranks free-text symptom descriptions for each detected lesion. We do **not**
compare retrieval against the heavyweight fusion VLM-Lesion — that model is the
prior work's main contribution (multi-positive + LocalGlobalFusion, purpose-built
for lesion→symptom retrieval), so a 0.2 M Linear head is not a fair opponent there.
The apples-to-apples baseline is a same-recipe *directly fine-tuned* SigLIP2
(`siglip2_base_patch16_224_finetuned_en_zh`), which like GROD just aligns to the
caption bank without the fusion machinery.

| method | region routing | symptom R@1 | symptom R@5 | symptom R@10 |
|---|---|---|---|---|
| directly fine-tuned SigLIP2 (same-recipe baseline) | isolated crop | 0.822 | 0.903 | 0.934 |
| **GROD (ours)** | DETR query + semantic head | 0.655 | 0.747 | 0.761 |

Retrieval is the "comparable, not winning" axis — the headline is faithfulness
(Table B) + efficiency. GROD currently trails the fine-tuned baseline here
(R@10 0.761 vs 0.934): its region feature is a frozen 256-d decoder query passed
through a single 0.2 M Linear head, whereas the baseline fine-tunes the whole
SigLIP2 vision tower. Closing this gap (a stronger semantic head) is open work —
see the head-capacity note below. (3738 valid lesions.)

### Faithfulness is shown at two levels (Table B + B′)

GROD's faithfulness is evidenced at two independent levels, hence two tables —
they answer different questions and compare different things, so they are kept
separate rather than merged:

- **Table B (embedding level)**: does GROD's region *vector* ground the symptom
  semantics on the lesion? Mask the lesion, watch the symptom probability drop.
  Compares **routings** (crop / fused / GROD).
- **Table B′ (pixel level)**: does GROD's *attention* point at the lesion? Take a
  saliency heatmap, check if its peak lands in the box. Compares **saliency
  methods** on the same GROD model (Grad-CAM / RISE / GROD attention / …).

Both holding up means the grounding is real at both the vector and spatial level.

### Table B — interpretability / faithfulness axis (region feature, CEAM-free)

The headline faithfulness metric is measured on **GROD's own region feature**,
with no cause-attribution module in the loop: mask the lesion pixels, re-encode,
and see how much the region's symptom probability drops. P = softmax over the 19
symptom anchors (rank-aware, so it is comparable across feature spaces and is not
skewed by absolute cosine scale). **Larger positive drop = more faithful**
(removing the lesion should destroy the symptom evidence). `region_faithfulness.py`,
3738 valid lesions.

This is a **routing controlled experiment**, not an OVD leaderboard: the semantic
anchor (frozen SigLIP2 text) is fixed and only the *routing* of the lesion feature
changes. Rows are routing variants, not competing detectors — a YOLO-World/GDINO
region feature would add an unaligned-feature confound (their region vectors live
in their own space, not SigLIP2 text space) without making the contrast cleaner;
isolated-crop already is the non-DETR / RoI-routing proxy.

| method | region routing | baseline P(symptom) | lesion-mask prob drop ↑ |
|---|---|---|---|
| isolated crop + raw SigLIP2 (= non-DETR RoI proxy) | isolated crop | 0.063 | +0.008 |
| fused VLM-Lesion (old prod, needs fusion bolt-on) | crop + fusion | 0.133 | +0.068 |
| **GROD (ours)** | DETR query + semantic head | **0.354** | **+0.276** (4× fused, 34× crop) |

The isolated crop barely moves (+0.008): masking the lesion does not change its
symptom evidence, i.e. the routing is **not grounded** on the lesion. GROD's
probability collapses by 0.276 — the region feature is grounded on the lesion by
construction. (An end-to-end variant *through* CEAM gives a smaller but same-sign
+0.0141; the CEAM-free number above is the cleaner, stronger headline.)

### Table B′ — pixel-level localization (pointing game)

Table B measures *embedding* faithfulness; this corroborates it at the **pixel**
level. Each method produces a saliency heatmap per lesion; a hit = the heatmap's
peak falls inside the GT box. Higher = better localization. (`pointing_game.py`,
3250 valid lesions.)

| method | heatmap source | pointing hit-rate ↑ |
|---|---|---|
| random | — (chance ≈ bbox area frac 0.092) | 0.088 |
| Grad-CAM | backbone-feature gradient | 0.210 (2.3× chance) |
| RISE | black-box random-masking (model-agnostic) | 0.287 (3.1× chance) |
| Grad-CAM++ | higher-order gradient weighting | 0.389 (4.2× chance) |
| **GROD attention** | decoder MSDeformAttn sampling pts | **0.533** (5.8× chance) |

random matches chance exactly (0.088 vs 0.092) → the metric is unbiased. Across
all standard saliency methods (Grad-CAM, Grad-CAM++, RISE) the detector's own
attention routing localizes the lesion best (0.533, 5.8× chance, 1.4× the strongest
baseline). So GROD's faithfulness
holds at both the embedding and pixel levels.

---

## Model & parameters

Base detector: **RF-DETR-Medium** (DINOv2 backbone, single class-agnostic
`ABNORMAL`), `num_queries=300`, `resolution=576`.

| component | value |
|---|---|
| total params | 33.6 M |
| trainable in joint (`freeze_encoder=True`) | 11.5 M (34.2%) — projector + decoder transformer + box/class/query heads + semantic head |
| frozen (DINOv2 encoder) | 22.1 M (65.8%) |
| **semantic head** (`semantic_embed`) | **197 K** — `Linear(256 → 768)` |
| `hs` (decoder query dim) | 256 |
| semantic dim (SigLIP2 text) | 768 (CLIP ablation: 512) |

The semantic head uses **stock (unmodified) `rfdetr 1.6.5.post0`** — no fork. The
optional head + `loss_semantic` are added to stock rfdetr at runtime by
[`rfdetr_patch.py`](rfdetr_patch.py) (monkeypatch, auto-applied on
`import diagnosis_model.grod`); inference instead builds via the vendored decoder
in [`detector/`](detector/) (`build_grod_detector`), which imports no rfdetr.
Toggled via env (`RFDETR_SEMANTIC_DIM` / `RFDETR_GLOBAL_DIM` / `RFDETR_BACKBONE`);
all default off → plain stock detector.

---

## Pipeline & commands

> **Paths below are old-tree illustrative examples.** For the current dataset
> version substitute the `current` tree: images/COCO under
> `data/processed/current/detection/{train,valid}/` (NOT `data/detection/coco/_merged`,
> which is a stale snapshot), artifacts (case_db, text_anchors, checkpoints) under
> `data/processed/current/artifacts/{db,models}/`. The `current` symptom taxonomy
> is 15 classes; rebuild `merged_semantic` on the current detection COCO before
> joint training (none ships in the current tree yet).
>
> **DINOv3 backbone:** prefix any build/train/extract command with
> `RFDETR_BACKBONE=dinov3_base` (or `dinov3_small`/`dinov3_large`). The global
> head input width changes with the backbone (DINOv2 1536 → DINOv3-base 3072), so
> the global head must be re-distilled per backbone. rfdetr is stock + runtime
> monkeypatch (auto-applied on `import diagnosis_model.grod`); no env beyond the
> `RFDETR_*` switches is needed.

### 0. One-time data prep

```bash
# Merged COCO: detection boxes + per-box symptom_category_id (joined 1:1 from vl_classifier COCO)
python -m diagnosis_model.grod.build_merged_coco \
    --det_root data/detection/coco/_merged \
    --vlc_root data/coco/_merged \
    --out_root data/detection/coco/_merged_semantic

# Frozen SigLIP2 per-symptom text anchors [19, 768]
python -m diagnosis_model.grod.build_text_anchors \
    --symptoms data/raw/symptoms.json \
    --out diagnosis_model/grod/outputs/text_anchors.pt
```

### 1. Joint training (production)

`freeze_encoder=True`; `semantic_loss_coef=2.0` and `batch_size=16` are the
validated defaults (sweep best operating point). ~7 min/epoch, ~3.5 h for 30
epochs on one 32 GB GPU.

```bash
python -m diagnosis_model.grod.train_joint \
    --dataset_dir data/detection/coco/_merged_semantic \
    --pretrain_weights diagnosis_model/detection/outputs/rfdetr/checkpoint_best_total.pth \
    --anchors diagnosis_model/grod/outputs/text_anchors.pt \
    --output_dir diagnosis_model/grod/outputs/joint_rfdetr
```

Key flags (defaults shown): `--epochs 30 --batch_size 16 --grad_accum_steps 1
--semantic_loss_coef 2.0 --semantic_dim 768 --semantic_temp 0.07 --lr 1e-4`.

### 2. Gate — verify faithfulness end-to-end

Extract the trained `z`, swap it into a case_db, then run the standard
cause_inference attribution gate.

```bash
# extract trained pred_semantic z, IoU-matched to each case_db GT lesion box
python -m diagnosis_model.grod.extract_z_joint \
    --case_db_dir diagnosis_model/cause_inference/outputs/case_db_raw \
    --joint_ckpt diagnosis_model/grod/outputs/joint_rfdetr/checkpoint_best_regular.pth \
    --anchors diagnosis_model/grod/outputs/text_anchors.pt \
    --image_root data/detection/coco/_merged \
    --output_dir diagnosis_model/grod/outputs/z_joint --splits train valid

# rebuild case_db with z as lesion_embs (--from_joint = z is already final)
python -m diagnosis_model.grod.rebuild_case_db \
    --src_case_db diagnosis_model/cause_inference/outputs/case_db_raw \
    --hs_dir diagnosis_model/grod/outputs/z_joint --from_joint \
    --out_case_db diagnosis_model/cause_inference/outputs/case_db_joint

# standard cause_inference gate on the rebuilt db
python -m diagnosis_model.cause_inference.preprocessing.build_train_candidate_pool \
    --case_db_dir diagnosis_model/cause_inference/outputs/case_db_joint \
    --output_path diagnosis_model/cause_inference/outputs/case_db_joint/train_candidate_pool.pt
python -m diagnosis_model.cause_inference.train_ceah \
    --case_db_dir diagnosis_model/cause_inference/outputs/case_db_joint \
    --train_pool_path diagnosis_model/cause_inference/outputs/case_db_joint/train_candidate_pool.pt \
    --output_dir diagnosis_model/cause_inference/outputs/ceah_joint \
    --attribution_mode softmax --scoring_mode multiplicative --lambda_sparsity 0.0 --text_dropout 0.5
python -m diagnosis_model.cause_inference.faithfulness_eval \
    --case_db_dir diagnosis_model/cause_inference/outputs/case_db_joint \
    --ceah_ckpt diagnosis_model/cause_inference/outputs/ceah_joint/best_ceah.pt \
    --output_dir diagnosis_model/cause_inference/outputs/ceah_joint_faithfulness \
    --attribution_mode softmax --scoring_mode multiplicative
```

### CLIP ablation (reviewer defense)

Re-run anchors + training with a CLIP text encoder to show the routing flip is
VLM-agnostic. CLIP-base is 512-d, so `--semantic_dim 512`:

```bash
python -m diagnosis_model.grod.build_text_anchors \
    --model_name openai/clip-vit-base-patch16 \
    --out diagnosis_model/grod/outputs/text_anchors_clip.pt
python -m diagnosis_model.grod.train_joint ... \
    --anchors diagnosis_model/grod/outputs/text_anchors_clip.pt --semantic_dim 512
```

---

## Scripts

| script | role |
|---|---|
| `build_merged_coco.py` | detection COCO + per-box `symptom_category_id` → `_merged_semantic/` |
| `build_text_anchors.py` | frozen VLM text anchors `[C, D]` (`--model_name` swaps VLM) |
| `train_joint.py` | **production** joint detection+semantic training (stock rfdetr + `rfdetr_patch` monkeypatch) |
| `extract_z_joint.py` | run joint model, IoU-match GT→query, dump trained `z` |
| `rebuild_case_db.py` | swap case_db `lesion_embs` with `z` (`--from_joint`) |
| `extract_hs.py` | frozen-probe (rung-1 evidence): hook raw `hs` from a frozen detector |
| `train_semantic_head.py` | frozen-probe: train a `Linear(256→768)` on cached `hs` |
| `calibrate_thresholds.py` | two decoupled GROD objectness thresholds → `data/processed/current/thresholds.json` (abstain=Youden image-level, display=F2 per-query) |
| `train_abstain_head.py` | **ablation / OOD-research** 健/病 abstain head → `models/disease_head/neck_disease_head.pt` (`--feat dino_neck` default = 1536 RF-DETR tap-B DINOv2 neck, beats the legacy 770 `--feat pooled`). **Not wired** — a constant objectness threshold ties it on the held-out test split incl. OOD (see `LESION_GATE.md`). Add OOD negatives to `data/healthy_images/<folder>/` + retrain. |
| `render_anomaly_heatmap.py` | objectness-splat heatmap (zero-training report display; all 300 queries as Gaussian blobs weighted by `w`) — the demo's step-① display |

The frozen-probe path (`extract_hs` → `train_semantic_head` →
`rebuild_case_db` *without* `--from_joint`) is kept as the zero-training rung-1
evidence, **not** a production step.

## Lesion-gate display: objectness heatmap & two thresholds

The demo's step-① display is an **objectness heatmap** (`render_anomaly_heatmap.py`):
all 300 queries splatted as `w`-weighted Gaussian blobs, zero-training, smooth. It
is the same signal as the boxes, so it does not surface detector-blind lesions —
of GT lesions below 0.5, **77.5 % have objectness < 0.1** (a detector recall limit
no display threshold reaches; only 12.8 % sit in the recoverable 0.3–0.5 band).

**Two decoupled thresholds** (`calibrate_thresholds.py`, dataset-calibrated
constants — a learned per-image τ ties a constant, see `train_disease_head_perquery.py`):
`abstain_thresh` (健/病, image-level max-objectness, Youden ≈ 0.39) vs
`display_thresh` (box display, F2 recall-leaning ≈ 0.32). The demo reads
`thresholds.json` for its slider defaults and abstains on the constant
`abstain_thresh` (max objectness). A learned 健/病 abstain head was tried (DINOv2
neck, `train_abstain_head.py`) but does **not** beat the constant on the held-out
test split incl. OOD (AUROC 0.9997 vs 0.9991, OOD reject ≤ the constant), so it is
kept as an ablation only — see `LESION_GATE.md`.

## Notes

- `outputs/` is gitignored; checkpoints (~134 MB each, ~1.8 GB/run) and caches
  live there and are not tracked.
- `data/`, `outputs/`, `*.pt`, `*.pth` are gitignored repo-wide — do not commit them.
- IoU matching uses `--iou_thresh 0.5`; on the fish data ~99% of GT lesions match
  a query (these are training boxes, so predicted boxes nearly coincide).
