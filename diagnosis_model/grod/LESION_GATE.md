# GROD lesion gate — design rationale & history

Single source of truth for how `gpu_infer.py` / `gpu_infer_soft.py` decide
**which queries are lesions** and **when to abstain (healthy)**. The inference
code itself is kept minimal — all the "why" lives here.

## Current production behaviour

A **fixed global threshold τ on objectness** `w_i = sigmoid(pred_logits[i,0])`:

- **hard** (`gpu_infer.py`): keep queries with `w_i > τ`; if none kept ⟹ abstain (healthy).
- **soft** (`gpu_infer_soft.py`): keep all queries with continuous weight `w`; abstain iff `max_i w_i < τ`.

`τ = DEFAULT_LESION_THRESH = 0.5`. No learned threshold, no disease head in the
`gpu_infer*.py` CLI inference path.

### Demo (`demo/app_gradio.py`) — two decoupled thresholds + heatmap display

The CLI keeps the single 0.5. The **demo** splits the one knob into two
dataset-calibrated constants (one threshold did two jobs badly — selection vs
abstain want different cuts):

- `abstain_thresh` (健/病 judgement, on image-level max objectness, Youden-optimal ≈ 0.39)
- `display_thresh` (which boxes to show → feeds the ② classification cards, F2
  recall-leaning ≈ 0.32)

Both come from `calibrate_thresholds.py` → `data/processed/current/thresholds.json`
(the demo reads it for slider defaults; falls back to 0.5/0.35). The demo's step-①
display is the objectness heatmap (`render_anomaly_heatmap.py`), not boxes. Note
the detector-blind 77.5 % (objectness < 0.1) is a detector recall limit no
threshold reaches.

## How τ is derived — `compute_lesion_threshold.py`

Per-query GT labels (`extract_disease_perquery.py`: IoU-match GT boxes → queries)
→ sweep a global threshold → pick the **train micro selection-F1 optimum**.

- τ\* = **0.497 ≈ 0.5**; val selection F1 = 0.806 (P 0.813 / R 0.799).
- F1 is flat over τ∈[0.41, 0.61] — selection is insensitive to the exact value.
- Writes `models/disease_head/lesion_threshold.json` (the authoritative value).
- Scope: GROD objectness only. NOT the base RF-DETR detector threshold (different score).
- `calibrate_thresholds.py` is the two-threshold sibling (abstain + display, see the
  demo subsection above); `compute_lesion_threshold.py` here is the single selection τ.

## Why a constant, not a learned gate (the experiments)

We tried to beat the constant and could not, defensibly. Harness:
`train_disease_head_perquery.py`, `probe_lesion_classifier.py`,
`nms_lesion_select.py`, `eval_gate_joint.py` (all kept under `grod/`).

| selection gate | val selection F1 |
|---|---|
| fixed constant τ≈0.5 | **0.806** |
| learned image-adaptive τ(g) | 0.805–0.807 (ties; residual-on-constant init guarantees ≥ const) |
| per-query classifier (objectness + centered z·anchor saliency + box) | 0.804–0.807 (ties, +0.002 = noise) |
| NMS de-duplication | no gain / hurts (duplicates are IoU<0.5; NMS removes real boxes when it fires) |

**Why there is no headroom:**
- Per-image lesion/background objectness separating interval is **wide (mean 0.764)** and image-stable → a constant lands inside it; the per-image optimal τ is **unpredictable from the image** (ridge held-out R² = −0.14).
- 22.8% of diseased images have an "objectness overlap" (a background query out-fires the weakest lesion). These are mostly **duplicate detections of the real lesion** — semantically *more* lesion-like than the matched box (saliency 0.603 vs 0.547, higher 60% of the time), so neither a threshold nor semantics can exclude them.
- Residual genuine FP vs TP separate only weakly on semantics (Cohen's d = 0.43, below the useful bar).
- Net: the residual selection error is **detector-level** (recall misses + genuine FP), not a selection-modelling gap.

### Joint abstain + selection (`eval_gate_joint.py`, val healthy+diseased)

| gate | box F1 | box P | box R | img sens | img spec |
|---|---|---|---|---|---|
| **A const (selection + "no box ⟹ healthy")** | **0.874** | 0.878 | 0.870 | 0.985 | 0.942 |
| B disease-head τ(g) both (former production) | 0.619 | 0.468 | 0.913 | 0.986 | 0.945 |
| C const selection + disease-head abstain | 0.874 | 0.878 | 0.870 | 0.985 | 0.942 |

The former τ(g) production gate **over-selects** (P 0.468). The constant matches the
learned abstain on healthy rejection (spec 0.942 vs 0.945) and sensitivity, so the
disease head is redundant for healthy-fish abstain → dropped from inference.

### Paper claim (defensible framing)

> Lesion selection on GROD objectness is solved by a single fixed threshold
> (F1≈0.88 jointly). A learned image-adaptive threshold and a per-query classifier
> (objectness + semantic saliency + geometry) do **not** improve it beyond noise,
> because the residual error is **detector-level** (recall misses + weakly-separable
> genuine false positives, d=0.43), not a selection-modelling gap.

Do **not** claim "perfect separation" — 22.8% objectness overlap exists (mostly
harmless duplicate detections). The claim is "no headroom for a learned selector
above the constant", with the detector as the real bottleneck.

## Disease heads — keep them distinct

| Head | Feature | Role | Where used |
|---|---|---|---|
| `ThresholdHead` | `g[768]` → τ(g), verdict `max_w ≥ τ(g)` | per-image **lesion-selection** τ (≈ adaptive `det_thresh`) | trained by `train_disease_head.py`; **not** wired — a constant τ ties it (above) |
| `DiseaseHead` **(1536)** | RF-DETR **tap-B DINOv2 neck** (4-scale patch-mean) → p, verdict `p ≥ τ*` | image-level **健/病 abstain** | `train_abstain_head.py --feat dino_neck`; **ablation only — not wired** (a constant ties it, below) |
| `DiseaseHead` (770, ablation) | `concat(g[768], max_w, Σw)` → p | same role, weaker | `--feat pooled`; kept for comparison |

**Why the neck, not `g`** (ablation 2026-06-10, same data/protocol): the distilled
global `g` is SigLIP2-aligned, which *strips* pure-visual signal (see
`distill_global_mlp.py`); the raw pre-distillation DINOv2 neck keeps it. The 1536
neck head beats the 770 on every axis — **OOD (sashimi) 50/50 vs 49/50, disease
recall 0.994 vs 0.983, AUROC 0.994 vs 0.986** (this is the feature-choice ablation;
both numbers are train-set-optimistic). Current ckpt:
`models/disease_head/neck_disease_head.pt` (dim 1536, val AUROC 0.994, τ*≈0.277).
Putting the bit on `g` directly (a "768+1" augmented dim) is the worst option — weakest
substrate **and** it risks CEAH faithfulness; the neck head touches neither.

**Not wired anywhere (2026-06-12).** Both `gpu_infer*.py` and `demo/app_gradio.py`
abstain on the constant objectness `abstain_thresh`. A held-out test-split A/B
(incl. structured OOD: sashimi / SalmonScan) showed the neck head does **not** beat
the constant — AUROC 0.9997 vs 0.9991, and OOD reject is ≤ the constant on every
source (sashimi 11/13 vs 13/13). It is kept only as the ablation / OOD-research head.

### Add OOD negatives + retrain

Negatives live under `data/healthy_images/`: **top-level loose files = healthy fish;
any sub-folder = OOD (non-fish)**. The 3-tier sampler (disease / healthy / OOD) gives
each ⅓ of every batch, so a few OOD images drive the gradient. OOD generalization
needs **variety, not count** (each image is repeated ~hundreds×/epoch — memorizes
otherwise).

```bash
# 1. drop the new negatives into a named sub-folder
mkdir -p data/healthy_images/sashimi && cp /path/to/sashimi_*.png data/healthy_images/sashimi/

# 2. retrain (features recompute by default; pass --use_cache to reuse the cache)
$PY -m diagnosis_model.grod.train_abstain_head        # --feat dino_neck is the default
```

It overwrites `models/disease_head/neck_disease_head.pt` (the ablation / OOD-research
head; not loaded by the demo). The run prints val AUROC + per-tier reject + an `[OOD sanity]` blocked count
(incl-train, optimistic) — watch the **held-out `val reject ood`** and **disease recall**
for the real signal.
