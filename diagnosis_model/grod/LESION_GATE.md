# GROD lesion gate — design rationale & history

Single source of truth for how `gpu_infer.py` / `gpu_infer_soft.py` decide
**which queries are lesions** and **when to abstain (healthy)**. The inference
code itself is kept minimal — all the "why" lives here.

## Current production behaviour

A **fixed global threshold τ on objectness** `w_i = sigmoid(pred_logits[i,0])`:

- **hard** (`gpu_infer.py`): keep queries with `w_i > τ`; if none kept ⟹ abstain (healthy).
- **soft** (`gpu_infer_soft.py`): keep all queries with continuous weight `w`; abstain iff `max_i w_i < τ`.

`τ = DEFAULT_LESION_THRESH = 0.5`. No learned threshold, no disease head in the
inference path.

## How τ is derived — `compute_lesion_threshold.py`

Per-query GT labels (`extract_disease_perquery.py`: IoU-match GT boxes → queries)
→ sweep a global threshold → pick the **train micro selection-F1 optimum**.

- τ\* = **0.497 ≈ 0.5**; val selection F1 = 0.806 (P 0.813 / R 0.799).
- F1 is flat over τ∈[0.41, 0.61] — selection is insensitive to the exact value.
- Writes `models/disease_head/lesion_threshold.json` (the authoritative value).
- Scope: GROD objectness only. NOT the base RF-DETR detector threshold (different score).

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

## The disease head (kept as standalone code, NOT in inference)

`disease_head.py` (`ThresholdHead`: per-image τ(g) abstain) + `train_disease_head.py`
remain available for experiments / true OOD work — the head reads global `g`, so it
*might* reject non-fish inputs better than "no box clears τ" (untested; the joint
eval used healthy fish only). It is **not** loaded by `gpu_infer*.py`. To use it,
import and wire it explicitly in a fork.
