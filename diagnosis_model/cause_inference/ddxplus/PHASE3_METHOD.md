# Phase 3 Cross-Domain Adaptation — Paper Method Prose

> Draft Method subsection for the DDXPlus cross-domain validation of the FaCE-R
> Phase 3 case encoder. Companion to [paper_tables.md](../paper_tables.md) §G
> (numbers) and [README.md](README.md) §"Phase 3 training rationale" (CLI +
> v1/v2 ablation). This file holds the camera-ready prose only.
>
> **Metric semantics** (why G tables report pathology R@K + DDX NDCG instead of
> fish's semantic/cluster R@K) are documented in the [paper_tables.md §G
> opening note](../paper_tables.md). The cause-ranking *mechanism*
> (`build_candidate_pool → score_candidates → argsort`) is identical across
> fish and DDXPlus; only the evaluation metrics track the GT structure.

---

### 4.3 Cross-Domain Phase 3 Adaptation

Applying the Phase 3 case-encoder distillation pipeline (Section 4.2) from the
fish dataset (≈12.8K cases, long-tailed free-text causes, 94.7% singletons) to
DDXPlus (~1.03M patients downsampled to 2×10⁵ train, 49-class structured
pathology with multi-positive differentials) requires changes along two
orthogonal axes: *scale infrastructure* to keep training and validation
tractable at the 16× larger bank and 84× larger query set, and *training-signal
calibration* to prevent the dual-objective from saturating trivially on
DDXPlus's narrow class taxonomy. Crucially, neither set of changes propagates to
fish — the schema mismatch is exposed through a single CLI surface, while the
fish defaults remain optimal for fish.

#### 4.3.1 Scale Infrastructure

Four components of the fish pipeline assume a case bank of N ≤ 10⁴ and become
intractable at N = 2×10⁵.

**On-the-fly teacher.** Fish precomputes the full N×N case-similarity teacher
table off-line (311 MB fp16, ~18 min compute). At N = 2×10⁵ this would require
80 GB of storage and approximately 14 hours to compute. We introduce an
on-the-fly teacher mode that caches L2-normalized global and lesion stacks on
GPU in bf16 and recomputes only the B×B intra-batch block per training step via
vectorized scatter-reduce (B = 256). Numerical equivalence to the precomputed
table was verified against the reference scorer on a fp32 synthetic test
(|Δ| ≤ 6×10⁻⁸). Only symmetric max-mean and max-mean-normalized lesion matching
are supported; the Hungarian assignment used as fish's default has no batched
equivalent and is infeasible at this scale regardless.

**Query-batched validation.** Fish's retrieval-metric helper materializes the
full [N_v, N_t] similarity matrix and performs full argsort. At N_v = 1.3×10⁵
and N_t = 2×10⁵ this requires 104 GB of GPU memory. We replace it with a
query-chunked top-K kernel: for each chunk of 512 valid queries, the [512, N_t]
partial matrix is computed and only the top-max(K) indices are retained,
dropping peak memory to ~400 MB.

**Resident-memory dataset.** Naively loading 2×10⁵ cases inflates CPU RAM by
4–5× through three coexisting copies — the upcasted bf16→fp32 case dictionaries
(~8 GB), the on-the-fly teacher's transient `torch.cat` staging buffer (~16 GB),
and the normalized records list inside `CaseEncoderDataset` (~8 GB). We
(a) chunk the teacher initialization (`case_chunk_size = 16384`) so the staging
tensor never exceeds ~500 MB, (b) introduce a `free_source` mode on the dataset
that pops embedding fields from each source dictionary immediately after the
normalized copy is appended, and (c) accept a pre-built dataset in `encode_all`
so each evaluation epoch reuses the existing records list rather than
reallocating. Peak CPU RAM drops from ~36 GB to ~10 GB.

**Positional case identifier.** Fish's dataset uses the dictionary field
`case_id` as the row index into the precomputed teacher. Under full loading,
`case_id` coincides with the position in the loaded list. Under subsampling
these diverge: `case_id` is a random index in [0, 10⁶), while the position lies
in [0, 2×10⁵). We switch to a positional identifier (`enumerate` index), so the
on-the-fly teacher's batch-block lookup remains correct regardless of
subsampling.

#### 4.3.2 Training-Signal Calibration

The fish distillation objective combines a listwise-KL term against the
case-similarity teacher with an auxiliary case→cause-text InfoNCE over the dual
cause-text vocabulary. This combination is well-calibrated for fish's
long-tailed singleton-cause structure: each case carries one positive in the
cause-text vocabulary, batches rarely contain co-occurring causes, and the
row-softmax of the teacher is naturally peaky.

On DDXPlus this combination saturates trivially. With 49 conditions × ~4×10³
cases per condition, each batch of 256 contains on average five same-condition
cases whose teacher scores cluster tightly, flattening the softmax target.
Simultaneously, the expanded ground-truth field `cause_emb_indices` =
[pathology, DDX₁, …, DDX₅] yields six positives per anchor in the InfoNCE term,
biasing the encoder toward a six-prototype centroid rather than the single
pathology embedding. We observed first-epoch saturation of the proxy `sem R@10`
metric at 99.92%, with the downstream pathology-R@1 stalling at 28.12% —
markedly worse than the explicit Phase 1 scorer α·cos(global) +
β·max-mean(lesions) at 53.83% R@1 (Table G1).

We restore discriminative signal through three schema-aware changes, each
motivated by the structural mismatch:

1. **Temperature sharpening.** The teacher-target and student-prediction softmax
   temperatures are halved (0.10 → 0.05) to counteract the flattening induced by
   same-condition near-collinear scores. The InfoNCE temperature is tightened
   analogously (0.07 → 0.05).

2. **Strict-pathology InfoNCE.** We add an `--infonce_positives` flag that
   switches positive selection from the expanded `cause_emb_indices` to the
   single `pathology_emb_idx`. This restores the one-positive InfoNCE geometry
   that fish enjoys natively and aligns the encoder embedding with the strict
   pathology centroid rather than the pathology-DDX mixture. The flag raises a
   runtime error on case databases lacking the `pathology_emb_idx` field,
   preventing silent misuse.

3. **Asymmetry with fish.** Fish defaults (temperature 0.10, InfoNCE over
   `cause_emb_indices`) remain optimal: lowering the temperature on fish's
   already-peaky teacher induces overconfidence and degrades retrieval, and fish
   lacks the `pathology_emb_idx` field. The cross-domain interface is the
   schema-aware `--infonce_positives` flag rather than a domain-specific
   temperature schedule, isolating the structural difference (number of GT
   positives per anchor) from the algorithmic core.

These three changes recover 24.86 pp pathology R@1 (28.12% → 52.98%, Table G1)
and bring Phase 3 within 0.85 pp of the explicit Phase 1 scorer at approximately
100× the per-query speed (single-vector cosine retrieval versus
Hungarian/max-mean over the lesion stack). We interpret the v1→v2 ablation as
evidence that the encoder *architecture* has sufficient capacity for both
domains, and that the bottleneck is schema-dependent *training signal*, which we
expose through a single configuration interface.

---

### Summary table (fish vs DDXPlus settings)

| Axis | Component | Fish setting | DDXPlus setting | Why fish setting fails on DDXPlus |
|---|---|---|---|---|
| Scale | Teacher | precomputed N×N (311 MB) | on-the-fly B×B block | 2×10⁵² ≈ 80 GB, ~14 h — infeasible to store/compute |
| Scale | Lesion match | Hungarian or max-mean | max-mean only | Hungarian = per-pair scipy LP loop; no batched form, infeasible at scale |
| Scale | bank_dtype | fp32 | bf16 | 2×10⁵ bank fp32 ≈ 8 GB GPU; halves bank + sim-matrix footprint |
| Scale | max_train_cases | −1 (full = 12.8K) | 2×10⁵ | full 1.03M overflows 64 GB CPU during bf16→fp32 upcast |
| Scale | max_valid_cases | −1 (full = 1573) | 5000 (early-stop) / −1 (final, chunked) | [N_v, N_t] sim matrix = 104 GB |
| Scale | Retrieval eval | full argsort | query-batched top-K | same 104 GB matrix |
| Scale | Dataset memory | plain init | free_source + chunked teacher | three coexisting copies → ~36 GB CPU peak |
| Scale | case_id | dict field `case_id` | positional `enumerate` index | subsample → `case_id` ≠ list position → teacher mis-index |
| Signal | temp_target / temp_pred | 0.10 | 0.05 | same-condition scores collinear → flat softmax target |
| Signal | infonce_temp | 0.07 | 0.05 | same flattening in the auxiliary term |
| Signal | infonce_positives | `cause_emb_indices` (1 pos) | `pathology` (1 strict pos) | DDXPlus `cause_emb_indices` = 6 positives → centroid bias; fish has no `pathology_emb_idx` |

**One-line framing.** The Phase 3 training framework is domain-agnostic, but its
hyperparameters track the **ground-truth structure** of the target dataset —
fish has long-tailed singleton causes; DDXPlus has a 49-class structured
taxonomy with multi-positive differentials. The schema-aware InfoNCE-positive
selector (`pathology` vs `cause_emb_indices`) is the portable cross-domain
interface that isolates this structural difference from the algorithmic core.
