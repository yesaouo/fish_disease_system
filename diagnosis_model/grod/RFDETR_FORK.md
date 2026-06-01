# RF-DETR fork dependency (semantic head)

The joint detection+semantic training in this folder (`train_joint.py`,
`extract_z_joint.py`) does **not** run on the pip release of `rfdetr`. It needs a
patched fork that adds an optional semantic projection head + `loss_semantic`.

This file records that dependency so the environment can be rebuilt from scratch.

## Where it lives

- **Fork working tree:** `/mnt/ssd/YJ/rf-detr` (sibling of this repo, *not* tracked here)
- **Branch:** `feat/semantic-head`
- **Base:** official tag `1.6.5.post0` (commit `eb9566c`) — byte-identical to the
  pip release, so all non-semantic behavior is unchanged.
- **Head commit:** `8871328`
- **Upstream remote (`origin`):** https://github.com/roboflow/rf-detr.git (read-only)
- **Your backup remote (`fork`):** push `feat/semantic-head` to your own GitHub
  (e.g. `git remote add fork <your-repo>.git && git push fork feat/semantic-head`)

## How it's installed (editable)

The fork is installed editable into the `SDM` conda env, so edits to its source
take effect immediately with no reinstall:

```bash
cd /mnt/ssd/YJ/rf-detr
/home/lab603/anaconda3/envs/SDM/bin/python -m pip install -e . --no-deps --no-build-isolation
```

Verify it's active:
```bash
/home/lab603/anaconda3/envs/SDM/bin/python -c "import rfdetr; print(rfdetr.__file__)"
# -> /mnt/ssd/YJ/rf-detr/src/rfdetr/__init__.py
```

## Rebuild from scratch (new machine / fresh env)

```bash
git clone https://github.com/roboflow/rf-detr.git /mnt/ssd/YJ/rf-detr
cd /mnt/ssd/YJ/rf-detr
git remote add fork https://github.com/yesaouo/rf-detr-semantic.git
git fetch fork && git checkout feat/semantic-head    # or: git pull fork feat/semantic-head
/home/lab603/anaconda3/envs/SDM/bin/python -m pip install -e . --no-deps --no-build-isolation
```

## Revert to stock rfdetr (for upstream comparison)

```bash
pip install rfdetr==1.6.5.post0          # overrides the editable install
# to switch back to the fork: re-run the `pip install -e .` above
```

## What changed (4 files, +110 lines, all additive; semantic_dim=0 => stock behavior)

| file | change |
|---|---|
| `src/rfdetr/models/lwdetr.py` | `LWDETR.semantic_dim` arg + `semantic_embed=Linear(hidden_dim, semantic_dim)`; `forward` emits `out["pred_semantic"]=semantic_embed(hs[-1])` (last decoder layer only); `build_model` threads `semantic_dim`; `build_criterion_and_postprocessors` loads anchors -> appends `"semantic"` loss + `weight_dict["loss_semantic"]` |
| `src/rfdetr/models/criterion.py` | `loss_semantic` (matched query `pred_semantic` -> L2norm -> multi-positive sigmoid toward frozen text anchors; target `symptom_labels`, -1 ignored); registered in `get_loss` loss_map; **skipped on aux/enc layers** (last layer only); `SetCriterion` gains `semantic_text_anchors` buffer + `semantic_temp` |
| `src/rfdetr/datasets/coco.py` | `ConvertCoco` reads per-box `symptom_category_id` -> `target["symptom_labels"]` (auto-synced through transforms' per-instance field filtering) |
| `src/rfdetr/_namespace.py` | `build_namespace` injects `semantic_dim`/`semantic_anchors_path`/`semantic_loss_coef`/`semantic_temp` from env vars (`RFDETR_SEMANTIC_*`), keeping them off the `extra="forbid"` Pydantic configs |

## How the semantic head is toggled at runtime

Off by default. Enabled via env vars (set automatically by `train_joint.py` /
`extract_z_joint.py`):

```bash
export RFDETR_SEMANTIC_DIM=768
export RFDETR_SEMANTIC_ANCHORS=/abs/path/to/text_anchors.pt
export RFDETR_SEMANTIC_LOSS_COEF=2.0   # best operating point (see merge_narrative.md)
export RFDETR_SEMANTIC_TEMP=0.07
```

With `RFDETR_SEMANTIC_DIM` unset or 0, the model and criterion are exactly the
stock detector — safe for all other detection scripts in this repo.
