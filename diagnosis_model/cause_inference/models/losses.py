"""
losses.py — Loss + evaluation metrics for cause inference.

Loss
----
Multi-label binary cross entropy over candidate cause nodes:
  - positives = candidates whose cause_id is in gt_cause_ids
  - negatives = the rest

Synonym-group dedup is baked in at the cause_id level (Phase 0 clustering
collapses paraphrases into one cause_id), so no extra dedup is needed here.

Metrics (all in cause_id space)
-------------------------------
- Recall@K
- MRR
- micro precision / recall / F1 at a fixed threshold

Healthy samples are filtered upstream by the detector and never reach the GNN,
so we no longer track NULL_CAUSE / health detection metrics here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from graph_builder import build_target_for_batch  # re-export


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def multi_label_bce(
    scores: Tensor,               # [sumK]
    targets: Tensor,              # [sumK] in {0., 1.}
    pos_weight: Optional[float] = None,
    label_smoothing: float = 0.0,
    reduction: str = 'mean',
) -> Tensor:
    """Per-candidate BCE with logits.

    pos_weight
        Multiplies the positive term. With ~4 GT positives per ~50 candidates,
        positives are ~8% of the population — a pos_weight ≈ 10 is a sane start.
    """
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

    if pos_weight is not None:
        pw = torch.tensor(float(pos_weight), device=scores.device, dtype=scores.dtype)
        return F.binary_cross_entropy_with_logits(
            scores, targets, pos_weight=pw, reduction=reduction,
        )
    return F.binary_cross_entropy_with_logits(scores, targets, reduction=reduction)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _iter_per_graph(batched, scores: Tensor):
    cand = batched.cand_cause_ids
    gt   = batched.gt_cause_ids
    num_cands = batched.num_cands.tolist()
    num_gts   = batched.num_gts.tolist()

    c_off, g_off = 0, 0
    for i, (kt, gg) in enumerate(zip(num_cands, num_gts)):
        yield (
            i,
            cand[c_off : c_off + kt].detach().cpu(),
            gt  [g_off : g_off + gg].detach().cpu(),
            scores[c_off : c_off + kt].detach().cpu(),
        )
        c_off += kt
        g_off += gg


def eval_metrics_for_batch(
    batched,
    scores: Tensor,
    ks: Tuple[int, ...] = (1, 3, 5, 10),
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute Recall@K (per-graph mean), MRR, and micro F1 at the threshold.

    Recall@K (set-based): fraction of GT cause_ids that appear in top-K
    predicted cause_ids for that image.
    """
    recall_at = {k: [] for k in ks}
    mrr_list: List[float] = []

    tp_counts, fp_counts, fn_counts = 0, 0, 0

    for _, cand_ids, gt_ids, sc in _iter_per_graph(batched, scores):
        gt_set = set(int(x) for x in gt_ids.tolist())
        probs = torch.sigmoid(sc)
        order = torch.argsort(sc, descending=True)
        ranked_cause_ids = [int(cand_ids[j]) for j in order.tolist()]

        for k in ks:
            topk = set(ranked_cause_ids[:k])
            r = len(topk & gt_set) / max(1, len(gt_set))
            recall_at[k].append(r)

        rank = 0
        for pos, cid in enumerate(ranked_cause_ids, start=1):
            if cid in gt_set:
                rank = pos
                break
        mrr_list.append(1.0 / rank if rank > 0 else 0.0)

        pred_set = {
            int(cand_ids[j]) for j in range(cand_ids.numel())
            if float(probs[j]) >= threshold
        }
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set  - pred_set)
        tp_counts += tp
        fp_counts += fp
        fn_counts += fn

    def _mean(xs): return float(sum(xs) / max(1, len(xs)))
    prec = tp_counts / max(1, tp_counts + fp_counts)
    rec  = tp_counts / max(1, tp_counts + fn_counts)
    f1   = 2 * prec * rec / max(1e-9, prec + rec)

    out = {f'recall@{k}': _mean(recall_at[k]) for k in ks}
    out['mrr'] = _mean(mrr_list)
    out['precision_micro'] = float(prec)
    out['recall_micro']    = float(rec)
    out['f1_micro']        = float(f1)
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from graph_builder import build_query_graph
    from torch_geometric.loader import DataLoader as PyGDataLoader

    torch.manual_seed(0)
    D = 32
    M = 60
    db_emb = F.normalize(torch.randn(M, D), dim=-1)
    db_ids = torch.arange(1, M + 1, dtype=torch.long)

    graphs = []
    for img_id, (n, gt) in enumerate([
        (3, [3, 17]),
        (4, [42, 9]),
        (1, [22]),
    ]):
        g = F.normalize(torch.randn(D), dim=-1)
        l = F.normalize(torch.randn(n, D), dim=-1)
        data, _ = build_query_graph(
            image_id=img_id, global_emb=g, lesion_embs=l,
            gt_cause_ids=torch.tensor(gt, dtype=torch.long),
            cause_db_emb=db_emb, cause_db_ids=db_ids,
            top_k_global=10, top_k_lesion=5,
        )
        graphs.append(data)

    loader = PyGDataLoader(graphs, batch_size=3)
    for batch in loader:
        targets = build_target_for_batch(batch)
        scores = torch.randn(targets.size(0), requires_grad=True)
        loss = multi_label_bce(scores, targets, pos_weight=10.0, label_smoothing=0.05)
        print("loss:", float(loss.detach()))

        metrics = eval_metrics_for_batch(batch, scores, ks=(1, 3, 5, 10))
        print("metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        loss.backward()
        print("grad ok")
        break
