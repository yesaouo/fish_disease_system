"""
graph_builder.py — Dynamic heterogeneous graph construction for cause inference.

Given a single query (one image's global + lesion embeddings) and a cause vector
database, this module builds one PyG HeteroData per image. The graph schema:

  Nodes
    - 'global' : 1 node, the whole-image VLM embedding (optionally fused with
                 a user-provided text description; that fusion happens in the
                 model, not here)
    - 'lesion' : N nodes (N >= 1), one per detected lesion crop embedding
    - 'cause'  : K nodes, the candidate causes selected by retrieval

  Edges (all stored as bidirectional pairs for PyG message passing)
    - ('global', 'to',         'lesion')   /  reverse
    - ('lesion', 'to',         'lesion')   symmetric, fully connected
    - ('lesion', 'supports',   'cause')    /  reverse  -- only if cause is in
                                                          that lesion's top-K
    - ('global', 'supports',   'cause')    /  reverse  -- only if cause is in
                                                          global's top-K

The 'supports' edges encode the *retrieval-as-edge* mechanism: a candidate
cause node's connectivity reflects which queries (global / which lesions)
endorsed it during the coarse retrieval phase. The downstream GNN learns
to weight these endorsements.

Healthy-sample handling
-----------------------
The upstream RF-DETR detector is a 2-class detector (healthy_region vs
lesion_region). If it produces no lesion_region for an image, the system
declares the fish healthy and the GNN is never invoked. As a result:
  - GNN training data only contains images with >=1 lesion bbox where
    isHealthy=False.
  - There is NO NULL_CAUSE node, NO is_null_slot mask, NO is_healthy field.
  - gt_cause_ids must be non-empty; build_query_graph asserts this.

All embeddings (global / lesion / cause) come from the SAME SigLIP2+Fusion
VLM, so they live in the same space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData


# ---------------------------------------------------------------------------
# Public dataclass for what the graph carries beyond raw tensors
# ---------------------------------------------------------------------------

@dataclass
class GraphMeta:
    """Per-graph metadata we attach to HeteroData for the loss layer to use."""
    image_id: int
    cand_cause_ids: Tensor  # Long[K], the cause_id of each 'cause' node in order
    gt_cause_ids:   Tensor  # Long[G], ground-truth cause_ids for this image (G >= 1)


# ---------------------------------------------------------------------------
# Retrieval helpers (cosine top-K against a fixed cause DB)
# ---------------------------------------------------------------------------

def _topk_cosine(query: Tensor, db_emb: Tensor, k: int) -> Tensor:
    """Return indices into db_emb of the top-k most cosine-similar rows."""
    if query.dim() == 1:
        query = query.unsqueeze(0)
    q = F.normalize(query, dim=-1)
    d = F.normalize(db_emb, dim=-1)
    sim = q @ d.t()
    k = min(k, d.size(0))
    return sim.topk(k, dim=-1).indices


def _union_unique_preserving_order(idx_lists: List[Tensor]) -> Tensor:
    """Concat several Long-1D tensors, deduplicate, preserve first-seen order."""
    if not idx_lists:
        return torch.empty(0, dtype=torch.long)
    cat = torch.cat([x.flatten() for x in idx_lists]).tolist()
    seen, out = set(), []
    for v in cat:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return torch.tensor(out, dtype=torch.long)


# ---------------------------------------------------------------------------
# Edge-builder helpers
# ---------------------------------------------------------------------------

def _bipartite_edges_full(n_src: int, n_dst: int) -> Tensor:
    if n_src == 0 or n_dst == 0:
        return torch.empty(2, 0, dtype=torch.long)
    src = torch.arange(n_src).repeat_interleave(n_dst)
    dst = torch.arange(n_dst).repeat(n_src)
    return torch.stack([src, dst], dim=0)


def _undirected_clique_edges(n: int) -> Tensor:
    """All ordered pairs (i, j) with i != j (covers both directions)."""
    if n <= 1:
        return torch.empty(2, 0, dtype=torch.long)
    idx = torch.arange(n)
    src = idx.repeat_interleave(n)
    dst = idx.repeat(n)
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


def _retrieval_edges_lesion_to_cause(
    lesion_topk_db_idx: Tensor,
    db_idx_to_local_cause_idx: dict,
) -> Tensor:
    n_lesion, k = lesion_topk_db_idx.shape
    if n_lesion == 0 or k == 0:
        return torch.empty(2, 0, dtype=torch.long)
    src_list, dst_list = [], []
    for i in range(n_lesion):
        for j in range(k):
            db_i = int(lesion_topk_db_idx[i, j])
            local_c = db_idx_to_local_cause_idx.get(db_i, None)
            if local_c is None:
                continue
            src_list.append(i)
            dst_list.append(local_c)
    if not src_list:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.stack([
        torch.tensor(src_list, dtype=torch.long),
        torch.tensor(dst_list, dtype=torch.long),
    ], dim=0)


def _retrieval_edges_global_to_cause(
    global_topk_db_idx: Tensor,
    db_idx_to_local_cause_idx: dict,
) -> Tensor:
    src_list, dst_list = [], []
    for j in range(global_topk_db_idx.numel()):
        db_i = int(global_topk_db_idx[j])
        local_c = db_idx_to_local_cause_idx.get(db_i, None)
        if local_c is None:
            continue
        src_list.append(0)
        dst_list.append(local_c)
    if not src_list:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.stack([
        torch.tensor(src_list, dtype=torch.long),
        torch.tensor(dst_list, dtype=torch.long),
    ], dim=0)


def _reverse(edge_index: Tensor) -> Tensor:
    if edge_index.numel() == 0:
        return edge_index
    return edge_index.flip(0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_query_graph(
    image_id: int,
    global_emb: Tensor,            # [D]
    lesion_embs: Tensor,           # [N, D]   N >= 1
    gt_cause_ids: Tensor,          # Long[G]  G >= 1
    cause_db_emb: Tensor,          # [M, D]
    cause_db_ids: Tensor,          # Long[M]
    top_k_global: int = 30,
    top_k_lesion: int = 20,
    fully_connect_lesions: bool = True,
) -> Tuple[HeteroData, GraphMeta]:
    """Construct one HeteroData representing the query graph for an image.

    Embeddings should be L2-normalized. All inputs must be on the same device.
    """
    device = global_emb.device
    D = global_emb.size(0)
    n_lesion = lesion_embs.size(0)
    assert n_lesion >= 1, "GNN expects >=1 lesion; healthy images should be filtered upstream"
    assert lesion_embs.dim() == 2 and lesion_embs.size(1) == D, "lesion_embs shape mismatch"
    assert cause_db_emb.dim() == 2 and cause_db_emb.size(1) == D, \
        "cause_db_emb must share the encoder space (and dim) of global/lesion"
    assert gt_cause_ids.numel() >= 1, "gt_cause_ids must be non-empty"

    # 1) Coarse retrieval against the cause DB
    global_topk_db = _topk_cosine(global_emb, cause_db_emb, top_k_global)[0]   # [k_g]
    lesion_topk_db = _topk_cosine(lesion_embs, cause_db_emb, top_k_lesion)     # [N, k_l]

    cand_db_idx = _union_unique_preserving_order(
        [global_topk_db.cpu()] + [lesion_topk_db[i].cpu() for i in range(n_lesion)]
    ).to(device)
    n_cands = cand_db_idx.numel()
    assert n_cands >= 1, "retrieval returned 0 candidates; check top_k or DB size"

    # 2) Cause-node features
    cause_x = cause_db_emb.index_select(0, cand_db_idx)                # [K, D]
    cand_cause_ids = cause_db_ids.index_select(0, cand_db_idx)         # [K]

    db_idx_to_local_cause_idx = {
        int(cand_db_idx[i]): i for i in range(n_cands)
    }

    # 3) Edges
    g2l = _bipartite_edges_full(1, n_lesion).to(device)
    l2g = _reverse(g2l)

    if fully_connect_lesions:
        l2l = _undirected_clique_edges(n_lesion).to(device)
    else:
        l2l = torch.empty(2, 0, dtype=torch.long, device=device)

    l2c = _retrieval_edges_lesion_to_cause(
        lesion_topk_db.cpu(), db_idx_to_local_cause_idx,
    ).to(device)
    c2l = _reverse(l2c)

    g2c = _retrieval_edges_global_to_cause(
        global_topk_db.cpu(), db_idx_to_local_cause_idx,
    ).to(device)
    c2g = _reverse(g2c)

    # 4) Pack into HeteroData
    data = HeteroData()
    data['global'].x = global_emb.unsqueeze(0)            # [1, D]
    data['lesion'].x = lesion_embs                        # [N, D]
    data['cause' ].x = cause_x                            # [K, D]

    data['global', 'to', 'lesion'].edge_index = g2l
    data['lesion', 'to', 'global'].edge_index = l2g
    data['lesion', 'to', 'lesion'].edge_index = l2l
    data['lesion', 'supports', 'cause'].edge_index = l2c
    data['cause',  'supported_by', 'lesion'].edge_index = c2l
    data['global', 'supports', 'cause'].edge_index = g2c
    data['cause',  'supported_by', 'global'].edge_index = c2g

    data.cand_cause_ids = cand_cause_ids                  # [K]
    data.gt_cause_ids   = gt_cause_ids.to(device)         # [G]
    data.image_id       = torch.tensor([image_id], dtype=torch.long, device=device)
    data.num_cands      = torch.tensor([cand_cause_ids.size(0)], dtype=torch.long, device=device)
    data.num_gts        = torch.tensor([gt_cause_ids.size(0)],   dtype=torch.long, device=device)

    meta = GraphMeta(
        image_id=image_id,
        cand_cause_ids=cand_cause_ids.detach().cpu(),
        gt_cause_ids=gt_cause_ids.detach().cpu(),
    )
    return data, meta


# ---------------------------------------------------------------------------
# Convenience: build the multi-positive target tensor from a batched graph
# ---------------------------------------------------------------------------

def build_target_for_batch(batched_data) -> Tensor:
    """Build a Tensor[total_K] of {0, 1} targets aligned with the 'cause'
    nodes' order in the PyG-batched graph."""
    cand = batched_data.cand_cause_ids
    num_cands = batched_data.num_cands
    num_gts   = batched_data.num_gts
    gt        = batched_data.gt_cause_ids

    targets = torch.zeros(cand.size(0), dtype=torch.float, device=cand.device)

    cand_offset, gt_offset = 0, 0
    B = num_cands.size(0)
    for b in range(B):
        kt = int(num_cands[b])
        gg = int(num_gts[b])
        cand_slice = cand[cand_offset : cand_offset + kt]
        gt_slice   = gt  [gt_offset   : gt_offset   + gg]
        gt_set = set(gt_slice.tolist())
        for k in range(kt):
            if int(cand_slice[k]) in gt_set:
                targets[cand_offset + k] = 1.0
        cand_offset += kt
        gt_offset   += gg

    return targets


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    D = 32
    M = 100

    g_emb = F.normalize(torch.randn(D), dim=-1)
    l_emb = F.normalize(torch.randn(3, D), dim=-1)
    db    = F.normalize(torch.randn(M, D), dim=-1)
    db_ids = torch.arange(1, M + 1, dtype=torch.long)

    data, meta = build_query_graph(
        image_id=42,
        global_emb=g_emb,
        lesion_embs=l_emb,
        gt_cause_ids=torch.tensor([3, 17, 88], dtype=torch.long),
        cause_db_emb=db,
        cause_db_ids=db_ids,
        top_k_global=10,
        top_k_lesion=5,
    )
    print(data)
    print("meta:", meta)

    # 1-lesion edge case
    data1, _ = build_query_graph(
        image_id=99,
        global_emb=g_emb,
        lesion_embs=F.normalize(torch.randn(1, D), dim=-1),
        gt_cause_ids=torch.tensor([3], dtype=torch.long),
        cause_db_emb=db,
        cause_db_ids=db_ids,
        top_k_global=10,
        top_k_lesion=5,
    )
    print("\n1-lesion graph:")
    print(data1)

    from torch_geometric.loader import DataLoader as PyGDataLoader
    loader = PyGDataLoader([data, data1], batch_size=2)
    for batch in loader:
        print("\nBatched:", batch)
        targets = build_target_for_batch(batch)
        print("targets sum:", targets.sum().item(), "shape:", tuple(targets.shape))
        break
