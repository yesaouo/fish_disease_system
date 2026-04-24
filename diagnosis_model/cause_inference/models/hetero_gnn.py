"""
hetero_gnn.py — Heterogeneous GNN for multi-lesion cause inference.

Design overview
---------------
A single query graph has three node types:
  - 'global' (1 node)   : VLM image embedding, optionally fused with a
                          user-provided text description (colloquial / medical)
  - 'lesion' (N nodes)  : VLM embeddings of detected lesion crops
  - 'cause'  (K nodes)  : top-K candidate cause descriptions from retrieval

ALL three live in the same encoder space (we use the trained SigLIP2+Fusion
VLM both for image-side features and for encoding cause descriptions). A
shared input projection (or per-type projection — controllable by a flag)
maps to the GNN hidden dim. We keep per-type projection by default because
even within one encoder, image and text features have different statistics.

Healthy samples are filtered upstream by the detector; the GNN only sees
images with >=1 lesion. There is NO NULL_CAUSE here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, GCNConv


# ---------------------------------------------------------------------------
# Per-node-type input projection
# ---------------------------------------------------------------------------

class NodeProjector(nn.Module):
    """Maps each node type's raw features into the shared hidden space.

    All node types share the same input dim (single VLM encoder), but we
    keep separate projection MLPs so the model can learn type-specific
    statistics (image-vs-text, global-vs-local).
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj_global = self._mlp(in_dim, hidden_dim, dropout)
        self.proj_lesion = self._mlp(in_dim, hidden_dim, dropout)
        self.proj_cause  = self._mlp(in_dim, hidden_dim, dropout)

    @staticmethod
    def _mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {
            'global': self.proj_global(x_dict['global']),
            'lesion': self.proj_lesion(x_dict['lesion']),
            'cause':  self.proj_cause (x_dict['cause']),
        }


# ---------------------------------------------------------------------------
# Global fusion: combine VLM image feature with optional text feature
# ---------------------------------------------------------------------------

class GlobalTextFusion(nn.Module):
    """Fuse the global image embedding with an optional user-provided text
    embedding (colloquial_zh or medical_zh, encoded by the VLM text encoder).

    Training-time augmentation: for each sample, the dataloader supplies
    either a real text embedding or a zero-tensor + text_mask = 0. A gated
    residual ensures that when text is absent the global feature is unchanged.

    Both inputs share the same encoder space, so we don't need an alignment
    Linear; we just fuse them with a small MLP.
    """

    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(in_dim * 2, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Parameter(torch.tensor(0.1))   # mirrors LocalGlobalFusionWrapper

    def forward(
        self,
        global_img: Tensor,    # [B, D]
        global_txt: Tensor,    # [B, D]   (zeros where absent)
        text_mask:  Tensor,    # [B]      1 = text present, 0 = absent
    ) -> Tensor:
        fused = self.fuse(torch.cat([global_img, global_txt], dim=-1))
        return global_img + self.gate * (text_mask.unsqueeze(-1) * fused)


# ---------------------------------------------------------------------------
# Edge types and per-relation conv factory
# ---------------------------------------------------------------------------

EDGE_TYPES: List[Tuple[str, str, str]] = [
    ('global', 'to',            'lesion'),
    ('lesion', 'to',            'global'),
    ('lesion', 'to',            'lesion'),
    ('lesion', 'supports',      'cause'),
    ('cause',  'supported_by',  'lesion'),
    ('global', 'supports',      'cause'),
    ('cause',  'supported_by',  'global'),
]


def _make_conv(gnn_type: str, in_dim: int, out_dim: int, heads: int = 4):
    if gnn_type in ('rgcn', 'sage'):
        return SAGEConv(in_dim, out_dim, aggr='mean')
    elif gnn_type == 'gat':
        return GATConv(in_dim, out_dim // heads, heads=heads, add_self_loops=False)
    elif gnn_type == 'gcn':
        return GCNConv(in_dim, out_dim, add_self_loops=False, normalize=False)
    else:
        raise ValueError(f"Unknown gnn_type: {gnn_type}")


class HeteroBlock(nn.Module):
    """One heterogeneous message passing layer with residual + LayerNorm."""

    def __init__(self, hidden_dim: int, gnn_type: str, dropout: float = 0.1):
        super().__init__()
        self.hetero_conv = HeteroConv(
            {et: _make_conv(gnn_type, hidden_dim, hidden_dim) for et in EDGE_TYPES},
            aggr='sum',
        )
        self.norm_global = nn.LayerNorm(hidden_dim)
        self.norm_lesion = nn.LayerNorm(hidden_dim)
        self.norm_cause  = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict:          Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        out = self.hetero_conv(x_dict, edge_index_dict)

        def _maybe(node_type: str, norm: nn.LayerNorm) -> Tensor:
            h = out[node_type] if node_type in out else x_dict[node_type]
            h = F.gelu(h)
            h = self.dropout(h)
            return norm(h + x_dict[node_type])   # residual

        return {
            'global': _maybe('global', self.norm_global),
            'lesion': _maybe('lesion', self.norm_lesion),
            'cause':  _maybe('cause',  self.norm_cause),
        }


# ---------------------------------------------------------------------------
# Ranking head
# ---------------------------------------------------------------------------

class RankingHead(nn.Module):
    """Scores each cause node.

    Two variants behind the same interface:
      - 'mlp'  : score = MLP(cause_h)
      - 'pair' : score = MLP([cause_h ; global_h_of_owning_graph])

    'pair' is the default — it lets the head re-use the global readout as
    a context anchor, which helps with calibration.
    """

    def __init__(self, hidden_dim: int, mode: str = 'pair', dropout: float = 0.1):
        super().__init__()
        assert mode in ('mlp', 'pair')
        self.mode = mode
        in_dim = hidden_dim if mode == 'mlp' else hidden_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        cause_h:     Tensor,   # [sumK, H]
        global_h:    Tensor,   # [B, H]
        cause_batch: Tensor,   # [sumK]  which graph each cause belongs to
    ) -> Tensor:
        if self.mode == 'mlp':
            return self.net(cause_h).squeeze(-1)
        ctx = global_h[cause_batch]              # [sumK, H]
        x = torch.cat([cause_h, ctx], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class CauseInferenceHGNN(nn.Module):
    """Heterogeneous GNN that takes a batched query graph and returns a per-
    cause-node score.

    Inputs (from the batched PyG HeteroData):
      x_dict:
        'global' : [B,    D]   VLM image feature for each graph
        'lesion' : [sumN, D]   VLM lesion features (sumN >= B since each graph
                               has at least one lesion)
        'cause'  : [sumK, D]   VLM-encoded cause descriptions

    Additional auxiliaries:
      global_txt : [B, D]   VLM-encoded optional colloquial / medical text
      text_mask  : [B]      1 if text present for this graph, else 0
      cause_batch: [sumK]   provided automatically by PyG batching
    """

    def __init__(
        self,
        in_dim: int,                       # = VLM hidden dim, e.g. 768
        hidden_dim: int = 256,
        num_layers: int = 2,
        gnn_type:   str = 'rgcn',
        ranker_mode: str = 'pair',
        dropout:    float = 0.1,
        use_global_text_fusion: bool = True,
        use_global_node: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.use_global_text_fusion = use_global_text_fusion
        self.use_global_node = use_global_node

        if use_global_text_fusion:
            self.global_fusion = GlobalTextFusion(in_dim, dropout=dropout)
        else:
            self.global_fusion = None

        self.proj = NodeProjector(in_dim, hidden_dim, dropout=dropout)
        self.blocks = nn.ModuleList([
            HeteroBlock(hidden_dim, gnn_type=gnn_type, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.head = RankingHead(hidden_dim, mode=ranker_mode, dropout=dropout)

    def forward(
        self,
        batched: HeteroData,
        global_txt: Tensor,       # [B, D]
        text_mask:  Tensor,       # [B]
    ) -> Dict[str, Tensor]:
        # 1) Optional global text fusion
        g = batched['global'].x                                   # [B, D]
        if self.use_global_text_fusion and self.global_fusion is not None:
            g = self.global_fusion(g, global_txt, text_mask)
        if not self.use_global_node:
            g = torch.zeros_like(g)

        x_dict = {
            'global': g,
            'lesion': batched['lesion'].x,
            'cause':  batched['cause'].x,
        }

        # 2) Input projection
        x_dict = self.proj(x_dict)

        # 3) Heterogeneous message passing
        edge_index_dict = {et: batched[et].edge_index for et in EDGE_TYPES}
        for blk in self.blocks:
            x_dict = blk(x_dict, edge_index_dict)

        # 4) Per-cause scores
        cause_batch = batched['cause'].batch
        scores = self.head(
            cause_h=x_dict['cause'],
            global_h=x_dict['global'],
            cause_batch=cause_batch,
        )
        return {
            'scores': scores,
            'cause_h': x_dict['cause'],
            'global_h': x_dict['global'],
            'lesion_h': x_dict['lesion'],
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from graph_builder import build_query_graph, build_target_for_batch
    from torch_geometric.loader import DataLoader as PyGDataLoader

    torch.manual_seed(0)
    D = 768
    H = 256
    M = 80

    model = CauseInferenceHGNN(
        in_dim=D, hidden_dim=H,
        num_layers=2, gnn_type='rgcn', ranker_mode='pair',
    )

    db_emb = F.normalize(torch.randn(M, D), dim=-1)
    db_ids = torch.arange(1, M + 1, dtype=torch.long)

    graphs = []
    for img_id, (n_lesion, gt) in enumerate([
        (3, [3, 17]),
        (5, [42, 9, 66]),
        (1, [22]),
    ]):
        g_emb = F.normalize(torch.randn(D), dim=-1)
        l_emb = F.normalize(torch.randn(n_lesion, D), dim=-1)
        data, _ = build_query_graph(
            image_id=img_id,
            global_emb=g_emb,
            lesion_embs=l_emb,
            gt_cause_ids=torch.tensor(gt, dtype=torch.long),
            cause_db_emb=db_emb,
            cause_db_ids=db_ids,
            top_k_global=10,
            top_k_lesion=5,
        )
        graphs.append(data)

    loader = PyGDataLoader(graphs, batch_size=3)
    for batch in loader:
        B = batch.num_cands.size(0)
        # half batch has text, half doesn't
        global_txt = torch.randn(B, D)
        text_mask  = torch.tensor([1.0, 0.0, 1.0])

        out = model(batch, global_txt=global_txt, text_mask=text_mask)
        print("scores shape:", out['scores'].shape)

        targets = build_target_for_batch(batch)
        loss = F.binary_cross_entropy_with_logits(out['scores'], targets)
        print("loss:", loss.item())

        loss.backward()
        # Verify gradients flow to all expected places
        for name, p in model.named_parameters():
            if p.grad is None:
                print(f"  WARNING: no grad for {name}")
        print("backward ok, gate value:", float(model.global_fusion.gate))
        break
