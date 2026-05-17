"""CRR-DeepRVQ: Compression-Residual Reranking for DeepSets-RVQ retrieval.

Post-hoc compression of frozen DeepSets case embeddings via Residual Vector
Quantization, plus a neural reranker that learns the compression residual
Δ ≈ qᵀe_i to recover the dense ranking. See README.md for full design.
"""
