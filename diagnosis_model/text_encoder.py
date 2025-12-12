from __future__ import annotations

from typing import Iterable, Optional
import numpy as np

from sentence_transformers import SentenceTransformer
from .config import DIM


class EmbeddingGemma:
    """
    Hugging Face: google/embeddinggemma-300m 的輕量包裝。

    - encode_document/encode_query：依用例使用適當提示模板與池化。
    - encode：維持與舊版相容，單筆文字 → 向量；mode 預設 'document'。
    - 支援 MRL 截斷（768/512/256/128）並重新單位化。
    - 依官方限制，請使用 float32 或 bfloat16。
    """

    _VALID_MRL = (768, 512, 256, 128)

    def __init__(
        self,
        dim: int = DIM,
        device: Optional[str] = None,          # 如 "cuda", "mps", "cpu"
        dtype: str = "float32",                # "float32" 或 "bfloat16"
    ) -> None:
        assert dim in self._VALID_MRL, f"EmbeddingGemma 只支援維度 {self._VALID_MRL}（MRL 截斷），收到 {dim}"
        if dtype not in ("float32", "bfloat16"):
            raise ValueError("EmbeddingGemma 不支援 float16，請改用 float32 或 bfloat16")

        # 直接從 Hub 載入官方 Sentence Transformers 介面
        # 其 encode_query/encode_document 會自動套用對應任務提示與池化策略
        # 參考模型頁「Usage」區段。 
        self.model = SentenceTransformer(
            "google/embeddinggemma-300m",
            device=device,
            model_kwargs={"dtype": dtype},
        )

        self.dim = dim
        self._full_dim = 768

    def _truncate_mrl(self, arr: np.ndarray) -> np.ndarray:
        """將 768d 向量截斷為較小 MRL 維度並重新單位化。"""
        if self.dim == self._full_dim:
            return arr
        arr = arr[..., : self.dim]
        norm = np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-8
        return arr / norm

    def encode_document(self, docs: Iterable[str]) -> np.ndarray:
        """用於建立索引／文件嵌入（retrieval_document）。"""
        arr = self.model.encode_document(
            list(docs),
            normalize_embeddings=True,   # 單位化
            convert_to_numpy=True,       # 直接回傳 np.ndarray
        )
        return self._truncate_mrl(arr)

    def encode_query(self, queries: Iterable[str]) -> np.ndarray:
        """用於查詢向量（retrieval_query）。"""
        arr = self.model.encode_query(
            list(queries),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return self._truncate_mrl(arr)

    # 與你先前 stub 相容：單筆文字 -> 向量；預設當作文件嵌入
    def encode(self, text: str, mode: str = "document") -> np.ndarray:
        if mode == "document":
            return self.encode_document([text])[0]
        if mode == "query":
            return self.encode_query([text])[0]
        raise ValueError("mode 必須是 'document' 或 'query'")
