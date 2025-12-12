from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from .config import DIM

# Pillow 10+ 的重採樣列舉相容
try:
    _RESAMPLING_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    _RESAMPLING_BILINEAR = Image.BILINEAR

ImageLike = Union[Image.Image, str, Path]


class VisionBackbone:
    """DINOv3 ViT-B/16（facebook/dinov3-vitb16-pretrain-lvd1689m）做為全域 token 抽取器。
    回傳：np.ndarray(shape=(768,), dtype=float32)，已做 L2 normalize。
    """

    def __init__(
        self,
        dim: int = DIM,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        device: str | None = None,
    ) -> None:
        if dim != 768:
            raise ValueError(f"This backbone produces 768-dim features, got dim={dim}.")

        self.dim = dim
        self._norm_eps: float = 1e-8
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 透過 transformers 載入處理器與模型（預設會下載權重與 config）
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _to_image(self, image: ImageLike) -> Image.Image:
        """讀入並轉成 RGB PIL.Image，妥善管理檔案資源。"""
        if isinstance(image, (str, Path)):
            with Image.open(image) as im:
                return im.convert("RGB").copy()
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError("image must be a PIL.Image or path-like (str/Path)")

    @torch.inference_mode()
    def extract(self, image: ImageLike) -> np.ndarray:
        """回傳 global_token: (768,) float32，單位長度。"""
        im = self._to_image(image)

        # 若輸入邊長不是 16 的倍數，官方會裁切到最近的較小倍數；這裡直接交給 processor 處理
        #（模型卡也示範了相同路徑）
        inputs = self.processor(images=im, return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)
        # 優先使用 pooled_output（已對 CLS 做合適的投影/正規化）
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            feat = outputs.pooler_output.squeeze(0)  # (768,)
        else:
            # 後備：取最後隱層的 CLS token
            last = outputs.last_hidden_state  # (1, tokens, C)
            feat = last[:, 0].squeeze(0)  # (C,)；對 ViT-B/16 應為 768

        feat = feat.to(dtype=torch.float32)
        feat = F.normalize(feat, p=2, dim=0, eps=self._norm_eps)
        return feat.detach().cpu().numpy().astype(np.float32, copy=False)
