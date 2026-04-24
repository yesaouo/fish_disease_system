from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .cause_texts import collect_cause_strings_from_coco


EncoderBackend = Literal["vlm", "hf", "sentence-transformers"]


def _autocast_device_type(device: str) -> str:
    if device.startswith("cuda"):
        return "cuda"
    if device.startswith("mps"):
        return "mps"
    return "cpu"


def encode_strings_with_vlm(
    texts: Sequence[str],
    vlm_path: str,
    device: str = "cuda",
    text_batch_size: int = 256,
    max_length: int = 64,
    use_amp: bool = True,
    text_template: str = "{cap}。",
) -> torch.Tensor:
    """Encode strings with the trained VLM text encoder."""
    from diagnosis_model.vl_classifier.eval import (  # type: ignore
        encode_text_features,
        load_model_and_processor,
    )

    print(f"[encode:vlm] loading VLM from {vlm_path}")
    model, processor = load_model_and_processor(
        vlm_path,
        device=device,
        force_fusion=True,
    )

    wrapped = [text_template.format(cap=text) for text in texts]
    print(f"[encode:vlm] encoding {len(wrapped)} strings (batch={text_batch_size})")
    features = encode_text_features(
        model,
        processor,
        wrapped,
        device=device,
        text_batch_size=text_batch_size,
        max_length=max_length,
        use_amp=use_amp,
    )
    return F.normalize(features.float().cpu(), dim=-1)


@torch.no_grad()
def encode_strings_with_hf(
    texts: Sequence[str],
    model_name_or_path: str,
    device: str = "cuda",
    text_batch_size: int = 256,
    max_length: int = 128,
    use_amp: bool = True,
    text_template: str = "{cap}",
    pooling: str = "mean",
    trust_remote_code: bool = False,
) -> torch.Tensor:
    """Encode strings with a generic Hugging Face AutoModel."""
    from transformers import AutoModel, AutoTokenizer

    print(f"[encode:hf] loading model from {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()

    wrapped = [text_template.format(cap=text) for text in texts]
    features: List[torch.Tensor] = []
    device_type = _autocast_device_type(device)
    enable_amp = bool(use_amp and device_type == "cuda")
    print(f"[encode:hf] encoding {len(wrapped)} strings (batch={text_batch_size}, pooling={pooling})")

    for start in range(0, len(wrapped), text_batch_size):
        batch = wrapped[start:start + text_batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.autocast(device_type=device_type, enabled=enable_amp):
            outputs = model(**inputs)
            if pooling == "pooler" and getattr(outputs, "pooler_output", None) is not None:
                batch_features = outputs.pooler_output
            else:
                last_hidden = outputs.last_hidden_state
                if pooling == "cls":
                    batch_features = last_hidden[:, 0]
                elif pooling == "mean":
                    mask = inputs["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
                    summed = (last_hidden * mask).sum(dim=1)
                    denom = mask.sum(dim=1).clamp_min(1e-12)
                    batch_features = summed / denom
                else:
                    raise ValueError(f"Unsupported HF pooling: {pooling}")

        features.append(F.normalize(batch_features.float().cpu(), dim=-1))

    return torch.cat(features, dim=0)


def encode_strings_with_sentence_transformers(
    texts: Sequence[str],
    model_name_or_path: str,
    device: str = "cuda",
    text_batch_size: int = 256,
    text_template: str = "{cap}",
) -> torch.Tensor:
    """Encode strings with sentence-transformers when that package is installed."""
    from sentence_transformers import SentenceTransformer

    wrapped = [text_template.format(cap=text) for text in texts]
    print(f"[encode:sentence-transformers] loading model from {model_name_or_path}")
    model = SentenceTransformer(model_name_or_path, device=device)
    features = model.encode(
        wrapped,
        batch_size=text_batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return features.float().cpu()


def encode_strings(
    texts: Sequence[str],
    backend: EncoderBackend,
    model_name_or_path: str,
    device: str = "cuda",
    text_batch_size: int = 256,
    max_length: int = 64,
    use_amp: bool = True,
    text_template: str = "{cap}。",
    hf_pooling: str = "mean",
    trust_remote_code: bool = False,
) -> torch.Tensor:
    if backend == "vlm":
        return encode_strings_with_vlm(
            texts,
            vlm_path=model_name_or_path,
            device=device,
            text_batch_size=text_batch_size,
            max_length=max_length,
            use_amp=use_amp,
            text_template=text_template,
        )
    if backend == "hf":
        return encode_strings_with_hf(
            texts,
            model_name_or_path=model_name_or_path,
            device=device,
            text_batch_size=text_batch_size,
            max_length=max_length,
            use_amp=use_amp,
            text_template=text_template,
            pooling=hf_pooling,
            trust_remote_code=trust_remote_code,
        )
    if backend == "sentence-transformers":
        return encode_strings_with_sentence_transformers(
            texts,
            model_name_or_path=model_name_or_path,
            device=device,
            text_batch_size=text_batch_size,
            text_template=text_template,
        )
    raise ValueError(f"Unsupported encoder backend: {backend}")


def save_text_embedding_cache(
    cache_dir: str | Path,
    texts: Sequence[str],
    embeddings: torch.Tensor,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    embeddings = F.normalize(embeddings.float().cpu(), dim=-1)

    torch.save(embeddings, cache / "embeddings.pt")
    with (cache / "texts.json").open("w", encoding="utf-8") as f:
        json.dump(list(texts), f, ensure_ascii=False)

    meta = dict(metadata or {})
    meta.update(
        {
            "n_texts": len(texts),
            "dim": int(embeddings.size(1)) if embeddings.ndim == 2 else None,
            "normalized": True,
        }
    )
    with (cache / "cache_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_text_embedding_cache(cache_dir: str | Path) -> Tuple[List[str], torch.Tensor, Dict[str, Any]]:
    cache = Path(cache_dir)
    with (cache / "texts.json").open("r", encoding="utf-8") as f:
        texts = json.load(f)

    try:
        embeddings = torch.load(cache / "embeddings.pt", map_location="cpu", weights_only=True)
    except TypeError:
        embeddings = torch.load(cache / "embeddings.pt", map_location="cpu")

    meta_path = cache / "cache_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    if len(texts) != int(embeddings.size(0)):
        raise ValueError(
            f"texts ({len(texts)}) and embeddings ({embeddings.size(0)}) are out of sync"
        )

    return texts, F.normalize(embeddings.float().cpu(), dim=-1), metadata


def build_text_embedding_cache(
    coco_files: Sequence[str | Path],
    cache_dir: str | Path,
    backend: EncoderBackend,
    model_name_or_path: str,
    device: str = "cuda",
    text_batch_size: int = 256,
    max_length: int = 64,
    use_amp: bool = True,
    text_template: str = "{cap}。",
    include_healthy: bool = False,
    hf_pooling: str = "mean",
    trust_remote_code: bool = False,
) -> Tuple[List[str], torch.Tensor]:
    texts = collect_cause_strings_from_coco(
        coco_files,
        skip_healthy=not include_healthy,
    )
    if not texts:
        raise RuntimeError("No cause strings found. Check --coco_files / --include_healthy.")

    embeddings = encode_strings(
        texts,
        backend=backend,
        model_name_or_path=model_name_or_path,
        device=device,
        text_batch_size=text_batch_size,
        max_length=max_length,
        use_amp=use_amp,
        text_template=text_template,
        hf_pooling=hf_pooling,
        trust_remote_code=trust_remote_code,
    )

    save_text_embedding_cache(
        cache_dir,
        texts,
        embeddings,
        metadata={
            "encoder_backend": backend,
            "model_name_or_path": model_name_or_path,
            "max_length": max_length,
            "include_healthy": bool(include_healthy),
            "text_template": text_template,
            "coco_files": [str(p) for p in coco_files],
            "hf_pooling": hf_pooling if backend == "hf" else None,
            "trust_remote_code": bool(trust_remote_code),
        },
    )
    print(f"[cache] wrote {len(texts)} embeddings of dim {embeddings.size(1)} to {cache_dir}")
    return texts, embeddings

