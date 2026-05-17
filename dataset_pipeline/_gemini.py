"""Shared Gemini client used by process_coco_gemini.py and process_folder_gemini.py.

Centralizes the retry loop, request shape (system instruction, temperature,
1280px thumbnail), strict-JSON parser, payload validator, and the
cache-or-call-or-reuse-legacy lookup used by both entry-points.

This version is compatible with the current google-genai SDK style:
- uses google.genai.Client
- uses pydantic.BaseModel for response_schema
- keeps the original public function names/signatures so existing imports do not need changes
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, TypeVar

from dotenv import load_dotenv
from PIL import Image, ImageOps
from pydantic import BaseModel

T = TypeVar("T")

SYSTEM_INSTRUCTION = "你是專精觀賞魚疾病的視覺助理，請輸出嚴格 JSON。"
TEMPERATURE = 0.2
THUMBNAIL_SIZE = (1280, 1280)


class _OverallSchema(BaseModel):
    colloquial_zh: str
    medical_zh: str


class _GeminiResponseSchema(BaseModel):
    overall: _OverallSchema
    global_causes_zh: list[str]
    global_treatments_zh: list[str]


def _classify_error(e: Exception) -> str:
    """Return one of: 'rate_limit', 'fatal', 'transient'."""
    try:
        from google.genai import errors as genai_errors
    except ImportError:
        return "transient"

    if isinstance(e, genai_errors.APIError):
        code = getattr(e, "code", None)
        if code == 429:
            return "rate_limit"
        if code in (400, 401, 403, 404):
            return "fatal"
        return "transient"
    return "transient"


def _retry_delay_from_error(e: Exception) -> float | None:
    """Best-effort: extract retryDelay seconds from a Gemini 429 error body."""
    msg = str(e)
    m = re.search(r'retryDelay"?\s*:\s*"?(\d+(?:\.\d+)?)\s*s', msg)
    if m:
        return float(m.group(1))
    m = re.search(r'retry_delay\s*\{\s*seconds:\s*(\d+)', msg)
    if m:
        return float(m.group(1))
    return None


def _retry(func: Callable[[], T]) -> T:
    """Error-aware retry.

    429 honors retryDelay then 30s/60s/120s... capped at 300s (max 6 attempts).
    5xx / network / unknown use 1*2^i + jitter (max 4 attempts).
    400/401/403/404 fail fast. Each retry is logged to stderr.
    """
    max_rate = 6
    max_transient = 4
    rate_attempts = 0
    transient_attempts = 0

    while True:
        try:
            return func()
        except Exception as e:
            cls = _classify_error(e)
            if cls == "fatal":
                raise
            if cls == "rate_limit":
                rate_attempts += 1
                if rate_attempts > max_rate:
                    raise
                delay = _retry_delay_from_error(e) or min(30 * (2 ** (rate_attempts - 1)), 300)
                delay += random.random() * 2
            else:
                transient_attempts += 1
                if transient_attempts > max_transient:
                    raise
                delay = 1.0 * (2 ** (transient_attempts - 1)) + random.random() * 0.5
            print(
                f"[retry {cls}] {type(e).__name__}: {str(e)[:200]}; sleep {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)


def call_gemini(image_path: str, prompt: str, model: str) -> str:
    from google import genai
    from google.genai import types as genai_types

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("請設定 GEMINI_API_KEY 或 GOOGLE_API_KEY")

    client = genai.Client(api_key=api_key)

    with Image.open(image_path) as raw_im:
        im = ImageOps.exif_transpose(raw_im).copy()
    im.thumbnail(THUMBNAIL_SIZE)

    def _do_req() -> str:
        resp = client.models.generate_content(
            model=model,
            contents=[prompt, im],
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=TEMPERATURE,
                response_mime_type="application/json",
                response_schema=_GeminiResponseSchema,
            ),
        )
        out = getattr(resp, "text", None) or getattr(resp, "output_text", None)
        if not out:
            raise RuntimeError("模型沒有回傳文字內容")
        return out

    try:
        return _retry(_do_req)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def parse_json_strict(text: str) -> Dict[str, Any]:
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I | re.M)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        raise ValueError("回應缺少 JSON 物件：\n" + text)
    return json.loads(m.group(0))


def validate_gemini_payload(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Gemini 回傳不是 JSON 物件")

    overall = data.get("overall")
    if not isinstance(overall, dict):
        raise ValueError("缺少 overall")
    for k in ("colloquial_zh", "medical_zh"):
        v = overall.get(k)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"overall.{k} 需為非空字串")

    for key in ("global_causes_zh", "global_treatments_zh"):
        arr = data.get(key)
        if not (isinstance(arr, list) and all(isinstance(x, str) for x in arr)):
            raise ValueError(f"{key} 需為 List[str]")

    if not isinstance(data.get("generated_by"), str):
        raise ValueError("缺少 generated_by")

    return data


def _try_reuse_legacy(
    legacy_dir: Path,
    stem: str,
) -> Dict[str, Any] | None:
    """Return a validated payload from an old annotations JSON, or None."""
    legacy_path = legacy_dir / f"{stem}.json"
    if not legacy_path.is_file():
        return None
    try:
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        if not all(k in legacy for k in ("overall", "global_causes_zh", "global_treatments_zh")):
            return None
        data = {
            "overall": legacy["overall"],
            "global_causes_zh": legacy["global_causes_zh"],
            "global_treatments_zh": legacy["global_treatments_zh"],
            "generated_by": legacy.get("generated_by", "legacy_import"),
        }
        return validate_gemini_payload(data)
    except Exception as e:
        print(f"[WARN] 復用舊 annotations 失敗 ({legacy_path}): {e}", file=sys.stderr)
        return None


def fetch_or_generate_json(
    image_path: str,
    prompt: str,
    model: str,
    json_path: Path,
    cache_only: bool,
    overwrite_cache: bool,
    reuse_annotations_dir: Path | None = None,
) -> Dict[str, Any]:
    """Resolve a payload for an image, in priority order:

    1. Reuse from `reuse_annotations_dir` if it has overall/causes/treatments.
    2. If `overwrite_cache`, always re-call Gemini.
    3. Read the cache at `json_path` if present.
    4. If `cache_only`, raise rather than call Gemini.
    5. Call Gemini.
    """
    stem = Path(image_path).stem

    if reuse_annotations_dir is not None:
        reused = _try_reuse_legacy(reuse_annotations_dir, stem)
        if reused is not None:
            return reused

    if overwrite_cache:
        text = call_gemini(image_path, prompt, model=model)
        data = parse_json_strict(text)
        data.setdefault("generated_by", model)
        return validate_gemini_payload(data)

    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if "generated_by" not in data:
                data["generated_by"] = model
            core = {
                "overall": data.get("overall"),
                "global_causes_zh": data.get("global_causes_zh"),
                "global_treatments_zh": data.get("global_treatments_zh"),
                "generated_by": data.get("generated_by", model),
            }
            return validate_gemini_payload(core)
        except Exception as e:
            print(f"[WARN] 讀取快取失敗，將重新呼叫 Gemini: {json_path} ({e})", file=sys.stderr)
            if cache_only:
                raise

    if cache_only:
        raise FileNotFoundError(f"找不到快取 JSON：{json_path}")

    text = call_gemini(image_path, prompt, model=model)
    data = parse_json_strict(text)
    data.setdefault("generated_by", model)
    return validate_gemini_payload(data)
