from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile

from ..config import Settings, get_settings
from ..dependencies import require_api_key

router = APIRouter(prefix="/api", tags=["diagnosis"])


@router.post("/diagnose")
async def diagnose(
    image: UploadFile = File(...),
    text: str = Form(""),
    mode: str = Form("grod_soft"),
    abstain_thresh: float | None = Form(None),
    display_thresh: float | None = Form(None),
    top_k_cases: int = Form(20),
    top_n_causes: int = Form(5),
    _token: str = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
) -> Response:
    """Proxy a diagnosis request to the GROD inference service (SDM env, GPU).

    Forwards the uploaded image + form fields unchanged and streams the JSON
    report back. Keeps the GPU service single-origin and behind this app's auth.
    """
    content = await image.read()
    files = {"image": (image.filename or "upload.jpg", content,
                       image.content_type or "application/octet-stream")}
    data = {"text": text, "mode": mode,
            "top_k_cases": str(top_k_cases), "top_n_causes": str(top_n_causes)}
    # Only forward thresholds when explicitly provided, so the service falls back
    # to its dataset-calibrated defaults otherwise.
    if abstain_thresh is not None:
        data["abstain_thresh"] = str(abstain_thresh)
    if display_thresh is not None:
        data["display_thresh"] = str(display_thresh)

    url = settings.inference_url.rstrip("/") + "/diagnose"
    try:
        async with httpx.AsyncClient(timeout=settings.inference_timeout_seconds) as client:
            resp = await client.post(url, files=files, data=data)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"inference service unreachable: {e}")
    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"))
