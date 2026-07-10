"""Retrieval-bank hot-update sync: mirror expert-submitted case changes into the
diagnosis serve service's in-memory case bank (HITL writeback loop).

The serve service (diagnosis_model/serve/app.py) holds the production retrieval bank
as frozen base ⊕ mutable delta. Whenever an expert submits / edits / deletes a task
in a writable (created_via:diagnosis) dataset, we push the change to serve so the new
case becomes retrievable immediately — no model retraining.

All calls are best-effort: serve being down or slow must never block or fail an
annotation write. Route handlers schedule these via FastAPI BackgroundTasks, so the
HTTP response returns before the sync runs; failures are only logged.
"""

from __future__ import annotations

import json
import logging

import httpx

from ..config import Settings
from ..services import datasets as datasets_service
from ..services import storage as storage_service

logger = logging.getLogger(__name__)

# Short connect timeout → fast fail when serve is down; longer read for the GPU
# forward on /bank/upsert.
_TIMEOUT = httpx.Timeout(connect=3.0, read=60.0, write=60.0, pool=3.0)


def _url(settings: Settings, path: str) -> str:
    return settings.inference_url.rstrip("/") + path


def sync_delete(dataset: str, task_id: str, settings: Settings) -> None:
    """Remove a case from the serve retrieval bank (idempotent; ok if absent)."""
    try:
        httpx.post(_url(settings, "/bank/delete"),
                   data={"source_dataset": dataset, "source_task_id": task_id},
                   timeout=_TIMEOUT)
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning("bank sync delete %s/%s failed: %s", dataset, task_id, e)


def sync_upsert(dataset: str, task_id: str, settings: Settings) -> None:
    """Add/replace a writable-dataset task in the serve retrieval bank.

    No-op for locked/official datasets. A healthy task (no detections) is removed
    from the bank instead — it is not a retrievable case."""
    if not datasets_service.is_writable(dataset, settings):
        return
    try:
        doc = storage_service.load_task(dataset, task_id, settings)
    except Exception as e:  # noqa: BLE001
        logger.warning("bank sync upsert: cannot load %s/%s: %s", dataset, task_id, e)
        return

    if len(doc.detections) == 0:
        sync_delete(dataset, task_id, settings)
        return

    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    image_path = storage_service.get_image_dir(dataset_dir, is_healthy=False) / doc.image_filename
    if not image_path.exists():
        logger.warning("bank sync upsert: image missing %s", image_path)
        return
    causes = [c for c in (doc.global_causes_zh or []) if str(c).strip()]
    try:
        with open(image_path, "rb") as f:
            httpx.post(
                _url(settings, "/bank/upsert"),
                data={"source_dataset": dataset, "source_task_id": task_id,
                      "image_path": str(image_path),
                      "causes_json": json.dumps(causes, ensure_ascii=False)},
                files={"image": (image_path.name, f, "application/octet-stream")},
                timeout=_TIMEOUT,
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("bank sync upsert %s/%s failed: %s", dataset, task_id, e)


def resync_all(settings: Settings) -> int:
    """Rebuild the serve delta bank from scratch: clear it, then re-push every
    non-healthy task from all writable datasets. Datasets are the single source of
    truth (used for serve-restart recovery). Returns the number of cases pushed."""
    try:
        httpx.post(_url(settings, "/bank/resync"), timeout=_TIMEOUT)
    except Exception as e:  # noqa: BLE001
        logger.warning("bank resync clear failed: %s", e)

    count = 0
    for d in datasets_service.list_datasets_with_meta(settings):
        if d.get("locked"):
            continue
        name = str(d["name"])
        for stem, doc in storage_service.load_all_tasks(name, settings):
            if len(doc.detections) == 0:
                continue
            sync_upsert(name, stem, settings)
            count += 1
    logger.info("bank resync pushed %d cases", count)
    return count
