from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from fastapi import HTTPException, status

from ..config import Settings, get_settings
from ..models import NextTaskResponse, TaskDocument
from ..services import datasets as datasets_service
from ..services import storage as storage_service

_settings = get_settings()

HEALTHY_LABEL = storage_service.HEALTHY_LABEL


def _ensure_settings(settings: Settings | None) -> Settings:
    return settings or _settings


def _now_iso() -> datetime:
    return datetime.now(timezone.utc)


def _is_blank(value: Any) -> bool:
    return not str(value or "").strip()


def _is_healthy_from_raw(raw: dict[str, Any]) -> bool:
    dets = raw.get("detections", [])
    if not isinstance(dets, list) or len(dets) == 0:
        return True
    for d in dets:
        if not isinstance(d, dict):
            return False
        if str(d.get("label") or "").strip() != HEALTHY_LABEL:
            return False
    return True


def _required_fields_ok(raw: dict[str, Any]) -> bool:
    # Healthy: no required fields.
    if _is_healthy_from_raw(raw):
        return True

    overall = raw.get("overall") if isinstance(raw.get("overall"), dict) else {}
    if _is_blank(overall.get("colloquial_zh")):
        return False
    if _is_blank(overall.get("medical_zh")):
        return False

    dets = raw.get("detections")
    if not isinstance(dets, list) or len(dets) == 0:
        return False

    for d in dets:
        if not isinstance(d, dict):
            return False
        label = str(d.get("label") or "").strip()
        if _is_blank(label):
            return False
        if label != HEALTHY_LABEL and d.get("evidence_index") in (None, ""):
            return False

    causes = raw.get("global_causes_zh")
    if not isinstance(causes, list) or len(causes) == 0:
        return False

    treatments = raw.get("global_treatments_zh")
    if not isinstance(treatments, list) or len(treatments) == 0:
        return False

    return True


def _is_expert_complete(
    *,
    expert_editors: list[str],
    has_comments: bool,
    required_fields_ok: bool,
) -> bool:
    # Completion is not based on "has expert editor" alone:
    # expert must have submitted AND required fields are filled (or there are comments to justify omissions).
    return bool(expert_editors) and (required_fields_ok or has_comments)


def select_next_task(
    dataset: str,
    editor_name: str,
    is_expert: bool,
    settings: Settings | None = None,
) -> NextTaskResponse:
    settings = _ensure_settings(settings)
    datasets_service.resolve_dataset_path(dataset, settings)

    image_filenames = storage_service.list_images(dataset, settings)
    total = len(image_filenames)
    if total <= 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="沒有可用任務")

    # Map task_id (stem) -> (index in `/images`, filename)
    index_map: dict[str, tuple[int, str]] = {}
    ordered_task_ids: list[str] = []
    for i, fn in enumerate(image_filenames, start=1):
        task_id = Path(fn).stem
        ordered_task_ids.append(task_id)
        index_map[task_id] = (i, fn)

    rows = storage_service.list_task_rows(dataset, settings)
    row_by_task_id = {str(r["task_id"]): r for r in rows if str(r["task_id"]) in index_map}

    candidates: list[str] = []
    for task_id in ordered_task_ids:
        row = row_by_task_id.get(task_id)
        if row is None:
            candidates.append(task_id)
            continue
        try:
            has_comments = int(row["comments_count"] or 0) > 0
            general_editors = json.loads(row["general_editors_json"] or "[]")
            expert_editors = json.loads(row["expert_editors_json"] or "[]")
            if not isinstance(general_editors, list):
                general_editors = []
            if not isinstance(expert_editors, list):
                expert_editors = []
            general_editors = [str(x).strip() for x in general_editors if str(x).strip()]
            expert_editors = [str(x).strip() for x in expert_editors if str(x).strip()]

            if editor_name in set(general_editors) or editor_name in set(expert_editors):
                continue

            raw = json.loads(row["doc_json"])
            required_ok = _required_fields_ok(raw)
            expert_complete = _is_expert_complete(
                expert_editors=expert_editors,
                has_comments=has_comments,
                required_fields_ok=required_ok,
            )
            if expert_complete:
                continue
            candidates.append(task_id)
        except Exception:
            candidates.append(task_id)

    if not candidates:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="沒有可用任務")

    task_id = random.choice(candidates)
    index, image_filename = index_map[task_id]

    storage_service.ensure_task_for_image(dataset, task_id, image_filename, settings)
    document = storage_service.load_task(dataset, task_id, settings)
    image_url = f"/api/datasets/{dataset}/images/{image_filename}"

    return NextTaskResponse(
        task_id=task_id,
        task=document,
        image_url=image_url,
        index=index,
        total_tasks=total,
    )


def validate_detections(
    dataset: str,
    document: TaskDocument,
    classes: List[str],
    *,
    require_evidence_index: bool,
    settings: Settings | None = None,
) -> None:
    """Validate detection labels (and evidence index, if required) against dataset config.

    - Allowed labels come from `symptoms.json` (preferred) or legacy `classes.txt`.
    - Evidence options are derived from `symptoms.json` and validated by index.
    """
    settings = _ensure_settings(settings)
    allowed = set(classes)
    evidence_options = datasets_service.load_evidence_options_zh(dataset, settings)

    for det in document.detections:
        if det.label == HEALTHY_LABEL:
            # Healthy regions are accepted even if they are not present in dataset classes/options.
            continue
        if det.label not in allowed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"label {det.label} 不在症狀類別清單中",
            )

        if det.evidence_index is None:
            if require_evidence_index:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="外觀敘述未選擇（請從下拉選單選一項）",
                )
            continue

        options = evidence_options.get(det.label) or []
        if not options:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"label {det.label} 未提供外觀敘述選項（請檢查 symptoms.json:data）",
            )
        if det.evidence_index < 0 or det.evidence_index >= len(options):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"evidence_index {det.evidence_index} 超出範圍（0-{len(options) - 1}）",
            )


def submit_task(
    dataset: str,
    task_id: str,
    incoming: TaskDocument,
    editor_name: str,
    is_expert: bool,
    settings: Settings | None = None,
) -> None:
    settings = _ensure_settings(settings)
    current = storage_service.load_task(dataset, task_id, settings)

    classes = datasets_service.load_classes(dataset, settings)
    allow_incomplete = bool(getattr(incoming, "comments", []) or [])
    is_healthy_task = (len(getattr(incoming, "detections", []) or []) == 0) or all(
        (getattr(d, "label", "") or "").strip() == HEALTHY_LABEL for d in (getattr(incoming, "detections", []) or [])
    )
    validate_detections(
        dataset,
        incoming,
        classes,
        require_evidence_index=(not allow_incomplete) and (not is_healthy_task),
        settings=settings,
    )

    # Extra completion checks (skip when user left comments to justify omissions)
    if (not allow_incomplete) and (not is_healthy_task):
        if _is_blank(getattr(incoming.overall, "colloquial_zh", "")):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="口語描述未填寫")
        if _is_blank(getattr(incoming.overall, "medical_zh", "")):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="醫學描述未填寫")
        if not (getattr(incoming, "global_causes_zh", []) or []):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="病因未填寫")
        if not (getattr(incoming, "global_treatments_zh", []) or []):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="處置未填寫")
        if not (getattr(incoming, "detections", []) or []):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="未新增任何框選目標")

    now = _now_iso()

    updated = incoming.model_copy()
    updated.dataset = dataset
    updated.image_filename = current.image_filename
    updated.image_width = getattr(current, "image_width", updated.image_width)
    updated.image_height = getattr(current, "image_height", updated.image_height)
    updated.last_modified_at = now
    updated.is_healthy = is_healthy_task

    # Append editor per role (list)
    updated.general_editor = list(getattr(current, "general_editor", []) or [])
    updated.expert_editor = list(getattr(current, "expert_editor", []) or [])
    if is_expert:
        if editor_name not in updated.expert_editor:
            updated.expert_editor.append(editor_name)
    else:
        if editor_name not in updated.general_editor:
            updated.general_editor.append(editor_name)

    storage_service.upsert_task(
        dataset,
        task_id,
        sort_index=None,
        document=updated,
        updated_by=editor_name,
        action="submit",
        settings=settings,
    )
    storage_service.append_audit_log(
        {
            "who": editor_name,
            "when": now.isoformat(),
            "dataset": dataset,
            "task_id": task_id,
            "action": "submit",
            "is_expert": is_expert,
        },
        settings,
    )


def get_task_by_index(
    dataset: str,
    index: int,
    editor_name: str,
    is_expert: bool,
    settings: Settings | None = None,
) -> NextTaskResponse:
    settings = _ensure_settings(settings)
    if index <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="index must be >= 1")
    datasets_service.resolve_dataset_path(dataset, settings)
    image_filenames = storage_service.list_images(dataset, settings)
    total = len(image_filenames)
    if total <= 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="沒有可用任務")
    if index > total:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="index out of range")

    image_filename = image_filenames[index - 1]
    task_id = Path(image_filename).stem

    storage_service.ensure_task_for_image(dataset, task_id, image_filename, settings)
    document = storage_service.load_task(dataset, task_id, settings)
    image_url = f"/api/datasets/{dataset}/images/{image_filename}"
    return NextTaskResponse(
        task_id=task_id,
        task=document,
        image_url=image_url,
        index=index,
        total_tasks=total,
    )

def get_healthy_task_by_index(
    dataset: str,
    index: int,
    editor_name: str,
    is_expert: bool,
    settings: Settings | None = None,
) -> NextTaskResponse:
    settings = _ensure_settings(settings)
    if index <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="index must be >= 1")
    datasets_service.resolve_dataset_path(dataset, settings)
    image_filenames = storage_service.list_healthy_images(dataset, settings)
    total = len(image_filenames)
    if total <= 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="瘝??舐隞餃?")
    if index > total:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="index out of range")

    image_filename = image_filenames[index - 1]
    task_id = Path(image_filename).stem

    storage_service.ensure_task_for_image(dataset, task_id, image_filename, settings, in_healthy_images=True)
    document = storage_service.load_task(dataset, task_id, settings)
    image_url = f"/api/datasets/{dataset}/healthy_images/{image_filename}"
    return NextTaskResponse(
        task_id=task_id,
        task=document,
        image_url=image_url,
        index=index,
        total_tasks=total,
    )


def skip_task(
    dataset: str,
    task_id: str,
    editor_name: str,
    is_expert: bool,
    settings: Settings | None = None,
) -> None:
    settings = _ensure_settings(settings)
    now = _now_iso()
    storage_service.append_audit_log(
        {
            "who": editor_name,
            "when": now.isoformat(),
            "dataset": dataset,
            "task_id": task_id,
            "action": "skip",
            "is_expert": is_expert,
        },
        settings,
    )


def save_task(
    dataset: str,
    task_id: str,
    incoming: TaskDocument,
    editor_name: str,
    is_expert: bool,
    settings: Settings | None = None,
) -> None:
    settings = _ensure_settings(settings)
    current = storage_service.load_task(dataset, task_id, settings)
    classes = datasets_service.load_classes(dataset, settings)
    validate_detections(dataset, incoming, classes, require_evidence_index=False, settings=settings)

    now = _now_iso()
    updated = incoming.model_copy()
    updated.dataset = dataset
    updated.image_filename = current.image_filename
    updated.image_width = getattr(current, "image_width", updated.image_width)
    updated.image_height = getattr(current, "image_height", updated.image_height)
    updated.last_modified_at = now
    updated.is_healthy = (len(getattr(updated, "detections", []) or []) == 0) or all(
        (getattr(d, "label", "") or "").strip() == HEALTHY_LABEL for d in (getattr(updated, "detections", []) or [])
    )
    # Do NOT set editors on save; keep completion state unchanged
    updated.general_editor = list(getattr(current, "general_editor", []) or [])
    updated.expert_editor = list(getattr(current, "expert_editor", []) or [])

    storage_service.upsert_task(
        dataset,
        task_id,
        sort_index=None,
        document=updated,
        updated_by=editor_name,
        action="save",
        settings=settings,
    )
    storage_service.append_audit_log(
        {
            "who": editor_name,
            "when": now.isoformat(),
            "dataset": dataset,
            "task_id": task_id,
            "action": "save",
            "is_expert": is_expert,
        },
        settings,
    )
