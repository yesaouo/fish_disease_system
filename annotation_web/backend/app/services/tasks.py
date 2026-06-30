from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

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


# Re-export storage's helpers so callers within this module stay terse and
# the completion-state logic lives in one place (it has to live in storage.py
# because the dispatch backfill and upsert path both compute it without a
# Pydantic model in hand).
_is_blank = storage_service.is_blank


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

    # Fetch only the columns dispatch needs — `doc_json` is intentionally NOT
    # selected, so we never pay for parsing the per-task payload here.
    # `expert_complete` is materialized at write time by upsert_task.
    rows = storage_service.list_dispatch_state(dataset, settings)
    skip: set[str] = set()
    for r in rows:
        task_id = str(r["task_id"])
        if task_id not in index_map:
            continue
        if int(r["expert_complete"] or 0) == 1:
            skip.add(task_id)
            continue
        # Editor lists are short (≤ a few names) — JSON-parsing them is cheap.
        try:
            general = json.loads(r["general_editors_json"] or "[]")
            expert = json.loads(r["expert_editors_json"] or "[]")
        except Exception:
            # Malformed editor JSON: be permissive (treat as dispatchable),
            # matching the pre-existing fallback behaviour.
            continue
        if not isinstance(general, list):
            general = []
        if not isinstance(expert, list):
            expert = []
        already_edited = editor_name in {str(x).strip() for x in general if str(x).strip()} \
            or editor_name in {str(x).strip() for x in expert if str(x).strip()}
        if already_edited:
            skip.add(task_id)

    candidates = [t for t in ordered_task_ids if t not in skip]
    if not candidates:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="沒有可用任務")

    # Resume from the editor's last submission: dispatch the first un-edited
    # task after it (in /images order), wrapping to the start once the editor
    # has reached the end. Falls back to the first candidate if the editor has
    # not submitted anything in this dataset yet. `candidates` preserves the
    # /images ordering, so a single forward scan suffices.
    last_submitted = storage_service.get_last_submitted_task_id(dataset, editor_name, settings)
    anchor_index = index_map[last_submitted][0] if last_submitted in index_map else 0
    task_id = next((t for t in candidates if index_map[t][0] > anchor_index), candidates[0])
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

    - Allowed labels come from dataset `symptoms.json`, or `DATA_ROOT/symptoms.json` fallback.
    - Evidence options are derived from the same `symptoms.json` source and validated by index.
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


def import_diagnosis_task(
    dataset: str,
    task_id: str,
    image_filename: str,
    image_width: int,
    image_height: int,
    incoming: TaskDocument,
    editor_name: str,
    settings: Settings | None = None,
) -> int:
    """Create a brand-new expert-submitted task from a diagnosis report.

    Inserts an empty row first (with the real image dims so submit_task preserves
    them), then runs the standard submit_task path for validation + expert
    bookkeeping. Returns the new version.
    """
    settings = _ensure_settings(settings)
    # Insert the placeholder row at the end of the dataset with correct dims.
    blank = TaskDocument(
        dataset=dataset,
        image_filename=image_filename,
        image_width=image_width,
        image_height=image_height,
    )
    storage_service.upsert_task(
        dataset,
        task_id,
        sort_index=storage_service.get_max_sort_index(dataset, settings) + 1,
        document=blank,
        updated_by="system",
        action="diagnosis_import_insert",
        settings=settings,
    )
    incoming = incoming.model_copy()
    incoming.version = 0
    return submit_task(
        dataset=dataset,
        task_id=task_id,
        incoming=incoming,
        editor_name=editor_name,
        is_expert=True,
        settings=settings,
    )


def submit_task(
    dataset: str,
    task_id: str,
    incoming: TaskDocument,
    editor_name: str,
    is_expert: bool,
    settings: Settings | None = None,
) -> int:
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
        # 病徵敘述擇一必填；處置建議改為選填。
        if _is_blank(getattr(incoming.overall, "colloquial_zh", "")) and _is_blank(
            getattr(incoming.overall, "medical_zh", "")
        ):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="請至少填寫通俗或醫學描述其中一項")
        if not (getattr(incoming, "global_causes_zh", []) or []):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="病因未填寫")
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

    new_version = storage_service.upsert_task(
        dataset,
        task_id,
        sort_index=None,
        document=updated,
        updated_by=editor_name,
        action="submit",
        expected_version=int(getattr(incoming, "version", 0) or 0),
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
    return new_version


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


def save_task(
    dataset: str,
    task_id: str,
    incoming: TaskDocument,
    editor_name: str,
    is_expert: bool,
    settings: Settings | None = None,
) -> int:
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

    new_version = storage_service.upsert_task(
        dataset,
        task_id,
        sort_index=None,
        document=updated,
        updated_by=editor_name,
        action="save",
        expected_version=int(getattr(incoming, "version", 0) or 0),
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
    return new_version
