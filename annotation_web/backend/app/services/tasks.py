from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Set
import random

from fastapi import HTTPException, status

from ..config import Settings, get_settings
from ..models import (
    NextTaskResponse,
    TaskDocument,
)
from ..services import datasets as datasets_service
from ..services import storage as storage_service

_settings = get_settings()


def _ensure_settings(settings: Settings | None) -> Settings:
    return settings or _settings


def _now_iso() -> datetime:
    return datetime.now(timezone.utc)


def _scan_task_metadata(dataset_dir: Path) -> List[Dict]:
    json_dir = storage_service.get_json_dir(dataset_dir)
    if not json_dir.exists():
        return []
    metadata: List[Dict] = []
    for path in json_dir.glob("*.json"):
        try:
            raw = storage_service.read_raw_json(path)
        except Exception:
            continue
        general_editor = str(raw.get("general_editor") or "").strip()
        expert_editor = str(raw.get("expert_editor") or "").strip()
        untouched = (not general_editor) and (not expert_editor)
        edited_by_general = bool(general_editor)
        edited_by_expert = bool(expert_editor)
        has_comments = bool(raw.get("comments") or [])
        metadata.append(
            {
                "stem": path.stem,
                "general_editor": general_editor,
                "expert_editor": expert_editor,
                "untouched": untouched,
                "edited_by_general": edited_by_general,
                "edited_by_expert": edited_by_expert,
                "has_comments": has_comments,
            }
        )
    return metadata


def _filter_candidates(
    metadata: List[Dict],
    current_user: str,
    is_expert: bool,
) -> List[Dict]:
    """
    Select candidate tasks for dispatch based on role.

    - 一般標註（is_expert=False）：
      不會拿到「專家已提交」的任務（有 expert_editor），其他都可以。
    - 專家標註（is_expert=True）：
      只要「尚未有專家標註」，就可派發（包含已由一般標註、或已經有留言的任務），
      如此可確保「專家沒標就不算標完」，且有註解的影像也會送給專家。
    - 任何角色都不會收到自己已經標註過的任務（避免自我複審）。
    """
    if is_expert:
        # Expert: any task without expert_editor, regardless of general_editor or comments,
        # as long as it wasn't edited by this user.
        return [
            m
            for m in metadata
            if (not m.get("edited_by_expert"))
            and current_user not in {m.get("general_editor"), m.get("expert_editor")}
        ]

    # General annotator: any task without expert_editor,
    # as long as it wasn't edited by this user (避免自己改自己).
    return [
        m
        for m in metadata
        if (not m.get("edited_by_expert"))
        and current_user not in {m.get("general_editor"), m.get("expert_editor")}
    ]


def select_next_task(
    dataset: str,
    editor_name: str,
    is_expert: bool,
    settings: Settings | None = None,
) -> NextTaskResponse:
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)

    metadata = _scan_task_metadata(dataset_dir)
    if not metadata:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="沒有可用任務")

    candidates = _filter_candidates(metadata, editor_name, is_expert)
    if not candidates:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="沒有可用任務")

    choice = random.choice(candidates)
    stem = choice["stem"]

    document = storage_service.load_task(dataset, stem, settings)

    image_url = f"/api/datasets/{dataset}/images/{document.image_filename}"

    # Compute index and total based on sorted task files
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    files = list(storage_service.iter_task_files(dataset_dir))
    stems = [p.stem for p in files]
    total = len(stems)
    try:
        index = stems.index(stem) + 1
    except ValueError:
        index = 0

    return NextTaskResponse(
        task_id=stem,
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
    validate_detections(dataset, incoming, classes, require_evidence_index=True, settings=settings)

    now = _now_iso()

    updated = incoming.model_copy()
    updated.dataset = dataset
    updated.image_filename = current.image_filename
    updated.image_width = getattr(current, "image_width", updated.image_width)
    updated.image_height = getattr(current, "image_height", updated.image_height)
    updated.last_modified_at = now

    # Set single editor per role
    if is_expert:
        updated.expert_editor = editor_name
        updated.general_editor = current.general_editor
    else:
        updated.general_editor = editor_name
        updated.expert_editor = current.expert_editor

    storage_service.save_task(updated, settings)
    # Snapshot versioned JSON for offline/long-term analysis.
    # Version files are named as `<image_stem>_vN.json` and are only
    # incremented on submit (not on intermediate saves).
    storage_service.save_task_version(updated, settings)
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
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    files = list(storage_service.iter_task_files(dataset_dir))
    if not files:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="沒有可用任務")
    total = len(files)
    if index > total:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="index out of range")
    target = files[index - 1]
    stem = target.stem
    document = storage_service.load_task(dataset, stem, settings)
    image_url = f"/api/datasets/{dataset}/images/{document.image_filename}"
    return NextTaskResponse(
        task_id=stem,
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
    # Do NOT set editors on save; keep completion state unchanged
    updated.general_editor = current.general_editor
    updated.expert_editor = current.expert_editor

    storage_service.save_task(updated, settings)
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
