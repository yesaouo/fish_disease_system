from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from .. import dependencies
from ..config import Settings
from ..models import (
    AnnotatedItem,
    AnnotatedListResponse,
    ClassesResponse,
    CommentedItem,
    CommentedListResponse,
    DatasetListResponse,
    EvidenceOptionsZhResponse,
    ImageListResponse,
    LabelMapZhResponse,
)
from ..services import datasets as datasets_service
from ..services import storage as storage_service

router = APIRouter(prefix="/api", tags=["datasets"])


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets(
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> DatasetListResponse:
    datasets = datasets_service.list_datasets(settings)
    return DatasetListResponse(datasets=datasets)


@router.get("/datasets/{dataset}/classes", response_model=ClassesResponse)
def get_classes(
    dataset: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> ClassesResponse:
    classes = datasets_service.load_classes(dataset, settings)
    return ClassesResponse(classes=classes)


@router.get("/datasets/{dataset}/images/{filename}")
def get_image(
    dataset: str,
    filename: str,
    settings: Settings = Depends(dependencies.get_app_settings),
) -> FileResponse:
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    image_dir = storage_service.get_image_dir(dataset_dir, is_healthy=False)
    file_path = (image_dir / filename).resolve()
    if ".." in Path(filename).parts:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    if not str(file_path).startswith(str(image_dir.resolve())):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    if file_path.suffix.lower() not in settings.image_extensions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    if not file_path.exists():
        storage_service.append_audit_log(
            {
                "who": "system",
                "when": datetime.now(timezone.utc).isoformat(),
                "dataset": dataset,
                "task_id": filename,
                "action": "missing_image",
            },
            settings,
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    return FileResponse(file_path)


@router.get("/datasets/{dataset}/healthy_images/{filename}")
def get_healthy_image(
    dataset: str,
    filename: str,
    settings: Settings = Depends(dependencies.get_app_settings),
) -> FileResponse:
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    image_dir = storage_service.get_image_dir(dataset_dir, is_healthy=True)
    file_path = (image_dir / filename).resolve()
    if ".." in Path(filename).parts:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    if not str(file_path).startswith(str(image_dir.resolve())):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    if file_path.suffix.lower() not in settings.image_extensions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    if not file_path.exists():
        storage_service.append_audit_log(
            {
                "who": "system",
                "when": datetime.now(timezone.utc).isoformat(),
                "dataset": dataset,
                "task_id": filename,
                "action": "missing_image",
            },
            settings,
        )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    return FileResponse(file_path)

@router.get("/datasets/{dataset}/healthy_images", response_model=ImageListResponse)
def list_healthy_images(
    dataset: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> ImageListResponse:
    images = storage_service.list_healthy_images(dataset, settings)
    return ImageListResponse(images=images)


@router.post("/datasets/{dataset}/images/{filename}/move_to_healthy_images")
def move_image_to_healthy_images(
    dataset: str,
    filename: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> dict[str, bool]:
    storage_service.move_image_to_healthy_images(dataset, filename, settings)
    storage_service.append_audit_log(
        {
            "who": "system",
            "when": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "task_id": filename,
            "action": "move_to_healthy_images",
        },
        settings,
    )
    return {"ok": True}

@router.post("/datasets/{dataset}/healthy_images/{filename}/move_to_images")
def move_image_to_images(
    dataset: str,
    filename: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> dict[str, bool]:
    storage_service.move_image_to_images(dataset, filename, settings)
    storage_service.append_audit_log(
        {
            "who": "system",
            "when": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "task_id": filename,
            "action": "move_to_images",
        },
        settings,
    )
    return {"ok": True}


@router.get("/datasets/{dataset}/labels_zh", response_model=LabelMapZhResponse)
def get_label_map_zh(
    dataset: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> LabelMapZhResponse:
    mapping = datasets_service.load_label_map_zh(dataset, settings)
    return LabelMapZhResponse(label_map_zh=mapping)


@router.get("/datasets/{dataset}/evidence_options_zh", response_model=EvidenceOptionsZhResponse)
def get_evidence_options_zh(
    dataset: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> EvidenceOptionsZhResponse:
    mapping = datasets_service.load_evidence_options_zh(dataset, settings)
    return EvidenceOptionsZhResponse(evidence_options_zh=mapping)




@router.get("/datasets/{dataset}/annotated", response_model=AnnotatedListResponse)
def list_annotated(
    dataset: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> AnnotatedListResponse:
    # Build list of annotated items (submitted by any role), sorted by last_modified_at descending
    items: list[AnnotatedItem] = []

    image_filenames = storage_service.list_images(dataset, settings)
    index_map = {Path(fn).stem: i + 1 for i, fn in enumerate(image_filenames)}

    for row in storage_service.list_task_rows(dataset, settings):
        task_id = str(row["task_id"])
        if task_id not in index_map:
            continue
        try:
            general = json.loads(row["general_editors_json"] or "[]")
            expert = json.loads(row["expert_editors_json"] or "[]")
            if not isinstance(general, list):
                general = []
            if not isinstance(expert, list):
                expert = []
        except Exception:
            general, expert = [], []

        if general or expert:
            items.append(
                AnnotatedItem(
                    dataset=dataset,
                    index=int(index_map[task_id]),
                    task_id=task_id,
                    image_filename=str(image_filenames[index_map[task_id] - 1]),
                    last_modified_at=str(row["last_modified_at"]),
                    general_editor=[str(x).strip() for x in general if str(x).strip()],
                    expert_editor=[str(x).strip() for x in expert if str(x).strip()],
                )
            )
    items.sort(key=lambda x: x.last_modified_at, reverse=True)
    return AnnotatedListResponse(items=items)


@router.get("/datasets/{dataset}/commented", response_model=CommentedListResponse)
def list_commented(
    dataset: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> CommentedListResponse:
    # Return tasks that have ANY comments, sorted by last_modified_at descending
    items: list[CommentedItem] = []

    image_filenames = storage_service.list_images(dataset, settings)
    index_map = {Path(fn).stem: i + 1 for i, fn in enumerate(image_filenames)}

    for row in storage_service.list_task_rows(dataset, settings):
        task_id = str(row["task_id"])
        if task_id not in index_map:
            continue
        comments_count = int(row["comments_count"] or 0)
        if comments_count > 0:
            items.append(
                CommentedItem(
                    dataset=dataset,
                    index=int(index_map[task_id]),
                    task_id=task_id,
                    image_filename=str(image_filenames[index_map[task_id] - 1]),
                    last_modified_at=str(row["last_modified_at"]),
                    comments_count=comments_count,
                )
            )

    # 最近有動到（包含留言/修改）的排最上面
    items.sort(key=lambda x: x.last_modified_at, reverse=True)

    return CommentedListResponse(items=items)
