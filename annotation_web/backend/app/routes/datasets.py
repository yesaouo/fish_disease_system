from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from .. import dependencies
from ..config import Settings
from ..models import ClassesResponse, DatasetListResponse, LabelMapZhResponse, AnnotatedListResponse, AnnotatedItem, CommentedListResponse, CommentedItem
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
    image_dir = storage_service.get_image_dir(dataset_dir)
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


@router.get("/datasets/{dataset}/labels_zh", response_model=LabelMapZhResponse)
def get_label_map_zh(
    dataset: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> LabelMapZhResponse:
    mapping = datasets_service.load_label_map_zh(dataset, settings)
    return LabelMapZhResponse(label_map_zh=mapping)




@router.get("/datasets/{dataset}/annotated", response_model=AnnotatedListResponse)
def list_annotated(
    dataset: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> AnnotatedListResponse:
    # Build list of annotated items (submitted by any role), sorted by last_modified_at descending
    items: list[AnnotatedItem] = []
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    files = list(storage_service.iter_task_files(dataset_dir))
    stems = [p.stem for p in files]
    index_map = {stem: i + 1 for i, stem in enumerate(stems)}
    for stem, doc in storage_service.load_all_tasks(dataset, settings):
        if getattr(doc, "general_editor", None) or getattr(doc, "expert_editor", None):
            items.append(
                AnnotatedItem(
                    dataset=dataset,
                    index=index_map.get(stem, 0),
                    task_id=stem,
                    image_filename=doc.image_filename,
                    last_modified_at=doc.last_modified_at,
                    general_editor=getattr(doc, "general_editor", None),
                    expert_editor=getattr(doc, "expert_editor", None),
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

    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    files = list(storage_service.iter_task_files(dataset_dir))
    stems = [p.stem for p in files]

    # 建 index_map 是為了對應前端用來 navigate(`/annotate/${item.index}`)
    index_map = {stem: i + 1 for i, stem in enumerate(stems)}

    for stem, doc in storage_service.load_all_tasks(dataset, settings):
        comments = getattr(doc, "comments", []) or []
        has_comments = bool(comments)

        if has_comments:
            items.append(
                CommentedItem(
                    dataset=dataset,
                    index=index_map.get(stem, 0),
                    task_id=stem,
                    image_filename=doc.image_filename,
                    last_modified_at=doc.last_modified_at,
                    comments_count=len(comments),
                )
            )

    # 最近有動到（包含留言/修改）的排最上面
    items.sort(key=lambda x: x.last_modified_at, reverse=True)

    return CommentedListResponse(items=items)
