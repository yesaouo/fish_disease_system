from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from .. import dependencies
from ..config import Settings
from ..models import (
    AnnotatedItem,
    AnnotatedListResponse,
    ClassesResponse,
    CommentedItem,
    CommentedListResponse,
    DatasetInfo,
    DatasetListResponse,
    EvidenceOptionsZhResponse,
    ImageListResponse,
    ImportTaskResponse,
    LabelMapZhResponse,
    TaskDocument,
    TaskLocatorResponse,
)
from ..services import datasets as datasets_service
from ..services import storage as storage_service
from ..services import tasks as tasks_service

router = APIRouter(prefix="/api", tags=["datasets"])


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets(
    settings: Settings = Depends(dependencies.get_app_settings),
    _role: str = Depends(dependencies.get_role),
) -> DatasetListResponse:
    datasets = [DatasetInfo(**d) for d in datasets_service.list_datasets_with_meta(settings)]
    return DatasetListResponse(datasets=datasets)


@router.get("/symptoms")
def get_global_symptoms(
    settings: Settings = Depends(dependencies.get_app_settings),
    _role: str = Depends(dependencies.get_role),
) -> dict:
    """Global symptoms (classes + zh labels + evidence options) for the diagnosis
    draft editor, whose target dataset may not exist yet (created on submit)."""
    return datasets_service.load_global_symptoms(settings)


@router.post("/datasets/{dataset}/tasks/import", response_model=ImportTaskResponse)
def import_diagnosis_task(
    dataset: str,
    image: UploadFile = File(...),
    doc_json: str = Form(...),
    editor_name: str = Form(...),
    settings: Settings = Depends(dependencies.get_app_settings),
    _role: str = Depends(dependencies.require_expert),
) -> ImportTaskResponse:
    """Persist an expert-edited diagnosis report as a new task. The target
    dataset is created on first submit (create-on-submit lifecycle): if it does
    not exist yet it is created here; locked/official datasets are rejected."""
    base = settings.data_root
    candidate = (base / dataset).resolve()
    exists = str(candidate).startswith(str(base.resolve())) and candidate.is_dir()
    if not exists:
        # First submit into a brand-new dataset → create it now.
        datasets_service.create_dataset(dataset, created_by=editor_name, settings=settings)
    elif not datasets_service.is_writable(dataset, settings):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="此資料集已鎖定，無法寫入")
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)

    try:
        incoming = TaskDocument.model_validate_json(doc_json)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"診斷資料格式錯誤: {exc}")
    # Dims come from the report (frontend sets image_width/height); no PIL needed.
    width, height = int(incoming.image_width), int(incoming.image_height)

    # Extension from the upload's filename / content-type (default jpg).
    ext = Path(image.filename or "").suffix.lower().lstrip(".")
    if f".{ext}" not in settings.image_extensions:
        ext = "jpg"

    # 無病灶（健康）→ 存到 healthy_images/（與既有健康影像同夾）；有病灶→ images/。
    is_healthy = len(incoming.detections) == 0
    content = image.file.read()
    stem = f"diag_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{secrets.token_hex(3)}"
    filename = f"{stem}.{ext}"
    target_dir = storage_service.get_image_dir(dataset_dir, is_healthy=is_healthy)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / filename).write_bytes(content)

    try:
        tasks_service.import_diagnosis_task(
            dataset=dataset,
            task_id=stem,
            image_filename=filename,
            image_width=width,
            image_height=height,
            incoming=incoming,
            editor_name=editor_name,
            settings=settings,
        )
    except HTTPException:
        (target_dir / filename).unlink(missing_ok=True)
        raise

    listing = (
        storage_service.list_healthy_images(dataset, settings)
        if is_healthy
        else storage_service.list_images(dataset, settings)
    )
    index = next((i + 1 for i, fn in enumerate(listing) if Path(fn).stem == stem), 1)
    return ImportTaskResponse(ok=True, task_id=stem, index=index, dataset=dataset, is_healthy=is_healthy)


@router.get("/datasets/{dataset}/classes", response_model=ClassesResponse)
def get_classes(
    dataset: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _role: str = Depends(dependencies.get_role),
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

@router.delete("/datasets/{dataset}/tasks/{task_id}")
def delete_dataset_task(
    dataset: str,
    task_id: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _role: str = Depends(dependencies.require_expert),
) -> dict[str, bool]:
    """Delete a task (and its image) from a diagnosis-created dataset. Locked
    (official) datasets cannot be modified."""
    datasets_service.resolve_dataset_path(dataset, settings)
    if not datasets_service.is_writable(dataset, settings):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="此資料集已鎖定，無法刪除")
    image_filename = storage_service.delete_task(dataset, task_id, settings)
    storage_service.append_audit_log(
        {
            "who": "expert",
            "when": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "task_id": task_id,
            "action": "delete_task",
            "image_filename": image_filename,
        },
        settings,
    )
    # Create-on-submit ⇒ remove-on-empty: a writable dataset exists only while it
    # holds tasks. When the last one is deleted, drop the whole dataset folder.
    dataset_removed = False
    if storage_service.count_tasks(dataset, settings) == 0:
        datasets_service.remove_dataset(dataset, settings)
        dataset_removed = True
    return {"ok": True, "dataset_removed": dataset_removed}


@router.get("/datasets/{dataset}/tasks/{task_id}/summary")
def get_task_summary(
    dataset: str,
    task_id: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _role: str = Depends(dependencies.get_role),
) -> dict:
    """A retrieved case's overall description + causes, for pre-filling the
    diagnosis draft editor from similar cases."""
    try:
        doc = storage_service.load_task(dataset, task_id, settings)
    except HTTPException:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task_not_found")
    return {
        "overall": {
            "colloquial_zh": doc.overall.colloquial_zh,
            "medical_zh": doc.overall.medical_zh,
        },
        "global_causes_zh": list(doc.global_causes_zh or []),
    }


@router.get("/datasets/{dataset}/task_locator", response_model=TaskLocatorResponse)
def locate_task(
    dataset: str,
    task_id: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _role: str = Depends(dependencies.get_role),
) -> TaskLocatorResponse:
    """Resolve a task_id (image stem) to its 1-based index in the dataset's
    `/images` listing — the index used by /annotate/:index. Used by the
    diagnosis report to link a retrieved case back to its annotation page."""
    image_filenames = storage_service.list_images(dataset, settings)
    for i, fn in enumerate(image_filenames):
        if Path(fn).stem == task_id:
            return TaskLocatorResponse(index=i + 1)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task_not_found")


@router.get("/datasets/{dataset}/healthy_images", response_model=ImageListResponse)
def list_healthy_images(
    dataset: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _role: str = Depends(dependencies.get_role),
) -> ImageListResponse:
    images = storage_service.list_healthy_images(dataset, settings)
    return ImageListResponse(images=images)


@router.post("/datasets/{dataset}/images/{filename}/move_to_healthy_images")
def move_image_to_healthy_images(
    dataset: str,
    filename: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _role: str = Depends(dependencies.require_editor),
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
    _role: str = Depends(dependencies.require_editor),
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
    _role: str = Depends(dependencies.get_role),
) -> LabelMapZhResponse:
    mapping = datasets_service.load_label_map_zh(dataset, settings)
    return LabelMapZhResponse(label_map_zh=mapping)


@router.get("/datasets/{dataset}/evidence_options_zh", response_model=EvidenceOptionsZhResponse)
def get_evidence_options_zh(
    dataset: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _role: str = Depends(dependencies.get_role),
) -> EvidenceOptionsZhResponse:
    mapping = datasets_service.load_evidence_options_zh(dataset, settings)
    return EvidenceOptionsZhResponse(evidence_options_zh=mapping)




@router.get("/datasets/{dataset}/annotated", response_model=AnnotatedListResponse)
def list_annotated(
    dataset: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _role: str = Depends(dependencies.get_role),
) -> AnnotatedListResponse:
    # Build list of annotated items (submitted by any role), sorted by last_modified_at descending
    items: list[AnnotatedItem] = []

    image_filenames = storage_service.list_images(dataset, settings)
    index_map = {Path(fn).stem: i + 1 for i, fn in enumerate(image_filenames)}
    last_submitters = storage_service.get_last_submitters(dataset, settings)

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
                    last_editor=last_submitters.get(task_id),
                )
            )
    items.sort(key=lambda x: x.last_modified_at, reverse=True)
    return AnnotatedListResponse(items=items)


@router.get("/datasets/{dataset}/commented", response_model=CommentedListResponse)
def list_commented(
    dataset: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _role: str = Depends(dependencies.get_role),
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
