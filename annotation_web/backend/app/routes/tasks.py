from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends

from .. import dependencies
from ..config import Settings
from ..models import (
    NextTaskRequest,
    NextTaskResponse,
    SubmitTaskRequest,
    SubmitTaskResponse,
    TaskByIndexRequest,
    SaveTaskRequest,
    SaveTaskResponse,
)
from ..services import bank_sync
from ..services import tasks as tasks_service

router = APIRouter(prefix="/api", tags=["tasks"])


@router.post("/tasks/next", response_model=NextTaskResponse)
def get_next_task(
    body: NextTaskRequest,
    settings: Settings = Depends(dependencies.get_app_settings),
    role: str = Depends(dependencies.get_role),
) -> NextTaskResponse:
    editor_name = body.editor_name or "anonymous"
    return tasks_service.select_next_task(body.dataset, editor_name, role == "expert", settings)


@router.post("/tasks/{task_id}/submit", response_model=SubmitTaskResponse)
def submit_task(
    task_id: str,
    request: SubmitTaskRequest,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(dependencies.get_app_settings),
    role: str = Depends(dependencies.require_editor),
) -> SubmitTaskResponse:
    editor_name = request.editor_name
    dataset = request.full_json.dataset
    new_version = tasks_service.submit_task(
        dataset=dataset,
        task_id=task_id,
        incoming=request.full_json,
        editor_name=editor_name,
        is_expert=role == "expert",
        settings=settings,
    )
    # Mirror the submitted case into the retrieval bank (writable datasets only;
    # healthy → removed). No-op for locked originals. Best-effort, after response.
    background_tasks.add_task(bank_sync.sync_upsert, dataset, task_id, settings)
    return SubmitTaskResponse(ok=True, version=new_version)


@router.post("/tasks/by_index", response_model=NextTaskResponse)
def get_task_by_index(
    body: TaskByIndexRequest,
    settings: Settings = Depends(dependencies.get_app_settings),
    role: str = Depends(dependencies.get_role),
) -> NextTaskResponse:
    editor_name = body.editor_name or "anonymous"
    return tasks_service.get_task_by_index(body.dataset, body.index, editor_name, role == "expert", settings)

@router.post("/healthy_tasks/by_index", response_model=NextTaskResponse)
def get_healthy_task_by_index(
    body: TaskByIndexRequest,
    settings: Settings = Depends(dependencies.get_app_settings),
    role: str = Depends(dependencies.get_role),
) -> NextTaskResponse:
    editor_name = body.editor_name or "anonymous"
    return tasks_service.get_healthy_task_by_index(body.dataset, body.index, editor_name, role == "expert", settings)

@router.post("/tasks/{task_id}/save", response_model=SaveTaskResponse)
def save_task(
    task_id: str,
    request: SaveTaskRequest,
    settings: Settings = Depends(dependencies.get_app_settings),
    role: str = Depends(dependencies.require_editor),
) -> SaveTaskResponse:
    editor_name = request.editor_name
    dataset = request.full_json.dataset
    new_version = tasks_service.save_task(
        dataset=dataset,
        task_id=task_id,
        incoming=request.full_json,
        editor_name=editor_name,
        is_expert=role == "expert",
        settings=settings,
    )
    return SaveTaskResponse(ok=True, version=new_version)
