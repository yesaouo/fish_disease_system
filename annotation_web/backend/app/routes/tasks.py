from __future__ import annotations

from fastapi import APIRouter, Depends

from .. import dependencies
from ..config import Settings
from ..models import (
    NextTaskRequest,
    NextTaskResponse,
    SkipTaskRequest,
    SkipTaskResponse,
    SubmitTaskRequest,
    SubmitTaskResponse,
    TaskByIndexRequest,
    SaveTaskRequest,
    SaveTaskResponse,
)
from ..services import tasks as tasks_service

router = APIRouter(prefix="/api", tags=["tasks"])


@router.post("/tasks/next", response_model=NextTaskResponse)
def get_next_task(
    body: NextTaskRequest,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> NextTaskResponse:
    editor_name = body.editor_name or "anonymous"
    return tasks_service.select_next_task(body.dataset, editor_name, body.is_expert, settings)


@router.post("/tasks/{task_id}/submit", response_model=SubmitTaskResponse)
def submit_task(
    task_id: str,
    request: SubmitTaskRequest,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> SubmitTaskResponse:
    editor_name = request.editor_name
    dataset = request.full_json.dataset
    tasks_service.submit_task(
        dataset=dataset,
        task_id=task_id,
        incoming=request.full_json,
        editor_name=editor_name,
        is_expert=request.is_expert,
        settings=settings,
    )
    return SubmitTaskResponse(ok=True)


@router.post("/tasks/by_index", response_model=NextTaskResponse)
def get_task_by_index(
    body: TaskByIndexRequest,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> NextTaskResponse:
    editor_name = body.editor_name or "anonymous"
    return tasks_service.get_task_by_index(body.dataset, body.index, editor_name, body.is_expert, settings)

@router.post("/healthy_tasks/by_index", response_model=NextTaskResponse)
def get_healthy_task_by_index(
    body: TaskByIndexRequest,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> NextTaskResponse:
    editor_name = body.editor_name or "anonymous"
    return tasks_service.get_healthy_task_by_index(body.dataset, body.index, editor_name, body.is_expert, settings)


@router.post("/tasks/{task_id}/skip", response_model=SkipTaskResponse)
def skip_task(
    task_id: str,
    request: SkipTaskRequest,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> SkipTaskResponse:
    editor_name = request.editor_name
    tasks_service.skip_task(
        dataset=request.dataset,
        task_id=task_id,
        editor_name=editor_name,
        is_expert=request.is_expert,
        settings=settings,
    )
    return SkipTaskResponse(ok=True)


@router.post("/tasks/{task_id}/save", response_model=SaveTaskResponse)
def save_task(
    task_id: str,
    request: SaveTaskRequest,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> SaveTaskResponse:
    editor_name = request.editor_name
    dataset = request.full_json.dataset
    tasks_service.save_task(
        dataset=dataset,
        task_id=task_id,
        incoming=request.full_json,
        editor_name=editor_name,
        is_expert=request.is_expert,
        settings=settings,
    )
    return SaveTaskResponse(ok=True)
