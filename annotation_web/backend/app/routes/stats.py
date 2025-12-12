from __future__ import annotations

from fastapi import APIRouter, Depends

from .. import dependencies
from ..config import Settings
from ..models import AdminStatsResponse, StatsResponse, AdminTasksResponse
from ..services import stats as stats_service

router = APIRouter(prefix="/api", tags=["stats"])


@router.get("/datasets/{dataset}/stats", response_model=StatsResponse)
def get_dataset_stats(
    dataset: str,
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> StatsResponse:
    return stats_service.get_dataset_stats(dataset, settings)


@router.get("/admin/stats", response_model=AdminStatsResponse)
def get_admin_stats(
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> AdminStatsResponse:
    return stats_service.get_admin_stats(settings)


@router.get("/admin/tasks", response_model=AdminTasksResponse)
def get_admin_tasks(
    settings: Settings = Depends(dependencies.get_app_settings),
    _token: str = Depends(dependencies.require_api_key),
) -> AdminTasksResponse:
    return stats_service.get_admin_task_summaries(settings)
