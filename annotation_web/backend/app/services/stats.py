from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple
import json
from pathlib import Path

from ..config import Settings, get_settings
from ..models import AdminStatsResponse, StatsResponse, AdminTasksResponse, TaskSummary
from ..services import datasets as datasets_service
from ..services import storage as storage_service
from ..utils.cache import TTLCache

_settings = get_settings()
_stats_cache = TTLCache[str, StatsResponse](_settings.stats_cache_seconds)


def _aggregate_submissions_from_audit(dataset: str, log_path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    try:
        if not log_path.exists():
            return counts
        with open(log_path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("dataset") != dataset:
                    continue
                if str(obj.get("action") or "") != "submit":
                    continue
                user = str(obj.get("who") or "").strip()
                if user:
                    counts[user] += 1
    except Exception:
        pass
    return counts

def _aggregate_submissions_from_audit_split(dataset: str, log_path: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    general: Dict[str, int] = defaultdict(int)
    expert: Dict[str, int] = defaultdict(int)
    try:
        if not log_path.exists():
            return general, expert
        with open(log_path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("dataset") != dataset:
                    continue
                if str(obj.get("action") or "") != "submit":
                    continue
                user = str(obj.get("who") or "").strip()
                if not user:
                    continue
                is_expert = bool(obj.get("is_expert", False))
                if is_expert:
                    expert[user] += 1
                else:
                    general[user] += 1
    except Exception:
        pass
    return general, expert


# Removed skip aggregation: skip metrics are no longer tracked


def _ensure_settings(settings: Settings | None) -> Settings:
    return settings or _settings


def _compute_stats(dataset: str, settings: Settings) -> StatsResponse:
    total = 0
    completed = 0
    duplicate_completed = 0
    general_completed = 0
    expert_completed = 0
    # Compute submissions directly from task JSON (editors), not audit log
    submissions: Dict[str, int] = defaultdict(int)
    general_submissions: Dict[str, int] = defaultdict(int)
    expert_submissions: Dict[str, int] = defaultdict(int)

    for _, doc in storage_service.load_all_tasks(dataset, settings):
        total += 1
        # Completed if any editor (general or expert) exists
        has_general = bool(getattr(doc, "general_editor", None))
        has_expert = bool(getattr(doc, "expert_editor", None))
        if has_general or has_expert:
            completed += 1
        if has_general:
            general_completed += 1
            editor = getattr(doc, "general_editor", None)
            if editor:
                general_submissions[editor] += 1
                submissions[editor] += 1
        if has_expert:
            expert_completed += 1
            editor = getattr(doc, "expert_editor", None)
            if editor:
                expert_submissions[editor] += 1
                submissions[editor] += 1
        # Count items edited by two roles
        try:
            if has_general and has_expert:
                duplicate_completed += 1
        except Exception:
            pass

    # Skip metrics removed

    completion_rate = (completed / total) if total else 0.0

    return StatsResponse(
        dataset=dataset,
        total_tasks=total,
        completed_tasks=completed,
        duplicate_completed=duplicate_completed,
        general_completed_tasks=general_completed,
        expert_completed_tasks=expert_completed,
        submissions_by_user=dict(submissions),
        general_submissions_by_user=dict(general_submissions),
        expert_submissions_by_user=dict(expert_submissions),
        completion_rate=completion_rate,
    )


def get_dataset_stats(dataset: str, settings: Settings | None = None) -> StatsResponse:
    settings = _ensure_settings(settings)
    datasets_service.resolve_dataset_path(dataset, settings)
    return _stats_cache.get_or_set(dataset, lambda: _compute_stats(dataset, settings))


def get_admin_stats(settings: Settings | None = None) -> AdminStatsResponse:
    settings = _ensure_settings(settings)
    datasets = datasets_service.list_datasets(settings)
    stats = [get_dataset_stats(dataset, settings) for dataset in datasets]
    return AdminStatsResponse(datasets=stats)


def get_admin_task_summaries(settings: Settings | None = None) -> AdminTasksResponse:
    settings = _ensure_settings(settings)
    tasks: List[TaskSummary] = []
    datasets = datasets_service.list_datasets(settings)
    for dataset in datasets:
        for _, doc in storage_service.load_all_tasks(dataset, settings):
            try:
                annotations = int(bool(getattr(doc, "general_editor", None))) + int(bool(getattr(doc, "expert_editor", None)))
            except Exception:
                annotations = 0
            tasks.append(
                TaskSummary(
                    dataset=dataset,
                    image_filename=doc.image_filename,
                    annotations_count=annotations,
                    general_editor=getattr(doc, "general_editor", None),
                    expert_editor=getattr(doc, "expert_editor", None),
                )
            )
    return AdminTasksResponse(tasks=tasks)
