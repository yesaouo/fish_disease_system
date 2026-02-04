from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple
import json
from pathlib import Path

from ..config import Settings, get_settings
from ..models import AdminStatsResponse, StatsResponse, AdminTasksResponse, TaskSummary
from ..services import datasets as datasets_service
from ..services import storage as storage_service
from ..utils.cache import TTLCache

_settings = get_settings()
_stats_cache = TTLCache[str, StatsResponse](_settings.stats_cache_seconds)

HEALTHY_LABEL = storage_service.HEALTHY_LABEL


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


def _is_blank(value: str) -> bool:
    return not (value or "").strip()


def _required_fields_ok(doc) -> bool:
    try:
        dets = getattr(doc, "detections", []) or []
        is_healthy = (len(dets) == 0) or all((getattr(d, "label", "") or "").strip() == HEALTHY_LABEL for d in dets)
        # Healthy: no required fields.
        if is_healthy:
            return True

        if _is_blank(getattr(doc.overall, "colloquial_zh", "")):
            return False
        if _is_blank(getattr(doc.overall, "medical_zh", "")):
            return False
        if not (getattr(doc, "detections", []) or []):
            return False
        for det in doc.detections:
            label = (getattr(det, "label", "") or "").strip()
            if _is_blank(label):
                return False
            if label != HEALTHY_LABEL and getattr(det, "evidence_index", None) is None:
                return False
        if not (getattr(doc, "global_causes_zh", []) or []):
            return False
        if not (getattr(doc, "global_treatments_zh", []) or []):
            return False
        return True
    except Exception:
        return False


def _compute_stats(dataset: str, settings: Settings) -> StatsResponse:
    image_filenames = storage_service.list_images(dataset, settings)
    task_ids = [Path(fn).stem for fn in image_filenames]
    task_id_set = set(task_ids)

    total = len(image_filenames)
    completed = 0
    duplicate_completed = 0
    general_completed = 0
    expert_completed = 0

    # Count submissions from audit log (supports multiple submits per task/user).
    general_submissions, expert_submissions = _aggregate_submissions_from_audit_split(dataset, settings.audit_log_path)
    submissions: Dict[str, int] = defaultdict(int)
    for user, count in general_submissions.items():
        submissions[user] += count
    for user, count in expert_submissions.items():
        submissions[user] += count

    rows = storage_service.list_task_rows(dataset, settings)
    row_by_task_id = {str(r["task_id"]): r for r in rows if str(r["task_id"]) in task_id_set}

    def _is_blank2(value: Any) -> bool:
        return not str(value or "").strip()

    def _required_fields_ok_raw(raw: dict) -> bool:
        dets = raw.get("detections", [])
        if not isinstance(dets, list) or len(dets) == 0:
            return True

        labels: list[str] = []
        for d in dets:
            if not isinstance(d, dict):
                return False
            label = str(d.get("label") or "").strip()
            if not label:
                return False
            labels.append(label)
            if label != HEALTHY_LABEL and d.get("evidence_index") in (None, ""):
                return False

        # Healthy (all boxes are healthy_region) => OK
        if all(label == HEALTHY_LABEL for label in labels):
            return True

        # Non-healthy requires overall + global fields
        overall = raw.get("overall") if isinstance(raw.get("overall"), dict) else {}
        if _is_blank2(overall.get("colloquial_zh")):
            return False
        if _is_blank2(overall.get("medical_zh")):
            return False
        causes = raw.get("global_causes_zh")
        if not isinstance(causes, list) or len(causes) == 0:
            return False
        treatments = raw.get("global_treatments_zh")
        if not isinstance(treatments, list) or len(treatments) == 0:
            return False
        return True

    for task_id in task_ids:
        row = row_by_task_id.get(task_id)
        if row is None:
            continue
        try:
            general_editors = json.loads(row["general_editors_json"] or "[]")
            expert_editors = json.loads(row["expert_editors_json"] or "[]")
            if not isinstance(general_editors, list):
                general_editors = []
            if not isinstance(expert_editors, list):
                expert_editors = []
            has_general = bool(general_editors)
            has_expert_submitted = bool(expert_editors)
            has_comments = int(row["comments_count"] or 0) > 0
            try:
                raw = json.loads(row["doc_json"])
            except Exception:
                raw = {}
            required_ok = _required_fields_ok_raw(raw) if isinstance(raw, dict) else False
            expert_complete = has_expert_submitted and (required_ok or has_comments)

            if has_general:
                general_completed += 1
            if expert_complete:
                expert_completed += 1
            if has_general and has_expert_submitted:
                duplicate_completed += 1
        except Exception:
            continue

    # Skip metrics removed

    # Overall completion is "expert complete".
    completed = expert_completed
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
                annotations = int(bool(getattr(doc, "general_editor", []) or [])) + int(bool(getattr(doc, "expert_editor", []) or []))
            except Exception:
                annotations = 0
            tasks.append(
                TaskSummary(
                    dataset=dataset,
                    image_filename=doc.image_filename,
                    annotations_count=annotations,
                    general_editor=list(getattr(doc, "general_editor", []) or []),
                    expert_editor=list(getattr(doc, "expert_editor", []) or []),
                )
            )
    return AdminTasksResponse(tasks=tasks)
