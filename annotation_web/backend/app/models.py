from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict

NAME_PATTERN = re.compile(r"^[\u4e00-\u9fffA-Za-z]{1,32}$")

def _ensure_aware(v: datetime) -> datetime:
    if isinstance(v, datetime) and v.tzinfo is None:
        return v.replace(tzinfo=timezone.utc)
    return v

class OverallText(BaseModel):
    model_config = ConfigDict(validate_default=True)

    colloquial_zh: str = ""
    medical_zh: str = ""

    @field_validator("*", mode="before")
    def _strip_newlines(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).replace("\n", " ").strip()

class Detection(BaseModel):
    model_config = ConfigDict(validate_default=True, extra="ignore")

    label: str
    box_xyxy: List[int]
    confidence: Optional[float] = None
    evidence_zh: str = ""

    @field_validator("label", mode="before")
    def _clean_label(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("evidence_zh", mode="before")
    def _cleanup_text(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).replace("\n", " ").strip()

    @field_validator("box_xyxy", mode="before")
    def _validate_box(cls, value: Any) -> List[int]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("box_xyxy must be a list of 4 numeric values")
        if len(value) != 4:
            raise ValueError("box_xyxy must contain 4 values")
        ints = [int(round(float(v))) for v in value]
        x1, y1, x2, y2 = ints
        for coord in ints:
            if coord < 0:
                raise ValueError("box_xyxy coordinates must be non-negative")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box_xyxy must satisfy x2>x1 and y2>y1")
        return ints

# Edit history removed

class Comment(BaseModel):
    model_config = ConfigDict(validate_default=True)

    author: str
    text: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("author", mode="before")
    def _clean_author(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("text", mode="before")
    def _clean_text(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).replace("\n", " ").strip()

class TaskDocument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dataset: str
    image_filename: str
    image_width: int = Field(default=1000)
    image_height: int = Field(default=1000)
    last_modified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Single editor per role
    general_editor: Optional[str] = None
    expert_editor: Optional[str] = None
    overall: OverallText = Field(default_factory=OverallText)
    detections: List[Detection] = Field(default_factory=list)
    global_causes_zh: List[str] = Field(default_factory=list)
    global_treatments_zh: List[str] = Field(default_factory=list)
    generated_by: Optional[str] = None
    comments: List[Comment] = Field(default_factory=list)
    # QA flags removed; no 'qa' field persisted or validated

    @field_validator("global_causes_zh", "global_treatments_zh", mode="before")
    def _strip_text(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            value = [value]
        return [str(v).replace("\n", " ").strip() for v in value if v is not None]

    @field_validator("image_width", "image_height", mode="before")
    def _validate_image_dims(cls, value: Any) -> int:
        if value is None:
            return 1000
        try:
            v = int(round(float(value)))
        except Exception:
            raise ValueError("image dimension must be numeric")
        if v <= 0:
            raise ValueError("image dimension must be positive")
        return v

    @model_validator(mode="after")
    def _validate_global_lists(self):
        for key in ("global_causes_zh", "global_treatments_zh"):
            items = getattr(self, key, []) or []
            if len(items) > 10:
                raise ValueError(f"{key} may contain at most 10 items")
            seen = set()
            deduped = []
            for item in items:
                if not item:
                    continue
                if item in seen:
                    raise ValueError(f"{key} contains duplicate entries")
                seen.add(item)
                deduped.append(item)
            setattr(self, key, deduped)
        return self

    @model_validator(mode="after")
    def _validate_boxes_within_image(self):
        width = getattr(self, "image_width", 0) or 0
        height = getattr(self, "image_height", 0) or 0
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        for det in self.detections:
            try:
                x1, y1, x2, y2 = det.box_xyxy
            except Exception:
                continue
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                raise ValueError("box_xyxy must stay within image_width/image_height")
        return self

class LoginRequest(BaseModel):
    name: str
    is_expert: bool = True
    api_key: str

    @field_validator("name")
    def _validate_name(cls, value: str) -> str:
        name = value.strip()
        if not NAME_PATTERN.match(name):
            raise ValueError("名稱僅限中英文，長度 1-32")
        return name

    @field_validator("api_key")
    def _validate_api_key(cls, value: str) -> str:
        key = (value or "").strip()
        if not key:
            raise ValueError("api_key 不可為空白")
        return key

class LoginResponse(BaseModel):
    token: str
    name: str

class DatasetListResponse(BaseModel):
    datasets: List[str]

class ClassesResponse(BaseModel):
    classes: List[str]

class LabelMapZhResponse(BaseModel):
    # Mapping from English class value to Chinese display text
    label_map_zh: Dict[str, str]

class NextTaskRequest(BaseModel):
    dataset: str
    editor_name: Optional[str] = None
    is_expert: bool = True

    @field_validator("dataset")
    def _validate_dataset(cls, value: str) -> str:
        return value.strip()

    @field_validator("editor_name")
    def _validate_editor(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        name = value.strip()
        if not NAME_PATTERN.match(name):
            raise ValueError("editor_name 必須為中英文，長度限制為 1-32")
        return name

class NextTaskResponse(BaseModel):
    task_id: str
    task: TaskDocument
    image_url: str
    # 1-based index within the dataset, and total task count
    index: int
    total_tasks: int

class TaskByIndexRequest(BaseModel):
    dataset: str
    index: int  # 1-based index
    editor_name: Optional[str] = None
    is_expert: bool = True

    @field_validator("dataset")
    def _validate_dataset(cls, value: str) -> str:
        return value.strip()

    @field_validator("editor_name")
    def _validate_editor(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        name = value.strip()
        if not NAME_PATTERN.match(name):
            raise ValueError("editor_name 必須為中英文，長度限制為 1-32")
        return name

class SubmitTaskRequest(BaseModel):
    full_json: TaskDocument
    editor_name: str
    is_expert: bool = True

    @field_validator("editor_name")
    def _validate_editor(cls, value: str) -> str:
        name = value.strip()
        if not NAME_PATTERN.match(name):
            raise ValueError("editor_name 僅限中英文，長度 1-32")
        return name

class SubmitTaskResponse(BaseModel):
    ok: bool

class SkipTaskRequest(BaseModel):
    dataset: str
    editor_name: str
    is_expert: bool = True

    @field_validator("dataset")
    def _validate_dataset(cls, value: str) -> str:
        return value.strip()

    @field_validator("editor_name")
    def _validate_editor(cls, value: str) -> str:
        name = value.strip()
        if not NAME_PATTERN.match(name):
            raise ValueError("editor_name 僅限中英文，長度 1-32")
        return name

class SkipTaskResponse(BaseModel):
    ok: bool = True

class StatsResponse(BaseModel):
    dataset: str
    total_tasks: int
    completed_tasks: int
    duplicate_completed: int
    # Split completion counts by role
    general_completed_tasks: int = 0
    expert_completed_tasks: int = 0
    submissions_by_user: Dict[str, int]
    # Split submissions by role (from audit log)
    general_submissions_by_user: Dict[str, int] = Field(default_factory=dict)
    expert_submissions_by_user: Dict[str, int] = Field(default_factory=dict)
    completion_rate: float

class AdminStatsResponse(BaseModel):
    datasets: List[StatsResponse]

class TaskSummary(BaseModel):
    dataset: str
    image_filename: str
    annotations_count: int
    general_editor: Optional[str] = None
    expert_editor: Optional[str] = None

class AdminTasksResponse(BaseModel):
    tasks: List[TaskSummary]


# Save task (partial) without marking completion
class SaveTaskRequest(BaseModel):
    full_json: TaskDocument
    editor_name: str
    is_expert: bool = True

    @field_validator("editor_name")
    def _validate_editor(cls, value: str) -> str:
        name = value.strip()
        if not NAME_PATTERN.match(name):
            raise ValueError("editor_name 僅限中英文，長度 1-32")
        return name

class SaveTaskResponse(BaseModel):
    ok: bool


class AnnotatedItem(BaseModel):
    dataset: str
    index: int
    task_id: str
    image_filename: str
    last_modified_at: datetime
    general_editor: Optional[str] = None
    expert_editor: Optional[str] = None

class AnnotatedListResponse(BaseModel):
    items: List[AnnotatedItem]


class CommentedItem(BaseModel):
    dataset: str
    index: int
    task_id: str
    image_filename: str
    last_modified_at: datetime
    comments_count: int

class CommentedListResponse(BaseModel):
    items: List[CommentedItem]


# NPUST symptoms mapping response
