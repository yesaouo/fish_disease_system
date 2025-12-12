from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

Label = Literal["HEALTHY", "ABNORMAL"]


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # [x1, y1, x2, y2] in pixel coords
    label: Label
    score: float


@dataclass
class CauseCandidate:
    cause_id: str
    desc: str
    similarity: float


@dataclass
class ActionCandidate:
    action_id: str
    desc: str
    similarity: float
    safety_note: str | None = None


@dataclass
class Report:
    detections: List[Detection]
    causes: List[CauseCandidate]
    actions: List[ActionCandidate]
    knowledge_base_version: str
    disclaimer: str
    # Optional human-readable summary (used by healthy path / fallbacks)
    summary: str | None = None
