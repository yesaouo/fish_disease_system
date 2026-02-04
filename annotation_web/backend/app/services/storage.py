from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from fastapi import HTTPException, status

from ..config import Settings, get_settings
from . import datasets as datasets_service
from ..models import TaskDocument

_settings = get_settings()

# Task is considered healthy when detections are empty OR all labels are "healthy_region".
HEALTHY_LABEL = "healthy_region"


def _ensure_settings(settings: Settings | None) -> Settings:
    return settings or _settings


def get_annotations_dir(dataset_dir: Path) -> Path:
    # Legacy JSON dir (kept for migration scripts)
    return dataset_dir / "annotations"


def get_image_dir(dataset_dir: Path, *, is_healthy: bool = False) -> Path:
    return dataset_dir / ("healthy_images" if is_healthy else "images")


def get_db_path(dataset_dir: Path, settings: Settings | None = None) -> Path:
    settings = _ensure_settings(settings)
    return dataset_dir / settings.db_filename


def _infer_image_size(image_path: Path) -> Tuple[Optional[int], Optional[int]]:
    """Try to read image dimensions, falling back silently if unavailable."""
    try:
        from PIL import Image  # type: ignore

        with Image.open(image_path) as img:
            width, height = img.size
            return int(width), int(height)
    except Exception:
        return (None, None)


def read_raw_json(path: Path) -> Dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def _normalize_task(
    raw: Dict,
    dataset: str,
    stem: str,
    image_filename: str,
    dataset_dir: Optional[Path] = None,
    settings: Settings | None = None,
) -> TaskDocument:
    settings = _ensure_settings(settings)
    dataset_dir = dataset_dir or datasets_service.resolve_dataset_path(dataset, settings)
    raw = dict(raw)

    # Resolve image path by filename (location is fixed by filesystem, not by annotation state).
    candidate_images = get_image_dir(dataset_dir, is_healthy=False) / image_filename
    candidate_healthy = get_image_dir(dataset_dir, is_healthy=True) / image_filename
    if candidate_images.exists():
        image_path = candidate_images
    elif candidate_healthy.exists():
        image_path = candidate_healthy
    else:
        image_path = candidate_images

    def _coerce_dim(value: Any) -> Optional[int]:
        try:
            v = int(round(float(value)))
            return v if v > 0 else None
        except Exception:
            return None

    width = _coerce_dim(raw.get("image_width"))
    height = _coerce_dim(raw.get("image_height"))
    if (width is None or height is None) and image_path.exists():
        inferred_w, inferred_h = _infer_image_size(image_path)
        width = width or inferred_w
        height = height or inferred_h
    raw["image_width"] = width or 1000
    raw["image_height"] = height or 1000

    raw.setdefault("dataset", dataset)
    raw["dataset"] = dataset
    raw.setdefault("image_filename", image_filename)
    raw.setdefault("last_modified_at", datetime.now(timezone.utc).isoformat())
    raw.setdefault("overall", {})
    raw.setdefault("detections", [])
    raw.setdefault("global_causes_zh", [])
    raw.setdefault("global_treatments_zh", [])
    raw.setdefault("comments", [])
    raw.setdefault("generated_by", raw.get("generated_by"))

    # Editors: migrated to list[str]
    raw.setdefault("general_editor", [])
    raw.setdefault("expert_editor", [])

    # Legacy compatibility: map box_2d -> box_xyxy (x1,y1,x2,y2 in pixel space)
    try:
        dets = raw.get("detections", [])
        if isinstance(dets, list):
            max_dim = max(raw["image_width"], raw["image_height"])

            def _clamp(val: float, low: float, high: float) -> float:
                try:
                    return min(max(val, low), high)
                except Exception:
                    return val

            for d in dets:
                if not isinstance(d, dict):
                    continue
                if "box_xyxy" not in d and d.get("box_2d") is not None:
                    box = d.get("box_2d")
                    try:
                        y1, x1, y2, x2 = [float(v) for v in box]
                    except Exception:
                        continue
                    x1_new, y1_new, x2_new, y2_new = x1, y1, x2, y2
                    # If legacy values exceed image dims, assume 0-1000 scale and rescale.
                    if max_dim and any(coord > max_dim for coord in (x1_new, x2_new, y1_new, y2_new)):
                        y1_new = (y1_new / 1000.0) * raw["image_height"]
                        y2_new = (y2_new / 1000.0) * raw["image_height"]
                        x1_new = (x1_new / 1000.0) * raw["image_width"]
                        x2_new = (x2_new / 1000.0) * raw["image_width"]
                    x1_new = _clamp(x1_new, 0, raw["image_width"])
                    x2_new = _clamp(x2_new, 0, raw["image_width"])
                    y1_new = _clamp(y1_new, 0, raw["image_height"])
                    y2_new = _clamp(y2_new, 0, raw["image_height"])
                    d["box_xyxy"] = [x1_new, y1_new, x2_new, y2_new]
                d.pop("box_2d", None)
    except Exception:
        pass

    # Backward/forward compatibility: map main/sub to label/evidence if present
    try:
        dets = raw.get("detections", [])
        if isinstance(dets, list):
            for d in dets:
                if isinstance(d, dict):
                    if not d.get("label") and d.get("main_category"):
                        d["label"] = str(d.get("main_category") or "")
                    if ("evidence_zh" not in d) and d.get("sub_category") is not None:
                        d["evidence_zh"] = str(d.get("sub_category") or "")
    except Exception:
        pass

    # Always derive is_healthy from detections for consistency.
    try:
        dets = raw.get("detections", [])
        if not isinstance(dets, list) or len(dets) == 0:
            raw["is_healthy"] = True
        else:
            raw["is_healthy"] = all(
                isinstance(d, dict) and str(d.get("label") or "").strip() == HEALTHY_LABEL for d in dets
            )
    except Exception:
        raw["is_healthy"] = False

    return TaskDocument.model_validate(raw)


def find_image_filename(
    dataset: str,
    dataset_dir: Path,
    stem: str,
    settings: Settings,
    *,
    is_healthy: bool | None = None,
) -> tuple[str, bool]:
    """Resolve image filename (and whether it's under healthy_images) for a task stem.

    If `is_healthy` is provided, only search that directory; otherwise search `images/`
    then `healthy_images/`.
    """
    dirs: list[tuple[bool, Path]]
    if is_healthy is None:
        dirs = [
            (False, get_image_dir(dataset_dir, is_healthy=False)),
            (True, get_image_dir(dataset_dir, is_healthy=True)),
        ]
    else:
        dirs = [(bool(is_healthy), get_image_dir(dataset_dir, is_healthy=bool(is_healthy)))]

    for healthy_flag, image_dir in dirs:
        if not image_dir.exists():
            continue
        for ext in settings.image_extensions:
            candidate = image_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate.name, healthy_flag
        matches = list(image_dir.glob(f"{stem}.*"))
        if matches:
            return matches[0].name, healthy_flag

    append_audit_log(
        {
            "who": "system",
            "when": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "task_id": stem,
            "action": "missing_image",
        },
        settings,
    )
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="影像不存在")


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # Best-effort pragmas for multi-reader/single-writer workloads.
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
    except Exception:
        pass
    return conn


def ensure_db(dataset_dir: Path, settings: Settings | None = None) -> Path:
    """Create/open dataset DB and ensure tables exist.

    Returns db path.
    """
    settings = _ensure_settings(settings)
    db_path = get_db_path(dataset_dir, settings)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                  key   TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                  task_id             TEXT PRIMARY KEY,
                  sort_index          INTEGER NOT NULL UNIQUE,
                  image_filename      TEXT NOT NULL,
                  is_healthy          INTEGER NOT NULL DEFAULT 0,
                  last_modified_at    TEXT NOT NULL,
                  general_editors_json TEXT NOT NULL DEFAULT '[]',
                  expert_editors_json  TEXT NOT NULL DEFAULT '[]',
                  comments_count      INTEGER NOT NULL DEFAULT 0,
                  doc_json            TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_last_modified_at ON tasks(last_modified_at);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_is_healthy ON tasks(is_healthy);")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS history (
                  id         INTEGER PRIMARY KEY AUTOINCREMENT,
                  task_id    TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  updated_by TEXT,
                  action     TEXT NOT NULL,
                  old_json   TEXT,
                  new_json   TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_task_id ON history(task_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_updated_at ON history(updated_at);")
            conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', '1');")
    finally:
        conn.close()
    return db_path


def resolve_image_subdir(dataset_dir: Path, image_filename: str) -> str:
    """Return the subdir that contains the image file: `images` or `healthy_images`.

    Falls back to `images` if missing.
    """
    p1 = get_image_dir(dataset_dir, is_healthy=False) / image_filename
    if p1.exists():
        return "images"
    p2 = get_image_dir(dataset_dir, is_healthy=True) / image_filename
    if p2.exists():
        return "healthy_images"
    return "images"


def list_images_in_dir(dataset_dir: Path, settings: Settings, *, in_healthy_images: bool) -> list[Path]:
    """List image files in `images/` or `healthy_images/` (sorted by filename)."""
    image_dir = get_image_dir(dataset_dir, is_healthy=in_healthy_images)
    if not image_dir.exists():
        return []
    exts = settings.image_extensions
    files: list[Path] = []
    try:
        for p in image_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in exts:
                continue
            files.append(p)
    except Exception:
        return []
    files.sort(key=lambda x: x.name.lower())
    return files


def list_images(dataset: str, settings: Settings | None = None) -> list[str]:
    """List images that should be shown on the frontend (only from `images/`)."""
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    return [p.name for p in list_images_in_dir(dataset_dir, settings, in_healthy_images=False)]

def list_healthy_images(dataset: str, settings: Settings | None = None) -> list[str]:
    """List images under `healthy_images/`."""
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    return [p.name for p in list_images_in_dir(dataset_dir, settings, in_healthy_images=True)]


def ensure_task_for_image(
    dataset: str,
    task_id: str,
    image_filename: str,
    settings: Settings | None = None,
    *,
    in_healthy_images: bool | None = None,
) -> None:
    """Ensure a DB row exists for an image task (creating a default document if missing)."""
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    db_path = ensure_db(dataset_dir, settings)

    if in_healthy_images is None:
        image_path = get_image_dir(dataset_dir, is_healthy=False) / image_filename
        if not image_path.exists():
            alt = get_image_dir(dataset_dir, is_healthy=True) / image_filename
            if alt.exists():
                image_path = alt
    else:
        image_path = get_image_dir(dataset_dir, is_healthy=bool(in_healthy_images)) / image_filename
    try:
        mtime = image_path.stat().st_mtime
        last_modified_at = datetime.fromtimestamp(mtime, tz=timezone.utc)
    except Exception:
        last_modified_at = datetime.now(timezone.utc)

    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT doc_json FROM tasks WHERE task_id = ?;",
            (task_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        # Assign a monotonically increasing sort_index (DB internal; frontend index is based on `/images` order).
        conn2 = _connect(db_path)
        try:
            max_row = conn2.execute("SELECT COALESCE(MAX(sort_index), 0) AS m FROM tasks;").fetchone()
            max_sort = int(max_row["m"] if max_row is not None else 0)
        finally:
            conn2.close()

        doc = TaskDocument(
            dataset=dataset,
            image_filename=image_filename,
            last_modified_at=last_modified_at,
        )
        upsert_task(
            dataset,
            task_id,
            sort_index=max_sort + 1,
            document=doc,
            updated_by="system",
            action="sync_image_insert",
            settings=settings,
        )
        return

    try:
        raw = json.loads(row["doc_json"])
    except Exception:
        raw = {}
    if str(raw.get("image_filename") or "") != image_filename:
        raw["image_filename"] = image_filename
        raw["dataset"] = dataset
        raw.setdefault("last_modified_at", last_modified_at.isoformat())
        doc = _normalize_task(raw, dataset, task_id, image_filename, dataset_dir, settings)
        upsert_task(
            dataset,
            task_id,
            sort_index=None,
            document=doc,
            updated_by="system",
            action="sync_image_filename",
            settings=settings,
        )


def move_image_to_healthy_images(
    dataset: str,
    filename: str,
    settings: Settings | None = None,
) -> None:
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)

    if ".." in Path(filename).parts or Path(filename).name != filename:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")

    src_dir = get_image_dir(dataset_dir, is_healthy=False)
    dst_dir = get_image_dir(dataset_dir, is_healthy=True)
    src = (src_dir / filename).resolve()
    dst = (dst_dir / filename).resolve()

    if not str(src).startswith(str(src_dir.resolve())):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    if not str(dst).startswith(str(dst_dir.resolve())):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")

    if not src.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="image_not_in_images")

    dst_dir.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="already_in_healthy_images")

    try:
        os.replace(src, dst)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="move_failed") from exc


def move_image_to_images(
    dataset: str,
    filename: str,
    settings: Settings | None = None,
) -> None:
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)

    if ".." in Path(filename).parts or Path(filename).name != filename:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")

    src_dir = get_image_dir(dataset_dir, is_healthy=True)
    dst_dir = get_image_dir(dataset_dir, is_healthy=False)
    src = (src_dir / filename).resolve()
    dst = (dst_dir / filename).resolve()

    if not str(src).startswith(str(src_dir.resolve())):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")
    if not str(dst).startswith(str(dst_dir.resolve())):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not_found")

    if not src.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="image_not_in_healthy_images")

    dst_dir.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="already_in_images")

    try:
        os.replace(src, dst)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="move_failed") from exc


def load_task(
    dataset: str,
    stem: str,
    settings: Settings | None = None,
) -> TaskDocument:
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    db_path = get_db_path(dataset_dir, settings)
    if not db_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="資料庫不存在")
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT task_id, image_filename, doc_json FROM tasks WHERE task_id = ?;",
            (stem,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="task 不存在")
        raw = json.loads(row["doc_json"])
        image_filename = str(row["image_filename"])
        return _normalize_task(raw, dataset, str(row["task_id"]), image_filename, dataset_dir, settings)
    finally:
        conn.close()


def load_all_tasks(
    dataset: str,
    settings: Settings | None = None,
) -> Generator[Tuple[str, TaskDocument], None, None]:
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    db_path = get_db_path(dataset_dir, settings)
    if not db_path.exists():
        return
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT task_id, image_filename, doc_json FROM tasks ORDER BY sort_index ASC;"
        ).fetchall()
        for row in rows:
            stem = str(row["task_id"])
            try:
                raw = json.loads(row["doc_json"])
                image_filename = str(row["image_filename"])
                doc = _normalize_task(raw, dataset, stem, image_filename, dataset_dir, settings)
                yield stem, doc
            except Exception:
                continue
    finally:
        conn.close()


def list_task_rows(
    dataset: str,
    settings: Settings | None = None,
) -> list[sqlite3.Row]:
    """Return raw DB rows for a dataset (ordered by sort_index)."""
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    db_path = get_db_path(dataset_dir, settings)
    if not db_path.exists():
        return []
    conn = _connect(db_path)
    try:
        return conn.execute(
            """
            SELECT task_id, sort_index, image_filename, is_healthy, last_modified_at,
                   general_editors_json, expert_editors_json, comments_count, doc_json
            FROM tasks
            ORDER BY sort_index ASC;
            """
        ).fetchall()
    finally:
        conn.close()


def count_tasks(dataset: str, settings: Settings | None = None) -> int:
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    db_path = get_db_path(dataset_dir, settings)
    if not db_path.exists():
        return 0
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT COUNT(1) AS n FROM tasks;").fetchone()
        return int(row["n"] if row is not None else 0)
    finally:
        conn.close()


def get_task_id_by_index(dataset: str, index: int, settings: Settings | None = None) -> str | None:
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    db_path = get_db_path(dataset_dir, settings)
    if not db_path.exists():
        return None
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT task_id FROM tasks WHERE sort_index = ?;", (index,)).fetchone()
        return str(row["task_id"]) if row is not None else None
    finally:
        conn.close()


def upsert_task(
    dataset: str,
    task_id: str,
    sort_index: int | None,
    document: TaskDocument,
    *,
    updated_by: str | None,
    action: str,
    settings: Settings | None = None,
) -> None:
    """Insert/update a task row and record a history entry on updates."""
    settings = _ensure_settings(settings)
    dataset_dir = datasets_service.resolve_dataset_path(dataset, settings)
    db_path = ensure_db(dataset_dir, settings)

    payload = document.model_dump(mode="json")
    # Derive is_healthy from detections for consistent querying/statistics.
    dets = payload.get("detections")
    if isinstance(dets, list) and len(dets) > 0:
        derived = all(str((d or {}).get("label") or "").strip() == HEALTHY_LABEL for d in dets if isinstance(d, dict))
    else:
        derived = True
    payload["is_healthy"] = bool(derived)
    new_json = json.dumps(payload, ensure_ascii=False)
    now_iso = datetime.now(timezone.utc).isoformat()
    general_json = json.dumps(getattr(document, "general_editor", []) or [], ensure_ascii=False)
    expert_json = json.dumps(getattr(document, "expert_editor", []) or [], ensure_ascii=False)
    comments_count = int(len(getattr(document, "comments", []) or []))
    is_healthy = 1 if bool(derived) else 0

    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE;")
        existing = conn.execute(
            "SELECT doc_json, sort_index FROM tasks WHERE task_id = ?;",
            (task_id,),
        ).fetchone()

        if existing is None:
            if sort_index is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="sort_index required for insert")
            conn.execute(
                """
                INSERT INTO tasks(
                  task_id, sort_index, image_filename, is_healthy, last_modified_at,
                  general_editors_json, expert_editors_json, comments_count, doc_json
                ) VALUES(?,?,?,?,?,?,?,?,?);
                """,
                (
                    task_id,
                    int(sort_index),
                    document.image_filename,
                    is_healthy,
                    str(payload.get("last_modified_at") or now_iso),
                    general_json,
                    expert_json,
                    comments_count,
                    new_json,
                ),
            )
        else:
            # Keep existing ordering for updates.
            sort_index = int(existing["sort_index"])
            old_json = str(existing["doc_json"])
            conn.execute(
                """
                INSERT INTO history(task_id, updated_at, updated_by, action, old_json, new_json)
                VALUES(?,?,?,?,?,?);
                """,
                (task_id, now_iso, updated_by, action, old_json, new_json),
            )
            conn.execute(
                """
                UPDATE tasks
                SET image_filename=?,
                    is_healthy=?,
                    last_modified_at=?,
                    general_editors_json=?,
                    expert_editors_json=?,
                    comments_count=?,
                    doc_json=?
                WHERE task_id=?;
                """,
                (
                    document.image_filename,
                    is_healthy,
                    str(payload.get("last_modified_at") or now_iso),
                    general_json,
                    expert_json,
                    comments_count,
                    new_json,
                    task_id,
                ),
            )
        conn.commit()
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="資料庫寫入衝突") from exc
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def append_audit_log(entry: Dict[str, str], settings: Settings | None = None) -> None:
    settings = _ensure_settings(settings)
    log_path = settings.audit_log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with open(log_path, "a", encoding="utf-8") as fp:
        fp.write(line + "\n")
