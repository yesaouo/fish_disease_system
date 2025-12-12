# Fish Disease Annotation Platform - Architecture Notes

## Overview
The system comprises a FastAPI backend serving JSON tasks and image streams from the existing `data/{dataset}` directories, and a Vite/React frontend that provides the expert annotation interface. All persistence remains file-backed; the API performs atomic writes and maintains lightweight in-memory caches.

## Backend (FastAPI)
- `app/main.py`: FastAPI entrypoint, middleware (CORS off by default), and router registration.
- `app/config.py`: Environment-driven settings (e.g., `DATA_ROOT`, `REVISIT_PROBABILITY`, cache TTLs).
- `app/models.py`: Pydantic schemas for requests/responses, internal task metadata, and validation.
- `app/services/datasets.py`: Dataset discovery, class loading from `symptoms.json` (or legacy `classes.txt`) with timestamp-aware caching.
- `app/services/tasks.py`: Task cataloguing, dispatch algorithm (5% light revisit), version conflict checks, JSON normalization, skip handling, and per-user exclusion of "unrecognizable" tasks.
- `app/services/storage.py`: Atomic JSON writes via temp files, audit-log append, and image path resolution.
- `app/services/stats.py`: Aggregated statistics with 60s in-memory cache and CSV rendering helpers.
- `app/utils/cache.py`: Shared memoization utilities keyed by dataset with TTL.
- `app/dependencies.py`: Common dependency providers (e.g., dataset validator, settings injection).

Data model notes:
- Each task corresponds to `{DATA_ROOT}/{dataset}/annotations/{stem}.json`.
- Images resolve via `{stem}` stem lookup within allowed extensions (`IMAGE_EXTS` from settings).
- JSON normalization ensures all system-managed fields (`dataset`, `version`, `editors`, etc.) exist; coordinate validation enforces integer 0-1000 bounds and ordering.
- Audit log stored at `{DATA_ROOT}/audit_log.jsonl` with append-only atomic writes.
- Do not persist `skip_history` in annotations; use `unrecognizable_users` to record users who marked the image as "unrecognizable".

## Frontend (Vite + React + TypeScript)
- Directory `frontend/` contains Vite project with Tailwind + shadcn UI and `react-konva` for canvas rendering.
- `src/app.tsx`: Router setup (login, dataset selection, annotation workspace, admin dashboard).
- `src/api/client.ts`: Fetch wrapper with token handling and TypeScript interfaces (mirroring backend schemas).
- `src/features/auth`: Name entry page storing display name in `localStorage`.
- `src/features/datasets`: Dataset picker, class list management, and React Query hooks for caches.
- `src/features/annotation`: Main workspace with canvas layer (Konva stage), bounding box editing tools, side panel forms, keyboard shortcuts, validation feedback, submit/skip flows.
- `src/features/admin`: Admin metrics view with charts/table and CSV export trigger.
- Shared utilities for coordinate transforms between normalized (0-1000) and pixel space, validation helpers, and drag-and-drop ordering for global causes/treatments.

## Task Dispatch Algorithm
- First exclude any tasks the requesting user previously marked as "unrecognizable" (via `unrecognizable_users`).
- Split the remaining tasks into two buckets:
  - A: `editors` is empty (never edited).
  - B: previously edited and `editors` does not include the current user.
- On each `/tasks/next` call: with probability `p=REVISIT_PROBABILITY` (default 0.05) choose randomly from B (fallback to A). Otherwise choose from A (fallback to B). Tasks missing JSON are skipped silently; missing images return HTTP 404 and are recorded in the audit log.

## Concurrency & Versioning
- Clients submit `base_version`; backend compares to JSON’s `version`. Mismatch -> HTTP 409 and no write.
- Successful submit increments version, appends editor name to `editors` (deduped), and appends a new `edit_history` entry.
- Skip actions do not change version:
  - Skip: only move to the next task; nothing is written to annotations (an audit log entry is still appended).
  - Unrecognizable: add the user to `unrecognizable_users` and save (also append to audit log); the task will not be assigned to that user again.
- Special case: submissions by username `test` save content and append to audit log but do not increment `version`, and do not modify `editors`/`edit_history`.

## Caching & Performance
- In-memory caches keyed by dataset for classes list and statistics with configurable TTL (default 60s).
- Dataset listing caches root directory snapshot for TTL.
- All writes occur via `NamedTemporaryFile` followed by `os.replace` for atomicity (NFS-safe).

## Security & Validation
- Display names validated to contain only CJK/Latin letters (no digits/punctuation).
- All API routes validate dataset name against directory, path traversal guarded.
- Image responses stream file contents with correct MIME type; no external URLs.

## Deployment Notes
- Docker compose with services: `frontend`, `backend`, and `nginx` (static frontend + reverse proxy).
- Environment variables supply roots and probabilities; defaults align with spec.
- Idle backup (optional): when no annotations occur for a configured idle window, the backend snapshots `annotations` for all datasets into a timestamped folder under `{DATA_ROOT}/backup/YYYYMMDD-HHMMSS/`. Controlled by env:
  - `IDLE_BACKUP_ENABLED` (default `true`)
  - `IDLE_BACKUP_SECONDS` (default `21600`, i.e. 6 hours)
  - `IDLE_CHECK_INTERVAL_SECONDS` (default `60`)
  - `BACKUP_DIRNAME` (default `backup`)
  The idle detector uses the mtime of `{DATA_ROOT}/audit_log.jsonl` (writes on submit/skip). One backup is created per unchanged activity period; a marker `.last_activity_mtime` prevents duplicates across restarts. Each snapshot includes a top-level `meta.json` with creation time and the source activity mtime.
