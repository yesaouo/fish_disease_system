# Fish Disease Annotation Platform - Architecture Notes

## Overview
The system comprises a FastAPI backend serving JSON tasks and image streams from the existing `data/{dataset}` directories, and a Vite/React frontend that provides the expert annotation interface. All persistence remains file-backed; the API performs atomic writes and maintains lightweight in-memory caches.

## Backend (FastAPI)
- `app/main.py`: FastAPI entrypoint, middleware (CORS off by default), and router registration.
- `app/config.py`: Environment-driven settings (e.g., `DATA_ROOT`, `ROOT_PATH`, cache TTLs, backup settings).
- `app/models.py`: Pydantic schemas for requests/responses, internal task metadata, and validation.
- `app/routes/*`: API routers (`/api/login`, `/api/tasks/*`, `/api/datasets/*`, `/api/*/stats`).
- `app/services/datasets.py`: Dataset discovery, class loading from dataset `symptoms.json` (or `DATA_ROOT/symptoms.json` fallback) with timestamp-aware caching.
- `app/services/tasks.py`: Task cataloguing, dispatch algorithm, and submit/save flows.
- `app/services/storage.py`: SQLite-backed task persistence, audit-log append, and image path resolution.
- `app/services/stats.py`: Aggregated statistics with 60s in-memory cache and CSV rendering helpers.
- `app/services/backup.py`: Daily backup worker that snapshots dataset SQLite files when they changed that day.
- `app/utils/cache.py`: Shared memoization utilities keyed by dataset with TTL.
- `app/dependencies.py`: Common dependency providers (e.g., dataset validator, settings injection).

Data model notes:
- Each dataset stores annotation state in `{DATA_ROOT}/{dataset}/annotations.db` and references images under `{DATA_ROOT}/{dataset}/images/` or `healthy_images/`.
- On submit/save, the current task state and prior version are stored in the dataset database (`tasks` + `history` tables).
- Task JSON uses role editors: `general_editor` / `expert_editor` (single editor name per role).
- Detection boxes are stored in `box_xyxy` as pixel coordinates and validated against `image_width` / `image_height`.
- Legacy inputs are normalized on load (e.g., `box_2d` -> `box_xyxy`, `main_category/sub_category` -> `label/evidence_zh`).
- Audit log is stored at `{DATA_ROOT}/audit_log.jsonl` (append-only).

## Frontend (Vite + React + TypeScript)
- Directory `frontend/` contains Vite project with Tailwind CSS and `react-konva` for canvas rendering.
- `src/App.tsx`: Router setup (login, dataset selection, annotation workspace, admin dashboard).
- `src/api/client.ts`: Fetch wrapper with token handling and TypeScript interfaces (mirroring backend schemas).
- `src/features/auth`: Name entry page storing display name in `localStorage`.
- `src/features/datasets`: Dataset picker, class list management, and React Query hooks for caches.
- `src/features/annotation`: Main workspace with canvas layer (Konva stage), bounding box editing tools, side panel forms, keyboard shortcuts, validation feedback, and submit flow.
- `src/features/admin`: Admin metrics view (tables + CSV export).
- Shared utilities for coordinate clamping/normalization, validation helpers, and ordered list editing (▲/▼ buttons) for global causes/treatments.

## Task Dispatch Algorithm
- `/api/tasks/next` scans dataset DB rows and returns a random candidate.
- A task is considered dispatchable when `expert_editor` is empty and the requesting user is not already the `general_editor` or `expert_editor` for that task.
- General and expert roles share the same candidate pool (tasks pending expert review); the difference is which editor field gets set on submit (`general_editor` vs `expert_editor`).
- `/api/tasks/by_index` returns a task by 1-based index in dataset DB order.

## Editing Model
- `submit`: sets the role editor (`general_editor` or `expert_editor`), saves the task row, writes a history entry, and appends an audit entry.
- `save`: saves the task row without changing editor fields (does not mark completion).
- No optimistic concurrency (no `version`/`base_version` checks): last write wins.

## Caching & Performance
- In-memory caches keyed by dataset for classes list and statistics with configurable TTL (default 60s).
- Dataset listing caches root directory snapshot for TTL.
- Backup files are written with SQLite's native backup API so WAL-mode writes are captured consistently.

## Security & Validation
- Display names validated to contain only CJK/Latin letters (no digits/punctuation).
- All API routes validate dataset name against directory, path traversal guarded.
- Image responses stream file contents with correct MIME type; no external URLs.

## Deployment Notes
- Dev: run backend (Uvicorn) + frontend (Vite); Vite proxies `/api` to the backend to avoid CORS.
- Production (example): run both via PM2 using `ecosystem.config.js` (backend + frontend processes).
- Environment variables supply data root, cache TTLs, and backup settings.
- Daily backup (optional): the backend checks whether any dataset DB changed on the current UTC date. If so, it creates one backup for that date under `{DATA_ROOT}/backup/{dataset}_{YYYYMMDD}.db`. Controlled by env:
  - `DAILY_BACKUP_ENABLED` (default `true`)
  - `DAILY_CHECK_INTERVAL_SECONDS` (default `60`)
  - `BACKUP_DIRNAME` (default `backup`)
  Legacy env names `IDLE_BACKUP_ENABLED` and `IDLE_CHECK_INTERVAL_SECONDS` are still accepted. DB activity is detected from the latest mtime across `annotations.db` and `annotations.db-wal`.
