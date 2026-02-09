# Fish Disease Annotation Platform

Expert-facing annotation platform for fish disease labeling. Backend is FastAPI (file-backed JSON tasks + images under `data/` with audit logging and atomic writes); frontend is React + Vite for bounding-box editing, label management, and basic admin metrics.

中文說明：`README_zh.md`

## Prerequisites

- Python 3.11+
- Node.js 18+ and `npm`
- Recommended: Conda/Mamba (lets you manage Python + Node.js in one env)
- Datasets under `data/{dataset}` with images and JSON cache

## Quick Start

1) Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # use `copy` on Windows; tweak DATA_ROOT/ROOT_PATH/etc. as needed
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Add at least one API key under `<DATA_ROOT>/<AUTH_KEYS_FILENAME>` (default `data/auth_keys.txt`), one key per line; login will be blocked without it.

2) Frontend

```bash
cd frontend
npm install
npm run dev -- --host --port 5173
```

3) Open UI

- Visit `http://localhost:5173`
- The dev server proxies `/api/*` to `http://localhost:8000` (adjust `frontend/vite.config.ts` if you run the backend on another port)

## Run with Conda + PM2 (Linux/systemd)

This repo includes `ecosystem.config.js` for running backend + frontend via PM2. The flow below follows a typical deployment on a Linux server.

```bash
git clone https://github.com/yesaouo/fish_disease_system.git
cd fish_disease_system/annotation_web

conda info --base
conda create -n annotation_web python=3.11.13
conda activate annotation_web
conda install -c conda-forge nodejs

cd frontend
npm install
cd ..
pip install -r backend/requirements.txt

npm install -g pm2

# Edit parameters in:
# - ecosystem.config.js (CONDA_BASE / CONDA_ENV / LOG_ROOT / ports)
# - backend/.env (DATA_ROOT / ROOT_PATH / cache settings / etc.)

pm2 start ecosystem.config.js
pm2 status
pm2 logs
```

Auto-start on boot (systemd):

```bash
pm2 startup
# If your Node/PM2 lives inside a conda env, systemd may not find `node` unless PATH is set.
# Example (replace placeholders with your own values):
sudo env PATH=$PATH:<conda_env_bin> <pm2_bin> startup systemd -u <user> --hp <home>
pm2 save
```

## Directory Layout

- `backend/` – FastAPI application (entry: `app.main:app`)
- `frontend/` – Vite + React UI (`npm install && npm run dev`)
- `docs/` – architecture notes and user guides
- `data/{dataset}` – images and JSON cache
- `backend/scripts/` – maintenance utilities
- `ecosystem.config.js` – PM2 process definition (backend + frontend)

## Configuration

Environment variables (case-insensitive; JSON for list-like values, e.g. IMAGE_EXTS=["jpg","png"]):

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATA_ROOT` | `<repo>/data` | Root containing datasets |
| `IMAGE_EXTS` | `["jpg","jpeg","png"]` | Allowed image extensions (JSON array) |
| `STATS_CACHE_SECONDS` | `60` | Cache window for stats endpoints |
| `DATASET_CACHE_SECONDS` | `60` | Cache window for dataset listing |
| `CLASSES_CACHE_SECONDS` | `60` | Cache window for class list |
| `AUDIT_LOG_FILENAME` | `audit_log.jsonl` | File name under `DATA_ROOT` for audit log |
| `AUTH_KEYS_FILENAME` | `auth_keys.txt` | API key file under `DATA_ROOT` (one key per line; blank lines/# ignored) |
| `IDLE_BACKUP_ENABLED` | `true` | Enable idle-time dataset backups |
| `IDLE_BACKUP_SECONDS` | `21600` | Idle duration before a backup (seconds) |
| `IDLE_CHECK_INTERVAL_SECONDS` | `60` | Polling interval for idle checks |
| `BACKUP_DIRNAME` | `backup` | Subfolder inside `DATA_ROOT` for backups |
| `ROOT_PATH` | `` | API root path prefix (e.g. `/fish`) |

Health check: `GET /healthz` returns `{status: "ok"}` when the API is healthy.

### Environment (.env)

- Backend reads environment variables from `backend/.env` by default. See `.env.example`.
- You can override the env file path via `ENV_FILE=...`.
- Frontend is configured via `frontend/vite.config.ts` (proxy/base). If you need Vite env vars, add `frontend/.env` following Vite conventions.


### Deploying Under a Subpath

If you need to serve the app under a subpath (for example `https://example.com/fish/`), configure both backend and frontend:

- Backend (FastAPI):
  - Set env `ROOT_PATH=/fish` (or your subpath without trailing slash). This sets FastAPI’s `root_path` so routes are served under `/fish` and generated links respect the prefix.
  - Alternatively, start Uvicorn with `--root-path /fish`.
  - If using a reverse proxy (nginx), also forward `X-Forwarded-Prefix: /fish`.

- Frontend (Vite + React):
  - Set the Vite `base` option in `frontend/vite.config.ts` (must end with a slash), for example `base: "/fish/"`, so assets and router paths are generated under the subpath. The current repo default is `"/annotation_web/"`—change it if you serve from another path.
  - The router and API client are wired to honor `import.meta.env.BASE_URL`, so navigation and API calls use `/fish/...` automatically.

- Proxy example (nginx):
  - `location /fish/ { try_files $uri /fish/index.html; }`
  - `location /fish/api/ { proxy_pass http://127.0.0.1:8000/; proxy_set_header X-Forwarded-Prefix /fish; }`

With the above, the UI runs at `/fish/`, API at `/fish/api/*`, and image links returned by the API include the prefix.

## API Highlights

- Auth: All API routes except image streaming require `Authorization: Bearer <api_key>` where keys are read from `<DATA_ROOT>/<AUTH_KEYS_FILENAME>`.
- `POST /api/login` – returns `{token, name}`; use `Authorization: Bearer <token>`
- `GET /api/datasets` / `GET /api/datasets/{dataset}/classes`
- `GET /api/datasets/{dataset}/labels_zh` – English→Chinese label mapping (if available)
- `GET /api/datasets/{dataset}/evidence_options_zh` – evidence caption options per label (Chinese preferred)
- `GET /api/datasets/{dataset}/annotated` – tasks that have been edited, sorted by last modified
- `GET /api/datasets/{dataset}/commented` – tasks with comments, sorted by last modified
- `POST /api/tasks/next` – random dispatch among untouched tasks
- `POST /api/tasks/by_index` – fetch task by 1-based index within dataset
- `POST /api/tasks/{task_id}/submit` – submit annotations for a task
- `POST /api/tasks/{task_id}/skip` – records skip history (no version bump)
- `POST /api/tasks/{task_id}/save` – save without marking completion
- `GET /api/datasets/{dataset}/stats`, `GET /api/admin/stats`
- `GET /api/admin/tasks` – summary of tasks across datasets
- `GET /api/datasets/{dataset}/images/{filename}` – image streaming; missing files are logged

All writes are atomic (temp file + rename). Audit entries append to `<DATA_ROOT>/audit_log.jsonl`.

### Notes

- Dispatch is role-aware: general annotators skip items already expert-reviewed, experts get anything lacking `expert_editor`, and no user sees a task they already edited (comments don’t block expert review).

## Frontend Highlights

- Login gate with name validation
- Dataset picker loads class list from dataset `symptoms.json` (falling back to `DATA_ROOT/symptoms.json` if needed); selection persisted locally
- Annotation workspace
  - `react-konva` canvas with drag/resize in normalized 0–1000 coordinates
  - Side panel for label + evidence (dropdown); ordered global causes/treatments
  - Shortcuts: `N` add, `Del` remove, `S` submit, `K` skip, `Ctrl+Z/Y` undo/redo
  - Validation for boxes, labels, and list constraints before submission
  - Skip reasons: `暫時跳過` / `無法辨識`
- Admin dashboard with aggregate metrics and CSV export

Unsaved changes trigger a `beforeunload` warning.

## Documentation

- Architecture: `docs/architecture.md`
- 影像標註 SOP: `docs/影像標註_SOP.md`
- 前端使用說明（養殖專家）: `docs/使用說明_養殖專家.md`

## Next Steps

- Harden token persistence (e.g. expiry refresh, revocation list)
- Expand admin dashboard with per-user charts and date filters
- Add automated tests (unit + integration) for task dispatch and file operations
- Containerize (Docker Compose) integrating frontend, backend, and nginx per deployment section of spec
- Wire monitoring hooks (structured logging already JSON friendly)
