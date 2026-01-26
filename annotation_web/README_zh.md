# 魚病標註平台（Annotation Web）

提供養殖專家/標註人員使用的病灶框選與病徵填寫平台。後端使用 FastAPI（讀寫 `data/` 下的任務 JSON 與影像，並以原子寫入與稽核日誌確保一致性）；前端使用 React + Vite（框選編輯、類別/敘述/原因/處置填寫與簡易統計）。

English version: `README.md`

## 需求環境

- Python 3.11+
- Node.js 18+（含 `npm`）
- 建議：Conda/Mamba（可把 Python + Node.js 裝在同一個環境）
- 資料集需放在 `data/{dataset}`（影像與對應的 JSON/快取）

## 快速開始（開發模式）

1) 後端

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env  # Windows 請改用 `copy` 或 PowerShell `Copy-Item`
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- 請在 `<DATA_ROOT>/<AUTH_KEYS_FILENAME>`（預設 `data/auth_keys.txt`）至少放入 1 組金鑰（每行一組），否則無法登入。

2) 前端

```bash
cd frontend
npm install
npm run dev -- --host --port 5173
```

3) 開啟

- 瀏覽 `http://localhost:5173`
- Vite 會把 `/api/*` 代理到 `http://localhost:8000`；若後端改用其他埠，請同步調整 `frontend/vite.config.ts`

## 運行流程 Conda + PM2 (Linux/systemd)

專案內建 `ecosystem.config.js` 可用 PM2 同時啟動後端與前端（適合在伺服器上常駐）。

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

# 需要先修改參數：
# - ecosystem.config.js（CONDA_BASE / CONDA_ENV / LOG_ROOT / ports）
# - backend/.env（DATA_ROOT / ROOT_PATH / cache 相關設定等）

pm2 start ecosystem.config.js
pm2 status
pm2 logs
```

開機自動啟動（systemd）：

```bash
pm2 startup
# 若 Node/PM2 安裝在 conda env 內，systemd 可能找不到 `node`，需要把 env 的 bin 加進 PATH。
# 範例（請把 <> 內容替換成你的實際路徑/帳號）：
sudo env PATH=$PATH:<conda_env_bin> <pm2_bin> startup systemd -u <user> --hp <home>
pm2 save
```

## 文件

- 架構說明：`docs/architecture.md`
- 影像標註 SOP：`docs/影像標註_SOP.md`
- 前端使用說明（養殖專家）：`docs/使用說明_養殖專家.md`

## 常用 PM2 指令

- 查看狀態：`pm2 status`
- 看 log：`pm2 logs`（或 `pm2 logs backend` / `pm2 logs frontend`）
- 重啟：`pm2 restart backend`、`pm2 restart frontend`
- 停止：`pm2 stop backend`、`pm2 stop frontend`
- 保存開機清單：`pm2 save`
