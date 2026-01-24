const path = require("path");

// conda info --base
const CONDA_BASE = "/opt/miniconda3";           // 改成你的 conda base
const CONDA_ENV  = "annotation_web";            // 固定 conda env 名稱
const LOG_ROOT   = "/opt/data/annotation/logs"; // 固定 log 目錄

// conda 可執行檔路徑
const CONDA =
  process.platform === "win32"
    ? path.join(CONDA_BASE, "Scripts", "conda.exe")
    : path.join(CONDA_BASE, "bin", "conda");

module.exports = {
  apps: [
    {
      name: "backend",
      cwd: "./backend",
      script: CONDA,
      args: `run -n ${CONDA_ENV} python -m uvicorn app.main:app --host 0.0.0.0 --port 5174`,
      watch: false,
      env: { NODE_ENV: "production" },

      out_file: path.join(LOG_ROOT, "backend.log"),
      error_file: path.join(LOG_ROOT, "backend.err.log"),
      merge_logs: true,
      max_memory_restart: "512M",
      instances: 1,
      exec_mode: "fork",
      autorestart: true,
    },
    {
      name: "frontend",
      cwd: "./frontend",
      script: "node",
      args: "./node_modules/vite/bin/vite.js --host 0.0.0.0 --port 5173",
      watch: false,
      env: { NODE_ENV: "production" },

      out_file: path.join(LOG_ROOT, "frontend.log"),
      error_file: path.join(LOG_ROOT, "frontend.err.log"),
      merge_logs: true,
      max_memory_restart: "512M",
      instances: 1,
      exec_mode: "fork",
      autorestart: true,
    },
  ],
};