const path = require("path");

// conda info --base
const CONDA_BASE = "/home/lab603/anaconda3";  // 改成你的 conda base
const CONDA_ENV  = "annotation_web";  // 固定 conda env 名稱
const LOG_ROOT   = "/home/lab603/桌面/YJ/fish_disease_system/data/annotation/logs"; // 固定 log 目錄
const PYTHON = path.join(CONDA_BASE, "envs", CONDA_ENV, "bin", "python");

module.exports = {
  apps: [
    {
      name: "backend",
      cwd: "./backend",
      script: PYTHON,
      args: `-m uvicorn app.main:app --host 0.0.0.0 --port 5174`,
      watch: false,

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
      script: "npm",
      args: "run dev -- --host 0.0.0.0 --port 5173",
      watch: false,

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