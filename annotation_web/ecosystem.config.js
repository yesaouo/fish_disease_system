module.exports = {
  apps: [
    {
      name: "backend",
      cwd: "./backend",
      // 用 uvicorn 啟動 FastAPI
      script: "uvicorn",
      // 將原本參數搬過來（是否用 --reload 視你的需求，見下方說明）
      args: "app.main:app --host 0.0.0.0 --port 5174 --reload",
      interpreter: "python3", // 或填入絕對路徑，例如 /usr/bin/python3
      watch: false,           // 若想由 PM2 監控檔案變動可改為 true（記得搭配 ignore_watch）
      env: {
        // 在這裡放後端需要的環境變數
        // NODE_ENV: "development"
      },
      out_file: "backend.log",    // 與你原本相同的檔名
      error_file: "backend.err.log",
      merge_logs: true,
      max_memory_restart: "512M",
      instances: 1,
      exec_mode: "fork"
    },
    {
      name: "frontend",
      cwd: "./frontend",
      // 直接用專案內的 Vite 執行檔
      script: "node",
      args: "./node_modules/vite/bin/vite.js --host --port 5173",
      watch: false,        // dev server 自己會熱更新，通常不需要 PM2 watch
      env: {
        // 前端需要的環境變數放這裡
      },
      out_file: "frontend.log",
      error_file: "frontend.err.log",
      merge_logs: true,
      max_memory_restart: "512M",
      instances: 1,
      exec_mode: "fork"
    }
  ]
}
