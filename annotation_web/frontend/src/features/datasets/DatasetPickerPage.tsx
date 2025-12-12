import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import {
  fetchClasses,
  fetchDatasets,
  fetchAdminStats
} from "../../api/client";
import { useDataset } from "../../context/DatasetContext";
import { useAuth } from "../../context/AuthContext";
import ProjectHeader from "../../components/ProjectHeader";
import React, { useState } from "react";

// Icons
import { Trash2, SquarePlus, Save, Undo2, Redo2, BarChart3, LogOut } from "lucide-react";

// ===== 小元件：Kbd / IconButton / Separator =====
const Kbd: React.FC<React.PropsWithChildren> = ({ children }) => (
  <kbd className="rounded border border-slate-300 bg-white px-1.5 text-[10px] font-medium text-slate-700 shadow-sm">
    {children}
  </kbd>
);

const DatasetPickerPage: React.FC = () => {
  const navigate = useNavigate();
  const { dataset, setDataset, setClasses } = useDataset();
  const { name, logout } = useAuth();

  const { data: datasets, isLoading, isError } = useQuery({
    queryKey: ["datasets"],
    queryFn: fetchDatasets
  });

  // Fetch aggregated stats for all datasets to show completed/total
  const { data: adminStats } = useQuery({
    queryKey: ["adminStats"],
    queryFn: fetchAdminStats
  });

  const handleSelect = async (selected: string) => {
    if (!selected) return;
    setDataset(selected);
    try {
      const classes = await fetchClasses(selected);
      setClasses(classes);
      navigate("/annotate", { replace: true });
    } catch (err) {
      console.error(err);
    }
  };

  // 免責聲明：顯示/隱藏
  const [showDisclaimer, setShowDisclaimer] = useState(true);
  const [expanded, setExpanded] = useState(true);

  return (
    <div className="mx-auto flex min-h-screen max-w-5xl flex-col gap-6 px-6 py-10">
      <ProjectHeader />

      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">資料集選擇</h1>
          <p className="flex items-center gap-2 text-sm text-slate-500">
            <span>您好，{name ?? "訪客"}</span>
            {/* 前往後台（AdminDashboard） */}
            <button
              type="button"
              onClick={() => navigate("/admin")}
              className="inline-flex h-6 w-6 items-center justify-center rounded text-slate-500 hover:bg-slate-100 hover:text-slate-700"
              title="提交概況"
              aria-label="提交概況"
            >
              <BarChart3 className="h-4 w-4" aria-hidden="true" />
            </button>
            {/* 登出並回到登入畫面 */}
            <button
              type="button"
              onClick={() => {
                // 清除登入與資料集，回到登入頁
                logout();
                setDataset(null);
                navigate("/login", { replace: true });
              }}
              className="inline-flex h-6 w-6 items-center justify-center rounded text-slate-500 hover:bg-slate-100 hover:text-slate-700"
              title="我要登出"
              aria-label="我要登出"
            >
              <LogOut className="h-4 w-4" aria-hidden="true" />
            </button>
          </p>
        </div>
        {dataset && (
          <span className="rounded-full bg-sky-100 px-4 py-1 text-sm text-sky-700">
            目前：{dataset}
          </span>
        )}
      </header>

      {/* 免責聲明卡片 */}
      {showDisclaimer && (
        <aside className="rounded-xl border border-amber-200 bg-amber-50 shadow">
          <div className="flex items-start gap-3 p-4">
            {/* 小圖示（不需額外套件） */}
            <svg
              className="mt-1 h-5 w-5 flex-shrink-0 text-amber-600"
              viewBox="0 0 20 20"
              fill="currentColor"
              aria-hidden="true"
            >
              <path
                fillRule="evenodd"
                d="M8.257 3.099c.765-1.36 2.72-1.36 3.485 0l6.518 11.592c.75 1.335-.213 2.989-1.742 2.989H3.48c-1.53 0-2.492-1.654-1.743-2.989L8.257 3.1zM11 14a1 1 0 10-2 0 1 1 0 002 0zm-1-2a.75.75 0 01-.75-.75v-3.5a.75.75 0 011.5 0v3.5A.75.75 0 0110 12z"
                clipRule="evenodd"
              />
            </svg>

            <div className="flex-1">
              <div className="flex items-center justify-between">
                <h2 className="text-base font-semibold text-amber-900">免責聲明</h2>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setExpanded((v) => !v)}
                    className="rounded px-2 py-1 text-xs text-amber-800 hover:bg-amber-100"
                    aria-expanded={expanded}
                    aria-controls="disclaimer-content"
                  >
                    {expanded ? "收合" : "展開"}
                  </button>
                  <button
                    onClick={() => setShowDisclaimer(false)}
                    className="rounded px-2 py-1 text-xs text-amber-800 hover:bg-amber-100"
                    aria-label="關閉免責聲明"
                  >
                    關閉
                  </button>
                </div>
              </div>

              {expanded && (
                <div id="disclaimer-content" className="mt-2 text-sm leading-6 text-amber-900">
                  <p className="mb-2">
                    本平台「說明文字」與圖像範例主要整理自公開資料庫之體表病徵影像與標註，並輔以生成式
                    AI 進行初步判讀與摘要，<span className="font-semibold">非經獸醫師或水產專業人員臨床診斷之醫療建議</span>。
                  </p>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>
                      影像來源（Roboflow Universe）：
                      <a
                        href="https://universe.roboflow.com/iot-uiah0/fish-disease-detection-n4fho"
                        target="_blank"
                        rel="noreferrer"
                        className="text-amber-800 underline underline-offset-2"
                      >
                        Fish Disease Detection
                      </a>{" "}
                      、{" "}
                      <a
                        href="https://universe.roboflow.com/healthy-ioj2n/fish-disease-fqkyc"
                        target="_blank"
                        rel="noreferrer"
                        className="text-amber-800 underline underline-offset-2"
                      >
                        Fish Disease
                      </a>{" "}
                      、{" "}
                      <a
                        href="https://universe.roboflow.com/fishv41/fish-disease-9911"
                        target="_blank"
                        rel="noreferrer"
                        className="text-amber-800 underline underline-offset-2"
                      >
                        Fish Disease 9911
                      </a>
                      。
                    </li>
                    <li>
                      <span className="font-semibold">資料集僅供 AI 模型建置與學術研究使用</span>，
                      不得作為實際魚病診斷之最終依據或治療決策。
                    </li>
                    <li>
                      若有健康疑慮或緊急情況，請洽具資格之專業人員（例如：獸醫師／水產專家）。
                    </li>
                  </ul>
                </div>
              )}
            </div>
          </div>
        </aside>
      )}

      <div className="rounded-xl bg-white p-6 shadow">
        {isLoading && <p className="text-slate-500">載入中...</p>}
        {isError && (
          <p className="text-red-600">載入資料集失敗，請稍後再試</p>
        )}
        {datasets && datasets.length === 0 && (
          <p className="text-slate-500">目前沒有可用的資料集</p>
        )}
        {datasets && datasets.length > 0 && (
          <ul className="grid gap-3 md:grid-cols-2">
            {datasets.map((item) => {
              const stats = adminStats?.datasets?.find((s) => s.dataset === item);
              const completed = stats?.expert_completed_tasks;
              const total = stats?.total_tasks;
              const label =
                typeof completed === "number" && typeof total === "number"
                  ? `${completed}/${total}`
                  : "—/—";
              return (
                <li key={item}>
                  <button
                    onClick={() => handleSelect(item)}
                    className="flex w-full items-center justify-between rounded border border-slate-200 px-4 py-3 text-left hover:border-sky-500 hover:bg-sky-50"
                    title="選擇資料集"
                  >
                    <span className="font-medium text-slate-700">{item}</span>
                    <span className="text-sm text-slate-400" title="標註完成/圖片數量">
                      {label}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <section className="rounded-xl bg-white p-6 shadow">
        <h2 className="mb-3 text-lg font-semibold text-slate-800">影像標註 — 操作SOP</h2>

        <ol className="list-decimal space-y-3 pl-5 text-sm text-slate-700">
          <li>
            <span className="font-semibold">檢查標註框</span>
            ：將每個框調整至正確位置與大小。框可拖曳移動、拉角縮放；但不能旋轉、不能超出影像範圍、不能小於最小尺寸。
          </li>

          <li>
            <span className="font-semibold">刪除多餘框</span>
            ：先點選目標框，再點工具列上的
            <Trash2 className="mx-1 inline h-4 w-4 align-[-2px]" aria-hidden="true" />
            圖示，或按 Del / Backspace。
          </li>

          <li>
            <span className="font-semibold">新增遺漏框</span>
            ：點工具列上的
            <SquarePlus className="mx-1 inline h-4 w-4 align-[-2px]" aria-hidden="true" />
            圖示，或按 <Kbd>N</Kbd>，再拉出正確位置與大小。新增時會沿用上一個框的類別與外觀敘述（如有），可再調整。
          </li>

          <li>
            <span className="font-semibold">確認類別與外觀敘述（選填）</span>
            ：於右側「標註內容」面板選擇表徵類別，並可填寫「外觀敘述」（單行文字）。
            必要欄位未填寫時，系統會以紅字提示。
          </li>

          <li>
            <span className="font-semibold">填寫病徵敘述</span>
            ：
            <ul className="mt-2 list-disc pl-4">
              <li>「通俗描述」：一般人可理解的描述。</li>
              <li>「醫學描述」：依醫學術語填寫。</li>
            </ul>
            以上欄位按 Enter 不會換行（僅接受單行）。
          </li>

          <li>
            <span className="font-semibold">填寫病徵原因與處置建議</span>
            ：
            <ul className="mt-2 list-disc pl-4">
              <li>「病徵原因」：依可能性高低排序（最多 10 項）。</li>
              <li>「處置建議」：依治療流程排序（最多 10 項）。</li>
            </ul>
            可按 Enter 新增項目，並可上移／下移或刪除。
          </li>

          <li>
            <span className="font-semibold">註解</span>
            ：在「註解」區輸入文字後按 Enter 或點「新增」即可新增一則註解；可個別移除。
          </li>

          <li>
            <span className="font-semibold">保存</span>
            ：點工具列上的
            <Save className="mx-1 inline h-4 w-4 align-[-2px]" aria-hidden="true" />
            圖示，或按 <Kbd>Ctrl</Kbd> + <Kbd>S</Kbd> 暫存。保存後會更新目前進度為基準，之後離開本頁不再跳出「未保存」提醒。
          </li>

          <li>
            <span className="font-semibold">資料驗證與錯誤定位（提交時）</span>
            ：若有缺漏，頁面上方會顯示警示條並提供「定位到第一個錯誤」。右側「標註清單」中有問題的框會以紅色標示並顯示錯誤數。
          </li>

          <li>
            <span className="font-semibold">切換影像</span>
            ：使用上方膠囊輸入列的「上一個／下一個」箭頭，或輸入編號後按「前往」。提交或跳過後會自動載入下一張影像。
          </li>

          <li>
            <span className="font-semibold">跳過與提交</span>
            ：
            <ul className="mt-2 list-disc pl-4">
              <li><span className="font-medium">跳過</span>：點擊「跳過」按鈕，可略過此影像（僅記錄於日誌，不影響派發）。若有未保存變更會先提示確認。</li>
              <li><span className="font-medium">提交</span>：點擊「提交」按鈕（會跳出確認視窗）。提交成功後，此影像將不再重新分派並自動前往下一張。</li>
            </ul>
          </li>

          <li>
            <span className="font-semibold">復原與重做</span>
            ：點工具列上的
            <Undo2 className="mx-1 inline h-4 w-4 align-[-2px]" aria-hidden="true" />
            /
            <Redo2 className="mx-1 inline h-4 w-4 align-[-2px]" aria-hidden="true" />
            圖示，或使用 <Kbd>Ctrl</Kbd> + <Kbd>Z</Kbd>、<Kbd>Ctrl</Kbd> + <Kbd>Y</Kbd>。
          </li>

          <li>
            <span className="font-semibold">未保存離開提醒</span>
            ：若存在未保存變更，直接離開或導向其他頁面時會跳出提醒；「提交」與「跳過」流程會暫時放行導頁。
          </li>
        </ol>

        <div className="mt-4 rounded border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
          <span className="font-medium">快捷鍵小抄：</span>
          <SquarePlus className="mx-1 inline h-3.5 w-3.5 align-[-2px]" aria-hidden="true" /> 新增框（N）；
          <Trash2 className="mx-1 inline h-3.5 w-3.5 align-[-2px]" aria-hidden="true" /> 刪除框（Del/Backspace）；
          <Undo2 className="mx-1 inline h-3.5 w-3.5 align-[-2px]" aria-hidden="true" /> 復原（Ctrl+Z）；
          <Redo2 className="mx-1 inline h-3.5 w-3.5 align-[-2px]" aria-hidden="true" /> 重做（Ctrl+Y）；
          <Save className="mx-1 inline h-3.5 w-3.5 align-[-2px]" aria-hidden="true" /> 保存（Ctrl+S）。
        </div>

        {/* 小提示：滑鼠懸停圖示可見完整工具提示（含快捷鍵），亦可用鍵盤操作。 */}
        <p className="mt-2 text-xs text-slate-500">
          提示：工具列圖示支援滑鼠懸停顯示名稱與快捷鍵，亦支援鍵盤操作。
        </p>
      </section>

    </div>
  );
};

export default DatasetPickerPage;
