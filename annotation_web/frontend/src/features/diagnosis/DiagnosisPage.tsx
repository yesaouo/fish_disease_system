import React, { useMemo, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Home, Upload, Stethoscope, Loader2, Download, FolderPlus } from "lucide-react";
import {
  diagnose,
  downloadReportPdf,
  locateTask,
  fetchDatasets,
  fetchGlobalSymptoms,
  fetchTaskSummary
} from "../../api/client";
import type { DiagnoseResponse, DatasetInfo } from "../../api/types";
import { reportToTaskDoc, invertLabelMap } from "../../lib/taskUtils";
import { useAuth } from "../../context/AuthContext";
import { useDataset } from "../../context/DatasetContext";
import ProjectHeader from "../../components/ProjectHeader";

const MODE_LABELS: Record<string, string> = {
  grod_soft: "OAVLE + CEAM",
  grod: "OAVLE（硬性門檻）",
  base: "分離式對照模型"
};

const Card: React.FC<React.PropsWithChildren<{ title: string; sub?: string; className?: string }>> = ({
  title,
  sub,
  className,
  children
}) => (
  <section className={"rounded-xl bg-white p-4 shadow sm:p-6" + (className ? ` ${className}` : "")}>
    <div className="mb-3 flex items-baseline justify-between">
      <h2 className="text-lg font-semibold text-slate-800">{title}</h2>
      {sub && <span className="text-xs text-slate-400">{sub}</span>}
    </div>
    {children}
  </section>
);

const DiagnosisPage: React.FC = () => {
  const navigate = useNavigate();
  const { name, isExpert } = useAuth();
  const { setClasses } = useDataset();

  // 送到資料集編輯（專家）：選可寫資料集或新增 → 帶報告草稿進標註編輯器。
  const [sendOpen, setSendOpen] = useState(false);
  const [writable, setWritable] = useState<DatasetInfo[]>([]);
  const [sendBusy, setSendBusy] = useState(false);
  const [sendErr, setSendErr] = useState<string | null>(null);
  const [newName, setNewName] = useState("");

  // 點相似案例 → 解析來源資料集中的 index → 跳到該案例的標註詳細頁（訪客唯讀可看）。
  const openCase = async (r: DiagnoseResponse["retrieved"][number]) => {
    if (!r.source_dataset || !r.source_task_id) return;
    const index = await locateTask(r.source_dataset, r.source_task_id);
    if (index == null) {
      window.alert("找不到此相似案例的來源標註（可能為健康負樣本）。");
      return;
    }
    setClasses([]); // 讓標註頁依新資料集重新載入類別
    navigate(`/annotate/${r.source_dataset}/${index}`);
  };

  const openSend = async () => {
    setSendErr(null);
    try {
      const all = await fetchDatasets();
      setWritable(all.filter((d) => !d.locked));
    } catch {
      setWritable([]);
    }
    setSendOpen(true);
  };

  // 帶報告草稿進標註編輯器。目標資料集可能尚未建立（提交時才落地），故類別/標籤
  // 一律用全域 symptoms；另外撈前幾個相似案例的整體描述當建議、預設帶最相似的。
  const goToEditor = async (datasetName: string) => {
    if (!mutation.data || !file) return;
    setSendBusy(true);
    setSendErr(null);
    try {
      const g = await fetchGlobalSymptoms();
      const zhToEn = invertLabelMap(g.label_map_zh);
      const doc = reportToTaskDoc(mutation.data, zhToEn);
      doc.dataset = datasetName;

      const cards = mutation.data.retrieved
        .filter((r) => r.source_dataset && r.source_task_id)
        .slice(0, 3);
      const overallSuggestions: { source: string; colloquial: string; medical: string }[] = [];
      for (const r of cards) {
        try {
          const s = await fetchTaskSummary(r.source_dataset!, r.source_task_id!);
          if (s.overall?.colloquial_zh || s.overall?.medical_zh) {
            overallSuggestions.push({
              source: `相似案例 #${r.rank}（${r.source_dataset}）`,
              colloquial: s.overall.colloquial_zh,
              medical: s.overall.medical_zh
            });
          }
        } catch {
          /* 撈不到就略過該案例 */
        }
      }
      if (overallSuggestions.length > 0) {
        doc.overall = {
          colloquial_zh: overallSuggestions[0].colloquial,
          medical_zh: overallSuggestions[0].medical
        };
      }

      setClasses([]);
      navigate(`/annotate/${datasetName}/new`, {
        state: { draft: true, dataset: datasetName, imageFile: file, doc, overallSuggestions }
      });
    } catch {
      setSendErr("開啟編輯器失敗，請稍後再試。");
      setSendBusy(false);
    }
  };

  // 新資料集：不在此即建（提交才落地），僅做名稱格式檢查後進編輯器。
  const createAndGo = async () => {
    const nm = newName.trim();
    if (!nm) return;
    if (!/^[A-Za-z0-9_-]+$/.test(nm)) {
      setSendErr("資料集名稱僅能使用英文、數字、底線(_)、連字號(-)，不可中文或空白。");
      return;
    }
    await goToEditor(nm);
  };

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [text, setText] = useState("");
  const mode = "grod_soft"; // 生產固定模式，不開放切換
  const [topN, setTopN] = useState(6);
  const [topK, setTopK] = useState(3);
  const [selectedCause, setSelectedCause] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const mutation = useMutation<DiagnoseResponse, Error>({
    mutationFn: () => {
      if (!file) throw new Error("尚未選擇影像");
      return diagnose(file, { text, mode, topKCases: topK, topNCauses: topN });
    },
    onSuccess: () => setSelectedCause(0)
  });

  const report = mutation.data;

  const onPickFile = (f: File | null) => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(f);
    setPreviewUrl(f ? URL.createObjectURL(f) : null);
    mutation.reset();
  };

  const cause = useMemo(
    () => report?.causes?.[selectedCause] ?? null,
    [report, selectedCause]
  );

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-4 py-6 sm:px-6 sm:py-10">
      <ProjectHeader />

      <header className="flex items-center justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-semibold text-slate-800 sm:text-2xl">
            <Stethoscope className="h-6 w-6 text-sky-600" /> AI 魚病診斷報告
          </h1>
          <p className="hidden text-sm text-slate-500 sm:block">您好，{name ?? "訪客"}　上傳魚體影像即可產生結構化診斷輔助報告</p>
        </div>
        <button
          type="button"
          onClick={() => navigate("/home")}
          className="inline-flex shrink-0 items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm hover:bg-slate-50 print:hidden"
          title="返回首頁"
          aria-label="返回首頁"
        >
          <Home className="h-4 w-4" /> <span className="hidden sm:inline">返回首頁</span>
        </button>
      </header>

      {/* ===== 輸入區 ===== */}
      <Card title="輸入" sub="影像為必填，文字描述選填" className="print:hidden">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <div>
            <div
              onClick={() => fileInputRef.current?.click()}
              className="flex aspect-video cursor-pointer items-center justify-center overflow-hidden rounded-lg border-2 border-dashed border-slate-300 bg-slate-50 hover:border-sky-400"
            >
              {previewUrl ? (
                <img src={previewUrl} alt="預覽" className="h-full w-full object-contain" />
              ) : (
                <span className="flex flex-col items-center gap-2 text-slate-400">
                  <Upload className="h-8 w-8" /> 點此選擇魚體影像
                </span>
              )}
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => onPickFile(e.target.files?.[0] ?? null)}
            />
          </div>

          <div className="flex flex-col gap-3">
            <label className="text-sm text-slate-600">
              文字描述（選填）
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={3}
                placeholder="例：體表潰瘍、紅腫，疑似感染"
                className="mt-1 w-full rounded border border-slate-300 px-3 py-2 text-sm"
              />
            </label>
            <details className="text-sm text-slate-600">
              <summary className="cursor-pointer text-slate-500">進階設定</summary>
              <div className="mt-2 grid grid-cols-2 gap-3">
                <label className="text-sm text-slate-600">
                  參考案例數量
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                    className="mt-1 w-full rounded border border-slate-300 px-2 py-2 text-sm"
                  />
                </label>
                <label className="text-sm text-slate-600">
                  病因顯示上限
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={topN}
                    onChange={(e) => setTopN(Number(e.target.value))}
                    className="mt-1 w-full rounded border border-slate-300 px-2 py-2 text-sm"
                  />
                </label>
              </div>
            </details>
            <button
              type="button"
              disabled={!file || mutation.isPending}
              onClick={() => mutation.mutate()}
              className="mt-1 flex w-full items-center justify-center gap-2 rounded bg-sky-600 px-4 py-2.5 font-medium text-white hover:bg-sky-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {mutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> 推論中…
                </>
              ) : (
                <>
                  <Stethoscope className="h-4 w-4" /> 產生診斷報告
                </>
              )}
            </button>
            {mutation.isError && (
              <p className="text-sm text-red-600">診斷失敗：{mutation.error.message}</p>
            )}
          </div>
        </div>
      </Card>

      {report && (
        <Report
          report={report}
          previewUrl={previewUrl}
          selectedCause={selectedCause}
          setSelectedCause={setSelectedCause}
          cause={cause}
          canSend={isExpert}
          onSendToDataset={openSend}
          onOpenCase={openCase}
        />
      )}

      {/* 送到資料集編輯（專家）：選現有可寫資料集或新增 */}
      {sendOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-md rounded-xl bg-white p-5 shadow-xl">
            <h3 className="mb-1 text-lg font-semibold text-slate-800">送到資料集編輯</h3>
            <p className="mb-3 text-xs text-slate-500">
              診斷結果會帶入標註編輯器，編輯完成後再提交存入。官方資料集已鎖定，無法選取。
            </p>
            {sendErr && (
              <div className="mb-3 rounded bg-red-50 px-3 py-2 text-sm text-red-600">{sendErr}</div>
            )}
            <div className="mb-3 max-h-48 overflow-y-auto rounded border border-slate-200">
              {writable.length === 0 ? (
                <p className="px-3 py-3 text-sm text-slate-400">尚無可寫資料集，請於下方新增。</p>
              ) : (
                writable.map((d) => (
                  <button
                    key={d.name}
                    disabled={sendBusy}
                    onClick={() => goToEditor(d.name)}
                    className="flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-sky-50 disabled:opacity-50"
                  >
                    <span className="text-slate-700">{d.name}</span>
                  </button>
                ))
              )}
            </div>
            <div className="mb-4 flex gap-2">
              <input
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="新資料集名稱（英數字 _ -）"
                className="flex-1 rounded border border-slate-300 px-3 py-2 text-sm"
              />
              <button
                disabled={sendBusy || !newName.trim()}
                onClick={createAndGo}
                className="inline-flex items-center gap-1 rounded bg-sky-600 px-3 py-2 text-sm text-white hover:bg-sky-700 disabled:bg-sky-300"
              >
                <FolderPlus className="h-4 w-4" /> 新增並編輯
              </button>
            </div>
            <div className="flex justify-end">
              <button
                onClick={() => setSendOpen(false)}
                disabled={sendBusy}
                className="rounded border border-slate-200 px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-100"
              >
                取消
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const Report: React.FC<{
  report: DiagnoseResponse;
  previewUrl: string | null;
  selectedCause: number;
  setSelectedCause: (i: number) => void;
  cause: DiagnoseResponse["causes"][number] | null;
  canSend: boolean;
  onSendToDataset: () => void;
  onOpenCase: (r: DiagnoseResponse["retrieved"][number]) => void;
}> = ({ report, previewUrl, selectedCause, setSelectedCause, cause, canSend, onSendToDataset, onOpenCase }) => {
  const m = report.meta;
  const [bgMode, setBgMode] = useState<"original" | "heatmap">("original");
  const [showBoxes, setShowBoxes] = useState(true);
  const [activeLesion, setActiveLesion] = useState<number | null>(null);
  const toggleLesion = (i: number) => setActiveLesion((p) => (p === i ? null : i));
  const [pdfPending, setPdfPending] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);
  const onDownloadPdf = async () => {
    setPdfPending(true);
    setPdfError(null);
    try {
      await downloadReportPdf(report);
    } catch (e) {
      const status = (e as { response?: { status?: number } })?.response?.status;
      setPdfError(
        status === 404
          ? "報告已過期，請重新產生診斷報告後再下載。"
          : "下載報告失敗，請稍後再試。"
      );
    } finally {
      setPdfPending(false);
    }
  };
  const [W, H] = report.image_size;
  const badge = Math.max(W, H) * 0.05; // box-corner index badge size (image px)
  const stroke = Math.max(W, H) * 0.006;

  return (
    <>
      <div className="flex flex-col items-end gap-1">
        <div className="flex items-center gap-2">
          {canSend && (
            <button
              type="button"
              onClick={onSendToDataset}
              className="inline-flex items-center gap-1 rounded border border-sky-200 bg-sky-50 px-3 py-1.5 text-sm text-sky-700 shadow-sm hover:bg-sky-100"
            >
              <FolderPlus className="h-4 w-4" />
              送到資料集編輯
            </button>
          )}
          <button
            type="button"
            onClick={onDownloadPdf}
            disabled={pdfPending}
            className="inline-flex items-center gap-1 rounded border border-slate-200 bg-white px-3 py-1.5 text-sm text-slate-600 shadow-sm hover:bg-slate-100 disabled:cursor-not-allowed disabled:text-slate-300"
          >
            {pdfPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
            下載診斷報告
          </button>
        </div>
        {pdfError && <p className="text-xs text-red-600">{pdfError}</p>}
      </div>
      {/* 基本資料 */}
      <Card title="基本資料">
        <dl className="grid grid-cols-1 gap-x-6 gap-y-2 text-sm sm:grid-cols-2 md:grid-cols-4">
          <Field k="病例編號" v={m.case_id} />
          <Field k="報告時間" v={m.timestamp.replace("T", " ")} />
          <Field k="分析模式" v={MODE_LABELS[m.mode] ?? m.mode} />
          <Field k="病灶數量" v={String(report.n_lesions)} />

          <div className="sm:col-span-2 md:col-span-4">
            <Field k="補充描述" v={m.text || "未提供"} />
          </div>

          <div className="sm:col-span-2 md:col-span-4">
            <Field
              k="判定結果"
              v={report.abstain ? "健康（未偵測到明顯異常）" : "疑似異常，進行病因分析"}
            />
          </div>
        </dl>
      </Card>

      {report.abstain ? (
        <Card title="判定結果">
          <p className="rounded-lg bg-emerald-50 px-4 py-3 text-emerald-700">
            🟢 系統未在魚體表面偵測到明顯異常，判定為<strong>健康</strong>，未進行病因分析。
          </p>
          {previewUrl && (
            <figure className="mt-4">
              <img src={previewUrl} alt="送檢影像" className="w-full rounded-lg" />
              <figcaption className="mt-1 text-center text-xs text-slate-400">送檢影像</figcaption>
            </figure>
          )}
        </Card>
      ) : (
        <div className="flex flex-col gap-6">
          {/* 區段一：病灶定位（左，較窄）＋ 病灶詳細（右，響應式網格） */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)] lg:items-start">
            <Card title="① 病灶定位" sub={`${report.n_lesions} 個病灶`}>
              <div className="mb-2 flex gap-2">
                <button
                  type="button"
                  onClick={() => setShowBoxes((v) => !v)}
                  className={
                    "rounded-full px-3 py-1 text-xs " +
                    (showBoxes ? "bg-sky-600 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200")
                  }
                >
                  顯示病灶框
                </button>
                <button
                  type="button"
                  onClick={() => setBgMode(bgMode === "heatmap" ? "original" : "heatmap")}
                  className={
                    "rounded-full px-3 py-1 text-xs " +
                    (bgMode === "heatmap" ? "bg-sky-600 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200")
                  }
                >
                  顯示異常熱力圖
                </button>
              </div>
              <div className="relative w-full overflow-hidden rounded-lg">
                <img
                  src={bgMode === "heatmap" ? report.heatmap : previewUrl ?? report.heatmap}
                  alt="病灶定位"
                  className="block w-full"
                />
                {showBoxes && (
                  <svg
                    viewBox={`0 0 ${W} ${H}`}
                    preserveAspectRatio="none"
                    className="absolute inset-0 h-full w-full"
                  >
                    {report.lesions.map((l) => {
                      const [x, y, w, h] = l.bbox_xywh;
                      const on = activeLesion === l.idx;
                      const color = on ? "#f59e0b" : "#dc2626";
                      return (
                        <g key={l.idx} onClick={() => toggleLesion(l.idx)} style={{ cursor: "pointer" }}>
                          <rect x={x} y={y} width={w} height={h} fill="transparent" />
                          <rect
                            x={x}
                            y={y}
                            width={w}
                            height={h}
                            fill="none"
                            stroke={color}
                            strokeWidth={on ? stroke * 1.8 : stroke}
                          />
                          <rect x={x} y={y} width={badge} height={badge} rx={badge * 0.15} fill={color} />
                          <text
                            x={x + badge / 2}
                            y={y + badge * 0.72}
                            textAnchor="middle"
                            fontSize={badge * 0.6}
                            fontWeight="bold"
                            fill="#fff"
                          >
                            {l.idx + 1}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                )}
              </div>
            </Card>

            <Card title="病灶詳細" sub="點卡片或影像框互相對應">
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 2xl:grid-cols-3">
                {report.lesions.map((l) => (
                  <LesionCard
                    key={l.idx}
                    l={l}
                    active={activeLesion === l.idx}
                    onClick={() => toggleLesion(l.idx)}
                  />
                ))}
              </div>
            </Card>
          </div>

          {/* 區段二：病因排序（左）＋ 病因詳細（右，含來源相似案例） */}
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            <Card title="② 疑似病因" sub="點選病因查看詳細">
              <ul className="flex min-w-0 flex-col gap-2">
                {report.causes.map((c, i) => (
                  <li key={c.rank} className="min-w-0">
                    <button
                      type="button"
                      onClick={() => setSelectedCause(i === selectedCause ? -1 : i)}
                      className={
                        "w-full rounded-lg border px-3 py-2 text-left text-sm " +
                        (i === selectedCause
                          ? "border-sky-500 bg-sky-50"
                          : "border-slate-200 hover:bg-slate-50")
                      }
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-medium text-slate-700">#{c.rank}</span>
                        <span className="text-xs text-slate-400">
                          s={c.score.toFixed(2)}
                          {c.support != null ? `　${c.support} 例` : ""}
                        </span>
                      </div>
                      <p className="mt-0.5 line-clamp-2 text-slate-600">{c.text}</p>
                    </button>
                    {/* 手機：手風琴就地展開詳細（電腦改用右欄） */}
                    {i === selectedCause && (
                      <div className="mt-2 md:hidden">
                        <CauseDetail cause={c} retrieved={report.retrieved} onOpenCase={onOpenCase} />
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            </Card>

            {/* 電腦：右側詳細欄 */}
            <Card title="病因詳細" sub="各證據貢獻" className="hidden md:block">
              {cause ? (
                <CauseDetail cause={cause} retrieved={report.retrieved} onOpenCase={onOpenCase} />
              ) : (
                <p className="py-2 text-sm text-slate-400">點左側病因，查看各證據的貢獻。</p>
              )}
            </Card>
          </div>

        </div>
      )}

      {/* 模組參數 + 延遲 */}
      <Card title="系統資訊" sub="模組參數量 · 各階段延遲">
        <details>
          <summary className="cursor-pointer text-sm text-slate-500">展開模組參數與推論延遲</summary>
          <div className="mt-3 grid gap-6 md:grid-cols-2">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-slate-400">
                  <th className="py-1">模組</th>
                  <th className="py-1 text-right">參數量</th>
                </tr>
              </thead>
              <tbody>
                {report.params.modules.map((p) => (
                  <tr key={p.name} className="border-t border-slate-100">
                    <td className="py-1 text-slate-600">{p.name}</td>
                    <td className="py-1 text-right text-slate-600">{p.count.toLocaleString()}</td>
                  </tr>
                ))}
                <tr className="border-t border-slate-200 font-semibold">
                  <td className="py-1">總計</td>
                  <td className="py-1 text-right">{(report.params.total / 1e6).toFixed(1)}M</td>
                </tr>
              </tbody>
            </table>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-slate-400">
                  <th className="py-1">階段</th>
                  <th className="py-1 text-right">ms</th>
                </tr>
              </thead>
              <tbody>
                {report.timings.map((t) => (
                  <tr key={t.stage} className="border-t border-slate-100">
                    <td className="py-1 text-slate-600">{t.stage}</td>
                    <td className="py-1 text-right text-slate-600">{t.ms.toFixed(2)}</td>
                  </tr>
                ))}
                <tr className="border-t border-slate-200 font-semibold">
                  <td className="py-1">總計</td>
                  <td className="py-1 text-right">
                    {report.timings.reduce((s, t) => s + t.ms, 0).toFixed(2)}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </details>
      </Card>
    </>
  );
};

const CauseDetail: React.FC<{
  cause: DiagnoseResponse["causes"][number];
  retrieved: DiagnoseResponse["retrieved"];
  onOpenCase: (r: DiagnoseResponse["retrieved"][number]) => void;
}> = ({ cause, retrieved, onOpenCase }) => {
  const supportCases = cause.support_cases
    .map((rk) => retrieved.find((r) => r.rank === rk))
    .filter((r): r is DiagnoseResponse["retrieved"][number] => !!r);
  return (
    <div className="flex flex-col gap-3">
      <div className="rounded-lg bg-slate-50 px-4 py-3 text-sm">
        <p className="font-semibold text-slate-800">Top-{cause.rank} 病因</p>
        <p className="mt-1 text-slate-700">{cause.text}</p>
        <p className="mt-2 text-xs text-slate-500">
          AI 評分：{cause.score.toFixed(3)}
          {cause.support != null && `　·　${cause.support} 個相似案例支持`}
        </p>
      </div>
      {supportCases.length > 0 && (
        <div className="min-w-0">
          <p className="mb-1 text-xs text-slate-400">此病因來自以下相似案例（點圖可查看原始標註）</p>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {supportCases.map((r) => {
              const linkable = !!(r.source_dataset && r.source_task_id);
              return (
                <figure
                  key={r.rank}
                  onClick={linkable ? () => onOpenCase(r) : undefined}
                  className={
                    "w-28 shrink-0 overflow-hidden rounded-lg border border-slate-100" +
                    (linkable ? " cursor-pointer hover:border-sky-400 hover:shadow" : "")
                  }
                  title={linkable ? `查看 ${r.source_dataset} 的原始標註` : undefined}
                >
                  <img src={r.image} alt={r.file_name} className="h-20 w-28 object-cover" />
                  <figcaption className="px-1 py-0.5 text-center text-[10px] text-slate-500">
                    相似度 {r.similarity.toFixed(3)}
                  </figcaption>
                </figure>
              );
            })}
          </div>
        </div>
      )}
      <div>
        <p className="mb-1 text-xs text-slate-400">各證據對此病因的貢獻</p>
        <img src={cause.breakdown} alt="α breakdown" className="w-full rounded-lg" />
      </div>
    </div>
  );
};

const LesionCard: React.FC<{
  l: DiagnoseResponse["lesions"][number];
  active?: boolean;
  onClick?: () => void;
}> = ({ l, active, onClick }) => (
  <div
    onClick={onClick}
    className={
      "rounded-lg border p-3 transition " +
      (onClick ? "cursor-pointer hover:bg-slate-50 " : "") +
      (active ? "border-amber-400 bg-amber-50 ring-1 ring-amber-300" : "border-slate-200")
    }
  >
    <div className="flex gap-3">
      <img src={l.crop} alt={`病灶 L${l.idx + 1}`} className="h-20 w-20 shrink-0 rounded object-cover" />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="rounded bg-red-600 px-1.5 text-xs font-bold text-white">L{l.idx + 1}</span>
          <span className="truncate font-medium text-slate-700">{l.label_zh}</span>
        </div>
        <p className="mt-0.5 text-xs text-slate-400">異常程度 {l.det_score.toFixed(2)}</p>
      </div>
    </div>
    <ul className="mt-2 space-y-1">
      {l.top_k.map((it, i) => (
        <li key={i} className="flex items-center gap-2 text-xs">
          <span className="w-20 shrink-0 truncate text-slate-500">{it.label_zh}</span>
          <span className="h-1.5 flex-1 rounded bg-slate-100">
            <span
              className="block h-full rounded bg-sky-400"
              style={{ width: `${Math.max(0, it.prob) * 100}%` }}
            />
          </span>
          <span className="w-10 shrink-0 text-right text-slate-400">
            {(it.prob * 100).toFixed(0)}%
          </span>
        </li>
      ))}
    </ul>
  </div>
);

const Field: React.FC<{ k: string; v: string }> = ({ k, v }) => (
  <div>
    <dt className="text-xs text-slate-400">{k}</dt>
    <dd className="text-slate-700">{v}</dd>
  </div>
);

export default DiagnosisPage;
