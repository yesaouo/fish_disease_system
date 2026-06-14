import React, { useMemo, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Upload, Stethoscope, Loader2 } from "lucide-react";
import { diagnose } from "../../api/client";
import type { DiagnoseResponse } from "../../api/types";
import { useAuth } from "../../context/AuthContext";
import ProjectHeader from "../../components/ProjectHeader";

const MODE_LABELS: Record<string, string> = {
  grod_soft: "GROD（soft gate）",
  grod: "GROD（hard gate）",
  base: "分離式對照組 base"
};

const Card: React.FC<React.PropsWithChildren<{ title: string; sub?: string }>> = ({
  title,
  sub,
  children
}) => (
  <section className="rounded-xl bg-white p-6 shadow">
    <div className="mb-3 flex items-baseline justify-between">
      <h2 className="text-lg font-semibold text-slate-800">{title}</h2>
      {sub && <span className="text-xs text-slate-400">{sub}</span>}
    </div>
    {children}
  </section>
);

const DiagnosisPage: React.FC = () => {
  const navigate = useNavigate();
  const { name } = useAuth();

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [mode, setMode] = useState("grod_soft");
  const [topN, setTopN] = useState(5);
  const [topK, setTopK] = useState(20);
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
    () => (report && report.causes.length > 0 ? report.causes[selectedCause] : null),
    [report, selectedCause]
  );

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-6 py-10">
      <ProjectHeader />

      <header className="flex items-center justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-semibold text-slate-800">
            <Stethoscope className="h-6 w-6 text-sky-600" /> AI 魚病診斷報告
          </h1>
          <p className="text-sm text-slate-500">您好，{name ?? "訪客"}　上傳魚體影像即可產生結構化診斷輔助報告</p>
        </div>
        <button
          type="button"
          onClick={() => navigate("/datasets")}
          className="inline-flex items-center gap-1 rounded border border-slate-200 px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-100"
        >
          <ArrowLeft className="h-4 w-4" /> 返回
        </button>
      </header>

      {/* ===== 輸入區 ===== */}
      <Card title="輸入" sub="影像為必填，文字描述選填">
        <div className="grid gap-6 md:grid-cols-2">
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
              文字描述（選填，作為 CEAH 文字證據）
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={3}
                placeholder="例：體表潰瘍、紅腫，疑似感染；留空＝vision-only"
                className="mt-1 w-full rounded border border-slate-300 px-3 py-2 text-sm"
              />
            </label>
            <div className="grid grid-cols-3 gap-3">
              <label className="text-sm text-slate-600">
                模式
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-300 px-2 py-2 text-sm"
                >
                  {Object.entries(MODE_LABELS).map(([k, v]) => (
                    <option key={k} value={k}>
                      {v}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-sm text-slate-600">
                top_n 病因
                <input
                  type="number"
                  min={1}
                  max={10}
                  value={topN}
                  onChange={(e) => setTopN(Number(e.target.value))}
                  className="mt-1 w-full rounded border border-slate-300 px-2 py-2 text-sm"
                />
              </label>
              <label className="text-sm text-slate-600">
                top_k 案例
                <input
                  type="number"
                  min={5}
                  max={50}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="mt-1 w-full rounded border border-slate-300 px-2 py-2 text-sm"
                />
              </label>
            </div>
            <button
              type="button"
              disabled={!file || mutation.isPending}
              onClick={() => mutation.mutate()}
              className="mt-1 inline-flex items-center justify-center gap-2 rounded bg-sky-600 px-4 py-2.5 font-medium text-white hover:bg-sky-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {mutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> 推論中…（首次載入模型較久）
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
        />
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
}> = ({ report, previewUrl, selectedCause, setSelectedCause, cause }) => {
  const m = report.meta;
  const [bgMode, setBgMode] = useState<"original" | "heatmap">("original");
  const [W, H] = report.image_size;
  const badge = Math.max(W, H) * 0.05; // box-corner index badge size (image px)
  const stroke = Math.max(W, H) * 0.006;
  return (
    <>
      {/* 基本資料 */}
      <Card title="基本資料">
        <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm md:grid-cols-4">
          <Field k="影像編號" v={m.case_id} />
          <Field k="上傳時間" v={m.timestamp.replace("T", " ")} />
          <Field k="模式" v={MODE_LABELS[m.mode] ?? m.mode} />
          <Field k="候選病因池" v={String(report.pool_size)} />
          <div className="col-span-2 md:col-span-4">
            <Field k="使用者描述" v={m.text || "（未填，vision-only）"} />
          </div>
        </dl>
      </Card>

      {report.abstain ? (
        <Card title="判定結果">
          <p className="rounded-lg bg-emerald-50 px-4 py-3 text-emerald-700">
            🟢 最高 objectness 未達 abstain 門檻（{m.thresholds.abstain.toFixed(2)}），判定為<strong>健康</strong>，不進行病因推論。
          </p>
        </Card>
      ) : (
        <>
          {/* ① 病灶定位與分析（熱力圖 ⇄ 原圖 + 框 + 卡片合併） */}
          <Card title="① 病灶定位與分析" sub={`${report.n_lesions} 個病灶 · z·anchor 分類`}>
            {/* 空間總覽：底圖可切換，病灶框為 SVG overlay */}
            <div className="mb-2 flex gap-2">
              {(["original", "heatmap"] as const).map((b) => (
                <button
                  key={b}
                  type="button"
                  onClick={() => setBgMode(b)}
                  className={
                    "rounded-full px-3 py-1 text-xs " +
                    (bgMode === b
                      ? "bg-sky-600 text-white"
                      : "bg-slate-100 text-slate-500 hover:bg-slate-200")
                  }
                >
                  {b === "original" ? "原圖" : "異常熱力圖"}
                </button>
              ))}
            </div>
            <div className="relative w-full overflow-hidden rounded-lg">
              <img
                src={bgMode === "heatmap" ? report.heatmap : previewUrl ?? report.heatmap}
                alt="病灶定位"
                className="block w-full"
              />
              <svg
                viewBox={`0 0 ${W} ${H}`}
                preserveAspectRatio="none"
                className="pointer-events-none absolute inset-0 h-full w-full"
              >
                {report.lesions.map((l) => {
                  const [x, y, w, h] = l.bbox_xywh;
                  return (
                    <g key={l.idx}>
                      <rect
                        x={x}
                        y={y}
                        width={w}
                        height={h}
                        fill="none"
                        stroke="#dc2626"
                        strokeWidth={stroke}
                      />
                      <rect x={x} y={y} width={badge} height={badge} rx={badge * 0.15} fill="#dc2626" />
                      <text
                        x={x + badge / 2}
                        y={y + badge * 0.72}
                        textAnchor="middle"
                        fontSize={badge * 0.6}
                        fontWeight="bold"
                        fill="#fff"
                      >
                        {l.idx}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
            <p className="mt-1 text-xs text-slate-400">
              框＝顯示門檻內的病灶；熱力圖＝全 query objectness 場（含門檻外訊號）。
            </p>

            {/* 病灶卡片（原生）：crop + top-k 症狀 + 信心 */}
            <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {report.lesions.map((l) => (
                <LesionCard key={l.idx} l={l} />
              ))}
            </div>
          </Card>

          {/* ② 相似案例 */}
          <Card title="② 相似案例" sub={`Top-${report.retrieved.length} 檢索`}>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-5">
              {report.retrieved.map((r) => (
                <figure key={r.rank} className="overflow-hidden rounded-lg border border-slate-100">
                  <img src={r.image} alt={r.file_name} className="h-28 w-full object-cover" />
                  <figcaption className="px-2 py-1 text-center text-xs text-slate-500">
                    #{r.rank}　sim={r.similarity.toFixed(3)}
                  </figcaption>
                </figure>
              ))}
            </div>
          </Card>

          {/* ③ 疑似病因 + 證據歸因 */}
          <Card title="③ 疑似病因排序 + α 證據歸因" sub="點選病因查看歸因">
            <div className="grid gap-5 md:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
              <ul className="flex flex-col gap-2">
                {report.causes.map((c, i) => (
                  <li key={c.rank}>
                    <button
                      type="button"
                      onClick={() => setSelectedCause(i)}
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
                  </li>
                ))}
              </ul>

              {cause && (
                <div className="flex flex-col gap-3">
                  <div className="rounded-lg bg-slate-50 px-4 py-3 text-sm">
                    <p className="font-semibold text-slate-800">Top-{cause.rank} 病因</p>
                    <p className="mt-1 text-slate-700">{cause.text}</p>
                    <p className="mt-2 text-xs text-slate-500">
                      CEAH score：{cause.score.toFixed(3)}
                      {cause.support != null && `　·　支持度：${cause.support} 個相似病例`}
                    </p>
                    {cause.members.length > 1 && (
                      <details className="mt-2 text-xs text-slate-500">
                        <summary className="cursor-pointer">已聚合 {cause.members.length - 1} 條相近病因</summary>
                        <ul className="mt-1 list-disc pl-5">
                          {cause.members.slice(1, 6).map((mm, j) => (
                            <li key={j}>{mm}</li>
                          ))}
                        </ul>
                      </details>
                    )}
                  </div>
                  <img src={cause.attribution} alt="α attribution" className="w-full rounded-lg" />
                  <img src={cause.breakdown} alt="α breakdown" className="w-full rounded-lg" />
                </div>
              )}
            </div>
          </Card>

          {/* 處置建議 / 專家覆核 — 佔位（未來工作） */}
          <Card title="處置建議與專家覆核" sub="保留欄位 · 未來 Human-In-The-Loop">
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <p className="mb-1 text-sm font-medium text-slate-600">處置建議</p>
                <textarea
                  disabled
                  rows={3}
                  placeholder="（保留欄位，待專家填寫）"
                  className="w-full rounded border border-dashed border-slate-300 bg-slate-50 px-3 py-2 text-sm"
                />
              </div>
              <div>
                <p className="mb-1 text-sm font-medium text-slate-600">專家覆核</p>
                <div className="flex flex-wrap gap-2">
                  {["接受", "修改", "補充", "退回"].map((b) => (
                    <button
                      key={b}
                      disabled
                      className="cursor-not-allowed rounded border border-dashed border-slate-300 bg-slate-50 px-3 py-1.5 text-sm text-slate-400"
                    >
                      {b}
                    </button>
                  ))}
                </div>
                <p className="mt-2 text-xs text-slate-400">此區為論文未來工作（回寫資料庫與再訓練閉環），目前僅保留欄位。</p>
              </div>
            </div>
          </Card>
        </>
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
                  <td className="py-1 text-right">{report.params.total.toLocaleString()}</td>
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

const LesionCard: React.FC<{ l: DiagnoseResponse["lesions"][number] }> = ({ l }) => (
  <div className="rounded-lg border border-slate-200 p-3">
    <div className="flex gap-3">
      <img src={l.crop} alt={`病灶 L${l.idx}`} className="h-20 w-20 shrink-0 rounded object-cover" />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="rounded bg-red-600 px-1.5 text-xs font-bold text-white">L{l.idx}</span>
          <span className="truncate font-medium text-slate-700">{l.label_zh}</span>
        </div>
        <p className="mt-0.5 text-xs text-slate-400">obj={l.det_score.toFixed(2)}</p>
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
