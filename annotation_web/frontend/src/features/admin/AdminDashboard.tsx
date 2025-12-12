import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchAdminStats, fetchAdminTasks } from "../../api/client";
import type { DatasetStats } from "../../api/types";
import ProjectHeader from "../../components/ProjectHeader";
import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";

const AdminDashboard: React.FC = () => {
  const navigate = useNavigate();
  const { data, isLoading, isError } = useQuery({
    queryKey: ["admin-stats"],
    queryFn: fetchAdminStats,
    staleTime: 60_000
  });

  const handleExportCsv = async () => {
    const { tasks } = await fetchAdminTasks();
    const rows = [[
      "dataset",
      "image_filename",
      "general",
      "expert"
    ]];
    tasks.forEach((t) => {
      rows.push([
        t.dataset,
        t.image_filename,
        t.general_editor ?? "",
        t.expert_editor ?? ""
      ]);
    });
    const csv = rows.map((r) => r.map((v) => String(v)).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "admin_tasks.csv";
    link.click();
    URL.revokeObjectURL(url);
  };

  const totals = useMemo(() => {
    if (!data) return null;
    return data.datasets.reduce(
      (acc, item) => {
        acc.total += item.total_tasks;
        acc.completed += item.completed_tasks;
        acc.generalCompleted += item.general_completed_tasks;
        acc.expertCompleted += item.expert_completed_tasks;
        acc.duplicate += item.duplicate_completed;
        return acc;
      },
      { total: 0, completed: 0, generalCompleted: 0, expertCompleted: 0, duplicate: 0 }
    );
  }, [data]);

  const submissionTotalsByUser = useMemo(() => {
    if (!data)
      return [] as { user: string; general: number; expert: number; total: number }[];
    const generalMap = new Map<string, number>();
    const expertMap = new Map<string, number>();
    for (const ds of data.datasets) {
      for (const [user, count] of Object.entries(ds.general_submissions_by_user)) {
        generalMap.set(user, (generalMap.get(user) ?? 0) + count);
      }
      for (const [user, count] of Object.entries(ds.expert_submissions_by_user)) {
        expertMap.set(user, (expertMap.get(user) ?? 0) + count);
      }
    }
    const users = new Set<string>([...generalMap.keys(), ...expertMap.keys()]);
    const rows = Array.from(users).map((user) => {
      const general = generalMap.get(user) ?? 0;
      const expert = expertMap.get(user) ?? 0;
      return { user, general, expert, total: general + expert };
    });
    rows.sort((a, b) => b.total - a.total);
    return rows;
  }, [data]);

  return (
    <div className="mx-auto flex min-h-screen max-w-5xl flex-col gap-6 px-6 py-10">
      <ProjectHeader />
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">
            後台統計
          </h1>
          <p className="text-sm text-slate-500">提交概況</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => navigate("/datasets")}
            className="rounded border border-slate-300 px-4 py-2 text-sm text-slate-700 hover:bg-slate-50"
            title="返回資料集"
          >
            <ArrowLeft className="mr-1 inline h-4 w-4 align-[-2px]" aria-hidden="true" />
            返回資料集
          </button>
          <button
            onClick={handleExportCsv}
            className="rounded bg-slate-800 px-4 py-2 text-sm text-white hover:bg-slate-900 disabled:bg-slate-500"
            title="匯出 CSV"
            disabled={!data}
          >
            匯出 CSV
          </button>
        </div>
      </header>

      {isLoading && <p className="text-slate-500">載入統計資料...</p>}
      {isError && (
        <p className="text-red-600">載入統計失敗，請稍後再試。</p>
      )}

      {data && (
        <>
          <section className="grid gap-4 sm:grid-cols-3">
            <StatCard
              title="總任務數"
              value={totals?.total ?? 0}
              accent="bg-sky-100 text-sky-700"
            />
            <StatCard
              title="已完成"
              value={totals?.completed ?? 0}
              accent="bg-emerald-100 text-emerald-700"
            />
            <StatCard
              title="完成率"
              value={
                totals && totals.total
                  ? `${((totals.completed / totals.total) * 100).toFixed(2)}%`
                  : "0%"
              }
              accent="bg-indigo-100 text-indigo-700"
            />
          </section>

          <section className="rounded-xl bg-white shadow">
            <table className="min-w-full divide-y divide-slate-200">
              <thead className="bg-slate-50 text-sm uppercase text-slate-500">
                <tr>
                  <th className="px-4 py-3 text-left">資料集</th>
                  <th className="px-4 py-3 text-right">總任務</th>
                  <th className="px-4 py-3 text-right">一般完成</th>
                  <th className="px-4 py-3 text-right">專家完成</th>
                  <th className="px-4 py-3 text-right">完成率</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-sm">
                {data.datasets.map((item) => (
                  <DatasetRow key={item.dataset} stats={item} />
                ))}
              </tbody>
            </table>
          </section>

          <section className="rounded-xl bg-white shadow">
            <table className="min-w-full divide-y divide-slate-200">
              <thead className="bg-slate-50 text-sm uppercase text-slate-500">
                <tr>
                  <th className="px-4 py-3 text-left">使用者</th>
                  <th className="px-4 py-3 text-right">一般標註</th>
                  <th className="px-4 py-3 text-right">專家標註</th>
                  <th className="px-4 py-3 text-right">合計</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 text-sm">
                {submissionTotalsByUser.length === 0 && (
                  <tr>
                    <td
                      colSpan={4}
                      className="px-4 py-4 text-center text-slate-500"
                    >
                      尚無統計資料
                    </td>
                  </tr>
                )}
                {submissionTotalsByUser.map(({ user, general, expert, total }) => (
                  <tr key={user} className="hover:bg-slate-50">
                    <td className="px-4 py-3 font-medium text-slate-700">
                      {user || "(未命名)"}
                    </td>
                    <td className="px-4 py-3 text-right text-slate-700">
                      {general.toLocaleString()}
                    </td>
                    <td className="px-4 py-3 text-right text-slate-700">
                      {expert.toLocaleString()}
                    </td>
                    <td className="px-4 py-3 text-right text-slate-700">
                      {total.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        </>
      )}
    </div>
  );
};

const DatasetRow: React.FC<{ stats: DatasetStats }> = ({ stats }) => (
  <tr className="hover:bg-slate-50">
    <td className="px-4 py-3 font-medium text-slate-700">{stats.dataset}</td>
    <td className="px-4 py-3 text-right text-slate-600">
      {stats.total_tasks.toLocaleString()}
    </td>
    <td className="px-4 py-3 text-right text-slate-600">
      {stats.general_completed_tasks.toLocaleString()}
    </td>
    <td className="px-4 py-3 text-right text-slate-600">
      {stats.expert_completed_tasks.toLocaleString()}
    </td>
    <td className="px-4 py-3 text-right text-slate-700">
      {(stats.completion_rate * 100).toFixed(2)}%
    </td>
  </tr>
);

const StatCard: React.FC<{
  title: string;
  value: number | string;
  accent: string;
}> = ({ title, value, accent }) => (
  <div className="rounded-xl bg-white p-5 shadow">
    <div className={`mb-2 inline-flex rounded px-2 py-1 text-xs ${accent}`}>
      {title}
    </div>
    <div className="text-2xl font-semibold text-slate-800">{value}</div>
  </div>
);

export default AdminDashboard;
