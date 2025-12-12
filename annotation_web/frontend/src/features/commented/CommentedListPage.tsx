import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { fetchCommentedList } from "../../api/client";
import { useDataset } from "../../context/DatasetContext";
import ProjectHeader from "../../components/ProjectHeader";
import { ArrowLeft, CheckCheck } from "lucide-react";

const CommentedListPage: React.FC = () => {
  const navigate = useNavigate();
  const { dataset } = useDataset();

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["commented", dataset],
    queryFn: () => fetchCommentedList(dataset!),
    enabled: !!dataset,
    staleTime: 15_000,
  });

  useEffect(() => {
    if (dataset) void refetch();
  }, [dataset, refetch]);

  return (
    <div className="mx-auto flex min-h-screen max-w-5xl flex-col gap-6 px-6 py-10">
      <ProjectHeader />
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">有註解清單</h1>
          <p className="text-sm text-slate-500">顯示最近註解的影像，點擊可前往編號</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => navigate("/annotated")}
            className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
          >
            <CheckCheck className="h-4 w-4" />
            查看已提交
          </button>
          <button
            onClick={() => navigate("/annotate")}
            className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
          >
            <ArrowLeft className="h-4 w-4" />
            返回標註
          </button>
        </div>
      </header>

      {isLoading && <p className="text-slate-500">載入中...</p>}
      {isError && <p className="text-red-600">載入失敗，請稍後再試。</p>}

      {data && (
        <section className="rounded-xl bg-white shadow">
          <table className="min-w-full divide-y divide-slate-200">
            <thead className="bg-slate-50 text-sm uppercase text-slate-500">
              <tr>
                <th className="px-4 py-3 text-left">編號</th>
                <th className="px-4 py-3 text-left">檔名</th>
                <th className="px-4 py-3 text-left">註解數</th>
                <th className="px-4 py-3 text-left">最後更新</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-sm">
              {data.items.length === 0 && (
                <tr>
                  <td colSpan={4} className="px-4 py-4 text-center text-slate-500">
                    尚無資料
                  </td>
                </tr>
              )}
              {data.items.map((item) => (
                <tr
                  key={`${item.dataset}-${item.task_id}`}
                  className="hover:bg-slate-50 cursor-pointer"
                  onClick={() => navigate(`/annotate/${item.index}`)}
                >
                  <td className="px-4 py-3 font-mono tabular-nums">{item.index}</td>
                  <td className="px-4 py-3">{item.image_filename}</td>
                  <td className="px-4 py-3">{item.comments_count}</td>
                  <td className="px-4 py-3 text-slate-600">
                    {new Date(item.last_modified_at).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
};

export default CommentedListPage;

