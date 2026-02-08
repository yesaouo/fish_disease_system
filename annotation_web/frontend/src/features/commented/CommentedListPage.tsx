import { useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, CheckCheck, Images } from "lucide-react";

import { fetchCommentedList } from "../../api/client";
import TaskImageBrowser, {
  TaskImageBrowserColumn,
  TaskImageBrowserItem,
} from "../../components/TaskImageBrowser";
import ProjectHeader from "../../components/ProjectHeader";
import { useDataset } from "../../context/DatasetContext";

const baseUrl = import.meta.env.BASE_URL || "/";
const apiBase = baseUrl.replace(/\/$/, "");

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

  const columns: TaskImageBrowserColumn[] = useMemo(
    () => [
      { key: "index", label: "編號", cellClassName: "font-mono tabular-nums" },
      { key: "comments_count", label: "註解數", headerClassName: "w-28" },
      { key: "updated_at", label: "最後更新", headerClassName: "w-56" },
    ],
    []
  );

  const items: TaskImageBrowserItem[] = useMemo(() => {
    return (data?.items ?? []).map((item) => ({
      id: `${item.dataset}-${item.task_id}`,
      filename: item.image_filename,
      imageUrl: `${apiBase}/api/datasets/${encodeURIComponent(
        dataset || item.dataset
      )}/images/${encodeURIComponent(item.image_filename)}`,
      onOpen: () => navigate(`/annotate/${item.index}`),
      fields: {
        index: item.index,
        comments_count: item.comments_count,
        updated_at: new Date(item.last_modified_at).toLocaleString(),
      },
    }));
  }, [data, dataset, navigate]);

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-6 py-10">
      <ProjectHeader />
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">有註解</h1>
          <p className="text-sm text-slate-500">顯示含註解的影像，可切換網格/列表檢視。</p>
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
            onClick={() => navigate("/healthy")}
            className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
          >
            <Images className="h-4 w-4" />
            健康影像
          </button>
          <button
            onClick={() => navigate("/annotate")}
            className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
          >
            <ArrowLeft className="h-4 w-4" />
            回到標註
          </button>
        </div>
      </header>

      <TaskImageBrowser
        items={items}
        columns={columns}
        loading={isLoading}
        error={isError}
        emptyText="目前沒有有註解項目。"
      />
    </div>
  );
};

export default CommentedListPage;
