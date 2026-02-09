import { useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Images, MessageSquareQuote } from "lucide-react";

import { fetchAnnotatedList } from "../../api/client";
import TaskImageBrowser, {
  TaskImageBrowserColumn,
  TaskImageBrowserItem,
} from "../../components/TaskImageBrowser";
import ProjectHeader from "../../components/ProjectHeader";
import { useDataset } from "../../context/DatasetContext";

const baseUrl = import.meta.env.BASE_URL || "/";
const apiBase = baseUrl.replace(/\/$/, "");

const AnnotatedListPage: React.FC = () => {
  const navigate = useNavigate();
  const { dataset } = useDataset();

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["annotated", dataset],
    queryFn: () => fetchAnnotatedList(dataset!),
    enabled: !!dataset,
    staleTime: 15_000,
  });

  useEffect(() => {
    if (dataset) void refetch();
  }, [dataset, refetch]);

  const columns: TaskImageBrowserColumn[] = useMemo(
    () => [
      { key: "index", label: "編號", cellClassName: "font-mono tabular-nums" },
      { key: "updated_at", label: "最後更新", headerClassName: "w-56" },
      { key: "editor", label: "最近提交者", headerClassName: "w-40" },
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
        updated_at: new Date(item.last_modified_at).toLocaleString(),
        editor:
          (item.expert_editor && item.expert_editor[item.expert_editor.length - 1]) ||
          (item.general_editor && item.general_editor[item.general_editor.length - 1]) ||
          "-",
      },
    }));
  }, [data, dataset, navigate]);

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-6 py-10">
      <ProjectHeader />
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">已提交</h1>
          <p className="text-sm text-slate-500">顯示已提交的影像，可切換網格/列表檢視。</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => navigate("/commented")}
            className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
          >
            <MessageSquareQuote className="h-4 w-4" />
            查看註解
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
        emptyText="目前沒有已提交項目。"
      />
    </div>
  );
};

export default AnnotatedListPage;
