import { useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, CheckCheck, MessageSquareQuote } from "lucide-react";

import { fetchHealthyImagesList } from "../../api/client";
import TaskImageBrowser, {
  TaskImageBrowserColumn,
  TaskImageBrowserItem,
} from "../../components/TaskImageBrowser";
import ProjectHeader from "../../components/ProjectHeader";
import { useDataset } from "../../context/DatasetContext";

const baseUrl = import.meta.env.BASE_URL || "/";
const apiBase = baseUrl.replace(/\/$/, "");

const HealthyImagesPage: React.FC = () => {
  const navigate = useNavigate();
  const { dataset } = useDataset();

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["healthy_images", dataset],
    queryFn: () => fetchHealthyImagesList(dataset!),
    enabled: !!dataset,
    staleTime: 15_000,
  });

  useEffect(() => {
    if (dataset) void refetch();
  }, [dataset, refetch]);

  const columns: TaskImageBrowserColumn[] = useMemo(
    () => [
      { key: "index", label: "編號", cellClassName: "font-mono tabular-nums" },
      { key: "bucket", label: "來源" },
    ],
    []
  );

  const items: TaskImageBrowserItem[] = useMemo(() => {
    return (data ?? []).map((filename, index) => ({
      id: filename,
      filename,
      imageUrl: `${apiBase}/api/datasets/${encodeURIComponent(
        dataset || ""
      )}/healthy_images/${encodeURIComponent(filename)}`,
      onOpen: () => navigate(`/healthy/${index + 1}`),
      fields: {
        index: index + 1,
        bucket: "healthy_images",
      },
    }));
  }, [data, dataset, navigate]);

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-6 py-10">
      <ProjectHeader />
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">健康影像</h1>
          <p className="text-sm text-slate-500">
            顯示健康的影像，可切換網格/列表檢視。
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => navigate("/annotated")}
            className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
          >
            <CheckCheck className="h-4 w-4" />
            查看提交
          </button>
          <button
            onClick={() => navigate("/commented")}
            className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
          >
            <MessageSquareQuote className="h-4 w-4" />
            查看註解
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
        emptyText="目前沒有健康影像。"
      />
    </div>
  );
};

export default HealthyImagesPage;
