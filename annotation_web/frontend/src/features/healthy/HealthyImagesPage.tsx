import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Search } from "lucide-react";

import { fetchHealthyImagesList } from "../../api/client";
import ProjectHeader from "../../components/ProjectHeader";
import { useDataset } from "../../context/DatasetContext";

const baseUrl = import.meta.env.BASE_URL || "/";
const apiBase = baseUrl.replace(/\/$/, "");

const HealthyImagesPage: React.FC = () => {
  const navigate = useNavigate();
  const { dataset } = useDataset();
  const [query, setQuery] = useState("");

  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ["healthy_images", dataset],
    queryFn: () => fetchHealthyImagesList(dataset!),
    enabled: !!dataset,
    staleTime: 15_000,
  });

  useEffect(() => {
    if (dataset) void refetch();
  }, [dataset, refetch]);

  const filtered = useMemo(() => {
    const images = data ?? [];
    const q = query.trim().toLowerCase();
    if (!q) return images;
    return images.filter((fn) => fn.toLowerCase().includes(q));
  }, [data, query]);

  const indexByFilename = useMemo(() => {
    const m = new Map<string, number>();
    (data ?? []).forEach((fn, i) => m.set(fn, i + 1));
    return m;
  }, [data]);

  const srcFor = (filename: string) =>
    `${apiBase}/api/datasets/${encodeURIComponent(dataset || "")}/healthy_images/${encodeURIComponent(filename)}`;

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-6 py-10">
      <ProjectHeader />

      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">健康影像</h1>
          <p className="text-sm text-slate-500">
            {dataset ? `資料集：${dataset}` : "尚未選擇資料集"}
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => navigate("/annotate")}
            className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
          >
            <ArrowLeft className="h-4 w-4" />
            回到標註
          </button>
        </div>
      </header>

      <div className="flex items-center gap-2">
        <div className="relative w-full max-w-md">
          <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full rounded-md border border-slate-300 bg-white py-2 pl-9 pr-3 text-sm text-slate-700 shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
            placeholder="搜尋檔名..."
          />
        </div>
        {data && (
          <div className="text-sm text-slate-500">
            {filtered.length} / {data.length}
          </div>
        )}
      </div>

      {isLoading && <p className="text-slate-500">載入中...</p>}
      {isError && <p className="text-red-600">載入失敗，請稍後再試。</p>}

      {data && (
        <section className="rounded-xl bg-white p-4 shadow">
          {filtered.length === 0 ? (
            <p className="py-10 text-center text-sm text-slate-500">
              沒有健康影像
            </p>
          ) : (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
              {filtered.map((filename) => {
                const index = indexByFilename.get(filename) ?? 1;
                return (
                  <button
                    key={filename}
                    type="button"
                    onClick={() => navigate(`/healthy/${index}`)}
                    className="group flex flex-col gap-2 rounded-lg border border-slate-200 bg-white p-2 text-left shadow-sm transition hover:border-slate-300 hover:shadow"
                    title={filename}
                  >
                    <div className="aspect-square overflow-hidden rounded-md bg-slate-100">
                      <img
                        src={srcFor(filename)}
                        alt={filename}
                        loading="lazy"
                        className="h-full w-full object-cover"
                      />
                    </div>
                    <div className="min-w-0">
                      <div className="truncate font-mono text-[11px] text-slate-700">
                        {filename}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </section>
      )}
    </div>
  );
};

export default HealthyImagesPage;
