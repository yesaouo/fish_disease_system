import { Grid3X3, List, Search } from "lucide-react";
import { useMemo, useState } from "react";

export type TaskImageBrowserColumn = {
  key: string;
  label: string;
  headerClassName?: string;
  cellClassName?: string;
};

export type TaskImageBrowserItem = {
  id: string;
  filename: string;
  imageUrl: string;
  onOpen: () => void;
  fields: Record<string, string | number>;
};

type TaskImageBrowserProps = {
  items: TaskImageBrowserItem[];
  columns: TaskImageBrowserColumn[];
  loading: boolean;
  error: boolean;
  emptyText: string;
  searchPlaceholder?: string;
  loadingText?: string;
  errorText?: string;
};

const TaskImageBrowser: React.FC<TaskImageBrowserProps> = ({
  items,
  columns,
  loading,
  error,
  emptyText,
  searchPlaceholder = "搜尋檔名...",
  loadingText = "載入中...",
  errorText = "載入失敗，請稍後再試。",
}) => {
  const [query, setQuery] = useState("");
  const [viewMode, setViewMode] = useState<"grid" | "list">("grid");

  const filteredItems = useMemo(() => {
    const key = query.trim().toLowerCase();
    if (!key) return items;
    return items.filter((item) => item.filename.toLowerCase().includes(key));
  }, [items, query]);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="relative w-full sm:max-w-md">
          <Search className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full rounded-md border border-slate-300 bg-white py-2 pl-9 pr-3 text-sm text-slate-700 shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
            placeholder={searchPlaceholder}
          />
        </div>

        <div className="flex items-center justify-between gap-2 sm:justify-end">
          <div className="text-sm text-slate-500">
            {filteredItems.length} / {items.length}
          </div>
          <div className="inline-flex rounded-md border border-slate-300 bg-white p-1 shadow-sm">
            <button
              type="button"
              onClick={() => setViewMode("grid")}
              className={`inline-flex items-center gap-1 rounded px-2 py-1 text-xs ${
                viewMode === "grid"
                  ? "bg-sky-100 text-sky-700"
                  : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              <Grid3X3 className="h-3.5 w-3.5" />
              網格
            </button>
            <button
              type="button"
              onClick={() => setViewMode("list")}
              className={`inline-flex items-center gap-1 rounded px-2 py-1 text-xs ${
                viewMode === "list"
                  ? "bg-sky-100 text-sky-700"
                  : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              <List className="h-3.5 w-3.5" />
              列表
            </button>
          </div>
        </div>
      </div>

      {loading && <p className="text-slate-500">{loadingText}</p>}
      {error && <p className="text-red-600">{errorText}</p>}

      {!loading && !error && (
        <section className="rounded-xl bg-white p-4 shadow">
          {filteredItems.length === 0 ? (
            <p className="py-10 text-center text-sm text-slate-500">{emptyText}</p>
          ) : viewMode === "grid" ? (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
              {filteredItems.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={item.onOpen}
                  className="group flex flex-col gap-2 rounded-lg border border-slate-200 bg-white p-2 text-left shadow-sm transition hover:border-slate-300 hover:shadow"
                  title={item.filename}
                >
                  <div className="aspect-square overflow-hidden rounded-md bg-slate-100">
                    <img
                      src={item.imageUrl}
                      alt={item.filename}
                      loading="lazy"
                      className="h-full w-full object-cover"
                    />
                  </div>
                  <div className="truncate font-mono text-[11px] text-slate-700">
                    {item.filename}
                  </div>
                </button>
              ))}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-slate-200">
                <thead className="bg-slate-50 text-sm uppercase text-slate-500">
                  <tr>
                    <th className="px-4 py-3 text-left">影像</th>
                    {columns.map((column) => (
                      <th
                        key={column.key}
                        className={`px-4 py-3 text-left ${column.headerClassName ?? ""}`}
                      >
                        {column.label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-sm">
                  {filteredItems.map((item) => (
                    <tr
                      key={item.id}
                      className="cursor-pointer hover:bg-slate-50"
                      onClick={item.onOpen}
                    >
                      <td className="px-4 py-3">
                        <div className="flex min-w-[240px] items-center gap-3">
                          <div className="h-12 w-12 shrink-0 overflow-hidden rounded border border-slate-200 bg-slate-100">
                            <img
                              src={item.imageUrl}
                              alt={item.filename}
                              loading="lazy"
                              className="h-full w-full object-cover"
                            />
                          </div>
                          <span className="font-mono text-xs text-slate-700">
                            {item.filename}
                          </span>
                        </div>
                      </td>
                      {columns.map((column) => (
                        <td
                          key={column.key}
                          className={`px-4 py-3 text-slate-700 ${column.cellClassName ?? ""}`}
                        >
                          {item.fields[column.key] ?? ""}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}
    </div>
  );
};

export default TaskImageBrowser;
