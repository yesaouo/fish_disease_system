import { useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import ProjectHeader from "../../components/ProjectHeader";
import { Stethoscope, ClipboardList, LogOut, ChevronRight } from "lucide-react";

const ROLE_LABEL: Record<string, string> = {
  guest: "訪客（唯讀）",
  editor: "標註人員",
  expert: "專家"
};

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const { name, role, canEdit, logout } = useAuth();

  const annotationSub = canEdit
    ? "標註、驗證與管理資料集"
    : "瀏覽標註資料（唯讀，需金鑰才能編輯）";

  return (
    <div className="mx-auto flex min-h-screen max-w-4xl flex-col gap-8 px-6 py-12">
      <ProjectHeader />

      <header className="flex items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">魚病診斷輔助系統</h1>
          <p className="flex items-center gap-2 text-sm text-slate-500">
            <span>您好，{name ?? "訪客"}</span>
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
              {ROLE_LABEL[role ?? "guest"] ?? "訪客"}
            </span>
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            logout();
            navigate("/login", { replace: true });
          }}
          className="inline-flex shrink-0 items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm hover:bg-slate-50"
          title={role === "guest" ? "登入金鑰" : "登出"}
          aria-label={role === "guest" ? "登入金鑰" : "登出"}
        >
          <LogOut className="h-4 w-4" />{" "}
          <span className="hidden sm:inline">{role === "guest" ? "登入金鑰" : "登出"}</span>
        </button>
      </header>

      <div className="grid gap-5 sm:grid-cols-2">
        {/* AI 診斷：所有人可用，置於主位 */}
        <button
          onClick={() => navigate("/diagnose")}
          className="group flex flex-col items-start gap-3 rounded-2xl border border-sky-200 bg-gradient-to-br from-sky-50 to-white p-6 text-left shadow-sm transition hover:border-sky-400 hover:shadow-md"
        >
          <span className="inline-flex h-12 w-12 items-center justify-center rounded-xl bg-sky-600 text-white">
            <Stethoscope className="h-6 w-6" />
          </span>
          <span className="text-lg font-semibold text-slate-800">AI 魚病診斷</span>
          <span className="text-sm text-slate-500">
            上傳魚體影像，即時取得病灶定位、相似案例與疑似病因之結構化報告。
          </span>
          <span className="mt-1 inline-flex items-center gap-1 text-sm font-medium text-sky-700">
            開始診斷 <ChevronRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
          </span>
        </button>

        {/* 資料標註與驗證 */}
        <button
          onClick={() => navigate("/datasets")}
          className="group flex flex-col items-start gap-3 rounded-2xl border border-slate-200 bg-white p-6 text-left shadow-sm transition hover:border-slate-400 hover:shadow-md"
        >
          <span className="inline-flex h-12 w-12 items-center justify-center rounded-xl bg-slate-700 text-white">
            <ClipboardList className="h-6 w-6" />
          </span>
          <span className="text-lg font-semibold text-slate-800">資料標註與驗證</span>
          <span className="text-sm text-slate-500">{annotationSub}</span>
          <span className="mt-1 inline-flex items-center gap-1 text-sm font-medium text-slate-700">
            進入資料集 <ChevronRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
          </span>
        </button>
      </div>
    </div>
  );
};

export default HomePage;
