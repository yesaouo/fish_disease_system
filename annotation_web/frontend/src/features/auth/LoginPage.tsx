import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../../context/AuthContext";
import ProjectHeader from "../../components/ProjectHeader";

const NAME_PATTERN = /^[\u4e00-\u9fffA-Za-z]{1,32}$/;

const LoginPage: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [isExpert, setIsExpert] = useState(true);
  const [apiKey, setApiKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (evt: FormEvent<HTMLFormElement>) => {
    evt.preventDefault();
    const trimmed = name.trim();
    if (!NAME_PATTERN.test(trimmed)) {
      setError("姓名僅能使用中英文，長度 1-32 字元");
      return;
    }
    if (!apiKey.trim()) {
      setError("請輸入金鑰");
      return;
    }
    setLoading(true);
    try {
      await login(trimmed, isExpert, apiKey.trim());
      navigate("/datasets", { replace: true });
    } catch (err) {
      console.error(err);
      setError("登入失敗，請稍後再試");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-200 to-slate-50">
      <div className="flex w-full max-w-md flex-col items-stretch px-4">
        <form
          onSubmit={handleSubmit}
          className="rounded-xl bg-white p-8 shadow-xl"
        >
          <ProjectHeader className="mb-4" />
          <label className="mb-2 block text-sm font-medium text-slate-600">
            顯示名稱
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => {
              setName(e.target.value);
              setError(null);
            }}
            placeholder="請輸入中英文姓名"
            className="mb-3 w-full rounded border border-slate-300 px-3 py-2 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
            maxLength={32}
          />
          <label className="mb-2 block text-sm font-medium text-slate-600">
            金鑰
          </label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => {
              setApiKey(e.target.value);
              setError(null);
            }}
            placeholder="請輸入金鑰"
            className="mb-3 w-full rounded border border-slate-300 px-3 py-2 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
          />
          {error && (
            <div className="mb-4 rounded bg-red-50 px-3 py-2 text-sm text-red-600">
              {error}
            </div>
          )}
          <div className="mb-4 flex items-center gap-2 text-sm text-slate-700">
            <input
              id="expert-flag"
              type="checkbox"
              checked={isExpert}
              onChange={(e) => setIsExpert(e.target.checked)}
            />
            <label htmlFor="expert-flag">我是養殖專家</label>
          </div>
          <button
            type="submit"
            disabled={loading}
            className="w-full rounded bg-sky-600 px-3 py-2 text-white hover:bg-sky-700 disabled:bg-sky-300"
          >
            {loading ? "登入中..." : "開始標註"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default LoginPage;
