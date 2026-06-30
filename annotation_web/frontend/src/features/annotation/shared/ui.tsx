import React from "react";
import { AlertTriangle, XCircle } from "lucide-react";

export const Kbd: React.FC<React.PropsWithChildren> = ({ children }) => (
  <kbd className="rounded border border-slate-300 bg-white px-1.5 text-[10px] font-medium text-slate-700 shadow-sm">
    {children}
  </kbd>
);

interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  label: string;
  shortcut?: string;
}
export const IconButton: React.FC<IconButtonProps> = ({ label, shortcut, className = "", children, ...rest }) => (
  <button
    type="button"
    aria-label={shortcut ? `${label}（${shortcut}）` : label}
    title={shortcut ? `${label}（${shortcut}）` : label}
    className={[
      "inline-flex h-9 w-9 items-center justify-center rounded-md",
      "border border-slate-200 bg-white/90 hover:bg-white",
      "shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500",
      "disabled:opacity-50 disabled:pointer-events-none",
      className,
    ].join(" ")}
    {...rest}
  >
    {children}
    <span className="sr-only">{label}{shortcut ? `（${shortcut}）` : ""}</span>
  </button>
);

export const Separator: React.FC<{ vertical?: boolean; className?: string }> = ({ vertical, className }) => (
  <div className={[vertical ? "h-6 w-px" : "h-px w-full", "bg-slate-200", className || ""].join(" ")} />
);

export const Banner: React.FC<{
  kind: "error" | "warning";
  children: React.ReactNode;
  onClose?: () => void;
}> = ({ kind, children, onClose }) => {
  const tone = kind === "error"
    ? "border-red-200 bg-red-50 text-red-700"
    : "border-amber-200 bg-amber-50 text-amber-800";
  const Icon = kind === "error" ? XCircle : AlertTriangle;

  return (
    <div
      role="alert"
      aria-live="assertive"
      className={`mt-2 rounded border px-3 py-2 text-sm ${tone} flex items-start justify-between`}
    >
      <div className="flex items-start gap-2">
        <Icon className="mt-0.5 h-4 w-4 shrink-0" />
        <div>{children}</div>
      </div>
      {onClose && (
        <button
          type="button"
          onClick={onClose}
          className="ml-3 inline-flex items-center rounded px-2 py-0.5 text-xs hover:bg-white/50"
          aria-label="關閉"
        >
          關閉
        </button>
      )}
    </div>
  );
};

export const SubmissionCapsules: React.FC<{
  generalEditor?: string[];
  expertEditor?: string[];
  commentsCount?: number;
}> = ({ generalEditor, expertEditor, commentsCount }) => {
  const general = (generalEditor ?? []).map((s) => String(s).trim()).filter(Boolean);
  const expert = (expertEditor ?? []).map((s) => String(s).trim()).filter(Boolean);
  const hasGeneral = general.length > 0;
  const hasExpert = expert.length > 0;
  const comments = commentsCount ?? 0;
  const hasComments = comments > 0;

  if (!hasGeneral && !hasExpert && !hasComments) return null;

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs leading-tight">
      {hasComments && (
        <span className="inline-flex items-center rounded-full bg-amber-50 text-amber-700 ring-1 ring-inset ring-amber-200 px-2 py-1 font-medium">
          <span>已註解</span>
          <span className="ml-1 text-[10px] text-amber-600/80 font-normal">({comments})</span>
        </span>
      )}
      {hasGeneral && (
        <span className="inline-flex items-center rounded-full bg-emerald-50 text-emerald-700 ring-1 ring-inset ring-emerald-200 px-2 py-1 font-medium">
          <span>用戶已提交</span>
          <span className="ml-1 text-[10px] text-emerald-600/80 font-normal">
            ({general[general.length - 1]}{general.length > 1 ? ` +${general.length - 1}` : ""})
          </span>
        </span>
      )}
      {hasExpert && (
        <span className="inline-flex items-center rounded-full bg-sky-50 text-sky-700 ring-1 ring-inset ring-sky-200 px-2 py-1 font-medium">
          <span>專家已提交</span>
          <span className="ml-1 text-[10px] text-sky-600/80 font-normal">
            ({expert[expert.length - 1]}{expert.length > 1 ? ` +${expert.length - 1}` : ""})
          </span>
        </span>
      )}
    </div>
  );
};

export const withBase = (path: string): string => {
  const base = import.meta.env.BASE_URL || "/";
  if (!path) return path;
  return path.startsWith("/") ? `${base.replace(/\/$/, "")}${path}` : path;
};
