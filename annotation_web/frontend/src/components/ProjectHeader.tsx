import React from "react";

type ProjectHeaderProps = {
  className?: string;
};

const withBase = (path: string) => {
  const base = import.meta.env.BASE_URL || "/";
  if (!path) return path;
  // 若是以 "/" 開頭的絕對路徑，補上 base（去掉 base 結尾的 "/" 避免重複）
  return path.startsWith("/")
    ? `${base.replace(/\/$/, "")}${path}`
    : path;
};

const ProjectHeader: React.FC<ProjectHeaderProps> = ({ className }) => {
  return (
    <div
      className={
        "flex items-center justify-center gap-4 sm:gap-6 " + (className ?? "")
      }
      aria-label="計畫標題與校徽"
    >
      <img
        src={withBase("/logos/NTOU_Logo.png")}
        alt="NTOU 校徽"
        className="h-10 w-auto sm:h-12"
        onError={(e) => ((e.target as HTMLImageElement).style.display = "none")}
      />
      <h2 className="text-center text-lg font-semibold text-slate-800 sm:text-2xl">
        透過表型體學精準辨識疾病
      </h2>
      <img
        src={withBase("/logos/NPUST_Logo.png")}
        alt="NPUST 校徽"
        className="h-10 w-auto sm:h-12"
        onError={(e) => ((e.target as HTMLImageElement).style.display = "none")}
      />
    </div>
  );
};

export default ProjectHeader;
