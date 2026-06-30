import axios from "axios";
import type {
  AdminStatsResponse,
  AdminTasksResponse,
  DatasetStats,
  ImageListResponse,
  LoginResponse,
  NextTaskResponse,
  SubmitTaskRequest,
  SubmitTaskResponse,
  LabelMapZhResponse,
  EvidenceOptionsZhResponse,
  DiagnoseResponse,
  SaveTaskRequest,
  SaveTaskResponse,
  AnnotatedListResponse,
  CommentedListResponse,
  DatasetInfo,
  TaskDocument
} from "./types";

const base = import.meta.env.BASE_URL || "/"; // always ends with "/" in Vite
const http = axios.create({
  // Ensure API requests also honor the base path (e.g., "/fish/api")
  baseURL: `${base}api`
});

export const setAuthToken = (token: string | null) => {
  if (token) {
    http.defaults.headers.common.Authorization = `Bearer ${token}`;
  } else {
    delete http.defaults.headers.common.Authorization;
  }
};

// Eagerly load token on module init to avoid 401 after hard refresh.
// Keep this key in sync with AuthContext.TOKEN_KEY ("annotatorToken").
try {
  const saved = typeof window !== "undefined" ? localStorage.getItem("annotatorToken") : null;
  if (saved) {
    http.defaults.headers.common.Authorization = `Bearer ${saved}`;
  }
} catch {
  // ignore if localStorage is not accessible
}

// Also attach token via request interceptor to handle any late changes.
http.interceptors.request.use((config) => {
  try {
    const t = typeof window !== "undefined" ? localStorage.getItem("annotatorToken") : null;
    if (t) {
      config.headers = config.headers ?? {};
      (config.headers as any).Authorization = `Bearer ${t}`;
    }
  } catch {
    // ignore
  }
  return config;
});

export const login = async (name: string, apiKey: string): Promise<LoginResponse> => {
  const { data } = await http.post<LoginResponse>("/login", { name, api_key: apiKey });
  return data;
};

export const fetchDatasets = async (): Promise<DatasetInfo[]> => {
  const { data } = await http.get<{ datasets: DatasetInfo[] }>("/datasets");
  return data.datasets;
};

export const deleteDatasetTask = async (
  dataset: string,
  taskId: string
): Promise<{ ok: boolean; dataset_removed?: boolean }> => {
  const { data } = await http.delete<{ ok: boolean; dataset_removed?: boolean }>(
    `/datasets/${encodeURIComponent(dataset)}/tasks/${encodeURIComponent(taskId)}`
  );
  return data;
};

export const importDiagnosisTask = async (
  dataset: string,
  imageFile: File,
  doc: TaskDocument,
  editorName: string
): Promise<{ ok: boolean; task_id: string; index: number; dataset: string; is_healthy: boolean }> => {
  const form = new FormData();
  form.append("image", imageFile);
  form.append("doc_json", JSON.stringify(doc));
  form.append("editor_name", editorName);
  const { data } = await http.post(
    `/datasets/${encodeURIComponent(dataset)}/tasks/import`,
    form,
    { headers: { "Content-Type": "multipart/form-data" } }
  );
  return data;
};

export const fetchClasses = async (dataset: string): Promise<string[]> => {
  const { data } = await http.get<{ classes: string[] }>(
    `/datasets/${encodeURIComponent(dataset)}/classes`
  );
  return data.classes;
};

export const fetchNextTask = async (
  dataset: string,
  editorName: string | undefined,
  isExpert: boolean
): Promise<NextTaskResponse> => {
  const payload: { dataset: string; editor_name?: string; is_expert: boolean } = { dataset, is_expert: isExpert };
  if (editorName) payload.editor_name = editorName;
  const { data } = await http.post<NextTaskResponse>("/tasks/next", payload);
  return data;
};

export const fetchTaskByIndex = async (
  dataset: string,
  index: number,
  editorName: string | undefined,
  isExpert: boolean
): Promise<NextTaskResponse> => {
  const payload: { dataset: string; index: number; editor_name?: string; is_expert: boolean } = { dataset, index, is_expert: isExpert };
  if (editorName) payload.editor_name = editorName;
  const { data } = await http.post<NextTaskResponse>("/tasks/by_index", payload);
  return data;
};

export const submitTask = async (
  taskId: string,
  payload: SubmitTaskRequest
): Promise<SubmitTaskResponse> => {
  const { data } = await http.post<SubmitTaskResponse>(
    `/tasks/${encodeURIComponent(taskId)}/submit`,
    payload
  );
  return data;
};

export const saveTask = async (
  taskId: string,
  payload: SaveTaskRequest
): Promise<SaveTaskResponse> => {
  const { data } = await http.post<SaveTaskResponse>(
    `/tasks/${encodeURIComponent(taskId)}/save`,
    payload
  );
  return data;
};

export const fetchDatasetStats = async (
  dataset: string
): Promise<DatasetStats> => {
  const { data } = await http.get<DatasetStats>(
    `/datasets/${encodeURIComponent(dataset)}/stats`
  );
  return data;
};

export const fetchAdminStats = async (): Promise<AdminStatsResponse> => {
  const { data } = await http.get<AdminStatsResponse>("/admin/stats");
  return data;
};

export const fetchAdminTasks = async (): Promise<AdminTasksResponse> => {
  const { data } = await http.get<AdminTasksResponse>("/admin/tasks");
  return data;
};

export const fetchLabelMapZh = async (
  dataset: string
): Promise<Record<string, string>> => {
  const { data } = await http.get<LabelMapZhResponse>(
    `/datasets/${encodeURIComponent(dataset)}/labels_zh`
  );
  return data.label_map_zh;
};

export const fetchEvidenceOptionsZh = async (
  dataset: string
): Promise<Record<string, string[]>> => {
  const { data } = await http.get<EvidenceOptionsZhResponse>(
    `/datasets/${encodeURIComponent(dataset)}/evidence_options_zh`
  );
  return data.evidence_options_zh;
};

export const fetchAnnotatedList = async (
  dataset: string
): Promise<AnnotatedListResponse> => {
  const { data } = await http.get<AnnotatedListResponse>(
    `/datasets/${encodeURIComponent(dataset)}/annotated`
  );
  return data;
};

export const fetchCommentedList = async (
  dataset: string
): Promise<CommentedListResponse> => {
  const { data } = await http.get<CommentedListResponse>(
    `/datasets/${encodeURIComponent(dataset)}/commented`
  );
  return data;
};

// 全域 symptoms（classes + zh 標籤 + 外觀敘述選項），供診斷草稿編輯器使用
// （目標資料集可能尚未建立 → 不能用需要資料夾存在的 dataset 範圍端點）。
export const fetchGlobalSymptoms = async (): Promise<{
  classes: string[];
  label_map_zh: Record<string, string>;
  evidence_options_zh: Record<string, string[]>;
}> => {
  const { data } = await http.get("/symptoms");
  return data;
};

// 取某案例的整體描述＋病因（供相似案例描述建議清單）。
export const fetchTaskSummary = async (
  dataset: string,
  taskId: string
): Promise<{ overall: { colloquial_zh: string; medical_zh: string }; global_causes_zh: string[] }> => {
  const { data } = await http.get(
    `/datasets/${encodeURIComponent(dataset)}/tasks/${encodeURIComponent(taskId)}/summary`
  );
  return data;
};

// 由 task_id（影像 stem）解析到資料集 /images 清單中的 1-based index（/annotate/:index 用）。
// 找不到（如資料夾掃描的健康負樣本）回 null。
export const locateTask = async (
  dataset: string,
  taskId: string
): Promise<number | null> => {
  try {
    const { data } = await http.get<{ index: number }>(
      `/datasets/${encodeURIComponent(dataset)}/task_locator`,
      { params: { task_id: taskId } }
    );
    return data.index;
  } catch {
    return null;
  }
};

export const fetchHealthyImagesList = async (
  dataset: string
): Promise<string[]> => {
  const { data } = await http.get<ImageListResponse>(
    `/datasets/${encodeURIComponent(dataset)}/healthy_images`
  );
  return data.images;
};

export const moveImageToHealthyImages = async (
  dataset: string,
  filename: string
): Promise<{ ok: boolean }> => {
  const { data } = await http.post<{ ok: boolean }>(
    `/datasets/${encodeURIComponent(dataset)}/images/${encodeURIComponent(filename)}/move_to_healthy_images`
  );
  return data;
};

export const moveHealthyImageToImages = async (
  dataset: string,
  filename: string
): Promise<{ ok: boolean }> => {
  const { data } = await http.post<{ ok: boolean }>(
    `/datasets/${encodeURIComponent(dataset)}/healthy_images/${encodeURIComponent(filename)}/move_to_images`
  );
  return data;
};

export type DiagnoseParams = {
  text?: string;
  mode?: string;
  topKCases?: number;
  topNCauses?: number;
};

export const diagnose = async (
  image: File,
  params: DiagnoseParams = {}
): Promise<DiagnoseResponse> => {
  const form = new FormData();
  form.append("image", image);
  form.append("text", params.text ?? "");
  form.append("mode", params.mode ?? "grod_soft");
  form.append("top_k_cases", String(params.topKCases ?? 3));
  form.append("top_n_causes", String(params.topNCauses ?? 6));
  const { data } = await http.post<DiagnoseResponse>("/diagnose", form, {
    headers: { "Content-Type": "multipart/form-data" }
  });
  return data;
};

// 以 case_id 向後端要固定模板 PDF（serve 端由快取的報告渲染，不重跑推論、不回傳大 JSON）。
export const downloadReportPdf = async (report: DiagnoseResponse): Promise<void> => {
  const { data } = await http.get("/diagnose/report.pdf", {
    params: { case_id: report.meta.case_id },
    responseType: "blob"
  });
  const url = URL.createObjectURL(data as Blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${report.meta.case_id || "report"}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
};

export const fetchHealthyTaskByIndex = async (
  dataset: string,
  index: number,
  editorName: string | undefined,
  isExpert: boolean
): Promise<NextTaskResponse> => {
  const { data } = await http.post<NextTaskResponse>("/healthy_tasks/by_index", {
    dataset,
    index,
    editor_name: editorName,
    is_expert: isExpert
  });
  return data;
};
