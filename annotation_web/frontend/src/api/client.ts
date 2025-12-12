import axios from "axios";
import type {
  AdminStatsResponse,
  AdminTasksResponse,
  DatasetStats,
  LoginResponse,
  NextTaskResponse,
  SkipTaskRequest,
  SkipTaskResponse,
  SubmitTaskRequest,
  SubmitTaskResponse,
  LabelMapZhResponse,
  SaveTaskRequest,
  SaveTaskResponse,
  AnnotatedListResponse,
  CommentedListResponse
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

export const login = async (name: string, isExpert: boolean, apiKey: string): Promise<LoginResponse> => {
  const { data } = await http.post<LoginResponse>("/login", { name, is_expert: isExpert, api_key: apiKey });
  return data;
};

export const fetchDatasets = async (): Promise<string[]> => {
  const { data } = await http.get<{ datasets: string[] }>("/datasets");
  return data.datasets;
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

export const skipTask = async (
  taskId: string,
  payload: SkipTaskRequest
): Promise<SkipTaskResponse> => {
  const { data } = await http.post<SkipTaskResponse>(
    `/tasks/${encodeURIComponent(taskId)}/skip`,
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
