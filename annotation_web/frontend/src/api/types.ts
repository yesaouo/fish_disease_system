export type Detection = {
  label: string;
  evidence_zh?: string;
  evidence_index?: number | null;
  box_xyxy: [number, number, number, number];
  confidence?: number;
};

export type Comment = {
  author: string;
  text: string;
  created_at?: string;
};

export type TaskDocument = {
  dataset: string;
  image_filename: string;
  is_healthy?: boolean;
  image_width: number;
  image_height: number;
  last_modified_at: string;
  general_editor?: string[];
  expert_editor?: string[];
  overall: {
    colloquial_zh: string;
    medical_zh: string;
  };
  detections: Detection[];
  global_causes_zh: string[];
  global_treatments_zh: string[];
  comments?: Comment[];
  generated_by?: string | null;
};

export type NextTaskResponse = {
  task_id: string;
  task: TaskDocument;
  image_url: string;
  index: number;
  total_tasks: number;
};

export type SubmitTaskRequest = {
  full_json: TaskDocument;
  editor_name: string;
  is_expert: boolean;
};

export type SubmitTaskResponse = {
  ok: boolean;
};

export type DatasetStats = {
  dataset: string;
  total_tasks: number;
  completed_tasks: number;
  duplicate_completed: number;
  // Split completions by role
  general_completed_tasks: number;
  expert_completed_tasks: number;
  submissions_by_user: Record<string, number>;
  // Split submissions by role
  general_submissions_by_user: Record<string, number>;
  expert_submissions_by_user: Record<string, number>;
  completion_rate: number;
};

export type AdminStatsResponse = {
  datasets: DatasetStats[];
};

export type TaskSummary = {
  dataset: string;
  image_filename: string;
  annotations_count: number;
  general_editor?: string[];
  expert_editor?: string[];
};

export type AdminTasksResponse = {
  tasks: TaskSummary[];
};

export type SaveTaskRequest = {
  full_json: TaskDocument;
  editor_name: string;
  is_expert: boolean;
};

export type SaveTaskResponse = {
  ok: boolean;
};

export type AnnotatedItem = {
  dataset: string;
  index: number;
  task_id: string;
  image_filename: string;
  last_modified_at: string;
  general_editor?: string[];
  expert_editor?: string[];
};

export type AnnotatedListResponse = {
  items: AnnotatedItem[];
};

export type CommentedItem = {
  dataset: string;
  index: number;
  task_id: string;
  image_filename: string;
  last_modified_at: string;
  comments_count: number;
};

export type CommentedListResponse = {
  items: CommentedItem[];
};

export type ImageListResponse = {
  images: string[];
};

export type LoginResponse = {
  token: string;
  name: string;
};

export type LabelMapZhResponse = {
  label_map_zh: Record<string, string>;
};

export type EvidenceOptionsZhResponse = {
  evidence_options_zh: Record<string, string[]>;
};
