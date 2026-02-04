import React, {
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";
import { AxiosError } from "axios";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { UNSAFE_NavigationContext as NavigationContext } from "react-router";
import {
  AlertTriangle,
  ArrowLeft,
  Redo2,
  Save,
  SquarePlus,
  Trash2,
  Undo2,
  XCircle,
} from "lucide-react";

import {
  fetchHealthyTaskByIndex,
  moveHealthyImageToImages,
  saveTask,
  submitTask,
} from "../../api/client";
import type {
  Comment as TaskComment,
  NextTaskResponse,
  TaskDocument,
} from "../../api/types";
import { useAuth } from "../../context/AuthContext";
import { useDataset } from "../../context/DatasetContext";
import AnnotationCanvas from "../annotation/components/AnnotationCanvas";
import {
  ValidationError,
  cloneTaskDocument,
  defaultDetection,
  documentsEqual,
  normalizeBox,
} from "../../lib/taskUtils";

const HEALTHY_LABEL = "healthy_region";
const HEALTHY_LABEL_DISPLAY = "健康特徵(healthy_region)";

const baseUrl = import.meta.env.BASE_URL || "/";

// --------- Helpers for reducer-based history ---------
const HISTORY_LIMIT = 100;

function clampHistory(hist: TaskDocument[]) {
  return hist.length > HISTORY_LIMIT
    ? hist.slice(hist.length - HISTORY_LIMIT)
    : hist;
}

function normalizeSelection(sel: number | null, detections: unknown[]) {
  if (!detections || detections.length === 0) return null;
  return sel == null ? null : Math.min(sel, detections.length - 1);
}

type State = {
  doc: TaskDocument | null;
  history: TaskDocument[];
  future: TaskDocument[];
  selectedIndex: number | null;
  validationErrors: ValidationError[];
};

type Action =
  | { type: "UNDO" }
  | { type: "REDO" }
  | { type: "APPLY_DOC"; next: TaskDocument; nextSelected?: number | null }
  | { type: "LOAD_DOC"; doc: TaskDocument }
  | { type: "SET_SELECTED"; index: number | null }
  | { type: "SET_ERRORS"; errors: ValidationError[] }
  | { type: "RESET_ERRORS" };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "UNDO": {
      const { doc, history } = state;
      if (!doc || history.length === 0) return state;

      const prev = history[history.length - 1];
      const nextDoc = cloneTaskDocument(prev);

      return {
        ...state,
        doc: nextDoc,
        history: history.slice(0, -1),
        future: [cloneTaskDocument(doc), ...state.future],
        selectedIndex: normalizeSelection(
          state.selectedIndex,
          nextDoc.detections
        ),
        validationErrors: [],
      };
    }

    case "REDO": {
      const { doc, future } = state;
      if (!doc || future.length === 0) return state;

      const [next, ...rest] = future;
      const nextDoc = cloneTaskDocument(next);

      return {
        ...state,
        doc: nextDoc,
        history: clampHistory([...state.history, cloneTaskDocument(doc)]),
        future: rest,
        selectedIndex: normalizeSelection(
          state.selectedIndex,
          nextDoc.detections
        ),
        validationErrors: [],
      };
    }

    case "APPLY_DOC": {
      const nextDoc = cloneTaskDocument(action.next);
      return {
        ...state,
        doc: nextDoc,
        history: state.doc
          ? clampHistory([...state.history, cloneTaskDocument(state.doc)])
          : state.history,
        future: [],
        selectedIndex:
          action.nextSelected !== undefined
            ? action.nextSelected
            : normalizeSelection(state.selectedIndex, nextDoc.detections),
        validationErrors: [],
      };
    }

    case "LOAD_DOC": {
      return {
        doc: cloneTaskDocument(action.doc),
        history: [],
        future: [],
        selectedIndex: null,
        validationErrors: [],
      };
    }

    case "SET_SELECTED":
      return { ...state, selectedIndex: action.index };

    case "SET_ERRORS":
      return { ...state, validationErrors: action.errors };

    case "RESET_ERRORS":
      return { ...state, validationErrors: [] };

    default:
      return state;
  }
}

// ===== UI helpers =====
const Kbd: React.FC<React.PropsWithChildren> = ({ children }) => (
  <kbd className="rounded border border-slate-300 bg-white px-1.5 text-[10px] font-medium text-slate-700 shadow-sm">
    {children}
  </kbd>
);

interface IconButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  label: string;
  shortcut?: string;
}
const IconButton: React.FC<IconButtonProps> = ({
  label,
  shortcut,
  className = "",
  children,
  ...rest
}) => (
  <button
    type="button"
    aria-label={shortcut ? `${label}（${shortcut}）` : label}
    title={shortcut ? `${label}（${shortcut}）` : label}
    className={[
      "inline-flex h-9 w-9 items-center justify-center rounded-md",
      "border border-slate-200 bg-white/90 hover:bg-white",
      "shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500",
      "disabled:pointer-events-none disabled:opacity-50",
      className,
    ].join(" ")}
    {...rest}
  >
    {children}
    <span className="sr-only">
      {label}
      {shortcut ? `（${shortcut}）` : ""}
    </span>
  </button>
);

const Separator: React.FC<{ vertical?: boolean; className?: string }> = ({
  vertical,
  className,
}) => (
  <div
    className={[
      vertical ? "h-6 w-px" : "h-px w-full",
      "bg-slate-200",
      className || "",
    ].join(" ")}
  />
);

const Banner: React.FC<{
  kind: "error" | "warning";
  children: React.ReactNode;
  onClose?: () => void;
}> = ({ kind, children, onClose }) => {
  const tone =
    kind === "error"
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
          className="ml-3 rounded px-2 py-1 text-xs hover:bg-white/60"
        >
          關閉
        </button>
      )}
    </div>
  );
};

// ===== Navigation guard =====
const useNavBlocker = (
  when: boolean,
  bypass?: (nextLocation: any) => boolean
) => {
  const { navigator } = useContext(NavigationContext) as any;
  useEffect(() => {
    if (!when || !navigator?.block) return;
    const unblock = navigator.block((tx: any) => {
      if (bypass?.(tx.location)) {
        unblock();
        tx.retry();
        return;
      }
      const ok = window.confirm(
        "目前有未儲存的變更，離開將放棄這些變更。確定要離開嗎？"
      );
      if (ok) {
        unblock();
        tx.retry();
      }
    });
    return unblock;
  }, [when, navigator, bypass]);
};

const validateHealthyDocument = (doc: TaskDocument): ValidationError[] => {
  const errors: ValidationError[] = [];
  const width = doc.image_width || 0;
  const height = doc.image_height || 0;

  doc.detections.forEach((det, idx) => {
    const [x1, y1, x2, y2] = det.box_xyxy;
    if (x1 < 0 || y1 < 0 || x2 > width || y2 > height) {
      errors.push({
        field: `detections.${idx}.box_xyxy`,
        message: "框必須在圖片範圍內",
      });
    }
    if (x2 <= x1 || y2 <= y1) {
      errors.push({
        field: `detections.${idx}.box_xyxy`,
        message: "框座標需滿足 x2>x1 且 y2>y1",
      });
    }

    const label = (det.label ?? "").trim();
    if (!label) {
      errors.push({
        field: `detections.${idx}.label`,
        message: "請選擇表徵類別",
      });
    } else if (label !== HEALTHY_LABEL) {
      errors.push({
        field: `detections.${idx}.label`,
        message: `僅允許：${HEALTHY_LABEL_DISPLAY}`,
      });
    }
  });

  return errors;
};

const withBase = (path: string) => {
  const base = import.meta.env.BASE_URL || "/";
  if (!path) return path;
  return path.startsWith("/") ? `${base.replace(/\/$/, "")}${path}` : path;
};

const HealthyAnnotationPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { index } = useParams();
  const routeIndex = index ? Number(index) : null;

  const { name, isExpert } = useAuth();
  const { dataset } = useDataset();

  const [task, setTask] = useState<NextTaskResponse | null>(null);
  const [{ doc, history, future, selectedIndex, validationErrors }, dispatch] =
    useReducer(reducer, {
      doc: null,
      history: [],
      future: [],
      selectedIndex: null,
      validationErrors: [],
    } as State);

  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gotoIndex, setGotoIndex] = useState<string>("");
  const [commentDraft, setCommentDraft] = useState("");

  const labelRef = useRef<HTMLSelectElement | null>(null);
  const listItemRefs = useRef<Record<number, HTMLButtonElement | null>>({});

  const allowNavRef = useRef(false);
  const runWithBypass = useCallback(async (fn: () => Promise<void> | void) => {
    allowNavRef.current = true;
    try {
      await fn();
    } finally {
      allowNavRef.current = false;
    }
  }, []);

  const dirty = useMemo(() => {
    if (!task || !doc) return false;
    return !documentsEqual(doc, task.task);
  }, [doc, task]);

  const dirtyRef = useRef(false);
  useEffect(() => {
    dirtyRef.current = dirty;
  }, [dirty]);

  useNavBlocker(dirty, () => allowNavRef.current);

  useEffect(() => {
    const handler = (evt: BeforeUnloadEvent) => {
      if (dirtyRef.current) {
        evt.preventDefault();
        evt.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, []);

  const confirmAndNavigate = useCallback(
    (to: string, options?: { replace?: boolean; state?: any }) => {
      if (dirtyRef.current) {
        const ok = window.confirm(
          "目前有未儲存的變更，離開將放棄這些變更。確定要離開嗎？"
        );
        if (!ok) return;
      }
      runWithBypass(() => navigate(to, options));
    },
    [navigate, runWithBypass]
  );

  const loadTask = useCallback(async () => {
    if (!dataset) return;
    if (routeIndex == null || Number.isNaN(routeIndex) || routeIndex <= 0) {
      confirmAndNavigate("/healthy", { replace: true });
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const resp = await fetchHealthyTaskByIndex(
        dataset,
        routeIndex,
        name ?? undefined,
        isExpert
      );
      setTask(resp);
      setGotoIndex(String(resp.index));
      dispatch({ type: "LOAD_DOC", doc: resp.task });
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      if (axiosErr.response?.status === 404) {
        await runWithBypass(() => navigate("/healthy", { replace: true }));
        return;
      }
      setError("載入失敗，請稍後再試。");
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [
    dataset,
    routeIndex,
    name,
    isExpert,
    confirmAndNavigate,
    navigate,
    runWithBypass,
  ]);

  useEffect(() => {
    if (location.pathname.startsWith("/healthy/")) {
      void loadTask();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.key]);

  const updateDoc = useCallback(
    (
      mutator: (draft: TaskDocument) => void,
      nextSelected?: (draft: TaskDocument) => number | null
    ) => {
      if (!doc) return;
      const next = cloneTaskDocument(doc);
      mutator(next);
      dispatch({
        type: "APPLY_DOC",
        next,
        nextSelected: nextSelected ? nextSelected(next) : undefined,
      });
    },
    [doc]
  );

  const selectedDetection = useMemo(() => {
    if (!doc || selectedIndex == null) return null;
    return doc.detections[selectedIndex] ?? null;
  }, [doc, selectedIndex]);

  const detectionErrors = useMemo(() => {
    const m = new Map<string, string>();
    validationErrors.forEach((e) => m.set(e.field, e.message));
    return m;
  }, [validationErrors]);

  const detectionErrorCounts = useMemo(() => {
    const counts: Record<number, number> = {};
    validationErrors.forEach((e) => {
      const match = e.field.match(/^detections\.(\d+)\./);
      if (!match) return;
      const idx = Number(match[1]);
      counts[idx] = (counts[idx] ?? 0) + 1;
    });
    return counts;
  }, [validationErrors]);

  const getDisplayLabel = useCallback((enLabel: string | undefined | null) => {
    const key = (enLabel || "").trim();
    if (!key) return "";
    if (key === HEALTHY_LABEL) return HEALTHY_LABEL_DISPLAY;
    return key;
  }, []);

  const handleSelectDetection = useCallback((idx: number) => {
    dispatch({ type: "SET_SELECTED", index: idx });
  }, []);

  const handleUpdateBox = useCallback(
    (idx: number, box: [number, number, number, number]) => {
      if (!doc) return;
      updateDoc((draft) => {
        const det = draft.detections[idx];
        if (!det) return;
        const [x1, y1, x2, y2] = box;
        det.box_xyxy = normalizeBox(
          x1,
          y1,
          x2,
          y2,
          draft.image_width,
          draft.image_height
        );
      });
    },
    [doc, updateDoc]
  );

  const handleAddDetection = useCallback(() => {
    if (!doc) return;
    updateDoc(
      (draft) => {
        const next = defaultDetection(
          draft.image_width,
          draft.image_height,
          HEALTHY_LABEL
        );
        next.evidence_index = null;
        next.evidence_zh = "";
        draft.detections = [...draft.detections, next];
      },
      (draft) => draft.detections.length - 1
    );
  }, [doc, updateDoc]);

  const handleRemoveDetection = useCallback(() => {
    if (!doc || selectedIndex == null) return;
    updateDoc(
      (draft) => {
        draft.detections.splice(selectedIndex, 1);
        draft.detections = [...draft.detections];
      },
      (draft) => {
        if (draft.detections.length === 0) return null;
        return Math.min(selectedIndex, draft.detections.length - 1);
      }
    );
  }, [doc, selectedIndex, updateDoc]);

  const handleDetectionLabel = useCallback(
    (idx: number, value: string) => {
      updateDoc((draft) => {
        const det = draft.detections[idx];
        if (!det) return;
        det.label = value;
        if ((value || "").trim() === HEALTHY_LABEL) {
          det.evidence_index = null;
          det.evidence_zh = "";
        }
      });
    },
    [updateDoc]
  );

  const handleAddComment = useCallback(() => {
    if (!name || !doc) return;
    const text = (commentDraft || "").replace(/\s*\n+\s*/g, " ").trim();
    if (!text) return;
    updateDoc((draft) => {
      const comments: TaskComment[] = (draft as any).comments ?? [];
      const entry: TaskComment = {
        author: name,
        text,
        created_at: new Date().toISOString(),
      };
      (draft as any).comments = [...comments, entry];
    });
    setCommentDraft("");
  }, [commentDraft, doc, name, updateDoc]);

  const handleRemoveComment = useCallback(
    (idx: number) => {
      updateDoc((draft) => {
        const comments: TaskComment[] = (draft as any).comments ?? [];
        if (idx < 0 || idx >= comments.length) return;
        comments.splice(idx, 1);
        (draft as any).comments = [...comments];
      });
    },
    [updateDoc]
  );

  const handleUndo = useCallback(() => dispatch({ type: "UNDO" }), []);
  const handleRedo = useCallback(() => dispatch({ type: "REDO" }), []);

  const jumpToFirstError = useCallback(() => {
    const first = validationErrors[0];
    if (!first) return;
    const match = first.field.match(/^detections\.(\d+)\./);
    if (match) {
      const idx = Number(match[1]);
      dispatch({ type: "SET_SELECTED", index: idx });
      setTimeout(() => {
        listItemRefs.current[idx]?.scrollIntoView({ block: "center" });
        labelRef.current?.focus();
      }, 0);
      return;
    }
    labelRef.current?.focus();
  }, [validationErrors]);

  const handleSave = useCallback(async () => {
    if (!doc || !task || !dataset || !name) return;
    setSaving(true);
    dispatch({ type: "RESET_ERRORS" });
    try {
      await saveTask(task.task_id, {
        full_json: doc,
        editor_name: name,
        is_expert: isExpert,
      });
      setTask((prev) =>
        prev ? { ...prev, task: cloneTaskDocument(doc) } : prev
      );
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      setError(axiosErr.response?.data?.detail || "保存失敗，請稍後再試。");
    } finally {
      setSaving(false);
    }
  }, [dataset, doc, isExpert, name, task]);

  const handleSubmit = useCallback(async () => {
    if (!doc || !task || !dataset || !name) return;
    const ok = window.confirm("確定要送出標註嗎？");
    if (!ok) return;

    const errors = validateHealthyDocument(doc);
    if (errors.length) {
      dispatch({ type: "SET_ERRORS", errors });
      setTimeout(() => jumpToFirstError(), 0);
      return;
    }

    setSaving(true);
    dispatch({ type: "RESET_ERRORS" });
    try {
      await submitTask(task.task_id, {
        full_json: doc,
        editor_name: name,
        is_expert: isExpert,
      });
      setTask((prev) =>
        prev ? { ...prev, task: cloneTaskDocument(doc) } : prev
      );
      window.alert("已提交。");
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      setError(axiosErr.response?.data?.detail || "提交失敗，請稍後再試。");
    } finally {
      setSaving(false);
    }
  }, [dataset, doc, isExpert, jumpToFirstError, name, task]);

  const handleMoveToImages = useCallback(async () => {
    if (!task || !dataset || !doc) return;

    if (dirty) {
      const ok = window.confirm("目前有未保存的修改。仍要把影像移回 /images 嗎？");
      if (!ok) return;
    }

    const confirm = window.confirm(
      "確定要把影像移回 /images 嗎？移回後將會出現在標註列表中。"
    );
    if (!confirm) return;

    setSaving(true);
    dispatch({ type: "RESET_ERRORS" });
    try {
      await moveHealthyImageToImages(dataset, doc.image_filename);
      const currentIdx = routeIndex ?? task.index;
      await runWithBypass(() =>
        navigate(`/healthy/${currentIdx}`, { replace: true })
      );
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      setError(axiosErr.response?.data?.detail || "移動失敗，請稍後再試。");
    } finally {
      setSaving(false);
    }
  }, [dataset, dirty, doc, navigate, routeIndex, runWithBypass, task]);

  const addDisabled = loading || saving || !doc;
  const removeDisabled = loading || saving || !doc || selectedIndex == null;
  const undoDisabled = loading || saving || !doc || history.length === 0;
  const redoDisabled = loading || saving || !doc || future.length === 0;
  const saveDisabled = loading || saving || !doc || !task || !dirty;
  const submitDisabled = loading || saving || !doc || !task;
  const moveDisabled = loading || saving || !doc || !task;

  useEffect(() => {
    const onKeydown = (evt: KeyboardEvent) => {
      const target = evt.target as HTMLElement | null;
      const tag = target?.tagName;
      const isTyping =
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        (target as any)?.isContentEditable;

      if ((evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === "s") {
        evt.preventDefault();
        void handleSave();
        return;
      }
      if ((evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === "z") {
        evt.preventDefault();
        handleUndo();
        return;
      }
      if ((evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === "y") {
        evt.preventDefault();
        handleRedo();
        return;
      }
      if (isTyping) return;
      if (evt.key.toLowerCase() === "n") {
        evt.preventDefault();
        handleAddDetection();
        return;
      }
      if (evt.key === "Delete") {
        evt.preventDefault();
        handleRemoveDetection();
      }
    };
    window.addEventListener("keydown", onKeydown);
    return () => window.removeEventListener("keydown", onKeydown);
  }, [
    handleAddDetection,
    handleRedo,
    handleRemoveDetection,
    handleSave,
    handleUndo,
  ]);

  const goToClampedIndex = useCallback(
    (value: string | number) => {
      if (!task) return;
      const n = Number(value);
      if (!Number.isFinite(n)) return;
      const clamped = Math.min(Math.max(Math.trunc(n), 1), task.total_tasks);
      confirmAndNavigate(`/healthy/${clamped}`);
    },
    [confirmAndNavigate, task]
  );

  const imageUrl = useMemo(() => {
    const url = task?.image_url ?? "";
    if (!url) return "";
    if (url.startsWith("/")) return `${baseUrl.replace(/\/$/, "")}${url}`;
    return url;
  }, [task]);

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/70 backdrop-blur supports-[backdrop-filter]:bg-white/60">
        <div className="mx-auto max-w-7xl px-4 py-3 sm:px-6">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div className="flex items-start gap-3">
              <div className="mt-0.5 hidden sm:block">
                <div className="flex h-10 items-center gap-2">
                  <img
                    src={withBase("/logos/NTOU_Logo.png")}
                    alt="NTOU Logo"
                    className="h-8 w-auto object-contain"
                    loading="lazy"
                    decoding="async"
                  />
                  <img
                    src={withBase("/logos/NPUST_Logo.png")}
                    alt="NPUST Logo"
                    className="h-8 w-auto object-contain"
                    loading="lazy"
                    decoding="async"
                  />
                </div>
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-xl font-semibold tracking-tight text-slate-800">
                    <span className="bg-gradient-to-r from-slate-800 via-slate-900 to-slate-700 bg-clip-text text-transparent">
                      健康標註
                    </span>
                  </h1>
                  <span className="inline-flex items-center rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-700">
                    {dataset ?? "未選擇"}
                  </span>
                </div>
                <p className="mt-0.5 text-xs text-slate-500">
                  使用者：{name ?? "-"}
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={() => confirmAndNavigate("/datasets")}
                className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
              >
                <ArrowLeft className="h-4 w-4" />
                返回資料集
              </button>
              <button
                onClick={() => confirmAndNavigate("/healthy")}
                className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
              >
                健康影像
              </button>
              <button
                onClick={() => confirmAndNavigate("/annotate")}
                className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
              >
                回到標註
              </button>
            </div>
          </div>

          <div className="mt-3 hidden md:block">
            <Separator />
          </div>

          <div className="mt-3 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white p-1 shadow-sm">
              <IconButton
                onClick={handleAddDetection}
                disabled={addDisabled}
                label="新增框"
                shortcut="N"
              >
                <SquarePlus className="h-4 w-4" />
              </IconButton>
              <IconButton
                onClick={handleRemoveDetection}
                disabled={removeDisabled}
                label="刪除框"
                shortcut="Del"
              >
                <Trash2 className="h-4 w-4" />
              </IconButton>

              <Separator vertical />

              <IconButton
                onClick={handleUndo}
                disabled={undoDisabled}
                label="復原"
                shortcut="Ctrl+Z"
              >
                <Undo2 className="h-4 w-4" />
              </IconButton>
              <IconButton
                onClick={handleRedo}
                disabled={redoDisabled}
                label="重做"
                shortcut="Ctrl+Y"
              >
                <Redo2 className="h-4 w-4" />
              </IconButton>

              <Separator vertical />

              <IconButton
                onClick={handleSave}
                disabled={saveDisabled}
                label="保存"
                shortcut="Ctrl+S"
              >
                <Save className="h-4 w-4" />
              </IconButton>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <div className="hidden items-center gap-2 text-xs text-slate-500 md:flex">
                <span className="inline-flex items-center gap-1">新增<Kbd>N</Kbd></span>
                <span className="inline-flex items-center gap-1">刪除<Kbd>Del</Kbd></span>
                <span className="inline-flex items-center gap-1">復原<Kbd>Ctrl+Z</Kbd></span>
                <span className="inline-flex items-center gap-1">重做<Kbd>Ctrl+Y</Kbd></span>
                <span className="inline-flex items-center gap-1">保存<Kbd>Ctrl+S</Kbd></span>
              </div>

              <Separator vertical className="hidden md:block" />

              <button
                type="button"
                onClick={handleMoveToImages}
                className="inline-flex items-center justify-center rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-500 disabled:pointer-events-none disabled:opacity-50"
                disabled={moveDisabled}
                title="移回 /images"
              >
                移回 /images
              </button>

              <button
                type="button"
                onClick={handleSubmit}
                className="inline-flex items-center justify-center rounded-md border border-transparent bg-sky-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-sky-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 disabled:pointer-events-none disabled:opacity-50"
                disabled={submitDisabled}
              >
                提交
              </button>
            </div>
          </div>

          {error && (
            <Banner kind="error" onClose={() => setError(null)}>
              {error}
            </Banner>
          )}
          {validationErrors.length > 0 && (
            <Banner kind="warning">
              提交失敗：共有 {validationErrors.length} 項欄位未填或有誤。
              <button
                type="button"
                onClick={jumpToFirstError}
                className="ml-2 underline underline-offset-2"
              >
                前往
              </button>
            </Banner>
          )}
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-4 px-4 py-6 lg:grid-cols-12 sm:px-6">
        <section className="min-w-0 lg:col-span-7 flex flex-col gap-4">
          {task && (
            <div className="rounded-xl bg-white p-3 shadow flex flex-col gap-3">
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  goToClampedIndex(gotoIndex);
                }}
                className="flex w-full items-stretch rounded-full ring-1 ring-slate-300 bg-white overflow-hidden
                 focus-within:ring-2 focus-within:ring-sky-500 divide-x divide-slate-200
                 text-sm text-slate-600"
              >
                <span className="shrink-0 px-3 flex items-center">編號</span>
                <div
                  className={`flex items-center flex-1 min-w-0 ${
                    loading || saving ? "opacity-60 pointer-events-none" : ""
                  }`}
                >
                  <button
                    type="button"
                    onClick={() =>
                      goToClampedIndex((Number(gotoIndex || task.index) || 1) - 1)
                    }
                    aria-label="上一張"
                    disabled={loading || saving || Number(gotoIndex || task.index) <= 1}
                    className="px-2.5 py-1.5 hover:bg-slate-50 disabled:opacity-40"
                  >
                    <svg viewBox="0 0 20 20" className="h-4 w-4" fill="currentColor" aria-hidden="true">
                      <path d="M12.5 15l-5-5 5-5" />
                    </svg>
                  </button>

                  <div className="relative flex-1 min-w-0">
                    <input
                      type="number"
                      inputMode="numeric"
                      step={1}
                      min={1}
                      max={task.total_tasks}
                      value={gotoIndex}
                      onChange={(e) => setGotoIndex(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          goToClampedIndex(gotoIndex);
                        }
                      }}
                      aria-label="前往編號"
                      disabled={loading || saving}
                      className="w-full text-center bg-transparent px-2 py-1.5 outline-none
                       [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none
                       font-mono tabular-nums pr-12"
                      style={{
                        paddingRight: `calc(1.75rem + ${String(task.total_tasks).length}ch)`,
                      }}
                    />
                    <span
                      aria-hidden="true"
                      className="pointer-events-none absolute inset-y-0 right-2 flex items-center text-slate-500 whitespace-nowrap font-mono tabular-nums"
                    >
                      / {task.total_tasks}
                    </span>
                  </div>

                  <button
                    type="button"
                    onClick={() =>
                      goToClampedIndex((Number(gotoIndex || task.index) || 1) + 1)
                    }
                    aria-label="下一張"
                    disabled={loading || saving || Number(gotoIndex || task.index) >= task.total_tasks}
                    className="px-2.5 py-1.5 hover:bg-slate-50 disabled:opacity-40"
                  >
                    <svg viewBox="0 0 20 20" className="h-4 w-4" fill="currentColor" aria-hidden="true">
                      <path d="M7.5 5l5 5-5 5" />
                    </svg>
                  </button>
                </div>

                <button
                  type="submit"
                  disabled={
                    loading ||
                    saving ||
                    !gotoIndex ||
                    Number(gotoIndex) < 1 ||
                    Number(gotoIndex) > task.total_tasks
                  }
                  className="px-3 py-1.5 bg-sky-600 text-white hover:bg-sky-700 disabled:bg-sky-300 transition-colors shrink-0"
                >
                  前往
                </button>
              </form>
            </div>
          )}

          <div className="rounded-xl bg-white p-4 shadow">
            {loading && <p className="text-slate-500">載入任務...</p>}
            {!loading && !doc && (
              <p className="text-slate-500">目前沒有可用任務。</p>
            )}
            {doc && (
              <AnnotationCanvas
                imageUrl={imageUrl}
                imageWidth={doc.image_width}
                imageHeight={doc.image_height}
                detections={doc.detections}
                selectedIndex={selectedIndex}
                onSelect={handleSelectDetection}
                onUpdate={handleUpdateBox}
                getDisplayLabel={getDisplayLabel}
              />
            )}
          </div>
        </section>

        <aside className="min-w-0 lg:col-span-5 flex flex-col gap-4">
          {doc && (
            <div className="rounded-xl bg-white p-4 shadow">
              <h2 className="mb-3 text-lg font-semibold text-slate-800">標註清單</h2>
              <div
                role="listbox"
                aria-label="標註清單"
                className="flex max-h-80 flex-col gap-2 overflow-y-auto"
              >
                {doc.detections.map((det, idx) => {
                  const errCount = detectionErrorCounts[idx] ?? 0;
                  const selected = idx === selectedIndex;
                  const itemClass = selected
                    ? "border-sky-500 bg-sky-50 text-sky-700"
                    : errCount > 0
                      ? "border-rose-300 bg-rose-50 text-rose-700"
                      : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50";
                  return (
                    <button
                      key={idx}
                      ref={(el) => {
                        listItemRefs.current[idx] = el;
                      }}
                      onClick={() => handleSelectDetection(idx)}
                      className={`flex items-center justify-between rounded border px-3 py-2 text-left text-sm ${itemClass}`}
                    >
                      <span className="flex items-center gap-2">
                        {getDisplayLabel((det as any).label) || `框${idx + 1}`}
                        {errCount > 0 && (
                          <span className="inline-flex items-center rounded-full bg-rose-100 px-2 py-0.5 text-[11px] font-medium text-rose-700">
                            {errCount}
                          </span>
                        )}
                      </span>
                      <span className="text-xs text-slate-400">
                        {det.box_xyxy.join(", ")}
                      </span>
                    </button>
                  );
                })}
                {!doc.detections.length && (
                  <p className="rounded border border-dashed border-slate-300 px-3 py-4 text-center text-sm text-slate-500">
                    尚無框。可按 N 新增。
                  </p>
                )}
              </div>
            </div>
          )}

          {selectedDetection && selectedIndex != null && (
            <div className="rounded-xl bg-white p-4 shadow">
              <h2 className="mb-3 text-lg font-semibold text-slate-800">框設定</h2>
              <label className="mb-2 block text-sm font-medium text-slate-600">表徵類別</label>
              <select
                ref={(el) => {
                  labelRef.current = el;
                }}
                value={(selectedDetection.label ?? "").trim()}
                onChange={(e) => handleDetectionLabel(selectedIndex, e.target.value)}
                className={`w-full rounded border px-3 py-2 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 ${
                  detectionErrors.get(`detections.${selectedIndex}.label`)
                    ? "border-red-400"
                    : "border-slate-300"
                }`}
              >
                <option value="" disabled>請選擇表徵類別</option>
                {(selectedDetection.label ?? "").trim() &&
                  (selectedDetection.label ?? "").trim() !== HEALTHY_LABEL && (
                    <option value={(selectedDetection.label ?? "").trim()} disabled>
                      目前：{(selectedDetection.label ?? "").trim()}
                    </option>
                  )}
                <option value={HEALTHY_LABEL}>{HEALTHY_LABEL_DISPLAY}</option>
              </select>
              {detectionErrors.get(`detections.${selectedIndex}.label`) && (
                <p className="mt-2 text-sm text-red-500">
                  {detectionErrors.get(`detections.${selectedIndex}.label`)}
                </p>
              )}
            </div>
          )}

          {doc && (
            <div className="rounded-xl bg-white p-4 shadow">
              <h2 className="mb-3 text-lg font-semibold text-slate-800">註解</h2>
              <div className="flex gap-2 mb-2">
                <input
                  type="text"
                  value={commentDraft}
                  onChange={(e) => setCommentDraft(e.target.value)}
                  className="flex-1 rounded border border-slate-300 px-3 py-2 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
                  placeholder="新增註解"
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      handleAddComment();
                    }
                  }}
                />
                <button
                  onClick={handleAddComment}
                  className="rounded bg-slate-800 px-3 py-2 text-white hover:bg-slate-900 disabled:opacity-50"
                  disabled={!commentDraft.trim()}
                >
                  新增
                </button>
              </div>

              <div className="flex flex-col gap-2 max-h-64 overflow-y-auto text-sm">
                {((doc as any).comments ?? []).length === 0 && (
                  <p className="rounded border border-dashed border-slate-300 px-3 py-4 text-center text-sm text-slate-500">
                    尚無註解。
                  </p>
                )}
                {(((doc as any).comments ?? []) as TaskComment[]).map((c, idx) => (
                  <div
                    key={`c-${idx}`}
                    className="flex items-center justify-between rounded border border-slate-200 px-3 py-2"
                  >
                    <div className="min-w-0 mr-2">
                      <p className="text-slate-800 truncate">{c.text}</p>
                      <p className="text-xs text-slate-500 mt-0.5">
                        {c.author}
                        {c.created_at ? ` · ${new Date(c.created_at).toLocaleString()}` : ""}
                      </p>
                    </div>
                    <button
                      className="rounded border border-rose-200 px-2 py-1 text-xs text-rose-600 hover:bg-rose-50"
                      onClick={() => handleRemoveComment(idx)}
                    >
                      移除
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </aside>
      </main>
    </div>
  );
};

export default HealthyAnnotationPage;
