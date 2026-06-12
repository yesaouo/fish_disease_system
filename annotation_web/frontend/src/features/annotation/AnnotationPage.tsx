import React from "react";
import { AxiosError } from "axios";
import { useCallback, useEffect, useMemo, useReducer, useState, useRef } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import { fetchNextTask, submitTask, fetchClasses, fetchLabelMapZh, fetchEvidenceOptionsZh, fetchTaskByIndex, saveTask, moveImageToHealthyImages } from "../../api/client";
import type {
  NextTaskResponse,
  TaskDocument
} from "../../api/types";
import { useAuth } from "../../context/AuthContext";
import { useDataset } from "../../context/DatasetContext";
import {
  ValidationError,
  cloneTaskDocument,
  defaultDetection,
  documentsEqual,
  ensureSingleLine,
  normalizeBox,
  validateTaskDocument
} from "../../lib/taskUtils";
import AnnotationCanvas from "./components/AnnotationCanvas";
import type { Comment as TaskComment } from "../../api/types";
import {
  annotationReducer,
  initialAnnotationState,
} from "./shared/annotationReducer";
import { Kbd, IconButton, Separator, Banner, SubmissionCapsules } from "./shared/ui";
import { useNavBlocker } from "./shared/useNavBlocker";

// Icons
import {
  SquarePlus,
  Trash2,
  Undo2,
  Redo2,
  Save,
  ArrowLeft,
  MessageSquareQuote,
  CheckCheck,
  Images,
  HeartPulse,
  Ban,
  ChevronUp,
  ChevronDown,
} from "lucide-react";

const AnnotationPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const baseUrl = import.meta.env.BASE_URL || "/";
  const params = useParams();
  const routeIndex = useMemo(() => {
    const raw = params.index;
    if (!raw) return null;
    const n = Number(raw);
    return Number.isFinite(n) && n > 0 ? Math.floor(n) : null;
  }, [params.index]);
  const [gotoIndex, setGotoIndex] = useState<string>("");
  useEffect(() => {
    setGotoIndex(routeIndex != null ? String(routeIndex) : "");
  }, [routeIndex]);
  const { dataset, classes, setClasses } = useDataset();
  const { name, isExpert } = useAuth();

  // reducer state
  const [state, dispatch] = useReducer(annotationReducer, initialAnnotationState);

  const { doc, history, future, selectedIndex, validationErrors } = state;

  const [task, setTask] = useState<NextTaskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [labelMapZh, setLabelMapZh] = useState<Record<string, string>>({});
  const [evidenceOptionsZh, setEvidenceOptionsZh] = useState<Record<string, string[]>>({});
  // 以 symptoms.json 派生的類別清單供應標籤；外觀敘述改為下拉選單（顯示中文、保存 index）
  const [commentDraft, setCommentDraft] = useState("");

  // 頂部導覽列：捲動時自動縮起以節省版面；使用者手動切換後在捲回頂端前不再自動變動
  const [navCollapsed, setNavCollapsed] = useState(false);
  const navManualRef = useRef(false);
  useEffect(() => {
    const onScroll = () => {
      const y = window.scrollY;
      if (y <= 4) navManualRef.current = false; // 回到頂端後恢復自動行為
      if (navManualRef.current) return;
      // 遲滯（hysteresis）：收合與展開用不同門檻，避免收合改變高度後在臨界值來回抖動
      setNavCollapsed((prev) => {
        if (!prev && y > 96) return true;
        if (prev && y < 32) return false;
        return prev;
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
  const toggleNav = useCallback(() => {
    navManualRef.current = true;
    setNavCollapsed((v) => !v);
  }, []);

  // 自動在 6 秒後清除 server/network 錯誤訊息（不影響驗證錯誤 Banner）
  useEffect(() => {
    if (!error) return;
    const t = setTimeout(() => setError(null), 6000);
    return () => clearTimeout(t);
  }, [error]);

  // Refs for error-jump UX
  const labelRef = useRef<HTMLSelectElement | null>(null);
  const evidenceRef = useRef<HTMLSelectElement | null>(null);
  const listItemRefs = useRef<Record<number, HTMLButtonElement | null>>({});

  // ✅ 只在我們允許的短時間內放行導頁（例如提交成功後）
  const allowNavRef = useRef(false);
  const runWithBypass = useCallback(async (fn: () => Promise<void> | void) => {
    allowNavRef.current = true;
    try { await fn(); } finally { allowNavRef.current = false; }
  }, []);
  // 404/派發完畢時只跳轉到編號 1 一次，避免卡在重導 loop
  const exhaustedRedirectedRef = useRef(false);
  useEffect(() => { exhaustedRedirectedRef.current = false; }, [dataset]);

  // ✅ 針對所有「我們自己呼叫的 navigate」做前置確認
  const dirtyRef = useRef(false);
  useEffect(() => { dirtyRef.current = !!(task && doc && !documentsEqual(doc, task.task)); }, [doc, task]);
  const confirmAndNavigate = useCallback(
    (to: string, options?: { replace?: boolean; state?: any }) => {
      if (dirtyRef.current) {
        const ok = window.confirm("目前有未儲存的變更，離開將放棄這些變更。確定要離開嗎？");
        if (!ok) return;
      }
      // 放行這一次（避免再被 useNavBlocker 二次彈窗）
      runWithBypass(() => navigate(to, options));
    },
    [navigate, runWithBypass]
  );

  const loadTask = useCallback(async () => {
    if (!dataset) return;
    setLoading(true);
    setError(null);
    try {
      let resp: NextTaskResponse;
      if (routeIndex != null) {
        resp = await fetchTaskByIndex(dataset, routeIndex, name ?? undefined, isExpert);
      } else {
        resp = await fetchNextTask(dataset, name ?? undefined, isExpert);
      }
      setTask(resp);
      if (routeIndex == null) setGotoIndex(String(resp.index));
      dispatch({ type: "LOAD_DOC", doc: resp.task });
      exhaustedRedirectedRef.current = false; // 取得新任務代表仍有資料，不需要 fallback 標記
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      const detail = axiosErr.response?.data?.detail;
      const shouldRedirectToFirst =
        axiosErr.response?.status === 404 &&
        !exhaustedRedirectedRef.current &&
        (detail === "沒有可用任務" || detail === "index out of range");

      if (shouldRedirectToFirst) {
        exhaustedRedirectedRef.current = true;
        window.alert("此資料集的任務已派發完畢。");
        await runWithBypass(() => navigate("/annotate/1", { replace: true }));
        return;
      }

      if (axiosErr.response?.status === 404) {
        setError("目前沒有可分派的任務或影像缺失。");
      } else {
        setError("取得任務失敗，請稍後再試。");
      }
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [dataset, name, isExpert, routeIndex, navigate, runWithBypass]);

  // 用 location.key 觸發載入
  useEffect(() => {
    if (location.pathname.startsWith("/annotate")) {
      void loadTask();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.key]);

  // Ensure dataset metadata (classes + zh mapping + evidence options) are loaded
  useEffect(() => {
    const loadDatasetMeta = async () => {
      if (!dataset) return;
      try {
        if (classes.length === 0) {
          const cls = await fetchClasses(dataset);
          setClasses(cls);
        }
        const [map, evidenceOpts] = await Promise.all([
          fetchLabelMapZh(dataset),
          fetchEvidenceOptionsZh(dataset),
        ]);
        setLabelMapZh(map);
        setEvidenceOptionsZh(evidenceOpts);
      } catch (err) {
        console.error("Failed to load classes/labels for dataset", dataset, err);
      }
    };
    void loadDatasetMeta();
  }, [dataset, classes.length, setClasses]);

  const getDisplayLabel = useCallback(
    (enLabel: string | undefined | null): string => {
      const key = (enLabel || "").trim();
      if (!key) return "";
      return labelMapZh[key] || key;
    },
    [labelMapZh]
  );

  // 移除自動套預設：由使用者選擇

  // 衍生 dirty（由目前 doc 與伺服器上的 task.task 比對）
  const dirty = useMemo(() => {
    if (!task || !doc) return false;
    return !documentsEqual(doc, task.task);
  }, [doc, task]);

  useEffect(() => {
    const handler = (evt: BeforeUnloadEvent) => {
      if (dirty) {
        evt.preventDefault();
        evt.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, [dirty]);

  // 單點更新工具：計算 next draft，若有變更則一次性提交到 reducer
  const updateDoc = useCallback(
    (
      mutator: (draft: TaskDocument) => void,
      selectIndex?: (draft: TaskDocument) => number | null
    ) => {
      if (!doc) return;
      const snapshot = cloneTaskDocument(doc);
      const draft = cloneTaskDocument(doc);
      mutator(draft);
      if (documentsEqual(snapshot, draft)) return;
      const nextSelected = selectIndex ? selectIndex(draft) : undefined;
      dispatch({ type: "APPLY_DOC", next: draft, nextSelected });
    },
    [doc]
  );

  const handleSelectDetection = (index: number) => {
    dispatch({ type: "SET_SELECTED", index });
  };

  const handleUpdateBox = (index: number, box: [number, number, number, number]) => {
    updateDoc((draft) => {
      if (!draft.detections[index]) return;
      draft.detections[index].box_xyxy = normalizeBox(
        box[0],
        box[1],
        box[2],
        box[3],
        draft.image_width,
        draft.image_height
      );
    });
  };

  const handleAddDetection = useCallback(() => {
    updateDoc(
      (draft) => {
        const prev =
          selectedIndex != null && draft.detections[selectedIndex]
            ? draft.detections[selectedIndex]
            : undefined;
        // Default to previous detection's label if exists; evidence must be selected each time
        const det = defaultDetection(
          draft.image_width,
          draft.image_height,
          (prev as any)?.label
        );
        draft.detections.push(det);
      },
      (draft) => draft.detections.length - 1
    );
  }, [updateDoc, selectedIndex]);

  const handleRemoveDetection = useCallback(() => {
    if (selectedIndex == null) return;
    updateDoc(
      (draft) => {
        draft.detections.splice(selectedIndex, 1);
      },
      (draft) => {
        if (!draft.detections.length) return null;
        return Math.min(selectedIndex, draft.detections.length - 1);
      }
    );
  }, [updateDoc, selectedIndex]);

  const detectionErrors = useMemo(() => {
    const map = new Map<string, string>();
    validationErrors.forEach((err) => {
      map.set(err.field, err.message);
    });
    return map;
  }, [validationErrors]);

  const detectionErrorCounts = useMemo(() => {
    const counts: Record<number, number> = {};
    validationErrors.forEach((e) => {
      const m = /^detections\.(\d+)\./.exec(e.field);
      if (m) {
        const idx = Number(m[1]);
        counts[idx] = (counts[idx] ?? 0) + 1;
      }
    });
    return counts;
  }, [validationErrors]);

  const jumpToFirstError = useCallback(() => {
    if (!validationErrors.length) return;
    // Find the first detection* error
    const first = validationErrors.find((e) => e.field.startsWith("detections."));
    if (!first) return;
    const m = /^detections\.(\d+)\.(.+)$/.exec(first.field);
    if (!m) return;
    const idx = Number(m[1]);
    const field = m[2]; // e.g., label or evidence_index
    dispatch({ type: "SET_SELECTED", index: idx });
    // Scroll list item into view
    const el = listItemRefs.current[idx];
    if (el) {
      el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
    // Focus appropriate control after DOM updates
    setTimeout(() => {
      if (field.includes("evidence_index")) {
        evidenceRef.current?.focus();
      } else if (field.includes("label")) {
        labelRef.current?.focus();
      } else {
        // default to first important control
        (labelRef.current || evidenceRef.current)?.focus();
      }
    }, 0);
  }, [validationErrors, dispatch]);

  const handleDetectionField = (
    index: number,
    field: "label" | "evidence_index",
    value: string
  ) => {
    updateDoc((draft) => {
      const target = draft.detections[index];
      if (!target) return;
      if (field === "label") {
        target.label = value;
        target.evidence_index = null;
      } else {
        if (!value) {
          target.evidence_index = null;
          return;
        }
        const parsed = Number.parseInt(value, 10);
        target.evidence_index = Number.isFinite(parsed) ? parsed : null;
      }
    });
  };

  const handleOverallChange = (
    field: "colloquial_zh" | "medical_zh",
    value: string
  ) => {
    updateDoc((draft) => {
      draft.overall[field] = ensureSingleLine(value);
    });
  };

  const handleGlobalListChange = (
    key: "global_causes_zh" | "global_treatments_zh",
    action: "add" | "remove" | "move",
    payload?: { value?: string; index?: number; direction?: -1 | 1 }
  ) => {
    updateDoc((draft) => {
      const list = draft[key];
      if (action === "add" && payload?.value) {
        const text = ensureSingleLine(payload.value);
        if (!text || list.includes(text) || list.length >= 10) return;
        list.push(text);
      }
      if (action === "remove" && payload?.index != null) {
        list.splice(payload.index, 1);
      }
      if (
        action === "move" &&
        payload?.index != null &&
        payload.direction != null
      ) {
        const from = payload.index;
        const to = from + payload.direction;
        if (to < 0 || to >= list.length) return;
        const [item] = list.splice(from, 1);
        list.splice(to, 0, item);
      }
    });
  };

  const handleAddComment = () => {
    if (!name || !doc) return;
    const text = ensureSingleLine(commentDraft);
    if (!text) return;
    updateDoc((draft) => {
      const comments: TaskComment[] = (draft as any).comments ?? [];
      const entry: TaskComment = {
        author: name,
        text,
        created_at: new Date().toISOString()
      };
      (draft as any).comments = [...comments, entry];
    });
    setCommentDraft("");
  };

  const handleRemoveComment = (index: number) => {
    updateDoc((draft) => {
      const comments: TaskComment[] = (draft as any).comments ?? [];
      if (index < 0 || index >= comments.length) return;
      comments.splice(index, 1);
      (draft as any).comments = [...comments];
    });
  };

  const handleUndo = useCallback(() => {
    dispatch({ type: "UNDO" });
  }, []);

  const handleRedo = useCallback(() => {
    dispatch({ type: "REDO" });
  }, []);

  // ✅ 只有提交成功時才由系統發新的一張（放行一次，不提示）
  const goNext = () => {
    navigate(`/annotate?refresh=${Date.now()}`, { replace: true });
  };

  const handleSubmit = async () => {
    if (!doc || !task || !dataset || !name) return;
    const confirm = window.confirm("確定要送出標註嗎？送出後此影像將視為完成，不會再次分派。");
    if (!confirm) return;
    const errors = validateTaskDocument(doc, classes, false);
    const hasComments = ((doc as any).comments ?? []).length > 0;
    if (errors.length && !hasComments) {
      dispatch({ type: "SET_ERRORS", errors });
      // Move selection and focus to the first error to guide user
      setTimeout(() => jumpToFirstError(), 0);
      return;
    }
    setSaving(true);
    dispatch({ type: "RESET_ERRORS" });
    try {
      await submitTask(task.task_id, {
        full_json: doc,
        editor_name: name,
        is_expert: isExpert
      });

      // 原本邏輯：提交後由系統派發下一張
      // await runWithBypass(() => goNext());

      // 新邏輯：提交後前往「下一個編號」
      const currentIdx = routeIndex != null ? routeIndex : task.index;
      const nextIdx = (currentIdx ?? 0) + 1;
      await runWithBypass(() => navigate(`/annotate/${nextIdx}`, { replace: true }));

    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      if (axiosErr.response?.status === 409) {
        // 樂觀並發衝突：留在原地，由使用者決定是否重新載入。
        setError(axiosErr.response.data?.detail || "此任務已被更新，請重新載入後再送出。");
      } else if (axiosErr.response?.data?.detail) {
        setError(axiosErr.response.data.detail);
      } else {
        setError("提交失敗，請稍後再試。");
      }
    } finally {
      setSaving(false);
    }
  };

  const handleMoveToHealthyImages = async () => {
    if (!task || !dataset || !doc) return;

    // Warn if there are unsaved changes.
    if (dirty) {
      const ok = window.confirm("目前有未保存的修改。仍要將這張影像判定為健康嗎？");
      if (!ok) return;
    }

    const confirm = window.confirm(
      "確定要將這張影像判定為健康嗎？判定後會移到健康影像集。"
    );
    if (!confirm) return;

    setSaving(true);
    dispatch({ type: "RESET_ERRORS" });

    try {
      await moveImageToHealthyImages(dataset, doc.image_filename);
      const currentIdx = routeIndex != null ? routeIndex : task.index;
      // Reload the same index; after removal, the next image becomes this index.
      await runWithBypass(() => navigate(`/annotate/${currentIdx}`, { replace: true }));
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      if (axiosErr.response?.data?.detail) {
        setError(axiosErr.response.data.detail);
      } else {
        setError("移動影像失敗。");
      }
    } finally {
      setSaving(false);
    }
  };

  // 標記為「無法使用」：自動加上一則註解後提交（沿用 comment 可跳過欄位驗證的流程）
  const handleMarkUnusable = async () => {
    if (!doc || !task || !dataset || !name) return;
    const confirm = window.confirm("確定要標記為「無法使用」並提交嗎？提交後此影像將視為完成，不會再次分派。");
    if (!confirm) return;
    setSaving(true);
    dispatch({ type: "RESET_ERRORS" });
    try {
      const submitDoc = cloneTaskDocument(doc);
      const comments: TaskComment[] = (submitDoc as any).comments ?? [];
      (submitDoc as any).comments = [
        ...comments,
        { author: name, text: "無法使用", created_at: new Date().toISOString() },
      ];
      await submitTask(task.task_id, {
        full_json: submitDoc,
        editor_name: name,
        is_expert: isExpert,
      });
      const currentIdx = routeIndex != null ? routeIndex : task.index;
      const nextIdx = (currentIdx ?? 0) + 1;
      await runWithBypass(() => navigate(`/annotate/${nextIdx}`, { replace: true }));
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      if (axiosErr.response?.status === 409) {
        setError(axiosErr.response.data?.detail || "此任務已被更新，請重新載入後再送出。");
      } else if (axiosErr.response?.data?.detail) {
        setError(axiosErr.response.data.detail);
      } else {
        setError("提交失敗，請稍後再試。");
      }
    } finally {
      setSaving(false);
    }
  };

  const handleSave = async () => {
    if (!doc || !task || !dataset || !name) return;
    setSaving(true);
    dispatch({ type: "RESET_ERRORS" });
    try {
      const resp = await saveTask(task.task_id, {
        full_json: doc,
        editor_name: name,
        is_expert: isExpert
      });

      // ✅ 把伺服器新發的 version 寫回 doc 與 baseline，
      // 下一次儲存才會用對的 version 做樂觀並發檢查（不清空 history/future）。
      const nextVersion = resp.version ?? ((doc.version ?? 0) + 1);
      dispatch({ type: "SET_VERSION", version: nextVersion });
      const savedDoc: TaskDocument = { ...cloneTaskDocument(doc), version: nextVersion };
      setTask((prev) => (prev ? { ...prev, task: savedDoc } : prev));
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      if (axiosErr.response?.status === 409) {
        setError(axiosErr.response.data?.detail || "此任務已被更新，請重新載入後再儲存。");
      } else if (axiosErr.response?.data?.detail) {
        setError(axiosErr.response.data.detail);
      } else {
        setError("保存失敗，請稍後再試。");
      }
    } finally {
      setSaving(false);
    }
  };

  // 左右方向鍵切換前後張（與膠囊列的上一個 / 下一個按鈕同邏輯）
  const handlePrevImage = useCallback(() => {
    if (!task) return;
    const target = (Number(gotoIndex || task.index) || 1) - 1;
    if (target < 1) return;
    confirmAndNavigate(`/annotate/${target}`);
  }, [task, gotoIndex, confirmAndNavigate]);

  const handleNextImage = useCallback(() => {
    if (!task) return;
    const total = task.total_tasks ?? 0;
    const target = (Number(gotoIndex || task.index) || 1) + 1;
    if (total > 0 && target > total) return;
    confirmAndNavigate(`/annotate/${target}`);
  }, [task, gotoIndex, confirmAndNavigate]);

  useEffect(() => {
    const onKeydown = (evt: KeyboardEvent) => {
      const target = evt.target as HTMLElement | null;
      const tag = target?.tagName;

      // 先攔 Ctrl/Cmd+S：就算在輸入框/可編輯元素內也要生效
      if ((evt.ctrlKey || evt.metaKey) && evt.key.toLowerCase() === "s") {
        evt.preventDefault(); // 避免瀏覽器另存新檔
        void handleSave();
        return;
      }

      // 焦點在可編輯元素：其餘快捷鍵一律不攔（保留輸入體驗）
      const inEditable =
        (tag && ["INPUT", "TEXTAREA", "SELECT"].includes(tag)) ||
        (target?.isContentEditable ?? false);
      if (inEditable) return;

      // 若按了 Alt/Shift/Control/Meta 的組合鍵且不是我們要的，直接略過
      const hasModifier = evt.altKey || evt.shiftKey || evt.ctrlKey || evt.metaKey;

      // Undo / Redo
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

      if (hasModifier) return;

      switch (evt.key) {
        case "n":
        case "N":
          evt.preventDefault();
          handleAddDetection();
          break;
        case "Delete":
        case "Backspace":
          if (selectedIndex != null) { // 避免不必要攔截
            evt.preventDefault();
            handleRemoveDetection();
          }
          break;
        case "ArrowLeft":
          evt.preventDefault();
          handlePrevImage();
          break;
        case "ArrowRight":
          evt.preventDefault();
          handleNextImage();
          break;
        default:
          break;
      }
    };
    window.addEventListener("keydown", onKeydown);
    return () => window.removeEventListener("keydown", onKeydown);
  }, [handleUndo, handleRedo, handleAddDetection, handleRemoveDetection, handleSave, handlePrevImage, handleNextImage, selectedIndex]);

  // ✅ 全域擋：除了我們明確放行（allowNavRef）以外，dirty 時一律跳提醒
  useNavBlocker(dirty, () => allowNavRef.current);

  const selectedDetection = useMemo(() => {
    return selectedIndex != null && doc
      ? doc.detections[selectedIndex]
      : null;
  }, [selectedIndex, doc]);

  const handleBackToDatasets = () => {
    confirmAndNavigate("/datasets");
  };

  const goToClampedIndex = useCallback((raw: string | number) => {
    if (raw === "" || raw === null || raw === undefined) return;
    const total = task?.total_tasks ?? 0;

    let n = Math.floor(Number(raw));
    if (!Number.isFinite(n)) return;

    const clamped = total > 0 ? Math.min(Math.max(1, n), total) : Math.max(1, n);
    confirmAndNavigate(`/annotate/${clamped}`);
  }, [confirmAndNavigate, task?.total_tasks]);

  // ===== Header 狀態（disabled） =====
  const addDisabled = !!(loading || saving);
  const removeDisabled = !!(loading || saving || selectedIndex == null);
  const undoDisabled = !!(loading || saving || !history?.length);
  const redoDisabled = !!(loading || saving || !future?.length);
  const saveDisabled = !!(loading || saving || !doc || !task);
  const submitDisabled = !!(loading || saving || !doc || !task);
  const moveToHealthyDisabled = !!(loading || saving || !doc || !task);

  const withBase = (p: string) => (p.startsWith("/") ? `${baseUrl.replace(/\/$/, "")}${p}` : p);

  return (
    <div className="flex min-h-screen flex-col bg-slate-100">
      {/* ======== Header ======== */}
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/70 backdrop-blur supports-[backdrop-filter]:bg-white/60">
        <div className="mx-auto max-w-7xl px-4 py-3 sm:px-6">
          {/* 可收合區：標題與導覽（捲動時自動縮起以節省版面） */}
          <div
            className={`overflow-hidden transition-all duration-300 ease-in-out ${
              navCollapsed ? "max-h-0 opacity-0" : "max-h-96 opacity-100"
            }`}
            aria-hidden={navCollapsed}
          >
          {/* 上層：標題與導覽 */}
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            {/* 左側：任務資訊 */}
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
                    <span className="bg-gradient-to-r from-slate-800 via-slate-900 to-slate-700 bg-clip-text text-transparent">標註任務</span>
                  </h1>
                  <span className="inline-flex items-center rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-xs text-slate-700">
                    {dataset ?? "未選擇"}
                  </span>
                </div>
                <p className="mt-0.5 text-xs text-slate-500">使用者：{name ?? "-"}</p>
              </div>
            </div>

            {/* 右側：導覽按鈕群 */}
            <div className="flex flex-wrap items-center gap-2">
              <button
                onClick={handleBackToDatasets}
                className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
              >
                <ArrowLeft className="h-4 w-4" />
                返回資料集
              </button>
              <button
                onClick={() => confirmAndNavigate("/commented")}
                className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
              >
                <MessageSquareQuote className="h-4 w-4" />
                查看註解
              </button>
              <button
                onClick={() => confirmAndNavigate("/annotated")}
                className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
              >
                <CheckCheck className="h-4 w-4" />
                查看提交
              </button>
              <button
                onClick={() => confirmAndNavigate("/healthy")}
                className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
              >
                <Images className="h-4 w-4" />
                健康影像
              </button>
            </div>
          </div>

          {/* 分隔線 */}
          <div className="mt-3 hidden md:block"><Separator /></div>
          </div>

          {/* 下層：工具列（Icon-only）＋ 行為按鈕 */}
          <div className="mt-3 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            {/* Icon Toolbar */}
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

              <Separator vertical />

              <IconButton
                onClick={handleMoveToHealthyImages}
                disabled={moveToHealthyDisabled}
                label="判定為健康"
                className="border-emerald-300 bg-emerald-50 text-emerald-700 hover:bg-emerald-100"
              >
                <HeartPulse className="h-4 w-4" />
              </IconButton>

              <IconButton
                onClick={handleMarkUnusable}
                disabled={submitDisabled}
                label="無法使用並提交"
                className="border-amber-300 bg-amber-50 text-amber-700 hover:bg-amber-100"
              >
                <Ban className="h-4 w-4" />
              </IconButton>
            </div>

            {/* 右側：操作按鈕與快捷鍵提示 */}
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
                onClick={handleSubmit}
                className="inline-flex items-center justify-center rounded-md border border-transparent bg-sky-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-sky-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 disabled:pointer-events-none disabled:opacity-50"
                disabled={submitDisabled}
              >
                提交
              </button>

              <IconButton
                onClick={toggleNav}
                label={navCollapsed ? "展開頂部選單" : "收起頂部選單"}
              >
                {navCollapsed ? <ChevronDown className="h-4 w-4" /> : <ChevronUp className="h-4 w-4" />}
              </IconButton>
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
                className="ml-2 inline-flex items-center rounded border border-amber-300 bg-white px-2 py-0.5 text-xs text-amber-800 hover:bg-amber-100"
              >
                定位到第一個錯誤
              </button>
            </Banner>
          )}
        </div>
      </header>

      <main className="grid grow gap-4 px-6 py-6 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <section className="min-w-0 flex flex-col gap-4">
          {task && (
            <div className="rounded-xl bg-white p-3 shadow flex flex-col gap-3">
              {/* 🆕 顯示「用戶已提交 / 專家已提交」的膠囊狀態 */}
              {doc && (
                <SubmissionCapsules
                  generalEditor={(doc as any).general_editor}
                  expertEditor={(doc as any).expert_editor}
                />
              )}

              {/* 膠囊式控制列：編號/跳轉 */}
              <form
                onSubmit={(e) => { e.preventDefault(); goToClampedIndex(gotoIndex); }}
                className="flex w-full items-stretch rounded-full ring-1 ring-slate-300 bg-white overflow-hidden
                 focus-within:ring-2 focus-within:ring-sky-500 divide-x divide-slate-200
                 text-sm text-slate-600"
              >
                {/* 左段：標籤 */}
                <span className="shrink-0 px-3 flex items-center">編號</span>

                {/* 中段：上一個 / （同格的）輸入 + 總數 / 下一個 */}
                <div className={`flex items-center flex-1 min-w-0 ${ (loading || saving) ? "opacity-60 pointer-events-none" : ""}`}>
                  {/* 上一個 */}
                  <button
                    type="button"
                    onClick={() => goToClampedIndex((Number(gotoIndex || task.index) || 1) - 1)}
                    aria-label="上一個"
                    disabled={loading || saving || (Number(gotoIndex || task.index) <= 1)}
                    className="px-2.5 py-1.5 hover:bg-slate-50 disabled:opacity-40"
                  >
                    <svg viewBox="0 0 20 20" className="h-4 w-4" fill="currentColor" aria-hidden="true">
                      <path d="M12.5 15l-5-5 5-5" />
                    </svg>
                  </button>

                  {/* 同格：編號輸入 + 疊在右側的「/ 總數」 */}
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
                        if (e.key === "Enter") { e.preventDefault(); goToClampedIndex(gotoIndex); }
                      }}
                      aria-label="目前編號"
                      disabled={loading || saving}
                      className="w-full text-center bg-transparent px-2 py-1.5 outline-none
                       [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none
                       font-mono tabular-nums pr-12"
                      style={{ paddingRight: `calc(1.75rem + ${String(task.total_tasks).length}ch)` }}
                    />
                    <span
                      aria-hidden="true"
                      className="pointer-events-none absolute inset-y-0 right-2 flex items-center text-slate-500 whitespace-nowrap font-mono tabular-nums"
                    >
                      / {task.total_tasks}
                    </span>
                  </div>

                  {/* 下一個 */}
                  <button
                    type="button"
                    onClick={() => goToClampedIndex((Number(gotoIndex || task.index) || 1) + 1)}
                    aria-label="下一個"
                    disabled={loading || saving || (Number(gotoIndex || task.index) >= task.total_tasks)}
                    className="px-2.5 py-1.5 hover:bg-slate-50 disabled:opacity-40"
                  >
                    <svg viewBox="0 0 20 20" className="h-4 w-4" fill="currentColor" aria-hidden="true">
                      <path d="M7.5 5l5 5-5 5" />
                    </svg>
                  </button>
                </div>

                {/* 右段：前往（同一顆膠囊內） */}
                <button
                  type="submit"
                  disabled={loading || saving || !gotoIndex || Number(gotoIndex) < 1 || Number(gotoIndex) > task.total_tasks}
                  className="px-3 py-1.5 bg-sky-600 text-white hover:bg-sky-700 disabled:bg-sky-300
                   transition-colors shrink-0"
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
                imageUrl={(() => {
                  const url = task?.image_url ?? "";
                  if (!url) return "";
                  // Prefix with base for absolute paths like "/api/..."
                  if (url.startsWith("/")) {
                    return `${baseUrl.replace(/\/$/, "")}${url}`;
                  }
                  return url;
                })()}
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

          {doc && (
            <div className="grid gap-4">
              <div className="rounded-xl bg-white p-4 shadow">
                <h2 className="mb-3 text-lg font-semibold text-slate-800">
                  病徵敘述
                </h2>
                <label className="mb-2 block text-sm font-medium text-slate-600">
                  通俗描述
                </label>
                <textarea
                  rows={4}
                  value={doc.overall.colloquial_zh}
                  onChange={(e) =>
                    handleOverallChange("colloquial_zh", e.target.value)
                  }
                  onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); } }}
                  className={`mb-3 w-full min-h-[6rem] resize-y rounded border px-3 py-2 leading-relaxed focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 ${
                    detectionErrors.get("overall.colloquial_zh")
                      ? "border-red-400"
                      : "border-slate-300"
                  }`}
                  placeholder="請輸入口語描述（Enter 不換行）"
                />
                {detectionErrors.get("overall.colloquial_zh") && (
                  <p className="mb-3 text-sm text-red-500">
                    {detectionErrors.get("overall.colloquial_zh")}
                  </p>
                )}
                <label className="mb-2 block text-sm font-medium text-slate-600">
                  醫學描述
                </label>
                <textarea
                  rows={4}
                  value={doc.overall.medical_zh}
                  onChange={(e) =>
                    handleOverallChange("medical_zh", e.target.value)
                  }
                  onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); } }}
                  className={`w-full min-h-[6rem] resize-y rounded border px-3 py-2 leading-relaxed focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 ${
                    detectionErrors.get("overall.medical_zh")
                      ? "border-red-400"
                      : "border-slate-300"
                  }`}
                  placeholder="請輸入醫學描述（Enter 不換行）"
                />
                {detectionErrors.get("overall.medical_zh") && (
                  <p className="mt-2 text-sm text-red-500">
                    {detectionErrors.get("overall.medical_zh")}
                  </p>
                )}
              </div>

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
                          {c.author}{c.created_at ? ` · ${new Date(c.created_at).toLocaleString()}` : ""}
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
            </div>
          )}
        </section>

        {doc && (
          <aside className="min-w-0 flex flex-col gap-4">
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
                      : "border-slate-200 hover:border-sky-400";
                  return (
                    <button
                      key={idx}
                      ref={(el) => { listItemRefs.current[idx] = el; }}
                      onClick={() => handleSelectDetection(idx)}
                      className={`flex items-center justify-between rounded border px-3 py-2 text-left text-sm ${itemClass}`}
                    >
                      <span className="flex items-center gap-2">
                        {getDisplayLabel((det as any).label) || `框 ${idx + 1}`}
                        {errCount > 0 && (
                          <span className="inline-flex items-center rounded-full bg-rose-100 px-2 py-0.5 text-[11px] font-medium text-rose-700">
                            {errCount}
                          </span>
                        )}
                      </span>
                      <span className="text-xs text-slate-400">{det.box_xyxy.join(", ")}</span>
                    </button>
                  );
                })}
                {!doc.detections.length && (
                  <p className="rounded border border-dashed border-slate-300 px-3 py-4 text-center text-sm text-slate-500">
                    尚未新增任何框，按下「新增框」開始。
                  </p>
                )}
              </div>
            </div>

            {selectedDetection && selectedIndex != null && (
              <div className="rounded-xl bg-white p-4 shadow">
                <h2 className="mb-3 text-lg font-semibold text-slate-800">
                  標註內容
                </h2>
                <label className="mb-2 block text-sm font-medium text-slate-600">表徵類別</label>
                <select
                  ref={(el) => { labelRef.current = el; }}
                  value={selectedDetection.label ?? ""}
                  onChange={(e) => handleDetectionField(selectedIndex, "label", e.target.value)}
                  className={`mb-3 w-full rounded border px-3 py-2 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 ${
                    detectionErrors.get(`detections.${selectedIndex}.label`)
                      ? "border-red-400"
                      : "border-slate-300"
                  }`}
                >
                  <option value="" disabled>選擇類別</option>
                  {classes.map((cls) => (
                    <option key={cls} value={cls}>{getDisplayLabel(cls)}</option>
                  ))}
                </select>
                <label className="mb-2 block text-sm font-medium text-slate-600">外觀敘述</label>
                <select
                  ref={(el) => { evidenceRef.current = el; }}
                  value={selectedDetection.evidence_index == null ? "" : String(selectedDetection.evidence_index)}
                  onChange={(e) => handleDetectionField(selectedIndex, "evidence_index", e.target.value)}
                  disabled={!selectedDetection.label}
                  className={`w-full rounded border px-3 py-2 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 ${
                    detectionErrors.get(`detections.${selectedIndex}.evidence_index`)
                      ? "border-red-400"
                      : "border-slate-300"
                  }`}
                >
                  <option value="" disabled>
                    {!selectedDetection.label
                      ? "請先選擇類別"
                      : (evidenceOptionsZh[selectedDetection.label] || []).length
                        ? "選擇外觀敘述"
                        : "（此類別沒有外觀敘述選項）"}
                  </option>
                  {(evidenceOptionsZh[selectedDetection.label] || []).map((txt, i) => (
                    <option key={`${selectedDetection.label}-${i}`} value={i}>
                      {txt}
                    </option>
                  ))}
                </select>
                {detectionErrors.get(`detections.${selectedIndex}.evidence_index`) && (
                  <p className="mt-2 text-sm text-red-500">
                    {detectionErrors.get(`detections.${selectedIndex}.evidence_index`)}
                  </p>
                )}
              </div>
            )}

            <div className="rounded-xl bg-white p-4 shadow">
              <h2 className="mb-3 text-lg font-semibold text-slate-800">
                病徵原因（依發生機率排序）
              </h2>
              <GlobalListEditor
                items={doc.global_causes_zh}
                onChange={(action, payload) =>
                  handleGlobalListChange("global_causes_zh", action, payload)
                }
                error={validationErrors.find((e) =>
                  e.field.startsWith("global_causes_zh")
                )}
              />
            </div>

            <div className="rounded-xl bg-white p-4 shadow">
              <h2 className="mb-3 text-lg font-semibold text-slate-800">
                處置建議（依治療流程排序）
              </h2>
              <GlobalListEditor
                items={doc.global_treatments_zh}
                onChange={(action, payload) =>
                  handleGlobalListChange(
                    "global_treatments_zh",
                    action,
                    payload
                  )
                }
                error={validationErrors.find((e) =>
                  e.field.startsWith("global_treatments_zh")
                )}
              />
            </div>

          </aside>
        )}
      </main>
    </div>
  );
};

type GlobalListEditorProps = {
  items: string[];
  onChange: (
    action: "add" | "remove" | "move",
    payload?: { value?: string; index?: number; direction?: -1 | 1 }
  ) => void;
  error?: ValidationError;
};

const GlobalListEditor: React.FC<GlobalListEditorProps> = ({
  items,
  onChange,
  error
}) => {
  const [draft, setDraft] = useState("");
  return (
    <div className="flex flex-col gap-3 text-sm text-slate-700">
      <div className="flex gap-2">
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          className="flex-1 rounded border border-slate-300 px-3 py-2 focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
          placeholder="新增項目 (Enter)"
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              onChange("add", { value: draft });
              setDraft("");
            }
          }}
        />
        <button
          onClick={() => {
            onChange("add", { value: draft });
            setDraft("");
          }}
          className="rounded bg-slate-800 px-3 py-2 text-white hover:bg-slate-900"
        >
          新增
        </button>
      </div>
      <div className="flex flex-col gap-2">
        {items.map((item, idx) => (
          <div
            key={`${item}-${idx}`}
            className="flex items-center justify-between rounded border border-slate-200 px-3 py-2"
          >
            <span className="flex-1">{item}</span>
            <div className="flex items-center gap-1">
              <button
                className="rounded border border-slate-200 px-2 py-1 text-xs text-slate-500 hover:border-slate-400"
                onClick={() => onChange("move", { index: idx, direction: -1 })}
                disabled={idx === 0}
              >
                ▲
              </button>
              <button
                className="rounded border border-slate-200 px-2 py-1 text-xs text-slate-500 hover:border-slate-400"
                onClick={() => onChange("move", { index: idx, direction: 1 })}
                disabled={idx === items.length - 1}
              >
                ▼
              </button>
              <button
                className="rounded border border-rose-200 px-2 py-1 text-xs text-rose-600 hover:bg-rose-50"
                onClick={() => onChange("remove", { index: idx })}
              >
                刪除
              </button>
            </div>
          </div>
        ))}
        {!items.length && (
          <p className="rounded border border-dashed border-slate-300 px-3 py-4 text-center text-sm text-slate-500">
            尚未新增資料。
          </p>
        )}
      </div>
      {error && (
        <div className="rounded border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600">
          {error.message}
        </div>
      )}
    </div>
  );
};

export default AnnotationPage;
