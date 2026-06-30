import React from "react";
import { AxiosError } from "axios";
import { useCallback, useEffect, useMemo, useReducer, useState, useRef } from "react";
import { useNavigate, useParams, useLocation } from "react-router-dom";
import { fetchNextTask, submitTask, fetchClasses, fetchLabelMapZh, fetchEvidenceOptionsZh, fetchTaskByIndex, saveTask, moveImageToHealthyImages, importDiagnosisTask, diagnose, fetchDatasets, deleteDatasetTask, fetchGlobalSymptoms } from "../../api/client";
import type {
  NextTaskResponse,
  TaskDocument,
  DiagnoseResponse
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
  validateTaskDocument,
  reportLesionToDetection,
  invertLabelMap
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
  Sparkles,
  Loader2,
  Trash,
  Menu,
  X,
  Pencil,
} from "lucide-react";

type GlobalListTab = "causes" | "treatments";

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
  // 資料集改由 URL 路徑提供（/annotate/:dataset/...），不再依 context/localStorage。
  const dataset = params.dataset ?? null;
  const { classes, setClasses, setDataset } = useDataset();
  const { name, isExpert, canEdit } = useAuth();

  // 同步進 context，讓共用 context 的清單頁（/annotated、/commented）與類別載入照常運作。
  useEffect(() => {
    if (dataset) setDataset(dataset);
  }, [dataset, setDataset]);

  // 草稿模式（從診斷報告帶資料進來；只在按提交時才落地成資料集任務）。
  // location.state 在首次 render 取一次即可。
  const [draft, setDraft] = useState(() => {
    const s = location.state as
      | {
          draft?: boolean;
          dataset?: string;
          imageFile?: File;
          doc?: TaskDocument;
          overallSuggestions?: { source: string; colloquial: string; medical: string }[];
        }
      | null;
    return s?.draft && s.dataset && s.imageFile && s.doc
      ? {
          dataset: s.dataset,
          imageFile: s.imageFile,
          doc: s.doc,
          overallSuggestions: s.overallSuggestions ?? []
        }
      : null;
  });

  // 此資料集是否可寫（診斷建立的資料集才能刪除任務；官方鎖定資料集不可）。
  const [datasetWritable, setDatasetWritable] = useState(false);
  useEffect(() => {
    let alive = true;
    if (!dataset) {
      setDatasetWritable(false);
      return;
    }
    fetchDatasets()
      .then((list) => {
        if (alive) setDatasetWritable(list.find((d) => d.name === dataset)?.locked === false);
      })
      .catch(() => alive && setDatasetWritable(false));
    return () => {
      alive = false;
    };
  }, [dataset]);

  // 無待標註病灶影像（含「健康回寫資料集 images/ 為空」）時的友善空狀態。
  const [noTasks, setNoTasks] = useState(false);

  // AI 建議：對當前影像跑診斷，把建議的框/病因逐項合併進標註（不取代既有內容）。
  const [aiBusy, setAiBusy] = useState(false);
  const [aiError, setAiError] = useState<string | null>(null);
  const [aiSuggest, setAiSuggest] = useState<DiagnoseResponse | null>(null);
  // 尚未採用的建議病灶（畫布上以幽靈框呈現、面板逐列可採用）＋ hover 連動索引
  const [aiPendingLesions, setAiPendingLesions] = useState<DiagnoseResponse["lesions"]>([]);
  const [aiHover, setAiHover] = useState<number | null>(null);

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
  const [mobileEditorOpen, setMobileEditorOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [mobileCanvasOpen, setMobileCanvasOpen] = useState(false);
  const [mobileCanvasZoom, setMobileCanvasZoom] = useState(1);
  const [mobileCanvasFormOpen, setMobileCanvasFormOpen] = useState(false);
  const [isMobileViewport, setIsMobileViewport] = useState(false);
  const [globalListTab, setGlobalListTab] = useState<GlobalListTab>("causes");

  useEffect(() => {
    const query = window.matchMedia("(max-width: 767px)");
    const update = () => {
      setIsMobileViewport(query.matches);
      if (!query.matches) {
        setMobileCanvasOpen(false);
        setMobileCanvasFormOpen(false);
      }
    };
    update();
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  // 頂部導覽列：捲動時自動縮起以節省版面；使用者手動切換後在捲回頂端前不再自動變動
  const [navCollapsed, setNavCollapsed] = useState(false);
  const navManualRef = useRef(false);
  const navToggleAtRef = useRef(0);
  useEffect(() => {
    if (isMobileViewport) {
      setNavCollapsed(false);
      return;
    }
    const onScroll = () => {
      const y = window.scrollY;
      if (y <= 4) navManualRef.current = false; // 回到頂端後恢復自動行為
      if (navManualRef.current) return;
      // 收/放會改變版面高度，連帶觸發 scroll 事件；剛切換後短暫忽略，避免版面位移回授
      // 造成在臨界範圍內不斷上下收放。配合較寬的遲滯帶（20↔140）雙重防抖。
      if (Date.now() - navToggleAtRef.current < 400) return;
      setNavCollapsed((prev) => {
        if (!prev && y > 140) {
          navToggleAtRef.current = Date.now();
          return true;
        }
        if (prev && y < 20) {
          navToggleAtRef.current = Date.now();
          return false;
        }
        return prev;
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, [isMobileViewport]);
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
  const globalListPanelRef = useRef<HTMLDivElement | null>(null);
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
    if (draft) return; // 草稿模式不向伺服器取任務
    if (!dataset) return;
    // #1：以「派發模式」進入（無編號）時重置 fallback 旗標，讓每次進資料集都有一次
    // 繞回第 1 張的機會（修正全部標完的資料集重入時誤報「沒有可分派的任務」）。
    if (routeIndex == null) exhaustedRedirectedRef.current = false;
    setLoading(true);
    setError(null);
    setNoTasks(false);
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
        // 任務派發完畢/編號越界時靜默繞回第一張，不再彈出提示。
        exhaustedRedirectedRef.current = true;
        await runWithBypass(() => navigate(`/annotate/${dataset}/1`, { replace: true }));
        return;
      }

      if (axiosErr.response?.status === 404) {
        // #2：無待標註病灶影像（含只收過健康影像的回寫資料集）→ 友善空狀態而非錯誤。
        setNoTasks(true);
      } else {
        setError("取得任務失敗，請稍後再試。");
      }
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [dataset, name, isExpert, routeIndex, navigate, runWithBypass, draft]);

  // 草稿模式：建立 blob URL、初始化資料集 context、合成 task、灌入預填 doc。
  // blob URL 的 create/revoke 配對在同一 effect 生命週期，避免 React 18 StrictMode
  // 的「掛載→卸載→重掛載」把仍在使用中的 URL 提前 revoke（會導致影像載入失敗）。
  useEffect(() => {
    if (!draft) return;
    const url = URL.createObjectURL(draft.imageFile);
    setDataset(draft.dataset);
    const draftDoc = { ...draft.doc, dataset: draft.dataset };
    setTask({
      task_id: "draft",
      task: draftDoc,
      image_url: url,
      index: 0,
      total_tasks: 0
    });
    dispatch({ type: "LOAD_DOC", doc: draftDoc });
    return () => URL.revokeObjectURL(url);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft]);

  // 用 location.key 觸發載入（草稿模式不取）
  useEffect(() => {
    if (!draft && location.pathname.startsWith("/annotate")) {
      void loadTask();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.key]);

  // Ensure dataset metadata (classes + zh mapping + evidence options) are loaded.
  // 草稿模式目標資料集可能尚未建立 → 一律用全域 symptoms。
  useEffect(() => {
    const loadDatasetMeta = async () => {
      try {
        if (draft) {
          const g = await fetchGlobalSymptoms();
          setClasses(g.classes);
          setLabelMapZh(g.label_map_zh);
          setEvidenceOptionsZh(g.evidence_options_zh);
          return;
        }
        if (!dataset) return;
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
        console.error("Failed to load classes/labels", dataset, err);
      }
    };
    void loadDatasetMeta();
  }, [dataset, draft, classes.length, setClasses]);

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
    if (isMobileViewport && !mobileCanvasOpen) setMobileEditorOpen(true);
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
    if (isMobileViewport && !mobileCanvasOpen) setMobileEditorOpen(true);
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
  }, [updateDoc, selectedIndex, isMobileViewport, mobileCanvasOpen]);

  const handleRemoveDetection = useCallback(() => {
    if (selectedIndex == null) return;
    if ((doc?.detections.length ?? 0) <= 1) setMobileEditorOpen(false);
    updateDoc(
      (draft) => {
        draft.detections.splice(selectedIndex, 1);
      },
      (draft) => {
        if (!draft.detections.length) return null;
        return Math.min(selectedIndex, draft.detections.length - 1);
      }
    );
  }, [updateDoc, selectedIndex, doc?.detections.length]);

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

  const jumpToFirstError = useCallback((errorsOverride?: ValidationError[]) => {
    const currentErrors = errorsOverride ?? validationErrors;
    if (!currentErrors.length) return;
    // Find the first detection* error
    const firstDetection = currentErrors.find((e) => e.field.startsWith("detections."));
    if (firstDetection) {
      const m = /^detections\.(\d+)\.(.+)$/.exec(firstDetection.field);
      if (!m) return;
      const idx = Number(m[1]);
      const field = m[2]; // e.g., label or evidence_index
      dispatch({ type: "SET_SELECTED", index: idx });
      if (isMobileViewport) {
        setMobileCanvasOpen(false);
        setMobileEditorOpen(true);
      }
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
      return;
    }

    const firstGlobal = currentErrors.find(
      (e) => e.field.startsWith("global_causes_zh") || e.field.startsWith("global_treatments_zh")
    );
    if (!firstGlobal) return;
    setGlobalListTab(firstGlobal.field.startsWith("global_treatments_zh") ? "treatments" : "causes");
    if (isMobileViewport) {
      setMobileCanvasOpen(false);
      setMobileEditorOpen(false);
    }
    setTimeout(() => {
      globalListPanelRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }, 0);
  }, [validationErrors, dispatch, isMobileViewport]);

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

  // 取得當前影像為 File（草稿用本地檔，一般任務 fetch 已提供的影像 URL）。
  const getCurrentImageFile = useCallback(async (): Promise<File | null> => {
    if (draft) return draft.imageFile;
    const url = task?.image_url;
    if (!url) return null;
    const full = url.startsWith("/") ? `${baseUrl.replace(/\/$/, "")}${url}` : url;
    const resp = await fetch(full);
    const blob = await resp.blob();
    return new File([blob], doc?.image_filename || "image.jpg", { type: blob.type });
  }, [draft, task, baseUrl, doc]);

  const runAiSuggest = useCallback(async () => {
    setAiError(null);
    setAiBusy(true);
    try {
      const f = await getCurrentImageFile();
      if (!f) {
        setAiError("找不到影像");
        return;
      }
      const report = await diagnose(f, { mode: "grod_soft" });
      setAiSuggest(report);
      setAiPendingLesions(report.lesions);
      setAiHover(null);
    } catch (err) {
      console.error(err);
      setAiError("AI 建議失敗，請稍後再試。");
    } finally {
      setAiBusy(false);
    }
  }, [getCurrentImageFile]);

  const applySuggestedLesion = useCallback(
    (lesion: DiagnoseResponse["lesions"][number]) => {
      const zhToEn = invertLabelMap(labelMapZh);
      updateDoc(
        (d) => {
          d.detections.push(
            reportLesionToDetection(lesion, zhToEn, d.image_width, d.image_height)
          );
        },
        (d) => d.detections.length - 1
      );
    },
    [updateDoc, labelMapZh]
  );

  // 草稿模式：套用某個相似案例的整體描述（可再修改）。
  const applyOverallSuggestion = useCallback(
    (idx: number) => {
      const s = draft?.overallSuggestions?.[idx];
      if (!s) return;
      updateDoc((d) => {
        d.overall.colloquial_zh = ensureSingleLine(s.colloquial);
        d.overall.medical_zh = ensureSingleLine(s.medical);
      });
    },
    [updateDoc, draft]
  );

  const applySuggestedCause = useCallback(
    (text: string) => {
      updateDoc((d) => {
        const t = ensureSingleLine(text);
        if (t && !d.global_causes_zh.includes(t) && d.global_causes_zh.length < 10) {
          d.global_causes_zh.push(t);
        }
      });
    },
    [updateDoc]
  );

  // 採用單一建議病灶（畫布幽靈框或面板列觸發）：加入標註並從待採清單移除。
  const acceptLesionSuggestion = useCallback(
    (sidx: number) => {
      const lesion = aiPendingLesions[sidx];
      if (!lesion) return;
      applySuggestedLesion(lesion);
      setAiPendingLesions((prev) => prev.filter((_, i) => i !== sidx));
      setAiHover(null);
    },
    [aiPendingLesions, applySuggestedLesion]
  );

  // 全部採用：一次 updateDoc 推入所有框（避免逐筆 setState 讀到舊 doc 只生效最後一筆）。
  const acceptAllLesions = useCallback(() => {
    if (!aiPendingLesions.length) return;
    const zhToEn = invertLabelMap(labelMapZh);
    updateDoc((d) => {
      aiPendingLesions.forEach((l) => {
        d.detections.push(reportLesionToDetection(l, zhToEn, d.image_width, d.image_height));
      });
    });
    setAiPendingLesions([]);
    setAiHover(null);
  }, [aiPendingLesions, labelMapZh, updateDoc]);

  // 待採病灶 → 畫布幽靈框（box 由 bbox_xywh 轉、label 顯示中文症狀）。
  const aiCanvasSuggestions = useMemo(
    () =>
      aiPendingLesions.map((l) => {
        const [x, y, w, h] = l.bbox_xywh;
        return {
          box_xyxy: [x, y, x + w, y + h] as [number, number, number, number],
          label: l.label_zh
        };
      }),
    [aiPendingLesions]
  );

  const closeAiSuggest = useCallback(() => {
    setAiSuggest(null);
    setAiPendingLesions([]);
    setAiHover(null);
    setAiError(null);
  }, []);

  const handleUndo = useCallback(() => {
    dispatch({ type: "UNDO" });
  }, []);

  const handleRedo = useCallback(() => {
    dispatch({ type: "REDO" });
  }, []);

  // ✅ 只有提交成功時才由系統發新的一張（放行一次，不提示）
  const goNext = () => {
    navigate(`/annotate/${dataset}?refresh=${Date.now()}`, { replace: true });
  };

  const handleSubmit = async () => {
    if (!doc || !name) return;
    // 草稿模式：提交＝把影像＋報告編輯結果寫進目標資料集，建立一筆專家任務後跳轉。
    if (draft) {
      const errors = validateTaskDocument(doc, classes, false);
      const hasComments = ((doc as any).comments ?? []).length > 0;
      if (errors.length && !hasComments) {
        dispatch({ type: "SET_ERRORS", errors });
        setTimeout(() => jumpToFirstError(errors), 0);
        return;
      }
      const isEmpty = (doc.detections ?? []).length === 0;
      const confirm = window.confirm(
        isEmpty
          ? `此影像未框選任何病灶，將存入「${draft.dataset}」的健康影像。確定嗎？`
          : `確定要提交到資料集「${draft.dataset}」嗎？`
      );
      if (!confirm) return;
      setSaving(true);
      dispatch({ type: "RESET_ERRORS" });
      try {
        const res = await importDiagnosisTask(draft.dataset, draft.imageFile, doc, name);
        setDataset(res.dataset);
        setDraft(null); // 退出草稿模式，讓導頁後載入剛建立的正式任務
        // 健康影像存到 healthy_images/，導向健康影像列表；有病灶導向該標註任務。
        await runWithBypass(() =>
          navigate(
            res.is_healthy ? `/healthy/${res.dataset}` : `/annotate/${res.dataset}/${res.index}`,
            { replace: true }
          )
        );
      } catch (err) {
        const axiosErr = err as AxiosError<{ detail?: string }>;
        setError(axiosErr.response?.data?.detail || "提交失敗，請稍後再試。");
      } finally {
        setSaving(false);
      }
      return;
    }
    if (!task || !dataset) return;
    const isEmpty = (doc.detections ?? []).length === 0;
    const confirm = window.confirm(
      isEmpty
        ? "此影像未框選任何病灶，提交後將標記為「健康」。確定嗎？"
        : "確定要送出標註嗎？送出後此影像將視為完成，不會再次分派。"
    );
    if (!confirm) return;
    const errors = validateTaskDocument(doc, classes, false);
    const hasComments = ((doc as any).comments ?? []).length > 0;
    if (errors.length && !hasComments) {
      dispatch({ type: "SET_ERRORS", errors });
      // Move selection and focus to the first error to guide user
      setTimeout(() => jumpToFirstError(errors), 0);
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
      await runWithBypass(() => navigate(`/annotate/${dataset}/${nextIdx}`, { replace: true }));

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
      await runWithBypass(() => navigate(`/annotate/${dataset}/${currentIdx}`, { replace: true }));
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
      await runWithBypass(() => navigate(`/annotate/${dataset}/${nextIdx}`, { replace: true }));
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

  // 刪除此影像與標註（僅診斷建立的可寫資料集，專家）。
  const handleDeleteTask = async () => {
    if (!task || !dataset || draft) return;
    const ok = window.confirm("確定要刪除這張影像與其標註嗎？此動作無法復原。");
    if (!ok) return;
    setSaving(true);
    dispatch({ type: "RESET_ERRORS" });
    try {
      const res = await deleteDatasetTask(dataset, task.task_id);
      if (res.dataset_removed) {
        // 刪掉最後一張後資料集已被移除 → 導回資料集列表，避免停留在空殼頁。
        await runWithBypass(() => navigate("/datasets", { replace: true }));
        return;
      }
      // 刪除後下一張會遞補到目前編號；直接重新載入（導向同一網址不會觸發重載）。
      await loadTask();
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: string }>;
      setError(axiosErr.response?.data?.detail || "刪除失敗，請稍後再試。");
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
    if (!task || draft) return;
    const target = (Number(gotoIndex || task.index) || 1) - 1;
    if (target < 1) return;
    confirmAndNavigate(`/annotate/${dataset}/${target}`);
  }, [task, gotoIndex, confirmAndNavigate, draft]);

  const handleNextImage = useCallback(() => {
    if (!task || draft) return;
    const total = task.total_tasks ?? 0;
    const target = (Number(gotoIndex || task.index) || 1) + 1;
    if (total > 0 && target > total) return;
    confirmAndNavigate(`/annotate/${dataset}/${target}`);
  }, [task, gotoIndex, confirmAndNavigate, draft]);

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
    confirmAndNavigate(`/annotate/${dataset}/${clamped}`);
  }, [confirmAndNavigate, task?.total_tasks]);

  // ===== Header 狀態（disabled） =====
  // 訪客（!canEdit）只能檢視：所有編輯動作一律停用。
  const addDisabled = !!(loading || saving || !canEdit || !doc);
  const removeDisabled = !!(loading || saving || selectedIndex == null || !canEdit);
  const undoDisabled = !!(loading || saving || !history?.length || !canEdit);
  const redoDisabled = !!(loading || saving || !future?.length || !canEdit);
  // 草稿模式只允許「提交」(=匯入建庫)；保存／判定健康／無法使用都停用。
  const saveDisabled = !!(loading || saving || !doc || !task || !canEdit || draft);
  const submitDisabled = !!(loading || saving || !doc || !task || !canEdit);
  const moveToHealthyDisabled = !!(loading || saving || !doc || !task || !canEdit || draft);
  const markUnusableDisabled = !!(submitDisabled || draft);

  const withBase = (p: string) => (p.startsWith("/") ? `${baseUrl.replace(/\/$/, "")}${p}` : p);
  const canvasImageUrl = useMemo(() => {
    const url = task?.image_url ?? "";
    if (!url) return "";
    return url.startsWith("/") ? `${baseUrl.replace(/\/$/, "")}${url}` : url;
  }, [task?.image_url, baseUrl]);

  const globalCauseError = validationErrors.find((e) => e.field.startsWith("global_causes_zh"));
  const globalTreatmentError = validationErrors.find((e) => e.field.startsWith("global_treatments_zh"));
  const renderGlobalListTabButton = (
    tab: GlobalListTab,
    label: string,
    count: number,
    hasError: boolean
  ) => {
    const active = globalListTab === tab;
    return (
      <button
        type="button"
        role="tab"
        id={`global-tab-${tab}`}
        aria-selected={active}
        aria-controls={`global-panel-${tab}`}
        onClick={() => setGlobalListTab(tab)}
        className={`inline-flex min-w-0 flex-1 items-center justify-center gap-1.5 rounded-md px-2.5 py-2 text-sm font-medium transition-colors ${
          active
            ? "bg-slate-900 text-white shadow-sm"
            : "text-slate-600 hover:bg-white hover:text-slate-900"
        }`}
      >
        <span className="truncate">{label}</span>
        <span
          className={`shrink-0 rounded-full px-1.5 py-0.5 text-[11px] leading-none ${
            active ? "bg-white/20 text-white" : "bg-white text-slate-500"
          }`}
        >
          {count}
        </span>
        {hasError && (
          <span
            aria-label={`${label}有錯誤`}
            className="inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-rose-500 text-[10px] font-semibold leading-none text-white"
          >
            !
          </span>
        )}
      </button>
    );
  };

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
            <div className="flex w-full items-start justify-between gap-3 md:w-auto md:justify-start">
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
              <button
                type="button"
                onClick={() => setMobileMenuOpen((open) => !open)}
                className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-slate-300 bg-white text-slate-700 shadow-sm md:hidden"
                aria-expanded={mobileMenuOpen}
                aria-label={mobileMenuOpen ? "關閉選單" : "開啟選單"}
                title={mobileMenuOpen ? "關閉選單" : "開啟選單"}
              >
                {mobileMenuOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
              </button>
            </div>

            {/* 右側：導覽按鈕群 */}
            <div className="hidden flex-wrap items-center gap-2 md:flex">
              <button
                onClick={handleBackToDatasets}
                className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
              >
                <ArrowLeft className="h-4 w-4" />
                返回資料集
              </button>
              <button
                onClick={() => confirmAndNavigate(`/commented/${dataset}`)}
                className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
              >
                <MessageSquareQuote className="h-4 w-4" />
                查看註解
              </button>
              <button
                onClick={() => confirmAndNavigate(`/annotated/${dataset}`)}
                className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
              >
                <CheckCheck className="h-4 w-4" />
                查看提交
              </button>
              <button
                onClick={() => confirmAndNavigate(`/healthy/${dataset}`)}
                className="inline-flex items-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-500"
              >
                <Images className="h-4 w-4" />
                健康影像
              </button>
            </div>

            {mobileMenuOpen && (
              <div className="grid grid-cols-2 gap-2 md:hidden">
                <button
                  type="button"
                  onClick={() => { setMobileMenuOpen(false); handleBackToDatasets(); }}
                  className="inline-flex items-center justify-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm"
                >
                  <ArrowLeft className="h-4 w-4" />
                  返回
                </button>
                <button
                  type="button"
                  onClick={() => { setMobileMenuOpen(false); confirmAndNavigate(`/commented/${dataset}`); }}
                  className="inline-flex items-center justify-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm"
                >
                  <MessageSquareQuote className="h-4 w-4" />
                  註解
                </button>
                <button
                  type="button"
                  onClick={() => { setMobileMenuOpen(false); confirmAndNavigate(`/annotated/${dataset}`); }}
                  className="inline-flex items-center justify-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm"
                >
                  <CheckCheck className="h-4 w-4" />
                  提交紀錄
                </button>
                <button
                  type="button"
                  onClick={() => { setMobileMenuOpen(false); confirmAndNavigate(`/healthy/${dataset}`); }}
                  className="inline-flex items-center justify-center gap-1 rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-700 shadow-sm"
                >
                  <Images className="h-4 w-4" />
                  健康影像
                </button>
                <button
                  type="button"
                  onClick={() => { setMobileMenuOpen(false); handleMoveToHealthyImages(); }}
                  disabled={moveToHealthyDisabled}
                  className="inline-flex items-center justify-center gap-1 rounded-md border border-emerald-300 bg-emerald-50 px-3 py-2 text-sm text-emerald-700 shadow-sm disabled:opacity-40"
                >
                  <HeartPulse className="h-4 w-4" />
                  判定健康
                </button>
                <button
                  type="button"
                  onClick={() => { setMobileMenuOpen(false); handleMarkUnusable(); }}
                  disabled={markUnusableDisabled}
                  className="inline-flex items-center justify-center gap-1 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-700 shadow-sm disabled:opacity-40"
                >
                  <Ban className="h-4 w-4" />
                  無法使用
                </button>
                {datasetWritable && isExpert && !draft && (
                  <button
                    type="button"
                    onClick={() => { setMobileMenuOpen(false); handleDeleteTask(); }}
                    disabled={!!(loading || saving || !task)}
                    className="inline-flex items-center justify-center gap-1 rounded-md border border-rose-300 bg-rose-50 px-3 py-2 text-sm text-rose-700 shadow-sm disabled:opacity-40"
                  >
                    <Trash className="h-4 w-4" />
                    刪除影像
                  </button>
                )}
              </div>
            )}
          </div>

          {/* 分隔線 */}
          <div className="mt-3 hidden md:block"><Separator /></div>
          </div>

          {/* 下層：工具列（Icon-only）＋ 行為按鈕 */}
          <div className="mt-3 hidden flex-col gap-2 md:flex md:flex-row md:items-center md:justify-between">
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
                onClick={runAiSuggest}
                disabled={!!(loading || saving || aiBusy || !doc || !canEdit)}
                label="AI 建議"
                className="border-violet-300 bg-violet-50 text-violet-700 hover:bg-violet-100"
              >
                {aiBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
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
                disabled={markUnusableDisabled}
                label="無法使用並提交"
                className="border-amber-300 bg-amber-50 text-amber-700 hover:bg-amber-100"
              >
                <Ban className="h-4 w-4" />
              </IconButton>

              {datasetWritable && isExpert && !draft && (
                <IconButton
                  onClick={handleDeleteTask}
                  disabled={!!(loading || saving || !task)}
                  label="刪除此影像"
                  className="border-rose-300 bg-rose-50 text-rose-700 hover:bg-rose-100"
                >
                  <Trash className="h-4 w-4" />
                </IconButton>
              )}
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
                onClick={() => jumpToFirstError()}
                className="ml-2 inline-flex items-center rounded border border-amber-300 bg-white px-2 py-0.5 text-xs text-amber-800 hover:bg-amber-100"
              >
                定位到第一個錯誤
              </button>
            </Banner>
          )}
        </div>
      </header>

      <main className="grid grow grid-cols-1 gap-4 px-4 py-4 pb-28 sm:px-6 sm:py-6 md:pb-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <section className={`min-w-0 flex flex-col gap-4 ${!doc && noTasks ? "lg:col-span-2" : ""}`}>
          {draft && (
            <div className="rounded-xl border border-sky-200 bg-sky-50 p-3 text-sm text-sky-800 shadow">
              診斷結果草稿 — 編輯完成後按「提交」即會存入資料集
              <span className="font-semibold">「{draft.dataset}」</span>。此頁不會自動保存，離開即放棄。
            </div>
          )}
          {task && !draft && (
            <div className="rounded-xl bg-white p-3 shadow flex flex-col gap-3">
              {/* 🆕 顯示「用戶已提交 / 專家已提交」的膠囊狀態 */}
              {doc && (
                <SubmissionCapsules
                  generalEditor={(doc as any).general_editor}
                  expertEditor={(doc as any).expert_editor}
                  commentsCount={((doc as any).comments ?? []).length}
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

          <div className={`rounded-xl bg-white p-4 shadow ${!doc && noTasks ? "mx-auto w-full max-w-md" : ""}`}>
            {loading && <p className="text-slate-500">載入任務...</p>}
            {!loading && !doc && noTasks && (
              <div className="flex flex-col items-center justify-center gap-3 py-8 text-center">
                <p className="text-slate-600">此資料集目前沒有待標註的病灶影像。</p>
                <p className="text-xs text-slate-400">健康影像會另存於健康影像區。</p>
              </div>
            )}
            {!loading && !doc && !noTasks && (
              <p className="text-slate-500">目前沒有可用任務。</p>
            )}
            {doc && (
              <div className="relative">
                <div className="absolute right-3 top-3 z-10 md:hidden">
                  <button
                    type="button"
                    onClick={() => {
                      setMobileEditorOpen(false);
                      setMobileCanvasZoom(1);
                      setMobileCanvasFormOpen(false);
                      setMobileCanvasOpen(true);
                    }}
                    disabled={!canEdit}
                    className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/80 bg-white/90 text-sky-700 shadow-lg backdrop-blur transition-colors hover:bg-white disabled:pointer-events-none disabled:opacity-40"
                    aria-label="調整標註框"
                    title="調整標註框"
                  >
                    <Pencil className="h-4 w-4" />
                  </button>
                </div>
                <AnnotationCanvas
                  imageUrl={canvasImageUrl}
                  imageWidth={doc.image_width}
                  imageHeight={doc.image_height}
                  detections={doc.detections}
                  selectedIndex={selectedIndex}
                  onSelect={handleSelectDetection}
                  onUpdate={handleUpdateBox}
                  getDisplayLabel={getDisplayLabel}
                  suggestions={aiCanvasSuggestions}
                  hoveredSuggestion={aiHover}
                  onAcceptSuggestion={acceptLesionSuggestion}
                  onHoverSuggestion={setAiHover}
                  editable={canEdit && !isMobileViewport}
                />
              </div>
            )}
          </div>

          {doc && (
            <div className="grid gap-4">
              <div className="rounded-xl bg-white p-4 shadow">
                <h2 className="mb-3 text-lg font-semibold text-slate-800">
                  病徵敘述
                  <span className="ml-2 text-xs font-normal text-slate-400">通俗／醫學描述擇一必填</span>
                </h2>
                {draft && draft.overallSuggestions.length > 0 && (
                  <div className="mb-3">
                    <label className="mb-1 block text-xs text-slate-500">
                      套用相似案例描述（已預帶最相似案例，可切換或自行修改）
                    </label>
                    <select
                      defaultValue=""
                      onChange={(e) => {
                        if (e.target.value !== "") applyOverallSuggestion(Number(e.target.value));
                      }}
                      className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500"
                    >
                      <option value="" disabled>
                        選擇要套用的相似案例描述
                      </option>
                      {draft.overallSuggestions.map((s, i) => (
                        <option key={i} value={i}>
                          {s.source}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
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
            <div className="hidden rounded-xl bg-white p-4 shadow md:block">
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
                      className={`flex items-center justify-between gap-3 rounded border px-3 py-2 text-left text-sm ${itemClass}`}
                    >
                      <span className="flex min-w-0 items-center gap-2">
                        <span className="truncate">
                          {getDisplayLabel((det as any).label) || `框 ${idx + 1}`}
                        </span>
                        {errCount > 0 && (
                          <span className="inline-flex shrink-0 items-center rounded-full bg-rose-100 px-2 py-0.5 text-[11px] font-medium text-rose-700">
                            {errCount}
                          </span>
                        )}
                      </span>
                      <span className="hidden shrink-0 text-xs text-slate-400 md:inline">{det.box_xyxy.join(", ")}</span>
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
              <div className="hidden rounded-xl bg-white p-4 shadow md:block">
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

            <div ref={globalListPanelRef} className="rounded-xl bg-white p-4 shadow">
              <h2 className="mb-3 text-lg font-semibold text-slate-800">整體判讀</h2>
              <div
                role="tablist"
                aria-label="病徵原因與處置建議"
                className="mb-4 flex rounded-lg bg-slate-100 p-1"
              >
                {renderGlobalListTabButton("causes", "病徵原因", doc.global_causes_zh.length, !!globalCauseError)}
                {renderGlobalListTabButton(
                  "treatments",
                  "處置建議",
                  doc.global_treatments_zh.length,
                  !!globalTreatmentError
                )}
              </div>

              <div
                id="global-panel-causes"
                role="tabpanel"
                aria-labelledby="global-tab-causes"
                hidden={globalListTab !== "causes"}
              >
                <p className="mb-3 text-xs text-slate-500">依發生機率排序 · 最多 10 項</p>
                <GlobalListEditor
                  items={doc.global_causes_zh}
                  onChange={(action, payload) =>
                    handleGlobalListChange("global_causes_zh", action, payload)
                  }
                  error={globalCauseError}
                  inputPlaceholder="新增病徵原因 (Enter)"
                  emptyMessage="尚未新增病徵原因。"
                />
              </div>

              <div
                id="global-panel-treatments"
                role="tabpanel"
                aria-labelledby="global-tab-treatments"
                hidden={globalListTab !== "treatments"}
              >
                <p className="mb-3 text-xs text-slate-500">選填 · 依治療流程排序 · 最多 10 項</p>
                <GlobalListEditor
                  items={doc.global_treatments_zh}
                  onChange={(action, payload) =>
                    handleGlobalListChange("global_treatments_zh", action, payload)
                  }
                  error={globalTreatmentError}
                  inputPlaceholder="新增處置建議 (Enter)"
                  emptyMessage="尚未新增處置建議。"
                />
              </div>
            </div>

          </aside>
        )}
      </main>

      {isMobileViewport && doc && mobileCanvasOpen && (
        <div className="fixed inset-0 z-[60] flex flex-col bg-slate-950 text-white md:hidden">
          <div className="shrink-0 border-b border-white/10 bg-slate-950 px-3 py-2 shadow-lg">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <h2 className="truncate text-base font-semibold">調整標註框</h2>
                <p className="text-xs text-white/60">
                  {selectedIndex != null ? `框 ${selectedIndex + 1} / ${doc.detections.length}` : `共 ${doc.detections.length} 框`}
                </p>
              </div>
              <button
                type="button"
                onClick={() => {
                  setMobileCanvasOpen(false);
                  setMobileCanvasFormOpen(false);
                }}
                className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-white/15 bg-white/10 text-white"
                aria-label="關閉調框"
                title="關閉調框"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="mt-2 flex items-center justify-between gap-2">
              <div className="inline-flex rounded-md border border-white/15 bg-white/10 p-1">
                {[1, 1.5, 2].map((value) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setMobileCanvasZoom(value)}
                    className={`rounded px-2.5 py-1 text-xs font-medium ${
                      mobileCanvasZoom === value ? "bg-white text-slate-950" : "text-white/80"
                    }`}
                  >
                    {value}x
                  </button>
                ))}
              </div>
              <button
                type="button"
                onClick={() => setMobileCanvasFormOpen((open) => !open)}
                disabled={selectedIndex == null}
                className={`inline-flex items-center gap-1.5 rounded-md border border-white/15 px-3 py-1.5 text-sm text-white disabled:pointer-events-none disabled:opacity-40 ${
                  mobileCanvasFormOpen ? "bg-white/20" : "bg-white/10"
                }`}
                aria-pressed={mobileCanvasFormOpen}
              >
                <Pencil className="h-4 w-4" />
                標註內容
              </button>
            </div>

            <div className="mt-2 grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={handleAddDetection}
                disabled={addDisabled}
                className="inline-flex items-center justify-center gap-1.5 rounded-md border border-white/15 bg-white/10 px-3 py-2 text-sm text-white disabled:pointer-events-none disabled:opacity-40"
              >
                <SquarePlus className="h-4 w-4" />
                新增框
              </button>
              <button
                type="button"
                onClick={handleRemoveDetection}
                disabled={removeDisabled}
                className="inline-flex items-center justify-center gap-1.5 rounded-md border border-white/15 bg-white/10 px-3 py-2 text-sm text-white disabled:pointer-events-none disabled:opacity-40"
              >
                <Trash2 className="h-4 w-4" />
                刪除框
              </button>
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-auto bg-slate-900 p-2">
            <AnnotationCanvas
              imageUrl={canvasImageUrl}
              imageWidth={doc.image_width}
              imageHeight={doc.image_height}
              detections={doc.detections}
              selectedIndex={selectedIndex}
              onSelect={handleSelectDetection}
              onUpdate={handleUpdateBox}
              getDisplayLabel={getDisplayLabel}
              suggestions={aiCanvasSuggestions}
              hoveredSuggestion={aiHover}
              onAcceptSuggestion={acceptLesionSuggestion}
              onHoverSuggestion={setAiHover}
              editable={canEdit}
              zoom={mobileCanvasZoom}
            />
          </div>

          {selectedDetection && selectedIndex != null && mobileCanvasFormOpen && (
            <div
              className="shrink-0 border-t border-white/10 bg-white px-3 py-3 text-slate-800 shadow-[0_-8px_24px_rgba(0,0,0,0.25)]"
              style={{ paddingBottom: "calc(env(safe-area-inset-bottom) + 0.75rem)" }}
            >
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-slate-800">標註內容</p>
                  <span className="text-xs text-slate-500">框 {selectedIndex + 1} / {doc.detections.length}</span>
                </div>
                <button
                  type="button"
                  onClick={() => setMobileCanvasFormOpen(false)}
                  className="shrink-0 rounded border border-slate-200 px-2 py-1 text-xs text-slate-600 hover:bg-slate-50"
                >
                  收起
                </button>
              </div>
              <div className="grid grid-cols-1 gap-3">
                <div>
                  <label className="mb-1 block text-xs font-medium text-slate-600">表徵類別</label>
                  <select
                    ref={(el) => { labelRef.current = el; }}
                    value={selectedDetection.label ?? ""}
                    onChange={(e) => handleDetectionField(selectedIndex, "label", e.target.value)}
                    className={`w-full rounded border px-3 py-2 text-sm focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 ${
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
                  {detectionErrors.get(`detections.${selectedIndex}.label`) && (
                    <p className="mt-1 text-xs text-red-500">
                      {detectionErrors.get(`detections.${selectedIndex}.label`)}
                    </p>
                  )}
                </div>
                <div>
                  <label className="mb-1 block text-xs font-medium text-slate-600">外觀敘述</label>
                  <select
                    ref={(el) => { evidenceRef.current = el; }}
                    value={selectedDetection.evidence_index == null ? "" : String(selectedDetection.evidence_index)}
                    onChange={(e) => handleDetectionField(selectedIndex, "evidence_index", e.target.value)}
                    disabled={!selectedDetection.label}
                    className={`w-full rounded border px-3 py-2 text-sm focus:border-sky-500 focus:outline-none focus:ring-1 focus:ring-sky-500 ${
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
                    <p className="mt-1 text-xs text-red-500">
                      {detectionErrors.get(`detections.${selectedIndex}.evidence_index`)}
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {isMobileViewport && doc && selectedDetection && selectedIndex != null && mobileEditorOpen && !mobileCanvasOpen && (
        <div
          className="fixed inset-x-0 bottom-0 z-50 max-h-[78vh] overflow-y-auto rounded-t-2xl border-t border-slate-200 bg-white p-4 shadow-2xl md:hidden"
          style={{ paddingBottom: "calc(env(safe-area-inset-bottom) + 1rem)" }}
        >
          <div className="mb-3 flex items-center justify-between gap-3">
            <div className="min-w-0">
              <p className="text-xs text-slate-500">標註內容</p>
              <h2 className="truncate text-lg font-semibold text-slate-800">
                框 {selectedIndex + 1} / {doc.detections.length}
              </h2>
            </div>
            <button
              type="button"
              onClick={() => setMobileEditorOpen(false)}
              className="shrink-0 rounded-md border border-slate-200 px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-50"
            >
              收起
            </button>
          </div>

          <div className="mb-3 grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => selectedIndex > 0 && handleSelectDetection(selectedIndex - 1)}
              disabled={selectedIndex <= 0}
              className="rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 disabled:opacity-40"
            >
              上一框
            </button>
            <button
              type="button"
              onClick={() => selectedIndex < doc.detections.length - 1 && handleSelectDetection(selectedIndex + 1)}
              disabled={selectedIndex >= doc.detections.length - 1}
              className="rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 disabled:opacity-40"
            >
              下一框
            </button>
          </div>

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
          {detectionErrors.get(`detections.${selectedIndex}.label`) && (
            <p className="mb-3 text-sm text-red-500">
              {detectionErrors.get(`detections.${selectedIndex}.label`)}
            </p>
          )}

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

      {isMobileViewport && doc && selectedDetection && selectedIndex != null && !mobileEditorOpen && !mobileCanvasOpen && (
        <button
          type="button"
          onClick={() => setMobileEditorOpen(true)}
          className="fixed left-1/2 z-40 -translate-x-1/2 rounded-full border border-sky-200 bg-white px-3 py-2 text-sm font-medium text-sky-700 shadow-lg md:hidden"
          style={{ bottom: "calc(env(safe-area-inset-bottom) + 4.75rem)" }}
        >
          編輯框 {selectedIndex + 1}
        </button>
      )}

      {doc && !mobileCanvasOpen && (
        <div
          className="fixed inset-x-0 bottom-0 z-40 border-t border-slate-200 bg-white/95 px-2 pt-2 shadow-[0_-8px_24px_rgba(15,23,42,0.12)] backdrop-blur md:hidden"
          style={{ paddingBottom: "calc(env(safe-area-inset-bottom) + 0.5rem)" }}
        >
          <div className="grid grid-cols-5 gap-1.5">
            <button
              type="button"
              onClick={handleUndo}
              disabled={undoDisabled}
              className="inline-flex min-w-0 flex-col items-center justify-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-2 text-[11px] text-slate-700 shadow-sm disabled:pointer-events-none disabled:opacity-40"
            >
              <Undo2 className="h-4 w-4" />
              <span>復原</span>
            </button>
            <button
              type="button"
              onClick={handleRedo}
              disabled={redoDisabled}
              className="inline-flex min-w-0 flex-col items-center justify-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-2 text-[11px] text-slate-700 shadow-sm disabled:pointer-events-none disabled:opacity-40"
            >
              <Redo2 className="h-4 w-4" />
              <span>重做</span>
            </button>
            <button
              type="button"
              onClick={handleSave}
              disabled={saveDisabled}
              className="inline-flex min-w-0 flex-col items-center justify-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-2 text-[11px] text-slate-700 shadow-sm disabled:pointer-events-none disabled:opacity-40"
            >
              <Save className="h-4 w-4" />
              <span>保存</span>
            </button>
            <button
              type="button"
              onClick={runAiSuggest}
              disabled={!!(loading || saving || aiBusy || !doc || !canEdit)}
              className="inline-flex min-w-0 flex-col items-center justify-center gap-1 rounded-md border border-violet-200 bg-violet-50 px-2 py-2 text-[11px] text-violet-700 shadow-sm disabled:pointer-events-none disabled:opacity-40"
            >
              {aiBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
              <span>AI</span>
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={submitDisabled}
              className="inline-flex min-w-0 flex-col items-center justify-center gap-1 rounded-md border border-sky-600 bg-sky-600 px-2 py-2 text-[11px] font-medium text-white shadow-sm disabled:pointer-events-none disabled:border-sky-300 disabled:bg-sky-300"
            >
              <CheckCheck className="h-4 w-4" />
              <span>提交</span>
            </button>
          </div>
        </div>
      )}

      {/* AI 建議：右側非阻擋抽屜，畫布保留可見幽靈框；逐項採用、合併不取代 */}
      {(aiSuggest || aiError) && (
        <div className="fixed inset-x-0 bottom-0 z-50 flex max-h-[85vh] w-full flex-col overflow-hidden rounded-t-2xl border-t border-slate-200 bg-white shadow-2xl md:inset-y-0 md:left-auto md:right-0 md:h-full md:max-h-none md:max-w-sm md:rounded-none md:border-l md:border-t-0">
          <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
            <h3 className="flex items-center gap-1.5 text-lg font-semibold text-slate-800">
              <Sparkles className="h-5 w-5 text-violet-600" /> AI 建議
            </h3>
            <button
              onClick={closeAiSuggest}
              className="rounded border border-slate-200 px-3 py-1 text-sm text-slate-600 hover:bg-slate-100"
            >
              完成
            </button>
          </div>
          <p className="px-4 py-2 text-xs text-slate-500">
            點影像上的紫色虛線框或下方「採用」即可加入標註；採用後仍可復原（Ctrl+Z）。
          </p>
          {aiError && (
            <div className="mx-4 mb-2 rounded bg-red-50 px-3 py-2 text-sm text-red-600">{aiError}</div>
          )}
          {aiSuggest && (
            <div className="flex-1 overflow-y-auto px-4 pb-6 md:pb-4">
              <div className="mb-1 flex items-center justify-between">
                <p className="text-sm font-medium text-slate-700">建議病灶（{aiPendingLesions.length}）</p>
                {aiPendingLesions.length > 0 && (
                  <button
                    onClick={acceptAllLesions}
                    className="rounded bg-violet-600 px-2 py-1 text-xs text-white hover:bg-violet-700"
                  >
                    全部採用
                  </button>
                )}
              </div>
              <div className="mb-4 flex flex-col gap-2">
                {aiPendingLesions.length === 0 && (
                  <p className="rounded border border-dashed border-slate-200 px-3 py-2 text-sm text-slate-400">
                    {aiSuggest.lesions.length === 0 ? "未偵測到明顯病灶。" : "建議病灶已全部採用。"}
                  </p>
                )}
                {aiPendingLesions.map((l, sidx) => (
                  <div
                    key={l.idx}
                    onMouseEnter={() => setAiHover(sidx)}
                    onMouseLeave={() => setAiHover(null)}
                    className={
                      "flex items-center gap-3 rounded border px-2 py-2 text-sm " +
                      (aiHover === sidx ? "border-pink-400 bg-pink-50" : "border-slate-200")
                    }
                  >
                    <img src={l.crop} alt={l.label_zh} className="h-12 w-12 shrink-0 rounded object-cover" />
                    <span className="min-w-0 flex-1 truncate text-slate-700">
                      {l.label_zh}
                      <span className="ml-2 text-xs text-slate-400">{Math.round(l.cls_score * 100)}%</span>
                    </span>
                    <button
                      onClick={() => acceptLesionSuggestion(sidx)}
                      className="shrink-0 rounded bg-violet-600 px-2 py-1 text-xs text-white hover:bg-violet-700"
                    >
                      採用
                    </button>
                  </div>
                ))}
              </div>
              <p className="mb-1 text-sm font-medium text-slate-700">建議病因（{aiSuggest.causes.length}）</p>
              <div className="flex flex-col gap-2">
                {aiSuggest.causes.map((c) => (
                  <div key={c.rank} className="flex items-center justify-between rounded border border-slate-200 px-3 py-2 text-sm">
                    <span className="min-w-0 truncate text-slate-700">{c.text}</span>
                    <button
                      onClick={() => applySuggestedCause(c.text)}
                      className="ml-2 shrink-0 rounded bg-violet-600 px-2 py-1 text-xs text-white hover:bg-violet-700"
                    >
                      採用
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
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
  inputPlaceholder?: string;
  emptyMessage?: string;
};

const GlobalListEditor: React.FC<GlobalListEditorProps> = ({
  items,
  onChange,
  error,
  inputPlaceholder,
  emptyMessage
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
          placeholder={inputPlaceholder ?? "新增項目 (Enter)"}
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
            className="flex items-center justify-between gap-2 rounded border border-slate-200 px-3 py-2"
          >
            <span className="min-w-0 flex-1 break-words leading-relaxed">{item}</span>
            <div className="flex shrink-0 flex-col items-center gap-1">
              <button
                type="button"
                className="inline-flex h-8 w-8 items-center justify-center rounded border border-slate-200 text-xs text-slate-500 hover:border-slate-400 disabled:opacity-40"
                onClick={() => onChange("move", { index: idx, direction: -1 })}
                disabled={idx === 0}
                aria-label="上移"
                title="上移"
              >
                ▲
              </button>
              <button
                type="button"
                className="inline-flex h-8 w-8 items-center justify-center rounded border border-slate-200 text-xs text-slate-500 hover:border-slate-400 disabled:opacity-40"
                onClick={() => onChange("move", { index: idx, direction: 1 })}
                disabled={idx === items.length - 1}
                aria-label="下移"
                title="下移"
              >
                ▼
              </button>
              <button
                type="button"
                className="inline-flex h-8 w-8 items-center justify-center rounded border border-rose-200 text-rose-600 hover:bg-rose-50"
                onClick={() => onChange("remove", { index: idx })}
                aria-label="刪除"
                title="刪除"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          </div>
        ))}
        {!items.length && (
          <p className="rounded border border-dashed border-slate-300 px-3 py-4 text-center text-sm text-slate-500">
            {emptyMessage ?? "尚未新增資料。"}
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
