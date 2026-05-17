import type { TaskDocument } from "../../../api/types";
import { ValidationError, cloneTaskDocument } from "../../../lib/taskUtils";

export const HISTORY_LIMIT = 100;

export function clampHistory(hist: TaskDocument[]) {
  return hist.length > HISTORY_LIMIT ? hist.slice(hist.length - HISTORY_LIMIT) : hist;
}

export function normalizeSelection(sel: number | null, detections: unknown[]) {
  if (!detections || detections.length === 0) return null;
  return sel == null ? null : Math.min(sel, detections.length - 1);
}

export type AnnotationState = {
  doc: TaskDocument | null;
  history: TaskDocument[];
  future: TaskDocument[];
  selectedIndex: number | null;
  validationErrors: ValidationError[];
};

export type AnnotationAction =
  | { type: "UNDO" }
  | { type: "REDO" }
  | { type: "APPLY_DOC"; next: TaskDocument; nextSelected?: number | null }
  | { type: "LOAD_DOC"; doc: TaskDocument }
  | { type: "SET_VERSION"; version: number }
  | { type: "SET_SELECTED"; index: number | null }
  | { type: "SET_ERRORS"; errors: ValidationError[] }
  | { type: "RESET_ERRORS" };

export const initialAnnotationState: AnnotationState = {
  doc: null,
  history: [],
  future: [],
  selectedIndex: null,
  validationErrors: [],
};

export function annotationReducer(state: AnnotationState, action: AnnotationAction): AnnotationState {
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
        selectedIndex: normalizeSelection(state.selectedIndex, nextDoc.detections),
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
        selectedIndex: normalizeSelection(state.selectedIndex, nextDoc.detections),
        validationErrors: [],
      };
    }
    case "APPLY_DOC": {
      const nextDoc = cloneTaskDocument(action.next);
      return {
        ...state,
        doc: nextDoc,
        history: state.doc ? clampHistory([...state.history, cloneTaskDocument(state.doc)]) : state.history,
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
    case "SET_VERSION": {
      // Update concurrency token on the current doc without disturbing
      // history/future/selection — used after a successful save.
      if (!state.doc) return state;
      return { ...state, doc: { ...state.doc, version: action.version } };
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
