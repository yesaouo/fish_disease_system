import type { Detection, TaskDocument } from "../api/types";

export const cloneTaskDocument = (doc: TaskDocument): TaskDocument =>
  JSON.parse(JSON.stringify(doc));

export const documentsEqual = (a: TaskDocument, b: TaskDocument): boolean =>
  JSON.stringify(a) === JSON.stringify(b);

export const clamp = (value: number, min: number, max: number): number => {
  return Math.min(Math.max(value, min), max);
};

export const normalizeBox = (
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  imageWidth: number,
  imageHeight: number
): [number, number, number, number] => {
  const safeWidth = Math.max(1, Math.round(imageWidth || 1000));
  const safeHeight = Math.max(1, Math.round(imageHeight || 1000));
  const box = [
    Math.round(clamp(x1, 0, safeWidth)),
    Math.round(clamp(y1, 0, safeHeight)),
    Math.round(clamp(x2, 0, safeWidth)),
    Math.round(clamp(y2, 0, safeHeight))
  ] as [number, number, number, number];
  box[0] = Math.min(box[0], box[2] - 1);
  box[1] = Math.min(box[1], box[3] - 1);
  box[2] = Math.max(box[2], box[0] + 1);
  box[3] = Math.max(box[3], box[1] + 1);
  return box;
};

export type ValidationError = {
  field: string;
  message: string;
};

export const validateTaskDocument = (
  doc: TaskDocument,
  classes: string[],
  hasHierarchical?: boolean
): ValidationError[] => {
  const errors: ValidationError[] = [];
  const classSet = new Set(classes);
  const width = doc.image_width || 0;
  const height = doc.image_height || 0;

  doc.detections.forEach((det, idx) => {
    const [x1, y1, x2, y2] = det.box_xyxy;
    if (x1 < 0 || y1 < 0 || x2 > width || y2 > height) {
      errors.push({
        field: `detections.${idx}.box_xyxy`,
        message: "框需位於圖片範圍內"
      });
    }
    if (x2 <= x1 || y2 <= y1) {
      errors.push({
        field: `detections.${idx}.box_xyxy`,
        message: "須滿足 x2>x1 與 y2>y1"
      });
    }
    // 標籤不可為空
    if (!det.label) {
      errors.push({
        field: `detections.${idx}.label`,
        message: `需選擇表徵類別`
      });
    }
    // evidence_zh 改為選填，不做強制檢查
  });

  const seenCauses = new Set<string>();
  doc.global_causes_zh.forEach((cause, idx) => {
    if (seenCauses.has(cause)) {
      errors.push({
        field: `global_causes_zh.${idx}`,
        message: "病徵原因不可重複"
      });
    }
    seenCauses.add(cause);
  });
  if (doc.global_causes_zh.length > 10) {
    errors.push({
      field: "global_causes_zh",
      message: "病徵原因最多 10 項"
    });
  }

  const seenTreatments = new Set<string>();
  doc.global_treatments_zh.forEach((t, idx) => {
    if (seenTreatments.has(t)) {
      errors.push({
        field: `global_treatments_zh.${idx}`,
        message: "處置不可重複"
      });
    }
    seenTreatments.add(t);
  });
  if (doc.global_treatments_zh.length > 10) {
    errors.push({
      field: "global_treatments_zh",
      message: "處置最多 10 項"
    });
  }

  return errors;
};

export const ensureSingleLine = (value: string) =>
  value.replace(/\s*\n+\s*/g, " ").trim();

export const defaultDetection = (
  imageWidth: number,
  imageHeight: number,
  label?: string,
  evidence?: string
): Detection => {
  const safeWidth = Math.max(1, Math.round(imageWidth || 1000));
  const safeHeight = Math.max(1, Math.round(imageHeight || 1000));
  const boxWidth = Math.round(safeWidth * 0.3);
  const boxHeight = Math.round(safeHeight * 0.3);
  const x1 = Math.round((safeWidth - boxWidth) / 2);
  const y1 = Math.round((safeHeight - boxHeight) / 2);

  return {
    label: label ?? "",
    evidence_zh: evidence ?? "",
    box_xyxy: normalizeBox(
      x1,
      y1,
      x1 + boxWidth,
      y1 + boxHeight,
      safeWidth,
      safeHeight
    ),
    confidence: 0.99,
  };
};
