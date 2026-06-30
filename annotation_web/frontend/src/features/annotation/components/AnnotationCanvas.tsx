import { useEffect, useMemo, useRef, useState } from "react";
import { Layer, Rect, Stage, Transformer, Label as KonvaLabel, Tag, Text } from "react-konva";
import type Konva from "konva";
import React from "react";

import type { Detection } from "../../../api/types";
import { clamp, normalizeBox } from "../../../lib/taskUtils";

type Suggestion = {
  box_xyxy: [number, number, number, number];
  label?: string;
};

type AnnotationCanvasProps = {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  detections: Detection[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
  onUpdate: (index: number, box: [number, number, number, number]) => void;
  getDisplayLabel?: (value: string) => string;
  // AI 建議的幽靈框（虛線、點擊即採用、與面板 hover 連動）
  suggestions?: Suggestion[];
  hoveredSuggestion?: number | null;
  onAcceptSuggestion?: (index: number) => void;
  onHoverSuggestion?: (index: number | null) => void;
  editable?: boolean;
  zoom?: number;
};

const AnnotationCanvas: React.FC<AnnotationCanvasProps> = ({
  imageUrl,
  imageWidth,
  imageHeight,
  detections,
  selectedIndex,
  onSelect,
  onUpdate,
  getDisplayLabel,
  suggestions = [],
  hoveredSuggestion = null,
  onAcceptSuggestion,
  onHoverSuggestion,
  editable = true,
  zoom = 1
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const transformerRef = useRef<Konva.Transformer | null>(null);
  const shapeRefs = useRef<Record<number, Konva.Rect | null>>({});
  const [containerWidth, setContainerWidth] = useState<number>(640);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [status, setStatus] = useState<"loading" | "loaded" | "failed">(
    "loading"
  );

  useEffect(() => {
    setStatus("loading");
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;
    img.onload = () => {
      setImage(img);
      setStatus("loaded");
    };
    img.onerror = () => {
      setImage(null);
      setStatus("failed");
    };
    return () => {
      setImage(null);
    };
  }, [imageUrl]);

  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      setContainerWidth(entry.contentRect.width);
    });
    ro.observe(containerRef.current);
    return () => {
      ro.disconnect();
    };
  }, []);

  const { stageWidth, stageHeight, scaleX, scaleY, naturalWidth, naturalHeight } = useMemo(() => {
    const fallbackWidth = imageWidth || containerWidth || 640;
    const fallbackHeight = imageHeight || Math.round((fallbackWidth || 640) * 0.75);
    const baseWidth = image?.width ?? fallbackWidth;
    const baseHeight = image?.height ?? fallbackHeight;
    const safeBaseWidth = Math.max(1, baseWidth);
    const safeZoom = Math.max(1, zoom);
    const width = (containerWidth || safeBaseWidth) * safeZoom;
    const scale = width / safeBaseWidth;
    return {
      stageWidth: width,
      stageHeight: Math.max(1, baseHeight) * scale,
      scaleX: scale,
      scaleY: scale,
      naturalWidth: safeBaseWidth,
      naturalHeight: Math.max(1, baseHeight),
    };
  }, [image, containerWidth, imageWidth, imageHeight, zoom]);

  useEffect(() => {
    const transformer = transformerRef.current;
    if (!transformer) return;
    if (!editable || selectedIndex == null) {
      transformer.nodes([]);
      return;
    }
    const node = shapeRefs.current[selectedIndex];
    if (node) {
      transformer.nodes([node]);
      transformer.getLayer()?.batchDraw();
    }
  }, [selectedIndex, detections, editable]);

  const handleDragEnd = (index: number, node: Konva.Rect) => {
    if (!naturalWidth || !naturalHeight) return;
    const x = clamp(node.x(), 0, stageWidth);
    const y = clamp(node.y(), 0, stageHeight);
    node.position({ x, y });

    const width = node.width() * node.scaleX();
    const height = node.height() * node.scaleY();
    node.scale({ x: 1, y: 1 });

    const scaleXFactor = stageWidth / naturalWidth;
    const scaleYFactor = stageHeight / naturalHeight;
    const x1 = x / scaleXFactor;
    const y1 = y / scaleYFactor;
    const x2 = (x + width) / scaleXFactor;
    const y2 = (y + height) / scaleYFactor;
    onUpdate(index, normalizeBox(x1, y1, x2, y2, naturalWidth, naturalHeight));
  };

  const handleTransformEnd = (index: number, node: Konva.Rect) => {
    if (!naturalWidth || !naturalHeight) return;
    const scaleXNode = node.scaleX();
    const scaleYNode = node.scaleY();
    const width = node.width() * scaleXNode;
    const height = node.height() * scaleYNode;
    node.scale({ x: 1, y: 1 });
    const x = clamp(node.x(), 0, stageWidth - width);
    const y = clamp(node.y(), 0, stageHeight - height);
    node.position({ x, y });

    const scaleXFactor = stageWidth / naturalWidth;
    const scaleYFactor = stageHeight / naturalHeight;
    const x1 = x / scaleXFactor;
    const y1 = y / scaleYFactor;
    const x2 = (x + width) / scaleXFactor;
    const y2 = (y + height) / scaleYFactor;
    onUpdate(index, normalizeBox(x1, y1, x2, y2, naturalWidth, naturalHeight));
  };

  // reset shape refs each render so stale keys do not linger
  shapeRefs.current = {};

  return (
    <div ref={containerRef} className="w-full min-w-0">
      <div className="rounded border border-slate-200 bg-slate-900/5">
        {status === "loading" && (
          <div className="flex h-64 items-center justify-center text-slate-500">
            影像載入中...
          </div>
        )}
        {status === "failed" && (
          <div className="flex h-64 items-center justify-center text-red-600">
            影像載入失敗
          </div>
        )}
        {image && status === "loaded" && (
          <Stage width={stageWidth} height={stageHeight} className="mx-auto">
            <Layer>
              <Rect
                listening={false}
                x={0}
                y={0}
                width={stageWidth}
                height={stageHeight}
                fillPatternImage={image}
                fillPatternScale={{ x: scaleX, y: scaleY }}
                fillPatternRepeat="no-repeat"
              />
              {detections.map((det, idx) => {
                const [x1, y1, x2, y2] = det.box_xyxy;
                const boxWidth = ((x2 - x1) / naturalWidth) * stageWidth;
                const boxHeight = ((y2 - y1) / naturalHeight) * stageHeight;
                const posX = (x1 / naturalWidth) * stageWidth;
                const posY = (y1 / naturalHeight) * stageHeight;

                // 盡量用穩定的 id；如果沒有，就先用 idx（但清單可能重排時不穩定）
                const itemKey = (det as any).id ?? (det as any).uuid ?? `det-${idx}`;

                return (
                  <React.Fragment key={itemKey}>
                    <Rect
                      ref={(node) => {
                        shapeRefs.current[idx] = node;
                      }}
                      x={posX}
                      y={posY}
                      width={boxWidth}
                      height={boxHeight}
                      stroke={idx === selectedIndex ? "#2563eb" : "#ef4444"}
                      strokeWidth={idx === selectedIndex ? 3 : 2}
                      listening
                      draggable={editable}
                      onClick={() => onSelect(idx)}
                      onTap={() => onSelect(idx)}
                      onDragEnd={(evt) => editable && handleDragEnd(idx, evt.target as Konva.Rect)}
                      onTransformEnd={(evt) =>
                        editable && handleTransformEnd(idx, evt.target as Konva.Rect)
                      }
                      dragBoundFunc={(pos) => {
                        const limitedX = clamp(pos.x, 0, stageWidth - boxWidth);
                        const limitedY = clamp(pos.y, 0, stageHeight - boxHeight);
                        return { x: limitedX, y: limitedY };
                      }}
                    />
                    {det.label && (
                      <KonvaLabel
                        x={Math.max(0, Math.min(stageWidth - 200, posX))}
                        y={Math.max(0, posY - 22)}
                        listening={false}
                      >
                        <Tag
                          fill={idx === selectedIndex ? "#2563eb" : "#ef4444"}
                          opacity={0.9}
                          cornerRadius={4}
                        />
                        <Text
                          text={(getDisplayLabel ? getDisplayLabel(det.label) : det.label) || ""}
                          padding={4}
                          fill="#ffffff"
                          fontSize={12}
                        />
                      </KonvaLabel>
                    )}
                  </React.Fragment>
                );
              })}
              {suggestions.map((sug, sidx) => {
                const [x1, y1, x2, y2] = sug.box_xyxy;
                const boxWidth = ((x2 - x1) / naturalWidth) * stageWidth;
                const boxHeight = ((y2 - y1) / naturalHeight) * stageHeight;
                const posX = (x1 / naturalWidth) * stageWidth;
                const posY = (y1 / naturalHeight) * stageHeight;
                const active = hoveredSuggestion === sidx;
                return (
                  <React.Fragment key={`sug-${sidx}`}>
                    <Rect
                      x={posX}
                      y={posY}
                      width={boxWidth}
                      height={boxHeight}
                      stroke="#ff1493"
                      strokeWidth={active ? 4 : 3}
                      dash={[9, 5]}
                      fill={active ? "rgba(255,20,147,0.22)" : "rgba(255,20,147,0.10)"}
                      shadowColor="#000000"
                      shadowBlur={2}
                      shadowOpacity={0.5}
                      listening
                      onClick={() => onAcceptSuggestion?.(sidx)}
                      onTap={() => onAcceptSuggestion?.(sidx)}
                      onMouseEnter={() => onHoverSuggestion?.(sidx)}
                      onMouseLeave={() => onHoverSuggestion?.(null)}
                    />
                    {sug.label && (
                      <KonvaLabel
                        x={Math.max(0, Math.min(stageWidth - 200, posX))}
                        y={Math.max(0, posY - 22)}
                        listening={false}
                      >
                        <Tag fill="#db2777" opacity={0.95} cornerRadius={4} />
                        <Text
                          text={`建議：${(getDisplayLabel ? getDisplayLabel(sug.label) : sug.label) || ""}`}
                          padding={4}
                          fill="#ffffff"
                          fontSize={12}
                        />
                      </KonvaLabel>
                    )}
                  </React.Fragment>
                );
              })}
              {editable && (
                <Transformer
                  ref={transformerRef}
                  rotateEnabled={false}
                  keepRatio={false}
                  enabledAnchors={[
                    "top-left",
                    "top-center",
                    "top-right",
                    "middle-left",
                    "middle-right",
                    "bottom-left",
                    "bottom-center",
                    "bottom-right"
                  ]}
                  boundBoxFunc={(oldBox, newBox) => {
                    if (newBox.width < 10 || newBox.height < 10) {
                      return oldBox;
                    }
                    return newBox;
                  }}
                />
              )}
            </Layer>
          </Stage>
        )}
      </div>
    </div>
  );
};

export default AnnotationCanvas;
