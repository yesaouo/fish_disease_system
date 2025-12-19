"""
規則摘要：
- 無 _vX 後綴：原版
- _v1/_v2/...：版本，取最大 v 當最終版；若沒有版本檔，最終版=原版 (v=0)
- 只要「最終版」JSON 的 comments 有內容，該 stem 不納入主 label 統計；但會另外統計到「被跳過」欄位
- 不做 IoU / 不做 index 對齊：label 與清單欄位都用「多重集合」的數量差/交集數量

輸出（2 個 CSV）：
- files_summary.csv：每個 stem 一列（含最終版本、最終是否有評論、overall 文字是否一致、global_* 清單的四個統計）
- label_delta.csv：每個 label 一列（主統計 + 被跳過(最終有評論) 的統計）
"""

from __future__ import annotations
from pathlib import Path
import json
import re
import argparse
from collections import Counter, defaultdict
import csv
from typing import Dict, Any, Optional, List, Tuple

Box = Tuple[float, float, float, float]

def comments_has_content(obj: Any) -> bool:
    if obj is None:
        return False
    if isinstance(obj, str):
        return obj.strip() != ""
    if isinstance(obj, (list, tuple, dict, set)):
        return len(obj) > 0
    return bool(obj)

def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def count_labels(data: Dict) -> Counter:
    dets = data.get("detections", []) or []
    c = Counter()
    for d in dets:
        c[d.get("label")] += 1
    return c

def multiset_intersection_size(a: List[str], b: List[str]) -> int:
    ca = Counter(a)
    cb = Counter(b)
    return sum(min(ca[k], cb[k]) for k in (set(ca) | set(cb)))

def group_files(folder: Path):
    rx = re.compile(r"^(?P<stem>.+?)_v(?P<v>\d+)\.json$", re.IGNORECASE)
    groups = defaultdict(list)  # stem -> list[(vnum, path, is_original)]
    for p in folder.rglob("*.json"):
        m = rx.match(p.name)
        if m:
            groups[m.group("stem")].append((int(m.group("v")), p, False))
        else:
            groups[p.stem].append((0, p, True))
    return groups

def safe_get_overall_text(data: Dict, key: str) -> str:
    overall = data.get("overall") or {}
    val = overall.get(key)
    return val if isinstance(val, str) else ""

def safe_get_str_list(data: Dict, key: str) -> List[str]:
    val = data.get(key)
    if isinstance(val, list):
        return [x for x in val if isinstance(x, str)]
    return []

def extract_boxes(data: Dict) -> List[Box]:
    dets = data.get("detections", []) or []
    boxes: List[Box] = []
    for d in dets:
        box = d.get("box_xyxy") or d.get("box") or d.get("bbox")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                x1, y1, x2, y2 = map(float, box)
                if x1 > x2: x1, x2 = x2, x1
                if y1 > y2: y1, y2 = y2, y1
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
            except Exception:
                pass
    return boxes

def union_area(rects: List[Box]) -> float:
    if not rects:
        return 0.0
    xs = sorted(set([x1 for x1,_,_,_ in rects] + [x2 for _,_,x2,_ in rects]))
    area = 0.0
    for i in range(len(xs) - 1):
        x_left, x_right = xs[i], xs[i+1]
        dx = x_right - x_left
        if dx <= 0:
            continue
        intervals: List[Tuple[float, float]] = []
        for x1, y1, x2, y2 in rects:
            if x1 < x_right and x2 > x_left:
                intervals.append((y1, y2))
        if not intervals:
            continue
        intervals.sort()
        merged_len = 0.0
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s > cur_e:
                merged_len += max(0.0, cur_e - cur_s)
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        merged_len += max(0.0, cur_e - cur_s)
        area += merged_len * dx
    return area

def boxes_global_iou(orig_boxes: List[Box], final_boxes: List[Box]) -> Optional[float]:
    if not orig_boxes or not final_boxes:
        return None
    area_o = union_area(orig_boxes)
    area_f = union_area(final_boxes)
    if area_o <= 0.0 and area_f <= 0.0:
        return None

    inter_rects: List[Box] = []
    for ax1, ay1, ax2, ay2 in orig_boxes:
        for bx1, by1, bx2, by2 in final_boxes:
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 > ix1 and iy2 > iy1:
                inter_rects.append((ix1, iy1, ix2, iy2))
    area_i = union_area(inter_rects)

    denom = area_o + area_f - area_i
    if denom <= 0.0:
        return None
    return area_i / denom

def analyze_dir(folder: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups = group_files(folder)

    # 主統計（不含 final comments 的 stem）
    orig_total = Counter()
    final_total = Counter()
    # 被跳過（final comments 的 stem）
    skipped_orig_total = Counter()
    skipped_final_total = Counter()

    files_rows: List[Dict[str, Any]] = []

    for stem, items in groups.items():
        # 原版（假設一定存在；若缺就略過）
        orig_candidates = [it for it in items if it[2] is True]
        if not orig_candidates:
            continue
        orig_path = sorted(orig_candidates, key=lambda x: x[1].name)[0][1]

        # 最終版（最大 v；無版本則 v=0 且=原版）
        versioned = [it for it in items if it[2] is False]
        if versioned:
            final_version, final_path, _ = max(versioned, key=lambda x: x[0])
        else:
            final_version = 0
            final_path = orig_path

        oj = load_json(orig_path)
        fj = load_json(final_path)

        final_has_comments = comments_has_content(fj.get("comments"))

        overlap = boxes_global_iou(extract_boxes(oj), extract_boxes(fj))
        overlap_str = "" if overlap is None else f"{overlap:.6f}"

        # overall 文本是否一致
        col_same = int(safe_get_overall_text(oj, "colloquial_zh") == safe_get_overall_text(fj, "colloquial_zh"))
        med_same = int(safe_get_overall_text(oj, "medical_zh") == safe_get_overall_text(fj, "medical_zh"))

        # detections 總數
        orig_det_n = len(oj.get("detections", []) or [])
        final_det_n = len(fj.get("detections", []) or [])

        # global_causes_zh / global_treatments_zh 四欄
        orig_causes = safe_get_str_list(oj, "global_causes_zh")
        final_causes = safe_get_str_list(fj, "global_causes_zh")
        causes_same_n = multiset_intersection_size(orig_causes, final_causes)

        orig_treats = safe_get_str_list(oj, "global_treatments_zh")
        final_treats = safe_get_str_list(fj, "global_treatments_zh")
        treats_same_n = multiset_intersection_size(orig_treats, final_treats)

        files_rows.append({
            "stem": stem,
            "final_version": final_version,
            "final_has_comments": int(final_has_comments),
            "bbox_overlap_ratio": overlap_str,
            "colloquial_zh_same_as_orig": col_same,
            "medical_zh_same_as_orig": med_same,
            "orig_detections": orig_det_n,
            "final_detections": final_det_n,
            "global_causes_zh_orig_count": len(orig_causes),
            "global_causes_zh_final_count": len(final_causes),
            "global_causes_zh_exact_same_count": causes_same_n,
            "global_causes_zh_final_minus_exact_same": len(final_causes) - causes_same_n,
            "global_treatments_zh_orig_count": len(orig_treats),
            "global_treatments_zh_final_count": len(final_treats),
            "global_treatments_zh_exact_same_count": treats_same_n,
            "global_treatments_zh_final_minus_exact_same": len(final_treats) - treats_same_n,
        })

        # label 統計：分兩組
        if final_has_comments:
            skipped_orig_total.update(count_labels(oj))
            skipped_final_total.update(count_labels(fj))
        else:
            orig_total.update(count_labels(oj))
            final_total.update(count_labels(fj))

    # label_delta rows（含 skipped_* 欄位）
    labels = sorted(set(orig_total) | set(final_total) | set(skipped_orig_total) | set(skipped_final_total))
    label_rows: List[Dict[str, Any]] = []
    for lab in labels:
        o = orig_total.get(lab, 0)
        f = final_total.get(lab, 0)
        so = skipped_orig_total.get(lab, 0)
        sf = skipped_final_total.get(lab, 0)
        label_rows.append({
            "label": lab,
            "orig_count_total": o,
            "final_count_total": f,
            "delta_final_minus_orig": f - o,
            "skipped_orig_count_total": so,
            "skipped_final_count_total": sf,
            "skipped_delta_final_minus_orig": sf - so,
        })

    return files_rows, label_rows

def write_csv(files_rows: List[Dict[str, Any]], label_rows: List[Dict[str, Any]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 中文欄頭對應
    files_map = [
        ("stem", "檔名"),
        ("final_version", "最終版本"),
        ("final_has_comments", "最終有評論"),
        ("bbox_overlap_ratio", "原始與最終框重疊比例"),
        ("colloquial_zh_same_as_orig", "通俗描述與原版完全相同"),
        ("medical_zh_same_as_orig", "醫學描述與原版完全相同"),
        ("orig_detections", "原版標註數量"),
        ("final_detections", "最終標註數量"),
        ("global_causes_zh_orig_count", "病徵原因原版數量"),
        ("global_causes_zh_final_count", "病徵原因最終數量"),
        ("global_causes_zh_exact_same_count", "病徵原因字串完全相同數量"),
        ("global_causes_zh_final_minus_exact_same", "病徵原因最終數量減完全相同"),
        ("global_treatments_zh_orig_count", "處置建議原版數量"),
        ("global_treatments_zh_final_count", "處置建議最終數量"),
        ("global_treatments_zh_exact_same_count", "處置建議字串完全相同數量"),
        ("global_treatments_zh_final_minus_exact_same", "處置建議最終數量減完全相同"),
    ]
    files_headers_cn = [cn for _, cn in files_map]

    # 用 UTF-8-SIG 讓 Excel 直接正確顯示中文
    with (out_dir / "files_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=files_headers_cn)
        w.writeheader()
        for row in sorted(files_rows, key=lambda r: r["stem"]):
            w.writerow({cn: row.get(k, "") for k, cn in files_map})

    label_map = [
        ("label", "類別"),
        ("orig_count_total", "原版總數"),
        ("final_count_total", "最終總數"),
        ("delta_final_minus_orig", "最終減原版"),
        ("skipped_orig_count_total", "跳過的原版總數"),
        ("skipped_final_count_total", "跳過的最終總數"),
        ("skipped_delta_final_minus_orig", "跳過的最終減原版"),
    ]
    label_headers_cn = [cn for _, cn in label_map]

    def sort_key(r):
        return (abs(r["delta_final_minus_orig"]), r["label"])

    with (out_dir / "label_delta.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=label_headers_cn)
        w.writeheader()
        for row in sorted(label_rows, key=sort_key, reverse=True):
            w.writerow({cn: row.get(k, "") for k, cn in label_map})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="annotations_versions", help="含 JSON 的資料夾（可遞迴）")
    ap.add_argument("--out", type=str, required=True, help="輸出資料夾（只會產生 2 個 CSV）")
    args = ap.parse_args()

    folder = Path(args.dir)
    if not folder.exists():
        raise SystemExit(f"找不到資料夾：{folder}")

    files_rows, label_rows = analyze_dir(folder)
    write_csv(files_rows, label_rows, Path(args.out))

    out = Path(args.out).resolve()
    print(f"已輸出：{out / 'files_summary.csv'}")
    print(f"已輸出：{out / 'label_delta.csv'}")

if __name__ == "__main__":
    main()
