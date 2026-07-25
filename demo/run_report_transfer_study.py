"""Run the exploratory external-report transfer study used in thesis Section 5.5.3.

Each directory below ``data/report`` is treated as one clinical case.  Its
``病例病因.txt`` contains the case history on the first non-empty line and the
expert diagnosis on the second.  Every image is inferred independently because
the current interactive system accepts one image at a time; the case history is
supplied as the optional text input.  Expert diagnoses are used only after
inference to check whether the displayed cause list contains the corresponding
disease entity.

This is a small, external, descriptive transfer study.  It is not intended to
estimate clinical sensitivity or specificity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diagnosis_model.grod.pipeline import (  # noqa: E402
    ABSTAIN_DEFAULT,
    DISPLAY_DEFAULT,
    data_version,
    encode_cause_texts,
    encode_text_slot,
    get_pipeline,
    load_shared,
    render_detection_image,
)
from diagnosis_model.grod.bank_delta import BankDelta, DeltaCase  # noqa: E402


# Disease-entity matching is deliberately lexical and conservative.  It checks
# the representative and all folded members shown by the production report.
# Keys are the report-directory names, which are the submitted case IDs.
TARGETS = {
    "AOD109224": [
        {"label": "分枝桿菌", "aliases": ["分枝桿", "mycobacter"]},
        {
            "label": "豚鼠氣單胞桿菌",
            "aliases": [
                "豚鼠氣單胞", "豚鼠產氣單胞", "豚鼠產氣單孢",
                "aeromonas caviae",
            ],
        },
        {
            "label": "腦膜膿毒性金黃桿菌",
            "aliases": [
                "腦膜膿毒性金黃桿菌", "腦膜膿毒金黃桿菌",
                "chryseobacterium meningosepticum",
                "elizabethkingia meningoseptica",
            ],
        },
    ],
    "AOD109265": [
        {"label": "車輪蟲", "aliases": ["車輪蟲", "trichodin"]},
        {"label": "指環蟲", "aliases": ["指環蟲", "dactylogyr"]},
        {"label": "三代蟲", "aliases": ["三代蟲", "gyrodactyl"]},
        {
            "label": "溫和產氣單胞菌",
            "aliases": [
                "溫和氣單胞", "溫和產氣單胞", "溫和產氣單孢",
                "aeromonas sobria",
            ],
        },
        {
            "label": "豚鼠產氣單胞菌",
            "aliases": [
                "豚鼠氣單胞", "豚鼠產氣單胞", "豚鼠產氣單孢",
                "aeromonas caviae",
            ],
        },
    ],
    "AOD109268": [
        {"label": "四膜蟲", "aliases": ["四膜蟲", "tetrahymen"]},
        {"label": "三代蟲", "aliases": ["三代蟲", "gyrodactyl"]},
        {"label": "指環蟲", "aliases": ["指環蟲", "dactylogyr"]},
        {
            "label": "溫和產氣單胞菌",
            "aliases": [
                "溫和氣單胞", "溫和產氣單胞", "溫和產氣單孢",
                "aeromonas sobria",
            ],
        },
        {
            "label": "豚鼠產氣單胞菌",
            "aliases": [
                "豚鼠氣單胞", "豚鼠產氣單胞", "豚鼠產氣單孢",
                "aeromonas caviae",
            ],
        },
    ],
    "AOD109313": [
        {
            "label": "甲狀腺增生",
            "aliases": ["甲狀腺增生", "thyroid hyperplasia"],
            "related_aliases": ["甲狀腺腫", "goiter", "goitre"],
        },
        {"label": "乳酸球菌", "aliases": ["乳酸球", "乳酸鏈球", "lactococc"]},
    ],
    "AOD109315": [
        {"label": "愛德華氏菌", "aliases": ["愛德華", "edwardsiella"]},
    ],
    "AOD110001": [
        {
            "label": "溫和產氣單胞菌",
            "aliases": [
                "溫和氣單胞", "溫和產氣單胞", "溫和產氣單孢",
                "aeromonas sobria",
            ],
        },
        {
            "label": "缺碘性甲狀腺腫",
            "aliases": [
                "缺碘性甲狀腺腫", "缺碘", "碘缺乏",
                "iodine-deficiency", "iodine deficiency",
            ],
            "related_aliases": ["甲狀腺腫", "goiter", "goitre"],
        },
    ],
    "AOD110211": [
        {"label": "乳酸鏈球菌", "aliases": ["乳酸鏈球", "乳酸球", "lactococc"]},
        {"label": "努卡氏菌", "aliases": ["努卡", "奴卡", "nocardi"]},
        {"label": "車輪蟲", "aliases": ["車輪蟲", "trichodin"]},
    ],
    "AOD110228": [
        {"label": "乳酸球菌", "aliases": ["乳酸球", "乳酸鏈球", "lactococc"]},
        {"label": "弧菌", "aliases": ["弧菌", "vibrio"]},
    ],
    "AOD111324": [
        {"label": "奴卡氏菌", "aliases": ["奴卡", "nocardi"]},
        {
            "label": "革蘭氏陰性桿菌",
            "aliases": ["革蘭氏陰性", "革蘭陰性", "gram-negative", "gram negative"],
        },
    ],
    "AOD111383": [
        {
            "label": "創傷弧菌",
            "aliases": ["創傷弧菌", "v. vulnificus", "vibrio vulnificus", "vulnificus"],
            "related_aliases": ["弧菌", "vibrio"],
        },
        {
            "label": "台灣石斑魚虹彩病毒",
            "aliases": ["台灣石斑魚虹彩病毒", "tgiv"],
            "related_aliases": ["虹彩病毒", "虹彩", "iridovirus"],
        },
    ],
    "LAM110079": [
        {"label": "鏈球菌", "aliases": ["鏈球", "streptococc"]},
        {"label": "弧菌（懷疑）", "aliases": ["弧菌", "vibrio"]},
    ],
}

INTERNAL_TERMS = (
    "內臟",
    "臟器",
    "腹腔",
    "肝",
    "脾",
    "腎",
    "心",
    "腦",
    "腸",
    "胃",
    "膽",
)
NO_OBVIOUS_LESION_TERMS = ("正常", "無明顯", "並無", "未見")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CJK_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
CASE_FIGURE_IDS = {
    case_id: case_id
    for case_id in TARGETS
}


def _read_case_text(path: Path) -> tuple[str, str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"expected at least two non-empty lines in {path}")
    return lines[0], lines[1]


def _image_modality(path: Path) -> str:
    return "internal_or_dissection" if any(term in path.stem for term in INTERNAL_TERMS) else "external"


def _finding_type(path: Path) -> str:
    return ("no_obvious_lesion" if any(term in path.stem for term in NO_OBVIOUS_LESION_TERMS)
            else "abnormal_or_unspecified")


def _cause_haystack(cause: dict) -> str:
    return "\n".join([str(cause.get("text", "")), *map(str, cause.get("members", []))]).casefold()


def _target_rank(causes: list[dict], aliases: list[str]) -> int | None:
    folded_aliases = [alias.casefold() for alias in aliases]
    for cause in causes:
        haystack = _cause_haystack(cause)
        if any(alias in haystack for alias in folded_aliases):
            return int(cause["rank"])
    return None


def _serialize_cause(cause: dict) -> dict:
    return {
        "rank": int(cause["rank"]),
        "text": str(cause["text"]),
        "score": round(float(cause["score"]), 6),
        "members": list(map(str, cause.get("members", []))),
    }


def _font(size: int, weight: str = "normal") -> FontProperties:
    kwargs = {"size": size, "weight": weight}
    if CJK_FONT.exists():
        kwargs["fname"] = str(CJK_FONT)
    return FontProperties(**kwargs)


def _truncate(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: max(limit - 1, 1)] + "…"


def _wrapped_diagnosis(text: str) -> str:
    compact = _truncate(text, 88)
    lines = textwrap.wrap(
        f"報告診斷：{compact}",
        width=46,
        break_long_words=True,
        break_on_hyphens=False,
    )
    return "\n".join(lines[:2])


def _figure_id(case_name: str) -> str:
    configured = CASE_FIGURE_IDS.get(case_name)
    if configured:
        return configured
    return hashlib.sha1(case_name.encode("utf-8")).hexdigest()[:10]


def _select_representative_images(records: list[dict], limit: int = 3) -> list[dict]:
    """Prefer images with displayed boxes, then rank by maximum objectness."""
    ranked = sorted(
        records,
        key=lambda record: (
            not record["abstain"] and record["n_lesions"] > 0,
            record["max_objectness"],
        ),
        reverse=True,
    )
    return ranked[:limit]


def _make_case_figure(case: dict, output_dir: Path) -> dict:
    selected = _select_representative_images(case["images"])
    if not selected:
        raise ValueError(f"no images available for {case['case_name']}")

    rows = len(selected)
    fig = plt.figure(figsize=(12.2, 2.75 * rows + 1.05), facecolor="white")
    grid = fig.add_gridspec(
        rows,
        2,
        width_ratios=(1.05, 1.55),
        left=0.035,
        right=0.985,
        top=0.87,
        bottom=0.025,
        hspace=0.12,
        wspace=0.055,
    )
    fig.suptitle(
        f"病例編號：{_truncate(case['case_name'], 58)}\n"
        f"{_wrapped_diagnosis(case['expert_diagnosis'])}",
        fontproperties=_font(18, "bold"),
        y=0.975,
        linespacing=1.3,
    )

    for row, record in enumerate(selected):
        image = ImageOps.exif_transpose(Image.open(record["image_path"])).convert("RGB")
        overlay = render_detection_image(image, record["lesions"])
        image_ax = fig.add_subplot(grid[row, 0])
        image_ax.imshow(overlay)
        image_ax.set_xticks([])
        image_ax.set_yticks([])

        text_ax = fig.add_subplot(grid[row, 1])
        text_ax.axis("off")
        lines: list[tuple[str, str, str]] = [
            (f"圖片描述：{_truncate(record['description'], 28)}", "bold", "#202020"),
        ]
        lines.append(("預測病灶類別（Top-3）：", "bold", "#9f2020"))
        if record["lesions"]:
            for lesion in record["lesions"][:3]:
                lines.append((
                    f"  • {_truncate(lesion['label'], 22)}（異常分數 {lesion['det_score']:.3f}）",
                    "normal",
                    "#202020",
                ))
        else:
            lines.append(("  • 無；未達異常判定門檻", "normal", "#606060"))

        lines.append(("模型預測病因（Top-3）：", "bold", "#1d4d8f"))
        causes = record["displayed_causes"][:3]
        if causes:
            for cause in causes:
                lines.append((
                    f"  {cause['rank']}. {_truncate(cause['text'], 28)}",
                    "normal",
                    "#202020",
                ))
        else:
            reason = "未執行病因排序（未達異常判定門檻）"
            lines.append((f"  {_truncate(reason, 36)}", "normal", "#606060"))

        y = 0.98
        step = min(0.105, 0.92 / max(len(lines) - 1, 1))
        for text_value, weight, color in lines:
            text_ax.text(
                0.01,
                y,
                text_value,
                va="top",
                ha="left",
                transform=text_ax.transAxes,
                fontproperties=_font(15, weight),
                color=color,
            )
            y -= step
        if row < rows - 1:
            text_ax.plot(
                [0.0, 1.0],
                [-0.04, -0.04],
                color="#cfcfcf",
                linewidth=0.8,
                transform=text_ax.transAxes,
                clip_on=False,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"report_transfer_{_figure_id(case['case_name'])}.png"
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)
    return {
        "path": str(output_path.resolve()),
        "selected_images": [record["file_name"] for record in selected],
        "selection": "non-abstained images with displayed boxes first, then descending maximum objectness",
    }


def _summarize_case(case: dict) -> dict:
    images = case["images"]
    n_targets = len(case["targets"])
    target_summary = []
    for target in case["targets"]:
        hits = [img for img in images if img["target_ranks"][target["label"]] is not None]
        related_hits = [
            img for img in images
            if img["related_family_ranks"].get(target["label"]) is not None
        ]
        target_summary.append({
            "label": target["label"],
            "images_with_display_hit": len(hits),
            "external_hits": sum(img["modality"] == "external" for img in hits),
            "internal_or_dissection_hits": sum(
                img["modality"] == "internal_or_dissection" for img in hits),
            "best_display_rank": min(
                (img["target_ranks"][target["label"]] for img in hits), default=None),
            "images_with_related_family_only": len(related_hits),
            "best_related_family_rank": min(
                (img["related_family_ranks"][target["label"]] for img in related_hits),
                default=None),
        })
    return {
        "case_name": case["case_name"],
        "expert_diagnosis": case["expert_diagnosis"],
        "n_images": len(images),
        "n_external": sum(img["modality"] == "external" for img in images),
        "n_internal_or_dissection": sum(
            img["modality"] == "internal_or_dissection" for img in images),
        "n_no_obvious_lesion": sum(img["finding_type"] == "no_obvious_lesion" for img in images),
        "n_abstain": sum(img["abstain"] for img in images),
        "n_images_with_any_target": sum(
            any(rank is not None for rank in img["target_ranks"].values()) for img in images),
        "n_images_with_all_targets": sum(
            n_targets > 0 and all(rank is not None for rank in img["target_ranks"].values())
            for img in images),
        "targets": target_summary,
        "case_union_targets_recovered": sum(t["images_with_display_hit"] > 0 for t in target_summary),
        "case_union_targets_total": n_targets,
    }


def _load_online_delta(pipe, data_root: Path) -> int:
    """Populate the in-process retrieval bank with the annotation_web online
    increment (base ⊕ delta), so the study can see results including expert-
    submitted cases without promoting them to a new dataset version.

    Mirrors serve /bank/upsert exactly (encode_case + encode_cause_texts +
    bank_upsert), but reads the writable datasets (created_via:diagnosis) straight
    off disk. Read-only on the annotation_web store; nothing is written back."""
    n = 0
    for meta_path in sorted(data_root.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("created_via") != "diagnosis" or meta.get("locked"):
            continue
        ds_dir = meta_path.parent
        ds = ds_dir.name
        db = ds_dir / "annotations.db"
        if not db.exists():
            continue
        con = sqlite3.connect(str(db))
        try:
            rows = con.execute(
                "SELECT task_id, doc_json FROM tasks WHERE is_healthy=0").fetchall()
        finally:
            con.close()
        for task_id, doc_json in rows:
            doc = json.loads(doc_json)
            if len(doc.get("detections", [])) == 0:
                continue
            causes = [c for c in (doc.get("global_causes_zh") or []) if str(c).strip()]
            if not causes:
                continue
            img_path = ds_dir / "images" / doc["image_filename"]
            if not img_path.exists():
                print(f"[delta] image missing, skip {ds}/{task_id}: {img_path}")
                continue
            image = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
            z = pipe.p.encode_case(image)
            if z is None:                       # healthy (no lesion clears threshold)
                continue
            rec = DeltaCase(
                case_id=BankDelta.make_case_id(ds, task_id),
                source_dataset=ds, source_task_id=task_id,
                image_path=str(img_path), file_name=img_path.name,
                z=z, cause_texts=causes, cause_embs=encode_cause_texts(causes))
            pipe.p.bank_upsert(rec)
            n += 1
    return n


def run(args: argparse.Namespace) -> dict:
    report_root = args.report_root.resolve()
    case_dirs = sorted(path for path in report_root.iterdir() if path.is_dir())
    unknown = [path.name for path in case_dirs if path.name not in TARGETS]
    if unknown:
        print(f"[transfer] no strict target matcher configured for: {unknown}")

    print(f"[transfer] cases={len(case_dirs)} device study starts")
    load_shared()
    pipe = get_pipeline("grod_soft")
    n_delta = 0
    if args.with_delta:
        n_delta = _load_online_delta(pipe, args.data_root.resolve())
        print(f"[transfer] online delta loaded: {n_delta} cases "
              f"(bank_size={pipe.p.bank_size})")
    vocabulary = list(map(str, pipe.p.cause_texts))
    target_vocabulary = {}
    for case_name, targets in TARGETS.items():
        target_vocabulary[case_name] = []
        for target in targets:
            strict_aliases = [alias.casefold() for alias in target["aliases"]]
            related_aliases = [alias.casefold() for alias in target.get("related_aliases", [])]
            target_vocabulary[case_name].append({
                "label": target["label"],
                "strict_vocabulary_matches": sum(
                    any(alias in text.casefold() for alias in strict_aliases)
                    for text in vocabulary),
                "related_family_vocabulary_matches": sum(
                    any(alias in text.casefold() for alias in related_aliases)
                    for text in vocabulary),
            })
    cases = []
    for case_dir in case_dirs:
        history, diagnosis = _read_case_text(case_dir / "病例病因.txt")
        text_emb = encode_text_slot(history)
        targets = TARGETS.get(case_dir.name, [])
        image_paths = sorted(
            path for path in case_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        records = []
        for image_path in image_paths:
            image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
            result = pipe.infer_rich(
                image,
                text_emb=text_emb,
                top_k_cases=args.top_k_cases,
                top_n=args.top_n_causes,
                abstain_thresh=ABSTAIN_DEFAULT,
                display_thresh=DISPLAY_DEFAULT,
            )
            causes = [_serialize_cause(cause) for cause in result.get("top_n", [])]
            obj_all = result.get("obj_all")
            max_objectness = (
                float(obj_all.max().item())
                if obj_all is not None and obj_all.numel()
                else 0.0
            )
            lesions = [
                {
                    "bbox_xywh": list(map(float, lesion["bbox_xywh"])),
                    "det_score": float(lesion["det_score"]),
                    "label": str(lesion["cls"]["label_zh"]),
                }
                for lesion in result.get("lesions", [])
            ]
            target_ranks = {
                target["label"]: _target_rank(causes, target["aliases"])
                for target in targets
            }
            related_family_ranks = {
                target["label"]: (
                    None if target_ranks[target["label"]] is not None
                    else _target_rank(causes, target.get("related_aliases", []))
                )
                for target in targets
            }
            records.append({
                "file_name": image_path.name,
                "image_path": str(image_path.resolve()),
                "description": image_path.stem,
                "modality": _image_modality(image_path),
                "finding_type": _finding_type(image_path),
                "abstain": bool(result.get("abstain", False)),
                "n_lesions": int(result.get("n_lesions", 0)),
                "max_objectness": round(max_objectness, 6),
                "lesions": lesions,
                "lesion_labels": [lesion["label"] for lesion in lesions],
                "target_ranks": target_ranks,
                "related_family_ranks": related_family_ranks,
                "displayed_causes": causes,
            })
            print(
                f"[transfer] case={case_dir.name} image={image_path.name} "
                f"modality={records[-1]['modality']} abstain={records[-1]['abstain']} "
                f"targets={target_ranks}"
            )
        case = {
            "case_name": case_dir.name,
            "case_history": history,
            "expert_diagnosis": diagnosis,
            "targets": targets,
            "images": records,
        }
        case["summary"] = _summarize_case(case)
        if not args.skip_figures:
            case["figure"] = _make_case_figure(case, args.figure_dir.resolve())
        cases.append(case)

    summaries = [case["summary"] for case in cases]
    target_rows = [target for summary in summaries for target in summary["targets"]]
    overall = {
        "n_cases": len(cases),
        "n_images": sum(summary["n_images"] for summary in summaries),
        "n_external": sum(summary["n_external"] for summary in summaries),
        "n_internal_or_dissection": sum(
            summary["n_internal_or_dissection"] for summary in summaries),
        "n_no_obvious_lesion": sum(summary["n_no_obvious_lesion"] for summary in summaries),
        "n_abstain": sum(summary["n_abstain"] for summary in summaries),
        "n_images_with_any_target": sum(
            summary["n_images_with_any_target"] for summary in summaries),
        "n_images_with_all_targets": sum(
            summary["n_images_with_all_targets"] for summary in summaries),
        "case_target_pairs_recovered_in_union": sum(
            target["images_with_display_hit"] > 0 for target in target_rows),
        "case_target_pairs_total": len(target_rows),
        "cases_with_all_targets_recovered_in_union": sum(
            summary["case_union_targets_total"] > 0
            and summary["case_union_targets_recovered"] == summary["case_union_targets_total"]
            for summary in summaries),
    }
    payload = {
        "study": {
            "design": "external exploratory transfer study; per-image inference with case-history text",
            "model_mode": "grod_soft",
            "data_version": data_version(),
            "online_delta_included": bool(args.with_delta),
            "online_delta_cases": n_delta,
            "bank_size": int(pipe.p.bank_size),
            "top_k_cases": args.top_k_cases,
            "top_n_causes": args.top_n_causes,
            "thresholds": {"abstain": ABSTAIN_DEFAULT, "display": DISPLAY_DEFAULT},
            "matching": "conservative lexical disease-entity match over displayed representative and folded members",
            "cause_vocabulary_size": len(vocabulary),
            "target_vocabulary": target_vocabulary,
        },
        "overall": overall,
        "case_summaries": summaries,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"overall": overall, "case_summaries": summaries}, ensure_ascii=False, indent=2))
    print(f"[transfer] wrote {args.output}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=REPO_ROOT / "data" / "report")
    # Output (results.json + figures) defaults into a per-mode folder under
    # paper/demo/ so the two runs never clobber each other:
    #   base         → paper/demo/base/
    #   --with-delta → paper/demo/with_delta/
    # (explicit --output / --figure-dir always win.)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--with-delta", action="store_true",
        help="include the annotation_web online increment (created_via:diagnosis "
             "writable datasets) in the retrieval bank; read-only, no dataset version built. "
             "Writes to paper/demo/with_delta/ instead of paper/demo/base/.")
    parser.add_argument(
        "--data-root", type=Path, default=REPO_ROOT / "data" / "annotation",
        help="annotation_web DATA_ROOT (writable datasets live here); used with --with-delta")
    parser.add_argument("--top-k-cases", type=int, default=3)
    parser.add_argument("--top-n-causes", type=int, default=3)
    args = parser.parse_args()
    mode_dir = REPO_ROOT / "paper" / "demo" / ("with_delta" if args.with_delta else "base")
    if args.output is None:
        args.output = mode_dir / "report_transfer_results.json"
    if args.figure_dir is None:
        args.figure_dir = mode_dir
    return args


if __name__ == "__main__":
    run(parse_args())
