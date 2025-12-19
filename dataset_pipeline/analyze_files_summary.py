import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def extract_aug(name: str) -> str:
    """
    從檔名抓 augmentation 類型：
    例：aug_xflip_123_jpg / aug_box_xflip_456_jpg
    """
    s = str(name)
    m = re.match(r"^aug_(.+?)_\d+_jpg", s)
    if m:
        return m.group(1)
    m = re.match(r"^aug_(.+?)_\d+_", s)
    if m:
        return m.group(1)
    return "unknown"


def pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x*100:.2f}%"


def safe_mean(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.mean()) if len(s) else np.nan


def safe_median(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.median()) if len(s) else np.nan


def build_flag_reasons(row: pd.Series) -> str:
    reasons = []

    overlap = row.get("原始與最終框重疊比例", np.nan)
    final_ver = row.get("最終版本", np.nan)
    has_comment = row.get("最終有評論", 0)
    ann_diff = row.get("標註數量差", 0)

    lay_same = row.get("通俗描述與原版完全相同", 1)
    med_same = row.get("醫學描述與原版完全相同", 1)
    cause_changed = row.get("病徵原因最終數量減完全相同", 0)
    action_changed = row.get("處置建議最終數量減完全相同", 0)

    if pd.isna(overlap):
        reasons.append("最終標註=0或無法計算重疊")
    else:
        if overlap == 0:
            reasons.append("重疊=0(框完全不同)")
        elif overlap < 0.5:
            reasons.append("重疊<0.5(大幅更動)")
        elif overlap < 0.8:
            reasons.append("重疊<0.8(中度更動)")

    if pd.notna(final_ver) and int(final_ver) >= 2:
        reasons.append(f"最終版本={int(final_ver)}")

    if int(has_comment) == 1:
        reasons.append("有評論")

    if pd.notna(ann_diff) and int(ann_diff) != 0:
        reasons.append(f"標註數量差={int(ann_diff)}")

    if int(lay_same) == 0:
        reasons.append("通俗描述變動")
    if int(med_same) == 0:
        reasons.append("醫學描述變動")

    if pd.notna(cause_changed) and int(cause_changed) != 0:
        reasons.append("病徵原因字串變動")
    if pd.notna(action_changed) and int(action_changed) != 0:
        reasons.append("處置建議字串變動")

    return "；".join(reasons)


def compute_priority_score(df: pd.DataFrame) -> pd.Series:
    """
    分數越高越建議優先回查。
    權重你可依專案規則調整。
    """
    score = np.zeros(len(df), dtype=float)

    overlap = df["原始與最終框重疊比例"]
    final_ver = df["最終版本"]
    ann_diff = df["標註數量差"].fillna(0)
    lay_changed = (df["通俗描述與原版完全相同"] == 0).astype(int)
    med_changed = (df["醫學描述與原版完全相同"] == 0).astype(int)
    cause_changed = (df["病徵原因最終數量減完全相同"] != 0).astype(int)
    action_changed = (df["處置建議最終數量減完全相同"] != 0).astype(int)
    has_comment = (df["最終有評論"] == 1).astype(int)

    score += overlap.isna().astype(int) * 5.0
    score += (overlap.fillna(1) < 0.5).astype(int) * 4.0
    score += (overlap.fillna(1) == 0).astype(int) * 2.0
    score += (final_ver >= 2).astype(int) * 1.5
    score += ann_diff.abs().clip(0, 5) * 0.5
    score += lay_changed * 0.5
    score += med_changed * 0.5
    score += cause_changed * 0.5
    score += action_changed * 0.5
    score += has_comment * 0.25

    return pd.Series(score, index=df.index)


def main():
    parser = argparse.ArgumentParser(description="分析 files_summary.csv 並輸出文字報告到 txt")
    parser.add_argument("--input", required=True, help="files_summary.csv 路徑")
    parser.add_argument("--output", default="output", help="輸出資料夾路徑（會輸出 txt + csv）")
    parser.add_argument("--top_n", type=int, default=50, help="txt 內列出前 N 筆優先檢查清單")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_txt = out_dir / "analysis_report.txt"
    out_candidates_csv = out_dir / "review_candidates.csv"

    # 讀檔（常見：utf-8 或 utf-8-sig）
    try:
        df = pd.read_csv(args.input, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(args.input, encoding="utf-8-sig")

    # 必要欄位檢查（依你提供的檔案格式）
    required_cols = [
        "檔名",
        "最終版本",
        "最終有評論",
        "原始與最終框重疊比例",
        "原版標註數量",
        "最終標註數量",
        "通俗描述與原版完全相同",
        "醫學描述與原版完全相同",
        "病徵原因原版數量",
        "病徵原因最終數量",
        "病徵原因最終數量減完全相同",
        "處置建議原版數量",
        "處置建議最終數量",
        "處置建議最終數量減完全相同",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要欄位：{missing}\n目前欄位：{list(df.columns)}")

    # 衍生欄位
    df["增強類型"] = df["檔名"].map(extract_aug)
    df["標註數量差"] = df["最終標註數量"] - df["原版標註數量"]
    df["病徵原因差"] = df["病徵原因最終數量"] - df["病徵原因原版數量"]
    df["處置建議差"] = df["處置建議最終數量"] - df["處置建議原版數量"]

    # 整體摘要
    n = len(df)
    overlap = df["原始與最終框重疊比例"]

    overall = {
        "檔案數": n,
        "最終版本=1比例": pct((df["最終版本"] == 1).mean()),
        "最終版本=2比例": pct((df["最終版本"] == 2).mean()),
        "最終版本=3比例": pct((df["最終版本"] == 3).mean()),
        "最終有評論比例": pct(df["最終有評論"].mean()),
        "框重疊比例(平均)": f"{safe_mean(overlap):.3f}",
        "框重疊比例(中位數)": f"{safe_median(overlap):.3f}",
        "框重疊=1比例": pct((overlap == 1).mean()),
        "框重疊<0.5比例": pct((overlap < 0.5).mean()),
        "框重疊<0.8比例": pct((overlap < 0.8).mean()),
        "通俗描述有變動比例": pct((df["通俗描述與原版完全相同"] == 0).mean()),
        "醫學描述有變動比例": pct((df["醫學描述與原版完全相同"] == 0).mean()),
        "標註數量有變動比例": pct((df["標註數量差"] != 0).mean()),
        "最終標註數量=0比例": pct((df["最終標註數量"] == 0).mean()),
        "病徵原因字串有變動比例": pct((df["病徵原因最終數量減完全相同"] != 0).mean()),
        "處置建議字串有變動比例": pct((df["處置建議最終數量減完全相同"] != 0).mean()),
    }

    # 依增強類型彙總
    by_type = (
        df.groupby("增強類型")
        .agg(
            n=("檔名", "count"),
            overlap_mean=("原始與最終框重疊比例", "mean"),
            overlap_median=("原始與最終框重疊比例", "median"),
            overlap_lt05=("原始與最終框重疊比例", lambda s: (s < 0.5).mean()),
            overlap_lt08=("原始與最終框重疊比例", lambda s: (s < 0.8).mean()),
            ann_changed=("標註數量差", lambda s: (s != 0).mean()),
            lay_changed=("通俗描述與原版完全相同", lambda s: (s == 0).mean()),
            med_changed=("醫學描述與原版完全相同", lambda s: (s == 0).mean()),
            final_ver_ge2=("最終版本", lambda s: (s >= 2).mean()),
            comment_rate=("最終有評論", "mean"),
        )
        .reset_index()
    )

    # 格式化
    by_type["overlap_mean"] = by_type["overlap_mean"].round(3)
    by_type["overlap_median"] = by_type["overlap_median"].round(3)
    for c in ["overlap_lt05", "overlap_lt08", "ann_changed", "lay_changed", "med_changed", "final_ver_ge2", "comment_rate"]:
        by_type[c] = (by_type[c] * 100).round(2)

    # 依最終版本彙總
    by_ver = (
        df.groupby("最終版本")
        .agg(
            n=("檔名", "count"),
            comment_rate=("最終有評論", "mean"),
            overlap_mean=("原始與最終框重疊比例", "mean"),
            overlap_median=("原始與最終框重疊比例", "median"),
            ann_changed=("標註數量差", lambda s: (s != 0).mean()),
            lay_changed=("通俗描述與原版完全相同", lambda s: (s == 0).mean()),
            med_changed=("醫學描述與原版完全相同", lambda s: (s == 0).mean()),
        )
        .reset_index()
        .sort_values("最終版本")
    )

    by_ver["overlap_mean"] = by_ver["overlap_mean"].round(3)
    by_ver["overlap_median"] = by_ver["overlap_median"].round(3)
    for c in ["comment_rate", "ann_changed", "lay_changed", "med_changed"]:
        by_ver[c] = (by_ver[c] * 100).round(2)

    # 優先檢查清單（含原因 + score）
    df["flag_reasons"] = df.apply(build_flag_reasons, axis=1)
    df["priority_score"] = compute_priority_score(df)

    candidates = df[
        (df["原始與最終框重疊比例"].isna())
        | (df["原始與最終框重疊比例"] < 0.5)
        | (df["最終版本"] >= 2)
        | (df["標註數量差"] != 0)
        | (df["最終有評論"] == 1)
    ].copy()

    candidates = candidates.sort_values(
        ["priority_score", "原始與最終框重疊比例"],
        ascending=[False, True],
        na_position="first",
    )

    out_cols = [
        "檔名",
        "增強類型",
        "priority_score",
        "flag_reasons",
        "最終版本",
        "最終有評論",
        "原始與最終框重疊比例",
        "原版標註數量",
        "最終標註數量",
        "標註數量差",
        "通俗描述與原版完全相同",
        "醫學描述與原版完全相同",
        "病徵原因原版數量",
        "病徵原因最終數量",
        "病徵原因最終數量減完全相同",
        "處置建議原版數量",
        "處置建議最終數量",
        "處置建議最終數量減完全相同",
    ]
    candidates_out = candidates[out_cols]

    # 輸出 CSV
    candidates_out.to_csv(out_candidates_csv, index=False, encoding="utf-8-sig")

    # ===== 寫入 TXT 報告 =====
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("========================================\n")
        f.write("files_summary.csv 分析報告\n")
        f.write(f"產生時間：{now}\n")
        f.write("========================================\n\n")

        # 文字敘述（你可以改成更符合你研究的語氣）
        f.write("【報告用途】\n")
        f.write("- 用於檢視原始標註與最終標註差異（框重疊/數量變動/描述變動）。\n")
        f.write("- 用於比較不同增強(augmentation)類型是否更容易導致需要人工修正。\n")
        f.write("- 產出優先檢查清單：重疊很低、版本>=2、標註數量改動、或有評論者優先回查。\n\n")

        f.write("【整體摘要（KPI）】\n")
        for k, v in overall.items():
            f.write(f"- {k}：{v}\n")
        f.write("\n")

        # 框重疊分位數
        f.write("【框重疊比例分位數】\n")
        q = overlap.dropna().quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        for idx, val in q.items():
            f.write(f"- P{int(idx*100):02d}：{float(val):.3f}\n")
        f.write("\n")

        # 增強類型排名
        f.write("【依增強類型彙總（重點欄位）】\n")
        f.write("說明：以下比例欄位皆為 %。\n\n")

        show_cols_type = [
            "增強類型", "n",
            "overlap_mean", "overlap_median",
            "overlap_lt05", "overlap_lt08",
            "ann_changed", "final_ver_ge2",
            "lay_changed", "med_changed",
            "comment_rate"
        ]

        # 依 n 排序（看量最大）
        f.write("－(A) 依樣本數 n 由大到小（前 20）\n")
        f.write(by_type.sort_values("n", ascending=False)[show_cols_type].head(20).to_string(index=False))
        f.write("\n\n")

        # 依 overlap_lt05 排序（看最容易大改的增強）
        f.write("－(B) 依「重疊<0.5」比例由大到小（前 20）\n")
        f.write(by_type.sort_values("overlap_lt05", ascending=False)[show_cols_type].head(20).to_string(index=False))
        f.write("\n\n")

        # 依 final_ver_ge2 排序（看反覆修正可能性）
        f.write("－(C) 依「最終版本>=2」比例由大到小（前 20）\n")
        f.write(by_type.sort_values("final_ver_ge2", ascending=False)[show_cols_type].head(20).to_string(index=False))
        f.write("\n\n")

        # 最終版本彙總
        f.write("【依最終版本彙總】\n")
        f.write("說明：比例欄位皆為 %。\n\n")
        f.write(by_ver.to_string(index=False))
        f.write("\n\n")

        # 建議解讀
        f.write("【解讀建議（可直接當報告文字）】\n")
        f.write("1) 若「重疊<0.5」比例高：代表框位置或框集合被大幅修正，建議回查增強轉換/標註流程。\n")
        f.write("2) 若某增強類型的「最終版本>=2」比例高：代表同類型樣本更常需要二次以上修改，可能有系統性問題。\n")
        f.write("3) 「標註數量有變動」若很高：可能是漏標/多標或增強造成目標消失/出現，需要確認規則。\n")
        f.write("4) 「最終標註數量=0」：通常代表全刪或完全不一致，建議獨立抽查並追原因。\n\n")

        # 優先檢查清單
        top_n = max(1, int(args.top_n))
        f.write(f"【優先檢查清單（前 {top_n} 筆，分數越高越優先）】\n")
        f.write("欄位說明：priority_score 為綜合分數；flag_reasons 為觸發原因。\n\n")
        f.write(
            candidates_out.head(top_n)[
                ["檔名", "增強類型", "priority_score", "原始與最終框重疊比例", "最終版本", "最終有評論", "標註數量差", "flag_reasons"]
            ].to_string(index=False)
        )
        f.write("\n\n")

        # 追加一些統計：最常見的觸發原因
        f.write("【常見觸發原因（粗略統計）】\n")
        reason_counts = (
            df["flag_reasons"]
            .fillna("")
            .str.split("；")
            .explode()
        )
        reason_counts = reason_counts[reason_counts != ""].value_counts().head(30)
        for r, c in reason_counts.items():
            f.write(f"- {r}：{int(c)}\n")
        f.write("\n")

        f.write("（完）\n")

    print(f"[OK] 已輸出 TXT 報告：{out_txt}")
    print(f"[OK] 已輸出 優先檢查CSV：{out_candidates_csv}")


if __name__ == "__main__":
    main()
