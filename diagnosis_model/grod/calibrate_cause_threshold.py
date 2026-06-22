"""校準 demo 病因顯示門檻 `cause_score_thresh` (τ) + `cause_max_n`。

demo 在每個 query 的候選池上以 CEAM 評分、min-max 正規化（top-1≡1.0）、再 folding
近義病因。原本固定顯示 top-n；本步改為「顯示 min-max score ≥ τ 的折疊病因，上限
`cause_max_n`、至少 1 個」。τ 用 grid 搜尋，於 fold 後的顯示集合上量 precision /
GT 覆蓋 / 平均顯示數，**目標＝平均顯示數 ≤ target 之下最大化 GT 覆蓋**（即滿足
avg≤target 的最小 τ）。與 calibrate_thresholds / calibrate_fold_threshold 同性質，
都是寫進 thresholds.json 的 demo 校準常數；γ-free（純 CEAM）。read-modify-write。

Run（預設指向 current 樹 artifacts、生產 k=3）:
  $PY -m diagnosis_model.grod.calibrate_cause_threshold
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from diagnosis_model.cause_inference.models import CEAH
from diagnosis_model.cause_inference.models.case_encoder import EncoderConfig, build_encoder
from diagnosis_model.cause_inference.phase1_baseline import build_candidate_pool, select_positive_top_cases
from diagnosis_model.grod.train_case_encoder_soft import encode_all_soft, load_soft
from diagnosis_model.grod.eval_ceah_soft_paper import minmax, select_lesions
from diagnosis_model.grod.pipeline import _fold_causes, FOLD_THRESH

ART = "data/processed/current/artifacts"


def _grid(spec: str) -> np.ndarray:
    a, b, s = (float(x) for x in spec.split(":"))
    return np.round(np.arange(a, b, s), 4)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case_db_dir", default=f"{ART}/db/case_db_jointDistRawP")
    ap.add_argument("--soft_dir", default=f"{ART}/db/soft_inputs")
    ap.add_argument("--encoder_ckpt", default=f"{ART}/models/encoder_grod_soft/best_encoder.pt")
    ap.add_argument("--bank_path", default=f"{ART}/models/encoder_grod_soft/bank_z_soft.pt")
    ap.add_argument("--ceah_ckpt", default=f"{ART}/models/ceah_grod_soft/best_ceah.pt")
    ap.add_argument("--out", default="data/processed/current/thresholds.json")
    ap.add_argument("--top_k_cases", type=int, default=3, help="生產操作點")
    ap.add_argument("--top_k_lesions", type=int, default=32)
    ap.add_argument("--semantic_threshold", type=float, default=0.95)
    ap.add_argument("--text", choices=["none", "medical", "colloquial"], default="medical")
    ap.add_argument("--n_max", type=int, default=6, help="顯示硬上限")
    ap.add_argument("--target_avg", type=float, default=5.0, help="平均顯示數上限（政策值）")
    ap.add_argument("--tau_grid", default="0.10:0.96:0.05")
    ap.add_argument("--max_queries", type=int, default=-1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    cdb = Path(args.case_db_dir)
    train_cases = torch.load(cdb / "train_cases.pt", weights_only=False)
    valid_cases = torch.load(cdb / "valid_cases.pt", weights_only=False)
    cause_embs = torch.load(cdb / "cause_text_embs.pt", weights_only=False)["embeddings"].float().to(dev)
    cause_embs_n = F.normalize(cause_embs, dim=-1)
    g_va, z_va, w_va, _ = load_soft(Path(args.soft_dir) / "valid.pt")

    enc_pkg = torch.load(args.encoder_ckpt, weights_only=False, map_location="cpu")
    encoder = build_encoder(EncoderConfig(**enc_pkg["encoder_config"])).to(dev).eval()
    encoder.load_state_dict(enc_pkg["encoder_state"])
    bank_z = torch.load(args.bank_path, weights_only=False)["bank_z"].to(dev)
    in_dim = cause_embs.size(1)
    ceah = CEAH(global_dim=in_dim, text_dim=in_dim, lesion_dim=in_dim, cause_dim=in_dim,
                common_dim=256, hidden_dim=512, dropout=0.0,
                attribution_mode="softmax", scoring_mode="multiplicative").to(dev).eval()
    ceah.load_state_dict(torch.load(args.ceah_ckpt, map_location=dev))
    H_va = encode_all_soft(encoder, g_va, z_va, w_va, dev).to(dev)

    nq = len(valid_cases) if args.max_queries < 0 else min(args.max_queries, len(valid_cases))
    # per query: list of (rep_score[min-max] desc, rep_correct bool, per-GT covered-by-rep bool[n_gt])
    per_q = []
    for qi in range(nq):
        gt = valid_cases[qi]["cause_emb_indices"]
        if not gt:
            continue
        sims = (H_va[qi:qi + 1] @ bank_z.T)[0].cpu().numpy()
        top_idx, _, _ = select_positive_top_cases(sims, args.top_k_cases)
        cand = build_candidate_pool(top_idx, train_cases)
        if len(cand) == 0:
            continue
        cand_t = torch.tensor(cand, device=dev, dtype=torch.long); P = len(cand)
        z_sel, w_sel = select_lesions(z_va[qi].float().to(dev), w_va[qi].float().to(dev), args.top_k_lesions)
        K = z_sel.size(0)
        t = valid_cases[qi][f"text_{args.text}_emb"].float().to(dev)
        s_ceah, _, _ = ceah(
            g_va[qi].float().to(dev).unsqueeze(0).expand(P, -1),
            t.unsqueeze(0).expand(P, -1), torch.ones(P, dtype=torch.bool, device=dev),
            z_sel.unsqueeze(0).expand(P, -1, -1).contiguous(),
            torch.ones(P, K, dtype=torch.bool, device=dev),
            cause_embs.index_select(0, cand_t),
            lesion_weights=w_sel.unsqueeze(0).expand(P, -1).contiguous())
        sc = minmax(s_ceah).cpu().numpy()
        cand_embs = cause_embs.index_select(0, cand_t)
        cand_embs_n = cause_embs_n.index_select(0, cand_t)
        gt_embs_n = cause_embs_n.index_select(0, torch.tensor(gt, device=dev, dtype=torch.long))
        match = ((gt_embs_n @ cand_embs_n.T) >= args.semantic_threshold).cpu().numpy()   # [n_gt, P]
        # fold（與生產同：pool-centered cosine、average linkage、cut=FOLD_THRESH），取全部 reps 排序
        groups = _fold_causes(cand_embs, s_ceah.detach().cpu().numpy(), top_n=P, fold_thresh=FOLD_THRESH)
        reps = []
        for rep_li, members in groups:
            rep_score = float(sc[rep_li])
            rep_correct = bool(match[:, members].any())
            gt_cov = match[:, members].any(axis=1)               # [n_gt] 此折疊群是否涵蓋各 GT
            reps.append((rep_score, rep_correct, gt_cov))
        reps.sort(key=lambda r: -r[0])
        per_q.append((reps, len(gt)))

    taus = _grid(args.tau_grid)
    curve = {}
    best = None
    for tau in taus:
        shown = corr = cov_n = cov_d = 0; counts = []
        for reps, n_gt in per_q:
            sel = [r for r in reps if r[0] >= tau][:args.n_max]
            if not sel:
                sel = reps[:1]
            counts.append(len(sel))
            shown += len(sel); corr += sum(1 for r in sel if r[1])
            covered = np.zeros(n_gt, dtype=bool)
            for r in sel:
                covered |= r[2]
            cov_n += int(covered.sum()); cov_d += n_gt
        prec = corr / max(shown, 1); cov = cov_n / max(cov_d, 1); avg = float(np.mean(counts))
        curve[f"{tau:.2f}"] = {"precision": round(prec, 4), "coverage": round(cov, 4), "avg_shown": round(avg, 3)}
        # 目標：avg ≤ target 下最大覆蓋 ＝ 滿足 avg≤target 的最小 τ（coverage 隨 τ 單調降）
        if avg <= args.target_avg and best is None:
            best = (float(tau), prec, cov, avg)

    if best is None:                       # 全段 avg 都 > target（不太可能）→ 取最大 τ
        tau = float(taus[-1]); c = curve[f"{tau:.2f}"]
        best = (tau, c["precision"], c["coverage"], c["avg_shown"])
    tau_star, prec, cov, avg = best

    p = Path(args.out)
    data = json.load(open(p, encoding="utf-8")) if p.exists() else {}
    data["cause_score_thresh"] = round(tau_star, 3)
    data["cause_max_n"] = int(args.n_max)
    data["cause"] = {"target_avg": args.target_avg, "avg_shown": round(avg, 3),
                     "precision": round(prec, 4), "coverage": round(cov, 4),
                     "top_k_cases": args.top_k_cases, "curve": curve}
    data["cause_note"] = ("demo 病因顯示門檻（min-max CEAM score ≥ τ、上限 cause_max_n、至少 1）。"
                          "fold 後計 precision/coverage/avg；τ=滿足 avg≤target_avg 之最小值（max coverage）。"
                          "γ-free、純 CEAM。pipeline `_load_cause_thresh()` 讀，缺鍵 fallback τ=0.6/n=6。")
    json.dump(data, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"queries={len(per_q)}  n_max={args.n_max}  target_avg={args.target_avg}")
    print(f"cause_score_thresh τ = {tau_star:.3f}  (precision {prec:.3f} / coverage {cov:.3f} / avg {avg:.2f})")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
