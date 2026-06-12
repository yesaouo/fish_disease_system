"""Gradio demo — three GROD-family pipelines on the new dataset tree.

Modes (UI dropdown, lazy-loaded, live-switchable):
  base       conventional separated baseline: RF-DETR + raw SigLIP2 global +
             standard-finetuned SigLIP2 lesion crops → DeepSets → dense retrieval → CEAH
  grod_soft  diagnosis_model/grod/gpu_infer_soft.py   (soft per-query weights, default)
  grod       same soft model + artifacts as grod_soft, but the per-query weights are
             hard-gated to {0,1} at display_thresh (objectness > τ). This is exactly
             the {0,1}-weight degenerate of the soft path (DeepSets/CEAH reduce
             bytes-exactly to the hard gate), so grod vs grod_soft is a pure
             input-selection ablation on one trained model — no separate hard
             encoder/CEAH. (Legacy hard CLI diagnosis_model/grod/gpu_infer.py is
             retired and no longer wired here.)

Display:
  detailed mode only:
    detection · per-lesion classification cards · retrieved cases · top-N causes + α
    + per-module parameter counts
    + per-stage latency averaged over N CUDA-synced runs (warm-up dropped)

Thresholds (two decoupled, dataset-calibrated — calibrate_thresholds.py writes
data/processed/current/thresholds.json; the sliders default to it):
  abstain_thresh : 健/病判定 (max objectness, Youden-optimal).
  display_thresh : 顯示/選框 (per-query, F2 recall-leaning) → feeds the ②
                   classification cards.
  Step ① shows the A objectness heatmap (all queries splatted), not boxes.
  Rationale: diagnosis_model/grod/LESION_GATE.md.

All artifacts live under data/processed/current/artifacts (15-symptom new tree).
Text box is optional and feeds the CEAH text slot (frozen SigLIP2 text space) for
every mode; the text-α row only appears when text was actually supplied. No grad
heatmap (GROD semantic z / finetuned lesion features have no per-patch grid to
back-prop through — interpretability is carried by the CEAH α attribution).

Run from repo root:
  /home/lab603/anaconda3/envs/SDM/bin/python demo/app_gradio.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "diagnosis_model" / "vl_classifier"))

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

from common import get_image_features, get_text_features  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Paths (new dataset tree)
# ---------------------------------------------------------------------------
ART = REPO_ROOT / "data/processed/current/artifacts"
DET_TRAIN = REPO_ROOT / "data/processed/current/detection/train"
DET_VALID = REPO_ROOT / "data/processed/current/detection/valid"
SYMPTOMS_JSON = REPO_ROOT / "data/processed/current/symptoms.json"
ANCHORS_PT = ART / "models/text_anchors.pt"
RAW_SIGLIP_NAME = "google/siglip2-base-patch16-224"

# shared joint / global / base detector
JOINT_CKPT = ART / "models/joint_rfdetr/checkpoint_best_regular.pth"
GLOBAL_SD = ART / "models/distilled_global_rawP/global_embed_state_dict.pt"
BASE_DETECTOR = ART / "models/rfdetr/checkpoint_best_total.pth"

# grod_soft (shared by grod hard-gate mode)
SOFT_ENC = ART / "models/encoder_grod_soft/best_encoder.pt"
SOFT_CEAH = ART / "models/ceah_grod_soft/best_ceah.pt"
SOFT_BANK = ART / "models/encoder_grod_soft/bank_z_soft.pt"
SOFT_CASE_DB = ART / "db/case_db_jointDistRawP"
# base
BASE_VLM_LESION = ART / "models/siglip2_base_finetuned"
BASE_ENC = ART / "models/encoder_base/best_encoder.pt"
BASE_CEAH = ART / "models/ceah_base/best_ceah.pt"
BASE_CASE_DB = ART / "db/case_db_base"

CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
MAX_TOPN_BUTTONS = 10
N_TIMING_RUNS = 20          # detailed mode: timed runs (first one dropped as warm-up)

# Two decoupled GROD objectness thresholds, dataset-calibrated by
# diagnosis_model/grod/calibrate_thresholds.py; fall back to legacy values.
THRESHOLDS_JSON = REPO_ROOT / "data/processed/current/thresholds.json"


def _load_thresholds():
    """(abstain_thresh, display_thresh) — abstain=健/病判定, display=顯示選框。"""
    try:
        d = json.load(open(THRESHOLDS_JSON))
        return float(d["abstain_thresh"]), float(d["display_thresh"])
    except Exception:
        return 0.5, 0.35


ABSTAIN_DEFAULT, DISPLAY_DEFAULT = _load_thresholds()


def _font(size: int = 12) -> FontProperties:
    if os.path.exists(CJK_FONT_PATH):
        return FontProperties(fname=CJK_FONT_PATH, size=size)
    return FontProperties(size=size)


def _sync_now() -> float:
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


def _nparams(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _detector_params(net) -> Dict[str, int]:
    """Architecture-level breakdown of the (LW-)DETR detector. query/refpoint
    embeddings are folded into the encoder/decoder line; GROD heads only appear
    on the joint model."""
    def g(name):
        return _nparams(getattr(net, name)) if hasattr(net, name) else 0
    d = {
        "backbone (DINOv2)": g("backbone"),
        "DETR encoder/decoder": g("transformer") + g("query_feat") + g("refpoint_embed"),
        "box head": g("bbox_embed"),
        "objectness head": g("class_embed"),
    }
    if hasattr(net, "semantic_embed"):
        d["semantic head"] = g("semantic_embed")
    if hasattr(net, "global_embed"):
        d["global head"] = g("global_embed")
    return d


# ---------------------------------------------------------------------------
# Shared resources (loaded once): anchors, symptom names, raw SigLIP2 text tower
# ---------------------------------------------------------------------------

class Shared:
    anchor_embs = None          # [C, 768] L2-normalized (index = symptom_category_id)
    cat_names = None            # {cat_id: zh name}
    lesion_cats = None          # [c for c in 1..C-1]  (exclude 0=healthy_region)
    raw_siglip = None           # for the global branch (base) + text-slot encoding (all modes)
    raw_siglip_proc = None


def load_shared():
    pack = torch.load(ANCHORS_PT, weights_only=False, map_location=DEVICE)
    Shared.anchor_embs = F.normalize(pack["anchor_embs"].float().to(DEVICE), dim=-1)
    C = Shared.anchor_embs.size(0)
    Shared.lesion_cats = [c for c in range(1, C)]   # 0 = healthy_region
    lm = json.load(open(SYMPTOMS_JSON, encoding="utf-8")).get("label_map", {})
    Shared.cat_names = {}
    for k, v in lm.items():
        Shared.cat_names[int(k)] = v["zh"] if isinstance(v, dict) else str(v)
    from transformers import AutoModel, AutoProcessor
    Shared.raw_siglip = AutoModel.from_pretrained(RAW_SIGLIP_NAME).to(DEVICE).eval()
    Shared.raw_siglip_proc = AutoProcessor.from_pretrained(RAW_SIGLIP_NAME)
    print(f"[shared] anchors={tuple(Shared.anchor_embs.shape)} "
          f"lesion_cats={len(Shared.lesion_cats)} raw SigLIP2 loaded")


@torch.no_grad()
def encode_text_slot(text: str) -> Optional[torch.Tensor]:
    """Encode free text into the CEAH text slot (frozen raw SigLIP2 text space).
    Returns [768] L2-normalized, or None when text is empty."""
    if not text or not text.strip():
        return None
    proc = Shared.raw_siglip_proc
    inp = proc(text=[text.strip()], return_tensors="pt",
               padding="max_length", truncation=True, max_length=64)
    inp = {k: v.to(DEVICE) for k, v in inp.items()}
    with torch.cuda.amp.autocast(enabled=DEVICE.startswith("cuda")):
        f = get_text_features(Shared.raw_siglip, inp["input_ids"], inp.get("attention_mask"))
    return F.normalize(f.float(), dim=-1)[0]


@torch.no_grad()
def classify_against_anchors(reps: torch.Tensor, top_k: int = 5) -> List[Dict]:
    """reps [N,768] (semantic z or lesion features) → per-region top-K symptom category."""
    if reps.numel() == 0:
        return []
    A = Shared.anchor_embs[torch.tensor(Shared.lesion_cats, device=DEVICE)]   # [L,768]
    sims = F.normalize(reps, dim=-1) @ A.t()                                  # [N,L]
    out: List[Dict] = []
    for i in range(sims.size(0)):
        sc, order = sims[i].sort(descending=True)
        ranked = [(Shared.lesion_cats[order[j].item()], float(sc[j].item()))
                  for j in range(order.numel())]
        top1c, top1s = ranked[0]
        out.append({
            "label_id": top1c, "label_zh": Shared.cat_names.get(top1c, str(top1c)),
            "score": top1s,
            "top_k": [{"label_zh": Shared.cat_names.get(c, str(c)), "score": s}
                      for c, s in ranked[:top_k]],
        })
    return out


# ---------------------------------------------------------------------------
# Geometry / crop helper
# ---------------------------------------------------------------------------

def scaled_rect_crop(img: Image.Image, bbox_xywh, scale: float = 1.0) -> Image.Image:
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2.0, y + h / 2.0
    nw, nh = w * scale, h * scale
    W, H = img.size
    x1 = max(0, min(W - 1, int(round(cx - nw / 2.0))))
    y1 = max(0, min(H - 1, int(round(cy - nh / 2.0))))
    x2 = max(1, min(W, int(round(cx + nw / 2.0))))
    y2 = max(1, min(H, int(round(cy + nh / 2.0))))
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    return img.crop((x1, y1, x2, y2))


# ===========================================================================
# Pipelines — each exposes .params (dict) and .infer_rich(image, text_emb, top_k, top_n)
# ===========================================================================

def _build_memb(cases, device):
    idx_lists = [c["cause_emb_indices"] for c in cases]
    max_c = max(len(x) for x in idx_lists)
    memb = torch.full((len(cases), max_c), -1, dtype=torch.long)
    mlen = torch.zeros(len(cases), dtype=torch.long)
    for i, xs in enumerate(idx_lists):
        memb[i, :len(xs)] = torch.tensor(xs, dtype=torch.long)
        mlen[i] = len(xs)
    return memb.to(device), mlen.to(device)


def _candidate_pool(zq, bank_z, memb, mlen, top_k_cases, device):
    s = zq @ bank_z.t()                                  # [1, Nt]
    k = min(top_k_cases, s.size(1))
    topw, topi = s[0].topk(k)
    rows, rlen = memb[topi], mlen[topi]
    cmask = torch.arange(rows.size(1), device=device)[None] < rlen[:, None]
    cand = torch.unique(rows[cmask]); cand = cand[cand >= 0]
    return cand, topi, topw


def _retrieved_cards(topi, topw, file_names, n=5):
    out = []
    for k in range(min(topi.numel(), n)):
        ci = int(topi[k].item()); fn = file_names[ci]
        p = DET_TRAIN / fn
        out.append({"file_name": fn, "similarity": float(topw[k].item()),
                    "image_path": str(p) if p.exists() else None})
    return out


def _load_fold_thresh():
    """Cause-folding cut, calibrated vs LLM taxonomy (grod/calibrate_fold_threshold.py)."""
    try:
        return float(json.load(open(THRESHOLDS_JSON))["fold_thresh"])
    except Exception:
        return 0.75


FOLD_THRESH = _load_fold_thresh()   # agglomerative cut (pool-centered cosine distance)


def _fold_causes(cand_embs, scores_np, top_n, fold_thresh=FOLD_THRESH):
    """Collapse near-duplicate free-text causes after CEAH ranking.

    Pool-center the candidate cause embeddings (SigLIP2 text space is anisotropic),
    agglomerative-cluster with average linkage on cosine distance, cut at fold_thresh.
    Each group's representative = its highest-CEAH-score member, so the global top-1
    cause is never demoted — folding only removes its paraphrases and surfaces the
    next distinct cause family. Returns top_n (rep_local_idx, member_local_idxs)
    ordered by representative score desc.
    """
    P = cand_embs.size(0)
    if P <= 1:
        labels = np.zeros(P, dtype=int)
    else:
        c = F.normalize(cand_embs - cand_embs.mean(0, keepdim=True), dim=-1).cpu().numpy()
        Z = linkage(c, method="average", metric="cosine")
        labels = fcluster(Z, t=fold_thresh, criterion="distance")
    groups = {}
    for li, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(li)
    reps = []
    for members in groups.values():
        members = sorted(members, key=lambda li: -scores_np[li])
        reps.append((scores_np[members[0]], members[0], members))
    reps.sort(key=lambda x: -x[0])
    return [(rep, mem) for _, rep, mem in reps[:top_n]]


def _case_cause_sets(topi, memb, mlen):
    """Per retrieved-case global cause-idx set (for case-level support count)."""
    if topi is None or memb is None or mlen is None:
        return None
    rows, rlen = memb[topi], mlen[topi]
    return [set(rows[j, :int(rlen[j])].tolist()) for j in range(topi.numel())]


def _ceah_score(ceah, g, z_lesions, cand, cause_embs, text_emb, device,
                lesion_weights=None):
    """Model stage (GPU): run CEAH over the candidate pool. Returns scored tensors
    for the non-model fold stage to consume. Timed as ⑤."""
    P = int(cand.numel())
    N = z_lesions.size(0)
    cand_embs = cause_embs[cand]
    g_e = g.unsqueeze(0).expand(P, -1).contiguous()
    l_e = z_lesions.unsqueeze(0).expand(P, -1, -1).contiguous()
    l_m = torch.ones(P, N, dtype=torch.bool, device=device)
    if text_emb is None:
        t_e = torch.zeros(P, g.size(-1), device=device)
        t_p = torch.zeros(P, dtype=torch.bool, device=device)
    else:
        t_e = text_emb.unsqueeze(0).expand(P, -1).contiguous()
        t_p = torch.ones(P, dtype=torch.bool, device=device)
    kw = {} if lesion_weights is None else {"lesion_weights": lesion_weights}
    s_ceah, alphas, _ = ceah(g_e, t_e, t_p, l_e, l_m, cand_embs, **kw)
    return {"cand": cand, "cand_embs": cand_embs, "s_ceah": s_ceah,
            "sc": _minmax(s_ceah), "alphas": alphas, "P": P}


def _fold_topn(scored, cause_texts, top_n, topi=None, memb=None, mlen=None):
    """Non-model stage (CPU): fold near-duplicate causes (agglomerative) + case-level
    support, build top_n output. No CEAH forward here. Timed as ⑥."""
    cand_embs = scored["cand_embs"]; sc = scored["sc"]; alphas = scored["alphas"]
    s_np = scored["s_ceah"].detach().cpu().numpy()
    cand_np = scored["cand"].cpu().numpy()
    groups = _fold_causes(cand_embs, s_np, top_n)
    case_sets = _case_cause_sets(topi, memb, mlen)
    out = []
    for rank, (rep_li, member_lis) in enumerate(groups, 1):
        gi = int(cand_np[rep_li])
        gids = {int(cand_np[li]) for li in member_lis}
        support = sum(1 for cs in case_sets if cs & gids) if case_sets is not None else None
        out.append({"rank": rank, "cause_idx": gi, "text": cause_texts[gi],
                    "score": float(sc[rep_li].item()),
                    "alpha": [float(a) for a in alphas[rep_li].cpu().tolist()],
                    "support": support,
                    "members": [cause_texts[int(cand_np[li])] for li in member_lis]})
    return out, scored["P"]


def _minmax(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


class GrodSoftPipeline:
    label = "grod_soft"
    hard_gate = False   # grod_soft: continuous w. Subclass GrodPipeline sets True
                        # to feed {0,1} weights (objectness > display_thresh) — the
                        # bytes-exact hard-gate degenerate of this same soft model.

    def __init__(self):
        from diagnosis_model.grod.gpu_infer_soft import GpuPipelineSoft
        self.p = GpuPipelineSoft(str(JOINT_CKPT), str(GLOBAL_SD), str(ANCHORS_PT),
                                 str(SOFT_ENC), str(SOFT_CEAH), str(SOFT_CASE_DB),
                                 str(SOFT_BANK), device=DEVICE)
        cases = torch.load(Path(SOFT_CASE_DB) / "train_cases.pt", weights_only=False)
        self.file_names = [c["file_name"] for c in cases]
        sfx = "hard gate" if self.hard_gate else "soft"
        self.params = {**_detector_params(self.p.net),
                       f"Aggregator (DeepSets, {sfx})": _nparams(self.p.enc),
                       f"CEAH ({sfx})": _nparams(self.p.ceah)}

    @torch.no_grad()
    def infer_rich(self, image, text_emb, top_k_cases, top_n,
                   abstain_thresh=ABSTAIN_DEFAULT, display_thresh=DISPLAY_DEFAULT):
        p = self.p; dev = DEVICE; W, H = image.size; K = p.top_k_lesions
        T = []
        t = _sync_now()
        px = TF.normalize(TF.resize(TF.to_tensor(image), [p.res, p.res]),
                          p.means, p.stds).unsqueeze(0).to(dev)
        out = p.net(px)
        logits = out["pred_logits"][0][:, 0]; z_all = out["pred_semantic"][0]
        g = out["pred_global"][0]; boxes = out["pred_boxes"][0]
        w = logits.sigmoid()
        scores, lidx = w.topk(min(K, w.numel()))
        T.append(("① backbone+DETR forward (4 head)", (_sync_now() - t) * 1000))

        # Two thresholds: abstain (健/病) + display (顯示選框). Aggregation always uses
        # all-Q (soft contract); CEAH always uses top-K; the thresholds only govern the
        # DISPLAY mask + abstain. Abstain = 健 iff max objectness < abstain_thresh.
        t = _sync_now()
        abstain = float(w.amax()) < abstain_thresh
        keep_mask = scores > display_thresh
        T.append(("② 健/病 + 病灶門檻 (abstain/display)", (_sync_now() - t) * 1000))
        if abstain or int(keep_mask.sum().item()) == 0:
            r = _empty_result(image, T); r["abstain"] = True
            r["obj_all"] = w; r["boxes_all"] = boxes; return r
        Kk = lidx.numel()
        zk = z_all[lidx]                                   # [Kk,768] — full top-K for CEAH
        M = max(int(keep_mask.sum().item()), 1)
        bn = boxes[lidx[:M]]
        cx, cy, bw, bh = bn.unbind(-1)
        bxywh = [[float((cx[i] - bw[i] / 2) * W), float((cy[i] - bh[i] / 2) * H),
                  float(bw[i] * W), float(bh[i] * H)] for i in range(M)]
        crops = [scaled_rect_crop(image, b) for b in bxywh]

        t = _sync_now()
        cls = classify_against_anchors(zk[:M].float())
        T.append(("③ 病灶分類 (z·anchor)", (_sync_now() - t) * 1000))

        t = _sync_now()
        # hard_gate (grod mode): binarize objectness at display_thresh → {0,1}, the
        # bytes-exact hard-gate degenerate. grod_soft keeps the continuous w.
        w_model = (w > display_thresh).float() if self.hard_gate else w.float()
        zq = p.enc(g.float().unsqueeze(0), z_all.float().unsqueeze(0),
                   torch.tensor([w.numel()], device=dev), lesion_weights=w_model.unsqueeze(0))
        cand, topi, topw = _candidate_pool(zq, p.bank_z, p.memb, p.mlen, top_k_cases, dev)
        T.append(("④ 聚合 + dense 檢索", (_sync_now() - t) * 1000))

        t = _sync_now()
        lw = ((scores > display_thresh).float() if self.hard_gate else scores.float()).unsqueeze(0)
        scored = _ceah_score(p.ceah, g.float(), zk.float(), cand, p.cause_embs,
                             text_emb, dev,
                             lesion_weights=lw.expand(int(cand.numel()), -1).contiguous())
        T.append(("⑤ CEAH 評分 + α", (_sync_now() - t) * 1000))

        t = _sync_now()
        top_n_out, P = _fold_topn(scored, p.cause_texts, top_n,
                                  topi=topi, memb=p.memb, mlen=p.mlen)
        T.append(("⑥ 病因聚合 (CPU)", (_sync_now() - t) * 1000))

        lesions = [{"idx": i, "bbox_xywh": bxywh[i], "det_score": float(scores[i]),
                    "crop": crops[i], "cls": cls[i]} for i in range(M)]
        return {"image_pil": image, "lesions": lesions, "n_lesions": M,
                "retrieved": _retrieved_cards(topi, topw, self.file_names),
                "top_n": top_n_out, "pool_size": P, "abstain": False,
                "obj_all": w, "boxes_all": boxes,
                "text_used": text_emb is not None, "timings": T}


class GrodPipeline(GrodSoftPipeline):
    """grod hard-gate mode: identical soft model/artifacts as grod_soft, but the
    per-query weights are binarized to {0,1} at display_thresh. The DeepSets pooling
    and CEAH attribution reduce bytes-exactly to the hard path when weights ∈ {0,1},
    so this is a pure input-selection variant — no separate hard encoder/CEAH."""
    label = "grod"
    hard_gate = True


class BasePipeline:
    """Conventional separated baseline (new tree). RF-DETR detect → raw SigLIP2 global
    + finetuned SigLIP2 lesion crops → DeepSets → dense retrieval → CEAH."""
    label = "base"

    def __init__(self):
        # base uses the PLAIN detector (no GROD semantic/global heads); clear any
        # env vars a previously-loaded grod pipeline may have set, so this RF-DETR
        # is built without those heads regardless of mode load order.
        for k in ("RFDETR_SEMANTIC_DIM", "RFDETR_SEMANTIC_ANCHORS", "RFDETR_GLOBAL_DIM"):
            os.environ.pop(k, None)
        from rfdetr import RFDETRMedium
        self.det = RFDETRMedium(pretrain_weights=str(BASE_DETECTOR), num_classes=1)
        self.det.optimize_for_inference(compile=False)
        from transformers import AutoModel
        self.lesion = AutoModel.from_pretrained(str(BASE_VLM_LESION)).to(DEVICE).eval()
        # aggregator + bank
        from diagnosis_model.cause_inference.models.case_encoder import EncoderConfig, build_encoder
        from diagnosis_model.cause_inference.train_case_encoder import encode_all
        pkg = torch.load(BASE_ENC, weights_only=False, map_location="cpu")
        cfg = pkg["encoder_config"]; cfg["dtype"] = torch.float32
        self.enc = build_encoder(EncoderConfig(**cfg)).to(DEVICE).eval()
        self.enc.load_state_dict(pkg["encoder_state"])
        cases = torch.load(Path(BASE_CASE_DB) / "train_cases.pt", weights_only=False)
        self.file_names = [c["file_name"] for c in cases]
        self.bank_z = encode_all(self.enc, cases, torch.device(DEVICE)).to(DEVICE).float()
        self.enc.eval()
        cte = torch.load(Path(BASE_CASE_DB) / "cause_text_embs.pt", weights_only=False)
        self.cause_embs = F.normalize(cte["embeddings"].float().to(DEVICE), dim=-1)
        self.cause_texts = cte["texts"]
        self.memb, self.mlen = _build_memb(cases, DEVICE)
        self.in_dim = self.cause_embs.size(-1)
        from diagnosis_model.cause_inference.models import CEAH
        self.ceah = CEAH(global_dim=self.in_dim, text_dim=self.in_dim,
                         lesion_dim=self.in_dim, cause_dim=self.in_dim,
                         common_dim=256, hidden_dim=512, dropout=0.1,
                         attribution_mode="softmax", scoring_mode="multiplicative").to(DEVICE).eval()
        self.ceah.load_state_dict(torch.load(BASE_CEAH, map_location=DEVICE))
        self.params = {**_detector_params(self.det.model.model),
                       "raw SigLIP2 (global, frozen)": _nparams(Shared.raw_siglip),
                       "微調 SigLIP2 (lesion)": _nparams(self.lesion),
                       "Aggregator (DeepSets)": _nparams(self.enc),
                       "CEAH": _nparams(self.ceah)}

    @torch.no_grad()
    def _enc_global(self, image):
        px = Shared.raw_siglip_proc(images=[image], return_tensors="pt")["pixel_values"].to(DEVICE)
        with torch.cuda.amp.autocast(enabled=DEVICE.startswith("cuda")):
            f = get_image_features(Shared.raw_siglip, px)
        return F.normalize(f.float(), dim=-1)[0]

    @torch.no_grad()
    def _enc_lesions(self, crops):
        if not crops:
            return torch.empty(0, self.in_dim, device=DEVICE)
        px = Shared.raw_siglip_proc(images=crops, return_tensors="pt")["pixel_values"].to(DEVICE)
        with torch.cuda.amp.autocast(enabled=DEVICE.startswith("cuda")):
            f = get_image_features(self.lesion, px)
        return F.normalize(f.float(), dim=-1)

    @torch.no_grad()
    def infer_rich(self, image, text_emb, top_k_cases, top_n,
                   abstain_thresh=ABSTAIN_DEFAULT, display_thresh=DISPLAY_DEFAULT):
        dev = DEVICE; T = []; W, H = image.size
        t = _sync_now()
        # detect at the (lower) display threshold so both the heatmap and the box
        # display see all candidates; abstain on the max confidence.
        pred = self.det.predict([image], threshold=display_thresh)
        det = pred[0] if isinstance(pred, list) else pred
        dets = sorted([([float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], float(s))
                       for b, s in zip(det.xyxy, det.confidence)], key=lambda x: -x[1])
        T.append(("① 偵測 (RF-DETR)", (_sync_now() - t) * 1000))
        # obj_all / boxes_all (cxcywh-norm) for the heatmap display
        obj_all = torch.tensor([d[1] for d in dets], device=dev) if dets else torch.zeros(0, device=dev)
        boxes_all = torch.tensor([[(d[0][0] + d[0][2] / 2) / W, (d[0][1] + d[0][3] / 2) / H,
                                   d[0][2] / W, d[0][3] / H] for d in dets],
                                 device=dev) if dets else torch.zeros(0, 4, device=dev)
        if not dets or max(d[1] for d in dets) < abstain_thresh:   # 沒框/最高分未達 → 健康
            r = _empty_result(image, T); r["abstain"] = True
            r["obj_all"] = obj_all; r["boxes_all"] = boxes_all; return r
        bxywh = [d[0] for d in dets]; scores = [d[1] for d in dets]
        crops = [scaled_rect_crop(image, b) for b in bxywh]

        t = _sync_now()
        g = self._enc_global(image); L = self._enc_lesions(crops)
        T.append(("② SigLIP2 編碼 (global+lesion)", (_sync_now() - t) * 1000))
        N = L.size(0)

        t = _sync_now()
        cls = classify_against_anchors(L)
        T.append(("③ 病灶分類 (lesion·anchor)", (_sync_now() - t) * 1000))

        t = _sync_now()
        boxes = torch.tensor(bxywh, dtype=torch.float32)
        Ls = L[torch.argsort(boxes[:, 2] * boxes[:, 3], descending=True)]
        lp = torch.zeros(1, max(N, 1), self.in_dim, device=dev); lp[0, :N] = Ls
        zq = self.enc(g.unsqueeze(0), lp, torch.tensor([N], device=dev)).float()
        cand, topi, topw = _candidate_pool(zq, self.bank_z, self.memb, self.mlen, top_k_cases, dev)
        T.append(("④ 聚合 + dense 檢索", (_sync_now() - t) * 1000))

        t = _sync_now()
        scored = _ceah_score(self.ceah, g, L, cand, self.cause_embs, text_emb, dev)
        T.append(("⑤ CEAH 評分 + α", (_sync_now() - t) * 1000))

        t = _sync_now()
        top_n_out, P = _fold_topn(scored, self.cause_texts, top_n,
                                  topi=topi, memb=self.memb, mlen=self.mlen)
        T.append(("⑥ 病因聚合 (CPU)", (_sync_now() - t) * 1000))

        lesions = [{"idx": i, "bbox_xywh": bxywh[i], "det_score": scores[i],
                    "crop": crops[i], "cls": cls[i]} for i in range(N)]
        return {"image_pil": image, "lesions": lesions, "n_lesions": N,
                "retrieved": _retrieved_cards(topi, topw, self.file_names),
                "top_n": top_n_out, "pool_size": P, "abstain": False,
                "obj_all": obj_all, "boxes_all": boxes_all,
                "text_used": text_emb is not None, "timings": T}


def _empty_result(image, T):
    return {"image_pil": image, "lesions": [], "n_lesions": 0, "retrieved": [],
            "top_n": [], "pool_size": 0, "abstain": False, "text_used": False, "timings": T}


# ---------------------------------------------------------------------------
# Lazy pipeline registry
# ---------------------------------------------------------------------------
_PIPELINES: Dict[str, object] = {}
_BUILDERS = {"base": BasePipeline, "grod": GrodPipeline, "grod_soft": GrodSoftPipeline}


def get_pipeline(mode: str):
    if mode not in _PIPELINES:
        print(f"[load] building pipeline '{mode}' ...")
        _PIPELINES[mode] = _BUILDERS[mode]()
        print(f"[load] '{mode}' ready")
    return _PIPELINES[mode]


# ===========================================================================
# Visualization
# ===========================================================================

def _put_label(draw, x, y, text, bg=(40, 40, 40), size=16):
    try:
        font = ImageFont.truetype(CJK_FONT_PATH, size)
    except OSError:
        font = ImageFont.load_default()
    b = draw.textbbox((x, y), text, font=font); pad = 3
    draw.rectangle((b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad), fill=bg)
    draw.text((x, y), text, font=font, fill=(255, 255, 255))


def render_heatmap_image(image, obj_all, boxes_all):
    """Step-1 display = A objectness heatmap (all queries splatted, absolute scale).
    Pure heatmap, no boxes (boxes appear in the ② classification cards). obj_all:[Q],
    boxes_all:[Q,4] cxcywh-normalized. Reuses render_anomaly_heatmap helpers."""
    from diagnosis_model.grod.render_anomaly_heatmap import splat_heatmap, overlay
    if obj_all is None or obj_all.numel() == 0:
        return image
    heat = splat_heatmap(obj_all, boxes_all, grid=200, agg="sum", normalize="absolute")
    return overlay(image, heat, gamma=0.7, max_alpha=0.6)


def render_detection_image(image, lesions):
    img = image.copy().convert("RGB"); draw = ImageDraw.Draw(img)
    for li, les in enumerate(lesions):
        x, y, w, h = [int(v) for v in les["bbox_xywh"]]
        for off in range(3):
            draw.rectangle((x - off, y - off, x + w + off, y + h + off), outline=(220, 50, 50))
        _put_label(draw, x + 2, max(2, y - 22),
                   f"L{li}: {les['cls']['label_zh']} ({les['det_score']:.2f})", bg=(180, 30, 30))
    return img


def make_lesion_card(les):
    fig = plt.figure(figsize=(9, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.0], wspace=0.06)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(les["crop"]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"L{les['idx']}  {les['cls']['label_zh']}  "
                 f"(obj={les['det_score']:.2f}, cls={les['cls']['score']:.2f})",
                 fontproperties=_font(12), pad=6)
    ax2 = fig.add_subplot(gs[0, 1]); ax2.axis("off")
    ax2.text(0.0, 0.98, "Top-K 症狀類別 (rep · anchor)", fontproperties=_font(11),
             transform=ax2.transAxes, fontweight="bold")
    yp = 0.86
    for k, it in enumerate(les["cls"]["top_k"]):
        ax2.text(0.0, yp, f"{k+1}. {it['label_zh']:<10}  {it['score']:.3f}",
                 fontproperties=_font(10), transform=ax2.transAxes, family="monospace")
        yp -= 0.10
    fig.tight_layout(); fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
    return Image.fromarray(out)


def make_missing_case_placeholder(rank, sim):
    img = Image.new("RGB", (320, 220), (245, 245, 245)); d = ImageDraw.Draw(img)
    try:
        tf = ImageFont.truetype(CJK_FONT_PATH, 20); bf = ImageFont.truetype(CJK_FONT_PATH, 16)
    except OSError:
        tf = bf = ImageFont.load_default()
    d.rectangle((0, 0, 319, 219), outline=(180, 180, 180), width=2)
    d.text((20, 30), f"Train case #{rank}", fill=(50, 50, 50), font=tf)
    d.text((20, 70), "圖片不存在", fill=(150, 45, 45), font=bf)
    d.text((20, 150), f"similarity = {sim:.3f}", fill=(70, 70, 70), font=bf)
    return img


def _alpha_rgba(a, base, a_max):
    inten = float(np.clip(a / max(a_max, 1e-6), 0.0, 1.0))
    return (*base, 0.3 + 0.7 * inten)


def make_alpha_attribution_image(image, lesion_boxes, alpha, n_lesions, cause_text,
                                 score, show_text):
    g_a = float(alpha[0]); t_a = float(alpha[1])
    les_a = [float(a) for a in alpha[2: 2 + n_lesions]]
    a_max = max([g_a] + ([t_a] if show_text else []) + les_a + [1e-6])
    fig = plt.figure(figsize=(11, 6.5)); ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"α attribution  (cause score={score:.3f})", fontproperties=_font(13), pad=8)
    for li, (bbox, a_val) in enumerate(zip(lesion_boxes, les_a)):
        x, y, w, h = [int(v) for v in bbox]
        inten = float(np.clip(a_val / a_max, 0.0, 1.0))
        edge = _alpha_rgba(a_val, (1.0, 0.15, 0.15), a_max)
        ax.add_patch(mpatches.Rectangle((x, y), w, h, linewidth=1.0 + 4.5 * inten,
                                        edgecolor=edge, facecolor="none"))
        ax.text(x + 3, max(y - 6, 12), f"L{li}\nα={a_val:.2f}", fontproperties=_font(11),
                color="white", bbox=dict(boxstyle="round,pad=0.25", facecolor=edge[:3],
                                         alpha=0.9, edgecolor="none"))
    gc = _alpha_rgba(g_a, (0.15, 0.4, 1.0), a_max)
    ax.text(8, 22, f"GLOBAL  α={g_a:.2f}", fontproperties=_font(11), color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=gc[:3], alpha=0.95, edgecolor="none"))
    if show_text:
        tc = _alpha_rgba(t_a, (1.0, 0.55, 0.0), a_max)
        ax.text(8, 50, f"TEXT  α={t_a:.2f}", fontproperties=_font(11), color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=tc[:3], alpha=0.95, edgecolor="none"))
    fig.text(0.5, 0.02, cause_text, ha="center", va="bottom", fontproperties=_font(12), wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
    return Image.fromarray(out)


def make_alpha_breakdown_chart(alpha, n_lesions, show_text):
    les_vals = [float(a) for a in alpha[2: 2 + n_lesions]]
    if show_text:
        labels = ["global", "text"] + [f"L{i}" for i in range(n_lesions)]
        vals = [float(alpha[0]), float(alpha[1])] + les_vals
        colors = ["#2660ff", "#ff8c00"] + ["#dc1e1e"] * n_lesions
    else:
        labels = ["global"] + [f"L{i}" for i in range(n_lesions)]
        vals = [float(alpha[0])] + les_vals
        colors = ["#2660ff"] + ["#dc1e1e"] * n_lesions
    fig, ax = plt.subplots(figsize=(7.5, 3.3)); bars = ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, max(0.6, max(vals) * 1.2)); ax.set_ylabel("α", fontproperties=_font(11))
    ax.set_title("Per-evidence α (softmax, sums to 1)", fontproperties=_font(11))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center",
                va="bottom", fontproperties=_font(10))
    ax.grid(axis="y", linestyle="--", alpha=0.3); fig.tight_layout(); fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
    return Image.fromarray(out)


# ---------------------------------------------------------------------------
# Detail tables (params + timing)
# ---------------------------------------------------------------------------

def _params_md(pipe) -> str:
    rows = ["", "**模組參數量**", "", "| 模組 | 參數量 |", "|---|---:|"]
    total = 0
    for name, n in pipe.params.items():
        if not name.startswith("  "):
            total += n
        rows.append(f"| {name} | {n:,} |")
    rows.append(f"| **總計 (不含子項重複)** | **{total:,}** |")
    return "\n".join(rows)


def _timing_md(stage_stats, n_runs) -> str:
    rows = ["", f"**各階段延遲 (CUDA-synced, {n_runs} 次平均, 丟首次 warm-up)**", "",
            "| 階段 | mean ms | ±std |", "|---|---:|---:|"]
    tot_mean = 0.0
    for label, mean, std in stage_stats:
        tot_mean += mean
        rows.append(f"| {label} | {mean:.2f} | {std:.2f} |")
    rows.append(f"| **總計** | **{tot_mean:.2f}** | — |")
    if tot_mean > 0:
        rows.append(f"\n吞吐 ≈ **{1000.0 / tot_mean:.1f} img/s**（單張、batch=1）")
    return "\n".join(rows)


def _aggregate_timings(runs: List[List[Tuple[str, float]]]):
    """runs: list of [(label, ms), ...] (same labels/order). Returns [(label, mean, std)]."""
    if not runs:
        return []
    labels = [l for l, _ in runs[0]]
    out = []
    for i, lab in enumerate(labels):
        vals = np.array([r[i][1] for r in runs], dtype=np.float64)
        out.append((lab, float(vals.mean()), float(vals.std())))
    return out


# ===========================================================================
# Gradio handlers
# ===========================================================================

def _empty_buttons():
    return [gr.update(visible=False) for _ in range(MAX_TOPN_BUTTONS)]


def handler_run(mode, image, text, abstain_thresh, display_thresh, top_k_cases,
                top_n_causes):
    if image is None:
        return (None, None, None, "請先上傳或選一張圖。", [], "", None, None, "",
                *_empty_buttons())
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    pipe = get_pipeline(mode)
    text_emb = encode_text_slot(text)
    top_k, top_n = int(top_k_cases), int(top_n_causes)
    ath, dth = float(abstain_thresh), float(display_thresh)

    res = pipe.infer_rich(image, text_emb, top_k, top_n, ath, dth)

    detailed = True
    timing_md = ""
    if detailed and res["n_lesions"] > 0:
        runs = [res["timings"]]
        for _ in range(N_TIMING_RUNS):
            runs.append(pipe.infer_rich(image, text_emb, top_k, top_n, ath, dth)["timings"])
        stats = _aggregate_timings(runs[1:])    # drop warm-up (first)
        timing_md = _timing_md(stats, N_TIMING_RUNS)

    det_img = render_heatmap_image(res["image_pil"], res.get("obj_all"), res.get("boxes_all"))
    gallery = [(make_lesion_card(l), f"L{l['idx']}: {l['cls']['label_zh']}")
               for l in res["lesions"]]

    if res.get("abstain"):
        info = f"🟢 **abstain：最高 objectness 未達 abstain_thresh={ath:.2f}，判定為健康**，不進行病因推論。"
    elif not res["lesions"]:
        info = "**display_thresh 下未顯示病灶。**"
    else:
        thr = f"abstain={ath:.2f} / display={dth:.2f}"
        info = (f"模式 **{mode}** ｜ 顯示 **{res['n_lesions']}** 個病灶（{thr}）｜ "
                f"候選病因池 = **{res['pool_size']}** ｜ top-{len(res['top_n'])} 病因"
                + ("（含文字證據）" if res["text_used"] else "（vision-only）"))
    if detailed:
        info += "\n\n" + _params_md(pipe)
        if timing_md:
            info += "\n\n" + timing_md

    btns = []
    for i in range(MAX_TOPN_BUTTONS):
        if i < len(res["top_n"]):
            r = res["top_n"][i]; txt = r["text"][:48] + "…" if len(r["text"]) > 50 else r["text"]
            sup = f"  {r['support']}例" if r.get("support") else ""
            btns.append(gr.update(value=f"#{r['rank']}  s={r['score']:.2f}{sup}  {txt}", visible=True))
        else:
            btns.append(gr.update(visible=False))

    retr_gallery = []
    for i, r in enumerate(res["retrieved"], 1):
        cap = f"#{i}  sim={r['similarity']:.3f}"
        retr_gallery.append((r["image_path"], cap) if r.get("image_path")
                            else (make_missing_case_placeholder(i, r["similarity"]), cap + " (missing)"))

    state = {"image_pil": res["image_pil"], "boxes": [l["bbox_xywh"] for l in res["lesions"]],
             "n_lesions": res["n_lesions"], "top_n": res["top_n"], "text_used": res["text_used"]}
    return (det_img, gallery, state, info, retr_gallery, "", None, None, "", *btns)


def handler_select(idx, state):
    if state is None or idx >= len(state.get("top_n", [])):
        return None, None, ""
    r = state["top_n"][idx]; n = state["n_lesions"]
    show_text = state.get("text_used", False)
    bbox = np.array(state["boxes"], dtype=np.float32)
    ai = make_alpha_attribution_image(state["image_pil"], bbox, r["alpha"], n,
                                      r["text"], r["score"], show_text)
    bar = make_alpha_breakdown_chart(r["alpha"], n, show_text)
    members = r.get("members") or []
    sup_md = f"- 支持度：**{r['support']}** 個相似病例指向此病因\n" if r.get("support") else ""
    fold_md = ""
    if len(members) > 1:
        extra = "\n".join(f"  - {m[:60]}" for m in members[1:6])
        more = f"\n  - …+{len(members) - 6}" if len(members) > 6 else ""
        fold_md = f"\n\n<details><summary>已聚合 {len(members) - 1} 條相近病因</summary>\n\n{extra}{more}\n</details>"
    explain = (f"### Top-{r['rank']} 病因\n**{r['text']}**\n\n"
               + sup_md
               + f"- CEAH score: **{r['score']:.3f}**\n"
               + fold_md)
    return ai, bar, explain


# ===========================================================================
# UI
# ===========================================================================
DESCRIPTION = """
# 🐟 GROD 魚病診斷流水線 demo

三個流水線，UI 下拉即時切換：

| 模式 | 架構 |
|---|---|
| **base** | 常規分離式對照組：偵測 (RF-DETR) + 凍結 SigLIP2 全域 + 微調 SigLIP2 病灶 → Aggregator (DeepSets) → dense 檢索 → CEAH |
| **grod_soft** | Backbone(DINOv2)+DETR forward → 四 head（box / objectness / semantic / global）→ Aggregator (DeepSets, per-query 連續權重 w) → dense 檢索 → CEAH（預設） |
| **grod** | 同 grod_soft 同一顆模型/artifacts，僅把 per-query 權重在 display_thresh 二值化成 {0,1}（硬閘）。DeepSets/CEAH 在 {0,1} 權重下 bytes-exactly 退化成硬路徑 → grod vs grod_soft ＝同一模型上的輸入選取消融 |
"""


def build_ui():
    with gr.Blocks(title="GROD Fish Disease Demo") as demo:
        gr.Markdown(DESCRIPTION)
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Dropdown(["base", "grod", "grod_soft"], value="grod_soft", label="模式")
                inp_image = gr.Image(label="魚體輸入圖", type="pil", height=300)
                inp_text = gr.Textbox(label="選填：文字描述", lines=2,
                                      placeholder="例：體表潰瘍、紅腫，疑似感染；留空＝vision-only")
                with gr.Accordion("可調參數", open=False):
                    sld_abstain = gr.Slider(0.1, 0.9, value=ABSTAIN_DEFAULT, step=0.01,
                                            label="abstain_thresh")
                    sld_display = gr.Slider(0.1, 0.9, value=DISPLAY_DEFAULT, step=0.01,
                                            label="display_thresh")
                    sld_topk = gr.Slider(5, 50, value=20, step=1, label="top_k_cases")
                    sld_topn = gr.Slider(1, MAX_TOPN_BUTTONS, value=5, step=1, label="top_n_causes")
                btn_run = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                out_det = gr.Image(label="① 異常熱力圖", type="pil", height=300)
                out_info = gr.Markdown()

        gr.Markdown("---\n## ② 病灶分類（rep · symptom anchor）")
        out_gallery = gr.Gallery(columns=1, height=520, show_label=False, object_fit="contain")
        gr.Markdown("---\n## ③ 檢索到的相似 case")
        out_retr = gr.Gallery(columns=5, height=220, show_label=True, object_fit="contain")
        gr.Markdown("---\n## ④ Top-N 病因 + α 歸因（點按鈕看解釋）")
        buttons = []
        with gr.Row():
            with gr.Column(scale=1):
                for _ in range(MAX_TOPN_BUTTONS):
                    buttons.append(gr.Button(value="", visible=False, size="sm"))
                out_retr_md = gr.Markdown()
            with gr.Column(scale=2):
                out_explain = gr.Markdown()
                out_alpha = gr.Image(label="α attribution", type="pil", height=400)
                out_bar = gr.Image(label="α breakdown", type="pil", height=230)

        run_outputs = [out_det, out_gallery, state, out_info, out_retr,
                       out_retr_md, out_alpha, out_bar, out_explain, *buttons]
        btn_run.click(handler_run,
                      [mode, inp_image, inp_text, sld_abstain, sld_display, sld_topk,
                       sld_topn],
                      run_outputs)
        for i, b in enumerate(buttons):
            b.click(lambda st, idx=i: handler_select(idx, st), [state],
                    [out_alpha, out_bar, out_explain])

        ex = _diseased_examples(8)
        if ex:
            gr.Examples(examples=ex, inputs=[inp_image], examples_per_page=8,
                        label="範例（valid，含病灶的魚）")
    return demo


def _diseased_examples(n=8) -> List[List[str]]:
    """Pick valid images that actually have lesion boxes (most-annotated first),
    so the examples exercise the full pipeline rather than abstaining on healthy fish."""
    coco_path = DET_VALID / "_annotations.coco.json"
    if not coco_path.exists():
        return []
    coco = json.load(open(coco_path, encoding="utf-8"))
    id2fn = {im["id"]: im["file_name"] for im in coco["images"]}
    from collections import Counter
    cnt = Counter(a["image_id"] for a in coco["annotations"])
    out = []
    for iid, _ in cnt.most_common():
        fn = id2fn.get(iid)
        if fn and (DET_VALID / fn).exists():
            out.append([str(DET_VALID / fn)])
        if len(out) >= n:
            break
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preload", nargs="*", default=[],
                    help="modes to load at startup (default: lazy on first use)")
    args = ap.parse_args()
    print(f"[init] device={DEVICE}")
    load_shared()
    for m in args.preload:
        get_pipeline(m)
    build_ui().queue(default_concurrency_limit=1).launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
