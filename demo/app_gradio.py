"""Gradio demo — three GROD-family pipelines on the new dataset tree.

Modes (UI dropdown, lazy-loaded, live-switchable):
  base       conventional separated baseline: RF-DETR + raw SigLIP2 global +
             standard-finetuned SigLIP2 lesion crops → DeepSets → dense retrieval → CEAH
  grod       diagnosis_model/grod/gpu_infer.py        (single 4-head RF-DETR forward)
  grod_soft  diagnosis_model/grod/gpu_infer_soft.py   (soft per-query weights)

Display:
  detailed mode only:
    detection · per-lesion classification cards · retrieved cases · top-N causes + α
    + per-module parameter counts
    + per-stage latency averaged over N CUDA-synced runs (warm-up dropped)

Lesion threshold:
  grod / grod_soft use a fixed objectness threshold (default 0.5 = τ* from
  compute_lesion_threshold.py); base uses its RF-DETR detector threshold.
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

# grod
GROD_ENC = ART / "models/encoder_grod/best_encoder.pt"
GROD_CEAH = ART / "models/ceah_jointDistRawP/best_ceah.pt"
GROD_CASE_DB = ART / "db/case_db_jointDistRawP"
# grod_soft
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
        "backbone (DINOv2 ViT)": g("backbone"),
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


def _ceah_topn(ceah, g, z_lesions, cand, cause_embs, cause_texts, text_emb,
               top_n, device, lesion_weights=None):
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
    sc = _minmax(s_ceah)
    order = torch.argsort(s_ceah, descending=True).cpu().numpy()
    cand_np = cand.cpu().numpy()
    out = []
    for r in range(min(top_n, len(order))):
        li = int(order[r]); gi = int(cand_np[li])
        out.append({"rank": r + 1, "cause_idx": gi, "text": cause_texts[gi],
                    "score": float(sc[li].item()),
                    "alpha": [float(a) for a in alphas[li].cpu().tolist()]})
    return out, P


def _minmax(x: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return torch.zeros_like(x)
    return (x - lo) / (hi - lo)


class GrodPipeline:
    """Wraps gpu_infer.GpuPipeline; adds viz + text-slot + per-stage timing."""
    label = "grod"

    def __init__(self):
        from diagnosis_model.grod.gpu_infer import GpuPipeline
        self.p = GpuPipeline(str(JOINT_CKPT), str(GLOBAL_SD), str(ANCHORS_PT),
                             str(GROD_ENC), str(GROD_CEAH), str(GROD_CASE_DB), device=DEVICE)
        cases = torch.load(Path(GROD_CASE_DB) / "train_cases.pt", weights_only=False)
        self.file_names = [c["file_name"] for c in cases]
        self.params = {**_detector_params(self.p.net),
                       "Aggregator (DeepSets)": _nparams(self.p.enc),
                       "CEAH": _nparams(self.p.ceah)}

    @torch.no_grad()
    def infer_rich(self, image, text_emb, top_k_cases, top_n, det_thresh=0.5):
        p = self.p; dev = DEVICE; W, H = image.size
        T = []
        t = _sync_now()
        px = TF.normalize(TF.resize(TF.to_tensor(image), [p.res, p.res]),
                          p.means, p.stds).unsqueeze(0).to(dev)
        out = p.net(px)
        logits = out["pred_logits"][0][:, 0]; z_all = out["pred_semantic"][0]
        g = out["pred_global"][0]; boxes = out["pred_boxes"][0]
        obj = logits.sigmoid()
        # lesion gate: fixed threshold; no box ⟹ healthy (abstain).
        tau_g = None
        keep = obj > det_thresh
        abstain = int(keep.sum()) == 0
        T.append(("① backbone+DETR forward → 4 head + 門檻", (_sync_now() - t) * 1000))
        if abstain:
            r = _empty_result(image, T); r["abstain"] = True; r["tau_g"] = tau_g; return r
        z = z_all[keep]; ok = obj[keep]; bn = boxes[keep]
        order = torch.argsort(ok, descending=True)
        z, ok, bn = z[order], ok[order], bn[order]; N = z.size(0)
        cx, cy, bw, bh = bn.unbind(-1)
        bxywh = [[float((cx[i] - bw[i] / 2) * W), float((cy[i] - bh[i] / 2) * H),
                  float(bw[i] * W), float(bh[i] * H)] for i in range(N)]
        crops = [scaled_rect_crop(image, b) for b in bxywh]

        t = _sync_now()
        cls = classify_against_anchors(z.float())
        T.append(("② 病灶分類 (z·anchor)", (_sync_now() - t) * 1000))

        t = _sync_now()
        zq = p.enc(g.unsqueeze(0), z.unsqueeze(0), torch.tensor([N], device=dev))
        cand, topi, topw = _candidate_pool(zq, p.bank_z, p.memb, p.mlen, top_k_cases, dev)
        T.append(("③ 聚合 + dense 檢索", (_sync_now() - t) * 1000))

        t = _sync_now()
        top_n_out, P = _ceah_topn(p.ceah, g, z, cand, p.cause_embs, p.cause_texts,
                                  text_emb, top_n, dev)
        T.append(("④ CEAH 排序 + α", (_sync_now() - t) * 1000))

        lesions = [{"idx": i, "bbox_xywh": bxywh[i], "det_score": float(ok[i]),
                    "crop": crops[i], "cls": cls[i]} for i in range(N)]
        return {"image_pil": image, "lesions": lesions, "n_lesions": N,
                "retrieved": _retrieved_cards(topi, topw, self.file_names),
                "top_n": top_n_out, "pool_size": P, "abstain": False, "tau_g": tau_g,
                "text_used": text_emb is not None, "timings": T}


class GrodSoftPipeline:
    label = "grod_soft"

    def __init__(self):
        from diagnosis_model.grod.gpu_infer_soft import GpuPipelineSoft
        self.p = GpuPipelineSoft(str(JOINT_CKPT), str(GLOBAL_SD), str(ANCHORS_PT),
                                 str(SOFT_ENC), str(SOFT_CEAH), str(SOFT_CASE_DB),
                                 str(SOFT_BANK), device=DEVICE)
        cases = torch.load(Path(SOFT_CASE_DB) / "train_cases.pt", weights_only=False)
        self.file_names = [c["file_name"] for c in cases]
        self.params = {**_detector_params(self.p.net),
                       "Aggregator (DeepSets, soft)": _nparams(self.p.enc),
                       "CEAH (soft)": _nparams(self.p.ceah)}

    @torch.no_grad()
    def infer_rich(self, image, text_emb, top_k_cases, top_n, det_thresh=0.5):
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
        T.append(("① backbone+DETR forward (all Q)", (_sync_now() - t) * 1000))

        # Lesion gate: fixed threshold; abstain iff no query clears it. Aggregation
        # always uses all-Q (soft contract); CEAH always uses top-K; the threshold
        # only governs the DISPLAY mask + abstain.
        t = _sync_now()
        tau_g = None
        keep_mask = scores > det_thresh
        abstain = bool(int(keep_mask.sum().item()) == 0)
        T.append(("② 病灶門檻 (det_thresh)", (_sync_now() - t) * 1000))
        if abstain:
            r = _empty_result(image, T); r["abstain"] = True; r["tau_g"] = tau_g; return r
        Kk = lidx.numel()
        zk = z_all[lidx]                                   # [Kk,768] — full top-K for CEAH
        M = max(int(keep_mask.sum().item()), 1)
        bn = boxes[lidx[:M]]
        cx, cy, bw, bh = bn.unbind(-1)
        bxywh = [[float((cx[i] - bw[i] / 2) * W), float((cy[i] - bh[i] / 2) * H),
                  float(bw[i] * W), float(bh[i] * H)] for i in range(M)]
        crops = [scaled_rect_crop(image, b) for b in bxywh]
        cls = classify_against_anchors(zk[:M].float())

        t = _sync_now()
        zq = p.enc(g.float().unsqueeze(0), z_all.float().unsqueeze(0),
                   torch.tensor([w.numel()], device=dev), lesion_weights=w.float().unsqueeze(0))
        cand, topi, topw = _candidate_pool(zq, p.bank_z, p.memb, p.mlen, top_k_cases, dev)
        T.append(("③ soft 聚合 (全 Q) + 檢索", (_sync_now() - t) * 1000))

        t = _sync_now()
        lw = scores.float().unsqueeze(0)
        top_n_out, P = _ceah_topn(p.ceah, g.float(), zk.float(), cand, p.cause_embs,
                                  p.cause_texts, text_emb, top_n, dev,
                                  lesion_weights=lw.expand(int(cand.numel()), -1).contiguous())
        T.append(("④ soft CEAH (top-K by w) + α", (_sync_now() - t) * 1000))

        lesions = [{"idx": i, "bbox_xywh": bxywh[i], "det_score": float(scores[i]),
                    "crop": crops[i], "cls": cls[i]} for i in range(M)]
        return {"image_pil": image, "lesions": lesions, "n_lesions": M,
                "retrieved": _retrieved_cards(topi, topw, self.file_names),
                "top_n": top_n_out, "pool_size": P, "abstain": False, "tau_g": tau_g,
                "text_used": text_emb is not None, "timings": T}


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
    def infer_rich(self, image, text_emb, top_k_cases, top_n, det_thresh=0.5):
        dev = DEVICE; T = []
        t = _sync_now()
        pred = self.det.predict([image], threshold=det_thresh)
        det = pred[0] if isinstance(pred, list) else pred
        dets = sorted([([float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], float(s))
                       for b, s in zip(det.xyxy, det.confidence)], key=lambda x: -x[1])
        T.append(("① 偵測 (RF-DETR)", (_sync_now() - t) * 1000))
        if not dets:                                       # 沒框就當健康
            r = _empty_result(image, T); r["abstain"] = True; return r
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
        top_n_out, P = _ceah_topn(self.ceah, g, L, cand, self.cause_embs, self.cause_texts,
                                  text_emb, top_n, dev)
        T.append(("⑤ CEAH 排序 + α", (_sync_now() - t) * 1000))

        lesions = [{"idx": i, "bbox_xywh": bxywh[i], "det_score": scores[i],
                    "crop": crops[i], "cls": cls[i]} for i in range(N)]
        return {"image_pil": image, "lesions": lesions, "n_lesions": N,
                "retrieved": _retrieved_cards(topi, topw, self.file_names),
                "top_n": top_n_out, "pool_size": P, "abstain": False,
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


def handler_run(mode, image, text, det_thresh, top_k_cases, top_n_causes):
    if image is None:
        return (None, None, None, "請先上傳或選一張圖。", [], "", None, None, "",
                *_empty_buttons())
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    pipe = get_pipeline(mode)
    text_emb = encode_text_slot(text)
    top_k, top_n, dth = int(top_k_cases), int(top_n_causes), float(det_thresh)

    res = pipe.infer_rich(image, text_emb, top_k, top_n, dth)

    detailed = True
    timing_md = ""
    if detailed and res["n_lesions"] > 0:
        runs = [res["timings"]]
        for _ in range(N_TIMING_RUNS):
            runs.append(pipe.infer_rich(image, text_emb, top_k, top_n, dth)["timings"])
        stats = _aggregate_timings(runs[1:])    # drop warm-up (first)
        timing_md = _timing_md(stats, N_TIMING_RUNS)

    det_img = render_detection_image(res["image_pil"], res["lesions"])
    gallery = [(make_lesion_card(l), f"L{l['idx']}: {l['cls']['label_zh']}")
               for l in res["lesions"]]

    if res.get("abstain"):
        info = f"🟢 **abstain：無病灶超過門檻（det_thresh={float(det_thresh):.2f}），判定為健康**，不進行病因推論。"
    elif not res["lesions"]:
        info = "**偵測閾值下未偵測到病灶。**"
    else:
        thr = f"det_thresh={float(det_thresh):.2f}"
        info = (f"模式 **{mode}** ｜ 偵測到 **{res['n_lesions']}** 個病灶（{thr}）｜ "
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
            btns.append(gr.update(value=f"#{r['rank']}  s={r['score']:.2f}  {txt}", visible=True))
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
    explain = (f"### Top-{r['rank']} 病因\n**{r['text']}**\n\n"
               f"- CEAH score: **{r['score']:.3f}**\n"
               f"- α 加總 = 1（softmax）；數字越大代表該證據貢獻越強"
               + ("" if show_text else "；本次為 vision-only，未顯示 text α"))
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
| **grod** | Backbone(DINOv2)+DETR forward → 四 head（box / objectness / semantic / global）→ Aggregator (DeepSets) → dense 檢索 → CEAH |
| **grod_soft** | 同 grod 架構，lesion 選取改 soft（per-query 連續權重 w）+ soft pooling/α |
"""


def build_ui():
    with gr.Blocks(title="GROD Fish Disease Demo") as demo:
        gr.Markdown(DESCRIPTION)
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Dropdown(["grod", "grod_soft", "base"], value="grod", label="模式")
                inp_image = gr.Image(label="魚體輸入圖", type="pil", height=300)
                inp_text = gr.Textbox(label="選填：文字描述", lines=2,
                                      placeholder="例：體表潰瘍、紅腫，疑似感染；留空＝vision-only")
                with gr.Accordion("可調參數", open=False):
                    sld_det = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="det_thresh")
                    sld_topk = gr.Slider(5, 50, value=20, step=1, label="top_k_cases")
                    sld_topn = gr.Slider(1, MAX_TOPN_BUTTONS, value=5, step=1, label="top_n_causes")
                btn_run = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                out_det = gr.Image(label="① 偵測", type="pil", height=300)
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
                      [mode, inp_image, inp_text, sld_det, sld_topk, sld_topn],
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
