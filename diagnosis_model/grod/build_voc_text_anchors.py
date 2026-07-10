"""建立 VOC 20 類名的凍結 SigLIP2 文字錨 [20, 768]（A1 OAVLE 文字語意匹配用）。

比照 build_text_anchors.py：每類用類名 prompt（"a photo of a {class}" + 中文名）過凍結
SigLIP2 text encoder、L2-norm、平均成一個 anchor。anchor_embs[i] 對應 VOC 類索引 i
（aeroplane=0..tvmonitor=19，= class-agnostic COCO 的 symptom_category_id）。temp 校準
（不含 logit_scale，criterion 走 /semantic_temp），與魚病 anchor 生產設定一致。

  $PY -m diagnosis_model.grod.build_voc_text_anchors
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

VL_DIR = Path(__file__).resolve().parents[1] / "vl_classifier"
if str(VL_DIR) not in sys.path:
    sys.path.insert(0, str(VL_DIR))
from common import get_text_features  # noqa: E402

BANK = "diagnosis_model/vl_classifier/voc_pipeline/voc_label_bank.json"
OUT = "diagnosis_model/grod/outputs/voc_text_anchors.pt"
MODEL = "google/siglip2-base-patch16-224"


@torch.no_grad()
def main() -> None:
    from transformers import AutoModel, AutoProcessor

    label_map = json.load(open(BANK))["label_map"]     # {"0": {"en","zh"}, ...}
    n = len(label_map)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()

    anchors = []
    for i in range(n):
        en, zh = label_map[str(i)]["en"], label_map[str(i)]["zh"]
        article = "an" if en[:1].lower() in "aeiou" else "a"
        texts = [f"a photo of {article} {en}", zh]
        ti = proc(text=texts, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        feats = get_text_features(model, ti["input_ids"], ti.get("attention_mask"))
        emb = F.normalize(F.normalize(feats.float(), dim=-1).mean(0, keepdim=True), dim=-1)
        anchors.append(emb)

    anchors = torch.cat(anchors, dim=0)                # [20, 768]
    pack = {"anchor_embs": anchors, "num_cats": n, "dim": anchors.size(-1), "model_name": MODEL}
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, OUT)
    print(f"[done] VOC anchors [{n}, {anchors.size(-1)}] -> {OUT}")
    print(f"  classes: {[label_map[str(i)]['en'] for i in range(n)]}")


if __name__ == "__main__":
    main()
