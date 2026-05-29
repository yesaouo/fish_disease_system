"""Build a FaCE-R-compatible case database from DDXPlus.

DDXPlus is text-only, so this adapter maps:
  - patient summary text -> global_emb
  - evidence texts       -> lesion_embs (compatibility key; semantically evidence)
  - pathology labels     -> causes

Outputs match build_case_database.py:
  train_cases.pt, valid_cases.pt, optional test_cases.pt,
  cause_text_embs.pt, meta.json
"""

from __future__ import annotations

import argparse
import gc
import ast
import csv
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

VL_CLASSIFIER_DIR = Path(__file__).resolve().parents[2] / "vl_classifier"
if str(VL_CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(VL_CLASSIFIER_DIR))

from common import get_text_features  # noqa: E402


def load_json(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Expected JSON object in {path}, got {type(payload).__name__}")


def iter_csv_rows(path: str) -> Iterable[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                # DDXPlus release zips store CSV content in extensionless files,
                # e.g. `release_train_patients`. Treat the first regular member
                # as CSV when no explicit *.csv entry exists.
                csv_names = [n for n in zf.namelist() if not n.endswith("/")]
            if not csv_names:
                raise FileNotFoundError(f"No readable file found inside {path}")
            name = sorted(csv_names)[0]
            with zf.open(name) as raw:
                txt = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                yield from csv.DictReader(txt)
        return

    with p.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def parse_cell(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    for parser in (ast.literal_eval, json.loads):
        try:
            return parser(text)
        except Exception:
            pass
    return text


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def get_nested_text(value: Any) -> str:
    """Extract an English display string from DDXPlus nested metadata values."""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, dict):
        for key in ("en", "english", "value_en", "name_en", "name"):
            if key in value:
                out = get_nested_text(value[key])
                if out:
                    return out
    return clean_text(value)


def lookup_evidence_meta(evidences_meta: Dict[str, Any], evidence_id: str) -> Dict[str, Any]:
    item = evidences_meta.get(evidence_id, {})
    if isinstance(item, dict):
        return item
    return {}


def lookup_value_meaning(meta: Dict[str, Any], value_id: str) -> str:
    for key in ("value_meaning", "value-meaning", "possible-values", "possible_values"):
        table = meta.get(key)
        if isinstance(table, dict):
            if value_id in table:
                return get_nested_text(table[value_id])
            try:
                int_key = int(value_id)
            except Exception:
                int_key = None
            if int_key is not None and int_key in table:
                return get_nested_text(table[int_key])
        elif isinstance(table, list):
            for item in table:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("id", item.get("value", item.get("code", ""))))
                if item_id == str(value_id):
                    return get_nested_text(item.get("en", item.get("label", item)))
    return clean_text(value_id)


def decode_evidence_token(token: Any, evidences_meta: Dict[str, Any]) -> Tuple[str, str]:
    """Return (evidence_id, display_text)."""
    raw = clean_text(token)
    if not raw:
        return "", ""

    if "_@_" in raw:
        evidence_id, value_id = raw.split("_@_", 1)
    else:
        evidence_id, value_id = raw, ""

    meta = lookup_evidence_meta(evidences_meta, evidence_id)
    question = clean_text(
        meta.get("question_en")
        or meta.get("question")
        or meta.get("name_en")
        or meta.get("name")
        or evidence_id
    )
    is_antecedent = bool(
        meta.get("is_antecedent")
        or meta.get("antecedent")
        or meta.get("is-antecedent")
    )
    kind = "Antecedent" if is_antecedent else "Symptom"
    if value_id:
        value_text = lookup_value_meaning(meta, value_id)
        return evidence_id, f"{kind}: {question} Answer: {value_text}."
    return evidence_id, f"{kind}: {question} Answer: yes."


def normalize_evidence_list(value: Any, evidences_meta: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    parsed = parse_cell(value)
    if parsed is None:
        items: List[Any] = []
    elif isinstance(parsed, (list, tuple, set)):
        items = list(parsed)
    else:
        items = [parsed]

    ids: List[str] = []
    texts: List[str] = []
    for item in items:
        evidence_id, text = decode_evidence_token(item, evidences_meta)
        if text:
            ids.append(evidence_id)
            texts.append(text)
    return ids, texts


def normalize_ddx(value: Any) -> List[dict]:
    parsed = parse_cell(value)
    if parsed is None:
        return []
    if not isinstance(parsed, (list, tuple)):
        return []

    out: List[dict] = []
    for item in parsed:
        disease = None
        prob = None
        if isinstance(item, dict):
            disease = item.get("name") or item.get("pathology") or item.get("condition")
            prob = item.get("probability", item.get("prob", item.get("score")))
        elif isinstance(item, (list, tuple)) and item:
            disease = item[0]
            if len(item) > 1:
                prob = item[1]
        disease_text = clean_text(disease)
        if not disease_text:
            continue
        try:
            prob_f = float(prob) if prob is not None else None
        except Exception:
            prob_f = None
        out.append({"name": disease_text, "prob": prob_f})
    return out


def pathology_from_row(row: dict) -> str:
    for key in ("PATHOLOGY", "pathology", "Pathology", "CONDITION", "condition"):
        if key in row:
            text = clean_text(row[key])
            if text:
                return text
    return ""


def row_id_from_row(row: dict, fallback: int) -> int:
    for key in ("PATIENT", "patient_id", "id", "ID"):
        if key not in row:
            continue
        text = clean_text(row[key])
        if not text:
            continue
        try:
            return int(float(text))
        except Exception:
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            return int(digest[:8], 16)
    return fallback


def build_global_text(age: str, sex: str, evidence_texts: Sequence[str]) -> str:
    pieces = []
    if age:
        pieces.append(f"Age: {age}.")
    if sex:
        pieces.append(f"Sex: {sex}.")
    if evidence_texts:
        pieces.append("Evidence: " + " ".join(evidence_texts))
    return " ".join(pieces).strip() or "Patient case."


def load_split_rows(path: str, split: str, evidences_meta: Dict[str, Any],
                    max_cases: int = -1, min_evidences: int = 1) -> List[dict]:
    cases: List[dict] = []
    for i, row in enumerate(iter_csv_rows(path)):
        pathology = pathology_from_row(row)
        if not pathology:
            continue

        initial_ids, initial_texts = normalize_evidence_list(
            row.get("INITIAL_EVIDENCE") or row.get("initial_evidence"),
            evidences_meta,
        )
        evidence_ids, evidence_texts = normalize_evidence_list(
            row.get("EVIDENCES") or row.get("evidences"),
            evidences_meta,
        )
        # Some releases keep the chief complaint in INITIAL_EVIDENCE while
        # others also include it in EVIDENCES. Preserve order and dedupe.
        merged_ids: List[str] = []
        merged_texts: List[str] = []
        seen_texts = set()
        for evidence_id, evidence_text in zip(initial_ids + evidence_ids,
                                             initial_texts + evidence_texts):
            if evidence_text in seen_texts:
                continue
            seen_texts.add(evidence_text)
            merged_ids.append(evidence_id)
            merged_texts.append(evidence_text)
        evidence_ids, evidence_texts = merged_ids, merged_texts
        if len(evidence_texts) < min_evidences:
            continue

        age = clean_text(row.get("AGE") or row.get("age"))
        sex = clean_text(row.get("SEX") or row.get("sex"))
        ddx = normalize_ddx(
            row.get("DIFFERENTIAL_DIAGNOSIS") or row.get("differential_diagnosis")
        )
        # global_text keeps the full descriptor (Age + Sex + Evidence
        # concatenated) so Phase 1 retrieval has a rich case-level vector to
        # match against. Lesion tokens get Age/Sex prepended as atomic units
        # so CEAH can attribute α to them separately from individual evidences.
        global_text = build_global_text(age, sex, evidence_texts)
        demographic_texts: List[str] = []
        demographic_ids: List[str] = []
        if age:
            demographic_texts.append(f"Age: {age}")
            demographic_ids.append("_demographic_age")
        if sex:
            demographic_texts.append(f"Sex: {sex}")
            demographic_ids.append("_demographic_sex")
        evidence_texts = demographic_texts + evidence_texts
        evidence_ids = demographic_ids + evidence_ids
        row_id = row_id_from_row(row, fallback=i)

        cases.append({
            "case_id": len(cases),
            "image_id": row_id,
            "split": split,
            "file_name": f"ddxplus/{split}/{row_id}",
            "age": age,
            "sex": sex,
            "global_text": global_text,
            "text_colloquial": global_text,
            "text_medical": global_text,
            "evidence_ids": evidence_ids,
            "evidence_texts": evidence_texts,
            "causes": [pathology],
            "ddx": ddx,
        })
        if max_cases > 0 and len(cases) >= max_cases:
            break
    print(f"[collect] {split}: kept={len(cases)} from {path}")
    return cases


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        text = clean_text(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def collect_cause_texts(splits: Sequence[List[dict]], include_ddx: bool) -> List[str]:
    items: List[str] = []
    for cases in splits:
        for c in cases:
            items.extend(c["causes"])
            if include_ddx:
                items.extend(d["name"] for d in c.get("ddx", []))
    return unique_preserve_order(items)


@torch.no_grad()
def encode_text_transformer(
    model, processor, texts: Sequence[str], device: str,
    batch_size: int, max_length: int, use_amp: bool,
) -> torch.Tensor:
    feats = []
    amp = bool(use_amp and str(device).startswith("cuda"))
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        ti = processor(
            text=batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        ti = {k: v.to(device) for k, v in ti.items()}
        with torch.cuda.amp.autocast(enabled=amp):
            f = get_text_features(model, ti["input_ids"], ti.get("attention_mask"))
        feats.append(F.normalize(f.float(), dim=-1).cpu())
    if not feats:
        return torch.empty(0)
    return torch.cat(feats, dim=0)


def load_transformer_encoder(model_name_or_path: str, device: str):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    model.eval()
    return model, processor


_TOKEN_RE = re.compile(r"[A-Za-z0-9_:+.-]+")


def encode_text_hash(texts: Sequence[str], dim: int) -> torch.Tensor:
    embs = torch.zeros(len(texts), dim, dtype=torch.float32)
    for row, text in enumerate(texts):
        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            tokens = [text.lower() or "<empty>"]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if int.from_bytes(digest[4:], "little") % 2 == 0 else -1.0
            embs[row, idx] += sign
    return F.normalize(embs, dim=-1)


class TextEncoder:
    def __init__(self, args):
        self.backend = args.embedding_backend
        self.dim = args.hash_dim
        self.model = None
        self.processor = None
        self.device = args.device
        self.batch_size = args.text_batch_size
        self.max_length = args.max_length
        self.use_amp = not args.no_amp

        if self.backend == "transformer":
            self.model, self.processor = load_transformer_encoder(args.text_encoder, args.device)
            self.dim = -1

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if self.backend == "hash":
            return encode_text_hash(texts, self.dim)
        embs = encode_text_transformer(
            self.model, self.processor, texts,
            device=self.device,
            batch_size=self.batch_size,
            max_length=self.max_length,
            use_amp=self.use_amp,
        )
        self.dim = int(embs.size(-1)) if embs.numel() else self.dim
        return embs


_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def parse_embedding_dtype(name: str) -> torch.dtype:
    key = name.lower()
    if key not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown --embedding_dtype {name!r}; choose from {sorted(_DTYPE_MAP)}"
        )
    return _DTYPE_MAP[key]


def save_shard(shard: List[dict], out_dir: Path, split: str, shard_id: int) -> str:
    """Save one shard and return its filename."""
    shard_name = f"{split}_cases_{shard_id:05d}.pt"
    torch.save(shard, out_dir / shard_name)
    print(f"[save] {shard_name} n={len(shard)} -> {out_dir}")
    return shard_name


def process_split_to_shards(
    cases: List[dict],
    split: str,
    encoder: TextEncoder,
    cause_text_to_idx: Dict[str, int],
    chunk_size: int,
    shard_size: int,
    output_dir: str,
    emb_dtype: torch.dtype = torch.float32,
) -> Tuple[List[str], int]:
    """Encode cases in small chunks, but save larger shards to avoid RAM growth.

    chunk_size controls encoding peak memory.
    shard_size controls how many encoded cases are written per .pt file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard: List[dict] = []
    shard_paths: List[str] = []
    shard_id = 0
    t0 = time.time()
    n = len(cases)

    for start in range(0, n, chunk_size):
        chunk = cases[start : start + chunk_size]

        global_embs = encoder.encode([c["global_text"] for c in chunk]).to(device="cpu", dtype=emb_dtype)

        evidence_flat: List[str] = []
        per_case_n: List[int] = []
        for c in chunk:
            per_case_n.append(len(c["evidence_texts"]))
            evidence_flat.extend(c["evidence_texts"])

        evidence_embs_flat = encoder.encode(evidence_flat).to(device="cpu", dtype=emb_dtype)

        cursor = 0
        for ci, (raw, n_evidence) in enumerate(zip(chunk, per_case_n)):
            ev = evidence_embs_flat[cursor : cursor + n_evidence].clone()
            cursor += n_evidence
            g = global_embs[ci].clone()

            # Pathology and DDX become separate fields:
            #   pathology_emb_idx — strict GT (single int, used for R@K pathology
            #     metric and CEAH positive_mask)
            #   cause_emb_indices — expanded list (pathology + DDX names that
            #     exist in cause_text_to_idx), used for candidate-pool
            #     construction so DDX alternatives can land in the pool and
            #     give NDCG@K real ranking space (DDXPlus pool was previously
            #     mean=1.08 — single pathology only). Fish builds don't set
            #     pathology_emb_idx; downstream consumers fall back to
            #     cause_emb_indices[0] when the field is absent.
            pathology_text = raw["causes"][0]
            pathology_idx = cause_text_to_idx[pathology_text]
            expanded_idxs: List[int] = [pathology_idx]
            seen_idxs = {pathology_idx}
            for d in raw.get("ddx", []) or []:
                name = d.get("name") if isinstance(d, dict) else None
                if not name or name not in cause_text_to_idx:
                    continue
                idx = cause_text_to_idx[name]
                if idx in seen_idxs:
                    continue
                seen_idxs.add(idx)
                expanded_idxs.append(idx)

            case_dict = {
                "case_id": raw["case_id"],
                "image_id": raw["image_id"],
                "split": split,
                "file_name": raw["file_name"],
                "global_emb": g,
                "text_colloquial_emb": g,
                "text_medical_emb": g,
                "lesion_embs": ev,
                "lesion_boxes_xywh": torch.zeros(n_evidence, 4, dtype=torch.long),
                "causes": list(raw["causes"]),
                "pathology_emb_idx": pathology_idx,
                "cause_emb_indices": expanded_idxs,
            }
            # Inspection / faithfulness fields are only consumed for query splits
            # (eval_retrieval / eval_ceah per_query dump, faithfulness_eval).
            # Train cases never need them, so drop ~3 KB/case (~400 MB at full
            # DDXPlus scale) and avoid pinning string lists in eval-time RAM.
            if split != "train":
                case_dict.update({
                    "ddx": list(raw.get("ddx", [])),
                    "evidence_ids": list(raw.get("evidence_ids", [])),
                    "evidence_texts": list(raw.get("evidence_texts", [])),
                    "age": raw.get("age", ""),
                    "sex": raw.get("sex", ""),
                    "global_text": raw.get("global_text", ""),
                })
            shard.append(case_dict)

        if len(shard) >= shard_size:
            shard_paths.append(save_shard(shard, out_dir, split, shard_id))
            shard_id += 1
            shard = []
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        done = min(start + len(chunk), n)
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-9)
        eta = (n - done) / max(rate, 1e-9)
        print(
            f"[{split}] {done}/{n} ({100.0 * done / max(n, 1):.1f}%) "
            f"rate={rate:.1f} cases/s ETA={eta/60:.1f} min "
            f"current_shard_cases={len(shard)}"
        )

        del chunk, global_embs, evidence_embs_flat, evidence_flat, per_case_n
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if shard:
        shard_paths.append(save_shard(shard, out_dir, split, shard_id))
        shard = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return shard_paths, n


def save_sharded_meta(
    args,
    train_count: int,
    valid_count: int,
    test_count: int,
    train_shards: List[str],
    valid_shards: List[str],
    test_shards: List[str],
    cause_texts: List[str],
    cause_embs: torch.Tensor,
) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_dtype = parse_embedding_dtype(args.embedding_dtype)
    cause_payload = {
        "texts": cause_texts,
        "embeddings": cause_embs.to(device="cpu", dtype=emb_dtype),
    }
    torch.save(cause_payload, out_dir / "cause_text_embs.pt")
    print(
        f"[save] cause_text_embs.pt n={len(cause_texts)} dim={cause_embs.size(-1)} "
        f"dtype={args.embedding_dtype}"
    )

    meta = {
        "dataset": "DDXPlus",
        "format": "sharded_cases",
        "embedding_backend": args.embedding_backend,
        "embedding_dtype": args.embedding_dtype,
        "text_encoder": args.text_encoder if args.embedding_backend == "transformer" else None,
        "global_dim": int(cause_embs.size(-1)),
        "lesion_dim": int(cause_embs.size(-1)),
        "lesion_semantics": "DDXPlus evidence token embeddings",
        "n_train_cases": train_count,
        "n_valid_cases": valid_count,
        "n_test_cases": test_count,
        "n_unique_causes": len(cause_texts),
        "include_ddx_causes": bool(args.include_ddx_causes),
        "train_csv": args.train_csv,
        "valid_csv": args.valid_csv,
        "test_csv": args.test_csv,
        "evidences_json": args.evidences_json,
        "conditions_json": args.conditions_json,
        "max_cases": args.max_cases,
        "min_evidences": args.min_evidences,
        "max_length": args.max_length,
        "text_batch_size": args.text_batch_size,
        "chunk_size": args.chunk_size,
        "shard_size": args.shard_size,
        "train_shards": train_shards,
        "valid_shards": valid_shards,
        "test_shards": test_shards,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] meta.json -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Build DDXPlus FaCE-R-compatible case DB.")
    ap.add_argument("--train_csv", type=str, required=True,
                    help="DDXPlus train CSV or release_train_patients.zip")
    ap.add_argument("--valid_csv", type=str, required=True,
                    help="DDXPlus validate CSV or release_validate_patients.zip")
    ap.add_argument("--test_csv", type=str, default=None,
                    help="Optional DDXPlus test CSV or release_test_patients.zip")
    ap.add_argument("--evidences_json", type=str, default=None,
                    help="release_evidences.json for decoding evidence IDs")
    ap.add_argument("--conditions_json", type=str, default=None,
                    help="release_conditions.json; stored for provenance for now")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--include_ddx_causes", action="store_true",
                    help="Include DIFFERENTIAL_DIAGNOSIS disease names in the cause table")
    ap.add_argument("--max_cases", type=int, default=-1,
                    help="Cap cases per split for smoke tests; -1 = all")
    ap.add_argument("--min_evidences", type=int, default=1)
    ap.add_argument("--chunk_size", type=int, default=8192,
                    help="Number of raw cases encoded at once; controls peak memory.")
    ap.add_argument("--shard_size", type=int, default=65536,
                    help="Number of encoded cases saved per shard file.")
    ap.add_argument("--embedding_backend", type=str, default="transformer",
                    choices=["transformer", "hash"])
    ap.add_argument("--text_encoder", type=str, default="google/siglip2-base-patch16-224")
    ap.add_argument("--hash_dim", type=int, default=768)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text_batch_size", type=int, default=8192)
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--embedding_dtype", type=str, default="float32",
                    choices=sorted(_DTYPE_MAP.keys()),
                    help="Storage dtype for saved embeddings (global, evidence/lesion, cause). "
                         "fp16/bf16 ~halve disk for full-DDXPlus shards.")
    args = ap.parse_args()
    emb_dtype = parse_embedding_dtype(args.embedding_dtype)

    evidences_meta = load_json(args.evidences_json)
    _ = load_json(args.conditions_json) if args.conditions_json else {}

    train_raw = load_split_rows(
        args.train_csv, "train", evidences_meta,
        max_cases=args.max_cases, min_evidences=args.min_evidences,
    )
    valid_raw = load_split_rows(
        args.valid_csv, "valid", evidences_meta,
        max_cases=args.max_cases, min_evidences=args.min_evidences,
    )
    test_raw = []
    if args.test_csv:
        test_raw = load_split_rows(
            args.test_csv, "test", evidences_meta,
            max_cases=args.max_cases, min_evidences=args.min_evidences,
        )

    cause_texts = collect_cause_texts(
        [train_raw, valid_raw, test_raw],
        include_ddx=args.include_ddx_causes,
    )
    cause_text_to_idx = {text: i for i, text in enumerate(cause_texts)}
    print(f"[cause-text] unique diseases={len(cause_texts)}")

    encoder = TextEncoder(args)
    cause_embs = encoder.encode(cause_texts)
    print(f"[cause-text] encoded shape={tuple(cause_embs.shape)}")

    train_shards, train_count = process_split_to_shards(
        train_raw,
        "train",
        encoder,
        cause_text_to_idx,
        args.chunk_size,
        args.shard_size,
        args.output_dir,
        emb_dtype=emb_dtype,
    )
    del train_raw
    gc.collect()

    valid_shards, valid_count = process_split_to_shards(
        valid_raw,
        "valid",
        encoder,
        cause_text_to_idx,
        args.chunk_size,
        args.shard_size,
        args.output_dir,
        emb_dtype=emb_dtype,
    )
    del valid_raw
    gc.collect()

    test_shards: List[str] = []
    test_count = 0
    if test_raw:
        test_shards, test_count = process_split_to_shards(
            test_raw,
            "test",
            encoder,
            cause_text_to_idx,
            args.chunk_size,
            args.shard_size,
            args.output_dir,
            emb_dtype=emb_dtype,
        )
        del test_raw
        gc.collect()

    save_sharded_meta(
        args,
        train_count,
        valid_count,
        test_count,
        train_shards,
        valid_shards,
        test_shards,
        cause_texts,
        cause_embs,
    )


if __name__ == "__main__":
    main()
