"""LLM-based cause string consolidation with Ollama.

This is an alternative to embedding + HDBSCAN clustering. It incrementally asks
an instruction LLM served by Ollama to map a small batch of raw cause strings
into existing canonical cause groups, or create new groups when no existing
group is similar.

The output JSON matches cause_cluster_json.py:
  - cause_id_to_canonical
  - original_to_cause_id
  - cluster_meta

The long-running state is checkpointed after every completed batch, so rerunning
the same command resumes from the last saved batch.

Sharded mode is available for large inputs. It runs local consolidation inside
bounded shards, then merges the shard-level canonical causes and rebuilds a
single global output. This prevents the prompt's Existing canonical causes list
from growing with the entire dataset during the first pass.

Prerequisite:
  ollama serve
  ollama pull <model>

Examples:
  # Original one-stage mode
  python cause_cluster_llm.py --model gemma4:26b --batch_size 50

  # Sharded mode: local clusters per 500 causes, then merge local canonicals
  python cause_cluster_llm.py --model gemma4:26b --batch_size 5 --shard_size 500 --merge_batch_size 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from diagnosis_model.cause_inference.preprocessing.cause_cluster_json import (  # type: ignore
        load_clusters_json,
        quality_report,
        save_clusters_json,
    )
    from diagnosis_model.cause_inference.preprocessing.cause_texts import (  # type: ignore
        collect_cause_strings_from_coco,
    )
else:
    from .cause_cluster_json import load_clusters_json, quality_report, save_clusters_json
    from .cause_texts import collect_cause_strings_from_coco


DEFAULT_COCO_FILES = [
    "data/detection/coco/_merged/train/_annotations.coco.json",
    "data/detection/coco/_merged/valid/_annotations.coco.json",
]
DEFAULT_MODEL = "gemma4:26b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OUTPUT = "diagnosis_model/cause_inference/outputs/cause_clusters_llm.json"
CHECKPOINT_VERSION = 1


SYSTEM_PROMPT = """You consolidate fish disease cause sentences.

Rules:
1. Existing canonical causes have priority. If a new input has the same meaning
   or is a more specific/general wording of an existing canonical cause, assign
   it to that existing id.
2. Create a new cluster only when none of the existing canonical causes and none
   of the new clusters in this batch express the same cause.
3. You may update an existing canonical sentence when the new input helps make
   the canonical wording clearer, but keep it concise and medically meaningful.
4. Preserve the meaning. Do not invent disease causes. Do not merge unrelated
   causes.
5. Every input_id from New input causes must appear exactly once in assignments.
6. Return raw JSON only. No markdown, no code fences, no explanation.
"""


USER_TEMPLATE = """{existing_note}Existing canonical causes:
{existing_json}

New input causes:
{batch_json}

Return exactly this JSON shape:
{{
  "canonical_updates": [
    {{"target": "E1", "canonical": "updated canonical sentence"}}
  ],
  "new_clusters": [
    {{"target": "N1", "canonical": "new canonical sentence"}}
  ],
  "assignments": [
    {{"input_id": 1, "target": "E1"}},
    {{"input_id": 2, "target": "N1"}}
  ]
}}

Use input_id values exactly as provided in New input causes. Do not copy the
input text into assignments. Use target values that are always strings. Use
"E1", "E2", ... only for existing ids shown in Existing canonical causes. Use
"N1", "N2", ... only for new clusters defined in new_clusters. If there are no
canonical updates or no new clusters, return an empty list for that field. If
Existing canonical causes is an empty list, canonical_updates must be [] and
every assignment must target a new cluster target from new_clusters.
"""


OUTPUT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "canonical_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "target": {"type": "string", "pattern": "^E[0-9]+$"},
                    "canonical": {"type": "string", "minLength": 1},
                },
                "required": ["target", "canonical"],
            },
        },
        "new_clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "target": {"type": "string", "pattern": "^N[0-9]+$"},
                    "canonical": {"type": "string", "minLength": 1},
                },
                "required": ["target", "canonical"],
            },
        },
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "input_id": {"type": "integer", "minimum": 1},
                    "target": {"type": "string", "pattern": "^(E[0-9]+|N[0-9]+)$"},
                },
                "required": ["input_id", "target"],
            },
        },
    },
    "required": ["canonical_updates", "new_clusters", "assignments"],
}


class ResponseValidationError(ValueError):
    """Raised when the LLM response is syntactically valid JSON but unusable."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _atomic_write_json(payload: Dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(output)


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _input_fingerprint(texts: Sequence[str]) -> str:
    payload = json.dumps(list(texts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_lines(path: str | Path) -> List[str]:
    with Path(path).open("r", encoding="utf-8") as f:
        return _dedupe_preserve_order(line.strip() for line in f)


def _read_cache_texts(cache_dir: str | Path) -> List[str]:
    path = Path(cache_dir) / "texts.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list of strings")
    return _dedupe_preserve_order(str(x) for x in payload if isinstance(x, str))


def load_input_texts(args: argparse.Namespace) -> List[str]:
    if args.input_texts:
        texts = _read_lines(args.input_texts)
        source = args.input_texts
    elif args.cache_dir:
        texts = _read_cache_texts(args.cache_dir)
        source = str(Path(args.cache_dir) / "texts.json")
    else:
        texts = collect_cause_strings_from_coco(
            args.coco_files,
            skip_healthy=not args.include_healthy,
            verbose=True,
        )
        source = ", ".join(args.coco_files)

    if args.max_items and args.max_items > 0:
        texts = texts[: args.max_items]
    print(f"[input] loaded {len(texts)} unique cause strings from {source}")
    return texts


def _int_key(value: Any) -> int:
    return int(value)


def _sorted_items_by_int_key(mapping: Dict[Any, Any]) -> List[Tuple[int, Any]]:
    return [(int(k), v) for k, v in sorted(mapping.items(), key=lambda kv: int(kv[0]))]

def _get_by_int_key(mapping: Dict[Any, Any], key: int) -> Any:
    if key in mapping:
        return mapping[key]
    str_key = str(key)
    if str_key in mapping:
        return mapping[str_key]
    raise KeyError(key)


def clusters_to_output(state: Dict[str, Any]) -> Dict[str, Any]:
    clusters = state["clusters"]
    return {
        "cause_id_to_canonical": {
            int(c["id"]): str(c["canonical"]) for c in clusters
        },
        "original_to_cause_id": {
            str(k): int(v) for k, v in state["original_to_cause_id"].items()
        },
        "cluster_meta": {
            int(c["id"]): {
                "size": len(c.get("members", [])),
                "members": list(c.get("members", [])),
            }
            for c in clusters
        },
    }


def _empty_state(fingerprint: str, total: int) -> Dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "input_fingerprint": fingerprint,
        "total": int(total),
        "processed_count": 0,
        "next_cause_id": 1,
        "clusters": [],
        "original_to_cause_id": {},
        "created_at": _now(),
        "updated_at": _now(),
    }


def _state_from_output(path: Path, texts: Sequence[str], fingerprint: str) -> Dict[str, Any]:
    clusters = load_clusters_json(path)
    original_to_cause_id = clusters["original_to_cause_id"]

    processed_count = 0
    for text in texts:
        if text not in original_to_cause_id:
            break
        processed_count += 1

    if processed_count != len(original_to_cause_id):
        raise ValueError(
            f"{path} contains non-contiguous processed inputs. "
            "Use the checkpoint file or rerun with --overwrite."
        )

    restored_clusters: List[Dict[str, Any]] = []
    for cause_id, meta in _sorted_items_by_int_key(clusters["cluster_meta"]):
        restored_clusters.append(
            {
                "id": int(cause_id),
                "canonical": str(_get_by_int_key(clusters["cause_id_to_canonical"], cause_id)),
                "members": list(meta.get("members", [])),
            }
        )

    next_id = max([0] + [int(c["id"]) for c in restored_clusters]) + 1
    return {
        "version": CHECKPOINT_VERSION,
        "input_fingerprint": fingerprint,
        "total": len(texts),
        "processed_count": processed_count,
        "next_cause_id": next_id,
        "clusters": restored_clusters,
        "original_to_cause_id": {str(k): int(v) for k, v in original_to_cause_id.items()},
        "created_at": _now(),
        "updated_at": _now(),
    }


def load_or_create_state(
    texts: Sequence[str],
    output_path: Path,
    checkpoint_path: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    fingerprint = _input_fingerprint(texts)
    if overwrite:
        print("[resume] --overwrite set; starting from an empty state")
        return _empty_state(fingerprint, len(texts))

    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("input_fingerprint") != fingerprint:
            raise ValueError(
                f"Checkpoint input fingerprint mismatch: {checkpoint_path}. "
                "Use the same inputs or rerun with --overwrite."
            )
        if int(state.get("version", 0)) != CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported checkpoint version in {checkpoint_path}")
        print(
            f"[resume] loaded checkpoint {checkpoint_path}; "
            f"processed={state['processed_count']}/{len(texts)}"
        )
        return state

    if output_path.exists():
        state = _state_from_output(output_path, texts, fingerprint)
        print(
            f"[resume] rebuilt state from {output_path}; "
            f"processed={state['processed_count']}/{len(texts)}"
        )
        return state

    return _empty_state(fingerprint, len(texts))


def save_state(state: Dict[str, Any], output_path: Path, checkpoint_path: Path) -> None:
    state["updated_at"] = _now()
    _atomic_write_json(state, checkpoint_path)
    save_clusters_json(clusters_to_output(state), output_path)


def build_prompt(
    clusters: Sequence[Dict[str, Any]],
    batch: Sequence[str],
    max_existing_in_prompt: int,
) -> Tuple[str, set[int]]:
    if max_existing_in_prompt > 0 and len(clusters) > max_existing_in_prompt:
        visible = list(clusters)[-max_existing_in_prompt:]
        hidden = len(clusters) - len(visible)
    else:
        visible = list(clusters)
        hidden = 0

    existing = [
        {"target": f"E{int(c['id'])}", "canonical": str(c["canonical"])}
        for c in visible
    ]
    existing_note = ""
    if hidden:
        existing_note = (
            f"Note: {hidden} older canonical causes were omitted because "
            "max_existing_in_prompt was reached. Only assign to targets shown below.\n\n"
        )

    indexed_batch = [
        {"input_id": idx, "text": text}
        for idx, text in enumerate(batch, start=1)
    ]

    prompt = USER_TEMPLATE.format(
        existing_note=existing_note,
        existing_json=json.dumps(existing, ensure_ascii=False, indent=2),
        batch_json=json.dumps(indexed_batch, ensure_ascii=False, indent=2),
    )
    return prompt, {int(c["id"]) for c in visible}


def _normalize_ollama_host(host: str) -> str:
    host = host.strip()
    if not host:
        raise ValueError("--ollama_host must not be empty")
    return host.rstrip("/")


def _ollama_timeout(value: float) -> Optional[float]:
    return None if value <= 0 else value


def _ollama_options(args: argparse.Namespace) -> Dict[str, Any]:
    options: Dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_predict": args.max_tokens,
    }
    if args.top_k is not None:
        options["top_k"] = args.top_k
    if args.num_ctx is not None:
        options["num_ctx"] = args.num_ctx
    if args.seed is not None:
        options["seed"] = args.seed
    return options


def _ollama_format(args: argparse.Namespace) -> str | Dict[str, Any]:
    if args.json_mode:
        return "json"
    return OUTPUT_JSON_SCHEMA


def _ollama_think(value: str) -> bool | str:
    if value == "true":
        return True
    if value == "false":
        return False
    return value


def _ollama_post(
    host: str,
    endpoint: str,
    payload: Dict[str, Any],
    timeout: Optional[float],
) -> Dict[str, Any]:
    url = f"{_normalize_ollama_host(host)}{endpoint}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama request failed: HTTP {exc.code} {exc.reason}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {host}. "
            "Make sure `ollama serve` is running and --ollama_host is correct."
        ) from exc

    try:
        decoded = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned non-JSON response: {response_body[:500]}") from exc

    if not isinstance(decoded, dict):
        raise RuntimeError(f"Ollama returned unexpected response type: {type(decoded).__name__}")
    if decoded.get("error"):
        raise RuntimeError(f"Ollama returned an error: {decoded['error']}")
    return decoded


def generate_response(
    args: argparse.Namespace,
    prompt: str,
    think_override: Optional[str] = None,
) -> str:
    payload_common: Dict[str, Any] = {
        "model": args.model,
        "stream": False,
        "think": _ollama_think(think_override if think_override is not None else args.think),
        "format": _ollama_format(args),
        "options": _ollama_options(args),
    }
    if args.keep_alive is not None:
        payload_common["keep_alive"] = args.keep_alive

    timeout = _ollama_timeout(args.request_timeout)

    if args.use_generate:
        payload = {
            **payload_common,
            "system": SYSTEM_PROMPT,
            "prompt": f"{prompt}\n\nJSON:",
        }
        response = _ollama_post(args.ollama_host, "/api/generate", payload, timeout)
        text = response.get("response", "")
    else:
        payload = {
            **payload_common,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        response = _ollama_post(args.ollama_host, "/api/chat", payload, timeout)
        message = response.get("message", {})
        if not isinstance(message, dict):
            raise RuntimeError(f"Ollama chat response missing message object: {response}")
        text = message.get("content", "")

    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"Ollama returned empty content: {response}")
    return text.strip()


def parse_json_strict(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if not (text.startswith("{") and text.endswith("}")):
        raise ResponseValidationError("response must be one raw JSON object")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ResponseValidationError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResponseValidationError("top-level JSON must be an object")
    return payload


def _require_keys(obj: Dict[str, Any], keys: Sequence[str], where: str) -> None:
    actual = set(obj)
    expected = set(keys)
    if actual != expected:
        raise ResponseValidationError(
            f"{where} keys must be {sorted(expected)}, got {sorted(actual)}"
        )


def _target_to_key(target: Any) -> Tuple[str, int | str]:
    if not isinstance(target, str):
        raise ResponseValidationError(f"invalid target: {target!r}; expected E<number> or N<number>")
    if re.fullmatch(r"E[0-9]+", target):
        return "existing", int(target[1:])
    if re.fullmatch(r"N[0-9]+", target):
        return "new", target
    raise ResponseValidationError(f"invalid target: {target!r}; expected E<number> or N<number>")


def validate_response(
    payload: Dict[str, Any],
    batch: Sequence[str],
    existing_ids: set[int],
) -> Dict[str, Any]:
    _require_keys(payload, ["canonical_updates", "new_clusters", "assignments"], "response")

    canonical_updates = payload["canonical_updates"]
    new_clusters = payload["new_clusters"]
    assignments = payload["assignments"]
    if not isinstance(canonical_updates, list):
        raise ResponseValidationError("canonical_updates must be a list")
    if not isinstance(new_clusters, list):
        raise ResponseValidationError("new_clusters must be a list")
    if not isinstance(assignments, list):
        raise ResponseValidationError("assignments must be a list")

    updates: Dict[int, str] = {}
    for idx, item in enumerate(canonical_updates):
        if not isinstance(item, dict):
            raise ResponseValidationError(f"canonical_updates[{idx}] must be an object")
        _require_keys(item, ["target", "canonical"], f"canonical_updates[{idx}]")
        target_kind, update_id = _target_to_key(item["target"])
        canonical = str(item["canonical"]).strip()
        if target_kind != "existing" or int(update_id) not in existing_ids:
            raise ResponseValidationError(f"canonical_updates[{idx}].target is not a shown existing target")
        if not canonical:
            raise ResponseValidationError(f"canonical_updates[{idx}].canonical is empty")
        updates[int(update_id)] = canonical

    new_defs: Dict[str, str] = {}
    for idx, item in enumerate(new_clusters):
        if not isinstance(item, dict):
            raise ResponseValidationError(f"new_clusters[{idx}] must be an object")
        _require_keys(item, ["target", "canonical"], f"new_clusters[{idx}]")
        target_kind, new_id = _target_to_key(item["target"])
        canonical = str(item["canonical"]).strip()
        if target_kind != "new":
            raise ResponseValidationError(f"new_clusters[{idx}].target must look like N1")
        new_id = str(new_id)
        if new_id in new_defs:
            raise ResponseValidationError(f"duplicate new target: {new_id}")
        if not canonical:
            raise ResponseValidationError(f"new_clusters[{idx}].canonical is empty")
        new_defs[new_id] = canonical

    id_to_text = {idx: text for idx, text in enumerate(batch, start=1)}
    assigned_ids: List[int] = []
    normalized_assignments: List[Tuple[str, str, int | str]] = []
    assigned_new_ids: set[str] = set()
    for idx, item in enumerate(assignments):
        if not isinstance(item, dict):
            raise ResponseValidationError(f"assignments[{idx}] must be an object")
        _require_keys(item, ["input_id", "target"], f"assignments[{idx}]")
        input_id = item["input_id"]
        if not isinstance(input_id, int) or input_id not in id_to_text:
            raise ResponseValidationError(
                f"assignments[{idx}].input_id must be an integer from 1 to {len(batch)}"
            )
        target_kind, target_key = _target_to_key(item["target"])
        if target_kind == "existing":
            if int(target_key) not in existing_ids:
                raise ResponseValidationError(
                    f"assignments[{idx}].target references unknown existing target"
                )
        else:
            if str(target_key) not in new_defs:
                raise ResponseValidationError(
                    f"assignments[{idx}].target references undefined new target"
                )
            assigned_new_ids.add(str(target_key))
        assigned_ids.append(input_id)
        normalized_assignments.append((id_to_text[input_id], target_kind, target_key))

    if len(assigned_ids) != len(batch):
        raise ResponseValidationError(
            f"expected {len(batch)} assignments, got {len(assigned_ids)}"
        )
    expected_ids = set(range(1, len(batch) + 1))
    actual_ids = set(assigned_ids)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ResponseValidationError(f"assignment input_id mismatch; missing={missing}, extra={extra}")
    if len(actual_ids) != len(assigned_ids):
        raise ResponseValidationError("duplicate assignment input_id")

    unused_new_ids = sorted(set(new_defs) - assigned_new_ids)
    if unused_new_ids:
        raise ResponseValidationError(f"new_clusters not used in assignments: {unused_new_ids}")

    return {
        "updates": updates,
        "new_defs": new_defs,
        "assignments": normalized_assignments,
    }


def apply_plan(state: Dict[str, Any], plan: Dict[str, Any]) -> None:
    clusters_by_id = {int(c["id"]): c for c in state["clusters"]}

    for cause_id, canonical in plan["updates"].items():
        clusters_by_id[int(cause_id)]["canonical"] = canonical

    new_id_to_real: Dict[str, int] = {}
    for new_id, canonical in sorted(
        plan["new_defs"].items(),
        key=lambda kv: int(kv[0][1:]),
    ):
        real_id = int(state["next_cause_id"])
        state["next_cause_id"] = real_id + 1
        cluster = {"id": real_id, "canonical": canonical, "members": []}
        state["clusters"].append(cluster)
        clusters_by_id[real_id] = cluster
        new_id_to_real[new_id] = real_id

    for text, target_kind, target_key in plan["assignments"]:
        if text in state["original_to_cause_id"]:
            continue
        real_id = int(target_key) if target_kind == "existing" else new_id_to_real[str(target_key)]
        clusters_by_id[real_id]["members"].append(text)
        state["original_to_cause_id"][text] = real_id


def _iter_batches(texts: Sequence[str], start: int, batch_size: int):
    for idx in range(start, len(texts), batch_size):
        yield idx, list(texts[idx : idx + batch_size])


def _iter_shards(texts: Sequence[str], shard_size: int):
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    for shard_idx, start in enumerate(range(0, len(texts), shard_size)):
        shard = list(texts[start : start + shard_size])
        yield shard_idx, start, shard


def _progress_bar(total: int, initial: int, desc: str = "llm-consolidate"):
    try:
        from tqdm import tqdm

        return tqdm(total=total, initial=initial, desc=desc, unit="cause")
    except Exception:  # pragma: no cover - optional dependency
        return None


def _save_failed_response(
    raw: str,
    output_path: Path,
    batch_start: int,
    attempt: int,
) -> None:
    failed_dir = output_path.parent / (output_path.name + ".failed_responses")
    failed_dir.mkdir(parents=True, exist_ok=True)
    failed_path = failed_dir / f"batch_{batch_start:06d}_attempt_{attempt}.txt"
    failed_path.write_text(raw, encoding="utf-8")
    print(f"[retry] raw response saved to {failed_path}")


def _print_ollama_config(args: argparse.Namespace) -> None:
    print(f"[ollama] host={args.ollama_host}")
    print(f"[ollama] model={args.model}")
    print("[ollama] output_format=" + ("json" if args.json_mode else "json_schema"))


def _attempt_batch_once(
    args: argparse.Namespace,
    state: Dict[str, Any],
    output_path: Path,
    checkpoint_path: Path,
    batch_start: int,
    batch: Sequence[str],
    attempt_label: str,
    failed_response_attempt: int,
    think_override: Optional[str] = None,
) -> Optional[Exception]:
    prompt, visible_existing_ids = build_prompt(
        state["clusters"],
        batch,
        max_existing_in_prompt=args.max_existing_in_prompt,
    )

    raw = generate_response(args=args, prompt=prompt, think_override=think_override)
    try:
        payload = parse_json_strict(raw)
        plan = validate_response(payload, batch, visible_existing_ids)
        apply_plan(state, plan)
        state["processed_count"] = batch_start + len(batch)
        save_state(state, output_path, checkpoint_path)
        return None
    except ResponseValidationError as exc:
        print(
            f"\n[retry] batch_start={batch_start} "
            f"batch_items={len(batch)} {attempt_label}: {exc}"
        )
        if args.save_failed_responses:
            _save_failed_response(raw, output_path, batch_start, failed_response_attempt)
        return exc


def _half_batch(batch: Sequence[str], batch_size: int) -> List[str]:
    half_size = max(1, batch_size // 2)
    return list(batch[: min(len(batch), half_size)])


def _try_one_batch(
    args: argparse.Namespace,
    state: Dict[str, Any],
    output_path: Path,
    checkpoint_path: Path,
    batch_start: int,
    batch: Sequence[str],
) -> int:
    """Try one logical batch and return how many inputs were committed.

    Retry policy:
      1. Try the original batch size for args.max_attempts attempts.
      2. If still invalid, try the first half-batch for args.half_batch_attempts attempts.
      3. If still invalid and thinking is disabled, try the half-batch once with think=true.

    When a half-batch succeeds, only that half is committed. The caller will resume
    from the updated processed_count, so the remaining items are retried next.
    """
    last_error: Optional[Exception] = None
    failed_response_attempt = 0

    for attempt in range(1, args.max_attempts + 1):
        failed_response_attempt += 1
        error = _attempt_batch_once(
            args=args,
            state=state,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            batch_start=batch_start,
            batch=batch,
            attempt_label=f"full_attempt={attempt}/{args.max_attempts}",
            failed_response_attempt=failed_response_attempt,
        )
        if error is None:
            return len(batch)
        last_error = error

    reduced_batch = _half_batch(batch, args.batch_size)
    print(
        f"\n[fallback] batch_start={batch_start}: "
        f"full batch failed; retrying with batch_size/2 ({len(reduced_batch)} item(s))"
    )
    for attempt in range(1, args.half_batch_attempts + 1):
        failed_response_attempt += 1
        error = _attempt_batch_once(
            args=args,
            state=state,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            batch_start=batch_start,
            batch=reduced_batch,
            attempt_label=f"half_attempt={attempt}/{args.half_batch_attempts}",
            failed_response_attempt=failed_response_attempt,
        )
        if error is None:
            return len(reduced_batch)
        last_error = error

    if args.think == "false":
        print(
            f"\n[fallback] batch_start={batch_start}: "
            "half batch failed; retrying once with think=true"
        )
        failed_response_attempt += 1
        error = _attempt_batch_once(
            args=args,
            state=state,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            batch_start=batch_start,
            batch=reduced_batch,
            attempt_label="half_think_attempt=1/1",
            failed_response_attempt=failed_response_attempt,
            think_override="true",
        )
        if error is None:
            return len(reduced_batch)
        last_error = error
    else:
        print(
            f"\n[fallback] batch_start={batch_start}: "
            f"think is already enabled ({args.think}); skipping extra think fallback"
        )

    total_attempts = args.max_attempts + args.half_batch_attempts + (1 if args.think == "false" else 0)
    raise SystemExit(
        f"LLM response failed validation after {total_attempts} scheduled attempt(s); "
        f"last error: {last_error}. Checkpoint remains at {checkpoint_path}"
    )


def run_consolidation_on_texts(
    args: argparse.Namespace,
    texts: Sequence[str],
    output_path: Path,
    checkpoint_path: Path,
    overwrite: bool,
    progress_desc: str = "llm-consolidate",
    print_done_summary: bool = False,
) -> Dict[str, Any]:
    """Run the existing incremental LLM clustering loop on an explicit text list."""
    state = load_or_create_state(
        texts=texts,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        overwrite=overwrite,
    )
    save_state(state, output_path, checkpoint_path)

    if int(state["processed_count"]) >= len(texts):
        print("[done] all inputs already processed")
        output = clusters_to_output(state)
        if print_done_summary:
            print(quality_report(output, top_n=args.report_top_n))
        return output

    pbar = _progress_bar(total=len(texts), initial=int(state["processed_count"]), desc=progress_desc)

    try:
        while int(state["processed_count"]) < len(texts):
            batch_start = int(state["processed_count"])
            batch = list(texts[batch_start : batch_start + args.batch_size])
            committed = _try_one_batch(
                args=args,
                state=state,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                batch_start=batch_start,
                batch=batch,
            )
            if pbar is not None:
                pbar.update(committed)
            else:
                print(
                    f"[progress] {state['processed_count']}/{len(texts)} "
                    f"({state['processed_count'] / max(1, len(texts)):.1%})"
                )
    except KeyboardInterrupt:
        save_state(state, output_path, checkpoint_path)
        print(f"\n[interrupt] saved checkpoint to {checkpoint_path}")
        raise SystemExit(130) from None
    finally:
        if pbar is not None:
            pbar.close()

    return clusters_to_output(state)


def _clone_args(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    payload = vars(args).copy()
    payload.update(overrides)
    return argparse.Namespace(**payload)


def _default_shard_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}.shards"


def _local_canonical_records(
    shard_outputs: Sequence[Dict[str, Any]],
) -> Tuple[List[str], Dict[Tuple[int, int], str]]:
    """Return unique local canonical texts and a local ref -> canonical text map.

    Multiple shard-local clusters can have exactly the same canonical sentence.
    The merge stage should see that sentence only once, while every local
    (shard_idx, local_cause_id) reference still maps to the resulting global id.
    """
    canonical_to_seen: Dict[str, None] = {}
    ref_to_canonical: Dict[Tuple[int, int], str] = {}

    for shard_idx, output in enumerate(shard_outputs):
        cause_id_to_canonical = output["cause_id_to_canonical"]
        for local_id, canonical in _sorted_items_by_int_key(cause_id_to_canonical):
            text = str(canonical).strip()
            if not text:
                raise ValueError(f"Shard {shard_idx} local cause id {local_id} has empty canonical")
            canonical_to_seen.setdefault(text, None)
            ref_to_canonical[(shard_idx, int(local_id))] = text

    return list(canonical_to_seen.keys()), ref_to_canonical


def _rebuild_global_output(
    shard_outputs: Sequence[Dict[str, Any]],
    local_ref_to_global_id: Dict[Tuple[int, int], int],
    global_cause_id_to_canonical: Dict[int, str],
) -> Dict[str, Any]:
    original_to_cause_id: Dict[str, int] = {}
    cluster_members: Dict[int, List[str]] = {
        int(global_id): [] for global_id in global_cause_id_to_canonical
    }

    for shard_idx, output in enumerate(shard_outputs):
        local_original_to_cause_id = output["original_to_cause_id"]
        for original, local_id in local_original_to_cause_id.items():
            ref = (shard_idx, int(local_id))
            if ref not in local_ref_to_global_id:
                raise ValueError(f"Missing global id for shard/local cause ref {ref}")
            global_id = local_ref_to_global_id[ref]
            original_to_cause_id[str(original)] = global_id
            cluster_members.setdefault(global_id, []).append(str(original))

    return {
        "cause_id_to_canonical": {
            int(global_id): global_cause_id_to_canonical[int(global_id)]
            for global_id in sorted(global_cause_id_to_canonical)
        },
        "original_to_cause_id": original_to_cause_id,
        "cluster_meta": {
            int(global_id): {
                "size": len(members),
                "members": members,
            }
            for global_id, members in sorted(cluster_members.items())
        },
    }


def merge_shard_outputs_with_llm(
    args: argparse.Namespace,
    shard_outputs: Sequence[Dict[str, Any]],
    output_path: Path,
    shard_dir: Path,
) -> Dict[str, Any]:
    merge_texts, ref_to_canonical = _local_canonical_records(shard_outputs)
    print(f"\n[merge] local canonical causes={len(merge_texts)}")

    if not merge_texts:
        return {
            "cause_id_to_canonical": {},
            "original_to_cause_id": {},
            "cluster_meta": {},
        }

    merge_args = _clone_args(args, batch_size=args.merge_batch_size)
    merge_output_path = shard_dir / f"{output_path.stem}.merge_stage.json"
    merge_checkpoint_path = shard_dir / f"{output_path.stem}.merge_stage.checkpoint.json"

    merge_output = run_consolidation_on_texts(
        args=merge_args,
        texts=merge_texts,
        output_path=merge_output_path,
        checkpoint_path=merge_checkpoint_path,
        overwrite=args.overwrite,
        progress_desc="llm-merge",
        print_done_summary=False,
    )

    merge_original_to_cause_id = {
        str(text): int(global_id)
        for text, global_id in merge_output["original_to_cause_id"].items()
    }
    global_cause_id_to_canonical = {
        int(global_id): str(canonical)
        for global_id, canonical in merge_output["cause_id_to_canonical"].items()
    }

    local_ref_to_global_id: Dict[Tuple[int, int], int] = {}
    for ref, canonical in ref_to_canonical.items():
        if canonical not in merge_original_to_cause_id:
            raise ValueError(f"Merge stage did not assign local canonical: {canonical!r}")
        local_ref_to_global_id[ref] = merge_original_to_cause_id[canonical]

    merged = _rebuild_global_output(
        shard_outputs=shard_outputs,
        local_ref_to_global_id=local_ref_to_global_id,
        global_cause_id_to_canonical=global_cause_id_to_canonical,
    )
    return merged


def cmd_run_sharded(args: argparse.Namespace) -> None:
    texts = load_input_texts(args)
    output_path = Path(args.output)
    shard_dir = Path(args.shard_dir) if args.shard_dir else _default_shard_dir(output_path)
    shard_dir.mkdir(parents=True, exist_ok=True)

    _print_ollama_config(args)
    print(f"[shard] shard_size={args.shard_size}")
    print(f"[shard] merge_batch_size={args.merge_batch_size}")
    print(f"[shard] work_dir={shard_dir}")

    shard_outputs: List[Dict[str, Any]] = []

    for shard_idx, start, shard_texts in _iter_shards(texts, args.shard_size):
        shard_output = shard_dir / f"shard_{shard_idx:04d}.json"
        shard_checkpoint = shard_dir / f"shard_{shard_idx:04d}.checkpoint.json"
        print(
            f"\n[shard] {shard_idx:04d} "
            f"items={len(shard_texts)} "
            f"range=[{start}, {start + len(shard_texts)})"
        )
        local_output = run_consolidation_on_texts(
            args=args,
            texts=shard_texts,
            output_path=shard_output,
            checkpoint_path=shard_checkpoint,
            overwrite=args.overwrite,
            progress_desc=f"llm-shard-{shard_idx:04d}",
            print_done_summary=False,
        )
        shard_outputs.append(local_output)

    merged = merge_shard_outputs_with_llm(
        args=args,
        shard_outputs=shard_outputs,
        output_path=output_path,
        shard_dir=shard_dir,
    )
    save_clusters_json(merged, output_path)

    print()
    print(quality_report(merged, top_n=args.report_top_n))
    print(f"\nSaved merged output: {output_path}")
    print(f"Shard/checkpoint directory: {shard_dir}")


def cmd_run(args: argparse.Namespace) -> None:
    if args.shard_size and args.shard_size > 0:
        cmd_run_sharded(args)
        return

    texts = load_input_texts(args)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_path.with_name(
        output_path.name + ".checkpoint.json"
    )

    _print_ollama_config(args)
    output = run_consolidation_on_texts(
        args=args,
        texts=texts,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        overwrite=args.overwrite,
        progress_desc="llm-consolidate",
        print_done_summary=False,
    )

    print()
    print(quality_report(output, top_n=args.report_top_n))
    print(f"\nSaved: {output_path}")
    print(f"Checkpoint: {checkpoint_path}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--coco_files", nargs="+", default=DEFAULT_COCO_FILES)
    parser.add_argument("--cache_dir", default=None, help="Read cause strings from cache_dir/texts.json.")
    parser.add_argument("--input_texts", default=None, help="Read one cause string per line from txt.")
    parser.add_argument("--include_healthy", action="store_true")
    parser.add_argument("--max_items", type=int, default=0, help="Debug limit; 0 means all inputs.")

    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path for non-sharded mode. Defaults to '<output>.checkpoint.json'.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing output/checkpoint and start from scratch.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=0,
        help=(
            "If > 0, split inputs into independent local shards, then merge "
            "the shard-level canonical causes. 0 keeps the original one-stage mode."
        ),
    )
    parser.add_argument(
        "--shard_dir",
        default=None,
        help="Directory for shard outputs/checkpoints. Defaults to '<output_stem>.shards'.",
    )
    parser.add_argument(
        "--merge_batch_size",
        type=int,
        default=50,
        help="Batch size for the final canonical-cause merge stage.",
    )

    parser.add_argument(
        "--ollama_host",
        default=DEFAULT_OLLAMA_HOST,
        help="Ollama server base URL.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model name. The model must be available in `ollama list`.",
    )
    parser.add_argument(
        "--keep_alive",
        default="5m",
        help="How long Ollama keeps the model loaded after a request. Use empty string to disable.",
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=600.0,
        help="HTTP request timeout in seconds. Use 0 for no timeout.",
    )
    parser.add_argument(
        "--json_mode",
        action="store_true",
        help="Use Ollama format='json' instead of passing the full JSON Schema.",
    )
    parser.add_argument(
        "--use_generate",
        action="store_true",
        help="Use Ollama /api/generate instead of /api/chat.",
    )
    parser.add_argument(
        "--think",
        choices=["false", "true", "low", "medium", "high"],
        default="false",
        help=(
            "Ollama thinking control. Default false disables thinking for thinking models; "
            "GPT-OSS-style models may accept low/medium/high."
        ),
    )

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_existing_in_prompt", type=int, default=0, help="0 means include all.")
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=3,
        help="Attempts using the original batch size before falling back to batch_size/2.",
    )
    parser.add_argument(
        "--half_batch_attempts",
        type=int,
        default=2,
        help="Attempts using batch_size/2 after original-batch attempts fail.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=65536,
        help="Maximum tokens to generate; mapped to Ollama options.num_predict.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument(
        "--num_ctx",
        type=int,
        default=None,
        help="Optional Ollama context window; mapped to options.num_ctx.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional Ollama seed for reproducibility.",
    )
    parser.add_argument("--save_failed_responses", action="store_true")
    parser.add_argument("--report_top_n", type=int, default=15)
    parser.set_defaults(func=cmd_run)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.merge_batch_size <= 0:
        raise ValueError("--merge_batch_size must be positive")
    if args.shard_size < 0:
        raise ValueError("--shard_size must be >= 0")
    if args.max_attempts <= 0:
        raise ValueError("--max_attempts must be positive")
    if args.half_batch_attempts <= 0:
        raise ValueError("--half_batch_attempts must be positive")
    if args.max_tokens <= 0:
        raise ValueError("--max_tokens must be positive")
    if args.keep_alive == "":
        args.keep_alive = None
    args.func(args)


if __name__ == "__main__":
    main()
