"""Entity-resolution style cause clustering (LLM-as-judge).

This is the successor to the greedy incremental ``cause_cluster_llm.py``. Instead
of streaming causes through a growing "existing canonical" prompt (which silently
under-merges synonyms once the relevant canonical scrolls out of the prompt), it
treats cause consolidation as **entity resolution**:

    Blocking (embedding kNN, high recall)
      -> pairwise LLM judge (same / different + reason, high precision)
      -> medoid-anchored leader clustering (Round 1)
      -> leader-vs-leader merge on a looser threshold (Round 2)

Every membership is backed by exactly one explicit ``member vs leader`` verdict,
so the whole partition is reproducible and auditable per pair (no transitive
closure, no LLM text generation). The cluster representative is the **medoid**
real string (reused from ``cause_cluster_json.assign_cause_ids`` / ``_pick_canonical``).

Recall is decoupled from precision: embedding kNN proposes candidates (loose
threshold, never decides), the LLM decides. The pure-embedding baseline is
available via ``--judge cosine`` (same iff cosine >= threshold) for ablation and
for running the full pipeline without Ollama.

Input is a case_db's ``cause_text_embs.pt`` (``texts`` + ``embeddings``); the
output is the same 3-key schema as the old clusterer.

Prerequisite for the LLM backend:
    ollama serve
    ollama pull <model>

Examples:
    # smoke test / pure-embedding baseline (no Ollama). NOTE these embeddings are
    # anisotropic (random-pair cosine median ~0.90), so the cosine baseline needs a
    # high threshold and still over-merges -- that is the point of the LLM judge.
    python -m diagnosis_model.cause_inference.preprocessing.cause_resolve_llm \
        --case_db data/processed/current/artifacts/db/case_db_jointDistRawP \
        --judge cosine --judge_cosine_tau 0.95 --max_items 3000

    # production LLM-judge run
    python -m diagnosis_model.cause_inference.preprocessing.cause_resolve_llm \
        --case_db data/processed/current/artifacts/db/case_db_jointDistRawP \
        --judge ollama --model gemma4:26b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from diagnosis_model.cause_inference.preprocessing.cause_cluster_json import (  # type: ignore
        assign_cause_ids,
        quality_report,
        save_clusters_json,
    )
    from diagnosis_model.cause_inference.preprocessing.cause_cluster_llm import (  # type: ignore
        _atomic_write_json,
        _normalize_ollama_host,
        _ollama_post,
    )
else:
    from .cause_cluster_json import assign_cause_ids, quality_report, save_clusters_json
    from .cause_cluster_llm import _atomic_write_json, _normalize_ollama_host, _ollama_post


DEFAULT_CASE_DB = "data/processed/current/artifacts/db/case_db_jointDistRawP"
DEFAULT_OUTPUT = "data/processed/current/artifacts/cause_clusters_llm.json"
DEFAULT_MODEL = "gemma4:26b"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


JUDGE_SYSTEM_PROMPT = """You are a fish-disease pathology expert performing entity \
resolution on free-text disease CAUSE statements.

For each pair you decide whether statement A and statement B refer to the SAME \
PRIMARY CAUSE.

Judge at the level of the primary etiological cause / mechanism. Treat the \
following as ILLUSTRATIVE detail, not as defining the cause -- if two statements \
share the same primary cause but differ only in these, answer same = true:
  - which specific pathogen species is named (e.g. Aeromonas vs Flavobacterium
    columnare are both "bacterial infection" -> same);
  - which specific water parameter is elevated (e.g. high ammonia vs high
    nitrate are both "poor water quality" -> same);
  - which specific injury scenario is described (e.g. netting vs fighting vs
    scraping are all "physical trauma" -> same);
  - paraphrase, language, or one wording being more specific/general.

same = false when the PRIMARY cause differs. These are distinct primary causes \
and must stay apart even at very high textual similarity:
  bacterial vs fungal vs viral vs parasitic infection; infection vs water-quality \
vs physical-trauma vs nutritional deficiency vs organ failure vs environmental \
(low oxygen, toxins). Also answer false when one statement is merely a symptom / \
downstream effect rather than the cause.

Return raw JSON only. No markdown, no code fences, no commentary."""


JUDGE_USER_TEMPLATE = """Judge each pair independently.

Pairs:
{pairs_json}

Return exactly this JSON shape, one verdict per pair_id, nothing else:
{{
  "verdicts": [
    {{"pair_id": 1, "same": true, "reason": "short justification"}},
    {{"pair_id": 2, "same": false, "reason": "short justification"}}
  ]
}}
"""


JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pair_id": {"type": "integer", "minimum": 1},
                    "same": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["pair_id", "same", "reason"],
            },
        }
    },
    "required": ["verdicts"],
}


# --------------------------------------------------------------------------- #
# Input / fingerprint
# --------------------------------------------------------------------------- #
def load_texts_embeddings(case_db: str, max_items: int) -> Tuple[List[str], torch.Tensor]:
    path = Path(case_db) / "cause_text_embs.pt"
    payload = torch.load(path, weights_only=False)
    texts = [str(t) for t in payload["texts"]]
    embeddings = payload["embeddings"].float()
    if max_items and max_items > 0:
        texts = texts[:max_items]
        embeddings = embeddings[:max_items]
    embeddings = F.normalize(embeddings, dim=-1)
    if embeddings.size(0) != len(texts):
        raise ValueError("texts and embeddings length mismatch")
    print(f"[input] {len(texts)} cause strings, emb dim {embeddings.size(1)} from {path}")
    return texts, embeddings


def _fingerprint(texts: Sequence[str], judge_sig: str) -> str:
    """Cache fingerprint over inputs AND the judge configuration.

    Including the judge backend + system prompt (or the cosine threshold) means a
    prompt edit auto-invalidates stale verdicts instead of silently reusing them.
    """
    h = hashlib.sha256()
    h.update(judge_sig.encode("utf-8"))
    h.update(b"\x00")
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _judge_signature(args: argparse.Namespace) -> str:
    if args.judge == "cosine":
        return f"cosine|{args.judge_cosine_tau}"
    return f"ollama|{args.model}|{JUDGE_SYSTEM_PROMPT}"


# --------------------------------------------------------------------------- #
# Blocking: embedding kNN candidate graph
# --------------------------------------------------------------------------- #
def knn_candidates(
    embeddings: torch.Tensor,
    k: int,
    tau: float,
    chunk: int = 2048,
) -> Tuple[List[List[Tuple[int, float]]], np.ndarray]:
    """Return per-node neighbors with cosine >= tau (top-k), and a density score.

    neighbors[i] is a cosine-descending list of (j, cos) with j != i, cos >= tau.
    density[i] is the sum of the (clamped) top-k cosines, used as the leader
    ordering key (denser nodes anchor clusters first).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = embeddings.to(device)
    n = emb.size(0)
    k_eff = min(k, max(1, n - 1))
    neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    density = np.zeros(n, dtype=np.float64)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = emb[start:end] @ emb.t()  # [c, n]
        rows = torch.arange(end - start, device=device)
        sims[rows, torch.arange(start, end, device=device)] = -1.0  # drop self
        topv, topi = sims.topk(k_eff, dim=1)
        density[start:end] = topv.clamp(min=0).sum(dim=1).cpu().numpy()
        topv_cpu = topv.cpu().tolist()
        topi_cpu = topi.cpu().tolist()
        for r in range(end - start):
            row = [
                (int(j), float(v))
                for j, v in zip(topi_cpu[r], topv_cpu[r])
                if v >= tau
            ]
            neighbors[start + r] = row
    return neighbors, density


# --------------------------------------------------------------------------- #
# LLM / cosine judge with a persistent pair cache
# --------------------------------------------------------------------------- #
class JudgeCache:
    """Persistent, fingerprint-guarded cache of pairwise verdicts.

    Key is "i-j" with i < j over global cause indices. Values store the verdict
    plus reason and cosine for the audit trail.
    """

    def __init__(self, path: Path, fingerprint: str, flush_every: int = 200):
        self.path = path
        self.fingerprint = fingerprint
        self.flush_every = flush_every
        self.pairs: Dict[str, Dict[str, Any]] = {}
        self._dirty = 0
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                blob = json.load(f)
            if blob.get("fingerprint") == fingerprint:
                self.pairs = blob.get("pairs", {})
                print(f"[cache] loaded {len(self.pairs)} cached verdicts from {path}")
            else:
                print(f"[cache] fingerprint mismatch in {path}; ignoring stale cache")

    @staticmethod
    def key(i: int, j: int) -> str:
        lo, hi = (i, j) if i < j else (j, i)
        return f"{lo}-{hi}"

    def get(self, i: int, j: int) -> Dict[str, Any] | None:
        return self.pairs.get(self.key(i, j))

    def put(self, i: int, j: int, same: bool, reason: str, cos: float) -> None:
        self.pairs[self.key(i, j)] = {
            "same": bool(same),
            "reason": reason,
            "cos": round(float(cos), 4),
        }
        self._dirty += 1
        if self._dirty >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        _atomic_write_json({"fingerprint": self.fingerprint, "pairs": self.pairs}, self.path)
        self._dirty = 0


def _ollama_judge_batch(
    args: argparse.Namespace,
    batch: List[Tuple[int, int, int, float]],
    texts: Sequence[str],
) -> Dict[int, bool]:
    """Judge a batch of pairs in one Ollama call. batch items: (pair_id, i, j, cos)."""
    pairs_json = json.dumps(
        [{"pair_id": pid, "A": texts[i], "B": texts[j]} for pid, i, j, _ in batch],
        ensure_ascii=False,
        indent=2,
    )
    prompt = JUDGE_USER_TEMPLATE.format(pairs_json=pairs_json)
    payload = {
        "model": args.model,
        "stream": False,
        "think": False,
        "format": JUDGE_SCHEMA,
        "options": {"temperature": 0.0, "top_p": 1.0, "num_predict": args.max_tokens},
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if args.keep_alive:
        payload["keep_alive"] = args.keep_alive
    timeout = None if args.request_timeout <= 0 else args.request_timeout
    response = _ollama_post(args.ollama_host, "/api/chat", payload, timeout)
    content = response.get("message", {}).get("content", "")
    parsed = json.loads(content)
    verdicts = {int(v["pair_id"]): bool(v["same"]) for v in parsed["verdicts"]}
    expected = {pid for pid, _, _, _ in batch}
    if set(verdicts) != expected:
        raise ValueError(f"verdict pair_id mismatch: expected {sorted(expected)}, got {sorted(verdicts)}")
    return verdicts


def judge_pairs(
    args: argparse.Namespace,
    pairs: List[Tuple[int, int, float]],
    texts: Sequence[str],
    cache: JudgeCache,
) -> Dict[Tuple[int, int], bool]:
    """Resolve same/different for each (i, j, cos), using cache + chosen backend."""
    result: Dict[Tuple[int, int], bool] = {}
    todo: List[Tuple[int, int, float]] = []
    for i, j, cos in pairs:
        hit = cache.get(i, j)
        if hit is not None:
            result[(i, j)] = bool(hit["same"])
        else:
            todo.append((i, j, cos))

    if not todo:
        return result

    if args.judge == "cosine":
        for i, j, cos in todo:
            same = cos >= args.judge_cosine_tau
            cache.put(i, j, same, "cosine>=tau" if same else "cosine<tau", cos)
            result[(i, j)] = same
        return result

    # ollama backend, batched
    for start in range(0, len(todo), args.pairs_per_call):
        sub = todo[start : start + args.pairs_per_call]
        batch = [(pid + 1, i, j, cos) for pid, (i, j, cos) in enumerate(sub)]
        verdicts: Dict[int, bool] | None = None
        last_err: Exception | None = None
        for _ in range(args.max_attempts):
            try:
                verdicts = _ollama_judge_batch(args, batch, texts)
                break
            except Exception as exc:  # noqa: BLE001 - retry any malformed response
                last_err = exc
        if verdicts is None:
            cache.flush()
            raise SystemExit(
                f"LLM judge failed after {args.max_attempts} attempts; last error: {last_err}. "
                f"Cache preserved at {cache.path}"
            )
        for pid, i, j, cos in batch:
            same = bool(verdicts[pid])
            reason = "llm:same" if same else "llm:different"
            cache.put(i, j, same, reason, cos)
            result[(i, j)] = same
    return result


# --------------------------------------------------------------------------- #
# Leader clustering on an index subset
# --------------------------------------------------------------------------- #
def resolve_round(
    args: argparse.Namespace,
    global_indices: List[int],
    emb_subset: torch.Tensor,
    texts: Sequence[str],
    cache: JudgeCache,
    tau: float,
    label: str,
) -> Dict[int, int]:
    """Medoid-anchored leader clustering over a subset.

    Returns global_index -> global_leader_index. Each non-leader's assignment is
    backed by one "member vs leader" verdict; leaders map to themselves.
    """
    neighbors, density = knn_candidates(emb_subset, args.k, tau)
    order = sorted(range(len(global_indices)), key=lambda loc: (-density[loc], loc))

    is_leader: set[int] = set()  # local indices
    assign_local: Dict[int, int] = {}  # local -> local leader

    pbar = _progress(len(order), f"{label} (tau={tau})")
    for loc in order:
        cand = [(nj, cos) for nj, cos in neighbors[loc] if nj in is_leader]
        cand.sort(key=lambda t: -t[1])
        cand = cand[: args.pairs_per_call]  # cap fan-out; round 2 mops up the rest

        chosen: int | None = None
        if cand:
            gi = global_indices[loc]
            pair_list = [(gi, global_indices[nj], cos) for nj, cos in cand]
            verdicts = judge_pairs(args, pair_list, texts, cache)
            for nj, cos in cand:  # cosine-descending: first "same" wins
                if verdicts[(gi, global_indices[nj])]:
                    chosen = nj
                    break

        if chosen is None:
            is_leader.add(loc)
            assign_local[loc] = loc
        else:
            assign_local[loc] = chosen
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()

    return {global_indices[loc]: global_indices[lead] for loc, lead in assign_local.items()}


def _progress(total: int, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="cause")
    except Exception:  # pragma: no cover - optional dependency
        return None


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> Dict[str, Any]:
    texts, embeddings = load_texts_embeddings(args.case_db, args.max_items)
    fingerprint = _fingerprint(texts, _judge_signature(args))
    output_path = Path(args.output)
    cache_path = output_path.with_name(output_path.name + ".judge_cache.json")
    cache = JudgeCache(cache_path, fingerprint)

    print(f"[judge] backend={args.judge}" + (f" model={args.model}" if args.judge == "ollama" else ""))

    # Round 1: full set at tau
    all_indices = list(range(len(texts)))
    leader1 = resolve_round(
        args, all_indices, embeddings, texts, cache, tau=args.tau, label="round1"
    )
    leaders = sorted(set(leader1.values()))
    print(f"[round1] {len(leaders)} leaders from {len(texts)} causes")

    # Round 2: merge leaders on a looser threshold
    super_leader = {L: L for L in leaders}
    if args.merge_rounds > 0 and len(leaders) > 1:
        leader_emb = embeddings[torch.tensor(leaders, dtype=torch.long)]
        super_leader = resolve_round(
            args, leaders, leader_emb, texts, cache, tau=args.tau_merge, label="round2"
        )
        n_super = len(set(super_leader.values()))
        print(f"[round2] {len(leaders)} leaders -> {n_super} clusters")

    cache.flush()

    # Compose final leader per cause, then contiguous labels
    final_leader = {i: super_leader[leader1[i]] for i in all_indices}
    label_of: Dict[int, int] = {}
    cluster_labels = np.empty(len(texts), dtype=np.int64)
    for i in all_indices:
        fl = final_leader[i]
        if fl not in label_of:
            label_of[fl] = len(label_of)
        cluster_labels[i] = label_of[fl]

    clusters = assign_cause_ids(texts, embeddings, cluster_labels)
    save_clusters_json(clusters, output_path)
    _dump_audit(cache, texts, output_path)

    print()
    print(quality_report(clusters, top_n=args.report_top_n))
    print(f"\nSaved: {output_path}")
    print(f"Cache: {cache_path}")
    print(f"Audit: {output_path.with_name(output_path.name + '.judgments.jsonl')}")
    return clusters


def _dump_audit(cache: JudgeCache, texts: Sequence[str], output_path: Path) -> None:
    audit_path = output_path.with_name(output_path.name + ".judgments.jsonl")
    with audit_path.open("w", encoding="utf-8") as f:
        for key, rec in cache.pairs.items():
            i, j = (int(x) for x in key.split("-"))
            f.write(
                json.dumps(
                    {
                        "a": texts[i],
                        "b": texts[j],
                        "cos": rec["cos"],
                        "same": rec["same"],
                        "reason": rec["reason"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--case_db", default=DEFAULT_CASE_DB,
                   help="case_db dir containing cause_text_embs.pt (texts + embeddings).")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--max_items", type=int, default=0, help="Debug limit; 0 means all.")

    p.add_argument("--judge", choices=["ollama", "cosine"], default="ollama",
                   help="cosine = pure-embedding baseline (same iff cosine>=tau), no Ollama.")
    p.add_argument("--judge_cosine_tau", type=float, default=0.78,
                   help="Decision threshold for the cosine judge backend.")

    p.add_argument("--k", type=int, default=32,
                   help="kNN fan-out for blocking; this (not tau) bounds candidate cost per node.")
    p.add_argument("--tau", type=float, default=0.90,
                   help="Round-1 candidate cosine floor. These cause embeddings are highly "
                        "anisotropic (random-pair cosine median ~0.90, true synonyms ~0.95+), so "
                        "the floor sits high; the LLM judge disambiguates the 0.90-0.99 band.")
    p.add_argument("--tau_merge", type=float, default=0.88,
                   help="Round-2 leader-merge cosine floor (looser; small leader set).")
    p.add_argument("--merge_rounds", type=int, default=1, help="0 disables Round-2 leader merge.")

    p.add_argument("--ollama_host", default=DEFAULT_OLLAMA_HOST)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--keep_alive", default="5m")
    p.add_argument("--request_timeout", type=float, default=600.0)
    p.add_argument("--pairs_per_call", type=int, default=20)
    p.add_argument("--max_attempts", type=int, default=3)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--report_top_n", type=int, default=15)
    p.set_defaults(func=run)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")
    if args.pairs_per_call <= 0:
        raise ValueError("--pairs_per_call must be positive")
    if args.max_attempts <= 0:
        raise ValueError("--max_attempts must be positive")
    if args.keep_alive == "":
        args.keep_alive = None
    args.func(args)


if __name__ == "__main__":
    main()
