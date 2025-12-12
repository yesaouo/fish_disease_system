from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from .config import DIM, KB_VERSION
from .text_encoder import EmbeddingGemma


@dataclass
class KBEntry:
    _id: str
    desc: str
    safety_note: str | None = None


class BruteForceIndex:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.vecs: np.ndarray | None = None  # (N,dim)
        self.ids: List[str] = []
        self.descs: List[str] = []
        self.notes: List[str | None] = []

    def add(self, ids: Iterable[str], vecs: np.ndarray, descs: Iterable[str], notes: Iterable[str | None]) -> None:
        assert vecs.ndim == 2 and vecs.shape[1] == self.dim
        if self.vecs is None:
            self.vecs = vecs.astype(np.float32)
        else:
            self.vecs = np.concatenate([self.vecs, vecs.astype(np.float32)], axis=0)
        self.ids.extend(list(ids))
        self.descs.extend(list(descs))
        self.notes.extend(list(notes))

    def search(self, q: np.ndarray, topk: int = 1) -> List[Tuple[int, float]]:
        if self.vecs is None or self.vecs.size == 0:
            return []
        q = q.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        M = self.vecs / (np.linalg.norm(self.vecs, axis=1, keepdims=True) + 1e-8)
        sims = M @ q  # cosine
        idx = np.argsort(sims)[::-1][:topk]
        return [(int(i), float(sims[i])) for i in idx]


class KnowledgeBase:
    def __init__(self, encoder: EmbeddingGemma | None = None, kb_dir: str | Path | None = None) -> None:
        self.encoder = encoder or EmbeddingGemma()
        self.kb_dir = Path(kb_dir or Path(__file__).parent / "knowledge_base")
        self.cause_index = BruteForceIndex(DIM)
        self.treat_index = BruteForceIndex(DIM)
        self.version = KB_VERSION
        self._load()

    def _load_jsonl(self, p: Path) -> List[KBEntry]:
        items: List[KBEntry] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(KBEntry(_id=obj["id"], desc=obj["desc"], safety_note=obj.get("safety_note")))
        return items

    def _load_embeddings(self, texts: List[str], cache_path: Path, source_mtime: float | None = None) -> np.ndarray:
        """Load cached embeddings if available and valid; otherwise encode and save."""
        if cache_path.exists():
            try:
                if source_mtime and cache_path.stat().st_mtime < source_mtime:
                    raise ValueError("stale cache")
                arr = np.load(cache_path)
                if arr.shape == (len(texts), DIM):
                    return arr.astype(np.float32)
            except Exception:
                pass  # Fallback to re-encode on any issue

        vecs = np.stack([self.encoder.encode(t) for t in texts], axis=0).astype(np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, vecs)
        return vecs

    def _load(self) -> None:
        causes_path = self.kb_dir / "causes.jsonl"
        actions_path = self.kb_dir / "actions.jsonl"

        causes = self._load_jsonl(causes_path)
        actions = self._load_jsonl(actions_path)

        # Encode entries using the same embedding space (with disk cache to avoid recompute)
        cause_vecs = self._load_embeddings(
            [e.desc for e in causes],
            self.kb_dir / "cause_embeddings.npy",
            source_mtime=causes_path.stat().st_mtime,
        )
        self.cause_index.add([e._id for e in causes], cause_vecs, [e.desc for e in causes], [e.safety_note for e in causes])

        action_vecs = self._load_embeddings(
            [e.desc for e in actions],
            self.kb_dir / "action_embeddings.npy",
            source_mtime=actions_path.stat().st_mtime,
        )
        self.treat_index.add([e._id for e in actions], action_vecs, [e.desc for e in actions], [e.safety_note for e in actions])

    def search_causes(self, queries: np.ndarray, topk: int = 1) -> List[Tuple[str, str, float]]:
        # returns list of (id, desc, sim) for all slots combined (with duplicates)
        out: List[Tuple[str, str, float]] = []
        for i in range(queries.shape[0]):
            for idx, sim in self.cause_index.search(queries[i], topk=topk):
                out.append((self.cause_index.ids[idx], self.cause_index.descs[idx], sim))
        return out

    def search_actions(self, queries: np.ndarray, topk: int = 1) -> List[Tuple[str, str, float, str | None]]:
        out: List[Tuple[str, str, float, str | None]] = []
        for i in range(queries.shape[0]):
            for idx, sim in self.treat_index.search(queries[i], topk=topk):
                out.append(
                    (
                        self.treat_index.ids[idx],
                        self.treat_index.descs[idx],
                        sim,
                        self.treat_index.notes[idx],
                    )
                )
        return out
