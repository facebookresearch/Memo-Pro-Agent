"""Loader for Contract B -- the built memory graph (see ``properagent.data.schema``).

One :class:`MemoryStore` per UID. The store owns the on-disk artifacts and the
invariants between them: the three ``.npy`` matrices are aligned to their JSON
counterparts by row order, which is exactly the kind of coupling that breaks
silently, so it is checked on load rather than trusted.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..backends.base import l2_normalize

MANIFEST_SCHEMA = "properagent_memory_v1"


class MemoryError_(RuntimeError):
    """Raised when memory artifacts are missing or mutually inconsistent."""


@dataclass(frozen=True)
class MemoryPaths:
    episodic: Path
    semantic: Path
    passages: Path
    entity_list: Path
    entity_emb: Path
    semantic_emb: Path
    passage_emb: Path
    manifest: Path

    @classmethod
    def for_uid(cls, root: str | Path, uid: str) -> "MemoryPaths":
        base = Path(root).expanduser().resolve()
        return cls(
            episodic=base / f"{uid}.episodic.json",
            semantic=base / f"{uid}.semantic.json",
            passages=base / f"{uid}.passages.json",
            entity_list=base / f"{uid}.entity_list.json",
            entity_emb=base / f"{uid}.entity_emb.npy",
            semantic_emb=base / f"{uid}.semantic_emb.npy",
            passage_emb=base / f"{uid}.passages_emb.npy",
            manifest=base / f"{uid}.manifest.json",
        )

    def required(self) -> list[Path]:
        return [self.episodic, self.passages, self.entity_list, self.entity_emb]

    def missing(self) -> list[Path]:
        return [p for p in self.required() if not p.is_file()]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class MemoryStore:
    """The built memory graph for one UID."""

    def __init__(
        self,
        uid: str,
        root: str | Path,
        caption_log: str | Path | None = None,
        verify_hashes: bool = False,
    ) -> None:
        self.uid = uid
        self.root = Path(root).expanduser().resolve()
        self.paths = MemoryPaths.for_uid(self.root, uid)

        missing = self.paths.missing()
        if missing:
            raise MemoryError_(
                f"{uid}: missing memory artifacts {[p.name for p in missing]} under {self.root}. "
                "Build them with `properagent-build-memory`."
            )

        self.manifest = self._load_manifest(verify_hashes)
        episodic = _read_json(self.paths.episodic)
        self.events: list[dict[str, Any]] = list(episodic.get("events", []))
        if not self.events:
            raise MemoryError_(f"{uid}: episodic graph has no events")

        self.entity_list: list[str] = _read_json(self.paths.entity_list)
        self.entity_emb = l2_normalize(np.load(self.paths.entity_emb))
        _require_aligned(uid, "entity", len(self.entity_list), len(self.entity_emb))

        self.passages: list[str] = _read_json(self.paths.passages)
        self.passage_emb = (
            l2_normalize(np.load(self.paths.passage_emb))
            if self.paths.passage_emb.is_file()
            else None
        )
        if self.passage_emb is not None:
            _require_aligned(uid, "passage", len(self.passages), len(self.passage_emb))

        self.semantic_triples: list[list[str]] = []
        self.semantic_emb: np.ndarray | None = None
        if self.paths.semantic.is_file():
            self.semantic_triples = _read_json(self.paths.semantic).get("triples", [])
            if self.paths.semantic_emb.is_file():
                self.semantic_emb = l2_normalize(np.load(self.paths.semantic_emb))
                _require_aligned(
                    uid, "semantic", len(self.semantic_triples), len(self.semantic_emb)
                )

        self._attach_recency(caption_log)
        self.passage_eids = self._map_passages_to_events()

    # ---- invariants ----

    def _load_manifest(self, verify_hashes: bool) -> dict[str, Any]:
        if not self.paths.manifest.is_file():
            # Artifacts built before manifests existed still load; the embedding
            # signature check is then skipped and reported as unknown.
            return {"schema": MANIFEST_SCHEMA, "embedding_signature": None}
        manifest = _read_json(self.paths.manifest)
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise MemoryError_(
                f"{self.uid}: manifest schema {manifest.get('schema')!r}, expected {MANIFEST_SCHEMA!r}"
            )
        if verify_hashes:
            for name, expected in (manifest.get("sha256") or {}).items():
                path = self.root / name
                if not path.is_file():
                    raise MemoryError_(f"{self.uid}: manifest lists missing file {name}")
                actual = sha256_file(path)
                if actual != expected:
                    raise MemoryError_(
                        f"{self.uid}: {name} changed since it was built "
                        f"(sha256 {actual[:12]} != {expected[:12]})"
                    )
        return manifest

    def check_embedding_signature(self, signature: str) -> None:
        """Fail loudly on an embedding-space mismatch.

        Querying a bge-built index with a different model returns plausible-looking
        nearest neighbours that are semantically unrelated, so retrieval degrades
        without ever raising. This is the check that catches it.
        """
        recorded = self.manifest.get("embedding_signature")
        if recorded is None:
            return
        if recorded != signature:
            raise MemoryError_(
                f"{self.uid}: memory was built with embedding {recorded!r} but the configured "
                f"backend is {signature!r}. Rebuild the memory or switch the backend."
            )

    # ---- derived fields ----

    def _attach_recency(self, caption_log: str | Path | None) -> None:
        """Give every event a ``days_ago``, the only recency signal retrieval has.

        Prefer the value stored at build time; fall back to reading it off the
        caption log for artifacts built before that was persisted.
        """
        if all(event.get("days_ago") is not None for event in self.events):
            return
        cid_to_days: dict[str, Any] = {}
        if caption_log and Path(caption_log).is_file():
            with Path(caption_log).open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    cid_to_days[row["cid"]] = row.get("t_days_before")
        for event in self.events:
            if event.get("days_ago") is not None:
                continue
            cids = event.get("cids") or []
            event["days_ago"] = cid_to_days.get(cids[0]) if cids else None

    def _map_passages_to_events(self) -> list[int | None]:
        """Map each passage row back to its event id.

        ``passages.json`` is the event rewrites in event order, so the common case
        is a positional match; the text index is the fallback for artifacts whose
        passage list was filtered.
        """
        if len(self.passages) == len(self.events):
            return [int(event["idx"]) for event in self.events]
        by_text = {str(event.get("rewrite", "")): int(event["idx"]) for event in self.events}
        return [by_text.get(passage) for passage in self.passages]

    # ---- stats ----

    def summary(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "events": len(self.events),
            "triples": sum(len(e.get("triples", [])) for e in self.events),
            "entities": len(self.entity_list),
            "semantic": len(self.semantic_triples),
            "passages": len(self.passages),
            "embedding_signature": self.manifest.get("embedding_signature"),
        }


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise MemoryError_(f"malformed JSON in {path}: {error}") from error


def _require_aligned(uid: str, name: str, n_json: int, n_rows: int) -> None:
    if n_json != n_rows:
        raise MemoryError_(
            f"{uid}: {name} embeddings have {n_rows} rows but the JSON has {n_json} entries. "
            "The two are aligned by row order; rebuild this UID's memory."
        )
