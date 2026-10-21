# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Retrieval over one clip's memory graph.

Given the object(s) the policy just named, return the memory worth showing the
gate. Two paths run in parallel, as described in the paper:

* **one-hop** -- embed-match the object to entity nodes, then pull the episodic
  triples hanging off those nodes plus semantic facts reached by entity link or
  direct similarity. Hub-immune and always on.
* **personalized PageRank** (``mode="slow"``) -- propagate the same entity seeds
  over the graph to reach event passages that never mention the object by name.

Both paths carry timestamps, because recency is part of whether an open loop is
still open. The output is rendered by :meth:`Retriever.render_for_gate` into the
fact block the gate reads.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..backends.base import EmbeddingBackend, l2_normalize
from .openloop import GENERIC_ENTITIES, HUB_ENTITIES, is_open_text, is_open_triple
from .store import MemoryStore


@dataclass(frozen=True)
class RetrievalConfig:
    """Defaults are the configuration used for the reported results."""

    mode: str = "slow"
    """``fast`` = one-hop only; ``slow`` = one-hop + personalized PageRank."""

    seed_threshold: float = 0.55
    """Minimum cosine for an entity to seed retrieval. Below this the object was
    not really mentioned in memory, and seeding anyway invents associations."""

    seeds_per_part: int = 8
    max_episodic: int = 60
    max_semantic: int = 6
    semantic_threshold: float = 0.50
    max_passages: int = 4
    ppr_events: int = 5
    ppr_damping: float = 0.5

    open_loop_bonus: float = 0.60
    """Added to a fact's score when it states an open loop. Large enough to lift a
    year-old open loop over a fresh routine fact, small enough that a weak entity
    match cannot be rescued by it alone."""

    recency_weight: float = 0.35
    recency_horizon_days: float = 365.0

    event_expand: bool = True
    """Pull in open-loop sibling facts from the best-ranked events. A constraint is
    often recorded on a neighbouring entity ("she flinched") rather than on the
    object the policy named ("cat")."""

    passage_vector_topk: int = 2
    """Direct passage-level vector recall, unioned with the graph paths."""

    synonym_threshold: float = 0.83
    synonym_neighbors: int = 6
    hub_degree: int = 25
    hub_penalty: float = 0.05
    semantic_link_weight: float = 1.5

    gate_max_episodic_lines: int = 24
    gate_passage_chars: int = 600


@dataclass
class RetrievalResult:
    query: str
    mode: str
    seeds: list[tuple[str, float]] = field(default_factory=list)
    episodic: list[dict[str, Any]] = field(default_factory=list)
    semantic: list[dict[str, Any]] = field(default_factory=list)
    passages: list[dict[str, Any]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.episodic or self.semantic or self.passages)

    def to_json(self, max_episodic: int = 24) -> dict[str, Any]:
        """Compact form for traces. Full passages are truncated."""
        return {
            "query": self.query,
            "mode": self.mode,
            "seeds": [[name, round(score, 3)] for name, score in self.seeds[:5]],
            "semantic": self.semantic[:6],
            "passages": [
                {
                    "eid": p.get("eid"),
                    "days_ago": p.get("days_ago"),
                    "passage": str(p.get("passage", ""))[:600],
                }
                for p in self.passages
            ],
            "episodic": [
                {"triple": e.get("triple"), "days_ago": e.get("days_ago")}
                for e in self.episodic[:max_episodic]
            ],
        }


class Retriever:
    """Retrieval over one UID's memory graph.

    Construction builds the in-memory graph index, which costs a second or two for
    a year-long history; reuse one instance for all decisions on a clip.
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: EmbeddingBackend,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        store.check_embedding_signature(embedder.signature)

        self.uid = store.uid
        self.entities = store.entity_list
        self.entity_emb = store.entity_emb
        self.semantic_triples = store.semantic_triples
        self.semantic_emb = store.semantic_emb

        self._flatten_episodic()
        self._build_graph_index()
        self._link_semantic()
        self._graph = None
        self._synonyms_added = False

    # ---- index construction ----

    def _flatten_episodic(self) -> None:
        """Flatten events into a fact table with provenance back to the event."""
        self.facts: list[tuple[str, str, str, Any, Any, Any, int]] = []
        self.events: dict[int, dict[str, Any]] = {}
        self.node_to_facts: dict[str, list[int]] = defaultdict(list)
        self.event_to_facts: dict[int, list[int]] = defaultdict(list)

        for event in self.store.events:
            eid = int(event["idx"])
            triples = [
                tuple(t)
                for t in event.get("triples", [])
                if len(t) == 3 and all(isinstance(x, str) and x.strip() for x in t)
            ]
            self.events[eid] = {
                "passage": event.get("rewrite", ""),
                "day": event.get("day"),
                "clock": event.get("start"),
                "days_ago": event.get("days_ago"),
                "triples": triples,
            }
            for subject, predicate, obj in triples:
                index = len(self.facts)
                self.facts.append(
                    (
                        subject,
                        predicate,
                        obj,
                        event.get("day"),
                        event.get("start"),
                        event.get("days_ago"),
                        eid,
                    )
                )
                self.event_to_facts[eid].append(index)
                for field_value in (subject, obj):
                    # Long phrases are not entities; indexing them bloats the table
                    # and never matches an Observe output anyway.
                    if len(field_value.split()) <= 4:
                        self.node_to_facts[field_value.strip().lower()].append(index)

    def _build_graph_index(self) -> None:
        self.nodes = sorted(
            {
                value
                for fact in self.facts
                for value in (fact[0].lower(), fact[2].lower())
                if value not in HUB_ENTITIES
            }
        )
        self.node_id = {name: i for i, name in enumerate(self.nodes)}
        self.edges: dict[tuple[int, int], float] = defaultdict(float)
        for subject, _, obj, *_ in self.facts:
            a, b = subject.lower(), obj.lower()
            if a in self.node_id and b in self.node_id and a != b:
                key = (min(self.node_id[a], self.node_id[b]), max(self.node_id[a], self.node_id[b]))
                self.edges[key] += 1.0
        self.event_nodes: dict[int, set[int]] = defaultdict(set)
        for subject, _, obj, _, _, _, eid in self.facts:
            for value in (subject.lower(), obj.lower()):
                if value in self.node_id:
                    self.event_nodes[eid].add(self.node_id[value])

    def _link_semantic(self) -> None:
        """Put the semantic layer in the same graph as the episodic one.

        A standing fact is free text, so it is linked by entity *mention*: every
        1-3 word n-gram that matches a known node becomes a link, and entities that
        exist only in the semantic layer are added as new nodes. Co-mentioned
        entities are then joined, which gives PageRank a hop the episodic graph does
        not have -- dog -> door through "I keep exterior doors shut because of my
        small dog".
        """
        self.semantic_node_to_facts: dict[str, list[int]] = defaultdict(list)
        self.semantic_nodes: dict[int, list[int]] = defaultdict(list)
        if not self.semantic_triples:
            return
        for index, triple in enumerate(self.semantic_triples):
            tokens = re.findall(r"[a-z0-9']+", " ".join(triple).lower())
            found: set[str] = set()
            for size in (3, 2, 1):
                for start in range(len(tokens) - size + 1):
                    gram = " ".join(tokens[start : start + size])
                    if gram in HUB_ENTITIES or gram in GENERIC_ENTITIES or len(gram) < 3:
                        continue
                    if gram in self.node_id:
                        found.add(gram)
            for field_value in (triple[0], triple[2]):
                name = field_value.strip().lower()
                if (
                    name
                    and name not in HUB_ENTITIES
                    and len(name.split()) <= 4
                    and name not in self.node_id
                ):
                    self.node_id[name] = len(self.nodes)
                    self.nodes.append(name)
                    found.add(name)
            for gram in found:
                self.semantic_node_to_facts[gram].append(index)
                self.semantic_nodes[index].append(self.node_id[gram])
            ids = sorted({self.node_id[g] for g in found})
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    key = (ids[i], ids[j])
                    self.edges[key] = max(self.edges.get(key, 0.0), self.config.semantic_link_weight)

    def _add_synonym_edges(self, block: int = 512) -> None:
        """Embedding-synonymy edges, so "milk carton" reaches "carton of milk".
        Only needed for PageRank, so it is deferred until the graph is built."""
        if self._synonyms_added:
            return
        lowered = [e.lower() for e in self.entities]
        emb = self.entity_emb
        for start in range(0, len(self.entities), block):
            sims = emb[start : start + block] @ emb.T
            for row in range(sims.shape[0]):
                i = start + row
                if lowered[i] not in self.node_id:
                    continue
                top = np.argpartition(-sims[row], min(self.config.synonym_neighbors, len(lowered) - 1))
                for j in top[: self.config.synonym_neighbors]:
                    if j == i or sims[row, j] < self.config.synonym_threshold:
                        continue
                    other = lowered[j]
                    if other in self.node_id and lowered[i] != other:
                        key = (
                            min(self.node_id[lowered[i]], self.node_id[other]),
                            max(self.node_id[lowered[i]], self.node_id[other]),
                        )
                        self.edges[key] = max(self.edges.get(key, 0.0), float(sims[row, j]))
        self._synonyms_added = True

    def _graph_penalty(self, node_index: int) -> float:
        if self.degree[node_index] > self.config.hub_degree:
            return self.config.hub_penalty
        if self.nodes[node_index] in GENERIC_ENTITIES:
            return self.config.hub_penalty
        return 1.0

    def _build_graph(self):
        if self._graph is not None:
            return self._graph
        try:
            import igraph
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError(
                "mode='slow' needs personalized PageRank; install the extra with "
                "`pip install memoproagent[ppr]`, or use mode='fast'"
            ) from error
        self._add_synonym_edges()
        self.degree = defaultdict(int)
        for a, b in self.edges:
            self.degree[a] += 1
            self.degree[b] += 1
        keys = list(self.edges.keys())
        graph = igraph.Graph(n=len(self.nodes), edges=keys)
        graph.es["weight"] = [
            self.edges[k] * self._graph_penalty(k[0]) * self._graph_penalty(k[1]) for k in keys
        ]
        self._graph = graph
        return graph

    # ---- scoring ----

    def _embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self.embedder.embed([text]))[0]

    def _score_fact(self, index: int, seed_cosine: float) -> float:
        """Score the fact, not just the entity it hangs off.

        Seed cosine is a property of the matched entity, so every fact under one
        seed would tie -- with hundreds of facts under "phone", the top-k cut would
        degenerate into "whatever the sort happened to put first". Openness and
        recency are what actually vary per fact.
        """
        _, predicate, obj, _, _, days_ago, _ = self.facts[index]
        score = float(seed_cosine)
        if is_open_triple(predicate, obj):
            score += self.config.open_loop_bonus
        if days_ago is not None:
            horizon = self.config.recency_horizon_days
            score += self.config.recency_weight * (1.0 - min(float(days_ago), horizon) / horizon)
        return score

    def _fact_row(self, index: int) -> dict[str, Any]:
        subject, predicate, obj, day, clock, days_ago, eid = self.facts[index]
        return {
            "triple": [subject, predicate, obj],
            "day": day,
            "clock": clock,
            "days_ago": days_ago,
            "eid": eid,
        }

    # ---- main entry ----

    def retrieve(self, query: str, config: RetrievalConfig | None = None) -> RetrievalResult:
        """Retrieve candidate memory for an Observe output.

        `query` may name several objects ("milk carton, fridge door"); each is
        seeded independently and the results are merged, so one unmatched object
        does not suppress the others.
        """
        config = config or self.config
        parts = [p.strip() for p in re.split(r",|;| and ", query) if p.strip()]
        if not parts:
            return RetrievalResult(query=query, mode=config.mode)

        candidates: dict[int, float] = {}
        semantic_best: dict[int, float] = {}
        seeds: list[tuple[str, float]] = []
        has_semantic = self.semantic_emb is not None and len(self.semantic_emb) > 0

        for part in parts:
            vector = self._embed_query(part)
            sims = self.entity_emb @ vector
            order = np.argsort(-sims)[: config.seeds_per_part]
            part_seeds = [
                (self.entities[i], float(sims[i])) for i in order if sims[i] >= config.seed_threshold
            ]
            seeds.extend(part_seeds)
            semantic_sims = self.semantic_emb @ vector if has_semantic else None

            for entity, cosine in part_seeds:
                for index in self.node_to_facts.get(entity.lower(), []):
                    candidates[index] = max(candidates.get(index, 0.0), cosine)
                for si in self.semantic_node_to_facts.get(entity.lower(), []):
                    # Score a linked standing fact by its own similarity plus a link
                    # bonus; inheriting the seed cosine would tie every fact on the node.
                    value = float(semantic_sims[si]) + 0.15 if has_semantic else cosine
                    semantic_best[si] = max(semantic_best.get(si, 0.0), value)

            if has_semantic:
                for si in np.argsort(-semantic_sims)[: config.max_semantic]:
                    if semantic_sims[si] < config.semantic_threshold:
                        break
                    semantic_best[int(si)] = max(
                        semantic_best.get(int(si), 0.0), float(semantic_sims[si])
                    )

        scored = {index: self._score_fact(index, cosine) for index, cosine in candidates.items()}
        ranked = sorted(scored, key=lambda i: scored[i], reverse=True)[: config.max_episodic]

        if config.event_expand:
            ranked = self._expand_events(ranked)

        result = RetrievalResult(
            query=query,
            mode=config.mode,
            seeds=seeds[:5],
            episodic=[self._fact_row(i) for i in ranked[: config.max_episodic]],
            semantic=[
                {"triple": self.semantic_triples[i], "score": round(semantic_best[i], 3)}
                for i in sorted(semantic_best, key=lambda i: -semantic_best[i])[: config.max_semantic]
            ],
        )
        self._attach_provenance_passages(result, ranked, config)
        if config.mode == "slow":
            self._attach_ppr(result, seeds, ranked, semantic_best, config)
        if config.passage_vector_topk:
            self._attach_vector_passages(result, query, config)
        return result

    def _expand_events(self, ranked: list[int]) -> list[int]:
        """Add open-loop sibling facts from the best-ranked events."""
        top_eids: list[int] = []
        for index in ranked[:6]:
            eid = self.facts[index][6]
            if eid not in top_eids:
                top_eids.append(eid)
        seen = set(ranked)
        extra: list[int] = []
        for eid in top_eids[:3]:
            for sibling in self.event_to_facts.get(eid, []):
                if sibling in seen:
                    continue
                if is_open_triple(self.facts[sibling][1], self.facts[sibling][2]):
                    extra.append(sibling)
                    seen.add(sibling)
        return ranked + extra[:8]

    def _attach_provenance_passages(
        self, result: RetrievalResult, ranked: Sequence[int], config: RetrievalConfig
    ) -> None:
        """Attach the source event passages of the best-ranked facts.

        Triples fragment a narrative -- "brushed her whiskers", "she flinched",
        "she swiped" each read as routine on their own. The event rewrite keeps the
        causal link the reminder is actually about.
        """
        for index in ranked[:12]:
            if len(result.passages) >= config.max_passages:
                return
            eid = self.facts[index][6]
            if any(p["eid"] == eid for p in result.passages):
                continue
            event = self.events[eid]
            result.passages.append(
                {
                    "eid": eid,
                    "day": event["day"],
                    "days_ago": event["days_ago"],
                    "passage": event["passage"],
                }
            )

    def _attach_ppr(
        self,
        result: RetrievalResult,
        seeds: Sequence[tuple[str, float]],
        ranked: Sequence[int],
        semantic_best: dict[int, float],
        config: RetrievalConfig,
    ) -> None:
        if not self.nodes:
            return
        reset = [0.0] * len(self.nodes)
        for entity, cosine in seeds:
            if entity.lower() in self.node_id:
                reset[self.node_id[entity.lower()]] += cosine
        if sum(reset) <= 0:
            return
        scores = self._build_graph().personalized_pagerank(
            reset=reset,
            damping=config.ppr_damping,
            weights="weight",
            directed=False,
            implementation="prpack",
        )
        ranked_events = sorted(
            ((sum(scores[n] for n in nodes), eid) for eid, nodes in self.event_nodes.items() if nodes),
            reverse=True,
        )
        seen = {self.facts[i][6] for i in ranked}
        for score, eid in ranked_events[: config.ppr_events]:
            if eid in seen:
                continue
            seen.add(eid)
            event = self.events[eid]
            result.passages.append(
                {
                    "eid": eid,
                    "day": event["day"],
                    "days_ago": event["days_ago"],
                    "passage": event["passage"],
                    "ppr_score": round(score, 5),
                }
            )
        # PageRank ranks the semantic layer too, now that it shares the graph: a
        # standing rule reachable only through an association surfaces here.
        have = set(sorted(semantic_best, key=lambda i: -semantic_best[i])[: config.max_semantic])
        ranked_semantic = sorted(
            ((sum(scores[n] for n in nodes), si) for si, nodes in self.semantic_nodes.items() if nodes),
            reverse=True,
        )
        for score, si in ranked_semantic[: config.max_semantic]:
            if si in have or score <= 0:
                continue
            have.add(si)
            result.semantic.append(
                {"triple": self.semantic_triples[si], "ppr_score": round(score, 5)}
            )

    def _attach_vector_passages(
        self, result: RetrievalResult, query: str, config: RetrievalConfig
    ) -> None:
        """Union in top-k passages by direct cosine, which catches events whose
        entities were never extracted cleanly."""
        if self.store.passage_emb is None or not len(self.store.passage_emb):
            return
        vector = self._embed_query(query)
        sims = self.store.passage_emb @ vector
        seen_eids = {p.get("eid") for p in result.passages if p.get("eid") is not None}
        seen_text = {_normalize(p.get("passage", "")) for p in result.passages}
        for i in np.argsort(-sims)[: config.passage_vector_topk]:
            index = int(i)
            passage = self.store.passages[index]
            eid = self.store.passage_eids[index]
            key = _normalize(passage)
            if (eid is not None and eid in seen_eids) or key in seen_text:
                continue
            event = self.events.get(eid, {}) if eid is not None else {}
            result.passages.append(
                {
                    "eid": eid,
                    "day": event.get("day"),
                    "days_ago": event.get("days_ago"),
                    "passage": passage,
                    "vector_score": round(float(sims[index]), 5),
                }
            )
            if eid is not None:
                seen_eids.add(eid)
            seen_text.add(key)

    # ---- rendering ----

    def render_for_gate(self, result: RetrievalResult, config: RetrievalConfig | None = None) -> str:
        """Render retrieval output into the fact block the gate reads.

        Order matters. Event passages and standing rules carry the causal link;
        the triple dump is mostly routine narration. Leading with the dump buries
        the signal, so passages come first.
        """
        config = config or self.config
        lines: list[str] = []
        for passage in result.passages:
            text = str(passage["passage"])[: config.gate_passage_chars]
            lines.append(f"EVENT ({passage['days_ago']}d ago): {text}")
        for standing in result.semantic:
            lines.append(f"[standing] {standing['triple']}")
        if result.episodic:
            if lines:
                lines.append("Individual facts about the object:")
            for fact in result.episodic[: config.gate_max_episodic_lines]:
                lines.append(f"- {fact['triple']} ({fact['days_ago']}d ago)")
        return "\n".join(lines)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())
