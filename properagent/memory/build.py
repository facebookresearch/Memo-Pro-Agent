"""Memory construction: caption stream -> episodic/semantic graph.

Five stages, run once per UID before any evaluation:

1. **Event segmentation** -- group the chronological caption stream into events;
   a gap larger than ``gap_seconds`` starts a new one.
2. **Faithful rewrite** -- turn each event's fragmented captions into one coherent
   first-person passage, preserving objects, states, quoted speech, and outcomes.
   Explicitly not a summary: the offhand "almost out of milk" is the signal.
3. **Episodic extraction** -- OpenIE entities and subject-relation-object triples
   over the rewrite, with first-person intentions *mirrored* onto their target
   object so an open loop is reachable from the thing the user can see.
4. **Semantic abstraction** -- standing facts (habits, preferences, dietary and
   safety constraints) distilled from groups of events.
5. **Indexing** -- embeddings for entities, semantic triples, and passages, plus a
   manifest recording the embedding signature and row counts.

The LLM prompts are verbatim from the implementation used in the paper.
"""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..backends.base import ChatBackend, EmbeddingBackend, GenerationConfig, Message
from ..data.schema import Caption
from .store import MANIFEST_SCHEMA, MemoryPaths, sha256_file

# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

REWRITE_SYSTEM = (
    "You are an Egocentric Event Transcription Specialist. You are given the raw, fragmented "
    'first-person caption stream of ONE event (captions are short, noisy, and may use vague '
    'references like "it", "that", "the thing"). Rewrite this stream into a single coherent, '
    "first-person account of the event.\n\n"
    "This is a FAITHFUL REWRITE, not a summary. Do not compress away detail.\n\n"
    "Requirements:\n"
    "1. Coherence & reference resolution: stitch the fragments into a smooth, chronologically "
    "ordered first-person narrative; resolve pronouns and vague references to explicit named "
    'entities (e.g. "it" -> "the oil bottle"). Keep the first person as "I".\n'
    "2. Preserve concrete details (priority): keep every object and its state/condition (e.g. low, "
    "empty, open), quantity, location and container relationship, spoken words / mutters (quote "
    "them verbatim), decision / intention / plan, and outcome. In THIS task small object states "
    "and offhand remarks ARE the most important signal; never drop them as trivial.\n"
    "3. Strict fidelity: include ONLY what is in the input; do not assume, infer motives, or invent "
    "facts; keep true chronological order.\n"
    "4. Phrasing for downstream extraction: prefer explicit subject-relation-object statements "
    "where natural.\n\n"
    "Output a single coherent first-person account as plain prose. No headings, bullets, or "
    "commentary."
)

NER_SYSTEM = (
    "Your task is to extract named entities from the given paragraph. "
    "Respond with a JSON list of entities.\n"
)

_NER_SHOT_INPUT = (
    "Radio City\n"
    "Radio City is India's first private FM radio station and was started on 3 July 2001.\n"
    "It plays Hindi, English and regional songs.\n"
    "Radio City recently forayed into New Media in May 2008 with the launch of a music portal - "
    "PlanetRadiocity.com that offers music related news, videos, songs, and other music-related "
    "features."
)
_NER_SHOT_OUTPUT = (
    '{"named_entities": ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", '
    '"PlanetRadiocity.com"]}'
)
_RE_SHOT_OUTPUT = (
    '{"triples": [["Radio City", "located in", "India"], ["Radio City", "is", "private FM radio '
    'station"], ["Radio City", "started on", "3 July 2001"], ["Radio City", "plays songs in", '
    '"Hindi"], ["Radio City", "plays songs in", "English"], ["Radio City", "forayed into", "New '
    'Media"], ["Radio City", "launched", "PlanetRadiocity.com"], ["PlanetRadiocity.com", "launched '
    'in", "May 2008"], ["PlanetRadiocity.com", "is", "music portal"]]}'
)

RE_SYSTEM = (
    "Your task is to construct an RDF (Resource Description Framework) graph from the given "
    "passage and named entity list.\n"
    "Respond with a JSON list of triples.\n\n"
    "Extract BOTH of the following:\n"
    "(A) Objective relationships between entities (actions, locations, containment, attributes).\n"
    "(B) PROACTIVE-MEMORY facts — you MUST also extract these even if they occur only once:\n"
    '    - Object states / conditions: e.g. ["oil bottle", "is", "low"].\n'
    '    - First-person intentions, plans, pending actions: e.g. ["I", "intend to", "refill the '
    'oil bottle before it runs out"].\n'
    "    - Spoken commitments or self-talk expressing a decision or reminder.\n\n"
    "MIRRORING RULE (important): whenever a first-person intention / pending action / reminder "
    "concerns a SPECIFIC object, ALSO emit a mirrored triple whose SUBJECT is that object, phrased "
    "as an open loop on the object itself. Emit BOTH triples.\n"
    '  Example: from "Next time I grab this, refill it before it\'s empty" about the oil bottle, '
    "emit both:\n"
    '    ["I", "intend to", "refill the oil bottle before it runs out"]\n'
    '    ["oil bottle", "needs", "refill before it runs out"]\n\n'
    "Requirements:\n"
    '- Keep first-person references as "I". First-person intention/state triples are always '
    "allowed.\n"
    "- The mirrored object triple must use the exact object node name so it attaches to that node.\n"
    "- Represent transient states and one-off intentions as triples; do NOT skip them.\n"
)

_RE_AUG_SHOT_PARAGRAPH = (
    "I picked up the milk carton and shook it; it felt almost empty. I muttered, "
    '"Almost out - grab milk on the next trip." I put the carton back in the fridge door.'
)
_RE_AUG_SHOT_ENTITIES = '{"named_entities": ["milk carton", "fridge door"]}'
_RE_AUG_SHOT_OUTPUT = json.dumps(
    {
        "triples": [
            ["I", "picked up", "milk carton"],
            ["milk carton", "is", "almost empty"],
            ["I", "intend to", "buy milk on the next trip"],
            ["milk carton", "needs", "restock on the next trip"],
            ["I", "put back", "milk carton"],
            ["milk carton", "located in", "fridge door"],
        ]
    }
)

SEMANTIC_SYSTEM = (
    "You extract the user's STANDING SEMANTIC memory from a stream of their past first-person "
    "activity. From the events, extract ONLY general, durable facts about the user that a "
    "proactive assistant would reuse later: habits/routines, preferences, dietary rules, "
    "allergies/health constraints, safety practices, standing rules, recurring intentions, and "
    "persistent possessions/relationships. Do NOT include one-off actions or momentary states. "
    "Prefer facts supported by RECURRING behavior or explicit statements. Output a JSON list of "
    "triples [subject, predicate, object], each a general standing fact, e.g. "
    '["I","keep exterior doors shut because of","my small dog"], ["I","avoid","dairy"], '
    '["I","limit","energy drinks to one per day"]. '
    "If no durable general fact is present, output []."
)


def _re_frame(passage: str, named_entity_json: str) -> str:
    return (
        "Convert the paragraph into a JSON dict, it has a named entity list and a "
        "triple list.\nParagraph:\n```\n" + passage + "\n```\n\n" + named_entity_json + "\n"
    )


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildConfig:
    gap_seconds: float = 60.0
    """A silence longer than this starts a new event."""

    semantic_chunk_events: int = 40
    workers: int = 12
    rewrite_max_tokens: int = 1200
    ner_max_tokens: int = 1000
    triple_max_tokens: int = 2500
    semantic_max_tokens: int = 800
    semantic_chunk_chars: int = 12000
    embed_batch: int = 64


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def segment_events(captions: Sequence[Caption], gap_seconds: float) -> list[list[Caption]]:
    """Stage 1. Split the caption stream wherever the gap exceeds `gap_seconds`."""
    if not captions:
        return []
    events: list[list[Caption]] = []
    current = [captions[0]]
    for previous, caption in zip(captions, captions[1:]):
        if caption.t - previous.t > gap_seconds:
            events.append(current)
            current = [caption]
        else:
            current.append(caption)
    events.append(current)
    return events


def _strip_fence(text: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()


def _complete_elements(text: str, start: int) -> list[Any]:
    """Recover the complete top-level elements of a possibly truncated array.

    An extraction response that runs into the token budget ends mid-element, so
    the whole array fails to parse. Everything before the cut is still valid and
    was still real work: dropping a 40-triple event because triple 41 got
    truncated loses most of an event's memory for no reason.
    """
    elements: list[Any] = []
    depth = 0
    in_string = False
    escaped = False
    element_start = -1
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "[{":
            if depth == 1:
                element_start = index
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth == 1 and element_start >= 0:
                try:
                    elements.append(json.loads(text[element_start : index + 1]))
                except json.JSONDecodeError:
                    pass
                element_start = -1
            elif depth == 0:
                break
    return elements


def _parse_json_list(raw: str, key: str) -> list[Any]:
    """Extract a JSON list from an extraction response.

    Models drift between the two shapes the prompt could be read as -- the
    documented ``{"<key>": [...]}`` and a bare ``[...]`` -- and which one you get
    varies by model and by call. Accepting both costs nothing; rejecting one
    silently drops whole events from the graph. A truncated response is salvaged
    down to its complete elements.
    """
    text = _strip_fence(raw)
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            continue
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
            # A single-key object under some other name is still unambiguous.
            lists = [v for v in parsed.values() if isinstance(v, list)]
            if len(lists) == 1:
                return lists[0]
            continue
        if isinstance(parsed, list):
            return parsed

    # Nothing parsed cleanly: the response is very likely truncated.
    opening = _first_array(text)
    if opening >= 0:
        salvaged = _complete_elements(text, opening)
        if salvaged:
            return salvaged
    raise ValueError(f"no JSON list for {key!r} in: {text[:200]!r}")


def _first_array(text: str) -> int:
    """Index of the first ``[`` that is not inside a string literal.

    For ``{"triples": [[...`` this is the array itself, so salvage starts at the
    right depth whether the model wrapped the list in an object or not.
    """
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "[":
            return index
    return -1


def _chat(backend: ChatBackend, messages: list[Message], max_tokens: int) -> str:
    return backend.chat(messages, GenerationConfig(max_tokens=max_tokens, temperature=0.0))


def rewrite_event(backend: ChatBackend, lines: Sequence[str], config: BuildConfig) -> str:
    """Stage 2."""
    return _chat(
        backend,
        [
            Message(role="system", text=REWRITE_SYSTEM),
            Message(role="user", text="Caption stream (one event):\n" + "\n".join(lines)),
        ],
        config.rewrite_max_tokens,
    ).strip()


def extract_entities(backend: ChatBackend, passage: str, config: BuildConfig) -> list[str]:
    """Stage 3a."""
    raw = _chat(
        backend,
        [
            Message(role="system", text=NER_SYSTEM),
            Message(role="user", text=_NER_SHOT_INPUT),
            Message(role="assistant", text=_NER_SHOT_OUTPUT),
            Message(role="user", text=passage),
        ],
        config.ner_max_tokens,
    )
    return [str(e) for e in _parse_json_list(raw, "named_entities") if str(e).strip()]


def extract_triples(
    backend: ChatBackend, passage: str, entities: Sequence[str], config: BuildConfig
) -> list[list[str]]:
    """Stage 3b. Includes the mirroring rule that makes open loops object-reachable."""
    raw = _chat(
        backend,
        [
            Message(role="system", text=RE_SYSTEM),
            Message(role="user", text=_re_frame(_NER_SHOT_INPUT, _NER_SHOT_OUTPUT)),
            Message(role="assistant", text=_RE_SHOT_OUTPUT),
            Message(role="user", text=_re_frame(_RE_AUG_SHOT_PARAGRAPH, _RE_AUG_SHOT_ENTITIES)),
            Message(role="assistant", text=_RE_AUG_SHOT_OUTPUT),
            Message(
                role="user",
                text=_re_frame(passage, json.dumps({"named_entities": list(entities)})),
            ),
        ],
        config.triple_max_tokens,
    )
    return [t for t in _parse_json_list(raw, "triples") if _is_triple(t)]


def extract_semantic(
    backend: ChatBackend, events: Sequence[dict[str, Any]], config: BuildConfig
) -> list[list[str]]:
    """Stage 4, for one chunk of events."""
    body = "\n".join(f"[{e.get('day')}] {e['rewrite']}" for e in events)
    raw = _chat(
        backend,
        [
            Message(role="system", text=SEMANTIC_SYSTEM),
            Message(
                role="user",
                text="Events:\n"
                + body[: config.semantic_chunk_chars]
                + "\n\nOutput the JSON triple list.",
            ),
        ],
        config.semantic_max_tokens,
    )
    try:
        parsed = _parse_json_list(raw, "triples")
    except ValueError:
        # An empty chunk legitimately yields no standing facts; the prompt asks
        # for [] and some models answer in prose instead.
        return []
    return [t for t in parsed if _is_triple(t)]


def _is_triple(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(x, str) and x.strip() for x in value)
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def build_memory(
    uid: str,
    captions: Sequence[Caption],
    out_root: str | Path,
    extractor: ChatBackend,
    embedder: EmbeddingBackend,
    config: BuildConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run the full pipeline for one UID and write Contract B artifacts.

    Events are processed concurrently; a per-event failure is recorded as
    ``ok: false`` with empty triples rather than aborting the UID, because one bad
    event out of ~900 should not cost the whole history. The failure count is
    returned and written into the manifest so it cannot pass unnoticed.
    """
    config = config or BuildConfig()
    out_root = Path(out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    paths = MemoryPaths.for_uid(out_root, uid)

    segments = segment_events(captions, config.gap_seconds)
    if not segments:
        raise ValueError(f"{uid}: caption stream produced no events")

    lock = threading.Lock()
    completed = [0]

    def build_one(index: int, segment: Sequence[Caption]) -> dict[str, Any]:
        event: dict[str, Any] = {
            "idx": index,
            "day": segment[0].day,
            "start": segment[0].clock,
            "end": segment[-1].clock,
            "episode_id": segment[0].episode_id,
            "title": segment[0].title,
            "n_caps": len(segment),
            "cids": [c.cid for c in segment],
            "days_ago": segment[0].t_days_before,
        }
        try:
            lines = [f"[{c.clock}] {c.text}" for c in segment]
            passage = rewrite_event(extractor, lines, config)
            entities = extract_entities(extractor, passage, config)
            triples = extract_triples(extractor, passage, entities, config)
            event["rewrite"] = passage
            event["triples"] = [t for t in triples if _is_triple(t)]
            event["ok"] = True
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            event["rewrite"] = ""
            event["triples"] = []
            event["ok"] = False
            event["error"] = str(error)[:300]
        with lock:
            completed[0] += 1
            if progress:
                progress(completed[0], len(segments))
        return event

    events: list[dict[str, Any] | None] = [None] * len(segments)
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = {pool.submit(build_one, i, s): i for i, s in enumerate(segments)}
        for future in as_completed(futures):
            events[futures[future]] = future.result()
    built = [e for e in events if e is not None]

    semantic = _build_semantic_layer(extractor, built, config)
    passages = [e["rewrite"] for e in built]
    entity_list = _collect_entities(built)

    episodic = {
        "history": uid,
        "gap_s": config.gap_seconds,
        "n_captions": len(captions),
        "n_events": len(built),
        "n_triples": sum(len(e["triples"]) for e in built),
        "n_failed_events": sum(1 for e in built if not e["ok"]),
        "events": built,
    }

    _write_json(paths.episodic, episodic)
    _write_json(paths.semantic, {"uid": uid, "n_semantic": len(semantic), "triples": semantic})
    _write_json(paths.passages, passages)
    _write_json(paths.entity_list, entity_list)

    np.save(paths.entity_emb, embedder.embed(entity_list))
    np.save(paths.semantic_emb, embedder.embed([" ".join(t) for t in semantic]))
    np.save(paths.passage_emb, embedder.embed(passages))

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "uid": uid,
        "embedding_signature": embedder.signature,
        "counts": {
            "events": len(built),
            "triples": episodic["n_triples"],
            "failed_events": episodic["n_failed_events"],
            "entities": len(entity_list),
            "semantic": len(semantic),
            "passages": len(passages),
        },
        "sha256": {
            path.name: sha256_file(path)
            for path in (
                paths.episodic,
                paths.semantic,
                paths.passages,
                paths.entity_list,
                paths.entity_emb,
                paths.semantic_emb,
                paths.passage_emb,
            )
        },
    }
    _write_json(paths.manifest, manifest)
    return manifest


def _build_semantic_layer(
    backend: ChatBackend, events: Sequence[dict[str, Any]], config: BuildConfig
) -> list[list[str]]:
    usable = [e for e in events if e.get("rewrite")]
    chunks = [
        usable[i : i + config.semantic_chunk_events]
        for i in range(0, len(usable), config.semantic_chunk_events)
    ]
    collected: list[list[str]] = []
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        for triples in pool.map(lambda c: extract_semantic(backend, c, config), chunks):
            collected.extend(triples)
    seen: set[tuple[str, str, str]] = set()
    deduped: list[list[str]] = []
    for triple in collected:
        key = tuple(x.strip().lower() for x in triple)  # type: ignore[assignment]
        if key in seen:
            continue
        seen.add(key)
        deduped.append([x.strip() for x in triple])
    return deduped


def _collect_entities(events: Sequence[dict[str, Any]]) -> list[str]:
    """Every distinct subject and object across all triples, in sorted order.

    Sorted rather than first-seen so a rebuild produces the same row order, which
    is what makes the ``.npy`` alignment reproducible.
    """
    entities: set[str] = set()
    for event in events:
        for subject, _, obj in event.get("triples", []):
            for value in (subject, obj):
                value = value.strip()
                if value:
                    entities.add(value)
    return sorted(entities)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
