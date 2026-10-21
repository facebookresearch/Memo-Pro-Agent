# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Schemas for the two data contracts MemoProAgent reads and writes.

Contract A — benchmark input, one directory per split::

    <data_root>/
      clips/<uid>.mp4              the egocentric video clip
      full_history/<uid>.jsonl     the caption stream (long-term memory, raw form)
      trigger/<uid>.json           trigger window + target
      memory/<uid>.json            the memory that justifies the intervention
      visibility/<uid>.json        per-timestep target visibility
      uids.txt                     newline-separated UIDs to evaluate

Contract B — built memory, one directory for all UIDs::

    <memory_root>/
      <uid>.episodic.json
      <uid>.semantic.json
      <uid>.passages.json
      <uid>.entity_list.json
      <uid>.entity_emb.npy
      <uid>.semantic_emb.npy
      <uid>.passages_emb.npy
      <uid>.manifest.json          embedding signature + row counts + content hashes

Contract B is produced by ``memoproagent.memory.build`` from Contract A's
``full_history``. Everything downstream reads only these two contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

VisibilityState = Literal["yes", "no", "uncertain"]


@dataclass(frozen=True)
class Caption:
    """One line of ``full_history/<uid>.jsonl``.

    `t_days_before` is how many days before the clip this caption occurred; it is the
    only recency signal retrieval has, so it must be present for ranking to work.
    """

    cid: str
    day: int
    t: float
    text: str
    t_days_before: int | None = None
    clock: str | None = None
    tag: str | None = None
    episode_id: str | None = None
    title: str | None = None

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "Caption":
        missing = {"cid", "day", "t", "text"} - set(row)
        if missing:
            raise ValueError(f"caption is missing required fields {sorted(missing)}: {row}")
        return cls(
            cid=str(row["cid"]),
            day=int(row["day"]),
            t=float(row["t"]),
            text=str(row["text"]),
            t_days_before=row.get("t_days_before"),
            clock=row.get("clock"),
            tag=row.get("tag"),
            episode_id=row.get("episode_id"),
            title=row.get("title"),
        )


@dataclass(frozen=True)
class MemoryItem:
    """One piece of evidence behind an intervention."""

    kind: str
    text: str
    day_offset: int | None = None


@dataclass(frozen=True)
class Target:
    """One target object and the open loop attached to it."""

    target: str
    worth_triggering: bool
    open_loop: str
    memories: tuple[MemoryItem, ...] = ()
    closes_when: str = ""

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "Target":
        return cls(
            target=str(row.get("target", "")),
            worth_triggering=bool(row.get("worth_triggering", False)),
            open_loop=str(row.get("open_loop", "")),
            memories=tuple(
                MemoryItem(
                    kind=str(m.get("kind", "")),
                    text=str(m.get("text", "")),
                    day_offset=m.get("day_offset"),
                )
                for m in row.get("memories", [])
            ),
            closes_when=str(row.get("closes_when", "")),
        )


@dataclass(frozen=True)
class Observation:
    t: float
    state: VisibilityState
    confidence: float = 1.0


Interval = tuple[float, float]


def normalize_windows(value: Any) -> tuple[Interval, ...]:
    """Accept ``[start, end]`` or ``[[s, e], ...]`` and return a tuple of intervals.

    Most clips have one trigger window, but a few have several disjoint ones --
    the object leaves view and comes back, and either moment is a legitimate time
    to speak. Normalizing here keeps every downstream check uniform.
    """
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"trigger_window must be [start, end] or [[s, e], ...], got {value!r}")
    pairs = value if isinstance(value[0], (list, tuple)) else [value]
    windows: list[Interval] = []
    for pair in pairs:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError(f"trigger window interval must be [start, end], got {pair!r}")
        start, end = float(pair[0]), float(pair[1])
        if end <= start:
            raise ValueError(f"empty trigger window interval {[start, end]}")
        windows.append((start, end))
    return tuple(sorted(windows))


@dataclass(frozen=True)
class Clip:
    """A fully resolved evaluation sample: Contract A for one UID.

    `windows` holds the trigger window(s) in seconds and is required for scoring.
    A clip with none is a *negative* clip: the system should stay silent
    throughout, and `worth_triggering` is False.
    """

    uid: str
    video_path: str
    caption_log: str
    duration: float
    windows: tuple[Interval, ...]
    target: str
    closes_when: str
    targets: tuple[Target, ...]
    visibility: tuple[Observation, ...] = ()

    @property
    def worth_triggering(self) -> bool:
        return any(t.worth_triggering for t in self.targets)

    @property
    def is_negative(self) -> bool:
        return not self.windows

    def in_window(self, timestamp: float, grace: float = 0.0) -> bool:
        return any(start <= timestamp <= end + grace for start, end in self.windows)

    def target_visible_at(self, timestamp: float, tol: float = 1.5) -> bool | None:
        if not self.visibility:
            return None
        nearest = min(self.visibility, key=lambda o: abs(o.t - timestamp))
        if abs(nearest.t - timestamp) > tol:
            return None
        return nearest.state == "yes"

    def gold_evidence(self) -> str:
        """The oracle evidence string: the open loop of the first worth-triggering
        target. Used by the oracle conditions and by the judge, never by the harness."""
        for target in self.targets:
            if target.worth_triggering and target.open_loop:
                return target.open_loop
        return ""


@dataclass
class Fire:
    """One intervention emitted by a system."""

    t: float
    answer: str
    observe: str = ""
    reason: str = ""
    evidence: str = ""


@dataclass
class ClipRun:
    """Everything one system produced on one clip."""

    uid: str
    paradigm: Literal["windowed", "streaming"]
    config_label: str
    fires: list[Fire] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    windows: tuple[Interval, ...] = ()
    health: dict[str, Any] | None = None
    """Optional per-run compatible health (streaming only). None for windowed
    runs and for traces written before health was recorded."""

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "uid": self.uid,
            "paradigm": self.paradigm,
            "config_label": self.config_label,
            "windows": [list(w) for w in self.windows],
            "fires": [
                {
                    "t": f.t,
                    "answer": f.answer,
                    "observe": f.observe,
                    "reason": f.reason,
                    "evidence": f.evidence,
                }
                for f in self.fires
            ],
            "steps": self.steps,
        }
        if self.health is not None:
            payload["health"] = self.health
        return payload


def split_fires(
    fires: Sequence[Fire], windows: Sequence[Interval], grace: float = 0.0
) -> tuple[list[Fire], list[Fire], list[Fire]]:
    """Partition fires into (before, in-window, after).

    With several windows, "before" means before the first one opens and "after"
    means after the last one closes; a fire in a gap between windows counts as
    late, since the moment it was aiming at has already passed.

    With no window at all, every fire is 'after' by convention: on a negative clip
    any fire is an error, and calling it 'early' would wrongly suggest the system
    had the right idea too soon.
    """
    if not windows:
        return [], [], list(fires)
    ordered = sorted(windows)
    first_start = ordered[0][0]
    before: list[Fire] = []
    inside: list[Fire] = []
    after: list[Fire] = []
    for fire in fires:
        if any(start <= fire.t <= end + grace for start, end in ordered):
            inside.append(fire)
        elif fire.t < first_start:
            before.append(fire)
        else:
            after.append(fire)
    return before, inside, after
