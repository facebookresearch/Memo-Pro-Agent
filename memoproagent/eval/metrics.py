# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Scoring: from runs to the numbers in the paper.

**Perfect** is the primary metric and is deliberately strict -- it requires the
system to speak *once*, *inside* the trigger window, with content the judge calls
GOOD. The remaining columns exist to show where a system that misses Perfect
actually loses it:

    Hit            at least one in-window fire judged GOOD
    over-trigger   fraction of clips with more than one fire
    early          fraction of clips with a fire before the window
    late           fraction of clips with a fire after the window (plus grace)
    fires/clip     total fires divided by clips

A system can score well on Hit and still score near zero on Perfect by firing
constantly: that gap is the whole point of reporting both.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..data.schema import ClipRun, Fire, Interval, split_fires
from .judge import Category, Verdict

# Serialized format identifiers stay stable across software package renames.
SCHEMA_VERSION = "properagent_metrics_v2"
EVALUATION_PROTOCOL = "properagent_allfire_v1"

DEFAULT_GRACE_SECONDS = 2.0
"""One-sided inclusive tolerance after each trigger window, never before it."""


def select_fires_for_judging(
    fires: Sequence[Fire],
    windows: Sequence[Interval],
    grace: float = DEFAULT_GRACE_SECONDS,
    *,
    max_in_window: int = 8,
    max_out_of_window: int = 5,
) -> list[tuple[Fire, bool]]:
    """Select exact-text representatives without changing the original fire counts.

    Ordering follows the first occurrence of each utterance, even if its first
    in-window occurrence is later. Text seen in-window is never judged off-window.
    """
    if max_in_window < 0 or max_out_of_window < 0:
        raise ValueError("judgment caps must be nonnegative")
    representatives: dict[str, list[Fire | None]] = {}
    for fire in fires:
        slots = representatives.setdefault(fire.answer, [None, None])
        in_window = any(start <= fire.t <= end + grace for start, end in windows)
        index = 0 if in_window else 1
        if slots[index] is None:
            slots[index] = fire
    inside = [(slots[0], True) for slots in representatives.values() if slots[0] is not None]
    outside = [
        (slots[1], False)
        for slots in representatives.values()
        if slots[0] is None and slots[1] is not None
    ]
    return inside[:max_in_window] + outside[:max_out_of_window]


@dataclass
class ScoredClip:
    """One clip's outcome, with everything the metrics need."""

    uid: str
    n_fires: int
    n_before: int
    n_in_window: int
    n_after: int
    category: str | None = None
    """Best selected in-window verdict; None when absent or timing-only."""
    timing: str = "no_fire"
    fire_times: list[float] = field(default_factory=list)
    judged_answer: str = ""
    judge_reason: str = ""
    scoring_mode: str = "judged"
    grace_seconds: float = DEFAULT_GRACE_SECONDS
    windows: list[list[float]] = field(default_factory=list)
    selected_judgments: list[dict[str, Any]] = field(default_factory=list)
    evaluation: dict[str, Any] = field(default_factory=dict)

    @property
    def hit(self) -> bool:
        content_correct = self.scoring_mode == "timing_only" or self.category == Category.GOOD.value
        return content_correct and self.n_in_window > 0

    @property
    def perfect(self) -> bool:
        return self.hit and self.n_fires == 1 and self.n_before == 0 and self.n_after == 0

    def to_json(self) -> dict[str, Any]:
        row = asdict(self)
        row["schema_version"] = SCHEMA_VERSION
        row["protocol"] = EVALUATION_PROTOCOL
        row["hit"] = self.hit
        row["perfect"] = self.perfect
        return row


def score_clip(
    run: ClipRun,
    verdict: Verdict | None,
    judged_fire: Fire | None = None,
    grace: float = DEFAULT_GRACE_SECONDS,
    *,
    scoring_mode: str = "judged",
    selected_judgments: Sequence[dict[str, Any]] = (),
    evaluation: dict[str, Any] | None = None,
) -> ScoredClip:
    """Combine all fire counts with the best selected in-window verdict."""
    if scoring_mode not in ("judged", "timing_only"):
        raise ValueError(f"unknown scoring mode: {scoring_mode}")
    if not math.isfinite(grace) or grace < 0:
        raise ValueError("grace must be finite and nonnegative")
    if scoring_mode == "timing_only" and (verdict is not None or selected_judgments):
        raise ValueError("timing-only scoring cannot carry content judgments")
    before, inside, after = split_fires(run.fires, run.windows, grace)
    return ScoredClip(
        uid=run.uid,
        n_fires=len(run.fires),
        n_before=len(before),
        n_in_window=len(inside),
        n_after=len(after),
        category=verdict.category.value if verdict else None,
        timing="in_window" if inside else ("mistimed" if run.fires else "no_fire"),
        fire_times=[round(f.t, 2) for f in run.fires],
        judged_answer=judged_fire.answer if judged_fire else "",
        judge_reason=verdict.reason if verdict else "",
        scoring_mode=scoring_mode,
        grace_seconds=grace,
        windows=[list(window) for window in run.windows],
        selected_judgments=list(selected_judgments),
        evaluation=dict(evaluation or {}),
    )


def summarize(rows: Iterable[ScoredClip], label: str = "") -> dict[str, Any]:
    """Aggregate scored clips into the reported table row."""
    scored = list(rows)
    n = len(scored)
    if n == 0:
        raise ValueError("cannot summarize an empty result set")

    first = scored[0]
    if any(
        (r.scoring_mode, r.grace_seconds, r.evaluation)
        != (first.scoring_mode, first.grace_seconds, first.evaluation)
        for r in scored[1:]
    ):
        raise ValueError("cannot summarize mixed evaluation protocols or scoring modes")

    perfect = sum(r.perfect for r in scored)
    hit = sum(r.hit for r in scored)
    over = sum(r.n_fires > 1 for r in scored)
    early = sum(r.n_before > 0 for r in scored)
    late = sum(r.n_after > 0 for r in scored)
    total_fires = sum(r.n_fires for r in scored)

    def metric(count: int) -> dict[str, float | int]:
        return {"count": count, "percent": 100.0 * count / n}

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": EVALUATION_PROTOCOL,
        "scoring_mode": first.scoring_mode,
        "grace_seconds": first.grace_seconds,
        "evaluation": first.evaluation,
        "label": label,
        "n": n,
        "perfect": metric(perfect),
        "hit": metric(hit),
        "over_trigger": metric(over),
        "early_trigger": metric(early),
        "late_trigger": metric(late),
        "total_fires": total_fires,
        "fires_per_clip": total_fires / n,
    }


def format_summary(metrics: dict[str, Any]) -> str:
    def show(name: str) -> str:
        value = metrics[name]
        return f"{value['percent']:.1f}% ({value['count']}/{metrics['n']})"

    label = metrics.get("label") or "run"
    if metrics.get("scoring_mode") == "timing_only":
        label += " [timing-only upper bound]"
    return (
        f"{label}\n"
        f"  PERFECT       {show('perfect')}\n"
        f"  HIT           {show('hit')}\n"
        f"  over-trigger  {show('over_trigger')}\n"
        f"  early         {show('early_trigger')}\n"
        f"  late          {show('late_trigger')}\n"
        f"  fires/clip    {metrics['fires_per_clip']:.2f}"
    )


def false_trigger_rate(rows: Sequence[ScoredClip]) -> dict[str, Any]:
    """Fraction of clips with any fire. Reported on negative clips, where the
    system should stay silent throughout and every fire is an interruption cost."""
    scored = list(rows)
    if not scored:
        raise ValueError("cannot compute a false-trigger rate over no clips")
    fired = sum(r.n_fires > 0 for r in scored)
    return {
        "n": len(scored),
        "clips_with_fire": fired,
        "false_trigger_rate": 100.0 * fired / len(scored),
        "fires_per_clip": sum(r.n_fires for r in scored) / len(scored),
    }


def write_json(path: str | Path, value: Any) -> None:
    """Write atomically, so an interrupted run never leaves a half-written result
    that looks complete to the next reader."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
