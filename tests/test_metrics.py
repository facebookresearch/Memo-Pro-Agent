# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Metric definitions, including the strictness of PERFECT."""
from __future__ import annotations

import pytest

from memoproagent.data.schema import ClipRun, Fire, split_fires
from memoproagent.eval.judge import Category, Verdict
from memoproagent.eval.metrics import false_trigger_rate, score_clip, summarize

WINDOW = (10.0, 20.0)


def verdict(category: Category) -> Verdict:
    return Verdict(
        category=category,
        memory_grounded=category is not Category.NON_MEMORY,
        correct=category in (Category.GOOD, Category.BASIC),
        useful=category is Category.GOOD,
    )


def run(uid: str, times: list[float]) -> ClipRun:
    return ClipRun(
        uid=uid,
        paradigm="windowed",
        config_label="test",
        windows=(WINDOW,),
        fires=[Fire(t=t, answer=f"reminder at {t}") for t in times],
    )


def test_split_fires():
    before, inside, after = split_fires(
        [Fire(t=t, answer="x") for t in (5.0, 15.0, 25.0)], (WINDOW,)
    )
    assert [f.t for f in before] == [5.0]
    assert [f.t for f in inside] == [15.0]
    assert [f.t for f in after] == [25.0]


def test_single_good_in_window_fire_is_perfect():
    scored = score_clip(run("a", [15.0]), verdict(Category.GOOD))
    assert scored.hit and scored.perfect


def test_extra_fire_keeps_hit_but_loses_perfect():
    """The gap between Hit and Perfect is the whole reason both are reported."""
    scored = score_clip(run("a", [5.0, 15.0]), verdict(Category.GOOD))
    assert scored.hit
    assert not scored.perfect
    assert scored.n_before == 1


@pytest.mark.parametrize("category", [Category.BASIC, Category.BAD, Category.NON_MEMORY])
def test_non_good_content_is_neither_hit_nor_perfect(category):
    scored = score_clip(run("a", [15.0]), verdict(category))
    assert not scored.hit and not scored.perfect


def test_silence_is_neither_hit_nor_perfect():
    scored = score_clip(run("a", []), None)
    assert not scored.hit and not scored.perfect
    assert scored.timing == "no_fire"


def test_late_grace_moves_a_boundary_fire_in_window():
    strict = score_clip(run("a", [21.0]), None, grace=0.0)
    assert strict.n_after == 1 and strict.n_in_window == 0

    lenient = score_clip(run("a", [21.0]), verdict(Category.GOOD), grace=2.0)
    assert lenient.n_after == 0 and lenient.n_in_window == 1
    assert lenient.perfect


def test_summary_counts_and_percentages():
    rows = [
        score_clip(run("a", [15.0]), verdict(Category.GOOD)),  # perfect
        score_clip(run("b", [5.0, 15.0]), verdict(Category.GOOD)),  # hit, early, over
        score_clip(run("c", [25.0]), None),  # late only
        score_clip(run("d", []), None),  # silent
    ]
    metrics = summarize(rows, label="test")
    assert metrics["n"] == 4
    assert metrics["perfect"]["count"] == 1
    assert metrics["hit"]["count"] == 2
    assert metrics["over_trigger"]["count"] == 1
    assert metrics["early_trigger"]["count"] == 1
    assert metrics["late_trigger"]["count"] == 1
    assert metrics["fires_per_clip"] == pytest.approx(4 / 4)
    assert metrics["perfect"]["percent"] == pytest.approx(25.0)


def test_summary_rejects_an_empty_result_set():
    with pytest.raises(ValueError):
        summarize([], label="empty")


def test_negative_clip_fires_are_all_errors():
    """With no window there is no 'too early': every fire is a false trigger."""
    negative = ClipRun(
        uid="n",
        paradigm="windowed",
        config_label="test",
        windows=(),
        fires=[Fire(t=3.0, answer="x")],
    )
    scored = score_clip(negative, None)
    assert scored.n_in_window == 0
    assert scored.n_before == 0
    assert scored.n_after == 1

    rates = false_trigger_rate([scored, score_clip(run("q", []), None)])
    assert rates["false_trigger_rate"] == pytest.approx(50.0)


def test_multiple_disjoint_windows():
    """A few clips have several trigger windows: the object leaves view and comes
    back, and either moment is a legitimate time to speak."""
    two = ClipRun(
        uid="m",
        paradigm="windowed",
        config_label="test",
        windows=((10.0, 14.6), (27.5, 45.0)),
        fires=[Fire(t=30.0, answer="x")],
    )
    scored = score_clip(two, verdict(Category.GOOD))
    assert scored.n_in_window == 1 and scored.perfect

    # A fire in the gap between windows is late, not early: the first moment has
    # already passed.
    gap = ClipRun(
        uid="m",
        paradigm="windowed",
        config_label="test",
        windows=((10.0, 14.6), (27.5, 45.0)),
        fires=[Fire(t=20.0, answer="x")],
    )
    scored = score_clip(gap, None)
    assert (scored.n_before, scored.n_in_window, scored.n_after) == (0, 0, 1)


def test_timing_only_is_an_upper_bound_without_a_good_verdict():
    row = score_clip(run("a", [15.0]), None, scoring_mode="timing_only")
    assert row.hit and row.perfect and row.category is None
    assert row.to_json()["scoring_mode"] == "timing_only"


def test_summary_rejects_mixed_modes_or_protocol_settings():
    judged = score_clip(run("a", [15.0]), verdict(Category.GOOD))
    timing = score_clip(run("b", [15.0]), None, scoring_mode="timing_only")
    with pytest.raises(ValueError, match="mixed"):
        summarize([judged, timing])
    strict = score_clip(run("b", [15.0]), verdict(Category.GOOD), grace=0)
    with pytest.raises(ValueError, match="mixed"):
        summarize([judged, strict])
    different = score_clip(run("b", [15.0]), verdict(Category.GOOD), evaluation={"cap": 1})
    with pytest.raises(ValueError, match="mixed"):
        summarize([judged, different])


@pytest.mark.parametrize("grace", [-1, float("nan"), float("inf")])
def test_invalid_grace_is_rejected(grace):
    with pytest.raises(ValueError, match="grace"):
        score_clip(run("a", []), None, grace=grace)


def test_invalid_mode_and_timing_only_content_are_rejected():
    with pytest.raises(ValueError, match="scoring mode"):
        score_clip(run("a", []), None, scoring_mode="unknown")
    with pytest.raises(ValueError, match="content judgments"):
        score_clip(run("a", [15]), verdict(Category.GOOD), scoring_mode="timing_only")


def test_default_grace_is_two_seconds_and_does_not_allow_early_fires():
    assert score_clip(run("a", [22]), verdict(Category.GOOD)).perfect
    assert not score_clip(run("a", [9]), verdict(Category.GOOD)).hit


def test_normalize_windows_accepts_both_shapes():
    from memoproagent.data.schema import normalize_windows

    assert normalize_windows([2.0, 45.5]) == ((2.0, 45.5),)
    assert normalize_windows([[10.0, 14.6], [27.5, 45.0]]) == ((10.0, 14.6), (27.5, 45.0))
    assert normalize_windows(None) == ()
    with pytest.raises(ValueError):
        normalize_windows([5.0, 5.0])
    with pytest.raises(ValueError):
        normalize_windows([1.0, 2.0, 3.0])
