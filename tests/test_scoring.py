"""All-fire scoring orchestration, without a model or video service."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from properagent.cli import run as cli
from properagent.data.frames import streaming_decisions, windowed_decisions
from properagent.data.schema import Clip, ClipRun, Fire
from properagent.eval.judge import JUDGE_SYSTEM, Category, JudgeConfig, Verdict
from properagent.eval.metrics import select_fires_for_judging, summarize


@pytest.fixture
def clip():
    return Clip(
        uid="u", video_path="unused", caption_log="", duration=30.0,
        windows=((10.0, 20.0),), target="", closes_when="", targets=(),
    )


def make_run(clip, fires):
    return ClipRun(
        uid=clip.uid, paradigm="windowed", config_label="test",
        windows=clip.windows, fires=[Fire(t, answer) for t, answer in fires],
    )


class StubJudge:
    def __init__(self, categories, config=None):
        self.categories = categories
        self.config = config or JudgeConfig()
        self.calls = []

    def __call__(self, clip, fire, frames, *, image_labels=None):
        self.calls.append((fire.answer, fire.t, frames, image_labels))
        category = self.categories[fire.answer]
        if isinstance(category, Exception):
            raise category
        return Verdict(
            category, category != Category.NON_MEMORY,
            category in (Category.GOOD, Category.BASIC), category == Category.GOOD,
            reason=fire.answer,
        )


@pytest.fixture(autouse=True)
def stub_frames(monkeypatch):
    calls = []

    def extract(path, t, n_frames, gap):
        calls.append((path, t, n_frames, gap))
        return [b"png"] * n_frames, ["label"] * n_frames

    monkeypatch.setattr(cli, "extract_judge_frames", extract)
    return calls


def test_bad_then_good_is_hit_and_all_selected_fires_are_judged(clip, stub_frames):
    run = make_run(clip, [(11, "bad"), (12, "good"), (13, "basic"), (25, "outside")])
    judge = StubJudge({
        "bad": Category.BAD, "good": Category.GOOD,
        "basic": Category.BASIC, "outside": Category.GOOD,
    })
    row = cli._score(clip, run, judge, {}, 2.0)
    assert row.hit and not row.perfect
    assert row.judged_answer == "good"
    assert [call[0] for call in judge.calls] == ["bad", "good", "basic", "outside"]
    assert (row.n_fires, row.n_in_window, row.n_after) == (4, 3, 1)
    assert len(row.selected_judgments) == 4
    assert row.selected_judgments[-1]["in_window"] is False
    assert all(call[2:] == (4, 1.0) for call in stub_frames)


def test_explicit_old_three_frame_setting_still_flows_to_extraction(clip, stub_frames):
    # Explicit old custom (3 x 2.5s) stays supported via the judge config.
    run = make_run(clip, [(11, "good")])
    judge = StubJudge({"good": Category.GOOD}, config=JudgeConfig(n_frames=3, frame_gap=2.5))
    row = cli._score(clip, run, judge, {}, 2.0)
    assert row.hit
    assert all(call[2:] == (3, 2.5) for call in stub_frames)


def test_actual_optional_single_frame_config_scoring_path(clip, stub_frames):
    # Existing config override: --set judge.n_frames=1 (current-only, no new flag).
    run = make_run(clip, [(11, "good")])
    judge = StubJudge({"good": Category.GOOD}, config=JudgeConfig(n_frames=1))
    row = cli._score(clip, run, judge, {}, 2.0)
    assert row.hit
    assert row.evaluation["judge_frames"]["n_frames"] == 1
    assert row.selected_judgments[0]["image_labels"] == ["label"]
    assert all(call[2:] == (1, 1.0) for call in stub_frames)


def test_repeated_text_prefers_first_in_window_and_keeps_all_counts(clip):
    run = make_run(clip, [(1, "same"), (11, "same"), (12, "same"), (25, "same")])
    judge = StubJudge({"same": Category.GOOD})
    row = cli._score(clip, run, judge, {}, 2.0)
    assert [(call[0], call[1]) for call in judge.calls] == [("same", 11)]
    assert row.n_fires == 4 and row.n_before == 1 and row.n_after == 1
    assert row.hit and not row.perfect


def test_exact_text_dedup_does_not_normalize_case_or_whitespace(clip):
    run = make_run(clip, [(11, "same"), (12, "Same"), (13, " same ")])
    assert len(select_fires_for_judging(run.fires, clip.windows)) == 3


def test_caps_follow_first_text_insertion_not_representative_timestamp(clip):
    run = make_run(clip, [(1, "a"), (11, "b"), (12, "a"), (13, "c"),
                          (25, "x"), (26, "y"), (27, "x")])
    selected = select_fires_for_judging(
        run.fires, clip.windows, max_in_window=2, max_out_of_window=1,
    )
    assert [(fire.answer, fire.t, inside) for fire, inside in selected] == [
        ("a", 12, True), ("b", 11, True), ("x", 25, False),
    ]


def test_default_caps_and_unjudged_good_do_not_count_as_hit(clip):
    fires = [(10 + i / 10, str(i)) for i in range(9)] + [(25, f"o{i}") for i in range(6)]
    judge = StubJudge({str(i): Category.BAD for i in range(8)} | {
        "8": Category.GOOD, **{f"o{i}": Category.GOOD for i in range(6)},
    })
    row = cli._score(clip, make_run(clip, fires), judge, {}, 2)
    assert len(judge.calls) == 13
    assert row.n_fires == 15 and not row.hit
    assert row.evaluation["max_in_window"] == 8
    assert row.evaluation["max_out_of_window"] == 5


@pytest.mark.parametrize("failed_answer", ["inside", "outside"])
def test_any_selected_judge_failure_raises_even_after_good(clip, failed_answer):
    judge = StubJudge({
        "good": Category.GOOD, "inside": Category.BAD, "outside": Category.BAD,
    } | {failed_answer: ValueError("broken judge")})
    run = make_run(clip, [(11, "good"), (12, "inside"), (25, "outside")])
    with pytest.raises(ValueError, match="broken judge"):
        cli._score(clip, run, judge, {}, 2)


@pytest.mark.parametrize("time, expected", [(9.99, False), (10, True), (22, True), (22.01, False)])
def test_one_sided_inclusive_grace(clip, time, expected):
    row = cli._score(clip, make_run(clip, [(time, "x")]), None, {}, 2)
    assert row.hit is expected and row.perfect is expected
    assert row.category is None and row.scoring_mode == "timing_only"


def test_current_window_overrides_cached_window_without_mutating_trace(clip):
    run = make_run(clip, [(11, "x")])
    run.windows = ((25, 28),)
    row = cli._score(clip, run, None, {}, 2)
    assert row.perfect
    assert row.windows == [[10.0, 20.0]]
    assert run.windows == ((25, 28),)


def test_multiple_windows_and_negative_clips(clip):
    multi = replace(clip, windows=((1, 3), (10, 12)))
    row = cli._score(multi, make_run(multi, [(4, "a"), (7, "b"), (13, "c")]), None, {}, 2)
    assert (row.n_in_window, row.n_after) == (2, 1)
    negative = replace(clip, windows=())
    judge = StubJudge({"x": Category.GOOD})
    row = cli._score(negative, make_run(negative, [(11, "x")]), judge, {}, 2)
    assert row.n_after == 1 and not row.hit
    assert len(judge.calls) == 1


def test_silence_never_opens_video_or_calls_judge(clip, stub_frames):
    judge = StubJudge({})
    row = cli._score(clip, make_run(clip, []), judge, {}, 2)
    assert not row.hit and not row.perfect
    assert not judge.calls and not stub_frames


def test_timing_only_summary_is_explicit_and_serializable(clip):
    row = cli._score(clip, make_run(clip, [(11, "x")]), None, {}, 2)
    metrics = summarize([row])
    assert metrics["scoring_mode"] == "timing_only"
    assert metrics["perfect"]["count"] == 1
    assert metrics["grace_seconds"] == 2
    assert metrics["evaluation"]["judge_frames"]["encoding"] == "png"
    assert row.to_json()["selected_judgments"] == []
    json.dumps(row.to_json())
    assert "timing-only upper bound" in cli.format_summary(metrics)


def test_score_evaluation_records_frame_shape_and_prompt_sha(clip, stub_frames):
    run = make_run(clip, [(11, "good")])
    judge = StubJudge({"good": Category.GOOD})
    row = cli._score(clip, run, judge, {}, 2.0)
    assert row.evaluation["judge_frames"] == {
        "n_frames": 4, "frame_gap": 1.0, "encoding": "png", "resolution": "original",
    }
    expected_sha = hashlib.sha256(JUDGE_SYSTEM.encode()).hexdigest()
    assert row.evaluation["judge_prompt_sha256"] == expected_sha
    # Hardcoded SHA of the confirmed tested prompt (no private fixture dep).
    assert expected_sha == "0e4884f100863324c3d386fc1c221ed1565615c14dfa030d2aa1ab2954dc9c3a"
    assert (row.evaluation["max_in_window"], row.evaluation["max_out_of_window"]) == (8, 5)


def test_negative_times_clamp_to_zero(clip, stub_frames):
    # Early fire clamped: extraction still called with the judge frame shape.
    run = make_run(clip, [(0.5, "good")])
    judge = StubJudge({"good": Category.GOOD})
    row = cli._score(clip, run, judge, {}, 2.0)
    assert row.n_before == 1 and not row.hit
    assert all(call[2:] == (4, 1.0) for call in stub_frames)


def test_summarize_rejects_mixed_prompt_or_frame_evaluation(clip):
    from properagent.eval.metrics import score_clip
    base = cli._score(clip, make_run(clip, [(11, "x")]), None, {}, 2)
    other = replace(base, evaluation={**base.evaluation, "judge_prompt_sha256": "other"})
    with pytest.raises(ValueError, match="mixed evaluation"):
        summarize([base, other])
    other_frames = replace(
        base, evaluation={**base.evaluation, "judge_frames": {"n_frames": 1}},
    )
    with pytest.raises(ValueError, match="mixed evaluation"):
        summarize([base, other_frames])


def test_resume_rejects_uid_and_paradigm_mismatch(clip, tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(make_run(clip, [(11, "x")]).to_json()))
    assert cli._load_run(path, expected_uid="u", expected_paradigm="windowed").uid == "u"
    with pytest.raises(ValueError, match="mismatched uid"):
        cli._load_run(path, expected_uid="another")
    with pytest.raises(ValueError, match="mismatched paradigm"):
        cli._load_run(path, expected_paradigm="streaming")


def test_empty_fire_answer_is_rejected_before_judging(clip, stub_frames):
    with pytest.raises(ValueError, match="empty fire answer"):
        cli._score(clip, make_run(clip, [(11, "")]), StubJudge({}), {}, 2)
    assert not stub_frames


def test_score_rejects_wrong_clip(clip):
    run = make_run(clip, [])
    run.uid = "other"
    with pytest.raises(ValueError, match="UID"):
        cli._score(clip, run, None, {}, 2)


@pytest.mark.parametrize("duration, expected", [(0, []), (4, [0, 2]), (4.1, [0, 2, 4])])
def test_windowed_schedule_excludes_terminal_time(duration, expected):
    class Source:
        def frame_at(self, t):
            return b"jpeg"
    assert [decision.t for decision in windowed_decisions(Source(), duration)] == expected


def test_streaming_schedule_excludes_terminal_time():
    class Source:
        def frame_at(self, t):
            return b"jpeg"
    assert [decision.t for decision in streaming_decisions(Source(), 3)] == [0, 1, 2]


@pytest.mark.parametrize("name", ["default", "example"])
def test_shipped_configs_match_canonical_defaults(name):
    from pathlib import Path
    from properagent.config import load_config
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / f"{name}.yaml")
    assert config["scoring"]["grace_seconds"] == 2
    judge = JudgeConfig(**config["judge"])
    assert (judge.n_frames, judge.frame_gap, judge.max_tokens) == (4, 1.0, 400)
    assert (judge.max_in_window, judge.max_out_of_window) == (8, 5)
