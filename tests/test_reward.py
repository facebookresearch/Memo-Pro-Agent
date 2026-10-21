# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""The GRPO reward, checked against the definition in the paper."""
from __future__ import annotations

import pytest

from memoproagent.rl.reward import (
    Action,
    StateClass,
    Trajectory,
    group_advantages,
    reward,
)

GOOD = lambda answer: True  # noqa: E731
BAD = lambda answer: False  # noqa: E731


def fire(answer: str = "Grab milk this trip.") -> Trajectory:
    return Trajectory(observe_closed=True, answer_closed=True, answer=answer, has_eos=True)


def silent() -> Trajectory:
    return Trajectory(observe_closed=True, answer_closed=True, answer="NO_REMINDER", has_eos=True)


def test_correct_silence_scores_plus_one():
    for state in (StateClass.EARLY, StateClass.ALREADY_SAID, StateClass.NO_EVIDENCE_GUARD):
        result = reward(state, silent())
        assert result.reward == 1.0
        assert result.branch == "correct_silence"


def test_correctly_timed_good_fire_scores_plus_one():
    result = reward(StateClass.ACTIVE_IN_WINDOW, fire(), judge=GOOD)
    assert result.reward == 1.0
    assert result.branch == "good_fire"


def test_correctly_timed_low_quality_fire_scores_minus_one():
    result = reward(StateClass.ACTIVE_IN_WINDOW, fire(), judge=BAD)
    assert result.reward == -1.0
    assert result.branch == "low_quality_fire"


def test_miss_scores_minus_one():
    result = reward(StateClass.ACTIVE_IN_WINDOW, silent(), judge=GOOD)
    assert result.reward == -1.0
    assert result.branch == "miss"


@pytest.mark.parametrize(
    "state", [StateClass.EARLY, StateClass.ALREADY_SAID, StateClass.NO_EVIDENCE_GUARD]
)
def test_false_fire_scores_minus_one(state):
    result = reward(state, fire(), judge=GOOD)
    assert result.reward == -1.0
    assert result.branch == "false_fire"


def test_repeated_in_window_fire_is_penalized_like_a_mistimed_one():
    """`already_said` is an in-window state whose gold action is SILENT, so a second
    intervention costs exactly what an early one does."""
    repeat = reward(StateClass.ALREADY_SAID, fire(), judge=GOOD)
    early = reward(StateClass.EARLY, fire(), judge=GOOD)
    assert repeat.reward == early.reward == -1.0


def test_malformed_generation_scores_minus_one_on_every_class():
    truncated = Trajectory(
        observe_closed=True, answer_closed=False, answer="Grab mi", has_eos=False
    )
    for state in StateClass:
        result = reward(state, truncated, judge=GOOD)
        assert result.reward == -1.0
        assert result.branch == "malformed"


def test_stalling_does_not_earn_the_silence_reward():
    """Running out of tokens before closing <observe> is a failure, not silence."""
    stalled = Trajectory(observe_closed=False, answer_closed=False, answer="", has_eos=False)
    assert reward(StateClass.EARLY, stalled).reward == -1.0


def test_mistimed_fire_never_reaches_the_judge():
    calls: list[str] = []

    def judge(answer: str) -> bool:
        calls.append(answer)
        return True

    reward(StateClass.EARLY, fire(), judge=judge)
    assert calls == []

    reward(StateClass.ACTIVE_IN_WINDOW, fire(), judge=judge)
    assert len(calls) == 1


def test_reward_is_binary_everywhere():
    judges = [GOOD, BAD]
    trajectories = [fire(), silent()]
    values = {
        reward(state, trajectory, judge=judge).reward
        for state in StateClass
        for trajectory in trajectories
        for judge in judges
    }
    assert values <= {1.0, -1.0}


def test_gold_action_mapping():
    assert StateClass.ACTIVE_IN_WINDOW.gold_action is Action.FIRE
    assert StateClass.EARLY.gold_action is Action.SILENT
    assert StateClass.ALREADY_SAID.gold_action is Action.SILENT
    assert StateClass.NO_EVIDENCE_GUARD.gold_action is Action.SILENT


def test_trajectory_parsing_from_raw_completion():
    completion = (
        "<observe> milk carton </observe>"
        "<reason> it is in view and unresolved </reason>"
        "<answer> Grab milk this trip. </answer>"
    )
    trajectory = Trajectory.parse(completion, has_eos=True)
    assert trajectory.well_formed
    assert trajectory.action is Action.FIRE

    trajectory = Trajectory.parse(
        "<observe> milk carton </observe><answer> NO_REMINDER </answer>", has_eos=True
    )
    assert trajectory.action is Action.SILENT


def test_group_advantages_centre_and_scale():
    advantages = group_advantages([1.0, 1.0, -1.0, -1.0])
    assert pytest.approx(sum(advantages), abs=1e-6) == 0.0
    assert advantages[0] > 0 > advantages[2]


def test_uniform_group_yields_no_gradient():
    assert group_advantages([1.0, 1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0, 0.0]
