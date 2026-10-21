# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""The GRPO reward, exactly as defined in the paper.

Let :math:`y_t` be the action at step *t* and *a* the reminder emitted. The gold
action depends on the trigger window **and** on what has already been said:

.. math::

    g_t = \\textsc{fire} \\iff t \\in W \\;\\wedge\\; \\nexists\\, t' < t
          \\text{ with } y_{t'} = \\textsc{fire}

so a repeated in-window intervention is penalized exactly like a mistimed one.
The reward is binary:

.. math::

    R(y_t, g_t, a) = \\begin{cases}
      +1 & y_t = g_t = \\textsc{silent} \\\\
      +1 & y_t = g_t = \\textsc{fire} \\wedge \\textsc{Judge}(a) = \\textsc{good} \\\\
      -1 & \\text{otherwise}
    \\end{cases}

The final branch absorbs every unsuccessful outcome: a miss, a false fire, a
low-quality reminder, and a malformed generation. A wrongly timed fire is never
sent to the judge.

Binary rather than shaped, on purpose: a graded reward gives the policy a
gradient to climb that is not "be right", and partial credit for partially
correct reminders is exactly the surface reward hacking exploits.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

from ..agent import prompts

REWARD_CORRECT = 1.0
REWARD_INCORRECT = -1.0


class Action(str, Enum):
    FIRE = "fire"
    SILENT = "silent"


class StateClass(str, Enum):
    """The four training state classes, and the gold action each implies."""

    ACTIVE_IN_WINDOW = "active_inwin"
    """Inside the window, nothing said yet -> gold is FIRE."""

    EARLY = "early"
    """Before the window -> gold is SILENT."""

    ALREADY_SAID = "already_said"
    """Inside the window but an intervention already happened -> gold is SILENT."""

    NO_EVIDENCE_GUARD = "none_guard"
    """The gate abstained -> gold is SILENT."""

    @property
    def gold_action(self) -> Action:
        return Action.FIRE if self is StateClass.ACTIVE_IN_WINDOW else Action.SILENT


@dataclass(frozen=True)
class Trajectory:
    """What the policy produced at one decision step."""

    observe_closed: bool
    answer_closed: bool
    answer: str
    has_eos: bool

    @property
    def well_formed(self) -> bool:
        """A trajectory counts only if both protocol rounds closed cleanly.

        Truncation is a failure, not silence. Without this, running out of tokens
        before writing anything would earn the +1 for correct silence on the three
        silent classes -- that is, the policy could learn to stall.
        """
        if not self.has_eos or not self.observe_closed:
            return False
        if prompts.is_no_reminder(self.answer):
            return True
        return self.answer_closed and bool(self.answer)

    @property
    def action(self) -> Action:
        if not self.answer or prompts.is_no_reminder(self.answer):
            return Action.SILENT
        return Action.FIRE

    @classmethod
    def parse(cls, completion: str, has_eos: bool) -> "Trajectory":
        observe_text = prompts.trim_observe(completion)
        _, observe_closed = prompts.parse_observe(observe_text)
        _, answer, answer_closed = prompts.parse_decision(completion)
        return cls(
            observe_closed=observe_closed,
            answer_closed=answer_closed,
            answer=answer,
            has_eos=has_eos,
        )


@dataclass
class RewardBreakdown:
    reward: float
    branch: str
    state_class: str
    action: str
    gold_action: str
    well_formed: bool
    judged: bool = False
    judge_good: bool | None = None


JudgeFn = Callable[[str], bool]
"""Maps a reminder to whether the judge calls it GOOD."""


def reward(
    state_class: StateClass | str,
    trajectory: Trajectory,
    judge: JudgeFn | None = None,
) -> RewardBreakdown:
    """Score one rollout.

    `judge` is called at most once, and only for a correctly timed fire. Pass
    None to score timing alone (useful for cheap offline checks); a fire that
    would need judging then scores as incorrect rather than being assumed good.
    """
    state_class = StateClass(state_class)
    gold = state_class.gold_action

    if not trajectory.well_formed:
        return RewardBreakdown(
            reward=REWARD_INCORRECT,
            branch="malformed",
            state_class=state_class.value,
            action=trajectory.action.value,
            gold_action=gold.value,
            well_formed=False,
        )

    action = trajectory.action

    if action is not gold:
        branch = "miss" if gold is Action.FIRE else "false_fire"
        return RewardBreakdown(
            reward=REWARD_INCORRECT,
            branch=branch,
            state_class=state_class.value,
            action=action.value,
            gold_action=gold.value,
            well_formed=True,
        )

    if gold is Action.SILENT:
        return RewardBreakdown(
            reward=REWARD_CORRECT,
            branch="correct_silence",
            state_class=state_class.value,
            action=action.value,
            gold_action=gold.value,
            well_formed=True,
        )

    good = bool(judge(trajectory.answer)) if judge is not None else False
    return RewardBreakdown(
        reward=REWARD_CORRECT if good else REWARD_INCORRECT,
        branch="good_fire" if good else "low_quality_fire",
        state_class=state_class.value,
        action=action.value,
        gold_action=gold.value,
        well_formed=True,
        judged=True,
        judge_good=good,
    )


def group_advantages(rewards: Sequence[float], eps: float = 1e-4) -> list[float]:
    """GRPO group-normalized advantage over the G rollouts of one state.

    When every rollout in a group gets the same reward the advantage is zero and
    the group contributes no gradient -- expected, and the reason G should not be
    too small on states the policy already answers consistently.
    """
    values = [float(r) for r in rewards]
    if len(values) <= 1:
        return [0.0] * len(values)
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = variance**0.5
    return [(v - mean) / (std + eps) for v in values]
