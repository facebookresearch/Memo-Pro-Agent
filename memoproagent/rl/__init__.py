# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""GRPO post-training of the generator."""
from __future__ import annotations

from .reward import (
    Action,
    RewardBreakdown,
    StateClass,
    Trajectory,
    group_advantages,
    reward,
)

__all__ = [
    "Action",
    "RewardBreakdown",
    "StateClass",
    "Trajectory",
    "group_advantages",
    "reward",
]
