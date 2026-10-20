"""The ProPerAgent harness: prompts, gate, and the two paradigm runners."""
from __future__ import annotations

from .gate import Gate, GateConfig, GateProtocolError, GateResult
from .prompts import (
    GATE_SYSTEM,
    NO_REMINDER,
    STREAMING_POLICY_SYSTEM,
    WINDOWED_POLICY_SYSTEM,
    is_fire,
    is_no_reminder,
    parse_decision,
    parse_gate,
    parse_observe,
)
from .runner import (
    PolicyConfig,
    StreamingConfig,
    StreamingRunner,
    WindowedConfig,
    WindowedRunner,
    iter_runs,
)

__all__ = [
    "GATE_SYSTEM",
    "Gate",
    "GateConfig",
    "GateProtocolError",
    "GateResult",
    "NO_REMINDER",
    "PolicyConfig",
    "STREAMING_POLICY_SYSTEM",
    "StreamingConfig",
    "StreamingRunner",
    "WINDOWED_POLICY_SYSTEM",
    "WindowedConfig",
    "WindowedRunner",
    "is_fire",
    "is_no_reminder",
    "iter_runs",
    "parse_decision",
    "parse_gate",
    "parse_observe",
]
