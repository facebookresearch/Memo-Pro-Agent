# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Streaming-compatible (OFFLINE_V2_COMPATIBLE) CPU regression tests.

Covers the runtime slice only: compatible parse/fire decisions with
termination metadata, chat-only backend fallback, per-step metadata, health
accounting, and ledger/chunk continuity after a degraded turn. No GPU, no
model server, no prompt-string changes.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from memoproagent.agent import prompts
from memoproagent.agent.runner import (
    OFFLINE_V2_COMPATIBLE,
    StreamingConfig,
    StreamingRunner,
    compatible_health,
)
from memoproagent.agent.runner import PolicyConfig
from memoproagent.backends.base import ChatResult
from memoproagent.data.schema import Clip


def R(text, finish=None, stop=None, prompt=None, completion=None, total=None):
    return ChatResult(
        text=text,
        finish_reason=finish,
        stop_reason=stop,
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


class StubGate:
    """Minimal gate double: the runner only reads evidence/raw/retrieval."""

    def __init__(self, evidence="the milk carton needs restocking"):
        self.evidence = evidence
        self.calls: list[str] = []

    def __call__(self, observation):
        self.calls.append(observation)
        return SimpleNamespace(evidence=self.evidence, raw="FACT: x", retrieval=None)


class MetaBackend:
    """Structured backend: pops queued ChatResults per round.

    Observe and final calls alternate starting with observe, so routing uses
    call order rather than message content (the prefix may be empty).
    """

    name = "meta-scripted"
    max_images_per_request = 32

    def __init__(self, observes, finals):
        self.observes: deque = deque(observes)
        self.finals: deque = deque(finals)
        self.n_calls = 0

    def _pop(self, queue):
        return queue[0] if len(queue) == 1 else queue.popleft()

    def chat_with_metadata(self, messages, config):
        self.n_calls += 1
        queue = self.observes if self.n_calls % 2 == 1 else self.finals
        item = self._pop(queue)
        return item if isinstance(item, ChatResult) else R(item)


class StringBackend:
    """Chat-only backend (no chat_with_metadata): unknown metadata fallback."""

    name = "string-scripted"
    max_images_per_request = 32

    def __init__(self, observes, finals):
        self.observes: deque = deque(observes)
        self.finals: deque = deque(finals)
        self.n_calls = 0

    def chat(self, messages, config):
        self.n_calls += 1
        queue = self.observes if self.n_calls % 2 == 1 else self.finals
        return queue[0] if len(queue) == 1 else queue.popleft()


OBS_CLOSED = R("<observe> milk carton </observe>", finish="stop", stop="</observe>")
FINAL_OK = "<reason> carton in view </reason><answer> Grab milk. </answer>"


def make_runner(policy, evidence="the milk carton needs restocking"):
    return StreamingRunner(
        policy, StubGate(evidence), StreamingConfig(frame_size=None, policy=PolicyConfig())
    )


def decide(policy, observe, final):
    runner = make_runner(policy)
    return runner._decide_streaming([], 1.0, [])


def test_streaming_default_frame_size_is_384x288():
    assert StreamingConfig().frame_size == (384, 288)
    # Explicit overrides (including None) are still honored.
    assert StreamingConfig(frame_size=None).frame_size is None


def test_bare_no_reminder_does_not_fire():
    step = decide(MetaBackend([OBS_CLOSED], [R("NO_REMINDER", finish="stop")]), None, None)
    assert step["fired"] is False
    assert "answer_unparsed" in step["degradation_reasons"]
    assert "reason_unparsed" in step["degradation_reasons"]


def test_reason_only_output_does_not_fire():
    step = decide(MetaBackend([OBS_CLOSED], [R("<reason> in view", finish="stop")]), None, None)
    assert step["fired"] is False
    assert step["reason_parsed"] is True
    assert "answer_unparsed" in step["degradation_reasons"]


def test_missing_reason_with_valid_answer_still_fires():
    """Missing <reason> is a diagnostic, not a fire-disqualification condition."""
    step = decide(
        MetaBackend([OBS_CLOSED], [R("<answer> Grab milk. </answer>", finish="stop")]),
        None,
        None,
    )
    assert step["fired"] is True
    assert step["reason_parsed"] is False
    assert "reason_unparsed" in step["degradation_reasons"]


def test_empty_answer_does_not_fire():
    step = decide(
        MetaBackend([OBS_CLOSED], [R("<reason> r </reason><answer>   </answer>", finish="stop")]),
        None,
        None,
    )
    assert step["fired"] is False
    assert "answer_empty" in step["degradation_reasons"]


def test_open_observe_still_retrieves_but_does_not_fire():
    policy = MetaBackend(
        [R("<observe> milk carton", finish="stop")], [R(FINAL_OK, finish="stop")]
    )
    runner = make_runner(policy)
    step = runner._decide_streaming([], 1.0, [])
    assert step["observe_closed"] is False
    assert step["fired"] is False
    assert "observe_unclosed" in step["degradation_reasons"]
    # Retrieval and final generation still executed after the incomplete Observe.
    assert len(runner.gate.calls) == 1


def test_observe_length_only_flags_does_not_disqualify():
    step = decide(
        MetaBackend(
            [R("<observe> milk carton </observe>", finish="length")],
            [R(FINAL_OK, finish="stop")],
        ),
        None,
        None,
    )
    assert "observe_length_truncated" in step["degradation_reasons"]
    assert step["fired"] is True


def test_final_length_with_closed_tags_does_not_fire():
    step = decide(
        MetaBackend([OBS_CLOSED], [R(FINAL_OK, finish="length")]), None, None
    )
    assert step["fired"] is False
    assert "final_length_truncated" in step["degradation_reasons"]


def test_eos_final_with_valid_tags_fires():
    step = decide(MetaBackend([OBS_CLOSED], [R(FINAL_OK, finish="eos")]), None, None)
    assert step["fired"] is True
    assert "final_length_truncated" not in step["degradation_reasons"]


def test_unknown_metadata_fires_without_invented_facts():
    step = decide(MetaBackend([R("<observe> milk </observe>")], [R(FINAL_OK)]), None, None)
    assert step["observe_finish_reason"] is None
    assert step["final_finish_reason"] is None
    assert step["fired"] is True


def test_string_backend_fallback_uses_unknown_metadata():
    policy = StringBackend(["<observe> milk carton </observe>"], [FINAL_OK])
    step = decide(policy, None, None)
    assert step["observe_finish_reason"] is None
    assert step["final_finish_reason"] is None
    assert step["fired"] is True
    assert step["degradation_reasons"] == []


def test_per_step_termination_and_token_metadata_recorded():
    policy = MetaBackend(
        [R("<observe> milk </observe>", finish="stop", stop="</observe>",
           prompt=10, completion=5, total=15)],
        [R(FINAL_OK, finish="stop", stop=None, prompt=20, completion=8, total=28)],
    )
    step = decide(policy, None, None)
    assert step["observe_finish_reason"] == "stop"
    assert step["observe_stop_reason"] == "</observe>"
    assert (step["observe_prompt_tokens"], step["observe_completion_tokens"],
            step["observe_total_tokens"]) == (10, 5, 15)
    assert step["final_finish_reason"] == "stop"
    assert (step["final_prompt_tokens"], step["final_completion_tokens"],
            step["final_total_tokens"]) == (20, 8, 28)
    assert step["protocol_mode"] == OFFLINE_V2_COMPATIBLE


def test_chat_result_contract_violation_raises():
    class BadBackend:
        name = "bad"
        max_images_per_request = 32

        def chat_with_metadata(self, messages, config):
            return 123  # neither ChatResult nor str

        def chat(self, messages, config):
            return "x"

    runner = make_runner(BadBackend())
    with pytest.raises(TypeError):
        runner._decide_streaming([], 1.0, [])


def test_compatible_final_parser_exact_reason_strings():
    _, _, rp, ap, fired, degradation = prompts.parse_compatible_final_output(
        "<answer> Grab milk. </answer>", observe_closed=True, final_finish_reason=None
    )
    assert (rp, ap, fired) == (False, True, True)
    assert degradation == ["reason_unparsed"]


def test_compatible_health_counts_degraded_not_failures():
    steps = [
        {"degradation_reasons": []},
        {"degradation_reasons": ["observe_unclosed"]},
        {"degradation_reasons": ["final_length_truncated", "answer_unparsed"]},
    ]
    health = compatible_health(steps)
    assert health["protocol_mode"] == OFFLINE_V2_COMPATIBLE
    assert health["n_steps"] == 3
    assert health["degraded_turn_count"] == 2
    assert health["degradation_count"] == 3
    assert health["degradation_reasons"] == {
        "observe_unclosed": 1,
        "final_length_truncated": 1,
        "answer_unparsed": 1,
    }
    assert health["protocol_failure_count"] == 0


def _write_clip_video(path: Path, duration: float = 4.0) -> None:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    for index in range(int(duration * 10)):
        writer.write(np.full((48, 64, 3), index % 256, dtype=np.uint8))
    writer.release()


def test_multi_turn_continuation_ledger_and_chunk(tmp_path):
    """A degraded turn commits NO_REMINDER; later turns still fire; chunk rolls."""
    video = tmp_path / "clip.mp4"
    _write_clip_video(video)
    clip = Clip(
        uid="stream-compat-0001",
        video_path=str(video),
        caption_log="",
        duration=4.0,
        windows=(),
        target="",
        closes_when="",
        targets=(),
    )
    finals = [
        R("<reason> r </reason><answer> Grab milk. </answer>", finish="stop"),
        R("<reason> r </reason><answer> Grab mi", finish="stop"),  # truncated
        R("<reason> r </reason><answer> Grab bread. </answer>", finish="stop"),
        R("<reason> r </reason><answer> NO_REMINDER </answer>", finish="stop"),
    ]
    policy = MetaBackend([OBS_CLOSED], finals)
    runner = StreamingRunner(
        policy,
        StubGate(),
        StreamingConfig(frame_size=None, chunk_turns=3, keep_turns=1, policy=PolicyConfig()),
    )
    run = runner.run(clip)
    assert len(run.steps) == 4
    assert [f.answer for f in run.fires] == ["Grab milk.", "Grab bread."]
    assert [s["context_turns"] for s in run.steps] == [0, 1, 2, 1]
    assert "answer_unparsed" in run.steps[1]["degradation_reasons"]
    assert run.health is not None
    assert run.health["degraded_turn_count"] == 1
    assert run.health["protocol_failure_count"] == 0
    # Trace round-trips through the backward-compatible health field.
    assert run.to_json()["health"] == run.health


@pytest.mark.parametrize("structured", [False, True])
def test_non_string_backend_text_is_fatal(structured):
    policy = (MetaBackend([R(None)], [R(FINAL_OK)]) if structured
              else StringBackend([None], [FINAL_OK]))
    with pytest.raises(TypeError, match="(non-string text|chat must return str)"):
        make_runner(policy)._decide_streaming([], 1.0, [])


def test_gate_failure_is_not_compatible_silence():
    policy = MetaBackend([OBS_CLOSED], [R(FINAL_OK)])
    runner = make_runner(policy)

    def failed_gate(observation):
        raise RuntimeError("gate unavailable")

    runner.gate = failed_gate
    with pytest.raises(RuntimeError, match="gate unavailable"):
        runner._decide_streaming([], 1.0, [])
    assert policy.n_calls == 1


def test_backend_failure_is_not_compatible_silence():
    class UnavailableBackend:
        name = "unavailable"
        max_images_per_request = 32

        def chat(self, messages, config):
            raise ConnectionError("backend unavailable")

    with pytest.raises(ConnectionError, match="backend unavailable"):
        make_runner(UnavailableBackend())._decide_streaming([], 1.0, [])
