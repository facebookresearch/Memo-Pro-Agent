# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""The MemoProAgent decision loop, in both deployment paradigms.

Each decision is the same two-round trajectory (Algorithm 1):

    observe -> retrieve -> reason -> answer

Round one asks the policy to name the object(s) in focus. The harness retrieves
memory for those objects and runs the gate. Round two continues the *same*
conversation with the gated evidence and the reminders already spoken, and the
policy either emits one grounded reminder or ``NO_REMINDER``.

The paradigms differ only in the visual context they carry:

* :class:`WindowedRunner` -- a decision every 2 s over the four most recent
  frames. Each decision is visually independent.
* :class:`StreamingRunner` -- a decision every 1 s, appending the new frame to a
  persistent context. The context holds at most ``chunk_turns`` turns; at the
  limit the latest ``keep_turns`` are carried into a fresh chunk.

Prior interventions cross both boundaries explicitly, through ``Already spoken``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from ..backends.base import ChatResult, GenerationConfig, Message, VisionChatBackend
from ..data.frames import DecisionFrames, FrameSource, streaming_decisions, windowed_decisions
from ..data.schema import Clip, ClipRun, Fire
from . import prompts
from .gate import Gate

OFFLINE_V2_COMPATIBLE = "offline_v2_compatible"


def _chat_result(policy: VisionChatBackend, messages: list[Message], config: GenerationConfig) -> ChatResult:
    """Structured chat with a chat-only fallback.

    Backends exposing a callable ``chat_with_metadata`` return their own
    :class:`ChatResult`. Chat-only (e.g. scripted) backends fall back to
    ``chat()`` with explicitly unknown termination metadata -- never invented
    as ``"length"``/``"stop"``/``"eos"``. A non-string, non-ChatResult return
    is a contract violation and raises.
    """
    metadata_fn = getattr(policy, "chat_with_metadata", None)
    if callable(metadata_fn):
        result = metadata_fn(messages, config)
        if isinstance(result, ChatResult):
            if not isinstance(result.text, str):
                raise TypeError("chat_with_metadata returned ChatResult with non-string text")
            return result
        if isinstance(result, str):
            return ChatResult(text=result)
        raise TypeError(
            f"chat_with_metadata must return ChatResult or str, got {type(result).__name__}"
        )
    text = policy.chat(messages, config)
    if not isinstance(text, str):
        raise TypeError(f"chat must return str, got {type(text).__name__}")
    return ChatResult(text=text)


def compatible_health(steps: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-run compatible health summary over streaming steps.

    Counts degraded (but completed) turns separately from fatal failures: a
    successful run always reports ``protocol_failure_count`` 0. Failed runs
    raise instead of producing a summary, so fatal tool/backend/config/decode
    errors never masquerade as benign degradation.
    """
    reason_counts: dict[str, int] = {}
    degraded = 0
    for step in steps:
        reasons = step.get("degradation_reasons") or []
        if reasons:
            degraded += 1
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "protocol_mode": OFFLINE_V2_COMPATIBLE,
        "n_steps": len(steps),
        "degraded_turn_count": degraded,
        "degradation_count": sum(reason_counts.values()),
        "degradation_reasons": reason_counts,
        "protocol_failure_count": 0,
    }


@dataclass(frozen=True)
class PolicyConfig:
    observe_max_tokens: int = 48
    answer_max_tokens: int = 160
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    observe_stop: bool = False
    """Canonical-profile stop preservation. When True the first (observe) call
    requests ``stop=('</observe>',)`` with ``include_stop_str_in_output=True``
    so a vLLM server returns the delimiter instead of stripping it. The second
    (reason/answer) call never takes a stop. Default False keeps payloads
    byte-identical for generic servers."""


def _normalize_frame_size(value: Any) -> tuple[int, int] | None:
    """Accept a ``(width, height)`` tuple or a YAML list; pass through None."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (value[0], value[1])
    return value  # FrameSource validation raises the precise error.


@dataclass(frozen=True)
class WindowedConfig:
    stride: float = 2.0
    n_frames: int = 4
    frame_gap: float = 1.0
    frame_size: tuple[int, int] | None = None
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_size", _normalize_frame_size(self.frame_size))


@dataclass(frozen=True)
class StreamingConfig:
    stride: float = 1.0
    frame_size: tuple[int, int] | None = (384, 288)
    chunk_turns: int = 30
    """Maximum turns held in the visual context before a new chunk starts."""
    keep_turns: int = 4
    """Turns carried over when a chunk rolls. Keeping a few preserves the
    immediate visual continuity that makes "is it still in view?" answerable."""
    policy: PolicyConfig = field(default_factory=PolicyConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_size", _normalize_frame_size(self.frame_size))


def already_spoken(fires: Sequence[Fire]) -> str:
    return "; ".join(f.answer for f in fires) or "none"


class _Runner:
    """Shared two-round decision logic."""

    streaming = False

    def __init__(self, policy: VisionChatBackend, gate: Gate, policy_config: PolicyConfig) -> None:
        self.policy = policy
        self.gate = gate
        self.config = policy_config

    @property
    def system_prompt(self) -> str:
        return prompts.STREAMING_POLICY_SYSTEM if self.streaming else prompts.WINDOWED_POLICY_SYSTEM

    def _decide(
        self, prefix: list[Message], timestamp: float, fires: Sequence[Fire]
    ) -> dict[str, Any]:
        """Run one two-round decision on an already-built message prefix."""
        observe_raw = self.policy.chat(
            prefix,
            GenerationConfig(
                max_tokens=self.config.observe_max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                seed=self.config.seed,
                stop=("</observe>",) if self.config.observe_stop else None,
                include_stop_str_in_output=self.config.observe_stop,
            ),
        )
        observe_action = prompts.trim_observe(observe_raw)
        observation, observe_closed = prompts.parse_observe(observe_action)

        gate_result = self.gate(observation)
        history = already_spoken(fires)
        decide_text = prompts.decide_user(
            timestamp,
            history,
            gate_result.evidence or "none",
            streaming=self.streaming,
        )

        messages = prefix + [
            Message(role="assistant", text=observe_action),
            Message(role="user", text=decide_text),
        ]
        answer_raw = self.policy.chat(
            messages,
            GenerationConfig(
                max_tokens=self.config.answer_max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                seed=self.config.seed,
            ),
        )
        reason, answer, answer_closed = prompts.parse_decision(answer_raw)
        fired = prompts.is_fire(
            answer, observe_closed=observe_closed, answer_closed=answer_closed
        )
        return {
            "t": round(timestamp, 2),
            "observe": observation,
            "observe_raw": observe_raw,
            "observe_closed": observe_closed,
            "evidence": gate_result.evidence,
            "gate_raw": gate_result.raw,
            "retrieval": gate_result.retrieval,
            "already_spoken": history,
            "reason": reason,
            "answer": answer,
            "answer_raw": answer_raw,
            "answer_closed": answer_closed,
            "fired": fired,
        }

    def _decide_streaming(
        self, prefix: list[Message], timestamp: float, fires: Sequence[Fire]
    ) -> dict[str, Any]:
        """Streaming two-round decision with termination-aware degradation.

        Mirrors the legacy OFFLINE_V2_COMPATIBLE path: retrieval and final
        generation still run after an incomplete/truncated Observe (empty
        evidence becomes ``"none"``); the fire rule additionally requires
        ``final_finish_reason != "length"``; a bad step commits NO_REMINDER and
        continues rather than dropping the run or raising. Windowed behavior is
        untouched -- it keeps using :meth:`_decide`.
        """
        observe_result = _chat_result(
            self.policy,
            prefix,
            GenerationConfig(
                max_tokens=self.config.observe_max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                seed=self.config.seed,
                stop=("</observe>",) if self.config.observe_stop else None,
                include_stop_str_in_output=self.config.observe_stop,
            ),
        )
        observe_raw = observe_result.text
        observe_action = prompts.trim_observe(observe_raw)
        observation, observe_closed = prompts.parse_observe(observe_action)
        degradation = prompts.compatible_observe_reasons(
            observe_closed=observe_closed,
            observe_finish_reason=observe_result.finish_reason,
        )

        gate_result = self.gate(observation)
        history = already_spoken(fires)
        decide_text = prompts.decide_user(
            timestamp,
            history,
            gate_result.evidence or "none",
            streaming=True,
        )

        messages = prefix + [
            Message(role="assistant", text=observe_action),
            Message(role="user", text=decide_text),
        ]
        final_result = _chat_result(
            self.policy,
            messages,
            GenerationConfig(
                max_tokens=self.config.answer_max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                seed=self.config.seed,
            ),
        )
        answer_raw = final_result.text
        (
            reason,
            answer,
            reason_parsed,
            answer_parsed,
            fired,
            final_degradation,
        ) = prompts.parse_compatible_final_output(
            answer_raw,
            observe_closed=observe_closed,
            final_finish_reason=final_result.finish_reason,
        )
        degradation.extend(final_degradation)
        return {
            "t": round(timestamp, 2),
            "protocol_mode": OFFLINE_V2_COMPATIBLE,
            "observe": observation,
            "observe_raw": observe_raw,
            "observe_closed": observe_closed,
            "observe_finish_reason": observe_result.finish_reason,
            "observe_stop_reason": observe_result.stop_reason,
            "observe_prompt_tokens": observe_result.prompt_tokens,
            "observe_completion_tokens": observe_result.completion_tokens,
            "observe_total_tokens": observe_result.total_tokens,
            "evidence": gate_result.evidence,
            "gate_raw": gate_result.raw,
            "retrieval": gate_result.retrieval,
            "already_spoken": history,
            "reason": reason,
            "reason_parsed": reason_parsed,
            "answer": answer,
            "answer_raw": answer_raw,
            "answer_closed": answer_parsed,
            "final_finish_reason": final_result.finish_reason,
            "final_stop_reason": final_result.stop_reason,
            "final_prompt_tokens": final_result.prompt_tokens,
            "final_completion_tokens": final_result.completion_tokens,
            "final_total_tokens": final_result.total_tokens,
            "degradation_reasons": degradation,
            "fired": fired,
        }


class WindowedRunner(_Runner):
    """Sliding-window paradigm: one decision every ``stride`` seconds."""

    streaming = False

    def __init__(
        self, policy: VisionChatBackend, gate: Gate, config: WindowedConfig | None = None
    ) -> None:
        self.window = config or WindowedConfig()
        super().__init__(policy, gate, self.window.policy)

    def run(self, clip: Clip, config_label: str = "memoproagent") -> ClipRun:
        run = ClipRun(
            uid=clip.uid,
            paradigm="windowed",
            config_label=config_label,
            windows=clip.windows,
        )
        with FrameSource(clip.video_path, frame_size=self.window.frame_size) as source:
            for decision in windowed_decisions(
                source,
                clip.duration,
                stride=self.window.stride,
                n_frames=self.window.n_frames,
                frame_gap=self.window.frame_gap,
            ):
                prefix = self._prefix(decision)
                step = self._decide(prefix, decision.t, run.fires)
                step["frame_times"] = list(decision.times)
                if self.window.frame_size is not None:
                    step["policy_frame_size"] = list(self.window.frame_size)
                run.steps.append(step)
                if step["fired"]:
                    run.fires.append(
                        Fire(
                            t=decision.t,
                            answer=step["answer"],
                            observe=step["observe"],
                            reason=step["reason"],
                            evidence=step["evidence"],
                        )
                    )
        return run

    def _prefix(self, decision: DecisionFrames) -> list[Message]:
        return [
            Message(role="system", text=self.system_prompt),
            Message(
                role="user",
                text=prompts.observe_user(decision.t, streaming=False),
                images=list(decision.images),
            ),
        ]


class StreamingRunner(_Runner):
    """Streaming paradigm: one new frame per step, appended to a chunked context."""

    streaming = True

    def __init__(
        self, policy: VisionChatBackend, gate: Gate, config: StreamingConfig | None = None
    ) -> None:
        self.stream = config or StreamingConfig()
        if self.stream.keep_turns >= self.stream.chunk_turns:
            raise ValueError("keep_turns must be smaller than chunk_turns")
        super().__init__(policy, gate, self.stream.policy)

    def run(self, clip: Clip, config_label: str = "memoproagent") -> ClipRun:
        run = ClipRun(
            uid=clip.uid,
            paradigm="streaming",
            config_label=config_label,
            windows=clip.windows,
        )
        # A turn is one frame plus the short answer the policy gave to it. Only the
        # answer is retained, not the observe/reason spans: the context is there to
        # carry what was *seen* and what was *said*, not the intermediate reasoning.
        turns: list[tuple[bytes, str]] = []

        with FrameSource(
            clip.video_path, frame_size=self.stream.frame_size, seek_by_frame=True
        ) as source:
            for decision in streaming_decisions(source, clip.duration, stride=self.stream.stride):
                if len(turns) >= self.stream.chunk_turns:
                    turns = turns[-self.stream.keep_turns :]

                prefix = self._prefix(turns, decision)
                step = self._decide_streaming(prefix, decision.t, run.fires)
                step["context_turns"] = len(turns)
                if self.stream.frame_size is not None:
                    step["policy_frame_size"] = list(self.stream.frame_size)
                run.steps.append(step)

                short_answer = (
                    step["answer"] if step["fired"] else prompts.NO_REMINDER
                )
                turns.append((decision.images[0], short_answer))

                if step["fired"]:
                    run.fires.append(
                        Fire(
                            t=decision.t,
                            answer=step["answer"],
                            observe=step["observe"],
                            reason=step["reason"],
                            evidence=step["evidence"],
                        )
                    )
        run.health = compatible_health(run.steps)
        return run

    def _prefix(
        self, turns: Sequence[tuple[bytes, str]], decision: DecisionFrames
    ) -> list[Message]:
        messages = [Message(role="system", text=self.system_prompt)]
        for image, answer in turns:
            # Historical user turns carry the frame only. Repeating the Observe
            # instruction at every past step would fill the context with
            # boilerplate and blur which frame is actually current.
            messages.append(Message(role="user", text="", images=[image]))
            messages.append(Message(role="assistant", text=answer))
        messages.append(
            Message(
                role="user",
                text=prompts.observe_user(decision.t, streaming=True),
                images=list(decision.images),
            )
        )
        return messages


def iter_runs(
    runner: WindowedRunner | StreamingRunner,
    clips: Sequence[Clip],
    config_label: str = "memoproagent",
) -> Iterator[ClipRun]:
    for clip in clips:
        yield runner.run(clip, config_label=config_label)
