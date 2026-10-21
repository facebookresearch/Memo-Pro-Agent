# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Canonical-policy focused tests: observe-stop preservation + policy frame size.

Covers `GenerationConfig.include_stop_str_in_output`, the OpenAICompatChat
stop restore, `PolicyConfig.observe_stop` first/second configs, FrameSource
`frame_size`, and judge-PNG independence. Nothing else is covered here.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from memoproagent.agent import runner as runner_module
from memoproagent.agent.runner import (
    PolicyConfig,
    StreamingConfig,
    StreamingRunner,
    WindowedConfig,
    WindowedRunner,
)
from memoproagent.backends.base import GenerationConfig, Message
from memoproagent.backends.openai_compat import (
    BackendError,
    OpenAICompatChat,
    OpenAICompatConfig,
)
from memoproagent.data.frames import FrameSource
from memoproagent.data.schema import Clip
from memoproagent.eval.judge import extract_judge_frames

cv2 = pytest.importorskip("cv2")

OBSERVE_STOP = "</observe>"
CANONICAL_SIZE = (384, 288)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _CapturingChat(OpenAICompatChat):
    """Stub transport: capture the payload, replay one scripted choice."""

    def __init__(self, config: OpenAICompatConfig, choice: dict) -> None:
        super().__init__(config)
        self.payloads: list[dict] = []
        self._choice = choice

    def _post(self, path: str, payload: dict) -> dict:
        self.payloads.append(payload)
        return {"choices": [self._choice]}


def _chat_config(**kw) -> OpenAICompatConfig:
    base: dict = {"model": "m", "base_url": "http://127.0.0.1:1/v1"}
    base.update(kw)
    return OpenAICompatConfig(**base)


def _gen(**kw) -> GenerationConfig:
    base: dict = {"max_tokens": 48, "temperature": 0.0}
    base.update(kw)
    return GenerationConfig(**base)


def _chat(config: OpenAICompatConfig, content: str, **choice_fields) -> _CapturingChat:
    choice = {"message": {"content": content}, "finish_reason": None}
    choice.update(choice_fields)
    return _CapturingChat(config, choice)


def _user() -> list[Message]:
    return [Message(role="user", text="hi")]


def test_defaults_keep_generic_payload_byte_identical():
    chat = _chat(_chat_config(), "<observe> milk </observe>")
    assert chat.chat(_user(), _gen()) == "<observe> milk </observe>"
    assert chat.payloads[0] == {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 48,
        "temperature": 0.0,
        "top_p": 1.0,
    }


def test_stop_without_flag_sends_stop_only():
    chat = _chat(_chat_config(), "x")
    chat.chat(_user(), _gen(stop=(OBSERVE_STOP,)))
    assert chat.payloads[0]["stop"] == [OBSERVE_STOP]
    assert "include_stop_str_in_output" not in chat.payloads[0]


def test_canonical_flag_sends_vllm_preserve():
    chat = _chat(_chat_config(), "x")
    chat.chat(_user(), _gen(stop=(OBSERVE_STOP,), include_stop_str_in_output=True))
    assert chat.payloads[0]["stop"] == [OBSERVE_STOP]
    assert chat.payloads[0]["include_stop_str_in_output"] is True


def test_restore_missing_stop_on_exact_metadata():
    chat = _chat(
        _chat_config(), "<observe> milk carton",
        finish_reason="stop", stop_reason=OBSERVE_STOP,
    )
    out = chat.chat(_user(), _gen(stop=(OBSERVE_STOP,), include_stop_str_in_output=True))
    assert out == "<observe> milk carton</observe>"


def test_already_closed_stop_not_duplicated():
    chat = _chat(
        _chat_config(), "<observe> milk </observe>",
        finish_reason="stop", stop_reason=OBSERVE_STOP,
    )
    out = chat.chat(_user(), _gen(stop=(OBSERVE_STOP,), include_stop_str_in_output=True))
    assert out == "<observe> milk </observe>"


def test_no_restore_without_exact_stop_metadata():
    cases = [
        {"finish_reason": "length", "stop_reason": OBSERVE_STOP},  # truncated
        {"finish_reason": "stop", "stop_reason": "<answer>"},  # other stop
        {"finish_reason": "stop"},  # missing stop_reason
        {"finish_reason": None, "stop_reason": OBSERVE_STOP},  # unknown reason
    ]
    for extra in cases:
        chat = _chat(_chat_config(), "<observe> milk carton", **extra)
        out = chat.chat(
            _user(), _gen(stop=(OBSERVE_STOP,), include_stop_str_in_output=True)
        )
        assert out == "<observe> milk carton", extra


def test_no_restore_when_flag_not_requested():
    chat = _chat(
        _chat_config(), "<observe> milk carton",
        finish_reason="stop", stop_reason=OBSERVE_STOP,
    )
    out = chat.chat(_user(), _gen(stop=(OBSERVE_STOP,)))
    assert out == "<observe> milk carton"


def test_canonical_requires_send_stop():
    chat = _chat(_chat_config(send_stop=False), "x")
    with pytest.raises(BackendError):
        chat.chat(_user(), _gen(stop=(OBSERVE_STOP,), include_stop_str_in_output=True))
    assert chat.payloads == []


def test_flag_without_stop_rejected():
    chat = _chat(_chat_config(), "x")
    with pytest.raises(BackendError):
        chat.chat(_user(), _gen(include_stop_str_in_output=True))


def test_extra_body_cannot_silently_drop_canonical_stop():
    chat = _chat(_chat_config(extra_body={"stop": ["<other>"]}), "x")
    with pytest.raises(BackendError):
        chat.chat(_user(), _gen(stop=(OBSERVE_STOP,), include_stop_str_in_output=True))


def test_extra_body_cannot_clear_preserve_flag():
    chat = _chat(_chat_config(extra_body={"include_stop_str_in_output": False}), "x")
    with pytest.raises(BackendError):
        chat.chat(_user(), _gen(stop=(OBSERVE_STOP,), include_stop_str_in_output=True))


def test_benign_extra_body_still_passes_through():
    chat = _chat(
        _chat_config(extra_body={"chat_template_kwargs": {"enable_thinking": False}}),
        "<observe> milk", finish_reason="stop", stop_reason=OBSERVE_STOP,
    )
    out = chat.chat(_user(), _gen(stop=(OBSERVE_STOP,), include_stop_str_in_output=True))
    assert out == "<observe> milk</observe>"
    assert chat.payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}


class _RecordingChat:
    name = "recording"
    max_images_per_request = 32

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.configs: list[GenerationConfig] = []

    def chat(self, messages, config: GenerationConfig) -> str:
        self.configs.append(config)
        return self.replies[len(self.configs) - 1]


def _gate():
    return lambda observation: SimpleNamespace(evidence="", retrieval=None, raw="")


_OBSERVE_REPLY = "<observe> milk carton </observe>"
_ANSWER_REPLY = "<reason> in view </reason><answer> NO_REMINDER </answer>"


def test_observe_stop_applies_to_first_call_only():
    policy = _RecordingChat([_OBSERVE_REPLY, _ANSWER_REPLY])
    runner = WindowedRunner(policy, _gate(), WindowedConfig(policy=PolicyConfig(observe_stop=True)))
    step = runner._decide(
        [Message(role="user", text="Begin the protocol with <observe>.")], 0.0, []
    )
    first, second = policy.configs
    assert first.stop == (OBSERVE_STOP,) and first.include_stop_str_in_output is True
    assert first.max_tokens == 48
    assert second.stop is None and second.include_stop_str_in_output is False
    assert second.max_tokens == 160
    assert step["observe_closed"] is True and step["fired"] is False


def test_observe_stop_default_off():
    policy = _RecordingChat([_OBSERVE_REPLY, _ANSWER_REPLY])
    runner = WindowedRunner(policy, _gate(), WindowedConfig(policy=PolicyConfig()))
    runner._decide([Message(role="user", text="Begin the protocol.")], 0.0, [])
    first, second = policy.configs
    assert first.stop is None and first.include_stop_str_in_output is False
    assert second.stop is None and second.include_stop_str_in_output is False


def test_frame_size_yaml_list_normalizes():
    assert WindowedConfig(frame_size=[384, 288]).frame_size == CANONICAL_SIZE
    assert StreamingConfig(frame_size=[384, 288]).frame_size == CANONICAL_SIZE
    assert WindowedConfig().frame_size is None
    assert StreamingConfig().frame_size == CANONICAL_SIZE
    assert StreamingConfig(frame_size=None).frame_size is None


def _clip() -> Clip:
    return Clip(
        uid="t", video_path="x.mp4", caption_log="c", duration=2.0,
        windows=(), target="milk", closes_when="", targets=(),
    )


def _fake_source_class(seen: dict):
    class FakeSource:
        def __init__(self, path, frame_size=None, *, seek_by_frame=False) -> None:
            seen["frame_size"] = frame_size
            seen["seek_by_frame"] = seek_by_frame

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

        def frame_at(self, t):
            return b"jpeg"

    return FakeSource


def test_windowed_runner_passes_frame_size(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(runner_module, "FrameSource", _fake_source_class(seen))
    policy = _RecordingChat([_OBSERVE_REPLY, _ANSWER_REPLY])
    run = WindowedRunner(
        policy, _gate(), WindowedConfig(frame_size=CANONICAL_SIZE)
    ).run(_clip())
    assert seen["frame_size"] == CANONICAL_SIZE
    assert seen["seek_by_frame"] is False
    assert run.steps[0]["policy_frame_size"] == [384, 288]


def test_streaming_runner_passes_frame_size(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(runner_module, "FrameSource", _fake_source_class(seen))
    policy = _RecordingChat([_OBSERVE_REPLY, _ANSWER_REPLY] * 2)
    run = StreamingRunner(
        policy, _gate(), StreamingConfig(frame_size=CANONICAL_SIZE)
    ).run(_clip())
    assert seen["frame_size"] == CANONICAL_SIZE
    assert seen["seek_by_frame"] is True
    assert run.steps[0]["policy_frame_size"] == [384, 288]


def test_default_runner_records_no_frame_size(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(runner_module, "FrameSource", _fake_source_class(seen))
    policy = _RecordingChat([_OBSERVE_REPLY, _ANSWER_REPLY])
    run = WindowedRunner(policy, _gate(), WindowedConfig()).run(_clip())
    assert seen["frame_size"] is None
    assert "policy_frame_size" not in run.steps[0]


@pytest.fixture
def clip_path(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    for index in range(60):
        writer.write(np.full((48, 64, 3), index % 256, dtype=np.uint8))
    writer.release()
    return path


def _decode(raw: bytes):
    array = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert array is not None
    return array


def test_frame_source_resizes_to_canonical_jpeg(clip_path):
    with FrameSource(clip_path, frame_size=CANONICAL_SIZE) as source:
        raw = source.frame_at(0.0)
    assert raw[:2] == b"\xff\xd8"
    assert _decode(raw).shape == (288, 384, 3)


def test_frame_source_default_keeps_original_resolution(clip_path):
    with FrameSource(clip_path) as source:
        raw = source.frame_at(0.0)
    assert _decode(raw).shape == (48, 64, 3)


def test_frame_source_accepts_yaml_list(clip_path):
    with FrameSource(clip_path, frame_size=[384, 288]) as source:
        assert source.frame_size == CANONICAL_SIZE


@pytest.mark.parametrize(
    "bad",
    [(0, 288), (-384, 288), (384.5, 288), (True, 288), ("384", 288),
     (None, 288), (384,), (384, 288, 1), "384x288"],
)
def test_frame_source_rejects_bad_sizes(clip_path, bad):
    with pytest.raises(ValueError):
        FrameSource(clip_path, frame_size=bad)


def test_judge_frames_stay_original_png(clip_path):
    images, _ = extract_judge_frames(clip_path, 3.0)
    assert images, "judge must decode frames"
    for raw in images:
        assert raw[:8] == PNG_MAGIC
        assert _decode(raw).shape == (48, 64, 3)
