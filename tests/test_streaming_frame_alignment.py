# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Streaming frame alignment: frame-index seek + 384x288 vs the Trunk V2 recipe.

Owns only the image-alignment slice of the release/streaming trunk-V2 plan:
`FrameSource(seek_by_frame=True)` mirrors the legacy
`session_runner.iter_video_turns` decode recipe (fps fallback 30,
``POS_FRAMES=int(round(t*fps))``, resize to 384x288, JPEG quality 85, no
nearby-frame substitution), while the default timestamp seek plus near-end
fallback (Windowed) is unchanged. Sampling stays 1 Hz and
`streaming.frame_size` is exposed in all three configs.

CPU-only synthetic video. Streaming JPEG bytes must exactly match the
legacy OpenCV recipe; Windowed retains its existing PIL encoding.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from memoproagent.agent.runner import PolicyConfig, StreamingConfig
from memoproagent.config import load_config, section
from memoproagent.data.frames import FrameSource, streaming_decisions

cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SIZE = (384, 288)  # (width, height), matching legacy VideoConfig.
FPS = 30.0
N_FRAMES = 90  # 3 s at 30 fps.


@pytest.fixture
def clip_path(tmp_path):
    """Synthetic clip with spatial color patterns that change per frame."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48)
    )
    assert writer.isOpened()
    yy, xx = np.indices((48, 64))
    for index in range(N_FRAMES):
        frame = np.stack(
            ((xx * 4 + index * 7) % 256,
             (yy * 5 + index * 11) % 256,
             ((xx + yy) * 3 + index * 13) % 256),
            axis=-1,
        ).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _legacy_frame(video_path, t: float) -> bytes:
    """Byte-for-byte replica of the legacy Trunk V2 `iter_video_turns` recipe."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = capture.read()
        assert ok and frame is not None, f"legacy recipe failed at t={t}"
        frame = cv2.resize(frame, CANONICAL_SIZE)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        assert ok
        return encoded.tobytes()
    finally:
        capture.release()


def _decode(raw: bytes):
    array = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert array is not None
    return array


def test_seek_by_frame_defaults_to_false():
    param = inspect.signature(FrameSource.__init__).parameters["seek_by_frame"]
    assert param.default is False


def test_seek_by_frame_matches_legacy_recipe(clip_path):
    with FrameSource(
        clip_path, frame_size=CANONICAL_SIZE, seek_by_frame=True
    ) as source:
        assert source.jpeg_quality == 85
        for t in (0.0, 1.0, 2.0):
            raw = source.frame_at(t)
            assert raw[:2] == b"\xff\xd8"
            got = _decode(raw)
            assert got.shape == (288, 384, 3)
            assert raw == _legacy_frame(clip_path, t), f"t={t}: JPEG bytes differ"
        # Frames actually advance: index math selects distinct source frames.
        first = _decode(source.frame_at(0.0)).astype(np.int16)
        second = _decode(source.frame_at(1.0)).astype(np.int16)
        assert np.abs(first - second).mean() > 5.0


def test_seek_by_frame_falls_back_to_30fps(clip_path):
    with FrameSource(
        clip_path, frame_size=CANONICAL_SIZE, seek_by_frame=True
    ) as source:
        source._ensure_open()
        source._fps = None  # container reports no fps; recipe uses 30.
        assert source.frame_at(1.0) == _legacy_frame(clip_path, 1.0)


def test_seek_by_frame_raises_without_substitution(clip_path):
    with FrameSource(
        clip_path, frame_size=CANONICAL_SIZE, seek_by_frame=True
    ) as source:
        with pytest.raises(RuntimeError):
            source.frame_at(1e6)


def test_windowed_timestamp_seek_and_fallback_unchanged(clip_path):
    with FrameSource(clip_path) as source:
        assert source.seek_by_frame is False
        assert source.frame_at(-2.0) == source.frame_at(0.0)  # clamp, not fewer frames
        # Past-the-end timestamp seeks still fall back instead of raising.
        assert source.frame_at(1e6)[:2] == b"\xff\xd8"


def test_streaming_sampling_stays_1hz():
    class StubSource:
        def __init__(self) -> None:
            self.requested: list[float] = []

        def frame_at(self, t: float) -> bytes:
            self.requested.append(t)
            return b"jpeg"

    source = StubSource()
    decisions = list(streaming_decisions(source, 3.5, stride=1.0))
    assert [decision.t for decision in decisions] == [0.0, 1.0, 2.0, 3.0]
    assert source.requested == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.parametrize("name", ["default.yaml", "example.yaml", "qwen_vllm.yaml"])
def test_streaming_frame_size_in_all_configs(name):
    config = load_config(ROOT / "configs" / name)
    assert config["streaming"] == {
        "stride": 1.0,
        "frame_size": [384, 288],
        "chunk_turns": 30,
        "keep_turns": 4,
    }
    assert "frame_size" not in config["windowed"]
    assert config["gate"]["prompt_variant"] == "version_free"
    assert config["policy_sampling"]["observe_max_tokens"] == 48
    assert config["policy_sampling"]["answer_max_tokens"] == 160


def test_streaming_config_constructs_from_yaml_section():
    config = load_config(ROOT / "configs" / "default.yaml")
    parsed = StreamingConfig(
        policy=PolicyConfig(), **section(config, "streaming")
    )
    assert parsed.frame_size == CANONICAL_SIZE
    assert (parsed.stride, parsed.chunk_turns, parsed.keep_turns) == (1.0, 30, 4)


def test_qwen_profile_preserves_prior_settings():
    config = load_config(ROOT / "configs" / "qwen_vllm.yaml")
    assert config["policy_sampling"]["observe_stop"] is True
    assert config["policy"]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert config["gate_backend"]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
