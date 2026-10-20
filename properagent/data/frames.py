"""Frame extraction for the two paradigms.

Windowed: a decision every ``stride`` seconds (2 s in the paper), each seeing the
current frame plus the three preceding ones sampled at 1 s intervals. Negative
frame times are clamped to 0, so early decisions repeat the first frame rather
than seeing fewer images -- the policy's visual input stays a fixed shape.

Streaming: one new frame every ``stride`` seconds (1 s in the paper), appended to
a context that the runner carries across decisions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from ..backends.base import encode_jpeg


@dataclass(frozen=True)
class DecisionFrames:
    """The visual input for one decision point."""

    t: float
    images: tuple[bytes, ...]
    times: tuple[float, ...]


def _validate_frame_size(value: Sequence[int] | None) -> tuple[int, int] | None:
    """Normalize to ``(width, height)``. Rejects bools, non-integers,
    non-positive values, and anything that is not exactly two ints."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ValueError(f"frame_size must be (width, height), got {value!r}")
    try:
        width, height = tuple(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"frame_size must be (width, height), got {value!r}") from error
    for dim in (width, height):
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError(f"frame_size dims must be ints, got {value!r}")
        if not math.isfinite(dim) or dim <= 0:
            raise ValueError(f"frame_size dims must be positive, got {value!r}")
    return (width, height)


class FrameSource:
    """Random-access JPEG frames from a clip, cached by requested timestamp.

    Decoding dominates runtime for long clips, and consecutive windowed decisions
    overlap by three of their four frames, so the cache is doing real work here.

    `frame_size` is an optional ``(width, height)`` pair: each BGR frame is
    resized to exactly that size before RGB conversion and JPEG encoding.
    ``None`` (default) keeps the original resolution.
    """

    def __init__(
        self,
        video_path: str | Path,
        jpeg_quality: int = 85,
        frame_size: Sequence[int] | None = None,
    ) -> None:
        self.path = Path(video_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"video not found: {self.path}")
        self.jpeg_quality = jpeg_quality
        self.frame_size = _validate_frame_size(frame_size)
        self._capture = None
        self._fps: float | None = None
        self._duration: float | None = None
        self._last_good: bytes | None = None

    def _ensure_open(self):
        if self._capture is None:
            import cv2

            capture = cv2.VideoCapture(str(self.path))
            if not capture.isOpened():
                raise RuntimeError(f"could not open video: {self.path}")
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            self._capture = capture
            self._fps = float(fps) if fps and fps > 0 else 30.0
            self._duration = float(count) / self._fps if count and count > 0 else None
        return self._capture

    @property
    def duration(self) -> float | None:
        self._ensure_open()
        return self._duration

    @lru_cache(maxsize=512)
    def frame_at(self, t: float) -> bytes:
        import cv2

        capture = self._ensure_open()
        t = max(0.0, float(t))
        capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = capture.read()

        if not ok or frame is None:
            # Explicit requests at duration lie past the last decodable index,
            # and some containers also refuse seeks near the tail.
            frame = self._decode_near_end(capture)

        if frame is None:
            if self._last_good is not None:
                return self._last_good
            raise RuntimeError(f"could not decode a frame at t={t:.2f}s from {self.path}")

        if self.frame_size is not None:
            frame = cv2.resize(frame, self.frame_size)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encoded = encode_jpeg(np.asarray(rgb), quality=self.jpeg_quality)
        self._last_good = encoded
        return encoded

    def _decode_near_end(self, capture, max_steps: int = 16):
        """Return the newest decodable frame, or None if the clip yields nothing."""
        import cv2

        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for step in range(max_steps):
            index = total - 1 - step
            if index < 0:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok and frame is not None:
                return frame
        # Some containers report a frame count they cannot seek to. Rewind and
        # read forward, keeping the last frame that decodes.
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        last = None
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                return last
            last = frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def windowed_decisions(
    source: FrameSource,
    duration: float,
    stride: float = 2.0,
    n_frames: int = 4,
    frame_gap: float = 1.0,
) -> Iterator[DecisionFrames]:
    """Yield one DecisionFrames per decision point, each with `n_frames` frames."""
    t = 0.0
    while t < duration:
        times = tuple(max(0.0, t - frame_gap * i) for i in range(n_frames - 1, -1, -1))
        yield DecisionFrames(t=t, images=tuple(source.frame_at(x) for x in times), times=times)
        t += stride


def streaming_decisions(
    source: FrameSource, duration: float, stride: float = 1.0
) -> Iterator[DecisionFrames]:
    """Yield one new frame per decision point."""
    t = 0.0
    while t < duration:
        yield DecisionFrames(t=t, images=(source.frame_at(t),), times=(t,))
        t += stride
