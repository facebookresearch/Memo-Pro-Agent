"""Content judging for emitted reminders.

Timing is scored from the trigger window alone; content needs a judge. The
all-fire evaluator selects exact-text representatives from both in-window and
out-of-window fires. The judge sees each selected reminder, frames at its time,
and gold memory, and returns one of four categories. Only in-window ``GOOD``
verdicts contribute to Hit and Perfect; out-of-window verdicts are diagnostic.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from ..backends.base import GenerationConfig, Message, VisionChatBackend
from ..data.schema import Clip, Fire


class Category(str, Enum):
    GOOD = "GOOD"
    BASIC = "BASIC"
    BAD = "BAD"
    NON_MEMORY = "NON_MEMORY"

    @property
    def is_good(self) -> bool:
        return self is Category.GOOD


_HOW_MEMORY = (
    "HOW MEMORY WORKS: the assistant should speak because it RECALLED a stored long-term "
    "memory relevant now. These memory facts are NOT visible in the current scene (they come "
    "from the user's past); if the answer correctly reflects such a fact, that proves it used "
    "memory. Anything already visible on screen, or already stated in recent context, does NOT "
    "count as memory use - that is merely reacting to what's in front of it.\n"
    "CRITICAL: merely NAMING or commenting on the visible product (its name, price, that it's "
    "on sale, that it's popular) is NOT memory use, EVEN IF the product name overlaps the memory "
    "topic. memory_grounded is TRUE only if the answer actually surfaces the STORED FACT itself "
    "(the allergy / past purchase / preference / plan), which is not visible. "
    "Example - memory='allergic to peanuts', scene shows peanut candy: "
    "'these peanut treats are on sale!' -> memory NOT used (pure reaction to the product); "
    "'careful, you're allergic to peanuts' -> memory used."
)

JUDGE_SYSTEM = (
    "You are an expert evaluator of a PROACTIVE assistant that watches the user's screen / "
    "first-person activity and, holding the user's long-term memory, speaks up to remind or help. "
    "Judge the QUALITY of ONE thing it said at a specific moment.\n" + _HOW_MEMORY + "\n\n"
    "Evaluate THREE criteria:\n"
    "1) memory_grounded: does ANSWER correctly draw on the RIGHT memory (explicit or implicit), "
    "not distorted, and not merely reacting to what's visible/in recent context?\n"
    "2) correct: accurate & safe for THIS moment - no hallucination (do NOT penalize omitting "
    "details not yet observable; DO penalize claiming things not visible), the memory->now link "
    "is valid, memory details not distorted, not harmful/wrong. NOTE: a bland/unhelpful but "
    "factually-accurate and safe reminder is still CORRECT (that is a usefulness problem -> BASIC, "
    "NOT a correctness problem). Set correct=false ONLY for hallucination, invalid link, distorted "
    "memory, or harmful/wrong advice.\n"
    "3) useful: offers real value/actionable point (not just restating), "
    "specificity fits what's knowable now, worth interrupting for.\n\n"
    "DECISION (in order): memory_grounded false -> NON_MEMORY; else correct false -> BAD; "
    "else useful false -> BASIC; else GOOD.\n"
    'OUTPUT ONLY JSON: {"memory_grounded":bool,"correct":bool,"useful":bool,'
    '"category":"GOOD|BASIC|BAD|NON_MEMORY","reason":"<=25 words (say WHY not grounded if '
    'NON_MEMORY: reactive-to-visible / generic / etc.)"}'
)


@dataclass
class Verdict:
    category: Category
    memory_grounded: bool
    correct: bool
    useful: bool
    reason: str = ""
    raw: str = ""

    @property
    def is_good(self) -> bool:
        return self.category.is_good

    def to_json(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "memory_grounded": self.memory_grounded,
            "correct": self.correct,
            "useful": self.useful,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class JudgeConfig:
    max_tokens: int = 400
    temperature: float = 0.0
    n_frames: int = 4
    frame_gap: float = 1.0
    max_in_window: int = 8
    max_out_of_window: int = 5

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens!r}")
        if self.n_frames <= 0:
            raise ValueError(f"n_frames must be positive, got {self.n_frames!r}")
        if not math.isfinite(self.frame_gap) or self.frame_gap <= 0:
            raise ValueError(f"frame_gap must be positive, got {self.frame_gap!r}")
        if self.max_in_window < 0 or self.max_out_of_window < 0:
            raise ValueError(
                "judgment caps must be nonnegative, got "
                f"max_in_window={self.max_in_window!r} "
                f"max_out_of_window={self.max_out_of_window!r}"
            )


def memory_block(clip: Clip) -> str:
    """The gold memory shown to the judge: the facts behind the intervention."""
    for target in clip.targets:
        if not target.worth_triggering:
            continue
        facts = "\n".join(f"  - ({m.kind}) {m.text}" for m in target.memories)
        return f"MEMORY (the correct stored memory the assistant should recall):\n{facts}"
    return "MEMORY (the correct stored memory the assistant should recall):\n  (none)"


def gt_block(clip: Clip) -> str:
    """Show only annotated memory facts, without a reason or reference answer."""
    return memory_block(clip)


def frame_labels(fire_time: float, n_frames: int, frame_gap: float) -> list[str]:
    """Canonical relative labels so the judge knows which frame is current.

    ``preceding(-{gap*k:.1f}s)`` for the context frames, ``CURRENT`` for the
    fire frame -- relative offsets, not absolute timestamps, matching the
    canonical ``frames_at`` labels. ``fire_time`` is kept for signature
    compatibility and is unused.
    """
    _ = fire_time
    labels = [f"preceding(-{frame_gap * (n_frames - 1 - i):.1f}s)" for i in range(n_frames - 1)]
    labels.append("CURRENT")
    return [f"[frame {label}]" for label in labels]


def extract_judge_frames(
    video_path: str | Path,
    fire_time: float,
    n_frames: int = 4,
    frame_gap: float = 1.0,
) -> tuple[list[bytes], list[str]]:
    """Extract the fire frame plus preceding frames as original-resolution PNGs.

    Seeks each target time with ``POS_MSEC``, falling back to ``POS_FRAMES``
    (``floor(t * fps)``) when the msec seek does not decode. Frames that still
    fail are skipped individually with their labels; raises when none decode.
    Lazy ``cv2`` import; the capture is always released. No tail fallback and
    no tempfiles: outputs are in-memory PNG bytes.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames!r}")
    if not math.isfinite(frame_gap) or frame_gap <= 0:
        raise ValueError(f"frame_gap must be finite and positive, got {frame_gap!r}")

    import cv2

    capture = None
    try:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        labels = frame_labels(fire_time, n_frames, frame_gap)
        images: list[bytes] = []
        kept: list[str] = []
        for offset in range(n_frames - 1, -1, -1):
            target = max(0.0, float(fire_time) - frame_gap * offset)
            frame = _seek_frame(capture, target, fps)
            if frame is None:
                continue
            ok, encoded = cv2.imencode(".png", frame)
            if not ok or encoded is None:
                continue
            images.append(bytes(encoded.tobytes()))
            kept.append(labels[n_frames - 1 - offset])
        if not images:
            raise RuntimeError(f"could not decode any judge frame from {video_path}")
        return images, kept
    finally:
        if capture is not None:
            capture.release()


def _seek_frame(capture: Any, target: float, fps: float) -> Any | None:
    """Decode one frame at `target` seconds: POS_MSEC first, POS_FRAMES fallback."""
    import cv2

    capture.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
    ok, frame = capture.read()
    if ok and frame is not None:
        return frame
    index = int(math.floor(max(0.0, target) * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if ok and frame is not None:
        return frame
    return None


class Judge:
    """Content judge for one fire."""

    def __init__(self, backend: VisionChatBackend, config: JudgeConfig | None = None) -> None:
        self.backend = backend
        self.config = config or JudgeConfig()

    def __call__(
        self,
        clip: Clip,
        fire: Fire,
        frames: Sequence[bytes],
        *,
        image_labels: Sequence[str] | None = None,
    ) -> Verdict:
        images = list(frames)
        labels = (
            list(image_labels)
            if image_labels is not None
            else frame_labels(fire.t, len(images), self.config.frame_gap)
        )
        user = Message(
            role="user",
            leading_text=f"ANSWER (judge this): {fire.answer}",
            text=gt_block(clip),
            images=images,
            image_labels=labels,
        )
        raw = self.backend.chat(
            [Message(role="system", text=JUDGE_SYSTEM), user],
            GenerationConfig(
                max_tokens=self.config.max_tokens, temperature=self.config.temperature
            ),
        )
        return parse_verdict(raw)


def parse_verdict(raw: str) -> Verdict:
    """Parse the judge's JSON, tolerating a code fence or surrounding prose.

    An unparseable verdict is NOT silently downgraded to BAD -- that would let a
    broken judge quietly deflate every score. It raises instead.
    """
    text = (raw or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"judge returned no JSON object: {text[:200]!r}")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise ValueError(f"judge returned malformed JSON: {match.group(0)[:200]!r}") from error

    name = str(payload.get("category", "")).strip().upper()
    try:
        category = Category(name)
    except ValueError as error:
        raise ValueError(f"judge returned unknown category {name!r}") from error

    return Verdict(
        category=category,
        memory_grounded=bool(payload.get("memory_grounded", False)),
        correct=bool(payload.get("correct", False)),
        useful=bool(payload.get("useful", False)),
        reason=str(payload.get("reason", "")),
        raw=text,
    )
