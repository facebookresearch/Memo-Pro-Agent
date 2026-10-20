"""Judge inputs, frame extraction, and verdict parsing."""
from __future__ import annotations

import base64
import hashlib
import math
import sys
import types

import pytest

from properagent.backends.base import Message
from properagent.data.schema import Clip, Fire, MemoryItem, Target
from properagent.eval.judge import (
    JUDGE_SYSTEM,
    Category,
    Judge,
    JudgeConfig,
    extract_judge_frames,
    frame_labels,
    gt_block,
    memory_block,
    parse_verdict,
)


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def clip_with(worth: bool = True) -> Clip:
    return Clip(
        uid="u", video_path="", caption_log="", duration=10.0, windows=((1.0, 5.0),),
        target="milk carton", closes_when="",
        targets=(
            Target(
                target="milk carton",
                worth_triggering=worth,
                open_loop="The milk is almost out.",
                memories=(MemoryItem(kind="episodic", text="Nine days ago the milk ran low."),),
            ),
        ),
    )


# ---- canonical labels -----------------------------------------------------

def test_frames_use_canonical_relative_labels():
    # Explicit custom setting (old 3 x 2.5s) stays supported.
    assert frame_labels(12.0, n_frames=3, frame_gap=2.5) == [
        "[frame preceding(-5.0s)]",
        "[frame preceding(-2.5s)]",
        "[frame CURRENT]",
    ]
    # fire_time is kept for API compatibility; labels are relative offsets.
    assert frame_labels(1.0, 3, 2.5) == frame_labels(99.0, 3, 2.5)
    assert frame_labels(10.0, n_frames=1, frame_gap=2.5) == ["[frame CURRENT]"]


def test_frames_default_to_four_one_second_labels():
    # Implicit defaults changed to n=4, gap=1.0 (tested ablation setting).
    assert frame_labels(10.0, n_frames=4, frame_gap=1.0) == [
        "[frame preceding(-3.0s)]",
        "[frame preceding(-2.0s)]",
        "[frame preceding(-1.0s)]",
        "[frame CURRENT]",
    ]


def test_labels_are_interleaved_before_each_image():
    message = Message(
        role="user", text="ANSWER", images=[b"a", b"b"],
        image_labels=["[frame preceding(-2.5s)]", "[frame CURRENT]"],
    )
    parts = message.to_openai()["content"]
    kinds = [p["type"] for p in parts]
    assert kinds == ["text", "image_url", "text", "image_url", "text"]
    assert parts[0]["text"] == "[frame preceding(-2.5s)]"
    assert parts[-1]["text"] == "ANSWER"


def test_label_count_must_match_image_count():
    with pytest.raises(ValueError, match="image_labels"):
        Message(role="user", images=[b"a", b"b"], image_labels=["only one"])


# ---- Message MIME + leading text -------------------------------------------

def test_jpeg_bytes_keep_jpeg_mime_unchanged():
    message = Message(role="user", text="t", images=[b"a"])
    url = message.to_openai()["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")


def test_png_bytes_sniffed_from_header():
    raw = PNG_MAGIC + b"\x00\x01\x02"
    message = Message(role="user", text="t", images=[raw])
    url = message.to_openai()["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == raw


def test_leading_text_comes_before_images_and_trailing_text_after():
    message = Message(
        role="user",
        leading_text="ANSWER (judge this): hi",
        text="MEMORY\nREF_ANSWER (ONE good example for this moment, NOT a target to match): ",
        images=[b"a", b"b"],
        image_labels=["[frame preceding(-2.5s)]", "[frame CURRENT]"],
    )
    parts = message.to_openai()["content"]
    kinds = [p["type"] for p in parts]
    assert kinds == ["text", "text", "image_url", "text", "image_url", "text"]
    assert parts[0]["text"].startswith("ANSWER (judge this):")
    assert parts[1]["text"] == "[frame preceding(-2.5s)]"
    assert parts[3]["text"] == "[frame CURRENT]"
    assert "REF_ANSWER" in parts[-1]["text"]


def test_default_message_order_unchanged_without_leading_text():
    message = Message(
        role="user", text="ANSWER", images=[b"a"],
        image_labels=["[frame CURRENT]"],
    )
    parts = message.to_openai()["content"]
    assert [p["type"] for p in parts] == ["text", "image_url", "text"]
    assert parts[0]["text"] == "[frame CURRENT]"
    assert parts[-1]["text"] == "ANSWER"


# ---- memory / gt block ------------------------------------------------------

def test_memory_block_carries_gold_facts_only_without_reason():
    text = memory_block(clip_with())
    assert "Nine days ago the milk ran low." in text
    assert "MEMORY" in text
    # Removed fields: no reason / reference answer in the judge input.
    assert "REMINDING_REASON" not in text
    assert "REF_ANSWER" not in text
    assert "The milk is almost out." not in text


def test_memory_block_handles_a_clip_with_nothing_worth_triggering():
    assert "(none)" in memory_block(clip_with(worth=False))


def test_gt_block_returns_same_memory_block_without_ref_constant():
    text = gt_block(clip_with())
    assert text == memory_block(clip_with())
    assert "MEMORY" in text and "Nine days ago the milk ran low." in text
    assert "REMINDING_REASON" not in text
    assert "REF_ANSWER" not in text


# ---- JudgeConfig ------------------------------------------------------------

def test_judge_config_canonical_defaults():
    config = JudgeConfig()
    assert config.max_tokens == 400
    assert config.n_frames == 4
    assert config.frame_gap == 1.0
    assert config.max_in_window == 8
    assert config.max_out_of_window == 5


def test_judge_config_accepts_explicit_old_three_frame_setting():
    # Explicit old custom (3 x 2.5s) stays supported.
    config = JudgeConfig(n_frames=3, frame_gap=2.5)
    assert (config.n_frames, config.frame_gap) == (3, 2.5)


@pytest.mark.parametrize("kwargs", [
    {"max_tokens": 0},
    {"max_tokens": -1},
    {"n_frames": 0},
    {"n_frames": -2},
    {"frame_gap": 0.0},
    {"frame_gap": -1.0},
    {"frame_gap": float("nan")},
    {"max_in_window": -1},
    {"max_out_of_window": -1},
])
def test_judge_config_rejects_nonpositive_or_negative(kwargs):
    with pytest.raises(ValueError):
        JudgeConfig(**kwargs)


def test_judge_config_allows_zero_caps():
    config = JudgeConfig(max_in_window=0, max_out_of_window=0)
    assert config.max_in_window == 0 and config.max_out_of_window == 0


# ---- Judge user order -------------------------------------------------------

class _FakeBackend:
    name = "fake"
    max_images_per_request = 32

    def __init__(self) -> None:
        self.messages = None

    def chat(self, messages, config):
        self.messages = list(messages)
        assert config.max_tokens == 400
        return (
            '{"memory_grounded": true, "correct": true, "useful": true, '
            '"category": "GOOD", "reason": "surfaces the stored fact"}'
        )


def test_judge_user_order_answer_then_frames_then_memory_only():
    backend = _FakeBackend()
    judge = Judge(backend)
    fire = Fire(t=10.0, answer="Drink some milk.")
    verdict = judge(clip_with(), fire, [b"a", b"b"])
    assert verdict.category is Category.GOOD

    user = backend.messages[1]
    assert user.leading_text == "ANSWER (judge this): Drink some milk."
    # Gold memory retained; removed fields absent.
    assert user.text == memory_block(clip_with())
    assert "Nine days ago the milk ran low." in user.text
    assert "REMINDING_REASON" not in user.text
    assert "The milk is almost out." not in user.text
    assert "REF_ANSWER" not in user.text
    # Default labels are canonical relative labels for the frame count
    # with the default gap (2 frames x 1.0s).
    assert user.image_labels == ["[frame preceding(-1.0s)]", "[frame CURRENT]"]

    parts = user.to_openai()["content"]
    texts = [p["text"] for p in parts if p["type"] == "text"]
    assert texts[0].startswith("ANSWER (judge this):")
    assert texts[1] == "[frame preceding(-1.0s)]"
    assert texts[2] == "[frame CURRENT]"
    assert texts[-1] is user.text


def test_judge_system_prompt_matches_tested_ablation_prompt():
    # Hardcoded SHA of the confirmed tested prompt (no private fixture dep).
    # Recorded in pilot200_gpt4o_frame_ablation settings as system_prompt_sha256.
    assert hashlib.sha256(JUDGE_SYSTEM.encode()).hexdigest() == (
        "0e4884f100863324c3d386fc1c221ed1565615c14dfa030d2aa1ab2954dc9c3a"
    )
    assert (
        "3) useful: offers real value/actionable point (not just restating), "
        "specificity fits what's knowable now, worth interrupting for." in JUDGE_SYSTEM
    )


def test_judge_accepts_caller_supplied_labels_for_partial_frames():
    backend = _FakeBackend()
    judge = Judge(backend)
    judge(
        clip_with(), Fire(t=10.0, answer="hi"), [b"a"],
        image_labels=["[frame CURRENT]"],
    )
    assert backend.messages[1].image_labels == ["[frame CURRENT]"]


# ---- extract_judge_frames with a faked cv2 ----------------------------------

class _FakeArray:
    def tobytes(self) -> bytes:
        return PNG_MAGIC + b"frame"


class _FakeCapture:
    """Scripted VideoCapture: one outcome per requested frame, in order.

    Outcomes: ``"ok"`` (POS_MSEC decodes), ``"frames-ok"`` (POS_MSEC fails,
    POS_FRAMES decodes), ``"fail"`` (neither decodes), ``"encode-fail"``
    (decodes, but ``imencode`` fails for that frame).
    """

    def __init__(self, script, record):
        self._script = list(script)
        self._record = record
        self._pending = None
        self._frame_index = -1
        self._current_outcome = "fail"
        self.released = False
        self.position_requests: list[tuple[str, float]] = []

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == "fps":
            return 10.0
        return 0

    def set(self, prop, value):
        self.position_requests.append((prop, value))
        if prop == "msec":
            self._frame_index += 1
            if self._frame_index < len(self._script):
                self._current_outcome = self._script[self._frame_index]
            else:
                self._current_outcome = "fail"
        self._pending = (prop, value)
        return True

    def read(self):
        if self._pending is None:
            return False, None
        kind = self._pending[0]
        if kind == "msec":
            if self._current_outcome in ("ok", "encode-fail"):
                return True, object()
            return False, None
        if self._current_outcome == "frames-ok":
            return True, object()
        return False, None

    def release(self):
        self.released = True


def _install_fake_cv2(monkeypatch, script, imencode_ok=True):
    holders = {}
    fake = types.SimpleNamespace()
    fake.CAP_PROP_FPS = "fps"
    fake.CAP_PROP_POS_MSEC = "msec"
    fake.CAP_PROP_POS_FRAMES = "frames"
    fake.COLOR_BGR2RGB = "cvt"

    def _video_capture(path):
        capture = _FakeCapture(script, holders)
        holders["capture"] = capture
        return capture

    def _imencode(ext, frame):
        assert ext == ".png"
        capture = holders.get("capture")
        if capture is not None and capture._current_outcome == "encode-fail":
            return False, None
        if not imencode_ok:
            return False, None
        return True, _FakeArray()

    fake.VideoCapture = _video_capture
    fake.imencode = _imencode
    fake.cvtColor = lambda frame, code: frame
    monkeypatch.setitem(sys.modules, "cv2", fake)
    return holders


def test_extract_judge_frames_happy_path_labels_and_png(monkeypatch):
    holders = _install_fake_cv2(monkeypatch, ["ok"] * 4)
    images, labels = extract_judge_frames("clip.mp4", 10.0)
    assert labels == [
        "[frame preceding(-3.0s)]",
        "[frame preceding(-2.0s)]",
        "[frame preceding(-1.0s)]",
        "[frame CURRENT]",
    ]
    assert len(images) == 4
    assert all(frame.startswith(PNG_MAGIC) for frame in images)
    assert holders["capture"].released is True
    # POS_MSEC seeks for 7.0, 8.0, 9.0, 10.0 in order.
    msec_targets = [v for k, v in holders["capture"].position_requests if k == "msec"]
    assert msec_targets == [7000.0, 8000.0, 9000.0, 10000.0]


def test_extract_judge_frames_explicit_old_three_frame_setting(monkeypatch):
    # Explicit old custom (3 x 2.5s) stays supported.
    holders = _install_fake_cv2(monkeypatch, ["ok"] * 3)
    images, labels = extract_judge_frames("clip.mp4", 10.0, n_frames=3, frame_gap=2.5)
    assert labels == [
        "[frame preceding(-5.0s)]",
        "[frame preceding(-2.5s)]",
        "[frame CURRENT]",
    ]
    assert len(images) == 3
    msec_targets = [v for k, v in holders["capture"].position_requests if k == "msec"]
    assert msec_targets == [5000.0, 7500.0, 10000.0]


def test_extract_judge_frames_falls_back_to_pos_frames_floor(monkeypatch):
    holders = _install_fake_cv2(monkeypatch, ["frames-ok"] * 4)
    images, labels = extract_judge_frames("clip.mp4", 10.0)
    assert len(images) == 4 and len(labels) == 4
    frame_targets = [v for k, v in holders["capture"].position_requests if k == "frames"]
    assert frame_targets == [
        math.floor(7.0 * 10.0),
        math.floor(8.0 * 10.0),
        math.floor(9.0 * 10.0),
        math.floor(10.0 * 10.0),
    ]
    assert holders["capture"].released is True


def test_extract_judge_frames_skips_failed_frames_preserving_labels(monkeypatch):
    holders = _install_fake_cv2(
        monkeypatch, ["ok", "fail", "ok", "ok"]
    )
    # Second frame fails both POS_MSEC and POS_FRAMES; its label is dropped
    # while the surviving frames keep their own canonical labels.
    images, labels = extract_judge_frames("clip.mp4", 10.0)
    assert labels == [
        "[frame preceding(-3.0s)]",
        "[frame preceding(-1.0s)]",
        "[frame CURRENT]",
    ]
    assert len(images) == 3
    assert holders["capture"].released is True


def test_extract_judge_frames_skips_encoding_failures(monkeypatch):
    _install_fake_cv2(monkeypatch, ["ok", "encode-fail", "ok", "ok"])
    images, labels = extract_judge_frames("clip.mp4", 10.0)
    assert labels == [
        "[frame preceding(-3.0s)]",
        "[frame preceding(-1.0s)]",
        "[frame CURRENT]",
    ]
    assert len(images) == 3


def test_extract_judge_frames_current_only_single_frame(monkeypatch):
    # Existing config override: --set judge.n_frames=1 (current-only).
    holders = _install_fake_cv2(monkeypatch, ["ok"])
    images, labels = extract_judge_frames("clip.mp4", 10.0, n_frames=1)
    assert labels == ["[frame CURRENT]"]
    assert len(images) == 1
    msec_targets = [v for k, v in holders["capture"].position_requests if k == "msec"]
    assert msec_targets == [10000.0]


def test_extract_judge_frames_raises_when_nothing_decodes(monkeypatch):
    holders = _install_fake_cv2(monkeypatch, ["fail"] * 4)
    with pytest.raises(RuntimeError, match="could not decode"):
        extract_judge_frames("clip.mp4", 10.0)
    assert holders["capture"].released is True


def test_extract_judge_frames_validates_frame_config():
    with pytest.raises(ValueError):
        extract_judge_frames("clip.mp4", 1.0, n_frames=0)
    for gap in (-1.0, 0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            extract_judge_frames("clip.mp4", 1.0, frame_gap=gap)


def test_real_judge_frames_are_original_resolution_pngs(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    video = tmp_path / "judge.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    for index in range(80):
        writer.write(np.full((48, 64, 3), index, dtype=np.uint8))
    writer.release()
    images, labels = extract_judge_frames(video, 6)
    assert labels == frame_labels(6, 4, 1.0)
    assert len(images) == 4
    for image in images:
        assert image.startswith(PNG_MAGIC)
        decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == (48, 64, 3)


# ---- verdict parsing (unchanged behavior) ------------------------------------

@pytest.mark.parametrize("fence", ["{}", "```json\n{}\n```", "Here you go:\n{}\n"])
def test_parse_verdict_tolerates_wrapping(fence):
    payload = (
        '{"memory_grounded": true, "correct": true, "useful": true, '
        '"category": "GOOD", "reason": "surfaces the stored fact"}'
    )
    verdict = parse_verdict(fence.replace("{}", payload))
    assert verdict.category is Category.GOOD and verdict.is_good


@pytest.mark.parametrize("category", ["BASIC", "BAD", "NON_MEMORY"])
def test_only_good_counts_as_content_correct(category):
    raw = (
        '{"memory_grounded": true, "correct": true, "useful": false, '
        f'"category": "{category}", "reason": "x"}}'
    )
    assert not parse_verdict(raw).is_good


def test_an_unparseable_verdict_raises_rather_than_scoring_zero():
    """A broken judge must not quietly deflate every score."""
    with pytest.raises(ValueError, match="no JSON"):
        parse_verdict("I cannot evaluate this.")
    with pytest.raises(ValueError, match="malformed"):
        parse_verdict("{not json}")
    with pytest.raises(ValueError, match="unknown category"):
        parse_verdict('{"category": "EXCELLENT"}')
