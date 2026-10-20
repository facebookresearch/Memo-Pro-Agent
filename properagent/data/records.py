"""Loader for Contract A (see ``properagent.data.schema``).

Deliberately strict: every field the harness depends on must be present in the
data, and a missing or malformed file raises. Silent fallbacks are how an
evaluation ends up scoring a different set than it reports.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from .schema import Caption, Clip, Observation, Target, normalize_windows

REQUIRED_SUBDIRS = ("clips", "full_history", "trigger", "memory")


class DatasetError(RuntimeError):
    pass


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise DatasetError(f"missing file: {path}")
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise DatasetError(f"malformed JSON in {path}: {error}") from error


class Dataset:
    """A ProPerMem-format directory.

    >>> dataset = Dataset("data/propermem")
    >>> clip = dataset.clip(dataset.uids()[0])
    >>> clip.trigger_window
    (2.0, 45.5)
    """

    def __init__(
        self,
        root: str | Path,
        uid_list: str | Path | None = None,
        windows: str | Path | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise DatasetError(f"data root does not exist: {self.root}")
        missing = [d for d in REQUIRED_SUBDIRS if not (self.root / d).is_dir()]
        if missing:
            raise DatasetError(
                f"{self.root} is missing required subdirectories {missing}; "
                "see properagent.data.schema for the expected layout"
            )
        self._uid_list = Path(uid_list).expanduser() if uid_list else self.root / "uids.txt"
        self._window_file = Path(windows).expanduser() if windows else None

    @lru_cache(maxsize=1)
    def _window_overrides(self) -> dict[str, Any]:
        """Authoritative trigger windows, keyed by UID.

        An evaluation set is defined by *which* clips and *when* each one should
        fire, so both travel together in ``evalsets/``. When this file is given it
        wins over ``trigger/<uid>.json``: the per-clip file is the dataset's own
        record, this is the frozen scoring contract for a particular eval set.
        """
        if self._window_file is None:
            return {}
        if not self._window_file.is_file():
            raise DatasetError(f"window file not found: {self._window_file}")
        overrides = _load_json(self._window_file)
        if not isinstance(overrides, dict):
            raise DatasetError(f"{self._window_file} must be a mapping of uid -> window(s)")
        return overrides

    # ---- the evaluation set ----

    @lru_cache(maxsize=1)
    def uids(self) -> tuple[str, ...]:
        """UIDs to evaluate, in file order.

        Reads ``uids.txt`` when present. Without it, falls back to every UID that has
        a caption log -- convenient for the bundled examples, but an evaluation run
        should always pin an explicit list so the reported `n` is reproducible.
        """
        if self._uid_list.is_file():
            lines = self._uid_list.read_text(encoding="utf-8").splitlines()
            uids = tuple(line.strip() for line in lines if line.strip() and not line.startswith("#"))
            if not uids:
                raise DatasetError(f"{self._uid_list} contains no UIDs")
            unknown = [u for u in uids if not (self.root / "full_history" / f"{u}.jsonl").is_file()]
            if unknown:
                raise DatasetError(
                    f"{len(unknown)} UID(s) in {self._uid_list} have no caption log, "
                    f"first few: {unknown[:5]}"
                )
            return uids
        return tuple(sorted(p.stem for p in (self.root / "full_history").glob("*.jsonl")))

    def __len__(self) -> int:
        return len(self.uids())

    def __iter__(self) -> Iterator[Clip]:
        for uid in self.uids():
            yield self.clip(uid)

    # ---- per-UID paths ----

    def video_path(self, uid: str) -> Path:
        return self.root / "clips" / f"{uid}.mp4"

    def caption_log_path(self, uid: str) -> Path:
        return self.root / "full_history" / f"{uid}.jsonl"

    # ---- loaders ----

    def captions(self, uid: str) -> list[Caption]:
        """The full caption stream, in file order.

        Order is taken as authoritative: it is the chronology the memory pipeline
        segments on. ~26K lines per UID is typical.
        """
        path = self.caption_log_path(uid)
        if not path.is_file():
            raise DatasetError(f"missing caption log: {path}")
        captions: list[Caption] = []
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    captions.append(Caption.from_json(json.loads(line)))
                except (json.JSONDecodeError, ValueError) as error:
                    raise DatasetError(f"{path}:{number}: {error}") from error
        if not captions:
            raise DatasetError(f"empty caption log: {path}")
        return captions

    @lru_cache(maxsize=4096)
    def clip(self, uid: str) -> Clip:
        trigger = _load_json(self.root / "trigger" / f"{uid}.json")
        memory = _load_json(self.root / "memory" / f"{uid}.json")
        if not isinstance(memory, list):
            raise DatasetError(f"memory/{uid}.json must be a list of targets")

        visibility_path = self.root / "visibility" / f"{uid}.json"
        visibility: dict[str, Any] = (
            _load_json(visibility_path) if visibility_path.is_file() else {}
        )

        overrides = self._window_overrides()
        if overrides and uid not in overrides:
            raise DatasetError(
                f"{uid} is in the evaluation set but has no entry in {self._window_file}. "
                "Add it, or drop the window file to score against trigger/."
            )
        raw_window = overrides[uid] if overrides else trigger.get("trigger_window")
        try:
            windows = normalize_windows(raw_window)
        except ValueError as error:
            where = self._window_file if overrides else self.root / "trigger" / f"{uid}.json"
            raise DatasetError(f"{where}: {error}") from error

        duration = visibility.get("clip_duration")
        if duration is None:
            duration = _probe_duration(self.video_path(uid))
        if duration is None:
            raise DatasetError(
                f"{uid}: no clip_duration in visibility/ and the video could not be probed"
            )

        video = self.video_path(uid)
        if not video.is_file():
            raise DatasetError(f"missing video: {video}")

        return Clip(
            uid=uid,
            video_path=str(video),
            caption_log=str(self.caption_log_path(uid)),
            duration=float(duration),
            windows=windows,
            target=str(trigger.get("target", "")),
            closes_when=str(trigger.get("closes_when", "")),
            targets=tuple(Target.from_json(row) for row in memory),
            visibility=tuple(
                Observation(
                    t=float(o["t"]),
                    state=str(o.get("state", "no")).lower(),  # type: ignore[arg-type]
                    confidence=float(o.get("confidence", 1.0)),
                )
                for o in visibility.get("observations", [])
                if "t" in o
            ),
        )


def _probe_duration(video: Path) -> float | None:
    if not video.is_file():
        return None
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        capture.release()
    if fps and frames and fps > 0:
        return float(frames) / float(fps)
    return None
