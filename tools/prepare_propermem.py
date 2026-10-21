#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Prepare a local ProPerMem dataset from histories and benchmark metadata.

Reads the shipped source-video/timestamp mapping or a supplied cut-spec file,
copies local histories and clips into Contract A, generates the three
annotation files from the metadata, and applies the selected scoring windows
to prepared triggers. Optional frozen Contract B memory imports preserve
payload bytes. No downloads, rebuilds or model calls.

Interface::

    python tools/prepare_propermem.py \\
      --history-root <flat history package> --clips-root <flat clips> \\
      [--output-root data/prepared] \\
      [--clip-specs evalsets/clip_specs.jsonl] \\
      [--uid UID ... | --uid-list evalsets/eval800.txt] \\
      [--windows evalsets/eval800_windows.json | --windows-none] \\
      [--memory-root <frozen indexes> [--memory-tag TAG]
       [--embedding-signature SIG]]

Only these inputs are read: a clip spec JSONL mapping IDs to source times
with ``annotations`` (trigger/memory/visibility references),
``<history-root>/<uid>.jsonl``, and clips at ``<uid>.mp4``
under ``--clips-root``. Clip file names are derived from the
validated UID instead of any metadata path string.

Output (under ``--output-root``)::

    propermem/clips/<uid>.mp4  propermem/full_history/<uid>.jsonl
    propermem/trigger/<uid>.json  propermem/memory/<uid>.json
    propermem/visibility/<uid>.json  propermem/uids.txt
    propermem/windows.json
    memory/<uid>.{episodic,semantic,passages,entity_list}.json
    memory/<uid>.{entity_emb,semantic_emb,passages_emb}.npy
    memory/<uid>.manifest.json

Provenance: ``propermem/windows.json`` holds the selected windows subset; each
``memory/<uid>.manifest.json`` is schema ``properagent_memory_v1`` with
``imported_not_rebuilt: true``, canonical ``sha256`` entries, row ``counts``,
and a ``source`` fingerprint (plus the source manifest hash when applicable).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from memoproagent.data import Dataset, normalize_windows
from memoproagent.data.schema import Caption, Observation, Target
from memoproagent.memory.store import MANIFEST_SCHEMA, MemoryPaths, MemoryStore, sha256_file

ANNOTATION_FILES = (
    ("trigger", ".json"),
    ("memory", ".json"),
    ("visibility", ".json"),
    ("full_history", ".jsonl"),
)
CORE_PAYLOAD = (
    "episodic.json",
    "passages.json",
    "entity_list.json",
    "entity_emb.npy",
    "passages_emb.npy",
)
SEMANTIC_PAYLOAD = ("semantic.json", "semantic_emb.npy")
CAPTION_SHA_KEYS = (
    "caption_sha256",
    "captions_sha256",
    "full_history_sha256",
    "caption_sha",
    "history_sha256",
)
DEFAULT_UID_LIST = "evalsets/eval800.txt"
DEFAULT_WINDOWS = "evalsets/eval800_windows.json"
DEFAULT_CLIP_SPECS = "evalsets/clip_specs.jsonl"
DEFAULT_OUTPUT_ROOT = "data/prepared"
SAFE_UID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*")
VALID_STATES = ("yes", "no", "uncertain")


class PrepError(RuntimeError):
    """A clean, actionable preparation failure (exit 1, no rollback)."""


def _fail(message: str) -> Any:
    raise PrepError(message)


def _check_safe_id(value: str, what: str) -> str:
    if not isinstance(value, str) or not SAFE_UID.fullmatch(value) or ".." in value:
        _fail(f"{what} is not a safe id (got {value!r}); refusing path traversal")
    return value


def _resolve_inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        _fail(f"{label} {path} resolves outside its root {root}; refusing")
    return resolved


def _resolve_default(given: str | None, default_rel: str, what: str) -> Path:
    if given:
        candidate = Path(given).expanduser()
        if not candidate.is_file() and not candidate.is_absolute():
            alt = REPO_ROOT / given
            candidate = alt if alt.is_file() else candidate
    else:
        candidate = Path(default_rel).expanduser()
        if not candidate.is_file():
            candidate = REPO_ROOT / default_rel
    if not candidate.is_file():
        _fail(f"{what} not found: {candidate}")
    return candidate.resolve()


# ---- selection ----


def _read_uids(args: argparse.Namespace) -> list[str]:
    if args.uid:
        uids = list(args.uid)
    else:
        path = _resolve_default(args.uid_list, DEFAULT_UID_LIST, "--uid-list")
        uids = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if not uids:
            _fail(f"{path} contains no UIDs")
    for uid in uids:
        _check_safe_id(uid, "selected UID")
    dupes = sorted({u for u in uids if uids.count(u) > 1})
    if dupes:
        _fail(f"duplicate UID(s) in selection: {dupes}")
    return uids


def _read_windows(args: argparse.Namespace, uids: list[str]) -> dict[str, Any]:
    if args.windows_none:
        return {}
    path = _resolve_default(args.windows, DEFAULT_WINDOWS, "--windows")
    try:
        overrides = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        _fail(f"malformed JSON in {path}: {error}")
    if not isinstance(overrides, dict):
        _fail(f"{path} must be a mapping of uid -> window(s)")
    missing = [u for u in uids if u not in overrides]
    if missing:
        _fail(
            f"{len(missing)} selected UID(s) have no entry in {path}, "
            f"first few: {missing[:5]}. Add them, or pass --windows-none to "
            "score against trigger/ instead."
        )
    selected: dict[str, Any] = {}
    for uid in uids:
        try:
            normalize_windows(overrides[uid])
        except ValueError as error:
            _fail(f"{path}: windows for {uid}: {error}")
        selected[uid] = overrides[uid]
    return selected


# ---- source metadata ----


def _check_number(row: dict[str, Any], key: str, uid: str) -> None:
    value = row.get(key)
    if value is None:
        return
    if isinstance(value, bool):
        _fail(f"spec {uid}: {key} must be a number, got {value!r}")
    if key == "expected_frames":
        if not isinstance(value, int) or value < 0:
            _fail(f"spec {uid}: {key} must be a non-negative int, got {value!r}")
        return
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"spec {uid}: {key} must be finite, got {value!r}")
    if key == "duration_s" and value <= 0:
        _fail(f"spec {uid}: duration_s must be positive, got {value!r}")
    if key == "source_start_s" and value < 0:
        _fail(f"spec {uid}: source_start_s must be non-negative, got {value!r}")
    if key == "source_end_s" and value <= 0:
        _fail(f"spec {uid}: source_end_s must be positive, got {value!r}")


def _read_clip_specs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        _fail(f"clip specifications not found: {path}")
    index: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                _fail(f"{path}:{number}: malformed JSON: {error}")
            if not isinstance(row, dict):
                _fail(f"{path}:{number}: spec row must be an object")
            uid = row.get("sample_id")
            _check_safe_id(uid, f"{path}:{number} sample_id")
            if uid in index:
                _fail(f"duplicate spec for UID {uid!r} in {path}")
            _check_number(row, "duration_s", uid)
            _check_number(row, "source_start_s", uid)
            _check_number(row, "source_end_s", uid)
            _check_number(row, "expected_frames", uid)
            index[uid] = row
    return index


def _find_clip(clips_root: Path, uid: str) -> Path:
    candidate = clips_root / f"{uid}.mp4"
    resolved = candidate.resolve()
    if root_contains(clips_root, resolved) and resolved.is_file():
        return resolved
    _fail(f"{uid}: clip not found as {uid}.mp4 under {clips_root}")


def root_contains(root: Path, path: Path) -> bool:
    return path == root or root in path.parents


def _generated_from_metadata(
    row: dict[str, Any], uid: str, specs_label: Path | str, selected_window: Any
) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
    """Validate metadata annotations and build generated annotation payloads.

    The top-level ``trigger_window`` is the sole window source; the nested
    ``annotations.trigger`` carries no window. Never fabricates references:
    every required field must be present. The source row is never mutated.
    """
    try:
        normalize_windows(selected_window)
    except ValueError as error:
        _fail(f"{uid}: invalid trigger_window in {specs_label}: {error}")
    ann = row.get("annotations")
    if not isinstance(ann, dict):
        _fail(
            f"{uid}: missing annotations in {specs_label}; each selected sample "
            "needs annotations.trigger/memory/visibility"
        )
    trig = ann.get("trigger")
    if not isinstance(trig, dict):
        _fail(f"{uid}: annotations.trigger must be an object with target/closes_when")
    if "trigger_window" in trig:
        _fail(
            f"{uid}: annotations.trigger must not contain trigger_window; "
            "the top-level trigger_window is the sole window"
        )
    if "target" not in trig or "closes_when" not in trig:
        _fail(
            f"{uid}: annotations.trigger is missing required fields "
            "(need target, closes_when)"
        )
    trig_target = trig.get("target")
    trig_close = trig.get("closes_when")
    if not isinstance(trig_target, str) or not trig_target:
        _fail(f"{uid}: annotations.trigger.target must be a nonempty string")
    if not isinstance(trig_close, str):
        _fail(f"{uid}: annotations.trigger.closes_when must be a string")
    mem = ann.get("memory")
    if not isinstance(mem, list):
        _fail(f"{uid}: annotations.memory must be a list of targets")
    for pos, entry in enumerate(mem):
        if not isinstance(entry, dict):
            _fail(f"{uid}: annotations.memory[{pos}] must be an object")
        for key in ("target", "worth_triggering", "open_loop", "closes_when", "memories"):
            if key not in entry:
                _fail(
                    f"{uid}: annotations.memory[{pos}] is missing required "
                    f"field {key!r}"
                )
        if not isinstance(entry.get("target"), str) or not entry["target"]:
            _fail(f"{uid}: annotations.memory[{pos}].target must be a nonempty string")
        if not isinstance(entry.get("worth_triggering"), bool):
            _fail(f"{uid}: annotations.memory[{pos}].worth_triggering must be a bool")
        if not isinstance(entry.get("open_loop"), str):
            _fail(f"{uid}: annotations.memory[{pos}].open_loop must be a string")
        if not isinstance(entry.get("closes_when"), str):
            _fail(f"{uid}: annotations.memory[{pos}].closes_when must be a string")
        memories = entry.get("memories")
        if not isinstance(memories, list):
            _fail(f"{uid}: annotations.memory[{pos}].memories must be a list")
        for jpos, item in enumerate(memories):
            if not isinstance(item, dict):
                _fail(f"{uid}: annotations.memory[{pos}].memories[{jpos}] must be an object")
            if "kind" not in item or "text" not in item or "day_offset" not in item:
                _fail(
                    f"{uid}: annotations.memory[{pos}].memories[{jpos}] is missing "
                    "required fields (need kind, text, day_offset)"
                )
            if not isinstance(item.get("kind"), str) or not isinstance(item.get("text"), str):
                _fail(
                    f"{uid}: annotations.memory[{pos}].memories[{jpos}] "
                    "kind/text must be strings"
                )
            day_offset = item.get("day_offset")
            if day_offset is not None and (
                isinstance(day_offset, bool) or not isinstance(day_offset, int)
            ):
                _fail(
                    f"{uid}: annotations.memory[{pos}].memories[{jpos}].day_offset "
                    "must be an int or null"
                )
        try:
            Target.from_json(entry)
        except Exception as error:
            _fail(f"{uid}: annotations.memory[{pos}] failed to parse: {error}")
    vis = ann.get("visibility")
    if not isinstance(vis, dict):
        _fail(f"{uid}: annotations.visibility must be an object")
    if "clip_duration" not in vis or "observations" not in vis:
        _fail(
            f"{uid}: annotations.visibility is missing required fields "
            "(need clip_duration, observations)"
        )
    duration = vis.get("clip_duration")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        _fail(
            f"{uid}: annotations.visibility.clip_duration must be a positive "
            f"finite number, got {duration!r}"
        )
    observations = vis.get("observations")
    if not isinstance(observations, list):
        _fail(f"{uid}: annotations.visibility.observations must be a list")
    for kpos, obs in enumerate(observations):
        if not isinstance(obs, dict):
            _fail(
                f"{uid}: annotations.visibility.observations[{kpos}] must be "
                "an object with t/state/confidence"
            )
        if "t" not in obs or "state" not in obs or "confidence" not in obs:
            _fail(
                f"{uid}: annotations.visibility.observations[{kpos}] is missing "
                "required fields (need t, state, confidence)"
            )
        stamp = obs.get("t")
        state = obs.get("state")
        confidence = obs.get("confidence")
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            _fail(
                f"{uid}: annotations.visibility.observations[{kpos}].t "
                "must be a finite number"
            )
        if not isinstance(state, str) or state.lower() not in VALID_STATES:
            _fail(
                f"{uid}: annotations.visibility.observations[{kpos}].state "
                f"must be one of {list(VALID_STATES)}, got {state!r}"
            )
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
        ):
            _fail(
                f"{uid}: annotations.visibility.observations[{kpos}].confidence "
                "must be a finite number"
            )
        try:
            Observation(t=float(stamp), state=state.lower(), confidence=float(confidence))  # type: ignore[arg-type]
        except Exception as error:
            _fail(
                f"{uid}: annotations.visibility.observations[{kpos}] "
                f"failed to parse: {error}"
            )
    trigger_json = {
        "trigger_window": copy.deepcopy(selected_window),
        "target": trig_target,
        "closes_when": trig_close,
    }
    memory_json = copy.deepcopy(mem)
    visibility_json = {
        "clip_duration": duration,
        "observations": copy.deepcopy(observations),
    }
    return trigger_json, memory_json, visibility_json


def _preflight_history(path: Path, uid: str) -> str:
    """Validate the source caption log and return its sha256."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        _fail(f"{uid}: cannot read source full_history/{uid}.jsonl: {error}")
    if not any(line.strip() for line in text.splitlines()):
        _fail(f"{uid}: empty source full_history/{uid}.jsonl")
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            _fail(f"{uid}: malformed source full_history/{uid}.jsonl line {number}: {error}")
        if not isinstance(item, dict):
            _fail(
                f"{uid}: malformed source full_history/{uid}.jsonl line {number}: "
                "must be an object"
            )
        try:
            Caption.from_json(item)
        except ValueError as error:
            _fail(
                f"{uid}: malformed source full_history/{uid}.jsonl line {number}: {error}"
            )
    return sha256_file(path)


# ---- memory import ----


def _tagged(paths: MemoryPaths, uid: str, tag: str) -> dict[str, Path]:
    return {
        name: getattr(paths, field).parent
        / getattr(paths, field).name.replace(f"{uid}.", f"{uid}__{tag}.", 1)
        for field, name in (
            ("episodic", "episodic.json"),
            ("semantic", "semantic.json"),
            ("passages", "passages.json"),
            ("entity_list", "entity_list.json"),
            ("entity_emb", "entity_emb.npy"),
            ("semantic_emb", "semantic_emb.npy"),
            ("passage_emb", "passages_emb.npy"),
            ("manifest", "manifest.json"),
        )
    }


def _preflight_memory(
    mem_root: Path, uid: str, tag: str | None, explicit_sig: str | None,
    full_history_sha: str,
) -> dict[str, Any]:
    canon = MemoryPaths.for_uid(mem_root, uid)
    names = (
        ("episodic.json", canon.episodic),
        ("semantic.json", canon.semantic),
        ("passages.json", canon.passages),
        ("entity_list.json", canon.entity_list),
        ("entity_emb.npy", canon.entity_emb),
        ("semantic_emb.npy", canon.semantic_emb),
        ("passages_emb.npy", canon.passage_emb),
    )
    if tag:
        tagged = _tagged(canon, uid, tag)
        names = tuple((n, tagged[n]) for n, _ in names)
        manifest_src = tagged["manifest.json"]
    else:
        manifest_src = canon.manifest
    kind = f"tagged:{tag}" if tag else "canonical"

    missing = [n for n, p in names if n in CORE_PAYLOAD and not p.is_file()]
    if missing:
        _fail(f"{uid}: incomplete frozen memory ({kind}): missing {missing} under {mem_root}")
    sem_files = {n: _by_name(names, n) for n in SEMANTIC_PAYLOAD}
    sem_present = [n for n, p in sem_files.items() if p.is_file()]
    if len(sem_present) == 1:
        _fail(
            f"{uid}: inconsistent frozen memory ({kind}): only {sem_present} "
            "present, need both or neither"
        )
    import_names = [n for n, p in names if p.is_file()]
    for name in import_names:
        _resolve_inside(_by_name(names, name), mem_root, f"{uid}: frozen memory file")
    dimension = _inspect_payload(uid, names, import_names)

    recorded_sig: str | None = None
    src_manifest_sha: str | None = None
    if manifest_src.is_file():
        _resolve_inside(manifest_src, mem_root, f"{uid}: source manifest")
        try:
            manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            _fail(f"{uid}: malformed source manifest {manifest_src}: {error}")
        if not isinstance(manifest, dict):
            _fail(f"{uid}: source manifest must be an object")
        if manifest.get("schema") != MANIFEST_SCHEMA:
            _fail(
                f"{uid}: source manifest schema {manifest.get('schema')!r}, "
                f"expected {MANIFEST_SCHEMA!r}"
            )
        if "uid" in manifest and manifest["uid"] != uid:
            _fail(f"{uid}: source manifest is for {manifest['uid']!r}")
        by_name = {path.name: path for _, path in names if path.is_file()}
        hashes = manifest.get("sha256", {})
        if not isinstance(hashes, dict):
            _fail(f"{uid}: source manifest sha256 must be a filename-to-hash mapping")
        for name, expected in hashes.items():
            src = by_name.get(name)
            if src is None:
                _fail(f"{uid}: source manifest lists an unknown or missing payload {name!r}")
            if not isinstance(expected, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected):
                _fail(f"{uid}: source manifest has an invalid hash for {name!r}")
            actual = sha256_file(src)
            if actual != expected:
                _fail(
                    f"{uid}: frozen {name} changed since built "
                    f"(sha {actual[:12]} != {expected[:12]})"
                )
        for key in CAPTION_SHA_KEYS:
            if manifest.get(key) not in (None, full_history_sha):
                _fail(f"{uid}: source manifest {key} does not match the source full_history")
        recorded_sig = manifest.get("embedding_signature")
        src_manifest_sha = sha256_file(manifest_src)
    if explicit_sig and recorded_sig and explicit_sig != recorded_sig:
        _fail(
            f"{uid}: --embedding-signature {explicit_sig!r} "
            f"conflicts with frozen {recorded_sig!r}"
        )
    signature = recorded_sig or explicit_sig
    if not signature:
        _fail(
            f"{uid}: no embedding signature: frozen memory has no manifest; "
            "pass --embedding-signature explicitly (e.g. 'BAAI/bge-large-en-v1.5:1024')"
        )
    if not isinstance(signature, str) or signature != signature.strip():
        _fail(f"{uid}: embedding signature must be a nonempty model:dimension string")
    model, separator, suffix = signature.rpartition(":")
    if not model or not separator or suffix != str(dimension):
        _fail(f"{uid}: embedding signature {signature!r} disagrees with matrix dimension {dimension}")
    checksums = {n: sha256_file(_by_name(names, n)) for n in import_names}
    fingerprint = hashlib.sha256(
        "\n".join(f"{n}:{checksums[n]}" for n in sorted(import_names)).encode()
    ).hexdigest()
    return {
        "names": names, "import_names": import_names, "signature": signature,
        "kind": kind, "tag": tag, "fingerprint": fingerprint,
        "src_manifest_sha": src_manifest_sha, "checksums": checksums,
        "history_sha256": full_history_sha,
    }


def _by_name(names: tuple[tuple[str, Path], ...], name: str) -> Path:
    return dict(names)[name]


def _inspect_payload(
    uid: str, names: tuple[tuple[str, Path], ...], import_names: list[str]
) -> int:
    import numpy as np

    by_name = dict(names)
    payloads: dict[str, Any] = {}
    for name in import_names:
        path = by_name[name]
        if name.endswith(".npy"):
            try:
                arr = np.load(path)
            except Exception as error:
                _fail(f"{uid}: malformed frozen {name}: {error}")
            if arr.ndim != 2 or arr.dtype.kind != "f" or not bool(np.isfinite(arr).all()):
                _fail(f"{uid}: frozen {name} must be a finite 2-D float matrix")
            payloads[name] = arr
        else:
            try:
                payloads[name] = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError) as error:
                _fail(f"{uid}: malformed frozen {name}: {error}")
    episodic = payloads["episodic.json"]
    if not isinstance(episodic, dict) or not isinstance(episodic.get("events"), list) or not episodic["events"]:
        _fail(f"{uid}: frozen episodic.json must contain a nonempty events list")
    if any(not isinstance(event, dict) for event in episodic["events"]):
        _fail(f"{uid}: frozen events must be objects")
    for name in ("entity_list.json", "passages.json"):
        if not isinstance(payloads[name], list) or any(not isinstance(item, str) for item in payloads[name]):
            _fail(f"{uid}: frozen {name} must be a string list")
    dims = {a.shape[1] for a in payloads.values() if hasattr(a, "shape")}
    if len(dims) != 1 or next(iter(dims)) <= 0:
        _fail(f"{uid}: frozen embedding dims disagree or are empty: {sorted(dims)}")
    checks = (
        ("entity_emb.npy", "entity_list.json", len(payloads["entity_list.json"])),
        ("passages_emb.npy", "passages.json", len(payloads["passages.json"])),
    )
    for npy_name, _, want in checks:
        if payloads[npy_name].shape[0] != want:
            _fail(
                f"{uid}: frozen {npy_name} has {payloads[npy_name].shape[0]} rows "
                f"but the JSON has {want} entries; row order must align"
            )
    if "semantic.json" in payloads:
        if not isinstance(payloads["semantic.json"], dict):
            _fail(f"{uid}: frozen semantic.json must be an object")
        triples = payloads["semantic.json"].get("triples", [])
        if not isinstance(triples, list) or payloads["semantic_emb.npy"].shape[0] != len(triples):
            _fail(f"{uid}: frozen semantic_emb rows != semantic triples")
    return next(iter(dims))


# ---- run ----


def _copy_bytes(src: Path, dst: Path, uid: str) -> int:
    if dst.exists():
        _fail(f"{uid}: refusing to overwrite existing {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout, length=1 << 20)
    return dst.stat().st_size


def _write_generated(payload: Any, dst: Path, uid: str) -> int:
    if dst.exists():
        _fail(f"{uid}: refusing to overwrite existing {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with dst.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
    except FileExistsError:
        _fail(f"{uid}: refusing to overwrite existing {dst}")
    return dst.stat().st_size


def _write_manifest(uid: str, mem_out: Path, plan: dict[str, Any]) -> Path:
    episodic = json.loads((mem_out / f"{uid}.episodic.json").read_text(encoding="utf-8"))
    events = episodic.get("events", [])
    entity_list = json.loads((mem_out / f"{uid}.entity_list.json").read_text(encoding="utf-8"))
    passages = json.loads((mem_out / f"{uid}.passages.json").read_text(encoding="utf-8"))
    sem_path = mem_out / f"{uid}.semantic.json"
    semantic = (
        json.loads(sem_path.read_text(encoding="utf-8")).get("triples", [])
        if sem_path.is_file() else []
    )
    failed = episodic.get("n_failed_events", 0)
    dest_names = [f"{uid}.{n}" for n in plan["import_names"]]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "uid": uid,
        "embedding_signature": plan["signature"],
        "imported_not_rebuilt": True,
        "counts": {
            "events": len(events),
            "triples": sum(len(e.get("triples", [])) for e in events),
            "failed_events": failed if isinstance(failed, int) else 0,
            "entities": len(entity_list),
            "semantic": len(semantic),
            "passages": len(passages),
        },
        "sha256": {n: sha256_file(mem_out / n) for n in dest_names},
        "source": {
            "kind": plan["kind"],
            "tag": plan["tag"],
            "fingerprint_sha256": plan["fingerprint"],
            "manifest_sha256": plan["src_manifest_sha"],
        },
    }
    out = mem_out / f"{uid}.manifest.json"
    if out.exists():
        _fail(f"{uid}: refusing to overwrite existing {out}")
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare local histories, benchmark annotations and clips for GPM."
    )
    parser.add_argument("--history-root", required=True,
                        help="flat history root holding <uid>.jsonl files")
    parser.add_argument("--clip-specs", default=None,
                        help="source-video/timestamp JSONL (defaults to shipped metadata)")
    parser.add_argument("--clips-root", required=True,
                        help="flat clips root holding <uid>.mp4 files")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help="fresh output directory")
    parser.add_argument("--uid", action="append", default=[],
                        help="repeatable; overrides --uid-list")
    parser.add_argument("--uid-list", default=DEFAULT_UID_LIST,
                        help="UID file (default: %(default)s)")
    parser.add_argument("--windows", default=DEFAULT_WINDOWS,
                        help="windows JSON (default: %(default)s)")
    parser.add_argument(
        "--windows-none", action="store_true",
        help="no window overrides; score against trigger/ instead",
    )
    parser.add_argument("--memory-root", default=None,
                        help="frozen canonical/legacy memory index dir")
    parser.add_argument("--memory-tag", default=None,
                        help="explicit legacy tag, e.g. qwen38fh")
    parser.add_argument("--embedding-signature", default=None,
                        help="required when frozen has no manifest")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.windows_none and args.windows != DEFAULT_WINDOWS:
            parser.error("--windows and --windows-none are mutually exclusive")
        if args.memory_tag and not args.memory_root:
            parser.error("--memory-tag needs --memory-root")
        if args.embedding_signature and not args.memory_root:
            parser.error("--embedding-signature needs --memory-root")

        history_root = Path(args.history_root).expanduser().resolve()
        clips_root = Path(args.clips_root).expanduser().resolve()
        if not history_root.is_dir():
            _fail(f"--history-root is not a directory: {history_root}")
        if not clips_root.is_dir():
            _fail(f"--clips-root is not a directory: {clips_root}")
        mem_root = Path(args.memory_root).expanduser().resolve() if args.memory_root else None
        if mem_root is not None and not mem_root.is_dir():
            _fail(f"--memory-root is not a directory: {mem_root}")

        out_root = Path(args.output_root).expanduser().resolve()
        for root in (history_root, clips_root, *( [mem_root] if mem_root else [])):
            if out_root == root or root_contains(root, out_root) or root_contains(out_root, root):
                _fail(f"--output-root {out_root} must not equal or overlap input root {root}")
        if out_root == REPO_ROOT or root_contains(out_root, REPO_ROOT):
            _fail(f"--output-root {out_root} must not be or contain the repo root {REPO_ROOT}")

        uids = _read_uids(args)
        windows = _read_windows(args, uids)
        if args.clip_specs:
            specs = Path(args.clip_specs).expanduser().resolve()
            if not specs.is_file():
                _fail(f"--clip-specs not found: {specs}")
        else:
            specs = REPO_ROOT / DEFAULT_CLIP_SPECS
            if not specs.is_file():
                _fail(f"--clip-specs not found: {specs}")
        specs = specs.resolve()
        if specs == out_root or root_contains(out_root, specs):
            _fail(
                f"--output-root {out_root} must not contain source metadata {specs}; "
                "refusing (source file never overwritten)"
            )
        index = _read_clip_specs(specs)
        unknown = [u for u in uids if u not in index]
        if unknown:
            _fail(
                f"{len(unknown)} selected UID(s) have no spec in {specs}, "
                f"first few: {unknown[:5]}"
            )
        if out_root.exists():
            if not out_root.is_dir() or any(out_root.iterdir()):
                _fail(f"--output-root {out_root} exists and is non-empty; refusing (no overwrite)")
        else:
            out_root.mkdir(parents=True)

        propermem_out = out_root / "propermem"
        mem_out = out_root / "memory"

        # ---- preflight everything before writing ----
        plans: list[dict[str, Any]] = []
        for uid in uids:
            hist_src = _resolve_inside(
                history_root / f"{uid}.jsonl",
                history_root, f"{uid}: full_history",
            )
            if not hist_src.is_file():
                _fail(
                    f"{uid}: missing source full_history/{uid}.jsonl under "
                    f"{history_root}"
                )
            history_sha = _preflight_history(hist_src, uid)
            clip = _find_clip(clips_root, uid)
            row = index[uid]
            selected_window = windows[uid] if windows else row.get("trigger_window")
            trigger_json, memory_json, visibility_json = _generated_from_metadata(
                row, uid, specs, selected_window
            )
            mem_plan = (
                _preflight_memory(mem_root, uid, args.memory_tag, args.embedding_signature,
                                  history_sha)
                if mem_root is not None else None
            )
            plans.append({"uid": uid, "history": hist_src, "history_sha": history_sha,
                          "clip": clip, "trigger": trigger_json,
                          "memory": memory_json, "visibility": visibility_json,
                          "mem": mem_plan})
        dests = [
            propermem_out / sub / f"{p['uid']}{ext}"
            for p in plans for sub, ext in (*ANNOTATION_FILES, ("clips", ".mp4"))
        ]
        if mem_root is not None:
            for p in plans:
                dests += [mem_out / f"{p['uid']}.{n}" for n in p["mem"]["import_names"]]
                dests.append(mem_out / f"{p['uid']}.manifest.json")
        dests += [propermem_out / "uids.txt", propermem_out / "windows.json"]
        clashes = [str(d) for d in dests if d.exists()]
        if clashes:
            _fail(f"refusing to overwrite {len(clashes)} existing file(s), first: {clashes[0]}")

        # ---- copy histories/clips, generate annotations, then validate per UID ----
        for sub in ("clips", "full_history", "trigger", "memory", "visibility"):
            (propermem_out / sub).mkdir(parents=True, exist_ok=True)
        if args.windows_none:
            validate_set = Dataset(propermem_out)
        else:
            window_file = _resolve_default(args.windows, DEFAULT_WINDOWS, "--windows")
            validate_set = Dataset(propermem_out, windows=window_file)
        total_bytes = 0
        for plan in plans:
            uid = plan["uid"]
            total_bytes += _copy_bytes(
                plan["history"], propermem_out / "full_history" / f"{uid}.jsonl", uid)
            total_bytes += _copy_bytes(plan["clip"], propermem_out / "clips" / f"{uid}.mp4", uid)
            total_bytes += _write_generated(
                plan["trigger"], propermem_out / "trigger" / f"{uid}.json", uid)
            total_bytes += _write_generated(
                plan["memory"], propermem_out / "memory" / f"{uid}.json", uid)
            total_bytes += _write_generated(
                plan["visibility"], propermem_out / "visibility" / f"{uid}.json", uid)
            if plan["mem"] is not None:
                by_name = dict(plan["mem"]["names"])
                for name in plan["mem"]["import_names"]:
                    total_bytes += _copy_bytes(by_name[name], mem_out / f"{uid}.{name}", uid)
                for name, expected in plan["mem"]["checksums"].items():
                    if sha256_file(mem_out / f"{uid}.{name}") != expected:
                        _fail(f"{uid}: frozen payload {name} changed during import")
                if sha256_file(propermem_out / "full_history" / f"{uid}.jsonl") != plan["mem"]["history_sha256"]:
                    _fail(f"{uid}: history changed during import")
                _write_manifest(uid, mem_out, plan["mem"])
                store = MemoryStore(
                    uid, mem_out,
                    caption_log=propermem_out / "full_history" / f"{uid}.jsonl",
                    verify_hashes=True,
                )
                store.check_embedding_signature(plan["mem"]["signature"])
            validate_set.captions(uid)
            validate_set.clip(uid)

        (propermem_out / "uids.txt").write_text("\n".join(uids) + "\n", encoding="utf-8")
        (propermem_out / "windows.json").write_text(
            json.dumps(windows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

        print(f"prepared {len(uids)} UID(s) -> {out_root} ({total_bytes / 1e6:.1f} MB payload)")
        for plan in plans:
            extra = f" + memory ({plan['mem']['kind']})" if plan["mem"] else ""
            print(f"  {plan['uid']} <- clip + history + generated annotations{extra}")
        print("config hint:")
        print(f"  data.root: {propermem_out}")
        print(f"  data.uid_list: {propermem_out / 'uids.txt'}")
        print(f"  data.windows: {propermem_out / 'windows.json'}"
              "  # or the shipped evalsets original")
        if mem_root is not None:
            print(f"  memory.root: {mem_out}")
        return 0
    except (RuntimeError, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        print(f"prepare_propermem: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
