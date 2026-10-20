"""Tests for tools/prepare_propermem.py (synthetic fixtures only, no real input)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import prepare_propermem as prep
from properagent.data import Dataset
from properagent.data.schema import Target, normalize_windows
from properagent.memory.store import MANIFEST_SCHEMA, MemoryPaths

UID = "0dca9b08_169-341"
UID2 = "1db856a0_99-265"
WIN = [[52.5, 75.0]]
CAPTION = {"cid": "D1:0", "day": 1, "t": 0.0, "text": "hi", "t_days_before": 9}
TRIG = {"target": "milk", "closes_when": "done"}
MEM = [{"target": "milk", "worth_triggering": True, "open_loop": "buy milk",
        "closes_when": "done", "memories": []}]
VIS = {"clip_duration": 172.0, "observations": []}


def _row(uid: str = UID, **over) -> dict:
    row = {
        "sample_id": uid,
        "ego4d_video_uid": "v" + uid,
        "source_start_s": 169.0,
        "source_end_s": 341.0,
        "duration_s": 172.0,
        "expected_frames": 100,
        "trigger_window": WIN,
        "annotations": {
            "trigger": dict(TRIG),
            "memory": [dict(m) for m in MEM],
            "visibility": {"clip_duration": 172.0, "observations": []},
        },
    }
    row.update(over)
    return row


def _specs(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as h:
        for row in rows:
            h.write(json.dumps(row) + "\n")
    return path


def _hist(root: Path, uid: str = UID, lines: list[dict] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = lines if lines is not None else [CAPTION]
    with (root / f"{uid}.jsonl").open("w", encoding="utf-8") as h:
        for item in rows:
            h.write(json.dumps(item) + "\n")
    return root


def _clips(root: Path, uid: str = UID, payload: bytes = b"fake-video") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{uid}.mp4").write_bytes(payload)
    return root


def _winfile(path: Path, uid: str = UID, win=WIN) -> Path:
    path.write_text(json.dumps({uid: win}))
    return path


def _frozen_mem(root: Path, uid: str = UID, tag: str | None = None,
                manifest: bool = True, dim: int = 8) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    name = f"{uid}__{tag}" if tag else uid
    rng = np.random.default_rng(7)
    events = [{"idx": 0, "day": 1, "cids": ["D1:0"], "rewrite": "I saw milk",
               "triples": [["I", "saw", "milk"]], "ok": True}]
    (root / f"{name}.episodic.json").write_text(json.dumps({"events": events}))
    (root / f"{name}.entity_list.json").write_text(json.dumps(["I", "milk"]))
    (root / f"{name}.passages.json").write_text(json.dumps(["I saw milk"]))
    (root / f"{name}.semantic.json").write_text(
        json.dumps({"uid": uid, "n_semantic": 1, "triples": [["I", "like", "milk"]]}))
    np.save(root / f"{name}.entity_emb.npy", rng.normal(size=(2, dim)).astype(np.float32))
    np.save(root / f"{name}.passages_emb.npy", rng.normal(size=(1, dim)).astype(np.float32))
    np.save(root / f"{name}.semantic_emb.npy", rng.normal(size=(1, dim)).astype(np.float32))
    if manifest:
        from properagent.memory.store import sha256_file

        names = [f"{name}.{s}" for s in
                 ("episodic.json", "semantic.json", "passages.json", "entity_list.json",
                  "entity_emb.npy", "semantic_emb.npy", "passages_emb.npy")]
        (root / f"{name}.manifest.json").write_text(json.dumps({
            "schema": MANIFEST_SCHEMA, "uid": uid,
            "embedding_signature": f"BAAI/bge-large-en-v1.5:{dim}",
            "counts": {"events": 1, "triples": 1, "entities": 2, "semantic": 1, "passages": 1},
            "sha256": {n: sha256_file(root / n) for n in names},
        }))
    return root


def _run(*argv: str) -> int:
    return prep.main(list(argv))


def _flat(tmp_path: Path, uid: str = UID, trigger_win=WIN,
          extra_row: dict | None = None) -> tuple[Path, Path, Path]:
    hist = _hist(tmp_path / "hist", uid)
    clips = _clips(tmp_path / "clips", uid)
    row = _row(uid, trigger_window=trigger_win)
    if extra_row:
        row.update(extra_row)
    specs = _specs(tmp_path / "specs.jsonl", [row])
    assert json.loads(specs.read_text().splitlines()[0])["trigger_window"] == WIN or uid != UID
    return hist, clips, specs


def _args(hist: Path, clips: Path, specs: Path, out: Path,
          uid: str = UID, wins: Path | None = None) -> list[str]:
    base = ["--history-root", str(hist), "--clips-root", str(clips),
            "--clip-specs", str(specs), "--output-root", str(out),
            "--uid", uid]
    if wins is None:
        base.append("--windows-none")
    else:
        base += ["--windows", str(wins)]
    return base


def _expected_from_row(row: dict, selected) -> tuple[dict, list, dict]:
    return (
        {"trigger_window": selected, "target": row["annotations"]["trigger"]["target"],
         "closes_when": row["annotations"]["trigger"]["closes_when"]},
        row["annotations"]["memory"],
        row["annotations"]["visibility"],
    )


# ---- happy paths ----


def test_flat_layout_end_to_end(tmp_path: Path):
    hist, clips, specs = _flat(tmp_path)
    out = tmp_path / "o"
    wins = _winfile(tmp_path / "w.json")
    assert _run(*_args(hist, clips, specs, out, wins=wins)) == 0
    got = out / "propermem"
    assert (got / "clips" / f"{UID}.mp4").read_bytes() == b"fake-video"
    assert (got / "full_history" / f"{UID}.jsonl").read_bytes() == (
        hist / f"{UID}.jsonl").read_bytes()
    assert (got / "uids.txt").read_text() == UID + "\n"
    assert json.loads((got / "windows.json").read_text()) == {UID: WIN}
    ds = Dataset(got, windows=wins)
    clip = ds.clip(UID)
    assert clip.duration == 172.0 and clip.target == "milk"
    assert ds.captions(UID)[0].cid == "D1:0"


def test_metadata_extra_fields_kept_end_to_end(tmp_path: Path):
    hist = _hist(tmp_path / "hist")
    clips = _clips(tmp_path / "clips")
    row = _row(extra_field="kept", codec={"c:v": "libx264"}, vf="scale=-2:540",
               clip_file=f"{UID}.mp4", expected_fps=30, source_asset="video_540ss")
    specs = _specs(tmp_path / "specs.jsonl", [row])
    out = tmp_path / "o"
    wins = _winfile(tmp_path / "w.json")
    assert _run(*_args(hist, clips, specs, out, wins=wins)) == 0
    got = out / "propermem"
    assert (got / "clips" / f"{UID}.mp4").is_file()
    assert json.loads((got / "windows.json").read_text()) == {UID: WIN}


def test_payload_bytes_preserved_and_memory_seven_plus_manifest(tmp_path: Path):
    hist, clips, specs = _flat(tmp_path)
    mem = _frozen_mem(tmp_path / "m")
    out = tmp_path / "o"
    wins = _winfile(tmp_path / "w.json")
    assert _run(*_args(hist, clips, specs, out, wins=wins),
                "--memory-root", str(mem)) == 0
    assert ((out / "propermem" / "full_history" / f"{UID}.jsonl").read_bytes()
            == (hist / f"{UID}.jsonl").read_bytes())
    assert ((out / "propermem" / "clips" / f"{UID}.mp4").read_bytes()
            == (clips / f"{UID}.mp4").read_bytes())
    for suffix in ("episodic.json", "semantic.json", "passages.json", "entity_list.json",
                   "entity_emb.npy", "semantic_emb.npy", "passages_emb.npy"):
        assert ((out / "memory" / f"{UID}.{suffix}").read_bytes()
                == (mem / f"{UID}.{suffix}").read_bytes())
    manifest = json.loads((out / "memory" / f"{UID}.manifest.json").read_text())
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["imported_not_rebuilt"] is True
    assert manifest["embedding_signature"] == "BAAI/bge-large-en-v1.5:8"
    assert manifest["source"]["manifest_sha256"] == (
        __import__("hashlib").sha256((mem / f"{UID}.manifest.json").read_bytes()).hexdigest())
    assert set(manifest["sha256"]) == {f"{UID}.{s}" for s in
        ("episodic.json", "semantic.json", "passages.json", "entity_list.json",
         "entity_emb.npy", "semantic_emb.npy", "passages_emb.npy")}
    store = MemoryPaths.for_uid(out / "memory", UID)
    assert store.manifest.is_file() and not (out / "memory" / f"{UID}__x.manifest.json").exists()


def test_tagged_exact_bytes_and_row_order_hashes(tmp_path: Path):
    hist, clips, specs = _flat(tmp_path)
    mem = _frozen_mem(tmp_path / "m", tag="qwen38fh")
    tagged_names = [f"{UID}__qwen38fh.{s}" for s in
                    ("episodic.json", "semantic.json", "passages.json", "entity_list.json",
                     "entity_emb.npy", "semantic_emb.npy", "passages_emb.npy")]
    before = {n: (mem / n).read_bytes() for n in tagged_names}
    out = tmp_path / "o"
    assert _run(*_args(hist, clips, specs, out),
                "--memory-root", str(mem), "--memory-tag", "qwen38fh") == 0
    canon = [f"{UID}.{s}" for s in
             ("episodic.json", "semantic.json", "passages.json", "entity_list.json",
              "entity_emb.npy", "semantic_emb.npy", "passages_emb.npy")]
    for tagged, name in zip(tagged_names, canon):
        assert (out / "memory" / name).read_bytes() == before[tagged]
    ent = np.load(out / "memory" / f"{UID}.entity_emb.npy")
    assert ent.shape == (2, 8)
    entity_list = json.loads((out / "memory" / f"{UID}.entity_list.json").read_text())
    assert entity_list == ["I", "milk"]
    manifest = json.loads((out / "memory" / f"{UID}.manifest.json").read_text())
    assert manifest["source"]["tag"] == "qwen38fh"


def test_wrong_signature_and_malformed_and_misaligned_fail(tmp_path: Path):
    hist, clips, specs = _flat(tmp_path)
    mem = _frozen_mem(tmp_path / "m")
    wins = _winfile(tmp_path / "w.json")
    base = ["--history-root", str(hist), "--clips-root", str(clips),
            "--clip-specs", str(specs), "--uid", UID,
            "--windows", str(wins), "--memory-root", str(mem)]
    assert _run(*base, "--output-root", str(tmp_path / "o1"),
                "--embedding-signature", "other:8") == 1
    (mem / f"{UID}.entity_list.json").write_text("{broken")
    assert _run(*base, "--output-root", str(tmp_path / "o2")) == 1
    _frozen_mem(mem, manifest=False)
    np.save(mem / f"{UID}.entity_emb.npy", np.zeros((5, 8), dtype=np.float32))
    assert _run(*base, "--output-root", str(tmp_path / "o3"),
                "--embedding-signature", "BAAI/bge-large-en-v1.5:8") == 1


def test_defaults_resolve_inside_repo(tmp_path: Path, monkeypatch):
    hist = _hist(tmp_path / "hist", uid="0dca9b08_169-341")
    clips = _clips(tmp_path / "clips", uid="0dca9b08_169-341")
    monkeypatch.chdir(ROOT)
    out = tmp_path / "o"
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--output-root", str(out), "--uid", "0dca9b08_169-341") == 0
    windows = json.loads((out / "propermem" / "windows.json").read_text())
    assert windows["0dca9b08_169-341"] == [[52.5, 75.0]]
    spec_row = None
    with (ROOT / "evalsets" / "clip_specs.jsonl").open(encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sample_id") == "0dca9b08_169-341":
                spec_row = row
                break
    assert spec_row is not None
    prepared_trig = json.loads(
        (out / "propermem" / "trigger" / "0dca9b08_169-341.json").read_text())
    assert prepared_trig["trigger_window"] == [[52.5, 75.0]]
    assert prepared_trig["target"] == spec_row["annotations"]["trigger"]["target"]
    assert prepared_trig["closes_when"] == spec_row["annotations"]["trigger"]["closes_when"]
    prepared_mem = json.loads(
        (out / "propermem" / "memory" / "0dca9b08_169-341.json").read_text())
    assert prepared_mem == spec_row["annotations"]["memory"]


def test_default_specs_resolve_to_repo_not_cwd(tmp_path: Path, monkeypatch):
    hist = _hist(tmp_path / "hist", uid="0dca9b08_169-341")
    clips = _clips(tmp_path / "clips", uid="0dca9b08_169-341")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "o"
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--output-root", str(out), "--uid", "0dca9b08_169-341",
                "--windows-none") == 0
    assert (out / "propermem" / "clips" / "0dca9b08_169-341.mp4").is_file()


def test_cli_help_and_existing_output_rejected(tmp_path: Path, capsys):
    with pytest.raises(SystemExit) as exc:
        prep.main(["--help"])
    assert exc.value.code == 0
    hist, clips, specs = _flat(tmp_path)
    out = tmp_path / "o"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    wins = _winfile(tmp_path / "w.json")
    assert _run(*_args(hist, clips, specs, out, wins=wins)) == 1
    capsys.readouterr()
    assert _run(*_args(hist, clips, specs, tmp_path / "o2", wins=wins)) == 0
    assert _run(*_args(hist, clips, specs, tmp_path / "o2", wins=wins)) == 1


def test_uid_repeat_and_list_order_preserved(tmp_path: Path):
    hist = _hist(tmp_path / "hist", UID)
    _hist(tmp_path / "hist", UID2)
    clips = _clips(tmp_path / "clips", UID)
    _clips(tmp_path / "clips", UID2)
    row2 = _row(UID2, trigger_window=[[0.0, 12.0]],
                annotations={"trigger": dict(TRIG),
                             "memory": [dict(m) for m in MEM],
                             "visibility": {"clip_duration": 99.5, "observations": []}})
    specs = _specs(tmp_path / "specs.jsonl", [_row(UID), row2])
    wins = tmp_path / "w.json"
    wins.write_text(json.dumps({UID: WIN, UID2: [[0.0, 12.0]]}))
    out = tmp_path / "o"
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(specs), "--output-root", str(out),
                "--uid", UID, "--uid", UID2, "--windows", str(wins)) == 0
    assert (out / "propermem" / "uids.txt").read_text() == UID + "\n" + UID2 + "\n"
    out2 = tmp_path / "o2"
    lst = tmp_path / "uids.txt"
    lst.write_text(UID2 + "\n" + UID + "\n")
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(specs), "--output-root", str(out2),
                "--uid-list", str(lst), "--windows", str(wins)) == 0
    assert (out2 / "propermem" / "uids.txt").read_text() == UID2 + "\n" + UID + "\n"


def test_windows_none_uses_top_window(tmp_path: Path):
    hist, clips, specs = _flat(tmp_path)
    out = tmp_path / "o"
    assert _run(*_args(hist, clips, specs, out)) == 0
    assert json.loads((out / "propermem" / "windows.json").read_text()) == {}
    trig = json.loads((out / "propermem" / "trigger" / f"{UID}.json").read_text())
    assert trig["trigger_window"] == WIN
    assert trig["target"] == "milk"
    ds = Dataset(out / "propermem")
    assert ds.clip(UID).windows == tuple(tuple(w) for w in WIN)


def test_explicit_window_override_applies_to_generated_trigger(tmp_path: Path):
    hist, clips, specs = _flat(tmp_path)
    selected = [[10.0, 14.6]]
    wins = _winfile(tmp_path / "w.json", win=selected)
    out = tmp_path / "out"
    assert _run(*_args(hist, clips, specs, out, wins=wins)) == 0
    trigger = json.loads((out / "propermem" / "trigger" / f"{UID}.json").read_text())
    assert trigger["trigger_window"] == selected
    assert trigger["target"] == "milk"
    assert normalize_windows(selected) == ((10.0, 14.6),)


def test_negative_top_window_none_permitted(tmp_path: Path):
    row = _row(trigger_window=None)
    specs = _specs(tmp_path / "specs.jsonl", [row])
    hist = _hist(tmp_path / "hist")
    clips = _clips(tmp_path / "clips")
    out = tmp_path / "o"
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(specs), "--output-root", str(out),
                "--uid", UID, "--windows-none") == 0
    trig = json.loads((out / "propermem" / "trigger" / f"{UID}.json").read_text())
    assert trig["trigger_window"] is None
    ds = Dataset(out / "propermem")
    clip = ds.clip(UID)
    assert clip.windows == ()
    assert clip.is_negative


def test_multiple_targets_preserved(tmp_path: Path):
    mem_list = [
        {"target": "milk", "worth_triggering": True, "open_loop": "buy milk",
         "closes_when": "done",
         "memories": [{"kind": "past", "text": "saw milk", "day_offset": 3}]},
        {"target": "keys", "worth_triggering": False, "open_loop": "",
         "closes_when": "", "memories": []},
    ]
    row = _row(annotations={"trigger": dict(TRIG), "memory": mem_list,
                            "visibility": dict(VIS)})
    specs = _specs(tmp_path / "specs.jsonl", [row])
    hist = _hist(tmp_path / "hist")
    clips = _clips(tmp_path / "clips")
    out = tmp_path / "o"
    assert _run(*_args(hist, clips, specs, out)) == 0
    prepared = json.loads((out / "propermem" / "memory" / f"{UID}.json").read_text())
    assert prepared == mem_list
    assert [Target.from_json(m).target for m in prepared] == ["milk", "keys"]


# ---- guards ----


def test_dangerous_ids_duplicates_and_unknown_uid_rejected(tmp_path: Path):
    hist, clips, specs = _flat(tmp_path)
    wins = _winfile(tmp_path / "w.json")
    for bad in ("../evil", "a/b", "", ".hidden"):
        assert _run("--history-root", str(hist), "--clips-root", str(clips),
                    "--clip-specs", str(specs),
                    "--output-root", str(tmp_path / f"o-{abs(hash(bad))}"),
                    "--uid", bad, "--windows-none") == 1
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(specs),
                "--output-root", str(tmp_path / "odup"), "--uid", UID, "--uid", UID,
                "--windows", str(wins)) == 1
    specs2 = _specs(tmp_path / "s2.jsonl", [_row("zzz_other_1")])
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(specs2),
                "--output-root", str(tmp_path / "ounk"), "--uid", "zzz_other_1",
                "--windows-none") == 1
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(specs),
                "--output-root", str(tmp_path / "ounk2"), "--uid", "zzz_missing_uid",
                "--windows-none") == 1


def test_clip_specs_duplicate_and_malformed_rejected(tmp_path: Path):
    hist, clips, _ = _flat(tmp_path)
    dup = _specs(tmp_path / "dup.jsonl", [_row(), _row()])
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(dup),
                "--output-root", str(tmp_path / "o1"), "--uid", UID,
                "--windows-none") == 1
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(bad),
                "--output-root", str(tmp_path / "o2"), "--uid", UID,
                "--windows-none") == 1
    nonobj = tmp_path / "nonobj.jsonl"
    nonobj.write_text("[1, 2]\n")
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(nonobj),
                "--output-root", str(tmp_path / "o3"), "--uid", UID,
                "--windows-none") == 1
    badnum = tmp_path / "badnum.jsonl"
    badnum.write_text(json.dumps(_row(duration_s=-3)) + "\n")
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(badnum),
                "--output-root", str(tmp_path / "o4"), "--uid", UID,
                "--windows-none") == 1
    badid = tmp_path / "badid.jsonl"
    badid.write_text(json.dumps(_row(uid="../evil")) + "\n")
    assert _run("--history-root", str(hist), "--clips-root", str(badid),
                "--clip-specs", str(badid),
                "--output-root", str(tmp_path / "o5"), "--uid", "../evil",
                "--windows-none") == 1


def test_symlink_escape_and_source_outside_rejected(tmp_path: Path):
    hist, clips, specs = _flat(tmp_path)
    (hist / f"{UID}.jsonl").unlink()
    (hist / f"{UID}.jsonl").symlink_to(tmp_path / "elsewhere.jsonl")
    (tmp_path / "elsewhere.jsonl").write_text(json.dumps(CAPTION) + "\n")
    assert _run(*_args(hist, clips, specs, tmp_path / "o1")) == 1
    hist2 = _hist(tmp_path / "hist2")
    specs2 = _specs(tmp_path / "specs2.jsonl", [_row()])
    clips2 = _clips(tmp_path / "clips2")
    linkroot = tmp_path / "linkclips"
    linkroot.symlink_to(tmp_path / "nowhere", target_is_directory=True)
    assert _run("--history-root", str(hist2), "--clips-root", str(linkroot),
                "--clip-specs", str(specs2),
                "--output-root", str(tmp_path / "o2"), "--uid", UID,
                "--windows-none") == 1


def test_missing_history_clip_and_window_gap_fail_clean(tmp_path: Path):
    hist, clips, specs = _flat(tmp_path)
    (clips / f"{UID}.mp4").unlink()
    wins = _winfile(tmp_path / "w.json")
    assert _run(*_args(hist, clips, specs, tmp_path / "o1", wins=wins)) == 1
    _clips(clips, UID)
    (hist / f"{UID}.jsonl").unlink()
    assert _run(*_args(hist, clips, specs, tmp_path / "o2", wins=wins)) == 1
    _hist(hist, UID)
    (tmp_path / "empty.json").write_text("{}")
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(specs),
                "--output-root", str(tmp_path / "o3"), "--uid", UID,
                "--windows", str(tmp_path / "empty.json")) == 1


def test_output_overlap_and_repo_root_rejected(tmp_path: Path, monkeypatch):
    hist, clips, specs = _flat(tmp_path)
    assert _run(*_args(hist, clips, specs, hist)) == 1
    assert _run(*_args(hist, clips, specs, clips)) == 1
    inner_specs = tmp_path / "o3" / "specs.jsonl"
    inner_specs.parent.mkdir(parents=True)
    inner_specs.write_text((_specs(tmp_path / "specs.jsonl", [_row()])).read_text())
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(inner_specs),
                "--output-root", str(tmp_path / "o3"), "--uid", UID,
                "--windows-none") == 1
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(prep, "REPO_ROOT", repo)
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(specs),
                "--output-root", str(repo), "--uid", UID,
                "--windows-none") == 1


@pytest.mark.parametrize("tag", [None, "qwen38fh"])
def test_original_manifest_filename_hashes_reject_changed_payload(tmp_path, tag):
    hist, clips, specs = _flat(tmp_path)
    mem = _frozen_mem(tmp_path / "m", tag=tag)
    prefix = UID if tag is None else f"{UID}__{tag}"
    path = mem / f"{prefix}.entity_list.json"
    path.write_text(json.dumps(["different entity", "milk"]))
    args = ["--history-root", str(hist), "--clips-root", str(clips),
            "--clip-specs", str(specs),
            "--uid", UID, "--windows-none", "--memory-root", str(mem),
            "--output-root", str(tmp_path / "out")]
    if tag:
        args += ["--memory-tag", tag]
    assert _run(*args) == 1
    assert not (tmp_path / "out" / "memory").exists()


@pytest.mark.parametrize("signature", ["model:1024", True, "model", " model:8"])
def test_signature_must_agree_with_actual_matrix_dimension(tmp_path, signature):
    hist, clips, specs = _flat(tmp_path)
    mem = _frozen_mem(tmp_path / "m")
    path = mem / f"{UID}.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["embedding_signature"] = signature
    path.write_text(json.dumps(manifest))
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(specs),
                "--uid", UID, "--windows-none", "--memory-root", str(mem),
                "--output-root", str(tmp_path / "out")) == 1


def test_source_mutation_between_preflight_and_copy_is_detected(tmp_path, monkeypatch):
    hist, clips, specs = _flat(tmp_path)
    mem = _frozen_mem(tmp_path / "m")
    original = prep._copy_bytes
    changed = False

    def copy(src, dst, uid):
        nonlocal changed
        if not changed:
            changed = True
            path = mem / f"{UID}.entity_list.json"
            path.write_text(path.read_text() + " ")
        return original(src, dst, uid)
    monkeypatch.setattr(prep, "_copy_bytes", copy)
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(specs),
                "--uid", UID, "--windows-none", "--memory-root", str(mem),
                "--output-root", str(tmp_path / "out")) == 1
    assert not (tmp_path / "out" / "memory" / f"{UID}.manifest.json").exists()


def test_malformed_history_fails_cleanly_instead_of_traceback(tmp_path, capsys):
    hist, clips, specs = _flat(tmp_path)
    (hist / f"{UID}.jsonl").write_text("{not json\n")
    assert _run(*_args(hist, clips, specs, tmp_path / "out")) == 1
    assert "prepare_propermem: error:" in capsys.readouterr().err


def test_default_output_can_be_inside_checkout(tmp_path, monkeypatch):
    hist = _hist(tmp_path / "r")
    clips = _clips(tmp_path / "c")
    specs = _specs(tmp_path / "specs.jsonl", [_row()])
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(prep, "REPO_ROOT", repo)
    monkeypatch.chdir(repo)
    assert _run("--history-root", str(hist), "--clips-root", str(clips),
                "--clip-specs", str(specs), "--uid", UID, "--windows-none") == 0
    assert (repo / "data" / "prepared" / "propermem" / "uids.txt").is_file()


def test_prepared_trigger_uses_selected_window_without_changing_source(tmp_path: Path):
    hist, clips, specs = _flat(tmp_path)
    before_specs = specs.read_bytes()
    before_hist = (hist / f"{UID}.jsonl").read_bytes()
    selected = [[10.0, 14.6]]
    wins = _winfile(tmp_path / "w.json", win=selected)
    out = tmp_path / "out"
    assert _run(*_args(hist, clips, specs, out, wins=wins)) == 0
    trigger = json.loads((out / "propermem" / "trigger" / f"{UID}.json").read_text())
    assert trigger["trigger_window"] == selected
    assert trigger["target"] == "milk"
    assert specs.read_bytes() == before_specs
    assert (hist / f"{UID}.jsonl").read_bytes() == before_hist


def test_missing_malformed_annotations_fail_without_partial_writes(tmp_path: Path):
    hist = _hist(tmp_path / "hist")
    clips = _clips(tmp_path / "clips")

    def expect_fail(rows: list[dict], outname: str):
        specs = _specs(tmp_path / f"{outname}.jsonl", rows)
        out = tmp_path / outname
        assert _run("--history-root", str(hist), "--clips-root", str(clips),
                    "--clip-specs", str(specs), "--output-root", str(out),
                    "--uid", UID, "--windows-none") == 1
        assert not (out / "propermem" / "trigger" / f"{UID}.json").exists()
        assert not (out / "propermem" / "memory" / f"{UID}.json").exists()
        assert not (out / "propermem" / "visibility" / f"{UID}.json").exists()
        assert not (out / "propermem" / "uids.txt").exists()

    bare = _row()
    del bare["annotations"]
    expect_fail([bare], "o-missing-ann")
    bad_trig = _row(annotations={"trigger": [1, 2], "memory": MEM,
                                "visibility": dict(VIS)})
    expect_fail([bad_trig], "o-bad-trig")
    bad_trig2 = _row(annotations={"trigger": {"target": "milk"}, "memory": MEM,
                                 "visibility": dict(VIS)})
    expect_fail([bad_trig2], "o-bad-trig2")
    bad_mem = _row(annotations={"trigger": dict(TRIG), "memory": {"target": "milk"},
                                "visibility": dict(VIS)})
    expect_fail([bad_mem], "o-bad-mem")
    bad_mem2 = _row(annotations={"trigger": dict(TRIG),
                                "memory": [{"target": "milk"}],
                                "visibility": dict(VIS)})
    expect_fail([bad_mem2], "o-bad-mem2")
    bad_vis = _row(annotations={"trigger": dict(TRIG), "memory": [dict(m) for m in MEM],
                                "visibility": {"clip_duration": "long", "observations": []}})
    expect_fail([bad_vis], "o-bad-vis")
    bad_obs = _row(annotations={"trigger": dict(TRIG), "memory": [dict(m) for m in MEM],
                                "visibility": {"clip_duration": 172.0,
                                               "observations": [{"t": 1.0}]}})
    expect_fail([bad_obs], "o-bad-obs")
    bad_obs2 = _row(annotations={"trigger": dict(TRIG), "memory": [dict(m) for m in MEM],
                                 "visibility": {"clip_duration": 172.0,
                                                "observations": "yes"}})
    expect_fail([bad_obs2], "o-bad-obs2")


def test_absent_external_annotation_dirs_still_succeed(tmp_path: Path):
    hist = _hist(tmp_path / "hist")
    clips = _clips(tmp_path / "clips")
    specs = _specs(tmp_path / "specs.jsonl", [_row()])
    for sub in ("trigger", "memory", "visibility"):
        assert not (hist / sub).exists()
    stray = hist / "trigger" / f"{UID}.json"
    stray.parent.mkdir(parents=True)
    stray.write_text('{"forged": true}')
    out = tmp_path / "o"
    assert _run(*_args(hist, clips, specs, out)) == 0
    trig = json.loads((out / "propermem" / "trigger" / f"{UID}.json").read_text())
    assert trig == {"trigger_window": WIN, "target": "milk", "closes_when": "done"}
    assert json.loads((out / "propermem" / "memory" / f"{UID}.json").read_text()) == MEM
    assert json.loads((out / "propermem" / "visibility" / f"{UID}.json").read_text()) == VIS


def test_generated_annotations_match_parsed_targets_and_observations(tmp_path: Path):
    mem_list = [
        {"target": "milk", "worth_triggering": True, "open_loop": "buy milk",
         "closes_when": "done",
         "memories": [{"kind": "past", "text": "saw milk", "day_offset": 3},
                      {"kind": "note", "text": "empty", "day_offset": None}]},
    ]
    vis = {"clip_duration": 172.0,
           "observations": [{"t": 1.5, "state": "yes", "confidence": 0.9},
                            {"t": 3.0, "state": "no", "confidence": 1.0}]}
    row = _row(annotations={"trigger": dict(TRIG), "memory": mem_list,
                            "visibility": vis})
    specs = _specs(tmp_path / "specs.jsonl", [row])
    hist = _hist(tmp_path / "hist")
    clips = _clips(tmp_path / "clips")
    out = tmp_path / "o"
    assert _run(*_args(hist, clips, specs, out)) == 0
    trig, mem_out, vis_out = _expected_from_row(row, WIN)
    assert json.loads((out / "propermem" / "trigger" / f"{UID}.json").read_text()) == trig
    assert json.loads((out / "propermem" / "memory" / f"{UID}.json").read_text()) == mem_out
    assert json.loads((out / "propermem" / "visibility" / f"{UID}.json").read_text()) == vis_out
    parsed = [Target.from_json(m) for m in mem_out]
    assert parsed[0].target == "milk" and parsed[0].memories[0].day_offset == 3
    assert parsed[0].memories[1].day_offset is None
    ds = Dataset(out / "propermem")
    assert ds.clip(UID).visibility[0].t == 1.5
