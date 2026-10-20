"""Release-only checks: packaging, reader entry points and publication boundaries."""
from __future__ import annotations

import hashlib
import importlib
import re
import subprocess
from pathlib import Path

import pytest

from properagent.config import load_config
from properagent.data.schema import Target, normalize_windows

ROOT = Path(__file__).resolve().parents[1]


def test_release_windows_and_order_are_pinned():
    import json

    windows_path = ROOT / "evalsets" / "eval800_windows.json"
    assert hashlib.sha256(windows_path.read_bytes()).hexdigest() == (
        "752860cddc28c00a1451e217eccc2b2aec26506372898e16f421159632ce5bfc"
    )
    windows = json.loads(windows_path.read_text())
    assert windows["0dca9b08_389-509"] == [[10.0, 14.6]]
    for value in windows.values():
        normalize_windows(value)
    uids_path = ROOT / "evalsets" / "eval800.txt"
    assert hashlib.sha256(uids_path.read_bytes()).hexdigest() == (
        "c90e560c2b997b65a25bffab96f074bbabc5800850a4af47252b57a5d6e93479"
    )
    uids = uids_path.read_text().splitlines()
    assert len(uids) == len(set(uids)) == 800
    assert set(uids) == set(windows)


def test_example_config_has_all_sections_and_no_included_data_assumption():
    config = load_config(ROOT / "configs" / "example.yaml")
    default = load_config(ROOT / "configs" / "default.yaml")
    assert set(default) == set(config)
    assert config["gate"]["prompt_variant"] == "version_free"
    assert config["policy"]["base_url"] == config["gate_backend"]["base_url"]
    assert config["memory"]["verify_hashes"] is True
    assert config["data"]["uid_list"].startswith("data/prepared/")
    assert config["data"]["windows"].startswith("data/prepared/")


def test_console_entry_points_resolve():
    tomllib = pytest.importorskip("tomllib")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    scripts = project["project"]["scripts"]
    assert set(scripts) == {"properagent-run", "properagent-build-memory"}
    for target in scripts.values():
        module, function = target.split(":")
        assert callable(getattr(importlib.import_module(module), function))
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == ["properagent*"]


@pytest.mark.parametrize("path", ["README.md", "docs/data.md", "docs/evaluation.md"])
def test_local_reader_links_resolve(path):
    source = ROOT / path
    text = source.read_text()
    links = re.findall(r"\]\(([^)]+)\)", text) + re.findall(r'src="([^"]+)"', text)
    for link in links:
        if link.startswith(("https://", "http://", "#")):
            continue
        target = (source.parent / link.split("#", 1)[0]).resolve()
        assert target.is_relative_to(ROOT)
        assert target.is_file(), (source, link)
    assert "/storage/" not in text
    assert "/lustre/" not in text


def test_no_experiment_or_data_assets_are_publishable():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    for path in result.stdout.splitlines():
        parts = Path(path).parts
        assert not set(parts) & {"baselines", "logs", "runs", "__pycache__", "checkpoints"}
        assert Path(path).suffix not in {".mp4", ".npy", ".npz", ".pyc", ".pt", ".safetensors"}
    assert not (ROOT / "baselines").exists()
    ignore = (ROOT / ".gitignore").read_text()
    assert "*.mp4" in ignore and "*.npy" in ignore
    assert "!examples/data" not in ignore
    publishable = set(result.stdout.splitlines())
    for module in (ROOT / "properagent").rglob("*.py"):
        assert str(module.relative_to(ROOT)) in publishable


def test_clip_metadata_contains_only_public_fields_and_matches_windows():
    import json
    import math

    rows = [json.loads(line) for line in (ROOT / "evalsets" / "clip_specs.jsonl").read_text().splitlines()]
    uids = (ROOT / "evalsets" / "eval800.txt").read_text().splitlines()
    windows = json.loads((ROOT / "evalsets" / "eval800_windows.json").read_text())
    assert [row["sample_id"] for row in rows] == uids
    for row in rows:
        assert set(row) == {
            "sample_id", "ego4d_video_uid", "source_asset", "source_start_s",
            "source_end_s", "duration_s", "expected_frames", "expected_fps",
            "expected_height", "codec", "vf", "clip_file", "trigger_window",
            "annotations",
        }
        annotations = row["annotations"]
        assert set(annotations) == {"trigger", "memory", "visibility"}
        assert set(annotations["trigger"]) == {"target", "closes_when"}
        assert isinstance(annotations["memory"], list)
        for target in annotations["memory"]:
            assert set(target) == {
                "target", "worth_triggering", "open_loop", "closes_when", "memories",
            }
            assert isinstance(target["worth_triggering"], bool)
            Target.from_json(target)
            for memory in target["memories"]:
                assert set(memory) == {"kind", "text", "day_offset"}
        visibility = annotations["visibility"]
        assert set(visibility) == {"clip_duration", "observations"}
        assert math.isfinite(visibility["clip_duration"]) and visibility["clip_duration"] > 0
        for observation in visibility["observations"]:
            assert set(observation) == {"t", "state", "confidence"}
            assert math.isfinite(observation["t"])
            assert math.isfinite(observation["confidence"])
            assert observation["state"] in {"yes", "no", "uncertain"}
        assert row["trigger_window"]
        assert row["clip_file"] == row["sample_id"] + ".mp4"
        assert math.isclose(row["source_end_s"] - row["source_start_s"], row["duration_s"], abs_tol=0.001)
        assert "/storage/" not in json.dumps(row)
        assert "/lustre/" not in json.dumps(row)


def test_reader_cut_example_uses_sample_id_and_source_timestamps(tmp_path, monkeypatch):
    import json

    doc = (ROOT / "docs" / "data.md").read_text()
    assert "export SAMPLE_ID=" in doc
    snippet = doc.partition("python - <<'PY'\n")[2].partition("\nPY\n")[0]
    assert snippet
    row = {
        "sample_id": "example_0", "ego4d_video_uid": "source_video_0",
        "source_start_s": 12.5, "duration_s": 30.0,
        "vf": "scale=-2:540", "clip_file": "example_0.mp4",
    }
    metadata_dir = tmp_path / "evalsets"
    metadata_dir.mkdir()
    (metadata_dir / "clip_specs.jsonl").write_text(json.dumps(row) + "\n")
    videos = tmp_path / "videos"
    videos.mkdir()
    source = videos / "source_video_0.mp4"
    source.write_bytes(b"test fixture; FFmpeg is not invoked")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SAMPLE_ID", "example_0")
    monkeypatch.setenv("EGO4D_VIDEO_DIR", str(videos))
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: calls.append((args, kwargs)))
    exec(compile(snippet, str(ROOT / "docs" / "data.md"), "exec"), {})
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0:4] == ["ffmpeg", "-n", "-ss", "12.5"]
    assert args[args.index("-i") + 1] == str(source)
    assert args[args.index("-t") + 1] == "30.0"
    assert args[-1] == "data/clips/example_0.mp4"
    assert kwargs == {"check": True}


def test_history_reader_uses_pinned_snapshot_and_local_index_build():
    doc = (ROOT / "docs" / "data.md").read_text()
    readme = (ROOT / "README.md").read_text()
    url = "https://huggingface.co/datasets/facebook/MemProBench"
    assert url in doc and url in readme
    revision = re.search(r"export HISTORY_REV=([a-f0-9]{40})\b", doc)
    assert revision
    assert '--revision "$HISTORY_REV"' in doc
    assert "full_history/01ef2c24_317-451.jsonl history_manifest.json uids.txt" in doc
    assert "--history-root data/hf/full_history" in doc
    assert "properagent-build-memory --config configs/example.yaml" in doc
    assert "--memory-root" not in doc
    assert "-k local_build_cli" in doc
