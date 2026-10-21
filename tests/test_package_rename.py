# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Clean-install contracts for the MemoProAgent software namespace."""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_distribution_and_entry_points_use_new_name():
    import memoproagent

    distribution = importlib.metadata.distribution("memoproagent")
    assert distribution.metadata["Name"] == "memoproagent"
    assert memoproagent.__version__ == distribution.version == "0.1.0"
    scripts = {entry.name: entry.value for entry in distribution.entry_points
               if entry.group == "console_scripts"}
    assert scripts == {
        "memoproagent-run": "memoproagent.cli.run:main",
        "memoproagent-build-memory": "memoproagent.cli.build_memory:main",
    }
    assert (ROOT / "memoproagent").is_dir()
    assert not (ROOT / "properagent").exists()


def test_clean_install_has_no_legacy_aliases():
    assert importlib.util.find_spec("properagent") is None
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        importlib.metadata.distribution("properagent")
    executable_directory = Path(sys.executable).parent
    assert not (executable_directory / "properagent-run").exists()
    assert not (executable_directory / "properagent-build-memory").exists()


@pytest.mark.parametrize("module", ["memoproagent.cli.run", "memoproagent.cli.build_memory"])
def test_new_module_help_is_offline(module):
    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("script", ["memoproagent-run", "memoproagent-build-memory"])
def test_new_installed_console_help_is_offline(script):
    executable = Path(sys.executable).parent / script
    assert executable.is_file()
    result = subprocess.run([str(executable), "--help"], cwd=ROOT,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_serialized_metric_identifiers_remain_stable():
    from memoproagent.eval.metrics import ScoredClip

    row = ScoredClip(uid="fixture", n_fires=0, n_before=0, n_in_window=0,
                     n_after=0, scoring_mode="timing_only").to_json()
    assert row["schema_version"] == "properagent_metrics_v2"
    assert row["protocol"] == "properagent_allfire_v1"
    assert not row["hit"] and not row["perfect"]


def test_existing_serialized_manifest_still_loads(tmp_path):
    from memoproagent.memory.store import MANIFEST_SCHEMA, MemoryStore

    assert MANIFEST_SCHEMA == "properagent_memory_v1"
    uid = "rename-fixture"
    payloads = {
        "episodic.json": {"events": [{"idx": 0, "cids": ["c0"], "days_ago": 1}]},
        "passages.json": ["An unfinished household task."],
        "entity_list.json": ["task"],
    }
    for suffix, payload in payloads.items():
        (tmp_path / f"{uid}.{suffix}").write_text(json.dumps(payload), encoding="utf-8")
    np.save(tmp_path / f"{uid}.entity_emb.npy", np.array([[1.0, 0.0]], dtype=np.float32))
    files = [tmp_path / f"{uid}.{suffix}" for suffix in payloads]
    files.append(tmp_path / f"{uid}.entity_emb.npy")
    manifest = {
        "schema": "properagent_memory_v1",
        "embedding_signature": "fixture:2",
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    (tmp_path / f"{uid}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    store = MemoryStore(uid, tmp_path, verify_hashes=True)
    assert store.manifest == manifest
    assert store.events[0]["idx"] == 0
    assert store.entity_emb.shape == (1, 2)
