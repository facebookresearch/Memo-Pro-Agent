# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""CPU CLI integration: real dataset, memory, retrieval, frames and cached scoring."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from memoproagent.backends.openai_compat import OpenAICompatChat, OpenAICompatConfig
from memoproagent.cli import run as run_cli
from memoproagent.config import load_config
from test_end_to_end import UID, ScriptedChat, StubEmbedding, write_dataset, write_memory


class PolicyTransport(OpenAICompatChat):
    """Exercise the real metadata adapter without sending an HTTP request."""

    def __init__(self):
        super().__init__(OpenAICompatConfig(model="fake-policy", base_url="http://127.0.0.1:1/v1"))
        self.requests = []

    def _post(self, path, payload):
        position = len(self.requests)
        self.requests.append(payload)
        turn, round_index = divmod(position, 2)
        finish = "stop"
        if round_index == 0:
            text = "<observe> milk carton </observe>"
            if turn == 0:
                text, finish = "<observe> milk carton", "length"
        elif turn in (1, 2):
            text = "<reason> carton in view </reason><answer> Grab milk. </answer>"
            if turn == 1:
                finish = "length"
        elif turn == 3:
            text = "NO_REMINDER"
        else:
            text = "<reason> no new need </reason><answer> NO_REMINDER </answer>"
        return {
            "choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 20, "completion_tokens": payload["max_tokens"],
                      "total_tokens": 20 + payload["max_tokens"]},
        }


def cli_environment(tmp_path, monkeypatch):
    data_root, memory_root = tmp_path / "data", tmp_path / "memory"
    embedder = StubEmbedding()
    write_dataset(data_root)
    write_memory(memory_root, embedder)
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    config["data"].update(root=str(data_root), uid_list=str(data_root / "uids.txt"), windows=None)
    config["memory"].update(root=str(memory_root), verify_hashes=False)
    config["output"]["root"] = str(tmp_path / "results")
    config["policy"]["model"] = "fake-policy"
    config["gate_backend"]["model"] = "fake-gate"
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    policy = PolicyTransport()
    gate = ScriptedChat([("Object(s):", "FACT: the milk carton needs restocking")])
    monkeypatch.setattr(
        run_cli, "build_chat_backend",
        lambda settings: policy if settings["model"] == "fake-policy" else gate,
    )
    monkeypatch.setattr(run_cli, "build_embedding_backend", lambda _: embedder)
    args = ["--config", str(config_path), "--paradigm", "streaming", "--no-judge",
            "--label", "alignment-cpu"]
    output = tmp_path / "results" / "alignment-cpu"
    return args, output, policy, gate


def test_cli_keeps_degraded_clip_and_reloads_health(tmp_path, monkeypatch, capsys):
    args, output, policy, gate = cli_environment(tmp_path, monkeypatch)
    assert run_cli.main(args) == 0
    assert len(policy.requests) == 12
    assert len(gate.calls) == 6  # Retrieval/gate still executes after open Observe.
    trace_path = output / "traces" / f"{UID}.json"
    original = trace_path.read_bytes()
    trace = json.loads(original)
    assert len(trace["steps"]) == 6
    assert [fire["t"] for fire in trace["fires"]] == [2.0]
    assert trace["steps"][1]["final_finish_reason"] == "length"
    assert trace["steps"][1]["fired"] is False
    assert all(step["policy_frame_size"] == [384, 288] for step in trace["steps"])
    assert trace["health"]["degraded_turn_count"] == 3
    assert trace["health"]["degradation_count"] == 5
    assert trace["health"]["protocol_failure_count"] == 0
    assert "Already spoken: none" in policy.requests[5]["messages"][-1]["content"]
    assert "Already spoken: Grab milk." in policy.requests[7]["messages"][-1]["content"]
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["n"] == 1  # The degraded clip remains in timing scoring.
    assert metrics["perfect"]["count"] == 1
    capsys.readouterr()
    assert run_cli.main(args) == 0
    assert len(policy.requests) == 12  # Cached re-scoring does not re-run the policy.
    assert len(gate.calls) == 6
    assert trace_path.read_bytes() == original
    assert "3 degraded turn(s)" in capsys.readouterr().out


def test_old_cached_trace_warns_without_rewriting(tmp_path, monkeypatch, capsys):
    args, output, policy, _ = cli_environment(tmp_path, monkeypatch)
    assert run_cli.main(args) == 0
    trace_path = output / "traces" / f"{UID}.json"
    trace = json.loads(trace_path.read_text())
    trace.pop("health")
    for step in trace["steps"]:
        for key in list(step):
            if key == "protocol_mode" or "finish_reason" in key or "stop_reason" in key:
                step.pop(key)
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    old_bytes = trace_path.read_bytes()
    capsys.readouterr()
    assert run_cli.main(args) == 0
    output_text = capsys.readouterr()
    assert "predates compatible health" in output_text.err
    assert "--force or a new label" in output_text.err
    assert "compatible health (streaming only)" not in output_text.out
    assert len(policy.requests) == 12
    assert trace_path.read_bytes() == old_bytes
    assert json.loads((output / "metrics.json").read_text())["n"] == 1
