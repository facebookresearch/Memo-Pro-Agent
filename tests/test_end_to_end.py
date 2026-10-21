# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""End-to-end smoke test with stub backends.

Exercises the whole path -- dataset -> memory store -> retrieval -> gate ->
two-round policy loop -> scoring -- without touching a model server. What it
proves is wiring and control flow: that the gate's abstention really suppresses a
fire, that `Already spoken` really reaches round two, and that a streaming
context really rolls at the chunk limit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from memoproagent.agent import Gate, GateConfig, StreamingRunner, WindowedRunner
from memoproagent.agent.runner import PolicyConfig, StreamingConfig, WindowedConfig
from memoproagent.backends.base import GenerationConfig, Message
from memoproagent.data import Dataset
from memoproagent.eval.metrics import score_clip
from memoproagent.memory import MemoryStore, RetrievalConfig, Retriever
from memoproagent.memory.store import MANIFEST_SCHEMA

cv2 = pytest.importorskip("cv2")

UID = "demo_0001"
EMBED_DIM = 8


class StubEmbedding:
    """Deterministic hash embedding. Not semantic, but stable and fast."""

    name = "stub"
    dim = EMBED_DIM

    @property
    def signature(self) -> str:
        return f"{self.name}:{self.dim}"

    def embed(self, texts):
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text.lower())) % (2**32))
            vector = rng.normal(size=self.dim).astype(np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, self.dim), np.float32)


class ScriptedChat:
    """Replies by matching the last user message against a rule list."""

    name = "scripted"
    max_images_per_request = 32

    def __init__(self, rules):
        self.rules = rules
        self.calls: list[list[Message]] = []

    def chat(self, messages, config: GenerationConfig) -> str:
        self.calls.append(list(messages))
        last = messages[-1].text
        for needle, reply in self.rules:
            if needle in last:
                return reply
        raise AssertionError(f"no scripted reply for: {last[:120]!r}")


def write_dataset(root: Path, duration: float = 6.0) -> None:
    for sub in ("clips", "full_history", "trigger", "memory", "visibility"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(root / "clips" / f"{UID}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    for index in range(int(duration * 10)):
        frame = np.full((48, 64, 3), index % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    with (root / "full_history" / f"{UID}.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(4):
            handle.write(
                json.dumps(
                    {
                        "cid": f"D1:{index}",
                        "day": 1,
                        "t": index * 5.0,
                        "clock": f"08:00:{index:02d}.00",
                        "text": "I shook the milk carton; it felt almost empty.",
                        "t_days_before": 9,
                    }
                )
                + "\n"
            )

    (root / "trigger" / f"{UID}.json").write_text(
        json.dumps({"sample_id": UID, "target": "milk carton", "trigger_window": [2.0, 4.0]})
    )
    (root / "memory" / f"{UID}.json").write_text(
        json.dumps(
            [
                {
                    "target": "milk carton",
                    "worth_triggering": True,
                    "open_loop": "The milk carton is almost empty and needs restocking.",
                    "memories": [
                        {"kind": "episodic", "text": "Nine days ago the milk ran low.", "day_offset": 9}
                    ],
                }
            ]
        )
    )
    (root / "visibility" / f"{UID}.json").write_text(
        json.dumps({"clip_duration": duration, "observations": [{"t": 2.0, "state": "yes"}]})
    )
    (root / "uids.txt").write_text(f"{UID}\n")


def write_memory(root: Path, embedder: StubEmbedding) -> None:
    root.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "idx": 0,
            "day": 1,
            "start": "08:00:00.00",
            "days_ago": 9,
            "cids": ["D1:0"],
            "rewrite": "I shook the milk carton; it felt almost empty and I said to grab milk.",
            "triples": [
                ["I", "picked up", "milk carton"],
                ["milk carton", "is", "almost empty"],
                ["milk carton", "needs", "restock on the next trip"],
            ],
            "ok": True,
        }
    ]
    entities = ["milk carton", "fridge door"]
    semantic = [["I", "avoid", "dairy after 8pm"]]
    passages = [events[0]["rewrite"]]

    (root / f"{UID}.episodic.json").write_text(json.dumps({"events": events}))
    (root / f"{UID}.semantic.json").write_text(json.dumps({"triples": semantic}))
    (root / f"{UID}.passages.json").write_text(json.dumps(passages))
    (root / f"{UID}.entity_list.json").write_text(json.dumps(entities))
    np.save(root / f"{UID}.entity_emb.npy", embedder.embed(entities))
    np.save(root / f"{UID}.semantic_emb.npy", embedder.embed([" ".join(t) for t in semantic]))
    np.save(root / f"{UID}.passages_emb.npy", embedder.embed(passages))
    (root / f"{UID}.manifest.json").write_text(
        json.dumps({"schema": MANIFEST_SCHEMA, "uid": UID, "embedding_signature": embedder.signature})
    )


@pytest.fixture
def env(tmp_path):
    embedder = StubEmbedding()
    data_root = tmp_path / "data"
    memory_root = tmp_path / "memory"
    write_dataset(data_root)
    write_memory(memory_root, embedder)
    dataset = Dataset(data_root)
    clip = dataset.clip(UID)
    store = MemoryStore(UID, memory_root, caption_log=clip.caption_log)
    retriever = Retriever(
        store,
        embedder,
        RetrievalConfig(mode="fast", seed_threshold=-1.0, passage_vector_topk=1),
    )
    return clip, retriever


def test_gate_evidence_produces_a_fire(env):
    clip, retriever = env
    gate_backend = ScriptedChat([("Object(s):", "FACT: the milk carton needs restocking")])
    policy = ScriptedChat(
        [
            ("Begin the protocol with <observe>.", "<observe> milk carton </observe>"),
            ("Continue the protocol", "<reason> in view </reason><answer> Grab milk. </answer>"),
        ]
    )
    runner = WindowedRunner(
        policy,
        Gate(retriever, gate_backend, GateConfig(retrieval=retriever.config)),
        WindowedConfig(stride=2.0, policy=PolicyConfig()),
    )
    run = runner.run(clip)
    assert run.fires, "evidence plus a positive decision should produce a fire"
    assert run.fires[0].evidence == "the milk carton needs restocking"


def test_gate_abstention_suppresses_the_fire(env):
    """The same policy that fires with evidence must see `none` without it."""
    clip, retriever = env
    gate_backend = ScriptedChat([("Object(s):", "FACT: none")])
    policy = ScriptedChat(
        [
            ("Begin the protocol with <observe>.", "<observe> milk carton </observe>"),
            ("<evidence>none</evidence>", "<reason> no evidence </reason><answer> NO_REMINDER </answer>"),
        ]
    )
    runner = WindowedRunner(
        policy,
        Gate(retriever, gate_backend, GateConfig(retrieval=retriever.config)),
        WindowedConfig(stride=2.0, policy=PolicyConfig()),
    )
    run = runner.run(clip)
    assert run.fires == []
    assert score_clip(run, None).timing == "no_fire"


def test_already_spoken_reaches_the_second_round(env):
    clip, retriever = env
    gate_backend = ScriptedChat([("Object(s):", "FACT: the milk carton needs restocking")])
    policy = ScriptedChat(
        [
            ("Begin the protocol with <observe>.", "<observe> milk carton </observe>"),
            ("Continue the protocol", "<reason> in view </reason><answer> Grab milk. </answer>"),
        ]
    )
    runner = WindowedRunner(
        policy,
        Gate(retriever, gate_backend, GateConfig(retrieval=retriever.config)),
        WindowedConfig(stride=2.0, policy=PolicyConfig()),
    )
    runner.run(clip)
    second_round_texts = [
        m[-1].text for m in policy.calls if "Continue the protocol" in m[-1].text
    ]
    assert second_round_texts[0].count("Already spoken: none") == 1
    assert "Already spoken: Grab milk." in second_round_texts[1]


def test_streaming_context_rolls_at_the_chunk_limit(env):
    clip, retriever = env
    gate_backend = ScriptedChat([("Object(s):", "FACT: none")])
    policy = ScriptedChat(
        [
            ("Begin the protocol with <observe>.", "<observe> milk carton </observe>"),
            ("Continue the protocol", "<reason> no </reason><answer> NO_REMINDER </answer>"),
        ]
    )
    runner = StreamingRunner(
        policy,
        Gate(retriever, gate_backend, GateConfig(retrieval=retriever.config)),
        StreamingConfig(stride=1.0, chunk_turns=3, keep_turns=1, policy=PolicyConfig()),
    )
    run = runner.run(clip)
    turns = [step["context_turns"] for step in run.steps]
    assert max(turns) < 3, "context must never exceed chunk_turns"
    assert turns[:5] == [0, 1, 2, 1, 2], f"expected a roll back to keep_turns, got {turns[:5]}"


def test_malformed_policy_output_is_not_a_fire(env):
    clip, retriever = env
    gate_backend = ScriptedChat([("Object(s):", "FACT: the milk carton needs restocking")])
    policy = ScriptedChat(
        [
            ("Begin the protocol with <observe>.", "<observe> milk carton </observe>"),
            # Truncated: <answer> never closes.
            ("Continue the protocol", "<reason> in view </reason><answer> Grab mi"),
        ]
    )
    runner = WindowedRunner(
        policy,
        Gate(retriever, gate_backend, GateConfig(retrieval=retriever.config)),
        WindowedConfig(stride=2.0, policy=PolicyConfig()),
    )
    assert runner.run(clip).fires == []


def test_memory_store_rejects_a_foreign_embedding_space(env, tmp_path):
    _, retriever = env

    class OtherEmbedding(StubEmbedding):
        name = "other"

    with pytest.raises(Exception, match="embedding"):
        Retriever(retriever.store, OtherEmbedding(), RetrievalConfig(mode="fast"))


def test_evalset_windows_override_the_trigger_file(tmp_path):
    """The frozen window file is the scoring contract and must win."""
    import json as _json

    data_root = tmp_path / "data"
    write_dataset(data_root)
    # trigger/ says [2.0, 4.0]; the eval set says otherwise.
    windows = tmp_path / "windows.json"
    windows.write_text(_json.dumps({UID: [[1.0, 1.5], [3.0, 5.0]]}))

    assert Dataset(data_root).clip(UID).windows == ((2.0, 4.0),)

    clip = Dataset(data_root, windows=windows).clip(UID)
    assert clip.windows == ((1.0, 1.5), (3.0, 5.0))
    assert clip.in_window(3.5) and not clip.in_window(2.5)


def test_uid_missing_from_the_window_file_is_an_error(tmp_path):
    import json as _json

    data_root = tmp_path / "data"
    write_dataset(data_root)
    windows = tmp_path / "windows.json"
    windows.write_text(_json.dumps({"some_other_uid": [[1.0, 2.0]]}))

    with pytest.raises(Exception, match="no entry in"):
        Dataset(data_root, windows=windows).clip(UID)


def test_local_build_cli_creates_loadable_retrieval_memory(tmp_path, monkeypatch):
    """Build, reload and use an index with scripted services, not a real model."""
    from memoproagent.cli import build_memory as build_cli
    from memoproagent.memory.store import MemoryPaths, sha256_file

    data_root = tmp_path / "data"
    memory_root = tmp_path / "memory"
    write_dataset(data_root)
    history = data_root / "full_history" / f"{UID}.jsonl"
    history_sha = sha256_file(history)
    config_root = tmp_path / "configs"
    config_root.mkdir()
    config = config_root / "example.yaml"
    config.write_text(
        f"data:\n  root: {data_root}\n  uid_list: {data_root / 'uids.txt'}\n"
        f"memory:\n  root: {memory_root}\n  verify_hashes: true\n"
        "gate_backend: {}\nembedding: {}\n"
    )
    embedder = StubEmbedding()
    extractor = ScriptedChat([
        ("Caption stream (one event):", "I shook the milk carton; it was almost empty."),
        ("Convert the paragraph into a JSON dict", json.dumps({"triples": [
            ["milk carton", "is", "almost empty"],
            ["milk carton", "needs", "restock on the next trip"],
        ]})),
        ("Events:", json.dumps([["I", "avoid", "dairy after 8pm"]])),
        ("I shook the milk carton", json.dumps({"named_entities": ["milk carton"]})),
    ])
    monkeypatch.setattr(build_cli, "build_chat_backend", lambda _: extractor)
    monkeypatch.setattr(build_cli, "build_embedding_backend", lambda _: embedder)
    args = ["--config", str(config), "--all", "--workers", "1"]
    assert build_cli.main(args) == 0
    assert len(extractor.calls) == 4
    assert sha256_file(history) == history_sha

    paths = MemoryPaths.for_uid(memory_root, UID)
    assert not paths.missing()
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["embedding_signature"] == embedder.signature
    assert manifest["counts"]["events"] == 1
    assert manifest["counts"]["failed_events"] == 0
    assert manifest["counts"]["semantic"] == 1
    for filename, digest in manifest["sha256"].items():
        assert sha256_file(memory_root / filename) == digest

    guide = Path(__file__).resolve().parents[1] / "docs" / "data.md"
    section = guide.read_text().partition("Verify that the generated files load")[2]
    snippet = section.partition("python - <<'PY'\n")[2].partition("\nPY\n")[0]
    assert snippet
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SAMPLE_ID", UID)
    exec(compile(snippet, str(guide), "exec"), {})

    dataset = Dataset(data_root)
    clip = dataset.clip(UID)
    store = MemoryStore(UID, memory_root, caption_log=clip.caption_log, verify_hashes=True)
    retriever = Retriever(store, embedder, RetrievalConfig(
        mode="fast", seed_threshold=-1.0, passage_vector_topk=1,
    ))
    gate_backend = ScriptedChat([("Object(s):", "FACT: the milk carton needs restocking")])
    policy = ScriptedChat([
        ("Begin the protocol with <observe>.", "<observe> milk carton </observe>"),
        ("Continue the protocol", "<reason> in view </reason><answer> Grab milk. </answer>"),
    ])
    runner = WindowedRunner(
        policy, Gate(retriever, gate_backend, GateConfig(retrieval=retriever.config)),
        WindowedConfig(stride=2.0, policy=PolicyConfig()),
    )
    assert runner.run(clip).fires
    calls_before_skip = len(extractor.calls)
    assert build_cli.main(args) == 0
    assert len(extractor.calls) == calls_before_skip
    assert sha256_file(history) == history_sha
