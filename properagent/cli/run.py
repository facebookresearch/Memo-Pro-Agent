"""Run ProPerAgent over an evaluation set and score it.

    properagent-run --paradigm windowed
    properagent-run --paradigm streaming --uid b1000_cleaning_laundry_0001
    properagent-run --paradigm windowed --no-judge      # timing only, no judge calls

Each clip writes a trace to ``<output.root>/<label>/traces/<uid>.json`` and is
resumable: an existing trace is reused unless --force. Scoring reads the traces,
so it can be re-run with a different judge without re-running the policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

from ..agent import Gate, GateConfig, StreamingConfig, StreamingRunner, WindowedConfig, WindowedRunner
from ..agent.runner import PolicyConfig
from ..backends import build_chat_backend, build_embedding_backend
from ..config import load_config, section
from ..data import Clip, Dataset, Fire, normalize_windows
from ..data.schema import ClipRun
from ..eval import Judge, JudgeConfig, format_summary, score_clip, summarize, write_json
from ..eval.judge import Category, JUDGE_SYSTEM, extract_judge_frames
from ..eval.metrics import select_fires_for_judging
from ..memory import MemoryStore, RetrievalConfig, Retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", dest="overrides", default=[], metavar="KEY=VALUE")
    parser.add_argument("--paradigm", choices=("windowed", "streaming"), required=True)
    parser.add_argument("--uid", action="append", default=[], help="restrict to these UIDs")
    parser.add_argument("--limit", type=int, default=None, help="first N UIDs (smoke tests)")
    parser.add_argument("--label", default=None, help="output subdirectory name")
    parser.add_argument("--no-judge", action="store_true", help="score timing only")
    parser.add_argument("--force", action="store_true", help="ignore cached traces")
    args = parser.parse_args(argv)

    config = load_config(args.config, args.overrides)
    label = args.label or f"properagent-{args.paradigm}"
    out_dir = Path(config["output"]["root"]) / label
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    dataset = Dataset(
        config["data"]["root"],
        config["data"].get("uid_list"),
        config["data"].get("windows"),
    )
    uids = args.uid or list(dataset.uids())
    if args.limit:
        uids = uids[: args.limit]
    if not uids:
        parser.error("evaluation set is empty")

    policy = build_chat_backend(section(config, "policy"))
    gate_backend = build_chat_backend(section(config, "gate_backend"))
    embedder = build_embedding_backend(section(config, "embedding"))

    retrieval = RetrievalConfig(**section(config, "retrieval"))
    gate_section = section(config, "gate")
    sampling = PolicyConfig(**section(config, "policy_sampling"))
    memory_root = config["memory"]["root"]
    verify_hashes = bool(config["memory"].get("verify_hashes", False))

    judge = None
    if not args.no_judge:
        judge = Judge(
            build_chat_backend(section(config, "judge_backend")),
            JudgeConfig(**section(config, "judge")),
        )

    grace = float(config["scoring"]["grace_seconds"])
    scored_rows = []

    for position, uid in enumerate(uids, start=1):
        trace_path = trace_dir / f"{uid}.json"
        clip = dataset.clip(uid)

        if trace_path.is_file() and not args.force:
            run = _load_run(trace_path, expected_uid=uid, expected_paradigm=args.paradigm)
        else:
            store = MemoryStore(
                uid, memory_root, caption_log=clip.caption_log, verify_hashes=verify_hashes
            )
            gate = Gate(
                Retriever(store, embedder, retrieval),
                gate_backend,
                GateConfig(retrieval=retrieval, **gate_section),
            )
            runner = _build_runner(args.paradigm, policy, gate, sampling, config)
            run = runner.run(clip, config_label=label)
            write_json(trace_path, run.to_json())

        row = _score(clip, run, judge, config, grace)
        scored_rows.append(row)
        status = "PERFECT" if row.perfect else ("hit" if row.hit else row.timing)
        print(f"[{position}/{len(uids)}] {uid}: {row.n_fires} fire(s), {status}", flush=True)

    metrics = summarize(scored_rows, label=label)
    write_json(out_dir / "scored.json", [r.to_json() for r in scored_rows])
    write_json(out_dir / "metrics.json", metrics)
    print()
    print(format_summary(metrics))
    if args.no_judge:
        print("\nNOTE: --no-judge was set, so PERFECT and HIT reflect timing only "
              "and are an upper bound on the judged numbers.")
    return 0


def _build_runner(paradigm: str, policy, gate, sampling: PolicyConfig, config: dict):
    if paradigm == "windowed":
        return WindowedRunner(
            policy, gate, WindowedConfig(policy=sampling, **section(config, "windowed"))
        )
    return StreamingRunner(
        policy, gate, StreamingConfig(policy=sampling, **section(config, "streaming"))
    )


def _load_run(
    path: Path, *, expected_uid: str | None = None, expected_paradigm: str | None = None
) -> ClipRun:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for field, expected in (("uid", expected_uid), ("paradigm", expected_paradigm)):
        if expected is not None and payload.get(field) != expected:
            raise ValueError(f"cached trace {path} has mismatched {field}; use --force to regenerate")
    return ClipRun(
        uid=payload["uid"],
        paradigm=payload["paradigm"],
        config_label=payload["config_label"],
        windows=normalize_windows(payload.get("windows") or None),
        fires=[
            Fire(
                t=float(f["t"]),
                answer=f["answer"],
                observe=f.get("observe", ""),
                reason=f.get("reason", ""),
                evidence=f.get("evidence", ""),
            )
            for f in payload.get("fires", [])
        ],
        steps=payload.get("steps", []),
    )


def _score(clip: Clip, run: ClipRun, judge: Judge | None, config: dict, grace: float):
    """Judge capped exact-text representatives; count every original fire."""
    if run.uid != clip.uid:
        raise ValueError("run UID does not match the current clip")
    if any(not fire.answer for fire in run.fires):
        raise ValueError("trace contains an empty fire answer; use --force to regenerate")
    # Re-scoring must use the current evaluation contract, not a cached window.
    run = replace(run, windows=clip.windows)
    judge_config = judge.config if judge is not None else JudgeConfig(**config.get("judge", {}))
    evaluation = {
        "selection": "exact_text_first_in_window",
        "judge_prompt_sha256": hashlib.sha256(JUDGE_SYSTEM.encode()).hexdigest(),
        "max_in_window": judge_config.max_in_window,
        "max_out_of_window": judge_config.max_out_of_window,
        "judge_frames": {
            "n_frames": judge_config.n_frames,
            "frame_gap": judge_config.frame_gap,
            "encoding": "png",
            "resolution": "original",
        },
        "judge_max_tokens": judge_config.max_tokens,
        "judge_temperature": judge_config.temperature,
        "judge_model": config.get("judge_backend", {}).get("model"),
    }
    if judge is None:
        return score_clip(run, None, grace=grace, scoring_mode="timing_only", evaluation=evaluation)

    selected = select_fires_for_judging(
        run.fires, run.windows, grace,
        max_in_window=judge_config.max_in_window,
        max_out_of_window=judge_config.max_out_of_window,
    )
    rank = {Category.GOOD: 3, Category.BASIC: 2, Category.BAD: 1, Category.NON_MEMORY: 0}
    judgments = []
    best_verdict = best_fire = None
    for fire, in_window in selected:
        frames, labels = extract_judge_frames(
            clip.video_path, fire.t, judge_config.n_frames, judge_config.frame_gap
        )
        verdict = judge(clip, fire, frames, image_labels=labels)
        judgments.append({
            "t": fire.t,
            "answer": fire.answer,
            "in_window": in_window,
            "image_labels": labels,
            **verdict.to_json(),
        })
        if in_window and (best_verdict is None or rank[verdict.category] > rank[best_verdict.category]):
            best_verdict, best_fire = verdict, fire
    return score_clip(
        run, best_verdict, best_fire, grace=grace,
        selected_judgments=judgments, evaluation=evaluation,
    )


if __name__ == "__main__":
    sys.exit(main())
