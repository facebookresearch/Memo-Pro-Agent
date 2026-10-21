# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""GRPO post-training entry point (MS-Swift).

This module is both the reward plugin and the launcher. MS-Swift loads it with
``--external_plugins``, finds the registered reward by name, and calls it once
per rollout group.

    python -m memoproagent.rl.train_grpo --print-command        # inspect the argv
    bash <(python -m memoproagent.rl.train_grpo --print-command)

The reward itself lives in ``memoproagent.rl.reward`` and is the binary one from
the paper. Two invariants this file is responsible for:

* **Evidence tokens are masked.** The gated evidence arrives as an environment
  message, not as something the model produced. Swift must report a zero loss
  mask for those spans; if it does not, the reward refuses to score the rollout
  rather than quietly training the model to reproduce its own tool output.
* **A malformed or truncated trajectory scores -1.** Handled in
  ``reward.Trajectory.well_formed``.

The judge is called at most once per rollout, only for a correctly timed fire.
Point ``MEMOPROAGENT_JUDGE_URL`` / ``MEMOPROAGENT_JUDGE_MODEL`` at a served judge;
serve it on its own GPU so it does not contend with rollout generation.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..backends.openai_compat import OpenAICompatChat, OpenAICompatConfig
from ..data.schema import Clip, Fire, MemoryItem, Target
from ..eval.judge import Judge, JudgeConfig
from .reward import StateClass, Trajectory, reward

REWARD_NAME = "memoproagent_binary_v1"
MAX_JUDGE_FRAMES = 4


class RewardContractError(RuntimeError):
    """A rollout violated an invariant the reward depends on."""


def _judge_backend() -> OpenAICompatChat:
    return OpenAICompatChat(
        OpenAICompatConfig(
            model=os.environ.get("MEMOPROAGENT_JUDGE_MODEL", "gpt-4o"),
            base_url=os.environ.get("MEMOPROAGENT_JUDGE_URL", "http://127.0.0.1:8003/v1"),
            api_key_env="MEMOPROAGENT_JUDGE_KEY",
            rpm=int(os.environ.get("MEMOPROAGENT_JUDGE_RPM", "0")),
        )
    )


class MemoProAgentBinaryReward:
    """Registered with MS-Swift as an ORM under :data:`REWARD_NAME`."""

    def __init__(self, args: Any = None, **kwargs: Any) -> None:
        self.args = args
        self._judge: Judge | None = None
        self.last_breakdowns: list[dict[str, Any]] = []

    @property
    def judge(self) -> Judge:
        if self._judge is None:
            self._judge = Judge(_judge_backend(), JudgeConfig())
        return self._judge

    def __call__(self, completions: Sequence[str], **kwargs: Any) -> list[float]:
        count = len(completions)
        classes = _column(kwargs, "class", count)
        infos = _column(kwargs, "rollout_infos", count)
        finish_reasons = _column(kwargs, "finish_reason", count)
        truncated = _column(kwargs, "is_truncated", count)
        gold = _column(kwargs, "gold_utterance", count)

        rewards: list[float] = []
        breakdowns: list[dict[str, Any]] = []

        for index, completion in enumerate(completions):
            info = infos[index]
            if not isinstance(info, Mapping):
                raise RewardContractError("rollout_infos entries must be mappings")
            mask = info.get("environment_loss_mask")
            if mask not in (None, 0):
                raise RewardContractError(
                    "evidence tokens must have a zero loss mask; the trainer reported "
                    f"environment_loss_mask={mask!r}"
                )
            if info.get("tool_ok") is False:
                raise RewardContractError("rollout records a retrieval/gate failure")

            trajectory = Trajectory.parse(
                completion, has_eos=_has_eos(finish_reasons[index], truncated[index])
            )
            breakdown = reward(
                StateClass(str(classes[index])),
                trajectory,
                judge=self._make_judge_fn(index, infos, gold),
            )
            rewards.append(breakdown.reward)
            breakdowns.append(
                {**breakdown.__dict__, "sample_key": info.get("sample_key")}
            )

        self.last_breakdowns = breakdowns
        return rewards

    def _make_judge_fn(self, index: int, infos: Sequence[Any], golds: Sequence[Any]):
        """Judge a correctly timed fire with the same inputs the metric uses.

        The reward is defined in terms of ``Judge(a)``, and the metric's judge
        sees the frames at the moment of the fire. A text-only judge here would
        make the training signal a different quantity from the reported one, so
        missing frames are an error rather than something to work around.
        """
        info = infos[index]

        def judge_fn(answer: str) -> bool:
            frames = _frame_bytes(info)
            clip = _clip_for_judging(info, golds[index])
            fire = Fire(t=float(info.get("t", 0.0)), answer=answer)
            return self.judge(clip, fire, frames).is_good

        return judge_fn


def _frame_bytes(info: Mapping[str, Any]) -> list[bytes]:
    """Pull the rollout's frames out of ``rollout_infos``.

    Accepts raw JPEG bytes or paths; the dataset builder writes paths, and the
    in-memory trainer path hands over bytes.
    """
    images = info.get("images")
    if not images:
        raise RewardContractError(
            "rollout_infos must carry the decision's frames under 'images' so the "
            "reward judge sees what the metric's judge sees"
        )
    frames: list[bytes] = []
    for image in images[-MAX_JUDGE_FRAMES:]:
        if isinstance(image, (bytes, bytearray)):
            frames.append(bytes(image))
        elif isinstance(image, str):
            frames.append(Path(image).read_bytes())
        else:
            raise RewardContractError(f"unsupported image entry: {type(image).__name__}")
    return frames


def _clip_for_judging(info: Mapping[str, Any], gold: Any) -> Clip:
    """A minimal Clip carrying just the gold memory the judge needs."""
    memories = info.get("memory_bundle") or ([{"kind": "", "text": str(gold)}] if gold else [])
    return Clip(
        uid=str(info.get("uid", "")),
        video_path="",
        caption_log="",
        duration=0.0,
        windows=(),
        target="",
        closes_when="",
        targets=(
            Target(
                target="",
                worth_triggering=True,
                open_loop=str(info.get("reminding_reason") or gold or ""),
                memories=tuple(
                    MemoryItem(kind=str(m.get("kind", "")), text=str(m.get("text", "")))
                    for m in memories
                ),
            ),
        ),
    )


def _column(kwargs: Mapping[str, Any], name: str, count: int) -> list[Any]:
    value = kwargs.get(name)
    if value is None:
        return [None] * count
    if len(value) != count:
        raise RewardContractError(f"column {name!r} has {len(value)} rows, expected {count}")
    return list(value)


def _has_eos(finish_reason: Any, truncated: Any) -> bool:
    if truncated:
        return False
    return str(finish_reason or "").lower() in ("stop", "eos", "stop_sequence", "end_turn")


# Registration is a side effect of import, which is how MS-Swift discovers the
# reward. Guarded so re-import under a reloader is not an error.
try:  # pragma: no cover - requires the MS-Swift runtime
    from swift.plugin import orms

    existing = orms.get(REWARD_NAME)
    if existing not in (None, MemoProAgentBinaryReward):
        raise RuntimeError(f"reward name already registered: {REWARD_NAME}")
    orms[REWARD_NAME] = MemoProAgentBinaryReward
except ImportError:
    orms = {REWARD_NAME: MemoProAgentBinaryReward}  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Launcher
# --------------------------------------------------------------------------

DEFAULTS = {
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "dataset": "data/posttrain/train.jsonl",
    "output_dir": "runs/grpo",
    "num_generations": 4,
    "learning_rate": 1e-6,
    "lora_rank": 16,
    "lora_alpha": 16,
    "epochs": 1,
}


def build_command(args: argparse.Namespace) -> list[str]:
    """Hyperparameters are the ones reported in the paper's appendix."""
    return [
        "swift",
        "rlhf",
        "--rlhf_type", "grpo",
        "--model", args.model,
        "--dataset", args.dataset,
        "--output_dir", args.output_dir,
        "--external_plugins", "memoproagent/rl/train_grpo.py",
        "--reward_funcs", REWARD_NAME,
        # G = 4 rollouts per state, group-normalized advantage.
        "--num_generations", str(args.num_generations),
        "--num_train_epochs", str(args.epochs),
        # LoRA on the language projections only; the vision encoder and aligner
        # stay frozen, so the harness's visual behaviour is unchanged.
        "--train_type", "lora",
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--freeze_vit", "true",
        "--freeze_aligner", "true",
        "--torch_dtype", "bfloat16",
        "--learning_rate", str(args.learning_rate),
        "--lr_scheduler_type", "constant",
        "--warmup_ratio", "0.0",
        "--weight_decay", "0.01",
        "--adam_beta1", "0.9",
        "--adam_beta2", "0.999",
        "--max_grad_norm", "1.0",
        "--epsilon", "0.2",
        "--beta", "0.01",          # KL coefficient
        "--temperature", "1.0",
        "--top_p", "0.95",
        "--max_completion_length", "528",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--save_strategy", "steps",
        "--save_steps", "100",
        "--logging_steps", "1",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch GRPO post-training with MS-Swift.")
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="print the swift invocation instead of running it",
    )
    args = parser.parse_args(argv)

    command = build_command(args)
    if args.print_command:
        print(" ".join(shlex.quote(part) for part in command))
        return 0

    import subprocess

    return subprocess.call(command)


if __name__ == "__main__":
    sys.exit(main())
