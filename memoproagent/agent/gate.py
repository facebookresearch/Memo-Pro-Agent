# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""The retrieval gate.

Retrieval answers "is this memory about the object?". The gate answers the
question retrieval cannot: "does this memory justify interrupting the user right
now?". It is a single LLM call that reads the candidates against the observed
object, drops the ones not worth intervening on, and refines the rest into one
piece of evidence -- or abstains.

Abstention is the common case and the point of the component: when the gate
returns nothing the policy sees ``<evidence>none</evidence>``, which is what
keeps irrelevant memory from inventing a reason to speak.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..backends.base import ChatBackend, GenerationConfig, Message
from ..memory.retrieval import RetrievalConfig, Retriever
from . import prompts


class GateProtocolError(RuntimeError):
    """The gate response did not contain a ``FACT:`` line."""


@dataclass
class GateResult:
    observation: str
    evidence: str = ""
    rendered_memory: str = ""
    gate_input: str = ""
    raw: str = ""
    retrieval: dict[str, Any] | None = None

    @property
    def fired(self) -> bool:
        """Whether the gate produced evidence rather than abstaining."""
        return bool(self.evidence)

    def to_json(self) -> dict[str, Any]:
        return {
            "observation": self.observation,
            "evidence": self.evidence,
            "gate_raw": self.raw,
            "retrieval": self.retrieval,
        }


@dataclass(frozen=True)
class GateConfig:
    max_tokens: int = 160
    temperature: float = 0.0
    strict_protocol: bool = True
    """Raise when the gate response has no ``FACT:`` line. On by default: a gate
    that rambles instead of following the contract is a configuration problem
    (wrong model, wrong chat template), and silently reading it as abstention
    would hide that behind a plausible all-silent result."""

    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    prompt_variant: str = prompts.DEFAULT_GATE_VARIANT

    def __post_init__(self) -> None:
        prompts.get_gate_system(self.prompt_variant)


class Gate:
    """Retrieve for an Observe output and reduce it to one actionable fact."""

    def __init__(
        self,
        retriever: Retriever,
        backend: ChatBackend,
        config: GateConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.backend = backend
        self.config = config or GateConfig()
        self.system_prompt = prompts.get_gate_system(self.config.prompt_variant)

    def __call__(self, observation: str) -> GateResult:
        result = GateResult(observation=observation)

        # "none" is the protocol's way of saying nothing is in focus. There is
        # nothing to key retrieval on, so skip straight to abstention.
        if observation.strip().lower() in ("", "none"):
            return result

        retrieved = self.retriever.retrieve(observation, self.config.retrieval)
        result.retrieval = retrieved.to_json()
        rendered = self.retriever.render_for_gate(retrieved, self.config.retrieval)
        result.rendered_memory = rendered
        if not rendered.strip():
            return result

        result.gate_input = prompts.gate_user(observation, rendered)
        raw = self.backend.chat(
            [
                Message(role="system", text=self.system_prompt),
                Message(role="user", text=result.gate_input),
            ],
            GenerationConfig(
                max_tokens=self.config.max_tokens, temperature=self.config.temperature
            ),
        ).strip()
        result.raw = raw

        if self.config.strict_protocol and not prompts.gate_followed_protocol(raw):
            raise GateProtocolError(
                f"gate did not emit a FACT: line for {observation!r}: {raw[:200]!r}"
            )
        result.evidence = prompts.parse_gate(raw)
        return result
