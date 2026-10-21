# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Protocol prompts and explicitly selectable gate variants.

Three strings drive the whole harness:

    GATE_SYSTEM              the shared retrieval gate (identical in both paradigms)
    WINDOWED_POLICY_SYSTEM   the policy under the sliding-window paradigm
    STREAMING_POLICY_SYSTEM  the policy under the streaming paradigm

The two policy prompts differ only in how they refer to the visual input: the
windowed one says "the frames", the streaming one says "the CURRENT/latest
frame". Everything else is byte-identical.

Do not reformat these strings. Whitespace and wording are part of the measured
configuration; a cosmetic edit changes results.

The default gate is VERSION_FREE, which replaces only the legacy v33/v45
comparison sentence in V51. The original V51 and the paper-displayed gate remain
explicit variants for reproducing their measured results.
"""
from __future__ import annotations

import re

NO_REMINDER = "NO_REMINDER"

# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------

_GATE_TRAILER = (
    "Output EXACTLY one line, either:\n"
    "FACT: <the specific fact + when>\n"
    "FACT: none"
)

_GATE_BODY = (
    "You are the memory gate of a proactive assistant. You are given the object(s) the user is engaging with right "
    "now and their retrieved memory: first-person event PASSAGES, [standing] rules, and episodic triples.\n"
    "High-recall rule: if ANY retrieved memory contains an unresolved open-loop, preference, precaution, prior "
    "mistake, discomfort, missing/low item, or next-time rule about the named object, you MUST output it as FACT. "
    "Do not require the reminder to feel important. Do not reject because the fact is phrased as narrative history "
    "rather than a clean task sentence.\n"
    "Translate prior mishaps into the still-open rule they imply for the same object: sore wrists with flat-palm "
    "push-ups -> use dumbbells as handles; mismatched shoes after checking only one -> check both shoes; a stretched "
    "shirt neckline from sunglasses -> remove sunglasses first; running out of taco shells -> keep/restock a spare.\n"
    "Output none ONLY when there is truly no open issue, preference, check, safety/use constraint, or next-time rule "
    "about the named object, or the only open loop is clearly about a different object.\n"
    "Rules: stay grounded in the memory; if a later memory clearly resolves the issue, output none; otherwise treat "
    "the open-loop as still relevant.\n"
    "Middle-ground willingness: be more willing than v33 only for OBJECT-MATCHED open loops, but do not use v45's "
    "broad narrative rule for unrelated spillover. First identify the current object family from Object(s), "
    "including direct context bridges: mannequin/rack/display -> garment on it; package/bag/box -> named product; "
    "fabric/scissors/needle/pin/template -> sewing/craft step; dog/leash/toy/grass -> dog handling or fetch rule; "
    "printer/tray/document/paper -> printed pages; backpack/locker/shoe -> climbing-shoe/key context; statue/"
    "plaque/path -> sculpture/photo intention.\n"
    "If a retrieved passage or fact contains an unresolved need/intention/check/avoid/safety/next-time rule within "
    "that object family, output it even if it is buried or phrased narratively. If the candidate is for a different "
    "object family, routine handling, or clearly resolved, output none. This variant should spend FR headroom "
    "only on matched-object recoveries.\n"
)

GATE_SYSTEM_V51 = _GATE_BODY + _GATE_TRAILER

# Replace only the legacy comparison sentence; preserve every other V51 byte.
GATE_SYSTEM_VERSION_FREE = GATE_SYSTEM_V51.replace(
    "Middle-ground willingness: be more willing than v33 only for OBJECT-MATCHED open loops, "
    "but do not use v45's broad narrative rule for unrelated spillover.",
    "Middle-ground willingness: be willing to surface OBJECT-MATCHED open loops, "
    "but do not use broad narrative rules for unrelated spillover.",
)

GATE_SYSTEM = GATE_SYSTEM_VERSION_FREE
DEFAULT_GATE_VARIANT = "version_free"

# Rendered System text from the manuscript's shared gate table.
GATE_SYSTEM_PAPER = (
    "You are the memory gate of a proactive assistant. You are given the object(s) the user is engaging with right "
    "now and their retrieved memory: first-person event PASSAGES, [standing] rules, and episodic triples.\n"
    "High-recall rule: if ANY retrieved memory contains an unresolved open-loop, preference, precaution, prior "
    "mistake, discomfort, missing/low item, or next-time rule about the named object, you MUST output it as FACT. "
    "Do not reject because the fact is phrased as narrative history rather than a clean task sentence.\n"
    "Translate prior mishaps into the still-open rule they imply for the same object.\n"
    "Output none ONLY when there is truly no open issue, preference, check, safety/use constraint, or next-time rule "
    "about the named object, or the only open loop is clearly about a different object.\n"
    "Rules: stay grounded in the memory; if a later memory clearly resolves the issue, output none; otherwise treat "
    "the open-loop as still relevant.\n"
    "Be willing to surface an open-loop, but ONLY when it belongs to the SAME object family the user is currently "
    "engaging with; do not spill over to open-loops about unrelated objects just because they appear in the narrative.\n"
) + _GATE_TRAILER

GATE_SYSTEM_VARIANTS = {
    "v51": GATE_SYSTEM_V51,
    "version_free": GATE_SYSTEM_VERSION_FREE,
    "paper": GATE_SYSTEM_PAPER,
}


def get_gate_system(variant: str = DEFAULT_GATE_VARIANT) -> str:
    """Select a gate prompt explicitly; never silently fall back on a typo."""
    try:
        return GATE_SYSTEM_VARIANTS[variant]
    except (KeyError, TypeError) as error:
        raise ValueError(f"unknown gate prompt variant {variant!r}") from error


GATE_USER_TEMPLATE = "Object(s): {observation}\nRetrieved memory:\n{memory}"


def gate_user(observation: str, memory: str) -> str:
    return GATE_USER_TEMPLATE.format(observation=observation, memory=memory)


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------

_POLICY_TEMPLATE = (
    "You are a proactive wearable assistant watching the user's first-person camera. You follow this exact "
    "protocol in order:\n"
    "<observe> the concrete object(s) the user is actively engaging with {observe_scope} (looking at/reaching for/"
    "holding/comparing/using) as short noun phrases, read brand/label, ONE or a FEW comma-separated; or none "
    "</observe>\n"
    "<evidence> refined memory about those object(s) is injected here by the retrieval system (may be empty) "
    "</evidence>\n"
    "<reason> ONE short sentence judging: is an object from EVIDENCE present {present_scope} AND does EVIDENCE state a "
    "specific open loop/constraint actionable at THIS moment AND not already spoken? </reason>\n"
    "<answer> ONE grounded reminder sentence, or NO_REMINDER </answer>\n"
    "Default to NO_REMINDER - for almost every moment that is correct. Speak ONLY IF ALL hold: (1) the object "
    "in EVIDENCE is present in {visual_scope}, (2) EVIDENCE states a specific open loop/constraint actionable "
    "at THIS moment, (3) not already said, and (4) if 'Already spoken' already covers this same need or item, "
    "output NO_REMINDER regardless of which objects are now in view or how you would word it. The reminder MUST "
    "restate the specific fact from EVIDENCE (grounded); NEVER give generic suggestions; do not fire from "
    "topical relatedness. If EVIDENCE is empty or not genuinely actionable now -> NO_REMINDER."
)

WINDOWED_POLICY_SYSTEM = _POLICY_TEMPLATE.format(
    observe_scope="right now",
    present_scope="now",
    visual_scope="the frames now",
)

STREAMING_POLICY_SYSTEM = _POLICY_TEMPLATE.format(
    observe_scope="in the CURRENT/latest frame",
    present_scope="in the CURRENT/latest frame",
    visual_scope="the CURRENT/latest frame now",
)

SECOND_TURN_INSTRUCTION = (
    "Continue the protocol. Respond EXACTLY:\n"
    "<reason> one short sentence </reason>\n"
    "<answer> ONE grounded reminder sentence, or NO_REMINDER </answer>"
)


def observe_user(timestamp: float, *, streaming: bool) -> str:
    """Round-one request. Carries no reminder history by design: the Observe step
    must describe the scene, not be steered by what has already been said."""
    if streaming:
        return f"(~{timestamp:.0f}s) CURRENT/latest frame. Begin the protocol with <observe>."
    return f"(~{timestamp:.0f}s) Begin the protocol with <observe>."


def decide_user(
    timestamp: float, already_spoken: str, evidence: str, *, streaming: bool
) -> str:
    """Round-two request, carrying the gated evidence and the fire history."""
    header = (
        f"(~{timestamp:.0f}s) CURRENT/latest frame.\n" if streaming else f"(~{timestamp:.0f}s)\n"
    )
    return (
        f"{header}"
        f"Already spoken: {sanitize_injected(already_spoken)}\n"
        f"<evidence>{sanitize_injected(evidence)}</evidence>\n"
        f"{SECOND_TURN_INSTRUCTION}"
    )


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_TAG_RE = re.compile(
    r"</?(?:observe|evidence|reason|answer)>|<\|(?:im_start|im_end|endoftext)\|>",
    re.IGNORECASE,
)
_OBSERVE_RE = re.compile(r"<observe>\s*(.*?)(?:</observe>|\Z)", re.DOTALL | re.IGNORECASE)
_OBSERVE_CLOSE_RE = re.compile(r"</observe>", re.IGNORECASE)
_REASON_RE = re.compile(r"<reason>\s*(.*?)\s*(?:</reason>|$)", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_FACT_LINE_RE = re.compile(r"(?im)^\s*FACT\s*:")


def sanitize_injected(value: str, fallback: str = "none") -> str:
    """Strip protocol tags from environment-authored text.

    Retrieved memory is model-generated and can contain a literal ``</evidence>``.
    Injecting it unescaped would let the environment close the span early and write
    into the policy's own region of the prompt.
    """
    cleaned = _TAG_RE.sub(" ", str(value or "").strip())
    return " ".join(cleaned.split()) or fallback


def trim_observe(raw: str) -> str:
    """Keep the sampled Observe span through its first real closing tag."""
    text = str(raw or "")
    match = _OBSERVE_CLOSE_RE.search(text)
    return text[: match.end()] if match else text


def parse_observe(raw: str) -> tuple[str, bool]:
    """Return ``(object_phrase, closed)``. `closed` is False when the model never
    emitted ``</observe>``, which marks the trajectory malformed."""
    text = str(raw or "")
    match = _OBSERVE_RE.search(text)
    if not match:
        return "", False
    observation = " ".join(match.group(1).split())
    return observation, bool(_OBSERVE_CLOSE_RE.search(text))


def parse_decision(raw: str) -> tuple[str, str, bool]:
    """Return ``(reason, answer, answer_closed)``."""
    text = str(raw or "")
    reason_match = _REASON_RE.search(text)
    answer_match = _ANSWER_RE.search(text)
    return (
        reason_match.group(1).strip() if reason_match else "",
        answer_match.group(1).strip() if answer_match else "",
        answer_match is not None,
    )


def is_no_reminder(answer: str) -> bool:
    normalized = answer.strip("\"'` .!?,;:").upper().replace(" ", "_").replace("-", "_")
    return normalized == NO_REMINDER


def is_fire(answer: str, *, observe_closed: bool, answer_closed: bool) -> bool:
    """A fire requires a well-formed two-round trajectory. A truncated generation is
    never counted as silence -- that would reward running out of tokens."""
    return bool(observe_closed and answer_closed and answer and not is_no_reminder(answer))


# --------------------------------------------------------------------------
# Streaming-compatible (OFFLINE_V2_COMPATIBLE) parsing helpers
# --------------------------------------------------------------------------
# These mirror the legacy reference
# ``e2e_inference/protocol_adapter.py`` + ``orchestrator.py`` compatible path
# without changing any prompt string. Reason/answer use the same independent
# regexes as :func:`parse_decision`; a missing ``<reason>`` is a diagnostic
# only, never an extra fire-disqualification condition. Only a final
# ``finish_reason == "length"`` disqualifies an otherwise eligible fire;
# an observe ``length`` only records a degradation flag.

def compatible_observe_reasons(
    *, observe_closed: bool, observe_finish_reason: str | None = None
) -> list[str]:
    """Degradation flags for one Observe output. Exact legacy reason strings."""
    reasons: list[str] = []
    if observe_finish_reason == "length":
        reasons.append("observe_length_truncated")
    if not observe_closed:
        reasons.append("observe_unclosed")
    return reasons


def parse_compatible_final_output(
    raw: str,
    *,
    observe_closed: bool,
    final_finish_reason: str | None = None,
) -> tuple[str, str, bool, bool, bool, list[str]]:
    """Return ``(reason, answer, reason_parsed, answer_parsed, fire_eligible,
    degradation_reasons)`` mirroring the legacy compatible path.

    ``fire_eligible`` requires a closed observe, a parsed nonempty non-NO_REMINDER
    answer, and ``final_finish_reason != "length"``. Unknown/EOS metadata is never
    invented as ``"length"`` or ``"stop"``.
    """
    text = str(raw or "")
    reason_match = _REASON_RE.search(text)
    answer_match = _ANSWER_RE.search(text)
    reason = reason_match.group(1).strip() if reason_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    reason_parsed = reason_match is not None
    answer_parsed = answer_match is not None
    fire_eligible = bool(
        is_fire(answer, observe_closed=observe_closed, answer_closed=answer_parsed)
        and final_finish_reason != "length"
    )
    degradation: list[str] = []
    if final_finish_reason == "length":
        degradation.append("final_length_truncated")
    if not reason_parsed:
        degradation.append("reason_unparsed")
    if not answer_parsed:
        degradation.append("answer_unparsed")
    elif not answer:
        degradation.append("answer_empty")
    return reason, answer, reason_parsed, answer_parsed, fire_eligible, degradation


def parse_gate(raw: str) -> str:
    """Return the gated evidence, or ``""`` when the gate abstains.

    The contract is one ``FACT:`` line. Anything else -- a refusal, a preamble with
    no FACT line, a rambling explanation -- is treated as abstention rather than
    guessed at, so a malformed gate response can never produce a fire.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    fact: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FACT:"):
            fact = stripped[5:].strip()
    if fact is None:
        return ""
    if fact.lower().rstrip(".") in ("none", ""):
        return ""
    return fact


def gate_followed_protocol(raw: str) -> bool:
    return bool(_FACT_LINE_RE.search(raw or ""))
