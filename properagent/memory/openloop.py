"""Open-loop detection.

An *open loop* is an unresolved intention, constraint, or object state -- the
thing that can still justify an interruption. Routine narration ("I hold the
phone") is not one. This module is the single place that distinction is made:
memory construction uses it to flag facts, and retrieval uses it to rank them.

The patterns are matched on the joined ``"<predicate> <object>"`` string with
word boundaries. Bare substring matching was tried first and produced false
positives on ordinary words ("low" inside "low hum", "out" inside "outside").
"""
from __future__ import annotations

import re

_INTENT = (
    r"needs?|needed|needing|need to|needs to|intends?|intended|intending|plans? to|planned to|"
    r"planning to|should|must|reminds?|reminder|forgot|forgotten|promis\w+|"
    r"wants? to|avoids?|avoided|avoiding|going to|has to|have to|meant to|"
    r"remember to|reminded to|look for|shop for|pick up|order|replace|refill|"
    r"return|put back|bring|take back|set aside|save|keep|reserve|restock|"
    r"clean|wash|wipe|repair|fix|check|verify|inspect|test|charge"
)

_STATE = (
    r"running low|is low|low on|nearly empty|almost empty|empty|out of|ran out|"
    r"last one|last of|broken|damaged|loose|spongy|worn|expired|unsent|dirty|"
    r"stuck|leaking|flinch\w*|winc\w*|sensitive|hurts?|sore|missing|misplaced|"
    r"lost|left behind|not charged|low battery|next time|later|again"
)

# Dietary avoidance is a distinct open-loop family: "allergic to peanuts" states no
# intention and no broken state, but it is exactly the kind of standing constraint
# worth interrupting for when the object is in hand.
_DIET = (
    r"allergic|allerg\w+|intoleran\w+|cannot eat|can'?t eat|cannot have|can'?t have|"
    r"does ?n'?t eat|do ?n'?t eat|not eat|won'?t eat|does ?n'?t contain|contains? no|"
    r"free of|dietary|restriction|lactose|gluten\W?free|too much|watch out for|"
    r"stay away from|no pork|no dairy|no soy|no egg|no nuts?|no gluten"
)

OPEN_LOOP_RE = re.compile(rf"\b(?:{_INTENT}|{_STATE}|{_DIET})\b", re.IGNORECASE)

# Entities too generic to anchor a retrieval hop; they connect everything to
# everything and turn personalized PageRank into uniform PageRank.
HUB_ENTITIES = frozenset({"i", "me", "my", "myself"})
GENERIC_ENTITIES = frozenset(
    {
        "purchase",
        "restock",
        "buy",
        "shopping list",
        "list",
        "storage",
        "cleaning",
        "use",
        "food",
        "water",
        "groceries",
    }
)


def is_open_triple(predicate: str, obj: str) -> bool:
    """Whether a ``(subject, predicate, object)`` triple states an open loop."""
    return bool(OPEN_LOOP_RE.search(f"{predicate} {obj}"))


def is_open_text(text: str) -> bool:
    """Whether a free-text passage or standing rule states an open loop."""
    return bool(OPEN_LOOP_RE.search(text or ""))
