"""Prompt-complexity classification for model routing.

Two stages: a cheap Haiku 4.5 call for semantic tiering, with a
keyword/size heuristic as fallback when the API call fails or returns
something unparseable. The classifier call costs a few hundred tokens
at $1/$5 per MTok — noise next to the downstream call it's routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import anthropic

Tier = Literal["low", "mid", "high"]

CLASSIFIER_MODEL = "claude-haiku-4-5"

# Complexity is judged from the head of the prompt plus size metadata;
# feeding the whole prompt to the classifier would defeat its purpose.
_CLASSIFY_SNIPPET_CHARS = 2000

_CLASSIFIER_SYSTEM = """\
You route coding prompts to one of three capability tiers. Reply with exactly one word.

low  - lookups, summaries, single-function edits, boilerplate, simple Q&A
mid  - debugging, security review, tricky edge cases, single-file refactors
high - multi-file refactors, migrations, architecture design, codebase-wide optimization

Reply with: low, mid, or high. Nothing else."""

_HIGH_KEYWORDS = (
    "migration",
    "architecture",
    "multi-file",
    "optimize codebase",
    "refactor across",
)
_MID_KEYWORDS = (
    "debug",
    "security audit",
    "edge case",
    "race condition",
    "fuzzing",
    "regex",
)

_TIER_ROUTES: dict[Tier, tuple[str, str]] = {
    "low": ("claude-sonnet-5", "medium"),
    "mid": ("claude-opus-4-8", "high"),
    "high": ("claude-fable-5", "xhigh"),
}


@dataclass(frozen=True)
class Route:
    model: str
    effort: str
    tier: Tier
    source: str  # "classifier" | "heuristic"


def heuristic_tier(prompt: str, context_tokens: int = 0) -> Tier:
    """Keyword/size fallback used when the classifier call fails."""
    p = prompt.lower()
    if any(k in p for k in _HIGH_KEYWORDS) or context_tokens > 400_000:
        return "high"
    if any(k in p for k in _MID_KEYWORDS) or context_tokens > 100_000:
        return "mid"
    return "low"


def classify(
    client: anthropic.Anthropic, prompt: str, context_tokens: int = 0
) -> Route:
    """Classify a prompt into a route; never raises on classifier failure."""
    tier: Tier | None = None
    source = "classifier"
    try:
        response = client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=8,
            system=_CLASSIFIER_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"context_tokens={context_tokens}\n---\n"
                        f"{prompt[:_CLASSIFY_SNIPPET_CHARS]}"
                    ),
                }
            ],
        )
        text = (
            next((b.text for b in response.content if b.type == "text"), "")
            .strip()
            .lower()
        )
        if text in _TIER_ROUTES:
            tier = text  # type: ignore[assignment]
    except anthropic.AnthropicError:
        pass
    if tier is None:
        tier = heuristic_tier(prompt, context_tokens)
        source = "heuristic"
    model, effort = _TIER_ROUTES[tier]
    return Route(model=model, effort=effort, tier=tier, source=source)
