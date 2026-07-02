"""Cost-aware model routing for the Claude API.

Classifies prompt complexity with a cheap Haiku call (keyword heuristic
as fallback) and executes on the cheapest capable model tier:

    trivial -> claude-haiku-4-5 (no thinking; bumped to low if context > 180K)
    low     -> claude-sonnet-5  @ medium effort
    mid     -> claude-opus-4-8  @ high effort (adaptive thinking)
    high    -> claude-fable-5   @ xhigh effort (server-side fallback to Opus 4.8)
"""

from .classifier import Route, Tier, classify, heuristic_tier
from .router import RefusalError, RoutedResult, build_request, execute, run

__all__ = [
    "Route",
    "Tier",
    "classify",
    "heuristic_tier",
    "RefusalError",
    "RoutedResult",
    "build_request",
    "execute",
    "run",
]
