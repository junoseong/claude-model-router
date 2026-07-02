"""Routed execution against the Messages API.

Handles the per-model API differences a naive router misses:

- Fable 5: thinking is always on (the param must be omitted), and
  refusals arrive as HTTP 200 with stop_reason "refusal" — so requests
  opt into the server-side fallback beta, which re-serves a declined
  request on Opus 4.8 inside the same call.
- Opus 4.8: thinking is OFF when omitted, so it is requested
  explicitly as adaptive.
- Sonnet 5: adaptive thinking is already the default when omitted;
  sending it explicitly is harmless and keeps the code uniform.
- All calls stream: xhigh turns can run minutes, and the SDK rejects
  non-streaming requests at this max_tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic

from .classifier import Route, classify

MAX_TOKENS = 64_000
FALLBACK_BETA = "server-side-fallback-2026-06-01"
FABLE_FALLBACK_MODEL = "claude-opus-4-8"


class RefusalError(RuntimeError):
    """Raised when a request is declined and no fallback rescued it."""

    def __init__(self, stop_details: Any) -> None:
        super().__init__(f"Request refused: {stop_details}")
        self.stop_details = stop_details


@dataclass(frozen=True)
class RoutedResult:
    text: str
    route: Route
    served_by: str  # actual model that answered; differs from route.model after a fallback
    stop_reason: str | None


def build_request(route: Route, messages: list[dict]) -> dict:
    kwargs: dict = {
        "model": route.model,
        "max_tokens": MAX_TOKENS,
        "messages": messages,
    }
    if route.effort is not None:
        kwargs["output_config"] = {"effort": route.effort}
    if route.model == "claude-fable-5":
        kwargs["betas"] = [FALLBACK_BETA]
        kwargs["fallbacks"] = [{"model": FABLE_FALLBACK_MODEL}]
    elif route.model.startswith("claude-haiku"):
        # Haiku 4.5 predates adaptive thinking and effort; sending either
        # returns a 400. Thinking defaults off, which is right for trivial work.
        pass
    else:
        kwargs["thinking"] = {"type": "adaptive"}
    return kwargs


def execute(
    client: anthropic.Anthropic, route: Route, messages: list[dict]
) -> RoutedResult:
    kwargs = build_request(route, messages)
    api = client.beta.messages if "betas" in kwargs else client.messages
    with api.stream(**kwargs) as stream:
        response = stream.get_final_message()
    if response.stop_reason == "refusal":
        raise RefusalError(getattr(response, "stop_details", None))
    text = "".join(b.text for b in response.content if b.type == "text")
    return RoutedResult(
        text=text,
        route=route,
        served_by=response.model,
        stop_reason=response.stop_reason,
    )


def run(
    client: anthropic.Anthropic, prompt: str, context_tokens: int = 0
) -> RoutedResult:
    """Classify a prompt and execute it on the cheapest capable model.

    Route once per conversation, not per turn: prompt caches are
    model-scoped, so reuse the returned RoutedResult.route (via
    execute()) for follow-up turns on the same conversation.
    """
    route = classify(client, prompt, context_tokens)
    return execute(client, route, [{"role": "user", "content": prompt}])
