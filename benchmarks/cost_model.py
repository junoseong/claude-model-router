"""Cost model: routed workload vs all-Fable, computed from list pricing.

This is a pricing calculation, not a live benchmark — it spends no API
tokens and makes its assumptions explicit so you can rerun with your own
workload shape. Run:

    python benchmarks/cost_model.py
"""

# $ per MTok (input, output) — list pricing, 2026-06.
# Sonnet 5 has intro pricing ($2/$10) through 2026-08-31; sticker used here.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-fable-5": (10.00, 50.00),
}

# Workload shape: (tier, share of prompts, avg input tokens, avg output tokens).
# Output includes thinking tokens. Token counts held equal across models,
# which is conservative — Fable at xhigh thinks longer than Haiku or Sonnet
# on the same prompt, so real all-Fable output cost runs higher than modeled.
WORKLOAD = [
    ("trivial", 0.30, 800, 200),
    ("low", 0.40, 2_000, 700),
    ("mid", 0.20, 6_000, 2_500),
    ("high", 0.10, 20_000, 12_000),
]

TIER_MODEL = {
    "trivial": "claude-haiku-4-5",
    "low": "claude-sonnet-5",
    "mid": "claude-opus-4-8",
    "high": "claude-fable-5",
}

N_PROMPTS = 1_000

# Classifier overhead: one Haiku call per prompt (system prompt + snippet in,
# one word out).
CLASSIFIER_IN, CLASSIFIER_OUT = 700, 4


def call_cost(model: str, tokens_in: float, tokens_out: float) -> float:
    price_in, price_out = PRICES[model]
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def main() -> None:
    rows = []
    routed_total = 0.0
    fable_total = 0.0
    for tier, share, avg_in, avg_out in WORKLOAD:
        n = N_PROMPTS * share
        model = TIER_MODEL[tier]
        routed = n * call_cost(model, avg_in, avg_out)
        fable = n * call_cost("claude-fable-5", avg_in, avg_out)
        routed_total += routed
        fable_total += fable
        rows.append((tier, int(n), model, routed, fable))

    classifier = N_PROMPTS * call_cost("claude-haiku-4-5", CLASSIFIER_IN, CLASSIFIER_OUT)
    routed_total += classifier

    print(f"Workload: {N_PROMPTS} prompts\n")
    print("| Tier | Prompts | Routed model | Routed cost | All-Fable cost |")
    print("|---|---|---|---|---|")
    for tier, n, model, routed, fable in rows:
        print(f"| {tier} | {n} | `{model}` | ${routed:.2f} | ${fable:.2f} |")
    print(f"| classifier overhead | {N_PROMPTS} | `claude-haiku-4-5` | ${classifier:.2f} | — |")
    print(f"| **total** | | | **${routed_total:.2f}** | **${fable_total:.2f}** |")
    savings = 1 - routed_total / fable_total
    print(f"\nRouted = {routed_total / fable_total:.1%} of all-Fable cost ({savings:.1%} saved).")


if __name__ == "__main__":
    main()
