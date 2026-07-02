"""Live benchmark: real API calls, real token counts, real dollars.

Runs a mixed prompt set through two arms and prices each call from the
usage block the API actually returned — no assumed token counts:

  routed    — classify with Haiku, execute on the tier the router picked
  all-fable — every prompt on claude-fable-5 at xhigh (the "no router" baseline)

THIS SPENDS REAL MONEY. Rough guide at list prices: --tiers trivial,low
is a few cents; the full set (incl. mid/high on Fable, twice) runs a few
dollars, dominated by how long Fable chooses to think.

Usage:

    ANTHROPIC_API_KEY=... python benchmarks/live_bench.py
    python benchmarks/live_bench.py --tiers trivial,low        # cheap smoke run
    python benchmarks/live_bench.py --arm routed               # skip the baseline
    python benchmarks/live_bench.py --out live_results.md      # save the report

Reads ANTHROPIC_API_KEY from the environment, falling back to a .env file
in the repo root.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402

from model_router import RefusalError, Route, classify, execute  # noqa: E402

# $ per MTok (input, output) — list pricing, 2026-06. Keep in sync with cost_model.py.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-fable-5": (10.00, 50.00),
}

BUGGY_SNIPPET = '''
def merge_intervals(intervals):
    intervals.sort(key=lambda iv: iv[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
'''

AUTH_SNIPPET = '''
@app.route("/api/user/<user_id>/settings", methods=["POST"])
def update_settings(user_id):
    token = request.headers.get("Authorization", "").removeprefix("Bearer ")
    payload = jwt.decode(token, options={"verify_signature": False})
    if payload.get("sub"):
        db.settings.update(user_id, request.json)
        return jsonify({"ok": True})
    return jsonify({"error": "unauthorized"}), 401
'''

# (expected tier, prompt). Expected tier is what a human would route it as —
# the classifier is free to disagree; disagreements show up in the report.
PROMPTS = [
    # trivial — extraction / reformatting / lookup
    ("trivial", "Extract every email address from this text, one per line: "
                "Contact sara.kim@acme.io or the ops team (ops@acme.io). "
                "Escalations: j.doe+urgent@partner.co.uk."),
    ("trivial", 'Convert this JSON to CSV with a header row: '
                '[{"name":"Ana","qty":3},{"name":"Ben","qty":7},{"name":"Ch","qty":1}]'),
    ("trivial", "Label the sentiment of this review as positive, negative, or mixed: "
                "'Shipping was fast but the case arrived cracked and support never replied.'"),
    ("trivial", "Reformat as a markdown bullet list: apples;oranges;two dozen eggs;oat milk"),
    ("trivial", "What HTTP status code means 'resource permanently moved'? Answer with the number only."),
    ("trivial", "Title-case this sentence: 'the quiet art of shipping small things every day'"),
    # low — short generation / explanation / single-function code
    ("low", "Write a Python function that deduplicates a list while preserving order. Include a docstring."),
    ("low", "Explain the difference between TCP and UDP in exactly three bullet points."),
    ("low", "Summarize in two sentences: Prompt caches on the Claude API are scoped to a "
            "model. If a router picks a different model on every turn of a conversation, "
            "each turn misses the cache and re-bills the full prefix at uncached input "
            "rates, which can cost more than the router saves."),
    ("low", "Write a regex that matches ISO-8601 dates (YYYY-MM-DD) and reject month 13. Explain briefly."),
    ("low", "Give a git command sequence to undo the last commit but keep its changes staged."),
    ("low", "Write a one-paragraph professional reply declining a meeting and proposing async notes instead."),
    # mid — debugging / security review
    ("mid", f"This function has a bug that produces wrong output on some inputs. Find it, "
            f"show a failing input, and fix it:\n```python{BUGGY_SNIPPET}```"),
    ("mid", f"Security-review this Flask handler. List each vulnerability with severity and a fix:"
            f"\n```python{AUTH_SNIPPET}```"),
    ("mid", "A Python service leaks ~30MB/hour. heapy shows growth in dict objects held by a "
            "module-level lru_cache on a function whose argument is a request-scoped dataclass "
            "(frozen, but holds a bytes payload). Explain the leak mechanism precisely and give "
            "two fixes with trade-offs."),
    # high — architecture / multi-constraint synthesis
    ("high", "Design a migration plan to move a 400k-LOC Django monolith (Postgres, Celery, "
             "12 engineers, 99.9% SLO) to services. The trigger: the reporting workload's "
             "table scans are starving OLTP. Deliver: target architecture, strangler-fig "
             "sequencing with the first three extractions named and justified, data-ownership "
             "strategy during the transition, rollback story per phase, and the top three "
             "risks with mitigations. Be concrete, not generic."),
    ("high", "Two teams propose incompatible event schemas for the same order-lifecycle stream: "
             "Team A wants fat events (full order snapshot per event, easy consumers, 40KB avg), "
             "Team B wants thin events (ids + changed fields, 2KB, consumers must call back). "
             "Throughput is 3k events/s peak; consumers include a fraud service with a 150ms "
             "budget and a nightly warehouse load. Recommend one design (or a hybrid), quantify "
             "the trade-offs, and specify the schema-evolution and backfill policy."),
]

TIER_ORDER = ["trivial", "low", "mid", "high"]


def load_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def call_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    # Price by the model that actually served the call (matters after a fallback).
    price_in, price_out = PRICES.get(model, PRICES["claude-fable-5"])
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def normalize_model(model_id: str) -> str:
    """Map dated API model ids (claude-sonnet-5-20260115) onto price-table keys."""
    for key in PRICES:
        if model_id.startswith(key):
            return key
    return model_id


class ClassifierMeter:
    """Wraps client.messages.create to record real classifier token usage."""

    def __init__(self, client: anthropic.Anthropic) -> None:
        self.calls: list[tuple[int, int]] = []
        self._orig = client.messages.create

        def metered(**kwargs):
            response = self._orig(**kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                self.calls.append((usage.input_tokens, usage.output_tokens))
            return response

        client.messages.create = metered  # type: ignore[method-assign]

    @property
    def cost(self) -> float:
        return sum(call_cost("claude-haiku-4-5", i, o) for i, o in self.calls)

    @property
    def tokens(self) -> tuple[int, int]:
        return (sum(i for i, _ in self.calls), sum(o for _, o in self.calls))


def run_one(client, route: Route, prompt: str) -> dict:
    started = time.monotonic()
    try:
        result = execute(client, route, [{"role": "user", "content": prompt}])
    except RefusalError as exc:
        return {"error": f"refused: {exc.stop_details}", "seconds": time.monotonic() - started}
    served = normalize_model(result.served_by)
    usage = result.usage
    return {
        "served_by": served,
        "tokens_in": usage.input_tokens,
        "tokens_out": usage.output_tokens,
        "cost": call_cost(served, usage.input_tokens, usage.output_tokens),
        "seconds": time.monotonic() - started,
        "chars_out": len(result.text),
    }


def run_arm(client, jobs: list[tuple[str, str, Route]], workers: int) -> list[dict]:
    """jobs: (expected_tier, prompt, route). Returns per-prompt result rows."""
    rows: list[dict | None] = [None] * len(jobs)

    def task(idx: int) -> None:
        expected, prompt, route = jobs[idx]
        row = run_one(client, route, prompt)
        row.update(expected=expected, routed_model=route.model, tier=route.tier,
                   source=route.source, prompt=prompt[:60])
        rows[idx] = row
        status = row.get("error") or f"${row['cost']:.4f} {row['tokens_out']}tok {row['seconds']:.0f}s"
        print(f"  [{expected}->{route.tier}] {route.model}: {status}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(task, range(len(jobs))))
    return [r for r in rows if r is not None]


def report(routed: list[dict], fable: list[dict], classifier: ClassifierMeter | None) -> str:
    lines = ["# Live benchmark results", ""]
    lines.append("Real API calls; costs computed from the `usage` block of each response "
                 "at list pricing. Priced by the model that served the call.")
    lines.append("")

    if routed:
        lines.append("## Routed arm (per prompt)")
        lines.append("")
        lines.append("| Expected | Classified | Model served | In tok | Out tok | Cost | Time |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in routed:
            if "error" in r:
                lines.append(f"| {r['expected']} | {r['tier']} | — | — | — | {r['error']} | {r['seconds']:.0f}s |")
                continue
            flag = "" if r["expected"] == r["tier"] else " ⚠"
            lines.append(
                f"| {r['expected']} | {r['tier']}{flag} | `{r['served_by']}` "
                f"| {r['tokens_in']} | {r['tokens_out']} | ${r['cost']:.4f} | {r['seconds']:.0f}s |")
        lines.append("")

    def arm_total(rows: list[dict]) -> tuple[float, float]:
        ok = [r for r in rows if "error" not in r]
        return sum(r["cost"] for r in ok), sum(r["seconds"] for r in ok)

    lines.append("## Totals")
    lines.append("")
    lines.append("| Arm | Prompts | Cost | Sum of latencies |")
    lines.append("|---|---|---|---|")
    routed_cost = fable_cost = None
    if routed:
        cost, secs = arm_total(routed)
        cls_cost = classifier.cost if classifier else 0.0
        routed_cost = cost + cls_cost
        lines.append(f"| routed (execution) | {len(routed)} | ${cost:.4f} | {secs:.0f}s |")
        if classifier:
            ti, to = classifier.tokens
            lines.append(f"| routed (classifier, {len(classifier.calls)} Haiku calls) "
                         f"| — | ${cls_cost:.4f} ({ti} in / {to} out) | — |")
        lines.append(f"| **routed total** | {len(routed)} | **${routed_cost:.4f}** | |")
    if fable:
        fable_cost, secs = arm_total(fable)
        lines.append(f"| **all-Fable baseline** | {len(fable)} | **${fable_cost:.4f}** | {secs:.0f}s |")
    lines.append("")
    if routed_cost is not None and fable_cost:
        saved = 1 - routed_cost / fable_cost
        lines.append(f"**Routed = {routed_cost / fable_cost:.1%} of the all-Fable bill "
                     f"({saved:.1%} saved) on this prompt mix.**")
        lines.append("")
    lines.append("Cost tells half the story — a router that saves money by giving worse "
                 "answers isn't saving anything. Read the transcripts before trusting the number.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["routed", "fable", "both"], default="both")
    parser.add_argument("--tiers", default="trivial,low,mid,high",
                        help="comma-separated subset of tiers to run")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", help="write the markdown report to this path")
    args = parser.parse_args()

    key = load_key()
    if not key:
        sys.exit("No ANTHROPIC_API_KEY in env or repo-root .env — aborting before spending money.")
    client = anthropic.Anthropic(api_key=key)

    wanted = {t.strip() for t in args.tiers.split(",")}
    prompts = [(t, p) for t, p in PROMPTS if t in wanted]
    if not prompts:
        sys.exit(f"No prompts match tiers {sorted(wanted)}")
    print(f"{len(prompts)} prompts, tiers: {', '.join(t for t in TIER_ORDER if t in wanted)}\n")

    routed_rows: list[dict] = []
    fable_rows: list[dict] = []
    classifier_meter = None

    if args.arm in ("routed", "both"):
        print("== Routed arm ==")
        classifier_meter = ClassifierMeter(client)
        jobs = []
        for expected, prompt in prompts:
            route = classify(client, prompt)
            jobs.append((expected, prompt, route))
        routed_rows = run_arm(client, jobs, args.workers)
        print()

    if args.arm in ("fable", "both"):
        print("== All-Fable baseline ==")
        fable_route = Route(model="claude-fable-5", effort="xhigh", tier="high", source="fixed")
        jobs = [(expected, prompt, fable_route) for expected, prompt in prompts]
        fable_rows = run_arm(client, jobs, args.workers)
        print()

    text = report(routed_rows, fable_rows, classifier_meter)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
