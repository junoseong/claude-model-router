# claude-model-router

**Route every prompt to the cheapest Claude model that can handle it.**

[![PyPI](https://img.shields.io/pypi/v/claude-model-router)](https://pypi.org/project/claude-model-router/)
[![CI](https://github.com/junoseong/claude-model-router/actions/workflows/test.yml/badge.svg)](https://github.com/junoseong/claude-model-router/actions/workflows/test.yml)
[![Python](https://img.shields.io/pypi/pyversions/claude-model-router)](https://pypi.org/project/claude-model-router/)
[![License](https://img.shields.io/github/license/junoseong/claude-model-router)](LICENSE)

[Before / After](#before--after) • [Why routers break](#why-naive-claude-routers-break) • [Install](#install) • [Benchmarks](#benchmarks) • [Claude Code hook](#the-claude-code-hook-your-subagents-are-billing-at-fable-rates)

One cheap Haiku call classifies each prompt's difficulty; execution lands on
the cheapest Claude model that can actually handle it. Ships with a Claude
Code hook that stops subagents from silently billing at your session model's
rate. One dependency, 27 tests, honest about where its numbers come from.

## Before / After

**Naive router (what most examples ship):**

```python
model = "claude-haiku-4-5" if len(prompt) < 200 else "claude-fable-5"
response = client.messages.create(model=model, max_tokens=1024,
                                  thinking={"type": "adaptive"}, messages=msgs)
print(response.content[0].text)
```

Five lines, four production bugs: the `thinking` param 400s on the Haiku
branch, Fable refusals arrive as **HTTP 200** with empty content so
`content[0]` raises IndexError, 1024 `max_tokens` truncates thinking turns,
and prompt length is not difficulty — a two-line prompt can be an
architecture question.

**This router:**

```python
from model_router import run

result = run(client, prompt)           # classified, routed, streamed, fallback-protected
print(result.text)
print(result.served_by)                # actual model (matters after a refusal fallback)
print(result.usage.output_tokens)      # the real bill, not an estimate
```

```
┌────────────────────────────────────────────────┐
│  MIXED-WORKLOAD COST         ▼ 26%             │
│  SUPPORT-BOT-SHAPED COST     ▼ 70%+            │
│  SUBAGENT SPAWN (hooked)     25.3k → 11.4k tok │
│  TESTS                       27 green          │
│  DEPENDENCIES                1 (anthropic)     │
└────────────────────────────────────────────────┘
```

## Why naive Claude routers break

Most routing examples treat model IDs as interchangeable strings. On the
Claude 5 family they are not:

| Trap | What actually happens | Where handled |
|---|---|---|
| Refusals are not exceptions | Fable 5 refusals return **HTTP 200** with `stop_reason: "refusal"` — a try/except fallback never fires | server-side fallback beta + explicit `stop_reason` check ([router.py](model_router/router.py)) |
| `thinking` differs per model | Fable 5: always on, explicit config → 400. Opus 4.8: omitted = **off**. Sonnet 5: omitted = adaptive. Haiku 4.5: adaptive → 400 | per-model request shape in `build_request()` |
| `effort` is not universal | `output_config: {"effort": ...}` → 400 on Haiku 4.5 | effort omitted on the trivial tier |
| `content[0].text` crashes | refusals ship empty `content`; thinking blocks precede text blocks | block-type-filtered extraction |
| Small `max_tokens` + high effort = truncation | thinking tokens spend from `max_tokens`; xhigh turns run minutes | 64K `max_tokens` + streaming everywhere |
| Cache dies on model switch | prompt caches are **model-scoped**; per-turn routing invalidates the cache every turn | route once per conversation, reuse via `execute()` |
| Context windows differ | Haiku 4.5 = 200K; every other tier model = 1M | trivial bumped to low above 180K context tokens |

## Tiers

| Tier | Model | Effort | Notes |
|------|-------|--------|-------|
| trivial | `claude-haiku-4-5` | — | reformatting, extraction, classification |
| low  | `claude-sonnet-5`  | medium | summaries, single-function edits, simple Q&A |
| mid  | `claude-opus-4-8`  | high   | debugging, security review, single-file refactors |
| high | `claude-fable-5`   | xhigh  | multi-file refactors, migrations, architecture |

The keyword-heuristic fallback (used when the classifier call fails) never
assigns trivial — misrouting real work to Haiku costs more in retries than
the tier saves, so only the classifier may pick it.

## Install

```bash
pip install claude-model-router
```

```python
import anthropic
from model_router import run, execute

client = anthropic.Anthropic()

result = run(client, "Refactor the auth module across services")
print(result.route.model, result.route.source)  # claude-fable-5 classifier
```

Multi-turn: prompt caches are model-scoped, so route once per conversation
and reuse the route for follow-ups —

```python
route = result.route
result2 = execute(client, route, full_message_history)
```

From source:

```bash
git clone https://github.com/junoseong/claude-model-router
cd claude-model-router
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

## Benchmarks

Two kinds, clearly labeled. Calculated numbers are calculated; live numbers
come from the `usage` block of real API responses. No third kind.

### Cost model (calculated, spends nothing)

1,000 prompts, mixed workload (30% trivial / 40% low / 20% mid / 10% high),
list pricing, output token counts held equal across models (conservative —
Fable at xhigh thinks longer than smaller models on the same prompt):

| Tier | Prompts | Routed model | Routed cost | All-Fable cost |
|---|---|---|---|---|
| trivial | 300 | `claude-haiku-4-5` | $0.54 | $5.40 |
| low | 400 | `claude-sonnet-5` | $6.60 | $22.00 |
| mid | 200 | `claude-opus-4-8` | $18.50 | $37.00 |
| high | 100 | `claude-fable-5` | $80.00 | $80.00 |
| classifier overhead | 1000 | `claude-haiku-4-5` | $0.72 | — |
| **total** | | | **$106.36** | **$144.40** |

**26% saved** on this mix — and the mix is the whole story: the 10% of
prompts that legitimately need Fable dominate the bill. A support-bot-shaped
workload (mostly trivial/low) saves 70%+. Rerun with your own shape:

```bash
python benchmarks/cost_model.py
```

### Live benchmark (real tokens, real dollars)

[`benchmarks/live_bench.py`](benchmarks/live_bench.py) runs a 17-prompt
mixed workload — real bug hunt, real security review, real architecture
questions — through both arms with **real API calls**, and prices every call
from the `usage` block the API actually returned, by the model that actually
served it (which matters after a fallback). Classifier overhead is metered
from real Haiku usage, not estimated. Misroutes are flagged per prompt so
you audit disagreements instead of trusting one savings number.

```bash
ANTHROPIC_API_KEY=... python benchmarks/live_bench.py --tiers trivial,low   # smoke run, cents
ANTHROPIC_API_KEY=... python benchmarks/live_bench.py --out live_results.md # full run, ~$2-6
```

It refuses to start without a key. This is the same cost-vs-quality
methodology RouterBench and RouteLLM run offline, at a scale one person can
afford to reproduce — read the transcripts before trusting the number,
because a router that saves money by giving worse answers saves nothing.

## The Claude Code hook: your subagents are billing at Fable rates

> **The hook graduated to its own repo:
> [subagent-bouncer](https://github.com/junoseong/subagent-bouncer)** —
> installable as a Claude Code plugin in two commands, with its own tests.
> The copy in [`hooks/`](hooks/) stays for pip users; new development
> happens there.

In Claude Code, **subagents inherit the session model by default** — run
your session on an expensive model and every spawned agent, including greps
and file listings, bills at that rate.

[`hooks/subagent-router.py`](hooks/subagent-router.py) is a PreToolUse hook
that denies any Agent spawn missing an explicit `model` param. The deny
reason carries a routing table, so the session model immediately re-issues
the spawn on the cheapest capable tier. Self-correcting, one file, zero
dependencies.

Measured on a live Claude Code session, same investigation task:

| | Tokens | Tool calls |
|---|---|---|
| No hook — spawn inherits session model | 25.3k | 3 |
| Hook — denied once, re-spawned with explicit cheap model | 11.4k | 1 |

The deny feedback didn't just move the spawn to a cheaper model — the
session model also wrote a tighter prompt on the retry.

Install:

```bash
cp hooks/subagent-router.py ~/.claude/hooks/
```

Then add to `~/.claude/settings.json`:

```json
"hooks": {
  "PreToolUse": [
    {
      "matcher": "Agent",
      "hooks": [
        {
          "type": "command",
          "command": "python3 ~/.claude/hooks/subagent-router.py"
        }
      ]
    }
  ]
}
```

New sessions then enforce: unrouted spawn → denied with policy → re-spawned
with explicit `model` (haiku for greps/locates, sonnet for single-file work
and reviews, opus for multi-file/hard debugging, session model only for
judgment-critical synthesis).

## Notes

- Fable 5 requires 30-day data retention; zero-data-retention orgs get 400s
  on the high tier — remap `high` to `claude-opus-4-8` in `_TIER_ROUTES`.
- Sonnet 5 has intro pricing ($2/$10 per MTok) through 2026-08-31; the cost
  model uses sticker prices ($3/$15).

---

If this saved you money, a ⭐ helps others find it.

MIT licensed.
