# claude-model-router

Reference implementation of cost-aware model routing for the Claude API —
plus a Claude Code hook that stops subagents from silently inheriting your
most expensive model.

A cheap Haiku 4.5 call classifies each prompt's complexity; the prompt then
executes on the cheapest capable tier. Small, tested, and built around the
parts naive routers get wrong.

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

## Usage

```python
import anthropic
from model_router import run, execute

client = anthropic.Anthropic()

result = run(client, "Refactor the auth module across services")
print(result.route.model, result.route.source)  # claude-fable-5 classifier
print(result.served_by)                          # may be opus-4-8 after a fallback
print(result.text)
```

Multi-turn: prompt caches are model-scoped, so route once per conversation
and reuse the route for follow-ups —

```python
route = result.route
result2 = execute(client, route, full_message_history)
```

## Cost model

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

This is a pricing calculation, not a live benchmark — it spends no tokens
and its assumptions are at the top of the script.

## Claude Code hook: stop subagents from burning Fable tokens

Separate deliverable, same philosophy. In Claude Code, **subagents inherit
the session model by default** — run your session on an expensive model and
every spawned agent (including greps and file listings) bills at that rate.

[`hooks/subagent-router.py`](hooks/subagent-router.py) is a PreToolUse hook
that denies any Agent spawn missing an explicit `model` param. The deny
reason carries a routing table, so the session model immediately re-issues
the spawn with the cheapest capable tier. Self-correcting, one file, no
dependencies.

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

## Install

```bash
pip install claude-model-router
```

Or from source:

```bash
git clone https://github.com/junoseong/claude-model-router
cd claude-model-router
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

## Notes

- Fable 5 requires 30-day data retention; zero-data-retention orgs get 400s
  on the high tier — remap `high` to `claude-opus-4-8` in `_TIER_ROUTES`.
- Sonnet 5 has intro pricing ($2/$10 per MTok) through 2026-08-31; the cost
  model uses sticker prices ($3/$15).

MIT licensed.
