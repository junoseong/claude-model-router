# model-router

Cost-aware model routing for the Claude API. A cheap Haiku 4.5 call
classifies prompt complexity (keyword/size heuristic as fallback), then
the prompt executes on the cheapest capable tier:

| Tier | Model | Effort | Notes |
|------|-------|--------|-------|
| trivial | `claude-haiku-4-5` | — | no thinking/effort params (both 400 on Haiku 4.5); bumped to low when context > 180K (200K window) |
| low  | `claude-sonnet-5`  | medium | adaptive thinking |
| mid  | `claude-opus-4-8`  | high   | adaptive thinking (explicit — omitted = off) |
| high | `claude-fable-5`   | xhigh  | thinking always on; server-side fallback to Opus 4.8 on refusal |

The keyword-heuristic fallback never assigns trivial — misrouting real
work to Haiku costs more in retries than the tier saves, so only the
classifier may pick it.

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

Multi-turn: prompt caches are model-scoped, so route once per
conversation and reuse the route for follow-ups —

```python
route = result.route
result2 = execute(client, route, full_message_history)
```

## Design notes

- All calls stream (`.stream()` + `get_final_message()`); xhigh turns
  can run minutes and the SDK rejects large non-streaming requests.
- Fable refusals are HTTP 200 + `stop_reason: "refusal"`, not
  exceptions — handled via the `server-side-fallback-2026-06-01` beta,
  with `RefusalError` raised if nothing rescued the request.
- Fable requires 30-day data retention; ZDR orgs will get 400s on the
  high tier — remap `high` to Opus 4.8 in `_TIER_ROUTES` if that's you.

## Tests

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
```
