#!/usr/bin/env python3
"""PreToolUse hook (matcher: Agent): enforce explicit subagent model routing.

An Agent call with no `model` param is denied with the routing policy as
the reason; Claude re-issues the spawn with an explicit model. Calls that
already set a model pass through untouched.
"""

import json
import sys


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return  # malformed input: never block the tool call
    if data.get("tool_name") != "Agent":
        return
    if (data.get("tool_input") or {}).get("model"):
        return

    reason = (
        "Model routing policy: re-issue this Agent call with an explicit `model` "
        "chosen by task difficulty — "
        "haiku: greps, 'where is X', file maps, mechanical renames/edits; "
        "sonnet: single-file builds, standard reviews, routine exploration; "
        "opus: multi-file implementation, hard debugging, adversarial verification; "
        "fable: synthesis or judgment-critical work only. "
        "Pick the cheapest capable tier."
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))


if __name__ == "__main__":
    main()
