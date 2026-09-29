"""Counterfactual data for learning when a follow-up turn pays off.

In our own harness a follow-up's effect needn't be inferred from different
runs: grade the run folder when the first turn ends (the "no follow-up"
outcome), send the follow-up, and grade again (the "follow-up" outcome).
Same run, both outcomes. This module holds the two pieces that needs from a
job's transcript:

  split_turns   the first run's messages vs the follow-up's (split at the
                second user message)
  features      what a plugin could compute when the first run ends - counts
                and labels only, never text - so anything learned from them
                can later run inside the plugin (xybernetex-openclaw) at
                decision time
"""
from __future__ import annotations

import json
import re

WRITE_TOOLS = {"write", "edit", "multi_edit", "multiedit", "apply_patch", "create_file", "str_replace"}
EXEC_TOOLS = {"exec", "bash", "shell", "terminal", "process"}
# Signals in the model's final answer, reduced to booleans here and (later)
# in the plugin: does it claim to have tested/checked, does it hedge.
CLAIMS_CHECKED = re.compile(r"\b(tests? (pass|passed|passing)|all \d+ (tests|checks)|verified|i (ran|checked|tested)|output matches)\b", re.I)
HEDGES = re.compile(r"\b(should (work|be)|might|may need|probably|i assume|not sure|couldn'?t|could not|unable to|todo)\b", re.I)

FEATURE_NAMES = ["tool_calls", "failed_calls", "fail_rate", "writes", "execs", "max_repeat", "tokens_k", "assistant_msgs",
                 "minutes", "reply_chars_k", "claims_checked", "hedges", "ended_stop", "ended_length", "empty_reply",
                 "last_call_failed"]


def split_turns(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(users) < 2:
        return messages, []
    return messages[:users[1]], messages[users[1]:]


def tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        u = m.get("usage") if m.get("role") == "assistant" else None
        if isinstance(u, dict):
            total += u.get("totalTokens") or (u.get("input") or 0) + (u.get("output") or 0)
    return int(total)


def _text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    return "".join(x.get("text") or "" for x in c if isinstance(x, dict) and x.get("type") == "text") if isinstance(c, list) else ""


def features(first: list[dict]) -> dict:
    assistant = [m for m in first if m.get("role") == "assistant"]
    calls = [c for m in assistant for c in (m.get("content") if isinstance(m.get("content"), list) else [])
             if isinstance(c, dict) and c.get("type") == "toolCall"]
    results = [m for m in first if m.get("role") == "toolResult"]
    failed = sum(1 for m in results if m.get("isError") is True)
    names = [str(c.get("name") or "").lower() for c in calls]
    keys = [(n, json.dumps(c.get("arguments"), sort_keys=True, default=str)) for n, c in zip(names, calls)]
    streak = best = 0
    for i, k in enumerate(keys):
        streak = streak + 1 if i and k == keys[i - 1] else 1
        best = max(best, streak)
    stamps = [m.get("timestamp") for m in first if isinstance(m.get("timestamp"), (int, float))]
    last = assistant[-1] if assistant else {}
    reply = _text(last)
    return {
        "tool_calls": len(calls), "failed_calls": failed, "fail_rate": failed / len(results) if results else 0.0,
        "writes": sum(n in WRITE_TOOLS for n in names), "execs": sum(n in EXEC_TOOLS for n in names), "max_repeat": best,
        "tokens_k": tokens(first) / 1000, "assistant_msgs": len(assistant),
        "minutes": (max(stamps) - min(stamps)) / 60_000 if len(stamps) > 1 else 0.0,
        "reply_chars_k": len(reply) / 1000, "claims_checked": int(bool(CLAIMS_CHECKED.search(reply))),
        "hedges": int(bool(HEDGES.search(reply))), "ended_stop": int(last.get("stopReason") == "stop"),
        "ended_length": int(last.get("stopReason") == "length"), "empty_reply": int(not reply.strip() and not calls),
        "last_call_failed": int(bool(results) and results[-1].get("isError") is True),
    }
