"""Deterministic, rule-based expert policy - the supervisor's equivalent of
worker-prod's rollout_generator.py SmarterTeacher.

Deliberately NOT an LLM judge - worker-prod's own production system uses
plain rule-based logic for this exact role, and that choice (not the
LLM-judge-scoring idea from earlier design discussion) is what this file
follows. Cheaper, deterministic, and reproducible across runs - the same
snapshot always produces the same label.

Rules, in priority order (first match wins):
  1. the most recent tool call's risk tier (chassis/state_vectorizer.py's
     effective_risk_tier(): its recorded risk label, else judged from the
     tool name) is "destructive" -> BLOCK_ACTION (refuse outright);
     "sensitive" -> REQUEST_USER (needs a human sign-off, not inherently
     wrong the way a stray delete is)
  2. the same tool+params has now repeated REPEAT_THRESHOLD+1 times in a
     row -> REPLAN (the loop-killer, mirrors worker-prod's
     RewardFunction.R_STAGNATION)
  3. ERROR_ESCALATION_THRESHOLD consecutive tool-call failures -> REQUEST_USER;
     a single failure alone -> INJECT_CONTEXT (a gentle nudge - flag what
     went wrong without blocking or escalating to a human yet)
  4. step or cost budget down to BUDGET_EXHAUSTION_FRACTION or less
     remaining -> TERMINATE. Worker-prod also gates this on "no aspect
     progress" - this supervisor has no aspect-completion signal yet (see
     data/trajectory_schema.py), so this rule is budget-only for now. Revisit
     once a progress signal exists; terminating purely on low budget can
     cut off a run that was about to succeed.
  5. otherwise -> CONTINUE (the default, low-intervention action - most
     steps in a healthy run should end up here)

Known gap, flagged rather than papered over: this rule set never produces
MODIFY_ACTION or NARROW_TOOLS, so a classifier trained only on this
teacher's labels will never learn to predict them - two of the 8 actions in
the vocabulary are currently unreachable. Both need information
TrajectorySnapshot doesn't carry yet: MODIFY_ACTION needs a *pending,
not-yet-executed* proposed tool call to alter (this snapshot only has
tool_history - calls that already happened), and NARROW_TOOLS needs some
sense of tool-usage breadth (is the agent flailing across many different
tools rather than committing to one approach). Revisit the schema before
treating this teacher as feature-complete.
"""
from __future__ import annotations

from chassis.action_space import ActionSpace
from chassis.state_vectorizer import consecutive_error_count, consecutive_repeat_count, effective_risk_tier

REPEAT_THRESHOLD: int = 2
ERROR_ESCALATION_THRESHOLD: int = 2
BUDGET_EXHAUSTION_FRACTION: float = 0.1


def select_action(snapshot) -> int:
    history = snapshot.tool_history

    if history:
        tier = effective_risk_tier(history[-1])
        if tier == "destructive":
            return ActionSpace.encode("BLOCK_ACTION")
        if tier == "sensitive":
            return ActionSpace.encode("REQUEST_USER")

    if consecutive_repeat_count(history) >= REPEAT_THRESHOLD:
        return ActionSpace.encode("REPLAN")

    if consecutive_error_count(history) >= ERROR_ESCALATION_THRESHOLD:
        return ActionSpace.encode("REQUEST_USER")
    if history and not history[-1].success:
        return ActionSpace.encode("INJECT_CONTEXT")

    step_remaining = (snapshot.max_steps - snapshot.step_number) / max(snapshot.max_steps, 1)
    cost_remaining = (
        (snapshot.cost_budget - snapshot.cost_so_far) / snapshot.cost_budget
        if snapshot.cost_budget > 0 else 1.0
    )
    if step_remaining <= BUDGET_EXHAUSTION_FRACTION or cost_remaining <= BUDGET_EXHAUSTION_FRACTION:
        return ActionSpace.encode("TERMINATE")

    return ActionSpace.encode("CONTINUE")
