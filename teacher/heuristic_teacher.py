"""Deterministic, rule-based expert policy - the supervisor's equivalent of
worker-prod's rollout_generator.py SmarterTeacher.

STUB (2026-09-24): design sketched, not yet implemented. Deliberately NOT an
LLM judge - worker-prod's own production system uses plain rule-based logic
for this exact role, and that choice (not the LLM-judge-scoring idea from
earlier design discussion) is what this file should follow. Cheaper,
deterministic, and reproducible across runs.

Planned rules, in roughly this priority order (first match wins):
  - same tool + params repeated N times in a row -> BLOCK_ACTION or REPLAN
    (the loop-killer, mirrors worker-prod's RewardFunction.R_STAGNATION)
  - a risky-tool-category bit is set (shell exec, mass delete, network
    write) -> BLOCK_ACTION or REQUEST_USER
  - the last tool call errored -> CONTINUE (let it retry) unless this is
    the Nth consecutive error, then REPLAN or REQUEST_USER
  - step/cost budget nearly exhausted with no aspect progress -> TERMINATE
  - otherwise -> CONTINUE (the default, low-intervention action - mirrors
    worker-prod's own action distribution being dominated by its most
    "let it keep working" primitives, not the terminal ones)

Takes a TrajectorySnapshot (see data/trajectory_schema.py) and returns an
action index (see chassis/action_space.py's ActionSpace), the same
select_action(info) -> int shape SmarterTeacher uses.
"""
from __future__ import annotations


def select_action(snapshot) -> int:
    raise NotImplementedError(
        "heuristic_teacher.select_action: not implemented - see this module's docstring for the planned rules"
    )
