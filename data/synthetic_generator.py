"""Fabricates plausible TrajectorySnapshot sequences and labels them with
the heuristic teacher - the supervisor's equivalent of worker-prod's
rollout_generator.py run_expert_generation(), which produced 1500 simulated
episodes entirely offline, at zero live cost, before any real orchestrator
traffic existed.

This is the actual starting point for training data, since replay_loader.py
is blocked on OpenClaw Pass 1 (observe-only) logging existing for real.

Five episode categories, chosen with equal probability per episode so every
teacher rule (see teacher/heuristic_teacher.py) gets meaningfully
represented in the resulting dataset - not weighted to mimic a "realistic"
CONTINUE-dominated distribution, since a classifier trained on a heavily
imbalanced corpus could get away with never learning the rare-but-important
interventions:
  - "normal": varied, unrelated, successful tool calls - mostly CONTINUE
  - "stuck_loop": the same tool+params repeated 4 times in a row somewhere
    in the episode - exercises REPLAN
  - "error_recovery": 3 consecutive failures somewhere in the episode
    (different params each, so this doesn't also trigger repeat-detection) -
    exercises INJECT_CONTEXT-then-REQUEST_USER as failures accumulate
  - "risky_tool": one destructive or sensitive call somewhere in the episode
    - exercises BLOCK_ACTION / REQUEST_USER
  - "budget_exhaustion": either the step budget or the cost budget (chosen
    per episode) is sized to run out right at the end - exercises TERMINATE

Every episode draws its own length, trigger position, cost-per-step, and
budget sizes. An earlier version fixed all of these (20 steps, trigger always
at the end, cost always hitting the budget exactly at step 20), which caused
two real problems found in code review (2026-09-25): every category - not
just budget_exhaustion - ended with 3 TERMINATE labels, since the cost budget
always ran out, so the model learned "late in the episode => TERMINATE"; and
every category's trigger was always paired with the same position/budget/
history state, so the dataset couldn't distinguish "learned the causal
feature" from "learned the position." Non-budget categories now keep at least
half their cost budget and two thirds of their step budget, so only
budget_exhaustion episodes ever reach TERMINATE.
"""
from __future__ import annotations

import random

import numpy as np

from chassis.action_space import ActionSpace
from chassis.state_vectorizer import vectorize
from data.trajectory_schema import ToolCallRecord, TrajectorySnapshot
from teacher.heuristic_teacher import select_action

_SAFE_TOOLS: tuple[str, ...] = (
    "web_search", "read_file", "list_directory", "fetch_url", "check_status", "query_read_only",
)
_DESTRUCTIVE_TOOLS: tuple[str, ...] = ("delete_file", "drop_table", "shell_exec", "purge_cache")
_SENSITIVE_TOOLS: tuple[str, ...] = ("send_email", "publish_post", "transfer_funds", "deploy_release")

_MIN_EPISODE_LENGTH: int = 12
_MAX_EPISODE_LENGTH: int = 30
_STUCK_RUN: int = 4
_ERROR_RUN: int = 3
_CATEGORIES: tuple[str, ...] = ("normal", "stuck_loop", "error_recovery", "risky_tool", "budget_exhaustion")


def _random_call(rng: random.Random, *, success: bool = True, risky_pool: tuple[str, ...] | None = None):
    tool_name = rng.choice(risky_pool) if risky_pool else rng.choice(_SAFE_TOOLS)
    params = {"arg": rng.randint(0, 10_000)}
    return ToolCallRecord(tool_name=tool_name, params=params, success=success)


def _build_episode_history(category: str, rng: random.Random, length: int) -> list[ToolCallRecord]:
    history = [_random_call(rng) for _ in range(length)]
    if category == "stuck_loop":
        start = rng.randint(0, length - _STUCK_RUN)
        stuck = _random_call(rng)
        for i in range(start, start + _STUCK_RUN):
            history[i] = ToolCallRecord(tool_name=stuck.tool_name, params=stuck.params, success=True)
    elif category == "error_recovery":
        start = rng.randint(0, length - _ERROR_RUN)
        for i in range(start, start + _ERROR_RUN):
            history[i] = _random_call(rng, success=False)
    elif category == "risky_tool":
        history[rng.randrange(length)] = _random_call(rng, risky_pool=_DESTRUCTIVE_TOOLS + _SENSITIVE_TOOLS)
    return history


def _episode_budgets(category: str, rng: random.Random, length: int) -> tuple[int, float, float]:
    """Returns (max_steps, cost_per_step, cost_budget). Only budget_exhaustion
    episodes are sized to run out; everything else ends with at least 2/3 of
    its step budget and 1/2 of its cost budget left, comfortably clear of the
    teacher's TERMINATE threshold."""
    cost_per_step = rng.uniform(0.1, 1.0)
    generous_cost = cost_per_step * length * rng.uniform(2.0, 5.0)
    generous_steps = length * rng.randint(3, 6)
    if category != "budget_exhaustion":
        return generous_steps, cost_per_step, generous_cost
    if rng.random() < 0.5:
        return length, cost_per_step, generous_cost                       # step budget runs out
    return generous_steps, cost_per_step, cost_per_step * length * rng.uniform(0.85, 1.0)  # cost runs out


def episode_snapshots(category: str, rng: random.Random) -> list[tuple[TrajectorySnapshot, int]]:
    """One episode of the given category -> per-step (snapshot, teacher action index).
    Kept separate from generate_episode so export/export_vectorizer_cases.py
    can ship the raw snapshots to xybernetex-cfworker's parity test."""
    length = rng.randint(_MIN_EPISODE_LENGTH, _MAX_EPISODE_LENGTH)
    max_steps, cost_per_step, cost_budget = _episode_budgets(category, rng, length)
    history = _build_episode_history(category, rng, length)

    steps: list[tuple[TrajectorySnapshot, int]] = []
    recent_action_names: list[str] = []
    for step in range(1, length + 1):
        snapshot = TrajectorySnapshot(
            run_id=f"synthetic-{category}",
            objective="synthetic training objective",
            step_number=step,
            max_steps=max_steps,
            recent_actions=list(recent_action_names),
            tool_history=history[:step],
            cost_so_far=min(step * cost_per_step, cost_budget),
            cost_budget=cost_budget,
        )
        action_index = select_action(snapshot)
        steps.append((snapshot, action_index))
        recent_action_names.append(ActionSpace.decode(action_index))
    return steps


def generate_episode(category: str, rng: random.Random) -> tuple[list[np.ndarray], list[int]]:
    """One episode of the given category -> (per-step state vectors, per-step teacher action indices)."""
    steps = episode_snapshots(category, rng)
    return [vectorize(snapshot) for snapshot, _ in steps], [action for _, action in steps]


def generate_episodes(num_episodes: int = 1500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Returns (states, actions): states is (N, STATE_DIM) float32, actions
    is (N,) int64 - one row per step across every generated episode, ready
    for train/bc_trainer.py's cross-entropy training."""
    rng = random.Random(seed)
    states: list[np.ndarray] = []
    actions: list[int] = []
    for _ in range(num_episodes):
        episode_states, episode_actions = generate_episode(rng.choice(_CATEGORIES), rng)
        states.extend(episode_states)
        actions.extend(episode_actions)
    return np.stack(states).astype(np.float32), np.array(actions, dtype=np.int64)


def generate_and_save(num_episodes: int = 1500, output_path: str = "data/synthetic_rollouts.npz",
                       seed: int = 42) -> str:
    states, actions = generate_episodes(num_episodes, seed=seed)
    np.savez(output_path, states=states, actions=actions)
    return output_path


if __name__ == "__main__":
    path = generate_and_save()
    print(f"Wrote synthetic rollouts to {path}")
