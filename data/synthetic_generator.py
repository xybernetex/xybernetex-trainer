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
  - "stuck_loop": the same tool+params repeated for the last 4 steps -
    exercises REPLAN
  - "error_recovery": the last 3 steps fail (different params each, so this
    doesn't also trigger repeat-detection) - exercises CONTINUE-then-
    REQUEST_USER as failures accumulate
  - "risky_tool": the final step is a destructive or sensitive tool call -
    exercises BLOCK_ACTION / REQUEST_USER
  - "budget_exhaustion": max_steps set equal to the episode length (instead
    of generously larger) so the step budget genuinely runs low near the
    end - exercises TERMINATE
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

_EPISODE_LENGTH: int = 20
_CATEGORIES: tuple[str, ...] = ("normal", "stuck_loop", "error_recovery", "risky_tool", "budget_exhaustion")


def _random_call(rng: random.Random, *, success: bool = True, risky_pool: tuple[str, ...] | None = None):
    tool_name = rng.choice(risky_pool) if risky_pool else rng.choice(_SAFE_TOOLS)
    params = {"arg": rng.randint(0, 10_000)}
    return ToolCallRecord(tool_name=tool_name, params=params, success=success)


def _build_episode_history(category: str, rng: random.Random) -> list[ToolCallRecord]:
    history: list[ToolCallRecord] = []
    stuck_call = None
    for i in range(_EPISODE_LENGTH):
        if category == "stuck_loop" and i >= _EPISODE_LENGTH - 4:
            if stuck_call is None:
                stuck_call = _random_call(rng)
            history.append(ToolCallRecord(tool_name=stuck_call.tool_name, params=stuck_call.params, success=True))
        elif category == "error_recovery" and i >= _EPISODE_LENGTH - 3:
            history.append(_random_call(rng, success=False))
        elif category == "risky_tool" and i == _EPISODE_LENGTH - 1:
            history.append(_random_call(rng, risky_pool=_DESTRUCTIVE_TOOLS + _SENSITIVE_TOOLS))
        else:
            history.append(_random_call(rng))
    return history


def generate_episodes(num_episodes: int = 1500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Returns (states, actions): states is (N, STATE_DIM) float32, actions
    is (N,) int64 - one row per step across every generated episode, ready
    for train/bc_trainer.py's cross-entropy training."""
    rng = random.Random(seed)
    states: list[np.ndarray] = []
    actions: list[int] = []

    for _ in range(num_episodes):
        category = rng.choice(_CATEGORIES)
        max_steps = _EPISODE_LENGTH if category == "budget_exhaustion" else _EPISODE_LENGTH * 5
        cost_budget = 10.0
        history = _build_episode_history(category, rng)

        recent_action_names: list[str] = []
        for step in range(1, len(history) + 1):
            snapshot = TrajectorySnapshot(
                run_id=f"synthetic-{category}",
                objective="synthetic training objective",
                step_number=step,
                max_steps=max_steps,
                recent_actions=list(recent_action_names),
                tool_history=history[:step],
                cost_so_far=min(step * (cost_budget / _EPISODE_LENGTH), cost_budget),
                cost_budget=cost_budget,
            )
            action_index = select_action(snapshot)
            states.append(vectorize(snapshot))
            actions.append(action_index)
            recent_action_names.append(ActionSpace.decode(action_index))

    return np.stack(states).astype(np.float32), np.array(actions, dtype=np.int64)


def generate_and_save(num_episodes: int = 1500, output_path: str = "data/synthetic_rollouts.npz",
                       seed: int = 42) -> str:
    states, actions = generate_episodes(num_episodes, seed=seed)
    np.savez(output_path, states=states, actions=actions)
    return output_path


if __name__ == "__main__":
    path = generate_and_save()
    print(f"Wrote synthetic rollouts to {path}")
