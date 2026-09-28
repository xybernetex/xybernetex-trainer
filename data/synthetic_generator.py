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
  - "error_recovery": 2-7 consecutive failures somewhere in the episode
    (different params each, so this doesn't also trigger repeat-detection) -
    exercises INJECT_CONTEXT-then-REQUEST_USER as failures accumulate, and
    (at 5+) the escalation holding once consecutive_error_count saturates
  - "risky_tool": one to three destructive or sensitive calls somewhere in
    the episode, each with an authorization label (requested / own_artifact
    / unrequested / not assessed) - exercises BLOCK_ACTION / REQUEST_USER
    when the user didn't ask, and CONTINUE when they did. Some risky calls
    fail and are retried, so a requested one that keeps failing escalates
    like any other error streak (INJECT_CONTEXT, then REQUEST_USER)
  - "budget_exhaustion": either the step budget or the cost budget (chosen
    per episode) is sized to run out partway through, with several steps
    still running afterward while it stays exhausted - exercises TERMINATE,
    including with TERMINATE already in the episode's own action history

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

A second real problem, found from live data rather than code review
(2026-09-25, the full-run1 scenario batch plus an offline ablation against
the deployed checkpoint - see xybernetex-openclaw/ALPHA.md and the
architecture doc): both budget_exhaustion and error_recovery episodes ended
at (or immediately after) their own trigger, so the model never trained on
what its OWN recommendation looks like once it's already in
recent_actions - exactly what happens in real observe-only deployment, since
nothing there ever stops a run. The ablation showed this precisely: on an
otherwise-identical real state, P(TERMINATE) jumped from 0.56 to 0.94 the
instant one prior TERMINATE appeared in the 16-slot action history, and
saturated immediately after - a hard switch the model had never been trained
against, not a gradient it learned. budget_exhaustion's trigger now lands
partway through the episode (not at the very end), so several further steps
run on afterward with the budget still exhausted and TERMINATE already
sitting in recent_actions - including step_number exceeding max_steps, which
real OpenClaw runs do (nothing hard-stops a run at its reported budget
either). error_recovery's streak length is now drawn from [2, 7] instead of
fixed at 3, so training actually reaches consecutive_error_count's
normalization cap (5) - real streaks of 6-8 had the model confidently
predicting the wrong action, worse the longer the streak ran, because it had
only ever seen exactly 3.
"""
from __future__ import annotations

import random

import numpy as np

from chassis.action_space import ActionSpace
from chassis.state_vectorizer import vectorize_v3
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
# Was fixed at 3: state_vectorizer.py's consecutive_error_count feature
# normalizes by _MAX_CONSECUTIVE_ERRORS=5, so a streak of exactly 3 (0.6)
# never reached the saturated region (streaks >=5, all 1.0) that real long
# streaks land in - see the module docstring's 2026-09-25 note.
_ERROR_RUN_MIN: int = 2
_ERROR_RUN_MAX: int = 7
# Floor on how many steps a budget_exhaustion episode keeps running once its
# budget first hits the teacher's TERMINATE threshold - see the module
# docstring. _episode_budgets derives the actual (variable) continuation
# length from this and the episode's own length.
_BUDGET_CONTINUATION_MIN: int = 2
_CATEGORIES: tuple[str, ...] = ("normal", "stuck_loop", "error_recovery", "risky_tool", "budget_exhaustion")
# Authorization of a risky call (src/authz.js in xybernetex-openclaw). Roughly
# the real mix: the 2026-09-27 replay of 565 runs labeled destructive calls
# ~55% requested, ~40% unrequested, the rest the agent's own artifacts; None
# covers sessions where no user turn was seen. More than one risky call per
# episode keeps each tier x label pair from being too thin to learn.
_AUTHORIZATIONS: tuple[tuple[str | None, float], ...] = (
    ("requested", 0.35), ("own_artifact", 0.15), ("unrequested", 0.35), (None, 0.15),
)
_MAX_RISKY_CALLS: int = 3
# Risky calls fail too, and a failed one is often retried. Until 2026-09-27
# every synthetic risky call succeeded, so a requested delete that failed twice
# (teacher: REQUEST_USER, via the error-escalation rule) had never been seen -
# the live v3 policy answered BLOCK_ACTION at 0.64 on exactly that, falling
# back on the destructive bit. A retry keeps the tool and label but not the
# params, so it reads as an error streak rather than a loop.
_RISKY_FAILURE_RATE: float = 0.3
_RISKY_RETRIES_MAX: int = 2
# Real agents often retry the identical command, so half of retries keep the
# params too (an error streak that turns into a loop -> REPLAN on the third).
# And a harmless failure elsewhere in a risky episode must still draw
# INJECT_CONTEXT with BLOCK_ACTION already in the history - the v3.1 replay
# found it drifting to CONTINUE there, having never seen that combination.
_RISKY_RETRY_IDENTICAL: float = 0.5
_SAFE_FAILURE_IN_RISKY_EPISODE: float = 0.4


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
        run_len = rng.randint(_ERROR_RUN_MIN, _ERROR_RUN_MAX)
        start = rng.randint(0, length - run_len)
        for i in range(start, start + run_len):
            history[i] = _random_call(rng, success=False)
    elif category == "risky_tool":
        labels, weights = zip(*_AUTHORIZATIONS)
        if rng.random() < _SAFE_FAILURE_IN_RISKY_EPISODE:
            history[rng.randrange(length)] = _random_call(rng, success=False)
        for position in rng.sample(range(length), rng.randint(1, _MAX_RISKY_CALLS)):
            call = _random_call(rng, risky_pool=_DESTRUCTIVE_TOOLS + _SENSITIVE_TOOLS)
            call.authorization = rng.choices(labels, weights)[0]
            call.success = rng.random() >= _RISKY_FAILURE_RATE
            history[position] = call
            if not call.success:
                # Retries of the same failing call, placed right after it.
                identical = rng.random() < _RISKY_RETRY_IDENTICAL
                for offset in range(1, rng.randint(0, _RISKY_RETRIES_MAX) + 1):
                    if position + offset < length:
                        params = call.params if identical else {"arg": rng.randint(0, 10_000)}
                        history[position + offset] = ToolCallRecord(
                            tool_name=call.tool_name, params=params, success=False,
                            authorization=call.authorization)
    return history


def _episode_budgets(category: str, rng: random.Random, length: int) -> tuple[int, float, float]:
    """Returns (max_steps, cost_per_step, cost_budget). Only budget_exhaustion
    episodes are sized to run out; everything else ends with at least 2/3 of
    its step budget and 1/2 of its cost budget left, comfortably clear of the
    teacher's TERMINATE threshold.

    budget_exhaustion's trigger lands partway through the episode, not at (or
    one step before) its very end, so continuation steps actually exist with
    the budget already exhausted and TERMINATE already in recent_actions -
    see the module docstring's 2026-09-25 note. The step-budget branch lets
    step_number run past max_steps for those continuation steps, same as real
    OpenClaw runs (nothing there hard-stops a run at its reported budget
    either) - state_vectorizer.py's s[1] goes negative and stays a clean
    TERMINATE signal; nothing clips it, deliberately."""
    cost_per_step = rng.uniform(0.1, 1.0)
    generous_cost = cost_per_step * length * rng.uniform(2.0, 5.0)
    generous_steps = length * rng.randint(3, 6)
    if category != "budget_exhaustion":
        if rng.random() < 0.4:
            # Tight but safe: stays clear of the teacher's exhaustion
            # threshold, but reaches into the 65-87% used zone other
            # categories otherwise never touch at all (a "generous" budget
            # never exceeds ~33% used across the episode's real length).
            # Without this, budget_exhaustion's own thin pre-trigger ramp
            # was the ONLY source of genuinely-CONTINUE examples in that
            # zone - too sparse to balance the extra TERMINATE mass the
            # post-trigger continuation fix adds there, which pulled the
            # model's decision boundary earlier than the teacher's true 90%
            # cutoff even after that fix worked as intended. See the module
            # docstring's 2026-09-25 note.
            denom = length / (1 - rng.uniform(0.13, 0.35))
            return max(length + 1, round(denom)), cost_per_step, cost_per_step * denom
        return generous_steps, cost_per_step, generous_cost
    # Trigger in the episode's last third: an offline retrain+ablation
    # (2026-09-25) found that letting the trigger land anywhere - including
    # very early in short episodes - starved training of genuine "70-90%
    # used, still CONTINUE" examples and shifted the model's decision
    # boundary earlier than the teacher's true 90% cutoff, trading the
    # history lock-in bug for a worse budget-calibration one. Keeping most
    # of the episode as a gradual, correctly-labeled approach avoids that
    # while still leaving room for genuine post-trigger continuation.
    trigger_step = max(2, min(length - _BUDGET_CONTINUATION_MIN, round(length * rng.uniform(0.65, 0.85))))
    if rng.random() < 0.5:
        return trigger_step, cost_per_step, generous_cost                 # step budget runs out
    return generous_steps, cost_per_step, cost_per_step * trigger_step    # cost budget runs out


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
    return [vectorize_v3(snapshot) for snapshot, _ in steps], [action for _, action in steps]


def generate_episodes(num_episodes: int = 1500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Returns (states, actions): states is (N, TRAINING_STATE_DIM) float32, actions
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
