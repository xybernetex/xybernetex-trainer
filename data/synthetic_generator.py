"""Fabricates plausible TrajectorySnapshot sequences for the heuristic
teacher to label - the supervisor's equivalent of worker-prod's
rollout_generator.py run_expert_generation(), which produced 1500 simulated
episodes entirely offline, at zero live cost, before any real orchestrator
traffic existed.

STUB (2026-09-24): not yet implemented. This is the actual starting point
for training data, since replay_loader.py is blocked on OpenClaw Pass 1
(observe-only) logging existing for real - see xybernetex-cfworker's sibling
project memory for that staged-rollout plan. Should generate a mix of
"normal" trajectories (no intervention needed), stuck-loop trajectories
(same tool call repeated), error-recovery trajectories, risky-tool
trajectories, and budget-exhaustion trajectories, so the heuristic teacher's
full rule set actually gets exercised in the resulting demonstrations.
"""
from __future__ import annotations


def generate_episodes(num_episodes: int = 1500):
    raise NotImplementedError(
        "synthetic_generator.generate_episodes: not implemented - see this module's docstring"
    )
