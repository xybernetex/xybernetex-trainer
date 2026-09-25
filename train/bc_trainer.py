"""Behavioral cloning trainer - cross-entropy on the heuristic teacher's
chosen actions, ported from worker-prod's ppo_trainer.py (V3, itself a
correction from an earlier A2C-style live-rollout approach that "was
fundamentally broken - the value function bootstraps from static data,
producing noisy advantages that cause the DONE logit to dominate globally").

STUB (2026-09-24): not yet implemented, waiting on chassis/actor_critic.py
and a real source of labeled (state, action) pairs (data/synthetic_generator.py
initially, data/replay_loader.py later). Expected shape once state_dim and
the dataset exist: minimize CE(policy(state), teacher_action) directly, no
advantage estimation, no value-function bootstrapping - matching worker-prod's
own explicit reasoning for why this is the right approach for learning from
a heuristic demonstrator, not just a cheaper bootstrap before "real" RL.
"""
from __future__ import annotations


def train(*_args, **_kwargs):
    raise NotImplementedError(
        "bc_trainer.train: not implemented - waiting on chassis/actor_critic.py and a labeled dataset"
    )
