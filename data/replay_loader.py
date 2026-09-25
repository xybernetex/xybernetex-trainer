"""Parses real logged OpenClaw sessions into TrajectorySnapshot sequences,
once such logging exists.

BLOCKED (2026-09-24): OpenClaw Pass 1 (observe-only logging, per the
staged-rollout plan - log what the supervisor would have decided vs. what
OpenClaw actually did, enforce nothing) hasn't been built yet. Until then
data/synthetic_generator.py is the only source of training data. This file
exists now as a placeholder for the interface real data will eventually need
to satisfy - not because there's anything to load yet.
"""
from __future__ import annotations


def load_sessions(log_dir: str):
    raise NotImplementedError(
        "replay_loader.load_sessions: not implemented - blocked on OpenClaw Pass 1 observe-only logging existing"
    )
