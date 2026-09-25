"""Actor-critic MLP for the OpenClaw supervisor MDP.

STUB (2026-09-24): architecture pattern is decided (ported from worker-prod's
rl_chassis.py ActorCritic - shared LayerNorm+GELU trunk, ~100K params at that
scale), but STATE_DIM is pinned to state_vectorizer.py's output width, which
isn't finalized yet. Filling this in is one of the first real tasks once
that module's feature set is settled - the class body below is expected to
look almost exactly like worker-prod's own, just with STATE_DIM TBD and
ACTION_COUNT = 8 (see action_space.py) instead of 6.

Reference shape (worker-prod's rl_chassis.py):
    trunk:  state_dim -> 256 (GELU + LayerNorm) -> 128 (GELU + LayerNorm)
    actor:  128 -> action_dim   (logits, no softmax - softmax applied by
            the caller, both here for training and in xybernetex-cfworker's
            src/index.js for serving)
    critic: 128 -> 1            (state value estimate)

Architecture is fixed once real training starts; changing it invalidates
saved weights.
"""
from __future__ import annotations

# import torch
# import torch.nn as nn

from .action_space import ActionSpace

ACTION_COUNT: int = ActionSpace.N
STATE_DIM: int | None = None  # TBD - see state_vectorizer.py


class ActorCritic:
    def __init__(self, *_args, **_kwargs):
        raise NotImplementedError(
            "ActorCritic: not implemented - waiting on state_vectorizer.py to fix STATE_DIM"
        )
