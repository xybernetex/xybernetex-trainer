"""Actor-critic MLP for the OpenClaw supervisor MDP.

Ported from worker-prod's rl_chassis.py ActorCritic - same shared
LayerNorm+GELU trunk, same layer widths. STATE_DIM comes from
state_vectorizer.py (140: 12 structural features + 16x8 action history);
ACTION_COUNT comes from action_space.py (8 discrete supervisor
interventions). 70,921 parameters at this scale - still comfortably small
enough for the pure-JS Cloudflare Worker forward pass in xybernetex-cfworker.

Architecture is fixed once real training starts; changing it invalidates
saved weights (both here and in xybernetex-cfworker's src/policy.js, which
must mirror this exactly).

Trunk:  state_dim (140) -> 256 (GELU + LayerNorm) -> 128 (GELU + LayerNorm)
Actor:  128 -> action_dim (8)   (logits, no softmax - softmax applied by
        the caller, both here for training and in xybernetex-cfworker's
        src/index.js for serving)
Critic: 128 -> 1                (state value estimate)
"""
from __future__ import annotations

import io

import torch
import torch.nn as nn

from .action_space import ActionSpace
from .state_vectorizer import TRAINING_STATE_DIM

ACTION_COUNT: int = ActionSpace.N


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = TRAINING_STATE_DIM, action_dim: int = ACTION_COUNT) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        features = self.trunk(x)
        return self.actor(features), self.critic(features)


def save_policy_bytes(model: ActorCritic) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def load_policy_bytes(data: bytes) -> ActorCritic:
    """Dimensions come from the checkpoint itself, so a v1 (140-input) and a
    v3 (143-input) checkpoint both load without the caller knowing which."""
    state = torch.load(io.BytesIO(data), map_location="cpu")
    model = ActorCritic(state_dim=state["trunk.0.weight"].shape[1], action_dim=state["actor.weight"].shape[0])
    model.load_state_dict(state)
    model.eval()
    return model


def count_parameters(model: ActorCritic) -> int:
    return sum(p.numel() for p in model.parameters())
