"""Discrete action set for the OpenClaw supervisor MDP.

Adapted from Xybernetex's own production orchestrator (worker-prod's
rl_chassis.py ActionSpace), which chooses an agent's OWN next move
(PLAN/ANALYZE/INVOKE_TOOL/...). This vocabulary is different on purpose:
the supervisor governs a SEPARATE agent (OpenClaw's own embedded LLM
reasoning), so its actions are interventions, not primitives an agent
executes directly.

Index -> primitive mapping is fixed; changing it invalidates any trained
weights (both here and in xybernetex-cfworker's src/action_space.js, which
must mirror this list exactly).
"""
from __future__ import annotations

import numpy as np


class ActionSpace:
    PRIMITIVES: tuple[str, ...] = (
        "CONTINUE",        # 0 - let the agent proceed unmodified
        "REPLAN",          # 1 - inject context prompting the agent to reconsider its approach
        "BLOCK_ACTION",    # 2 - refuse the proposed tool call outright
        "MODIFY_ACTION",   # 3 - alter the proposed tool call's parameters before it runs
        "NARROW_TOOLS",    # 4 - restrict the tool surface available for the next step
        "INJECT_CONTEXT",  # 5 - add corrective context without blocking anything
        "REQUEST_USER",    # 6 - pause and ask a human to weigh in
        "TERMINATE",       # 7 - end the run
    )

    N: int = len(PRIMITIVES)
    _IDX: dict[str, int] = {p: i for i, p in enumerate(PRIMITIVES)}

    @classmethod
    def decode(cls, index: int) -> str:
        if not 0 <= index < cls.N:
            raise ValueError(f"Invalid action index {index}")
        return cls.PRIMITIVES[index]

    @classmethod
    def encode(cls, action: str) -> int:
        try:
            return cls._IDX[action]
        except KeyError:
            raise ValueError(f"Unknown action {action!r}") from None

    @classmethod
    def sample(cls) -> int:
        """Uniform random action - offline testing only, never used at inference."""
        return int(np.random.randint(0, cls.N))

    @classmethod
    def mask(cls, valid: list[str]) -> np.ndarray:
        """Boolean mask over the action space. True = action available.

        e.g. TERMINATE/REQUEST_USER only valid once certain conditions hold -
        mirrors worker-prod's own hard-masking of INVOKE_TOOL when a goal
        doesn't need it.
        """
        m = np.zeros(cls.N, dtype=bool)
        for a in valid:
            if a in cls._IDX:
                m[cls._IDX[a]] = True
        return m
