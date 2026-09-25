"""Opt-in alpha state layout. Never feeds an existing 140-input checkpoint.

Keep the entire v1 vector at [0:140], followed by eight explicitly versioned
features. Breadth indicates tool variety, NOT lack of progress or permission.
"""
from dataclasses import replace
import numpy as np
from data.trajectory_schema import ProposedToolCall

from .state_vectorizer import vectorize, effective_risk_tier, RISK_LABELS

FEATURE_SCHEMA = "xybernetex.state.v2"
STATE_DIM = 148
EXTRA_FEATURES = (
    "proposed_present", "proposed_sensitive", "proposed_destructive", "proposed_risk_assessed",
    "recent_tool_breadth", "proposed_repeat_count", "proposed_new_tool", "goal_drift_available",
)


def vectorize_proposal(snapshot):
    if snapshot.feature_schema != FEATURE_SCHEMA:
        raise ValueError("proposal vectorizer requires xybernetex.state.v2")
    base = vectorize(replace(snapshot, feature_schema="xybernetex.state.v1", proposed_call=None))
    extra = np.zeros(len(EXTRA_FEATURES), dtype=np.float32)
    recent = snapshot.tool_history[-8:]
    if recent:
        extra[4] = len({c.tool_name for c in recent}) / len(recent)
    extra[7] = snapshot.goal_drift_score is not None
    proposed = snapshot.proposed_call
    if proposed is not None:
        if not isinstance(proposed, ProposedToolCall):
            raise ValueError("proposed_call must be a pending call without execution outcomes")
        if not isinstance(proposed.tool_name, str) or not isinstance(proposed.params, dict):
            raise ValueError("proposed_call needs a string tool_name and object params")
        if proposed.risk is not None and proposed.risk not in RISK_LABELS:
            raise ValueError("invalid proposed_call risk")
        extra[0] = 1
        tier = effective_risk_tier(proposed)
        extra[1] = tier == "sensitive"
        extra[2] = tier == "destructive"
        extra[3] = proposed.risk is not None
        count = 0
        for call in reversed(recent):
            if call.tool_name != proposed.tool_name or call.params != proposed.params:
                break
            count += 1
        extra[5] = min(count / 5, 1)
        extra[6] = bool(recent) and proposed.tool_name not in {c.tool_name for c in recent}
    return np.concatenate([base, extra]).astype(np.float32)
