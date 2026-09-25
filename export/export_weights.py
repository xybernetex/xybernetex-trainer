"""Packs a trained ActorCritic checkpoint into {manifest.json, weights.bin} -
the exact pair xybernetex-cfworker's scripts/upload_weights.js pushes to
Workers KV, and src/policy.js's decodeWeights()/policyForward() expect.

STUB (2026-09-24): not yet implemented, waiting on chassis/actor_critic.py's
final architecture. This is the ONLY place in either repo that needs to know
the model's tensor names/shapes - xybernetex-cfworker's upload script stays
a dumb file-mover, unlike the old design where the JS repo hand-duplicated
the PyTorch class to stay import-free (fragile, needed manual upkeep - see
xybernetex-cfworker's git history before its 2026-09-24 rewrite).

Expected shape once actor_critic.py exists:
    manifest.json: {state_dim, action_dim, tensors: [{name, shape, offset, length}]}
    weights.bin:   raw float32 bytes, one buffer, tensors index into it by offset

Usage (once implemented):
    python export_weights.py --checkpoint models/policy_v1.pt --output-dir ./out
"""
from __future__ import annotations


def pack_weights(*_args, **_kwargs):
    raise NotImplementedError(
        "export_weights.pack_weights: not implemented - waiting on chassis/actor_critic.py"
    )
