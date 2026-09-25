"""Packs a trained ActorCritic checkpoint into {manifest.json, weights.bin} -
the exact pair xybernetex-cfworker's scripts/upload_weights.js pushes to
Workers KV, and src/policy.js's decodeWeights()/policyForward() expect.

This is the ONLY place in either repo that needs to know the model's tensor
names/shapes - xybernetex-cfworker's upload script stays a dumb file-mover,
unlike the old design where the JS repo hand-duplicated the PyTorch class to
stay import-free (fragile, needed manual upkeep - see xybernetex-cfworker's
git history before its 2026-09-24 rewrite).

manifest.json: {state_dim, action_dim, layer_norm_eps, tensors: [{name, shape, offset, length}]}
weights.bin:   raw float32 bytes, one buffer, tensors index into it by offset
               (offset/length are in FLOATS, not bytes - the JS side multiplies
               by 4 itself, matching the old hierarchy-policy export's convention)

Tensor list is read directly off model.state_dict() rather than hardcoded,
so it automatically tracks chassis/actor_critic.py's actual architecture -
the manifest is the single source of truth the JS side reads, not a
hardcoded assumption duplicated on either side.

Usage (run as a module, not a script directly - it needs the repo root on
sys.path for the chassis package import):
    python -m export.export_weights --checkpoint models/policy_v1.pt --output-dir ./out
    python -m export.export_weights --random-seed 1234 --output-dir ./out   # local dev, no real checkpoint needed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chassis.actor_critic import ActorCritic, load_policy_bytes


def pack_weights(model: ActorCritic) -> tuple[dict, bytes]:
    """Returns (manifest, raw_bytes) for model's current weights."""
    tensors_meta = []
    chunks = []
    offset = 0
    for name, tensor in model.state_dict().items():
        flat = tensor.detach().cpu().numpy().astype(np.float32).flatten()
        tensors_meta.append({
            "name": name,
            "shape": list(tensor.shape),
            "offset": offset,
            "length": int(flat.size),
        })
        chunks.append(flat.tobytes())
        offset += flat.size

    # eps isn't in state_dict, so read it off the modules rather than
    # hardcoding it - the manifest carries a single value, so every LayerNorm
    # must agree.
    eps_values = {m.eps for m in model.modules() if isinstance(m, torch.nn.LayerNorm)}
    if len(eps_values) != 1:
        raise ValueError(f"expected every LayerNorm to share one eps, found {sorted(eps_values)}")

    manifest = {
        "state_dim": model.state_dim,
        "action_dim": model.action_dim,
        "layer_norm_eps": eps_values.pop(),
        "tensors": tensors_meta,
    }
    return manifest, b"".join(chunks)


def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun 7.1.26 approximation, ~1.5e-7 max absolute error.
    NumPy has no built-in erf and neither does JavaScript, so both the
    reference_forward() below and xybernetex-cfworker's src/policy.js use
    this exact formula - torch.nn.GELU()'s default is the EXACT erf-based
    GELU, not the cruder tanh approximation, and matching it this closely
    (rather than switching to the tanh approximation on both sides) keeps
    the same tight numerical parity this project's other JS ports already
    hit (hierarchy's ReLU-only policy verified to ~1e-7 - see
    xybernetex-cfworker's git history)."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return sign * y


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))


def reference_forward(manifest: dict, weights_bytes: bytes, state_vector: np.ndarray) -> tuple[np.ndarray, float]:
    """Pure NumPy forward pass reconstructed ONLY from manifest + raw bytes -
    no PyTorch, no reference to ActorCritic's class definition. Proves the
    exported format actually carries everything needed to reproduce the
    real model's output, and doubles as the spec xybernetex-cfworker's pure-JS
    src/policy.js (same architecture, same _erf approximation) is checked
    against - see tests/test_export_weights.py for the numerical comparison
    against the real PyTorch model (measured max abs diff: ~4.6e-7 over 200
    random inputs - matches this project's established precedent for these
    ports, e.g. hierarchy's earlier ReLU-only policy)."""
    flat = np.frombuffer(weights_bytes, dtype=np.float32)
    tensors = {t["name"]: flat[t["offset"]: t["offset"] + t["length"]].reshape(t["shape"])
               for t in manifest["tensors"]}

    def linear(x, name):
        return x @ tensors[f"{name}.weight"].T + tensors[f"{name}.bias"]

    def layer_norm(x, name):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + manifest["layer_norm_eps"])
        return normalized * tensors[f"{name}.weight"] + tensors[f"{name}.bias"]

    x = linear(state_vector, "trunk.0")
    x = _gelu(layer_norm(x, "trunk.1"))
    x = linear(x, "trunk.3")
    x = _gelu(layer_norm(x, "trunk.4"))

    action_logits = linear(x, "actor")
    value = linear(x, "critic")
    return action_logits, float(value[0])


def pack_and_write(model: ActorCritic, output_dir: str) -> tuple[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest, raw_bytes = pack_weights(model)
    manifest_path = out / "manifest.json"
    weights_path = out / "weights.bin"
    manifest_path.write_text(json.dumps(manifest))
    weights_path.write_bytes(raw_bytes)
    return str(manifest_path), str(weights_path)


def main():
    parser = argparse.ArgumentParser(
        description="Pack a trained ActorCritic checkpoint into manifest.json + weights.bin."
    )
    parser.add_argument("--checkpoint", help="Path to a .pt state_dict file (train/bc_trainer.py's output)")
    parser.add_argument("--random-seed", type=int,
                         help="Skip --checkpoint and pack a random fixed-seed model instead - "
                              "for local dev/testing without a real trained checkpoint")
    parser.add_argument("--output-dir", default="./out")
    args = parser.parse_args()

    if args.checkpoint and args.random_seed is not None:
        parser.error("pass --checkpoint or --random-seed, not both")
    elif args.checkpoint:
        with open(args.checkpoint, "rb") as f:
            model = load_policy_bytes(f.read())
    elif args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        model = ActorCritic()
        model.eval()
    else:
        parser.error("must pass either --checkpoint or --random-seed")

    manifest_path, weights_path = pack_and_write(model, args.output_dir)
    print(f"Wrote {manifest_path} and {weights_path}")


if __name__ == "__main__":
    main()
