"""Behavioral cloning trainer - cross-entropy on the heuristic teacher's
chosen actions, ported from worker-prod's ppo_trainer.py (V3, itself a
correction from an earlier A2C-style live-rollout approach that "was
fundamentally broken - the value function bootstraps from static data,
producing noisy advantages that cause the DONE logit to dominate globally").

No advantage estimation, no value-function bootstrapping, no live rollouts -
minimize CE(policy(state), teacher_action) directly, same reasoning
worker-prod's own docstring gives for why this is the right approach for
learning from a heuristic demonstrator, not just a cheaper bootstrap before
"real" RL.

One addition beyond worker-prod's own trainer: a held-out validation split
with best-checkpoint tracking (by val_loss, not final epoch), the same
lesson Agent Drift Lab's hierarchy experiment learned the hard way after
observing severe overfitting on small datasets - train_loss improving is no
guarantee val_loss still is.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

from chassis.action_space import ActionSpace
from chassis.actor_critic import ActorCritic, count_parameters, save_policy_bytes

DEFAULT_DATA_GLOB = "data/*.npz"
DEFAULT_MODEL_PATH = "models/policy_v1.pt"

BATCH_SIZE: int = 512
EPOCHS: int = 120
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-4
LABEL_SMOOTH: float = 0.05
VAL_FRACTION: float = 0.1
# The first full training run's val_loss plateaued by ~epoch 20 of 120 - the
# remaining ~80% of runtime moved val_loss by 0.3%.
EARLY_STOP_PATIENCE: int = 15
SEED: int = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RolloutDataset(Dataset):
    """Loads and concatenates every {states, actions} .npz file matching
    data_glob - the same shape data/synthetic_generator.py's
    generate_and_save() writes, and eventually data/replay_loader.py's
    output once real sessions exist."""

    def __init__(self, data_glob: str = DEFAULT_DATA_GLOB):
        files = glob.glob(data_glob)
        if not files:
            raise FileNotFoundError(
                f"No .npz files found matching {data_glob!r} - run "
                "data/synthetic_generator.py's generate_and_save() first"
            )

        states_list, actions_list = [], []
        for f in files:
            with np.load(f) as data:
                states_list.append(data["states"])
                actions_list.append(data["actions"])

        self.states = torch.tensor(np.concatenate(states_list), dtype=torch.float32)
        self.actions = torch.tensor(np.concatenate(actions_list), dtype=torch.long)

        counts = torch.bincount(self.actions, minlength=ActionSpace.N)
        print("\n[Dataset] Action distribution:")
        for i, name in enumerate(ActionSpace.PRIMITIVES):
            pct = 100.0 * counts[i].item() / len(self.actions)
            print(f"  {name:16s}: {counts[i].item():6d}  ({pct:5.1f}%)")

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


def _class_weights(actions: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Sqrt-dampened inverse-frequency class weights, so a class like
    BLOCK_ACTION at <1% of the data doesn't get drowned out by CONTINUE at
    85% - found necessary in practice, not just in theory: an unweighted
    model trained on this dataset predicted CONTINUE for hand-crafted
    destructive-tool and repeated-call snapshots the teacher itself would
    clearly flag, despite those being unambiguous rules (see
    tests/test_bc_trainer.py's TestBehavioralCloningActuallyLearns, which is
    exactly what caught this).

    Plain inverse frequency (sklearn's "balanced" formula, n/(k*count)) was
    tried first and overcorrected badly: on this dataset it produces up to a
    ~200x weight spread between the rarest and most common present class,
    and the resulting model started predicting the rare class (BLOCK_ACTION)
    almost everywhere, including plainly healthy and REPLAN-worthy
    snapshots - worse than the unweighted failure it was meant to fix.
    Square-rooting the ratio keeps rare classes meaningfully upweighted
    (~15x spread on this same dataset) without that instability.

    A class with zero examples (MODIFY_ACTION, NARROW_TOOLS as of this
    dataset - see teacher/heuristic_teacher.py's docstring) gets weight 0
    rather than a division error; there's nothing to weight since nothing to
    learn from."""
    counts = torch.bincount(actions, minlength=num_classes).float()
    weights = torch.zeros(num_classes)
    present = counts > 0
    weights[present] = torch.sqrt(len(actions) / (present.sum() * counts[present]))
    return weights


def _run_epoch(model: ActorCritic, loader: DataLoader, optimizer=None, class_weights=None):
    """One pass over loader - trains (and steps optimizer) if optimizer is
    given, otherwise evaluates with grad disabled. Returns (avg_loss, accuracy)."""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(train_mode):
        for states, actions in loader:
            states, actions = states.to(device), actions.to(device)
            logits, _value = model(states)
            loss = F.cross_entropy(logits, actions, weight=class_weights, label_smoothing=LABEL_SMOOTH)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(actions)
            correct += (logits.argmax(dim=1) == actions).sum().item()
            n += len(actions)

    return total_loss / n, correct / n


def train_from_arrays(states: np.ndarray, actions: np.ndarray, *, epochs: int = EPOCHS,
                       batch_size: int = BATCH_SIZE, lr: float = LR, val_fraction: float = VAL_FRACTION,
                       patience: int = EARLY_STOP_PATIENCE, seed: int = SEED, verbose: bool = True):
    """Trains a fresh ActorCritic on already-loaded (states, actions) arrays.
    Returns (model, history) - model has the best-val_loss weights restored,
    not necessarily the final epoch's. Stops early once val_loss hasn't
    improved for `patience` epochs. Seeded end to end (init, split, shuffle),
    so the same data and seed produce the same checkpoint. Separated from
    RolloutDataset/train() so this core loop is testable against a small
    in-memory dataset without needing real .npz files on disk."""
    torch.manual_seed(seed)
    actions_t = torch.tensor(actions, dtype=torch.long)
    dataset = TensorDataset(torch.tensor(states, dtype=torch.float32), actions_t)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_set, batch_size=batch_size)

    class_weights = _class_weights(actions_t[train_set.indices], ActionSpace.N).to(device)

    model = ActorCritic(state_dim=states.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_state = None
    epochs_since_best = 0
    history = []

    for epoch in range(epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, optimizer, class_weights)
        val_loss, val_acc = _run_epoch(model, val_loader, optimizer=None, class_weights=class_weights)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        stopping = epochs_since_best >= patience
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1 or stopping):
            print(f"  epoch {epoch:4d}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        if stopping:
            if verbose:
                print(f"  early stop: no val_loss improvement for {patience} epochs")
            break

    if best_state is None:
        # Every val_loss was NaN/inf (NaN < inf is False), so nothing was ever
        # worth keeping - fail loudly rather than load_state_dict(None).
        raise RuntimeError("val_loss was never finite - training diverged; no checkpoint to restore")

    model.load_state_dict(best_state)
    model.eval()
    return model, history


def train(data_glob: str = DEFAULT_DATA_GLOB, model_path: str = DEFAULT_MODEL_PATH, **kwargs):
    dataset = RolloutDataset(data_glob)
    model, history = train_from_arrays(dataset.states.numpy(), dataset.actions.numpy(), **kwargs)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    with open(model_path, "wb") as f:
        f.write(save_policy_bytes(model))

    best_epoch = min(history, key=lambda h: h["val_loss"])
    print(f"\nBest checkpoint: epoch {best_epoch['epoch']} (val_loss={best_epoch['val_loss']:.4f}, "
          f"val_acc={best_epoch['val_acc']:.4f})")
    print(f"Saved policy checkpoint: {model_path} ({count_parameters(model)} parameters)")
    return model_path, history


if __name__ == "__main__":
    train()
