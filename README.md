# xybernetex-trainer

Training pipeline for the OpenClaw supervisor policy served by
[`xybernetex-cfworker`](https://github.com/xybernetex/xybernetex-cfworker).
Python, runs wherever training actually happens (not on the edge) - a
deliberate split from the serving repo, which stays edge-only JS with zero
heavy dependencies.

## Where this design came from

Adapted from Xybernetex's own production agent orchestrator
(`Agent Orchestrator/worker-prod`'s `rl_chassis.py`), not from Agent Drift
Lab's hierarchy experiment. That experiment's approach - a continuous
belief-correction policy, behavioral-cloned offline then PPO-fine-tuned
against live rollouts - was tested end to end (12 matched pairs against an
LLM baseline) and lost decisively: 0/12 pairs, p=0.0005, median relative
error ~5x the baseline's. worker-prod's own version history shows an
independent, prior instance of the same failure class: an earlier
A2C-style live-rollout policy gradient "was fundamentally broken - the
value function bootstraps from static data, producing noisy advantages
that cause the DONE logit to dominate globally" (see `ppo_trainer.py`'s
docstring there). They fixed it by dropping live RL entirely in favor of
pure behavioral cloning against a rule-based heuristic expert - this repo
follows that same recipe, not hierarchy's.

## Design

- **Action space** (`chassis/action_space.py`): 8 discrete supervisor
  interventions - `CONTINUE`, `REPLAN`, `BLOCK_ACTION`, `MODIFY_ACTION`,
  `NARROW_TOOLS`, `INJECT_CONTEXT`, `REQUEST_USER`, `TERMINATE`. A
  supervisor governing a *separate* agent (OpenClaw's own embedded
  reasoning), not an agent choosing its own moves - different problem from
  worker-prod's own `PLAN`/`ANALYZE`/`INVOKE_TOOL`/etc., same chassis shape.
- **State** (`chassis/state_vectorizer.py`): hand-engineered structural
  features from a `TrajectorySnapshot` (`data/trajectory_schema.py`) - step
  budget, repeated-tool-call detection, error telemetry, a two-tier
  risky-tool flag (destructive vs. sensitive), one-hot action history, and a
  goal-drift signal reusing Agent Drift Lab's already-proven Cloudflare
  embedding call (precomputed upstream, never called from inside the
  vectorizer - see its own docstring on why). `STATE_DIM = 140`. Not a text
  embedding of the whole trajectory - worker-prod's pattern of
  deterministic, structural features is what actually made its policy work,
  and that's the part worth keeping.
- **Model** (`chassis/actor_critic.py`): shared LayerNorm+GELU trunk (same
  layer widths as worker-prod's own) -> actor head (8-way softmax) + critic
  head, 70,921 parameters - still small enough for the same pure-JS
  Cloudflare Worker forward pass already proven for hierarchy's
  policy.
- **Teacher** (`teacher/heuristic_teacher.py`): deterministic rules, not an
  LLM judge - same choice worker-prod's `SmarterTeacher` makes. Generates
  the (state, action) labels behavioral cloning trains against. **Known
  gap**: the current rule set only ever produces 6 of the 8 actions -
  `MODIFY_ACTION` and `NARROW_TOOLS` are unreachable because they need
  information `TrajectorySnapshot` doesn't carry yet (a pending
  not-yet-executed proposed call, and a tool-usage-breadth signal,
  respectively). A classifier trained on this teacher's labels alone will
  never learn to predict those two - see the module's own docstring before
  treating it as feature-complete.
- **Training** (`train/bc_trainer.py`): cross-entropy behavioral cloning
  against the teacher's labels, with sqrt-dampened inverse-frequency class
  weighting (plain inverse frequency overcorrected badly - see the module's
  own docstring) and best-checkpoint-by-val_loss tracking, the same lesson
  Agent Drift Lab's hierarchy experiment learned about overfitting on small
  datasets. No live rollouts, no policy-gradient RL, no reward function
  needed at all for v1 - this sidesteps the sparse/delayed-reward problem a
  supervisor's task naturally has, rather than trying to solve it.
- **Data** (`data/`): `synthetic_generator.py` fabricates plausible
  trajectories fully offline, at zero cost, across five categories
  (normal / stuck-loop / error-recovery / risky-tool / budget-exhaustion) -
  the actual starting point for training, mirroring worker-prod's own
  1500-episode offline bootstrap. Each episode draws its own length (12-30
  steps), trigger position, and budgets, and only budget-exhaustion
  episodes actually run out - an earlier fixed-shape version ended every
  episode in TERMINATE and pinned each trigger to one position (see the
  module docstring). 1500 episodes -> ~31,500 labeled steps; `CONTINUE`
  dominates (91%) but every reachable action is represented (see the known
  gap above for the two that aren't). `replay_loader.py` is a placeholder
  for parsing real logged OpenClaw sessions once Pass 1 (observe-only)
  logging exists - blocked for now.
- **Export** (`export/export_weights.py`): packs a trained checkpoint into
  `manifest.json` + `weights.bin` for `xybernetex-cfworker`, reading the
  tensor list directly off `model.state_dict()` rather than hardcoding
  names - the *only* place in either repo that needs to know the model's
  architecture. Includes a pure-NumPy `reference_forward()` (no PyTorch)
  that reconstructs the model's output from ONLY the packed bytes, proving
  the format actually carries everything needed - matched the real
  PyTorch model to ~4.6e-7 max abs diff over 200 random inputs. GELU has no
  native NumPy or JavaScript implementation, so both this reference and
  `xybernetex-cfworker`'s eventual `src/policy.js` use the same
  Abramowitz & Stegun erf approximation rather than the cruder tanh-GELU
  shortcut, to keep that same tight numerical parity on the JS side too.
- **Vectorizer parity** (`export/export_vectorizer_cases.py`): the Worker
  builds features itself from a raw snapshot, using a JS port of
  `chassis/state_vectorizer.py`. This script runs the real Python
  vectorizer over edge cases and seeded synthetic episodes and writes the
  (snapshot, features) pairs `xybernetex-cfworker`'s tests must reproduce
  bit-for-bit. **Any change to the vectorizer needs this rerun and the
  output committed to cfworker**, or training and serving silently
  disagree:
  `python -m export.export_vectorizer_cases --output ../xybernetex-cfworker/test/fixtures/vectorizer_cases.json`

## Status (2026-09-25)

Every module is real and tested (68 tests). The current checkpoint scores
100% on every reachable action except `CONTINUE` (99.6%) on a fresh,
never-trained-on evaluation set, and matches the teacher on hand-crafted
snapshots with isolated triggers - e.g. step 19 of 100 -> `CONTINUE`, step
19 of 20 -> `TERMINATE` - so it's reading budgets, not episode position.

Known, documented limitations:
- `MODIFY_ACTION`/`NARROW_TOOLS` are unreachable by the current teacher
  (see Teacher above).
- The policy partly leans on its own previous action: two consecutive
  failures with an *empty* action history get `INJECT_CONTEXT` instead of
  the teacher's `REQUEST_USER` (the same situation mid-episode, where the
  first failure already produced `INJECT_CONTEXT`, is 300/300). Matters for
  sessions the supervisor joins partway through.

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
cp .env.example .env            # only needed once the goal-drift feature is implemented
```

## Test

### Pending-call review (v2, not training data yet)

Run the plugin's `python scripts/control_smoke.py --proposals` against the
configured disposable OpenClaw sandbox, then audit its two captured logs:

```bash
python -m scenarios.proposal_report ../xybernetex-openclaw/out/<batch>/observe-telemetry.jsonl ../xybernetex-openclaw/out/<batch>/enforce-telemetry.jsonl --output out/proposal-review.json
```

This validates v2 vectors and joins proposals, local rule decisions and local
completion callbacks by run and host call ID. Missing outcomes remain unknown;
ambiguous, reordered or modified calls are flagged. A completion callback is
not independent proof of filesystem effects or task success. All exported
targets remain null until reviewed: explicit test restrictions are not universal
teacher labels. Split future datasets by whole run before training.

```bash
python -m unittest discover -s tests
```
