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
- **Training** (`train/bc_trainer.py`, not yet implemented): cross-entropy
  behavioral cloning against the teacher's labels. No live rollouts, no
  policy-gradient RL, no reward function needed at all for v1 - this
  sidesteps the sparse/delayed-reward problem a supervisor's task naturally
  has, rather than trying to solve it.
- **Data** (`data/`): `synthetic_generator.py` fabricates plausible
  trajectories fully offline, at zero cost, across five categories
  (normal / stuck-loop / error-recovery / risky-tool / budget-exhaustion) -
  the actual starting point for training, mirroring worker-prod's own
  1500-episode offline bootstrap. 1500 episodes -> 30,000 labeled steps;
  `CONTINUE` still dominates (85%) but every reachable action is represented
  (see the known gap above for the two that aren't). `replay_loader.py` is a
  placeholder for parsing real logged OpenClaw sessions once Pass 1
  (observe-only) logging exists - blocked for now.
- **Export** (`export/export_weights.py`, not yet implemented): packs a
  trained checkpoint into `manifest.json` + `weights.bin` for
  `xybernetex-cfworker`. The *only* place in either repo that needs to know
  the model's architecture - `xybernetex-cfworker`'s upload script is a
  dumb file-mover, not a hand-duplicated copy of this model's class
  (the old, more fragile pattern hierarchy's cfworker repo used).

## Status (2026-09-24)

`chassis/`, `data/trajectory_schema.py`, `data/synthetic_generator.py`, and
`teacher/heuristic_teacher.py` are all real and tested (42 tests). Only
`train/bc_trainer.py` and `export/export_weights.py` remain stubs. Known
limitation: the heuristic teacher never produces `MODIFY_ACTION` or
`NARROW_TOOLS` - see the Teacher section above. Build order:
`bc_trainer.py` (unblocked - a real labeled dataset now exists via
`synthetic_generator.generate_and_save()`) -> `export_weights.py`.

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
pip install -r requirements.txt
cp .env.example .env            # only needed once the goal-drift feature is implemented
```

## Test

```bash
python -m unittest discover -s tests
```
