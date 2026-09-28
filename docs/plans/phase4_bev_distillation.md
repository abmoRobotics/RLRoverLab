# Phase 4 — Driving on the persistent BEV: distillation first

Written 27 September 2026. Implements Step 3 of the [next steps](next_steps.md) as decided: distillation before PPO, and
the network–lift disagreement as the first perception-uncertainty signal
([Phase-3 results](../results/phase3_reliability.md)).

**Status (28 September):** steps 1–4 are done; see the [Phase-4 results](../results/phase4_distillation.md).

- **First criterion: met.** The full student reaches 98% of the teacher's success up to 2× drift.
- **Second criterion: not met.** No variant degrades consistently with drift up to 4×.
- **Camera dropouts** were added to the evaluation, since the current view alone suffices on this task. Under them,
  memory proves to be worth 9–22 points of success.
- **Step 5, PPO fine-tuning,** is the next step, under wider drift and dropouts.

## Goal

A driving policy that sees only what the rover can build itself: the persistent BEV from its camera and odometry, with
provenance and reliability, plus the goal. It is distilled from the height-map PPO teacher, which sees exact terrain.
The research question is whether knowing where each piece of terrain information came from, and how much to trust it,
changes driving under odometry drift of unknown size.

## Student observation

- `proprio`: distance and angles to the goal, as for the RGB student.
- `bev` [C, 120, 120], the same grid as all phases, from the settled Phase-3 pipeline:
  - noisy depth;
  - the frozen Phase-1 models, the noisy-depth one for classes;
  - the memory stores the raw lifted elevation and averages class probabilities.

  Drift is drawn per episode from 0.25× to 4× and varies with location. The policy never sees the drift level.

Channels of the full variant:

1. **elevation** relative to the rover (m);
2. **rock probability**;
3. **known**: current or remembered;
4. **current**: observed in this frame;
5. **expected elevation error**, in decimetres. This is the Phase-3 reliability map: the pose effect at the drift level
   the rover measures (`DriftMonitor`), plus the network–lift disagreement.

Unknown cells are marked by the known channel; the other channels are zero there, never NaN.

## Variants (the plan's ablations)

| Variant | Channels | Question |
|---|---|---|
| instantaneous | the current frame only: elevation, rock, observed | What does memory add? |
| persistent | memory: elevation, rock, known | Does memory help without knowing its provenance? |
| full | memory with provenance and reliability (all five) | Does knowing where terrain came from help under drift? |

## Student and training

- **Model.** A small CNN over the BEV, joined with `proprio`, then a GRU (2 × 256) and an MLP head with tanh output, as
  in the RGB student. The CNN replaces the frozen DINOv3 encoder and is trained.
- **Perception outside the student.** The perception runs once per step in an environment wrapper. The rollout storage
  keeps only `policy` (teacher), `proprio` and `bev`, never the raw images.
- **Training.** RSL-RL distillation with the RGB student's settings, for 400 iterations per variant, `mars`, seed 42.

## Evaluation

- **Protocol.** Each policy drives for a fixed number of steps with drift drawn per episode, on held-out seeds.
- **Outcomes per episode:** success, collision, timeout or leaving the area; path length over straight-line distance;
  time to goal.
- **Reported:** by drift octave, and pooled.
- **References.** The teacher and the RGB student are unaffected by drift; they give the upper reference and the
  image-only baseline.
- **Provisional criteria:**
  - the full variant reaches at least 90% of the teacher's success rate up to 2× drift;
  - its success rate drops less from the lowest to the highest drift octave than that of the persistent variant.

## Order of work

1. Disagreement (3.6): store it per cell, evaluate it, and export the reliability map.
2. Pipeline, environment wrapper, student, task and tests; then a short training run to check everything runs.
3. Train the full variant; build the evaluation; evaluate the teacher, the RGB student and the full variant.
4. Train and evaluate the ablations.
5. PPO fine-tuning only if distillation leaves a clear gap to the teacher.
