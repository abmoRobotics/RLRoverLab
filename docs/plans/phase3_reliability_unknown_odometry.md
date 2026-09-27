# Phase 3 — Reliability of the persistent BEV when the odometry quality is unknown

Written 27 September 2026. Refines Steps 1 and 2 of the [next steps](next_steps.md) after the
[Phase-2 results](../results/phase2_temporal_memory.md).

**Status (27 September, evening):** 3.0–3.4 are done; see the [Phase-3 results](../results/phase3_reliability.md).

- **3.3.** The keyframe registration proved too noisy at 1×. The drift is measured instead from each cell's current
  against its first observation (`DriftMonitor`). The success criteria of 3.4 are met on held-out runs.
- **3.5.** Decided: measure only; do not correct the pose.

## The constraint

The rover does not know how good its odometry is. On the real rover the drift depends on where it drives: wheels slip
more in loose sand and on slopes, and visual–inertial odometry drifts faster on texture-poor ground. The drift also
differs from one run to the next. A covariance reported by an odometry system is at best an uncalibrated hint.

This has four consequences for all work below:

1. **Nothing on the robot side may use the true drift level or the true noise model.** The memory and the reliability
   estimate receive the estimated pose only. They may assume the *shape* of a nominal noise model, e.g. that heading
   errors displace far cells more than near ones, but never its scale.
2. **Evaluations draw the drift level per run and vary it along the route.** Each episode draws its own severity from a
   wide log-uniform range, and a smooth random field over the terrain scales it further by location. The true level is
   logged for analysis only.
3. **Reliability is judged on held-out runs over the whole range of drift levels.** A map that is right only on average
   over-trusts the memory in bad runs and under-trusts it in good ones. In Phase 2, terrain remembered over 4–8 m had
   1.9 cm error with exact poses, 3.6 cm at 1× drift and 10.9 cm at 4×: a fixed calibration is off by about a factor of
   three at one end or the other.
4. **The actual drift can only be learned from the robot's own data**, i.e. from terrain it observes again. If a new
   frame disagrees with what the rover saw of the same terrain a metre earlier, the pose has drifted in between.

## Steps

### 3.0 Close Phase 2 (next steps, Step 1)

Unchanged in intent, with the drift runs replaced by the protocol of 3.1:

1. **Stored elevation under noisy depth.** `AAURoverEnvSimpleBEVNoisyDepth-v0`, clean (`model_249.pt`) and noisy-depth
   (`model_119.pt`) models, network and raw lifted elevation. The plan's rule decides: store the source with the lower
   remembered-cell MAE. All later runs use noisy depth and that source.
2. **Class evidence fusion.** Running mean of the class probabilities and clamped log-odds accumulation, next to
   "latest". Circles and goal-driving, exact poses and varying drift (averaging may blur rocks under drift).
   *Success:* remembered rock IoU while circling ≥ 0.80, without losing goal-driving IoU.
3. **Error bars.** A second seed and the `debug` terrain, with exact poses and varying drift.

### 3.1 Drift of unknown and varying size

- `OdometryDrift` moves into the package (`learning/bev/odometry.py`) with the Phase-2 error structure unchanged. A fixed
  severity reproduces the Phase-2 drift exactly.
- `severity` may be a range: each episode draws its own severity log-uniformly, by default from 0.25× to 4×.
- A random field over the world (log standard deviation 0.5, correlation length 5 m, shared by all environments like a
  property of the terrain) multiplies the level locally. Biases stay continuous: a unit Gauss–Markov state is scaled by
  the current level.
- The evaluation reports the memory's error by the run's severity. It can also dump a per-cell sample of remembered
  cells with the trajectories, so reliability models can be fitted and compared offline without the simulator.

### 3.2 Reliability from observable signals, with a fixed calibration

Per remembered cell, all available on the robot:

- distance travelled and time since the cell was observed (existing);
- the range at which it was observed;
- **U_pose with unit scale.** For a random-walk heading error, a cell at world position c that was observed at the
  start of path P is displaced with variance ∝ ∫_P |c − p|² ds (the lever arm), plus ∝ length(P) for translation and
  height. These integrals only need running path moments, so they are cheap;
- local relief in the persistent map: a horizontal shift only causes elevation error where the terrain is not flat.

Each signal gets a monotone map to the expected elevation error (and rock-confusion probability), fitted on training runs
(`mars`, seed 42) and tested on held-out runs (seed 43, `debug`).

- **Ranking:** risk–coverage curves and their area (AURC), Spearman correlation with the actual error.
- **Calibration:** observed over predicted error per severity bin, on cells remembered over at least 1 m.
- **Decision:** keep U_pose only if it ranks errors better than the distance (next steps, 2.2).

*Expected:* good ranking, but a calibration that fails at both ends of the severity range, because nothing observable
in this step reveals the run's drift level.

### 3.3 Measure the drift online from re-observed terrain

Register each frame's depth lift against a keyframe about 1 m back along the route. The registration finds the small
correction (x, y, yaw, z) that makes the two agree on the cells both observe, with a robust cost and bilinear sampling,
from a coarse grid refined by Gauss–Newton. The correction divided by the distance travelled measures the local drift
rate. An average over the last few keyframes follows changes along the route.

- The depth lift is the most precise geometry the robot has, and it is available with noisy depth too.
- Flat, featureless ground makes the horizontal correction unobservable. There the estimator must report low confidence
  instead of low drift. Horizontal drift also causes little elevation error on flat ground.
- *Check (simulation only):* estimated against true drift per keyframe interval, the estimation floor with exact poses,
  and the bias per severity bin. *Provisional criterion:* Spearman ≥ 0.7 between estimated and true drift, and the
  estimate within a factor of 1.5 of the truth from 1× upward.

### 3.4 Adaptive reliability

Accumulate U_pose with the measured rates instead of unit rates. This is a path integral, so a bad stretch only raises
the uncertainty of cells observed before it. Compare it with 3.2 on the held-out runs.

*Success:* observed over predicted error between 0.67 and 1.5 in every severity bin, where the fixed calibration of 3.2
falls outside, with an AURC no worse than 3.2.

### 3.5 Decision point: measure only, or also correct?

If 3.3 measures the drift well, the same registration can correct the pose, i.e. terrain-relative odometry. That
changes what the memory stores and needs its own evaluation, so it is decided after 3.3.

### 3.6 Perception uncertainty (next steps, 2.3)

Free signals first: class entropy, and the disagreement between the network and the raw lifted elevation.

## Consequences for Phase 4 (RL)

- Drift is drawn per episode during training and evaluation, from the same protocol; the policy never sees the level.
- Its reliability channel is the adaptive estimate of 3.4. The ablations add the fixed calibration of 3.2.
- Results are reported by severity bin, not only pooled.

## Order of work

1. Code: class fusion in `BevMemory`, the varying drift model, its evaluation flags and the sample dump, with unit
   tests. Then the 3.0 runs.
2. Offline analysis of 3.2 on the dumped samples.
3. 3.3, then 3.4, then the decision of 3.5.
