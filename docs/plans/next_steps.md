# Next steps after Phase 2

Written 27 September 2026, after the Phase-2 evaluation ([results](../results/phase2_temporal_memory.md),
[summary](../results/summary.md)).

## Where we stand

- **Phase 1 (instantaneous BEV) works.** 1.2 cm elevation MAE and rock IoU 0.90 on visible cells. Geometry from depth
  carries the elevation; DINOv3 adds a small, consistent gain on rocks.
- **Phase 2 (deterministic world-aligned memory) works.** It covers 99% of terrain seen earlier at 2.0 cm and rock IoU
  0.82, against 31 cm and 0.10 for the network's guesses, and it does not degrade the current view.
- **What limits it, in order of impact:**
  1. **Odometry drift.** Remembered terrain degrades with the distance travelled since it was seen: at 4× drift, 11 cm
     and rock IoU 0.46 after 4–8 m.
  2. **The network's elevation is less precise than its own depth input.** Storing the raw lifted elevation halves the
     memory's error.
  3. **Rocks in memory.** Remembered rock IoU drops right after rocks leave the view (0.70) and while turning (0.74).
  4. **Idealized sensing.** Depth is perfect and poses are exact unless drift is added, with one map, one seed and one
     checkpoint.

The next work should first remove the idealizations that could change the Phase-2 conclusions (step 1). It should then
turn provenance into a trust signal (step 2) before the persistent BEV becomes an RL observation (step 3). Real-sensor
work (step 4) can run in parallel whenever hardware time is available.

## Step 1 — Close Phase 2 (small, a few days)

Keep the architecture and the deterministic memory; only settle the open choices.

1. **Stored elevation under realistic depth.** Rerun the memory evaluation on `AAURoverEnvSimpleBEVNoisyDepth-v0` with
   two elevation sources: the network's and the raw lifted elevation. Evaluate both the frozen clean model
   (`model_249.pt`) and the noisy-depth ablation model (`logs/bev/rover_bev_noisy_depth/…/model_119.pt`).
   - *Decision:* store whichever gives the lower remembered-cell MAE under noise. If the raw lift wins there too, the
     memory stores geometry, and the network supplies classes and fills cells without depth.
   - Needs a `--task` switch that already exists; no code changes.
2. **Class evidence fusion.** Replace "latest class wins" with a running average of the class probabilities, or
   log-odds accumulation, per cell (elevation stays latest or closest). Evaluate on the circle runs, where repeated
   views matter.
   - *Success:* remembered rock IoU while circling ≥ 0.80 (from 0.74), without losing goal-driving IoU (0.82).
   - This is still a simple deterministic rule, as the Phase-2 plan allows.
3. **Robustness of the numbers.** Repeat the baseline and 1× drift evaluations with a second seed and on a second
   terrain (`debug` or a generated terrain). Report run-to-run spread, so later comparisons have error bars.
4. **Optional: cleaner evidence.** 3% of never-visible cells receive memory values, mostly depth points on rock faces.
   Try requiring a minimum number of points per cell, or rejecting cells whose points span a large height range. Keep
   it only if it lowers the unknown-cell error without losing coverage.

## Step 2 — Phase 3: reliability of every cell

The persistent BEV already records the time and distance since each cell was observed. Phase 3 should turn that into a
calibrated estimate of how wrong each cell is likely to be, because a reliability-aware policy needs it.

1. **Empirical error model (cheap, uses existing data).** From the evaluation runs, fit the expected elevation error
   and rock-confusion probability of remembered cells as functions of distance since observation, age, observation
   range and drift level. Output a per-cell reliability map, e.g. expected error in cm.
   - *Check:* risk–coverage curves. Discarding the least reliable cells first should lower the error of the kept cells
     monotonically.
2. **Pose uncertainty instead of a distance proxy.** Propagate the odometry covariance to a per-cell spatial
   uncertainty (U_pose), as in the earlier BEV experiments' Day 5–7 work. Compare it with the distance proxy. Keep
   U_pose only if it predicts error better on the drift runs.
3. **Perception uncertainty (needs a decision).** Either add the plan's optional log σ elevation head to the Phase-1
   network, which means retraining Phase 1, or start with free signals: class entropy, and the disagreement between the
   network and the raw lifted elevation on observed cells.
   - *Recommendation:* start with the free signals. Retrain only if they fail to rank errors.

## Step 3 — Phase 4: the persistent BEV as the RL observation

Same interface, now used by a driving policy.

1. **Observation.** Channels of the persistent BEV: elevation (relative to the rover), rock probability, provenance
   (current, remembered or unknown) and reliability, all from the frozen Phase-1 model and the memory in the loop.
   Unknown cells get an explicit mask and are never filled with zeros.
2. **Distillation first.** Distil the existing height-map PPO teacher into a student that sees the persistent BEV and
   the goal. This reuses the RGB-student distillation pipeline, with the BEV encoder replacing the DINOv3 pooling
   encoder. It is the fastest end-to-end test of the whole stack.
   - Compare with the existing RGB student and the teacher on success rate, collisions and path length, with exact
     poses and at 1× and 4× drift.
3. **Then PPO fine-tuning** of the student on the persistent BEV, if distillation leaves a gap to the teacher.
4. **Ablations that answer the research question.** Instantaneous BEV only, persistent BEV without provenance, and
   persistent BEV with provenance and reliability. This shows whether knowing where the terrain information came from
   changes driving under drift.

Cost to watch: the Phase-1 model then runs inside the RL step, which matches today's evaluation of about 0.2 s per
step for 64 environments. That is fine for distillation but slow for long PPO runs. A geometry-only memory (no DINOv3)
is the cheap fallback, since Phase 1 showed geometry carries the elevation.

## Step 4 — Towards the real rover (in parallel, needs hardware time)

1. **Record ZED X sequences** with RGB, depth, IMU and visual-inertial odometry poses over Mars-like ground. Run the
   frozen model and the memory offline.
   - Compare the real depth noise with `StereoDepthCfg` (disparity noise, holes) and recalibrate it.
   - Measure the real odometry drift, to choose the severity used in Steps 1–3.
2. **Check the appearance gap.** Look at DINOv3 class predictions on real frames. Fine-tune only the adapter on a small
   labelled set, if rocks are missed.

## Supporting work

- **Terrain semantics** ([spec](../specs/terrain_semantics.md)): registry-level classes, visible lunar rocks and USD
  labels. Needed before training on lunar terrains or adding classes; can go to another agent.
- **Longer and broader Phase-1 training.** The long run was still improving when it stopped. Run it to 1000–3000
  iterations on several terrains, as a detached job, before the RL work depends on it.

## Not recommended yet

- **Learned memory (GRU/ConvGRU), storing latent DINOv3 features, BEV completion.** The measured failures are drift and
  class fusion, which a learned memory would not fix by itself. Revisit after Step 2 has shown what reliability signals
  exist.
- **Retraining Phase 1 on noisy depth.** Wait until Step 1.1 has shown whether noise actually hurts the memory.

## Decisions for you

1. Store raw lifted elevation or network elevation? This depends on Step 1.1.
2. For perception uncertainty: free signals first, or add the log σ head now (retrains Phase 1)?
3. For Phase 4: distillation first (recommended) or straight to PPO?
4. Hardware: when can ZED X sequences be recorded for Step 4?
