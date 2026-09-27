# Phase 1 — instantaneous BEV input ablations

Date: 26 September 2026. Decision: keep the full model (geometry + DINOv3) as the Phase-1 architecture.

## Setup

- **Tasks:** `AAURoverEnvSimpleBEV-v0` (full model) and the three ablations `AAURoverEnvSimpleBEVGeometryOnly-v0`,
  `AAURoverEnvSimpleBEVDinoOnly-v0` and `AAURoverEnvSimpleBEVNoisyDepth-v0`.
- **Training:** 64 environments on `mars`, 120 iterations each (about 122,000 frames), seed 42, otherwise the default
  `RoverBevTrainerCfg`.
- **Metrics:** computed on each frame's first use, before the model trains on it, over camera-visible cells only (see
  `losses.py`). All runs train and are scored on the same map, so this is not a test on unseen terrain.
- **Statistics:** median over the last 40 iterations, with the interquartile range of iterations in brackets.

| Run | U-Net input | Elevation MAE | Rock IoU | Learning time per iteration |
|---|---|---|---|---|
| Full | Geometry + DINOv3, perfect depth | **1.53 cm** [1.26–1.92] | **0.882** [0.864–0.904] | 11.6 s |
| Geometry only | Lifted geometry, perfect depth; no DINOv3 | 1.58 cm [1.35–1.90] | 0.872 [0.858–0.894] | 5.7 s |
| DINOv3 only | Lifted DINOv3 features; depth still places them | 7.67 cm [7.42–8.03] | 0.825 [0.812–0.841] | 11.6 s |
| Noisy depth | Geometry + DINOv3 from stereo-like depth* | 1.55 cm [1.37–1.88] | 0.853 [0.836–0.868] | 11.7 s |

\* Disparity noise of 0.2 px correlated over 9×9 pixels (about 2.5 cm depth error at 2 m and 10 cm at 4 m) and 5%
of the image as holes (`StereoDepthCfg`). Its depth lift alone scores 1.16 cm elevation MAE on observed cells, against
0.43 cm with perfect depth.

## Findings

1. **Perfect depth solves nearly all of the elevation.** The full and geometry-only runs saw exactly the same frames
   (identical depth-lift error in all 120 iterations), so they can be compared iteration by iteration. Elevation is a
   tie: full is 0.10 cm better on average and wins 23 of 40 iterations. Geometry-only trains twice as fast.
2. **DINOv3 adds a small but consistent gain on rocks.** In the same paired comparison, full beats geometry-only on rock
   IoU in 40 of 40 iterations, by 0.010 on average.
3. **Appearance alone cannot give metric height.** Without the geometry channels the elevation error is five times
   larger (7.7 cm), although rocks are still found reasonably well (IoU 0.825).
4. **Stereo-like depth noise barely hurts elevation but costs rock IoU.** Elevation is unchanged (1.55 cm), while rock
   IoU drops by about 0.03.

## Caveats

- One seed and only 120 iterations. The longer run of the full model (`logs/bev/rover_bev/2026-09-26_18-40-41`,
  stopped at iteration 294) kept improving: over its last 40 iterations it reached 1.16 cm elevation MAE
  [0.95–1.38] and rock IoU 0.895 [0.884–0.903]. Its checkpoint `model_249.pt` is the frozen Phase-1 model used for
  the Phase-2 memory evaluation.
- Only the full vs geometry-only comparison is paired. The DINOv3-only and noisy-depth runs saw different frames.
- Differences of a few tenths of a centimetre in elevation MAE are within the iteration-to-iteration noise.

## Where the runs are

- Run folders: `logs/bev/<experiment>/<timestamp>/`, with checkpoints and BEV visualizations. The full run is
  `logs/bev/rover_bev/2026-09-26_20-17-25`.
- Console logs: `logs/bev_ablation_{Full,GeometryOnly,DinoOnly,NoisyDepth}.log`.
- W&B project `rlroverlab`: runs `rover_bev_*`.
