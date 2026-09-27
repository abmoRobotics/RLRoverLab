# Phase 2 — Temporal BEV Memory

## Context

Phase 1 is complete and should remain unchanged:

```text
RGB-D
 → DINO + geometric lift
 → BEV U-Net
 → elevation [B,120,120]
 → semantic logits [B,2,120,120]
```

The instantaneous BEV covers a 6 × 6 m area at 5 cm resolution. Only currently visible cells were supervised during Phase 1.

The purpose of Phase 2 is **only to retain terrain that was previously observed but is no longer visible**.

Do not retrain or modify the Phase-1 network.

## Desired architecture

```text
Frozen instantaneous BEV
        ↓
current elevation + semantics + observation evidence
        ↓
world-aligned temporal memory
        ↓
persistent ego-centric BEV
        ├─ elevation
        ├─ semantics
        ├─ current / remembered / unknown
        ├─ age since observation
        └─ distance travelled since observation
```

### Core requirements

1. **Keep memory in a world-aligned frame.**  
   Avoid recursively warping an ego-centric memory every timestep.

2. **Only store actual observational evidence.**  
   Determine observation evidence from information available on the robot, preferably the existing depth/lift information. Do not use simulator GT visibility for the memory itself.

3. **Do not store hallucinated BEV predictions.**  
   If a cell has no supporting current observation, the Phase-1 prediction for that cell must not become new memory.

4. **Initially use simple deterministic fusion.**  
   Start with the latest valid observation replacing the previous one. Do not introduce learned fusion, GRUs, uncertainty weighting, etc.

5. **Use the repository's existing coordinate/frame conventions.**  
   Inspect the implementation first rather than assuming axis ordering, height reference, camera frame, pose timing or binning conventions.

6. **Support batched environments and resets.**

A world memory around **12 × 12 m at 5 cm resolution** is a reasonable starting point, but adapt this if the existing repo suggests a cleaner implementation. The purpose is simply to retain terrain around the rover while the output remains 6 × 6 m.

## Stage 1 — Inspect and implement the basic memory

First inspect:

- how BEV cell coordinates are defined;
- what height reference is used;
- which pose corresponds to each rendered observation;
- how depth/lift evidence can be identified;
- how resets work.

Then implement the simplest world-aligned memory consistent with the repo.

It should be possible to:

```text
current BEV + pose
        ↓ write
world memory
        ↓ read at new pose
persistent 120×120 BEV
```

Start with **perfect simulator poses**.

Add focused tests demonstrating:

- write/read at the same pose;
- translation;
- rotation;
- correct reset/isolation between environments;
- remembered terrain remains spatially aligned as the rover moves.

Do not over-engineer the memory implementation.

## Stage 2 — Evaluate temporal memory

Run the frozen Phase-1 model during simulated driving.

Separate cells into:

```text
CURRENT
currently observable

REMEMBERED
observed previously but not now

UNKNOWN
never observed
```

Evaluate at minimum:

- elevation MAE;
- rock IoU;
- coverage;
- performance versus distance since last observation.

Ground-truth visibility/history may be used **only to calculate these evaluation groups**, not as an input to memory.

The critical comparison is:

```text
instantaneous BEV
vs
persistent BEV
```

particularly on cells that were seen previously but are now outside the camera view or occluded.

Add visualizations showing instantaneous BEV, persistent BEV and current/remembered/unknown status.

## Stage 3 — Stress the memory

Only once the basic memory works correctly, test:

- different driving patterns, including turns and revisiting terrain;
- imperfect pose estimates / odometry drift;
- alternative simple write rules if repeated observations become relevant;
- network elevation versus raw lifted elevation if useful.

Do **not** add learned memory yet.

The purpose of this stage is to determine where simple geometric memory fails.

## Later

Not part of this implementation:

- learned memory;
- GRU/ConvGRU;
- storing latent DINO features;
- uncertainty-aware fusion;
- depth-noise retraining;
- BEV completion;
- RL integration.

Those should only be considered after the deterministic memory baseline is understood.

## Important design principle

Preserve the explicit distinction between:

- **current**
- **remembered**
- **never observed / unknown**

This provenance should remain available in the persistent BEV because later reliability-aware RL will depend on knowing where each piece of terrain information came from.
