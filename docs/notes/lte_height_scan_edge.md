# Known issue: height scan at the edge of LTE terrains

Height-scanner rays that miss the terrain return `inf`, so `height_scan_rover`
yields `-inf` for them. On the LTE packages (tested on `lunar_lte100` and
`lunar_lte500`) the 5 x 5 m scan is complete from 3 m inside the terrain edge;
at 2 m about 10% of rays miss, at 1 m about 30%.

Spawns (>= 10 m from the edge), goals (>= 8 m) and the far-from-target
termination (12 m) keep normal episodes away from the edge: 2 x 38,400
random-action env-steps gave no non-finite observation. A policy can still
wander into the last 3 m, which would put `-inf` into its input.

Not fixed yet. Options: clip the height-scan observation, or end episodes near
the terrain edge.
