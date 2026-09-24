#!/usr/bin/env python3
"""Run the AAU particle smoke benchmark in separate backend processes."""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--voxel_size", type=float, default=0.08)
parser.add_argument("--particles_per_cell", type=int, default=2)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--output_dir", type=Path, default=Path("/tmp/aau_particle_benchmark"))
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
benchmark = Path(__file__).with_name("debug_aau_particles.py")
runs = {"physx": [], "newton": []}

for repeat in range(args.repeats):
    for backend in (("physx", "newton") if repeat % 2 == 0 else ("newton", "physx")):
        stem = f"{backend}_{repeat + 1}"
        result_path = args.output_dir / f"{stem}.json"
        log_path = args.output_dir / f"{stem}.log"
        command = [
            sys.executable,
            str(benchmark),
            "--backend", backend,
            "--num_envs", str(args.num_envs),
            "--voxel_size", str(args.voxel_size),
            "--particles_per_cell", str(args.particles_per_cell),
            "--warmup", str(args.warmup),
            "--steps", str(args.steps),
            "--device", args.device,
            "--viz", "none",
            "--output", str(result_path),
        ]
        print(f"Running {stem}; log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        runs[backend].append(json.loads(result_path.read_text(encoding="utf-8")))

physx_rate = statistics.median(run["env_steps_per_s"] for run in runs["physx"])
newton_rate = statistics.median(run["env_steps_per_s"] for run in runs["newton"])
comparison = {
    "num_envs": args.num_envs,
    "particles_per_env": runs["physx"][0]["particles_per_env"],
    "median_env_steps_per_s": {"physx": physx_rate, "newton": newton_rate},
    "newton_over_physx": newton_rate / physx_rate,
    "runs": runs,
}
summary_path = args.output_dir / "comparison.json"
summary_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in comparison.items() if key != "runs"}, indent=2))
print(f"Saved runs and logs to {args.output_dir}")
