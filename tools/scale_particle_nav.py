#!/usr/bin/env python3
"""Measure how the MPM soil navigation task scales with the number of rovers.

Runs ``tools/benchmark_particle_nav.py`` headless ``--repeats`` times per rover count, each run
in its own process, and writes the per-run JSON and logs plus a table of median results to
``--output_dir``. Arguments after ``--`` go to every run, for example ``-- --physics_dt 0.0166667``.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, nargs="+", default=[2, 8, 32, 64, 128])
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--output_dir", type=Path, default=Path("/tmp/particle_nav_scaling"))
args, benchmark_args = parser.parse_known_args()
if benchmark_args[:1] == ["--"]:
    benchmark_args = benchmark_args[1:]


def combine(values: list):
    """Median of numeric results, ``all`` of flags, and the first run's value otherwise."""
    if all(isinstance(value, bool) for value in values):
        return all(values)
    numbers = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not numbers:
        return values[0]
    median = statistics.median(numbers)
    return int(median) if all(isinstance(number, int) for number in numbers) and median == int(median) else median


args.output_dir.mkdir(parents=True, exist_ok=True)
benchmark = Path(__file__).with_name("benchmark_particle_nav.py")
results = []
for num_envs in args.num_envs:
    runs = []
    for repeat in range(1, args.repeats + 1):
        stem = f"envs_{num_envs}_run_{repeat}"
        result_path = args.output_dir / f"{stem}.json"
        log_path = args.output_dir / f"{stem}.log"
        command = [
            sys.executable,
            str(benchmark),
            "--num_envs", str(num_envs),
            "--warmup", str(args.warmup),
            "--steps", str(args.steps),
            "--device", args.device,
            "--viz", "none",
            "--output", str(result_path),
            *benchmark_args,
        ]
        print(f"Running {stem}; log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        runs.append(json.loads(result_path.read_text(encoding="utf-8")))
    results.append({"median": {key: combine([run[key] for run in runs]) for key in runs[0]}, "runs": runs})

columns = [
    ("Rovers", "num_envs", "{}"),
    ("Env steps/s", "env_steps_per_s", "{:.1f}"),
    ("Policy steps/s", "policy_steps_per_s", "{:.2f}"),
    ("Real-time factor", "real_time_factor", "{:.2f}"),
    ("GPU memory [GB]", "gpu_mem_used_gb", "{:.1f}"),
    ("Startup [s]", "startup_s", "{:.0f}"),
    ("Targets reached", "targets_reached", "{}"),
    ("Mean speed [m/s]", "mean_forward_speed_m_s", "{:.2f}"),
]
table = [
    "| " + " | ".join(name for name, _, _ in columns) + " |",
    "|" + "---:|" * len(columns),
    *("| " + " | ".join(fmt.format(result["median"][key]) for _, key, fmt in columns) + " |" for result in results),
]
summary = {"repeats": args.repeats, "results": results, "table": table}
(args.output_dir / "scaling.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("\n".join(table))
print(f"Saved runs and logs to {args.output_dir}")
