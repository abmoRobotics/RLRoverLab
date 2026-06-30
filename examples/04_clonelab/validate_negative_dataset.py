#!/usr/bin/env python3
"""Validate an isolated negative-example HDF5 shard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rover_envs.integrations.clonelab.negative_dataset_tools import validate_negative_dataset


parser = argparse.ArgumentParser("Validate a negative-example HDF5 shard.")
parser.add_argument("dataset")
parser.add_argument("--allow_missing_negative_metadata", action="store_true", default=False)
parser.add_argument("--allow_missing_termination_metadata", action="store_true", default=False)
parser.add_argument("--allow_unverified_collision_penalty", action="store_true", default=False)
parser.add_argument("--require_timeout_penalty", action="store_true", default=False)
parser.add_argument("--terminal_reward_max", type=float, default=-1e-6)
parser.add_argument("--json_out", default=None)


def main() -> int:
    args = parser.parse_args()
    result = validate_negative_dataset(
        args.dataset,
        require_negative_metadata=not args.allow_missing_negative_metadata,
        require_termination_metadata=not args.allow_missing_termination_metadata,
        require_collision_penalty=not args.allow_unverified_collision_penalty,
        require_timeout_penalty=args.require_timeout_penalty,
        terminal_reward_max=args.terminal_reward_max,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
