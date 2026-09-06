#!/usr/bin/env python3
"""Report and verify the pinned RLRoverLab simulation stack."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path


EXPECTED_ISAAC_SIM_VERSION = "6.0.1"
EXPECTED_ISAAC_LAB_REF = "v3.0.0-beta2.patch1"
EXPECTED_ISAAC_LAB_SHA = "ffff603eafc6b74264a5261cc0183d6a65390d78"


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    except OSError as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-pip-check",
        action="store_true",
        help="Fail when pip reports third-party dependency conflicts.",
    )
    parser.add_argument("--skip-pip-check", action="store_true", help="Do not run pip check.")
    args = parser.parse_args()

    isaac_sim_path = Path(os.environ.get("ISAAC_SIM_PATH", "/isaac-sim"))
    isaac_lab_path = Path(os.environ.get("ISAAC_LAB_PATH", "/workspace/isaac_lab"))
    errors: list[str] = []

    version_file = isaac_sim_path / "VERSION"
    sim_build_version = version_file.read_text(encoding="utf-8").strip() if version_file.is_file() else None
    sim_package_version = _package_version("isaacsim")
    detected_sim_version = sim_build_version or sim_package_version
    if detected_sim_version is None or not detected_sim_version.startswith(EXPECTED_ISAAC_SIM_VERSION):
        errors.append(
            f"Isaac Sim version must start with {EXPECTED_ISAAC_SIM_VERSION!r}; found {detected_sim_version!r}."
        )

    if sys.version_info[:2] != (3, 12):
        errors.append(f"Python must be 3.12.x; found {platform.python_version()}.")

    lab_sha_result = _run(["git", "rev-parse", "HEAD"], cwd=isaac_lab_path)
    lab_sha = lab_sha_result.stdout.strip() if lab_sha_result.returncode == 0 else None
    if lab_sha != EXPECTED_ISAAC_LAB_SHA:
        errors.append(f"Isaac Lab must be {EXPECTED_ISAAC_LAB_SHA}; found {lab_sha!r}.")

    tags_result = _run(["git", "tag", "--points-at", "HEAD"], cwd=isaac_lab_path)
    lab_tags = sorted(tags_result.stdout.split()) if tags_result.returncode == 0 else []
    if EXPECTED_ISAAC_LAB_REF not in lab_tags:
        errors.append(f"Isaac Lab HEAD is not tagged {EXPECTED_ISAAC_LAB_REF!r}; tags: {lab_tags!r}.")

    gpu_result = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    gpu = gpu_result.stdout.strip().splitlines() if gpu_result.returncode == 0 else []

    pip_check_exit_code = None
    pip_check_output = None
    if not args.skip_pip_check:
        pip_result = _run([sys.executable, "-m", "pip", "check"])
        pip_check_exit_code = pip_result.returncode
        pip_check_output = (pip_result.stdout + pip_result.stderr).strip()
        if args.strict_pip_check and pip_result.returncode != 0:
            errors.append("pip check reported dependency conflicts.")

    packages = {
        name: _package_version(name)
        for name in (
            "isaaclab",
            "isaaclab_physx",
            "isaaclab_rl",
            "isaaclab_tasks",
            "gymnasium",
            "numpy",
            "skrl",
            "torch",
            "torchvision",
        )
    }

    try:
        import torch

        cuda_version = torch.version.cuda
    except ImportError:
        cuda_version = None

    report = {
        "status": "pass" if not errors else "fail",
        "expected": {
            "isaac_sim": EXPECTED_ISAAC_SIM_VERSION,
            "isaac_lab_ref": EXPECTED_ISAAC_LAB_REF,
            "isaac_lab_sha": EXPECTED_ISAAC_LAB_SHA,
            "python": "3.12.x",
        },
        "actual": {
            "isaac_sim_build": sim_build_version,
            "isaac_sim_package": sim_package_version,
            "isaac_lab_sha": lab_sha,
            "isaac_lab_tags": lab_tags,
            "python": platform.python_version(),
            "packages": packages,
            "cuda": cuda_version,
            "gpu": gpu,
        },
        "pip_check": {
            "exit_code": pip_check_exit_code,
            "output": pip_check_output,
        },
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
