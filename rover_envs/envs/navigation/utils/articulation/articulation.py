import re
from typing import Sequence

import omni.usd
from pxr import PhysxSchema, Sdf, Usd, UsdGeom


def _get_current_stage() -> Usd.Stage:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")
    return stage


def prepare_rover_contact_sensors(report_targets: Sequence[str]) -> None:
    """Attach rover contact report pairs for all matching prims in the stage."""
    if not report_targets:
        raise ValueError("At least one rover contact report target is required.")

    stage = _get_current_stage()
    pattern = "/World/envs/env_.*/Robot/.*_(Drive|Steer|Boogie|Bogie|Body|Rocker)$"
    matching_prims = []
    prim: Usd.Prim
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xform):
            prim_path: Sdf.Path = prim.GetPath()
            if re.match(pattern, prim_path.pathString):
                matching_prims.append(prim_path)

    for prim in matching_prims:
        contact_api: PhysxSchema.PhysxContactReportAPI = PhysxSchema.PhysxContactReportAPI.Get(stage, prim)
        for report_target in report_targets:
            contact_api.CreateReportPairsRel().AddTarget(report_target)


def prepare_franka_contact_sensors() -> None:
    """Attach Franka contact report pairs for all matching prims in the stage."""
    stage = _get_current_stage()
    pattern = "/World/envs/env_.*/Robot/.*(link1|link2|link3|link4|link5|link6|link7|hand)$"
    matching_prims = []
    prim: Usd.Prim
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xform):
            prim_path: Sdf.Path = prim.GetPath()
            if re.match(pattern, prim_path.pathString):
                matching_prims.append(prim_path)

    prims_per_env = 8
    for idx, prim in enumerate(matching_prims):
        env_idx = idx // prims_per_env
        contact_api: PhysxSchema._physxSchema.PhysxContactReportAPI = PhysxSchema._physxSchema.PhysxContactReportAPI.Get(stage, prim)
        contact_api.CreateReportPairsRel().AddTarget(f"/World/envs/env_{env_idx}/Table/Collisions/Cube")
