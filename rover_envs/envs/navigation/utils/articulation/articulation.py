import re

from isaacsim.core.utils.stage import get_current_stage
from pxr import PhysxSchema, Sdf, Usd, UsdGeom


def prepare_rover_contact_sensors() -> None:
    """Attach rover contact report pairs for all matching prims in the stage."""
    stage = get_current_stage()
    pattern = "/World/envs/env_.*/Robot/.*(Drive|Steer|Boogie|Bogie|Body)$"
    matching_prims = []
    prim: Usd.Prim
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Xform):
            prim_path: Sdf.Path = prim.GetPath()
            if re.match(pattern, prim_path.pathString):
                matching_prims.append(prim_path)

    for prim in matching_prims:
        contact_api: PhysxSchema.PhysxContactReportAPI = PhysxSchema.PhysxContactReportAPI.Get(stage, prim)
        contact_api.CreateReportPairsRel().AddTarget("/World/terrain/obstacles/obstacles")


def prepare_franka_contact_sensors() -> None:
    """Attach Franka contact report pairs for all matching prims in the stage."""
    stage = get_current_stage()
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
