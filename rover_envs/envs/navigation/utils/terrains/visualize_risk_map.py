#!/usr/bin/env python3
"""Export static debug images for terrain obstacle risk maps."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_mesh(usd_path: str) -> tuple[np.ndarray, np.ndarray]:
    import pymeshlab
    from rover_envs.envs.navigation.utils.terrains.usd_utils import (
        get_triangles_and_vertices_from_prim_standalone,
    )

    faces, vertices = get_triangles_and_vertices_from_prim_standalone(usd_path)
    mesh = pymeshlab.Mesh(vertices, faces)
    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(mesh)
    processed = meshset.current_mesh()
    return processed.vertex_matrix().astype("float32"), processed.face_matrix().astype("uint32")


def _save_map(path: Path, data: np.ndarray, title: str, cmap: str, *, vmin=None, vmax=None) -> None:
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    image = ax.imshow(data, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_overlay(
    path: Path,
    heightmap: np.ndarray,
    obstacle_map: np.ndarray,
    cost_map: np.ndarray,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    ax.imshow(heightmap, cmap="terrain", origin="lower", alpha=0.9)

    risk_overlay = np.ma.masked_where(cost_map <= 0.01, cost_map)
    ax.imshow(risk_overlay, cmap="magma", origin="lower", alpha=0.55, vmin=0.0, vmax=1.0)

    rock_overlay = np.ma.masked_where(obstacle_map == 0, obstacle_map)
    ax.imshow(rock_overlay, cmap="Reds", origin="lower", alpha=0.95, vmin=0, vmax=1)

    ax.set_title(title)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export obstacle risk-map debug images for a registered terrain.")
    parser.add_argument("--terrain", type=str, default="mars", help="Registered terrain name.")
    parser.add_argument("--output", type=str, default="logs/risk_map_debug", help="Output directory.")
    parser.add_argument("--resolution", type=float, default=0.05, help="Grid resolution in meters.")
    parser.add_argument("--decay-distance", type=float, default=1.5, help="Risk decay distance L in meters.")
    parser.add_argument("--decay-scale", type=float, default=4.6, help="Risk decay scale.")
    parser.add_argument("--decay-exponent", type=float, default=2.5, help="Risk decay exponent.")
    parser.add_argument("--min-obstacle-area", type=float, default=0.0, help="Minimum connected rock area in m^2.")
    parser.add_argument("--min-obstacle-diameter", type=float, default=0.0, help="Minimum connected rock diameter in meters.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output) / args.terrain / f"L{args.decay_distance:g}"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from rover_envs.assets.terrains import get_terrain
        from rover_envs.envs.navigation.utils.terrains.risk_map import (
            ObstacleRiskMap,
            ObstacleRiskMapCfg,
        )
        from rover_envs.envs.navigation.utils.terrains.terrain_utils import HeightmapManager
    except ImportError as exc:
        raise SystemExit(
            f"Missing terrain debug dependency: {exc}\n"
            "Run this script inside the same environment/container used for Isaac Lab training."
        ) from exc

    terrain = get_terrain(args.terrain)
    terrain_vertices, terrain_faces = _load_mesh(terrain.files.terrain_only)
    rock_vertices, rock_faces = _load_mesh(terrain.files.rocks_merged)

    combined_vertices = np.vstack([terrain_vertices, rock_vertices])
    combined_faces = np.vstack([terrain_faces, rock_faces + len(terrain_vertices)])
    heightmap_manager = HeightmapManager(args.resolution, combined_vertices, combined_faces, device="cpu")

    bounds = (
        heightmap_manager.min_x,
        heightmap_manager.min_y,
        heightmap_manager.max_x,
        heightmap_manager.max_y,
    )
    risk_cfg = ObstacleRiskMapCfg(
        decay_distance_m=args.decay_distance,
        decay_scale=args.decay_scale,
        decay_exponent=args.decay_exponent,
        min_obstacle_area_m2=args.min_obstacle_area,
        min_obstacle_diameter_m=args.min_obstacle_diameter,
    )
    risk_map = ObstacleRiskMap.from_rock_mesh(
        rock_vertices=rock_vertices,
        rock_faces=rock_faces,
        heightmap_shape=heightmap_manager.heightmap.shape,
        bounds=bounds,
        device="cpu",
        cfg=risk_cfg,
    )

    _save_map(output_dir / "heightmap.png", heightmap_manager.heightmap, "Combined terrain heightmap", "terrain")
    _save_map(output_dir / "rock_binary.png", risk_map.obstacle_map, "Projected rock cells", "gray_r", vmin=0, vmax=1)
    _save_map(output_dir / "risk_cost.png", risk_map.cost_map, "Obstacle risk cost C(d)", "magma", vmin=0, vmax=1)
    distance_map = np.where(np.isfinite(risk_map.distance_map), risk_map.distance_map, np.nan)
    _save_map(output_dir / "distance_to_rock.png", distance_map, "Distance to nearest rock in meters", "viridis")
    _save_overlay(
        output_dir / "risk_overlay.png",
        heightmap_manager.heightmap,
        risk_map.obstacle_map,
        risk_map.cost_map,
        f"{args.terrain} obstacle risk overlay",
    )

    stats = {
        "terrain": args.terrain,
        "terrain_only_usd": terrain.files.terrain_only,
        "rocks_merged_usd": terrain.files.rocks_merged,
        "bounds": {
            "min_x": float(bounds[0]),
            "min_y": float(bounds[1]),
            "max_x": float(bounds[2]),
            "max_y": float(bounds[3]),
        },
        "risk_map": risk_map.statistics(),
    }
    with open(output_dir / "stats.json", "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    print(f"Wrote risk-map debug outputs to: {output_dir}")
    print(json.dumps(stats["risk_map"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
