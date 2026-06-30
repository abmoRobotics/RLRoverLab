"""Obstacle risk map utilities for rover navigation rewards."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage
import torch


@dataclass
class ObstacleRiskMapCfg:
    """Configuration for converting projected rocks into a smooth risk cost."""

    decay_distance_m: float = 1.5
    decay_scale: float = 4.6
    decay_exponent: float = 2.5
    min_obstacle_area_m2: float = 0.0
    min_obstacle_diameter_m: float = 0.0
    outside_cost: float = 1.0
    outside_distance_m: float = 0.0
    close_kernel_cells: int = 3


class ObstacleRiskMap:
    """Binary obstacle map plus distance-derived risk cost."""

    def __init__(
        self,
        obstacle_map: np.ndarray,
        bounds: Tuple[float, float, float, float],
        device: str | torch.device,
        cfg: ObstacleRiskMapCfg | None = None,
    ) -> None:
        if obstacle_map.ndim != 2:
            raise ValueError(f"Obstacle map must be 2D, got shape {obstacle_map.shape}")

        self.cfg = cfg or ObstacleRiskMapCfg()
        self.obstacle_map = (obstacle_map > 0).astype(np.uint8)
        self.min_x, self.min_y, self.max_x, self.max_y = bounds
        self.height, self.width = self.obstacle_map.shape
        self.cell_size_x = (self.max_x - self.min_x) / max(self.width - 1, 1)
        self.cell_size_y = (self.max_y - self.min_y) / max(self.height - 1, 1)
        self.device = torch.device(device)

        self.distance_map = self._compute_distance_map()
        self.cost_map = self._compute_cost_map()
        self._initialize_tensors()

    @classmethod
    def empty(
        cls,
        heightmap_shape: Tuple[int, int],
        bounds: Tuple[float, float, float, float],
        device: str | torch.device,
        cfg: ObstacleRiskMapCfg | None = None,
    ) -> "ObstacleRiskMap":
        obstacle_map = np.zeros(heightmap_shape, dtype=np.uint8)
        return cls(obstacle_map, bounds=bounds, device=device, cfg=cfg)

    @classmethod
    def from_rock_mesh(
        cls,
        rock_vertices: np.ndarray,
        rock_faces: np.ndarray,
        heightmap_shape: Tuple[int, int],
        bounds: Tuple[float, float, float, float],
        device: str | torch.device,
        cfg: ObstacleRiskMapCfg | None = None,
    ) -> "ObstacleRiskMap":
        cfg = cfg or ObstacleRiskMapCfg()
        obstacle_map = cls._project_rocks_to_grid(
            rock_vertices=rock_vertices,
            rock_faces=rock_faces,
            heightmap_shape=heightmap_shape,
            bounds=bounds,
        )
        obstacle_map = cls._clean_obstacle_map(obstacle_map, cfg)
        obstacle_map = cls._filter_obstacles_by_size(obstacle_map, bounds, cfg)
        return cls(obstacle_map, bounds=bounds, device=device, cfg=cfg)

    @staticmethod
    def _project_rocks_to_grid(
        rock_vertices: np.ndarray,
        rock_faces: np.ndarray,
        heightmap_shape: Tuple[int, int],
        bounds: Tuple[float, float, float, float],
    ) -> np.ndarray:
        height, width = heightmap_shape
        min_x, min_y, max_x, max_y = bounds
        cell_size_x = (max_x - min_x) / max(width - 1, 1)
        cell_size_y = (max_y - min_y) / max(height - 1, 1)

        obstacle_map = np.zeros((height, width), dtype=np.uint8)
        if rock_faces is None or len(rock_faces) == 0:
            return obstacle_map

        face_vertices = rock_vertices[rock_faces]
        grid_x = np.rint((face_vertices[:, :, 0] - min_x) / cell_size_x).astype(np.int32)
        grid_y = np.rint((face_vertices[:, :, 1] - min_y) / cell_size_y).astype(np.int32)
        grid_x = np.clip(grid_x, 0, width - 1)
        grid_y = np.clip(grid_y, 0, height - 1)

        triangles = np.stack((grid_x, grid_y), axis=-1)
        for triangle in triangles:
            cv2.fillPoly(obstacle_map, [triangle.reshape((-1, 1, 2))], color=1)

        return obstacle_map

    @staticmethod
    def _clean_obstacle_map(obstacle_map: np.ndarray, cfg: ObstacleRiskMapCfg) -> np.ndarray:
        cleaned = obstacle_map.astype(bool)
        if cfg.close_kernel_cells > 1:
            structure = np.ones((cfg.close_kernel_cells, cfg.close_kernel_cells), dtype=bool)
            cleaned = ndimage.binary_closing(cleaned, structure=structure)
        cleaned = ndimage.binary_fill_holes(cleaned)
        return cleaned.astype(np.uint8)

    @staticmethod
    def _filter_obstacles_by_size(
        obstacle_map: np.ndarray,
        bounds: Tuple[float, float, float, float],
        cfg: ObstacleRiskMapCfg,
    ) -> np.ndarray:
        if cfg.min_obstacle_area_m2 <= 0.0 and cfg.min_obstacle_diameter_m <= 0.0:
            return obstacle_map

        height, width = obstacle_map.shape
        min_x, min_y, max_x, max_y = bounds
        cell_size_x = (max_x - min_x) / max(width - 1, 1)
        cell_size_y = (max_y - min_y) / max(height - 1, 1)
        cell_area = cell_size_x * cell_size_y

        labels, num_labels = ndimage.label(obstacle_map > 0)
        if num_labels == 0:
            return obstacle_map

        filtered = np.zeros_like(obstacle_map, dtype=np.uint8)
        slices = ndimage.find_objects(labels)
        for label_id, label_slice in enumerate(slices, start=1):
            if label_slice is None:
                continue
            component = labels[label_slice] == label_id
            area_m2 = float(component.sum()) * cell_area
            rows = label_slice[0].stop - label_slice[0].start
            cols = label_slice[1].stop - label_slice[1].start
            diameter_m = max(rows * cell_size_y, cols * cell_size_x)

            area_ok = area_m2 >= cfg.min_obstacle_area_m2
            diameter_ok = diameter_m >= cfg.min_obstacle_diameter_m
            if area_ok and diameter_ok:
                filtered[labels == label_id] = 1

        return filtered

    def _compute_distance_map(self) -> np.ndarray:
        if not np.any(self.obstacle_map):
            return np.full_like(self.obstacle_map, np.inf, dtype=np.float32)

        free_cells = self.obstacle_map == 0
        return ndimage.distance_transform_edt(
            free_cells,
            sampling=(self.cell_size_y, self.cell_size_x),
        ).astype(np.float32)

    def _compute_cost_map(self) -> np.ndarray:
        with np.errstate(over="ignore"):
            cost = np.exp(
                -self.cfg.decay_scale
                * np.power(self.distance_map / self.cfg.decay_distance_m, self.cfg.decay_exponent)
            )
        cost[~np.isfinite(cost)] = 0.0
        return cost.astype(np.float32)

    def _initialize_tensors(self) -> None:
        self.distance_map_tensor = torch.as_tensor(self.distance_map, dtype=torch.float32, device=self.device)
        self.cost_map_tensor = torch.as_tensor(self.cost_map, dtype=torch.float32, device=self.device)
        self.offset_tensor = torch.tensor([self.min_x, self.min_y], dtype=torch.float32, device=self.device)
        self.cell_size_tensor = torch.tensor(
            [self.cell_size_x, self.cell_size_y],
            dtype=torch.float32,
            device=self.device,
        )

    def _sample_grid_tensor(self, xy: torch.Tensor, grid_tensor: torch.Tensor, outside_value: float) -> torch.Tensor:
        squeeze_result = False
        if xy.ndim == 1:
            xy = xy.unsqueeze(0)
            squeeze_result = True
        if xy.shape[-1] != 2:
            raise ValueError(f"XY tensor must have shape (..., 2), got {xy.shape}")

        original_device = xy.device
        query_xy = xy.to(grid_tensor.device)
        grid = ((query_xy - self.offset_tensor) / self.cell_size_tensor).long()
        valid = (
            (query_xy[:, 0] >= self.min_x)
            & (query_xy[:, 0] <= self.max_x)
            & (query_xy[:, 1] >= self.min_y)
            & (query_xy[:, 1] <= self.max_y)
        )

        grid[:, 0] = torch.clamp(grid[:, 0], 0, self.width - 1)
        grid[:, 1] = torch.clamp(grid[:, 1], 0, self.height - 1)
        values = grid_tensor[grid[:, 1], grid[:, 0]]
        outside_values = torch.full_like(values, float(outside_value))
        values = torch.where(valid, values, outside_values)

        if values.device != original_device:
            values = values.to(original_device)
        return values.squeeze(0) if squeeze_result else values

    def cost_at_world_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """Return risk costs for world-frame XY positions."""
        return self._sample_grid_tensor(xy, self.cost_map_tensor, self.cfg.outside_cost)

    def distance_at_world_xy(self, xy: torch.Tensor) -> torch.Tensor:
        """Return distance in meters to the nearest projected rock for world-frame XY positions."""
        return self._sample_grid_tensor(xy, self.distance_map_tensor, self.cfg.outside_distance_m)

    def statistics(self) -> dict:
        total_cells = self.height * self.width
        obstacle_cells = int(self.obstacle_map.sum())
        return {
            "shape": (self.height, self.width),
            "obstacle_cells": obstacle_cells,
            "obstacle_percentage": float(obstacle_cells / total_cells * 100.0),
            "cost_min": float(np.min(self.cost_map)),
            "cost_max": float(np.max(self.cost_map)),
            "cost_mean": float(np.mean(self.cost_map)),
            "decay_distance_m": self.cfg.decay_distance_m,
        }
