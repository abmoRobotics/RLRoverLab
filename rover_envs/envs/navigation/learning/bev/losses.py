"""BEV training loss and evaluation metrics, restricted to the camera-visible target cells.

Targets are the BEV task's ``bev_target`` observation [B, 3, X, Y]: the elevation in metres, the class index and the
camera visibility (see ``mdp.bev_ground_truth``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bev_loss(
    elevation: torch.Tensor, semantics: torch.Tensor, target: torch.Tensor, semantic_weight: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the total loss, the elevation Huber loss and the class cross-entropy over the visible cells."""
    visible = target[:, 2] > 0.5
    elevation_loss = F.smooth_l1_loss(elevation[visible], target[:, 0][visible], beta=0.05)
    semantic_loss = F.cross_entropy(semantics.permute(0, 2, 3, 1)[visible], target[:, 1][visible].long())
    return elevation_loss + semantic_weight * semantic_loss, elevation_loss, semantic_loss


@torch.no_grad()
def bev_statistics(
    elevation: torch.Tensor, semantics: torch.Tensor, lifted: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Per-batch sums over the visible cells; add them up over batches and pass them to ``bev_metrics``."""
    visible = target[:, 2] > 0.5
    # Visible cells that received depth points: where the lifted mean elevation is defined.
    observed = visible & (lifted[:, 2] > 0)
    predicted, truth = semantics.argmax(dim=1)[visible], target[:, 1][visible].long()
    classes = torch.arange(semantics.shape[1], device=truth.device).unsqueeze(1)
    return {
        "visible": visible.sum(),
        "observed": observed.sum(),
        "elevation_error": (elevation - target[:, 0]).abs()[visible].sum(),
        "lift_error": (lifted[:, 0] - target[:, 0]).abs()[observed].sum(),
        "correct": (predicted == truth).sum(),
        "intersection": ((predicted == classes) & (truth == classes)).sum(dim=1),
        "union": ((predicted == classes) | (truth == classes)).sum(dim=1),
    }


def bev_metrics(statistics: dict[str, torch.Tensor], classes: tuple[str, ...]) -> dict[str, float]:
    """Metrics from summed ``bev_statistics``; errors in metres, undefined ratios are NaN.

    ``lift_elevation_mae`` scores the depth lift alone (its mean point elevation) on the cells it observes, which
    ``lift_coverage`` is the visible fraction of.
    """
    s = {key: value.double() for key, value in statistics.items()}
    iou = s["intersection"] / s["union"]
    metrics = {
        "elevation_mae": s["elevation_error"] / s["visible"],
        "lift_elevation_mae": s["lift_error"] / s["observed"],
        "lift_coverage": s["observed"] / s["visible"],
        "semantic_accuracy": s["correct"] / s["visible"],
        **{f"iou_{name}": iou[k] for k, name in enumerate(classes)},
    }
    return {key: value.item() for key, value in metrics.items()}
