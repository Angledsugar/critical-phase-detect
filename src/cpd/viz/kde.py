"""KDE visualisation. Paper Fig 3 — kernel-weighted f̃_± heatmap."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure


def _gaussian_density(
    points: torch.Tensor, grid: torch.Tensor, h: float
) -> torch.Tensor:
    """Evaluate Gaussian KDE of `points` (N, 2) on `grid` (G, 2)."""
    if points.shape[0] == 0:
        return torch.zeros(grid.shape[0])
    diff = grid.unsqueeze(1) - points.unsqueeze(0)  # (G, N, 2)
    sq = (diff**2).sum(dim=-1)  # (G, N)
    log_kernel = -0.5 * sq / (h**2)
    return torch.logsumexp(log_kernel, dim=1).exp() / points.shape[0]


def plot_kde_2d(
    kde_stats: object,
    axis_dims: tuple[int, int] = (0, 1),
    *,
    grid_size: int = 80,
) -> Figure:
    """Overlay filled contour of f̃_+ and f̃_- on two latent dimensions.

    `kde_stats` must expose attributes:
        h: float                       (bandwidth)
        latents_pos: torch.Tensor (N+, d)
        latents_neg: torch.Tensor (N-, d)

    Returns a matplotlib Figure.
    """
    h = float(kde_stats.h)  # type: ignore[attr-defined]
    latents_pos = kde_stats.latents_pos  # type: ignore[attr-defined]
    latents_neg = kde_stats.latents_neg  # type: ignore[attr-defined]
    i, j = axis_dims

    pos_2d = (
        latents_pos[:, [i, j]].detach().cpu()
        if latents_pos.numel() > 0
        else torch.empty(0, 2)
    )
    neg_2d = (
        latents_neg[:, [i, j]].detach().cpu()
        if latents_neg.numel() > 0
        else torch.empty(0, 2)
    )

    # Determine joint bounding box, with padding to avoid contour edge clipping.
    all_pts = torch.cat([pos_2d, neg_2d], dim=0)
    if all_pts.numel() == 0:
        # Empty input: draw an empty axes with informative title.
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_title(f"KDE 2D (h={h:.3g}) — empty buffers")
        return fig
    lo = all_pts.min(dim=0).values - max(h, 1e-6)
    hi = all_pts.max(dim=0).values + max(h, 1e-6)
    xs = torch.linspace(lo[0].item(), hi[0].item(), grid_size)
    ys = torch.linspace(lo[1].item(), hi[1].item(), grid_size)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1)

    f_pos = _gaussian_density(pos_2d, grid, h).reshape(grid_size, grid_size).numpy()
    f_neg = _gaussian_density(neg_2d, grid, h).reshape(grid_size, grid_size).numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    Xn, Yn = X.numpy(), Y.numpy()
    if pos_2d.numel() > 0:
        ax.contourf(Xn, Yn, f_pos, levels=10, cmap="Blues", alpha=0.6)
        ax.scatter(
            pos_2d[:, 0].numpy(),
            pos_2d[:, 1].numpy(),
            s=8,
            color="navy",
            label="positive",
        )
    if neg_2d.numel() > 0:
        ax.contourf(Xn, Yn, f_neg, levels=10, cmap="Reds", alpha=0.4)
        ax.scatter(
            neg_2d[:, 0].numpy(),
            neg_2d[:, 1].numpy(),
            s=8,
            color="darkred",
            marker="x",
            label="negative",
        )
    ax.set_xlabel(f"latent dim {i}")
    ax.set_ylabel(f"latent dim {j}")
    ax.set_title(f"Kernel-weighted f̃_+ / f̃_- (h={h:.3g})")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# Convenience for downstream callers; keep np re-export silent for ruff.
__all__ = ["plot_kde_2d"]
_ = np
