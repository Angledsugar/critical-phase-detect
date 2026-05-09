"""KDE — silverman bandwidth, density vectorization, integration sanity."""
import math

import pytest
import torch

from cpd.core.buffer import TrajectoryBuffer
from cpd.core.kde import KDEStats, compute_kde, silverman_bandwidth
from cpd.core.trajectory import Trajectory


def _labeled(latents: torch.Tensor, success: bool) -> Trajectory:
    T = latents.shape[0]
    return Trajectory(
        raw_states=list(range(T)), latents=latents, actions=(), success=success
    )


def test_silverman_returns_positive_scalar():
    torch.manual_seed(0)
    h = silverman_bandwidth(torch.randn(100, 4))
    assert isinstance(h, float)
    assert h > 0.0
    assert math.isfinite(h)


def test_silverman_validates_shape_and_size():
    with pytest.raises(ValueError):
        silverman_bandwidth(torch.randn(5))  # not 2-D
    with pytest.raises(ValueError):
        silverman_bandwidth(torch.zeros(1, 3))  # N < 2


def test_density_vectorized_over_batch():
    torch.manual_seed(0)
    pos = torch.randn(20, 3)
    neg = torch.randn(20, 3)
    stats = KDEStats(h=0.3, latents_pos=pos, latents_neg=neg)
    x = torch.randn(7, 3)
    out = stats.density(x, positive=True)
    assert out.shape == (7,)
    assert torch.all(out >= 0.0)
    assert torch.all(torch.isfinite(out))
    out_neg = stats.density(x, positive=False)
    assert out_neg.shape == (7,)


def test_density_single_point():
    pos = torch.randn(10, 2)
    stats = KDEStats(h=0.4, latents_pos=pos, latents_neg=torch.empty(0, 2))
    z = torch.zeros(2)
    val = stats.density(z, positive=True)
    assert val.shape == ()
    assert float(val.item()) >= 0.0


def test_density_empty_bucket_returns_zero():
    pos = torch.randn(5, 2)
    stats = KDEStats(h=0.3, latents_pos=pos, latents_neg=torch.empty(0, 2))
    out = stats.density(torch.randn(4, 2), positive=False)
    assert out.shape == (4,)
    assert torch.all(out == 0.0)


def test_density_integrates_to_approximately_one_on_uniform_grid():
    """A 1-D Gaussian KDE over points in [0,1] should integrate to ~1 on a fine grid
    that covers the support (extended a few bandwidths each side)."""
    torch.manual_seed(0)
    pts = torch.rand(50, 1)
    h = silverman_bandwidth(pts)
    stats = KDEStats(h=h, latents_pos=pts, latents_neg=torch.empty(0, 1))
    pad = 5.0 * h
    grid = torch.linspace(-pad, 1.0 + pad, 4000).unsqueeze(-1)
    f = stats.density(grid, positive=True)
    integral = float(torch.trapz(f, grid.squeeze(-1)).item())
    assert 0.9 <= integral <= 1.1


def test_compute_kde_from_buffer():
    torch.manual_seed(0)
    buf = TrajectoryBuffer()
    buf.add(_labeled(torch.randn(8, 3), success=True))
    buf.add(_labeled(torch.randn(7, 3), success=False))
    stats = compute_kde(buf)
    assert isinstance(stats, KDEStats)
    assert stats.h > 0.0
    assert stats.latents_pos.shape == (8, 3)
    assert stats.latents_neg.shape == (7, 3)


def test_compute_kde_empty_raises():
    buf = TrajectoryBuffer()
    with pytest.raises(ValueError):
        compute_kde(buf)
