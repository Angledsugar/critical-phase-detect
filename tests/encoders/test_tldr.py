"""TLDR encoder + triplet trainer — shape, learning, save/load."""
from __future__ import annotations

import numpy as np
import torch

from cpd.encoders.tldr import TLDREncoder
from cpd.encoders.tldr_train import TLDRTrainer


class _SimpleTraj:
    def __init__(self, raw_states: np.ndarray) -> None:
        self.raw_states = raw_states


def _make_synthetic(
    num_trajectories: int = 8,
    traj_length: int = 80,
    state_dim: int = 8,
    drift: float = 0.1,
    noise: float = 0.05,
    seed: int = 0,
) -> list[_SimpleTraj]:
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(num_trajectories):
        direction = rng.standard_normal(state_dim)
        direction /= np.linalg.norm(direction) + 1e-8
        s0 = rng.standard_normal(state_dim) * 0.5
        states = np.empty((traj_length, state_dim), dtype=np.float32)
        states[0] = s0
        for t in range(1, traj_length):
            states[t] = states[t - 1] + drift * direction + rng.standard_normal(state_dim) * noise
        trajs.append(_SimpleTraj(states))
    return trajs


def test_encode_unbatched_shape():
    enc = TLDREncoder(state_dim=10, latent_dim=4)
    z = enc.encode(torch.randn(10))
    assert z.shape == (4,)


def test_encode_batched_shape():
    enc = TLDREncoder(state_dim=10, latent_dim=4)
    z = enc.encode(torch.randn(7, 10))
    assert z.shape == (7, 4)


def test_encode_accepts_numpy_and_list():
    enc = TLDREncoder(state_dim=5, latent_dim=3)
    z_np = enc.encode(np.zeros(5, dtype=np.float32))
    z_list = enc.encode([0.0] * 5)
    assert z_np.shape == (3,)
    assert z_list.shape == (3,)
    # both routes should produce the same encoding for the same input.
    assert torch.allclose(z_np, z_list, atol=1e-6)


def test_latent_dim_property():
    enc = TLDREncoder(state_dim=4, latent_dim=16, hidden_dim=32, num_layers=2)
    assert enc.latent_dim == 16


def test_triplet_loss_decreases():
    torch.manual_seed(0)
    enc = TLDREncoder(state_dim=8, latent_dim=8, hidden_dim=64, num_layers=2)
    trainer = TLDRTrainer(
        enc, lr=3e-3, k_pos=2, K_neg=20, margin=1.0, batch_size=128
    )
    trajs = _make_synthetic(seed=0)

    initial = trainer.train_epoch(trajs, num_batches=1, seed=42)["loss"]

    losses = []
    for ep in range(15):
        m = trainer.train_epoch(trajs, num_batches=10, seed=100 + ep)
        losses.append(m["loss"])
    # final-window mean should be well below the initial single-batch loss
    final = sum(losses[-3:]) / 3
    assert final < initial * 0.7, (
        f"loss did not decrease: initial={initial:.4f}, final={final:.4f}"
    )


def test_save_load_roundtrip(tmp_path):
    enc = TLDREncoder(state_dim=6, latent_dim=4, hidden_dim=16, num_layers=2)
    enc.eval()
    x = torch.randn(3, 6)
    with torch.no_grad():
        z_before = enc.encode(x)

    path = tmp_path / "tldr.pt"
    enc.save(path)
    enc2 = TLDREncoder.load(path)
    enc2.eval()
    with torch.no_grad():
        z_after = enc2.encode(x)

    assert enc2.latent_dim == enc.latent_dim
    assert torch.allclose(z_before, z_after, atol=1e-6)


def test_encoder_conforms_to_phi_protocol():
    from cpd.encoders.base import PhiEncoder

    enc = TLDREncoder(state_dim=4, latent_dim=2)
    # runtime structural conformance — Protocol checks attributes exist.
    assert isinstance(enc.latent_dim, int)
    assert callable(getattr(enc, "encode", None))
    # also passes Protocol structural matching (Protocol with @runtime_checkable
    # would assert directly; PhiEncoder isn't, so we duck-type).
    _ = PhiEncoder  # imported for symmetry
