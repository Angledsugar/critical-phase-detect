"""IdentityEncoder — shape, dtype, batch handling."""
import pytest
import torch

from cpd.encoders.identity import IdentityEncoder


def test_encode_single_state_shape():
    enc = IdentityEncoder(latent_dim=3)
    z = enc.encode([0.1, 0.2, 0.3])
    assert z.shape == (3,)
    assert z.dtype == torch.float32


def test_encode_batched_shape():
    enc = IdentityEncoder(latent_dim=2)
    z = enc.encode(torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]))
    assert z.shape == (3, 2)
    assert z.dtype == torch.float32


def test_encode_validates_dim_single():
    enc = IdentityEncoder(latent_dim=3)
    with pytest.raises(ValueError):
        enc.encode([0.1, 0.2])


def test_encode_validates_dim_batched():
    enc = IdentityEncoder(latent_dim=3)
    with pytest.raises(ValueError):
        enc.encode(torch.zeros(5, 4))


def test_encode_from_numpy_like_list():
    enc = IdentityEncoder(latent_dim=2)
    z = enc.encode([1, 2])  # int → float32
    assert z.dtype == torch.float32
    assert torch.allclose(z, torch.tensor([1.0, 2.0]))
