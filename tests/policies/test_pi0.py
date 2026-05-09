"""Pi0Policy import-guarded smoke test."""
import pytest

pytest.importorskip("openpi")

from cpd.policies.pi0 import Pi0Policy  # noqa: E402


def test_pi0_requires_checkpoint():
    with pytest.raises(ValueError, match="checkpoint_dir"):
        Pi0Policy(model_id="pi0_libero", checkpoint_dir=None, device="cpu")


def test_pi0_unknown_config():
    with pytest.raises(ValueError):
        Pi0Policy(model_id="not_a_real_config", checkpoint_dir="/tmp/nope", device="cpu")
