"""Pi05Policy import-guarded smoke test."""
import pytest

pytest.importorskip("openpi")

from cpd.policies.pi0_5 import Pi05Policy  # noqa: E402


def test_pi05_requires_checkpoint():
    with pytest.raises(ValueError, match="checkpoint_dir"):
        Pi05Policy(model_id="pi05_libero", checkpoint_dir=None, device="cpu")


def test_pi05_unknown_config():
    with pytest.raises(ValueError):
        Pi05Policy(model_id="not_a_real_config", checkpoint_dir="/tmp/nope", device="cpu")
