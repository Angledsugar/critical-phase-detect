"""PR0 smoke test: every module under cpd.* must be importable.

Stubs raise NotImplementedError when called, but importing the modules
(and the protocols / dataclasses they define) must succeed so downstream
PRs can build against the framework.
"""


def test_top_level():
    import cpd

    assert cpd.__version__


def test_core():
    from cpd.core import (  # noqa: F401
        Trajectory,
        TrajectoryBuffer,
        conf,
        kde,
        labeler,
        pipeline,
        reward,
    )


def test_protocols():
    from cpd.encoders.base import PhiEncoder  # noqa: F401
    from cpd.envs.base import DemoSource, Env  # noqa: F401
    from cpd.eval.base import Metric  # noqa: F401
    from cpd.policies.base import Policy  # noqa: F401


def test_wandb_utils_module():
    from cpd import wandb_utils

    assert callable(wandb_utils.init_run)
    assert callable(wandb_utils.log_metrics)
