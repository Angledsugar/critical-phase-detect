"""Core framework — env-/encoder-/policy-agnostic.

Public surface: Trajectory, TrajectoryBuffer, Labeler protocol.
KDE, reward, conf, pipeline live in submodules and are imported lazily.
"""
from cpd.core.buffer import TrajectoryBuffer
from cpd.core.trajectory import Trajectory

__all__ = ["Trajectory", "TrajectoryBuffer"]
