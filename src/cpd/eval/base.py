"""Metric protocol. Paper §6.3."""
from __future__ import annotations

from typing import Any, Protocol


class Metric(Protocol):
    """A single metric comparing predictions to targets.

    Implementations: F1 (vs RLT label), success_rate, transfer F1, conf_fallback.
    """

    name: str

    def compute(self, predictions: Any, targets: Any) -> float: ...
