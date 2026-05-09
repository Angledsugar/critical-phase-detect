"""Track 5 — RLT-style ground-truth handoff labeling pipeline.

Public surface: rubric, store, and the Streamlit annotator entrypoint. The
annotator imports streamlit lazily so test code that only touches storage /
rubric does not require the labeling extra.
"""
from cpd.labeling.rubric import RubricV1
from cpd.labeling.store import (
    compute_kappa,
    load_labels,
    load_split,
    save_label,
    split_dataset,
)

__all__ = [
    "RubricV1",
    "compute_kappa",
    "load_labels",
    "load_split",
    "save_label",
    "split_dataset",
]
