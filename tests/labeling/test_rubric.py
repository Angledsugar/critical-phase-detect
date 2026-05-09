"""RubricV1 — Markdown export and bilingual docstring sanity check."""
from __future__ import annotations

from cpd.labeling import rubric as rubric_mod
from cpd.labeling.rubric import RubricV1


def test_to_markdown_non_empty_with_all_criteria() -> None:
    md = RubricV1().to_markdown()
    assert md.strip()
    # All criteria field names should be discoverable in the rendered doc.
    for substr in (
        "Pre-handoff signal",
        "Peak uncertainty zone",
        "Post-handoff recovery",
        "Binary handoff yes/no",
        "Tie-breaker",
    ):
        assert substr in md, f"missing {substr!r} in to_markdown()"


def test_module_docstring_is_bilingual() -> None:
    """At least one Hangul codepoint must appear in the module docstring."""
    doc = rubric_mod.__doc__ or ""
    assert any(0xAC00 <= ord(ch) <= 0xD7A3 for ch in doc), (
        "rubric module docstring should contain Korean characters"
    )


def test_rubric_dataclass_fields_round_trip() -> None:
    r = RubricV1()
    # All five criteria fields are non-empty strings (so to_markdown can use them).
    for field in (
        "pre_handoff_signal",
        "peak_uncertainty_zone",
        "post_handoff_recovery",
        "binary_handoff_yesno",
        "tie_breaker",
    ):
        value = getattr(r, field)
        assert isinstance(value, str) and value.strip(), f"{field} empty"
    assert r.version == "v1"
