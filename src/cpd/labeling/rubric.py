"""Rubric v1 — RLT-style critical-phase / handoff annotation guide.

EN: A handoff is the single step at which the VLA's competence visibly
breaks down and an RL refinement should take over. Pick the earliest step
that satisfies all four signals (pre-handoff, peak uncertainty, post-handoff
recovery, binary yes/no). Use the tie-breaker only when two steps are
equally plausible.

KO: Handoff(인계점)란, VLA 의 수행 능력이 처음으로 명확히 흔들려서 RL 로
넘기는 게 합리적인 단일 step 을 말한다. 네 가지 신호(pre-handoff,
peak uncertainty, post-handoff recovery, binary yes/no)를 모두 만족하는
가장 이른 step 을 고른다. 두 step 이 동등하면 tie-breaker 로 결정.

Run `python -c "from cpd.labeling.rubric import RubricV1; print(RubricV1().to_markdown())"`
to dump the full rubric Markdown.
"""
from __future__ import annotations

from dataclasses import dataclass

_PRE = """\
**Pre-handoff signal.** Within the 5-10 steps preceding the candidate, the
end-effector starts to drift or oscillate around the target (e.g. lateral
jitter near a peg, repeated near-misses on a grasp). Action norms inflate
or alternate sign. *If the VLA is still smoothly progressing, the handoff
has not happened yet.*
"""

_PEAK = """\
**Peak uncertainty zone.** The candidate step itself is where corrective
action is most needed: contact ambiguity, occluded view, sub-millimetre
alignment, or a stage transition (e.g. release → reach). One should be able
to point at the frame and say "from here, vanilla VLA is unlikely to
finish without help."
"""

_POST = """\
**Post-handoff recovery.** After the candidate, the trajectory either (a)
fails outright, (b) limps to a low-quality success, or (c) recovers only by
luck. If the post-handoff window looks confidently correct, the true
handoff is *later*; move the marker.
"""

_BINARY = """\
**Binary handoff yes/no.** The trajectory must contain exactly one
handoff. If no step satisfies the three signals above, mark the trajectory
as "no handoff" (handoff_step = -1). Do not split the label across
multiple steps.
"""

_TIE = """\
**Tie-breaker.** When two adjacent steps are equally plausible, prefer the
*earlier* one (RL needs lead time). When the candidate spans a stage
transition, place the handoff on the last step of the *outgoing* stage,
not the first of the incoming stage.
"""


@dataclass(frozen=True)
class RubricV1:
    """Five-criterion rubric for RLT-style handoff annotation."""

    pre_handoff_signal: str = _PRE
    peak_uncertainty_zone: str = _PEAK
    post_handoff_recovery: str = _POST
    binary_handoff_yesno: str = _BINARY
    tie_breaker: str = _TIE

    version: str = "v1"

    def to_markdown(self) -> str:
        """Render full rubric as a Markdown document for annotator UI."""
        sections = [
            "# Rubric v1 — Handoff Annotation",
            "",
            "## 1. Pre-handoff signal",
            self.pre_handoff_signal,
            "## 2. Peak uncertainty zone",
            self.peak_uncertainty_zone,
            "## 3. Post-handoff recovery",
            self.post_handoff_recovery,
            "## 4. Binary handoff yes/no",
            self.binary_handoff_yesno,
            "## 5. Tie-breaker",
            self.tie_breaker,
        ]
        return "\n".join(sections)
