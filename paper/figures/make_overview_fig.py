"""Figure: CPD framework overview (paper §1).

Top row    — five-stage pipeline schematic (Input → Encode → Self-Label → Density → Detect).
Middle row — three detail panels expanding the key visual concepts.
Bottom     — one-line caption.

Run:
    python paper/figures/make_overview_fig.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent / "overview.png"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10,
        "mathtext.fontset": "cm",
        "axes.edgecolor": "#444",
        "axes.linewidth": 0.8,
    }
)

# ── palette ────────────────────────────────────────────────────────────
NAVY = "#1f4e79"
BLUE = "#2563eb"
TEAL = "#0e7c66"
VIOLET = "#7c3aed"
AMBER = "#b45309"
CRIMSON = "#b91c1c"
SUCCESS = "#15803d"
FAILURE = "#b91c1c"
KDE_P = "#bbf7d0"
KDE_N = "#fecaca"
INK = "#1f2937"
GRAY = "#6b7280"
PAGE_BG = "#fbfbfd"

STAGE_COLS = [NAVY, BLUE, VIOLET, AMBER, CRIMSON]

# ── figure layout ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(15.5, 8.0), facecolor=PAGE_BG)
gs = fig.add_gridspec(
    2, 3,
    height_ratios=[0.95, 2.4],
    hspace=0.36,
    wspace=0.22,
    left=0.035,
    right=0.975,
    top=0.95,
    bottom=0.10,
)

ax_top = fig.add_subplot(gs[0, :])
ax_a = fig.add_subplot(gs[1, 0])
ax_b = fig.add_subplot(gs[1, 1])
ax_c = fig.add_subplot(gs[1, 2])

rng = np.random.default_rng(7)

# ═══════════════════════════════════════════════════════════════════════
# TOP — pipeline schematic
# ═══════════════════════════════════════════════════════════════════════
ax_top.set_xlim(0, 1)
ax_top.set_ylim(0, 1)
ax_top.axis("off")

# subtle background band for training vs inference partition
ax_top.add_patch(
    Rectangle((0.018, 0.05), 0.768, 0.92, facecolor="#eef2ff", edgecolor="none", alpha=0.55, zorder=0)
)
ax_top.add_patch(
    Rectangle((0.804, 0.05), 0.178, 0.92, facecolor="#fef2f2", edgecolor="none", alpha=0.55, zorder=0)
)
ax_top.text(0.41, 0.97, "TRAINING — build buffer & KDE (offline)", fontsize=8.5, ha="center",
            color="#4338ca", style="italic")
ax_top.text(0.892, 0.97, "INFERENCE (online)", fontsize=8.5, ha="center", color="#9f1239", style="italic")

stages = [
    ("1", "Input", r"$N$ demos $\;+\;M$ rollouts", "states $s \in \mathbb{R}^8$"),
    ("2", "Encode", r"TLDR $\varphi:\mathbb{R}^8\!\to\!\mathbb{R}^{64}$", "triplet contrastive"),
    ("3", "Self-Label", r"$\|\varphi(s_T)-g\|<\varepsilon$", r"split into $B^+,\,B^-$"),
    ("4", "Density", r"KDE on $B^\pm$", r"$\tilde{f}^+(z),\;\tilde{f}^-(z)$"),
    ("5", "Detect", r"$r_t=\log\frac{\tilde{f}^+}{\tilde{f}^-}$", r"$r_t<0$ run $+\;K^*$ rule"),
]

n = len(stages)
box_w = 0.155
y_center = 0.50
box_h = 0.62

# evenly distribute boxes
total_w = n * box_w
gap = (0.94 - total_w) / (n - 1)
x_start = 0.03

for i, (num, name, line1, line2) in enumerate(stages):
    col = STAGE_COLS[i]
    x_left = x_start + i * (box_w + gap)

    # main box (rounded, filled)
    box = FancyBboxPatch(
        (x_left, y_center - box_h / 2),
        box_w,
        box_h,
        boxstyle="round,pad=0.005,rounding_size=0.018",
        facecolor=col,
        edgecolor=col,
        linewidth=1.0,
        alpha=0.95,
        zorder=2,
    )
    ax_top.add_patch(box)

    # numbered badge (small circle, top-left)
    badge_r = 0.018
    badge_cx = x_left + 0.022
    badge_cy = y_center + box_h / 2 - 0.06
    ax_top.add_patch(plt.Circle((badge_cx, badge_cy), badge_r, facecolor="white", edgecolor=col, linewidth=1.2, zorder=3))
    ax_top.text(badge_cx, badge_cy, num, fontsize=10, ha="center", va="center", color=col, fontweight="bold", zorder=4)

    # stage name (large, top)
    ax_top.text(x_left + box_w / 2, y_center + box_h / 2 - 0.10, name,
                fontsize=12.5, ha="center", color="white", fontweight="bold", zorder=4)
    # primary math/concept
    ax_top.text(x_left + box_w / 2, y_center, line1,
                fontsize=11, ha="center", va="center", color="white", zorder=4)
    # caption
    ax_top.text(x_left + box_w / 2, y_center - box_h / 2 + 0.10, line2,
                fontsize=9, ha="center", color="white", alpha=0.92, zorder=4)

    # arrow to next stage
    if i < n - 1:
        a_start = x_left + box_w + 0.005
        a_end = a_start + gap - 0.01
        ax_top.add_patch(
            FancyArrowPatch(
                (a_start, y_center),
                (a_end, y_center),
                arrowstyle="-|>",
                mutation_scale=20,
                color="#374151",
                lw=1.8,
                zorder=3,
            )
        )

# ═══════════════════════════════════════════════════════════════════════
# Panel A — TLDR encoder pre-training (zoom on stage ②)
# ═══════════════════════════════════════════════════════════════════════
ax_a.set_title("(A) Stage 2 · TLDR Triplet Encoder", color=BLUE, pad=10, fontweight="bold")
ax_a.set_xlim(0, 1)
ax_a.set_ylim(0, 1)
ax_a.axis("off")
ax_a.set_facecolor("#ffffff")

# subtle panel background
ax_a.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, facecolor="white", edgecolor="#e5e7eb", linewidth=1.0, zorder=0))

# four demo trajectories with anchored markers on top one
y_demos = [0.80, 0.69, 0.58, 0.47]
ts_demo = np.linspace(0.10, 0.48, 200)
for i, y in enumerate(y_demos):
    wave = 0.018 * np.sin(2 * np.pi * (ts_demo - 0.10) * (3 + i * 0.7))
    ax_a.plot(ts_demo, y + wave, color="#9ca3af", lw=1.4, zorder=1)

# triplet markers (anchor / positive / negative) on the top demo
y_top = 0.80
t_a, t_p, t_n = 0.20, 0.225, 0.38
for tx, col, lab, y_off in [(t_a, NAVY, "a", 0.05), (t_p, SUCCESS, "p", 0.075), (t_n, FAILURE, "n", 0.05)]:
    y_pt = y_top + 0.018 * np.sin(2 * np.pi * (tx - 0.10) * 3)
    ax_a.scatter([tx], [y_pt], s=85, color=col, zorder=5, edgecolor="white", linewidth=1.5)
    ax_a.text(tx, y_pt + y_off, f"$s_{{t_{lab}}}$", fontsize=10, ha="center", color=col, fontweight="bold")

ax_a.text(0.29, 0.40, r"$N$ state-only demos $\tau^{(i)}$", fontsize=10, ha="center", color=INK)

# arrow → encoder
ax_a.add_patch(FancyArrowPatch((0.52, 0.65), (0.62, 0.65), arrowstyle="-|>", mutation_scale=18, color="#374151", lw=1.6))

# encoder box (φ)
phi_box = FancyBboxPatch(
    (0.63, 0.48), 0.30, 0.34, boxstyle="round,pad=0.01,rounding_size=0.025",
    facecolor=BLUE, edgecolor=BLUE, linewidth=1.2, alpha=0.95
)
ax_a.add_patch(phi_box)
ax_a.text(0.78, 0.69, r"$\varphi$", fontsize=26, ha="center", va="center", color="white", fontweight="bold")
ax_a.text(0.78, 0.54, r"$\mathbb{R}^8 \to \mathbb{R}^{64}$", fontsize=10.5, ha="center", va="center", color="white")

# loss & hyperparameters at the bottom
ax_a.add_patch(Rectangle((0.06, 0.06), 0.88, 0.30, facecolor="#f8fafc", edgecolor="#e5e7eb", linewidth=0.8))
ax_a.text(0.50, 0.29, "Triplet contrastive loss", fontsize=10, ha="center", color=INK, fontweight="bold")
ax_a.text(0.50, 0.20, r"$\mathcal{L}=\max(0,\ \|z_a-z_p\|^2-\|z_a-z_n\|^2+m)$",
          fontsize=10.5, ha="center", color=INK)
ax_a.text(0.50, 0.11, r"$t_p=t_a+k_{\mathrm{pos}}\;(\text{near}),\quad t_n=t_a+K_{\mathrm{neg}}\;(\text{far})$",
          fontsize=9, ha="center", color=GRAY, style="italic")

# ═══════════════════════════════════════════════════════════════════════
# Panel B — G2 self-labeling + KDE on latent space (zoom on stages ③④)
# ═══════════════════════════════════════════════════════════════════════
ax_b.set_title("(B) Stages 3 & 4 · G2 Label + KDE in Latent Space", color=VIOLET, pad=10, fontweight="bold")
ax_b.set_xlim(-3.6, 3.6)
ax_b.set_ylim(-2.9, 3.1)
ax_b.set_xticks([])
ax_b.set_yticks([])
ax_b.set_aspect("equal")
ax_b.set_facecolor("#ffffff")
for s in ("top", "right", "bottom", "left"):
    ax_b.spines[s].set_color("#e5e7eb")
    ax_b.spines[s].set_linewidth(1.0)

# success / failure terminal latents
n_pos, n_neg = 95, 26
pos_pts = rng.multivariate_normal([1.25, 0.95], [[0.45, 0.08], [0.08, 0.38]], n_pos)
neg_pts = rng.multivariate_normal([-1.40, -0.95], [[0.55, -0.10], [-0.10, 0.45]], n_neg)

# soft concentric ellipses (KDE contours)
for sigma_mult, alpha_v in [(2.4, 0.10), (1.8, 0.16), (1.25, 0.24), (0.75, 0.32)]:
    ax_b.add_patch(
        Ellipse((1.25, 0.95), 2 * sigma_mult * np.sqrt(0.45), 2 * sigma_mult * np.sqrt(0.38),
                facecolor=KDE_P, alpha=alpha_v, edgecolor="none")
    )
    ax_b.add_patch(
        Ellipse((-1.40, -0.95), 2 * sigma_mult * np.sqrt(0.55), 2 * sigma_mult * np.sqrt(0.45),
                facecolor=KDE_N, alpha=alpha_v, edgecolor="none")
    )

# scatter the buffer points
ax_b.scatter(pos_pts[:, 0], pos_pts[:, 1], s=14, color=SUCCESS, alpha=0.78, edgecolor="white", linewidth=0.6, zorder=3)
ax_b.scatter(neg_pts[:, 0], neg_pts[:, 1], s=14, color=FAILURE, alpha=0.78, edgecolor="white", linewidth=0.6, zorder=3)

# centroid g and ε boundary on success side
ax_b.scatter([1.25], [0.95], marker="*", s=260, color=SUCCESS, edgecolor="white", linewidth=1.4, zorder=6)
ax_b.text(1.25, 1.65, r"$g=\mathrm{mean}_i\,\varphi(s_T^{(i)})$",
          fontsize=9.5, ha="center", color=SUCCESS, fontweight="bold")
ax_b.add_patch(
    Ellipse((1.25, 0.95), 2.0, 2.0, facecolor="none", edgecolor=SUCCESS, ls="--", lw=1.4, zorder=5)
)
ax_b.annotate("", xy=(2.25, 0.95), xytext=(1.25, 0.95),
              arrowprops=dict(arrowstyle="-", color=SUCCESS, lw=1.1))
ax_b.text(1.80, 0.55, r"$\varepsilon$", fontsize=14, color=SUCCESS, fontweight="bold")

# KDE labels (right/left aligned, well outside clouds)
ax_b.text(3.40, 2.55, r"$\tilde{f}^+(z)$  on $B^+$",
          fontsize=11.5, ha="right", color=SUCCESS, fontweight="bold")
ax_b.text(-3.40, -2.30, r"$\tilde{f}^-(z)$  on $B^-$",
          fontsize=11.5, ha="left", color=FAILURE, fontweight="bold")

# bottom-of-panel small axis-label
ax_b.text(0.50, -0.10, r"latent space $\mathbb{R}^{64}$ (2D PCA view)",
          fontsize=9, ha="center", color=GRAY, transform=ax_b.transAxes, style="italic")

# ═══════════════════════════════════════════════════════════════════════
# Panel C — Per-step r_t with debounced detection (zoom on stage ⑤)
# ═══════════════════════════════════════════════════════════════════════
ax_c.set_title("(C) Stage 5 · Per-step $r_t$ + Detection", color=CRIMSON, pad=10, fontweight="bold")
ax_c.set_facecolor("#ffffff")
for s in ("top", "right"):
    ax_c.spines[s].set_visible(False)
for s in ("bottom", "left"):
    ax_c.spines[s].set_color("#9ca3af")

T_traj = 220
t_axis = np.arange(T_traj)
base = 1.2 - 0.003 * t_axis
dip = -3.4 * np.exp(-((t_axis - 130) ** 2) / 1200)
noise = rng.normal(0, 0.18, T_traj)
r_t = base + dip + noise

# detect runs ≥ 3 of r_t < 0
crit_step = r_t < 0
runs = []
i = 0
while i < T_traj:
    if crit_step[i]:
        j = i
        while j < T_traj and crit_step[j]:
            j += 1
        if j - i >= 3:
            runs.append((i, j))
        i = j
    else:
        i += 1

# shaded critical region
for s, e in runs:
    ax_c.axvspan(s, e, alpha=0.20, color=FAILURE, ec="none", zorder=1)

# τ = 0
ax_c.axhline(0, color="#1f2937", lw=1.0, ls="--", alpha=0.7, zorder=2)
# r_t curve
ax_c.plot(t_axis, r_t, color=NAVY, lw=1.8, zorder=3)
# fill between r_t and 0 where r_t<0 to emphasise
ax_c.fill_between(t_axis, r_t, 0, where=r_t < 0, color=FAILURE, alpha=0.25, zorder=2)

ax_c.set_xlabel(r"time step $t$", color=INK, fontsize=10.5)
ax_c.set_ylabel(r"$r_t=\log\tilde{f}^+(z_t)-\log\tilde{f}^-(z_t)$", color=INK, fontsize=10.5)
ax_c.tick_params(colors=GRAY)
ax_c.set_xlim(0, T_traj)
ax_c.set_ylim(-3.8, 2.5)

# τ = 0 marker
ax_c.text(T_traj - 4, 0.18, r"$\tau=0$", fontsize=10, color=INK, ha="right")

# critical-phase annotation
if runs:
    s, e = runs[0]
    ax_c.annotate(
        "critical phase\n" + r"($r_t<0$, run length $\geq 3$)",
        xy=((s + e) / 2, -1.0),
        xytext=((s + e) / 2 - 30, 1.7),
        ha="center",
        fontsize=9.5,
        color=FAILURE,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=FAILURE, lw=1.0, connectionstyle="arc3,rad=0.18"),
    )

# trajectory-level decision (top-right callout)
n_crit_total = sum(e - s for s, e in runs)
ax_c.text(
    0.97, 0.97,
    rf"$n_\mathrm{{crit\_steps}}={n_crit_total} \;\geq\; K^*$" + "\n" + r"$\Longrightarrow$ rollout is critical",
    transform=ax_c.transAxes,
    fontsize=9.5,
    ha="right",
    va="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=FAILURE, lw=1.1),
)

# ═══════════════════════════════════════════════════════════════════════
# Bottom — one-line caption
# ═══════════════════════════════════════════════════════════════════════
fig.text(
    0.50,
    0.02,
    r"Critical Phase Detection — pre-trained encoder $\varphi$, self-supervised buffer split, "
    r"KDE log-ratio scoring, debounced detection. No oracle labels, no human annotation, no classifier retraining.",
    ha="center",
    fontsize=10,
    color=INK,
    style="italic",
)

plt.savefig(OUT, dpi=220, bbox_inches="tight", facecolor=PAGE_BG)
print(f"Wrote {OUT}")
