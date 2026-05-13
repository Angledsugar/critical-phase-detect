"""Figure: Temporal-distance triplet sampling (paper §4.1).

Left panel  — one demo timeline with anchor/positive/negative markers.
Right panel — 2D latent-space projection with pull/push arrows.

Run:
    python paper/figures/make_triplet_fig.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

T = 120
T_A, T_P, T_N = 50, 52, 70
K_POS, K_NEG = 2, 20

OUT = Path(__file__).parent / "triplet_sampling.png"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "mathtext.fontset": "cm",
    }
)

fig, (ax_t, ax_z) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.28})

# ────────────────────────── LEFT: timeline ──────────────────────────
ax_t.set_title(r"One demo $\tau^{(i)}$ (state space)", pad=12)
ax_t.set_xlim(-5, T + 5)
ax_t.set_ylim(-1.6, 1.4)
ax_t.set_yticks([])
ax_t.set_xlabel(r"time step $t$")
for s in ("top", "right", "left"):
    ax_t.spines[s].set_visible(False)

# trajectory backbone (a smooth curve that loosely suggests "task progress")
ts = np.linspace(0, T, 400)
curve = 0.55 * np.sin(2 * np.pi * ts / T * 0.55) - 0.05
ax_t.plot(ts, curve, color="#888", lw=1.4, zorder=1)

# tick markers every 10 step
tick_steps = np.arange(0, T + 1, 10)
ax_t.scatter(tick_steps, 0.55 * np.sin(2 * np.pi * tick_steps / T * 0.55) - 0.05,
             s=10, color="#bbbbbb", zorder=2)

# anchor / positive / negative markers
y_of = lambda t: 0.55 * np.sin(2 * np.pi * t / T * 0.55) - 0.05
COL_A, COL_P, COL_N = "#1f4e79", "#2e7d32", "#c62828"

ax_t.scatter([T_A], [y_of(T_A)], s=95, color=COL_A, zorder=5, edgecolor="white", linewidth=1.4)
ax_t.scatter([T_P], [y_of(T_P)], s=95, color=COL_P, zorder=5, edgecolor="white", linewidth=1.4)
ax_t.scatter([T_N], [y_of(T_N)], s=95, color=COL_N, zorder=5, edgecolor="white", linewidth=1.4)

# labels above markers — staggered heights to avoid overlap
ax_t.annotate(r"$s_{t_a}$ (anchor)", xy=(T_A, y_of(T_A)), xytext=(T_A - 18, 1.05),
              ha="center", color=COL_A, fontsize=11,
              arrowprops=dict(arrowstyle="-", color=COL_A, lw=0.8))
ax_t.annotate(r"$s_{t_p}$ (positive)", xy=(T_P, y_of(T_P)), xytext=(T_P + 6, 1.25),
              ha="left", color=COL_P, fontsize=11,
              arrowprops=dict(arrowstyle="-", color=COL_P, lw=0.8))
ax_t.annotate(r"$s_{t_n}$ (negative)", xy=(T_N, y_of(T_N)), xytext=(T_N + 8, 0.55),
              ha="left", color=COL_N, fontsize=11,
              arrowprops=dict(arrowstyle="-", color=COL_N, lw=0.8))

# brackets for k_pos / K_neg below — taller tick marks so they read clearly
def bracket(ax, x0, x1, y, label, color, tick=0.12):
    ax.plot([x0, x0, x1, x1], [y + tick, y, y, y + tick], color=color, lw=1.5)
    ax.text((x0 + x1) / 2, y - 0.13, label, ha="center", va="top", color=color, fontsize=10)

bracket(ax_t, T_A, T_P, -0.55, fr"$k_{{\mathrm{{pos}}}}={K_POS}$", COL_P)
bracket(ax_t, T_A, T_N, -1.05, fr"$K_{{\mathrm{{neg}}}}={K_NEG}$", COL_N)

# trajectory endpoint labels
ax_t.text(0, -0.45, r"$t=0$", ha="center", fontsize=9, color="#666")
ax_t.text(T, -0.45, rf"$t=T={T}$", ha="center", fontsize=9, color="#666")

# ────────────────────────── RIGHT: latent space ──────────────────────────
ax_z.set_title(r"Latent space $\mathbb{R}^{64}$ (2D projection)", pad=12)
ax_z.set_xlim(-3.2, 3.2)
ax_z.set_ylim(-2.6, 2.6)
ax_z.set_xticks([])
ax_z.set_yticks([])
ax_z.set_aspect("equal")
for s in ("top", "right", "bottom", "left"):
    ax_z.spines[s].set_color("#aaaaaa")

# axis labels (subtle)
ax_z.set_xlabel(r"$z_1$", fontsize=10, color="#888", labelpad=2)
ax_z.set_ylabel(r"$z_2$", fontsize=10, color="#888", labelpad=2)

# latent positions
z_a = np.array([0.0, 0.0])
z_p = np.array([0.55, -0.35])      # close
z_n = np.array([-2.1, 1.7])        # far

# distance dashed lines
ax_z.plot([z_a[0], z_p[0]], [z_a[1], z_p[1]], color=COL_P, ls=":", lw=1.0, alpha=0.8)
ax_z.plot([z_a[0], z_n[0]], [z_a[1], z_n[1]], color=COL_N, ls=":", lw=1.0, alpha=0.8)

# midpoint distance labels
mp_ap = (z_a + z_p) / 2 + np.array([0.05, 0.25])
mp_an = (z_a + z_n) / 2 + np.array([0.25, 0.05])
ax_z.text(mp_ap[0], mp_ap[1], r"$\|z_a{-}z_p\|$" + "\n(small)", color=COL_P, fontsize=9, ha="left")
ax_z.text(mp_an[0], mp_an[1], r"$\|z_a{-}z_n\|$" + "\n(large)", color=COL_N, fontsize=9, ha="left")

# pull arrow: z_p toward z_a
pull = FancyArrowPatch(
    tuple(z_p + 0.18 * (z_a - z_p) / np.linalg.norm(z_a - z_p)),
    tuple(z_a + 0.22 * (z_p - z_a) / np.linalg.norm(z_p - z_a)),
    arrowstyle="-|>", mutation_scale=14, color=COL_P, lw=2.0,
)
ax_z.add_patch(pull)
ax_z.text(z_p[0] + 0.35, z_p[1] - 0.35, "pull", color=COL_P, fontsize=10, fontweight="bold")

# push arrow: z_n away from z_a
direction_n = (z_n - z_a) / np.linalg.norm(z_n - z_a)
push_start = z_n + 0.18 * (z_a - z_n) / np.linalg.norm(z_a - z_n)
push_end = z_n + 0.55 * direction_n
push = FancyArrowPatch(
    tuple(push_start), tuple(push_end),
    arrowstyle="-|>", mutation_scale=14, color=COL_N, lw=2.0,
)
ax_z.add_patch(push)
ax_z.text(z_n[0] + 0.05, z_n[1] + 0.35, "push", color=COL_N, fontsize=10, fontweight="bold")

# latent dots
ax_z.scatter([z_a[0]], [z_a[1]], s=180, color=COL_A, zorder=5, edgecolor="white", linewidth=1.8)
ax_z.scatter([z_p[0]], [z_p[1]], s=140, color=COL_P, zorder=5, edgecolor="white", linewidth=1.5)
ax_z.scatter([z_n[0]], [z_n[1]], s=140, color=COL_N, zorder=5, edgecolor="white", linewidth=1.5)

ax_z.text(z_a[0] - 0.05, z_a[1] - 0.45, r"$z_a$", color=COL_A, fontsize=12, ha="center", fontweight="bold")
ax_z.text(z_p[0] + 0.15, z_p[1] - 0.05, r"$z_p$", color=COL_P, fontsize=12, ha="left", fontweight="bold")
ax_z.text(z_n[0] - 0.05, z_n[1] - 0.45, r"$z_n$", color=COL_N, fontsize=12, ha="center", fontweight="bold")

# ────────────────────────── overall caption (bottom) ──────────────────────────
fig.text(
    0.5, 0.01,
    r"$\varphi$ pulls $z_p$ toward $z_a$ and pushes $z_n$ away — enforcing "
    r"$\|z_a{-}z_p\|^2 + m < \|z_a{-}z_n\|^2$. Both positive and negative come from the "
    r"same demo; $k_\mathrm{pos}, K_\mathrm{neg}$ are forward step offsets, not sample counts.",
    ha="center", fontsize=9.5, color="#444",
)

plt.subplots_adjust(top=0.92, bottom=0.14, left=0.04, right=0.98)
plt.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Wrote {OUT}")
