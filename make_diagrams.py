"""Two schematic figures for the paper, drawn to print cleanly in one column."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "figure.dpi": 400,
})

INK = "#1a1a1a"
FILL = "#f2f2f2"
ACCENT = "#d9d9d9"


def box(ax, x, y, w, h, text, fill=FILL, fontsize=6.4, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.7, edgecolor=INK, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, linespacing=1.35, weight=weight)


def arrow(ax, x1, y1, x2, y2, style="-|>", dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=7,
        linewidth=0.7, color=INK, shrinkA=0, shrinkB=0,
        linestyle=(0, (2.5, 2)) if dashed else "solid"))


# ---------------------------------------------------------------- Figure: pipeline
fig, ax = plt.subplots(figsize=(3.4, 2.35))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

box(ax, 0.02, 0.72, 0.28, 0.20, "Raw touch events\n$x,\\ y,\\ p,\\ a,\\ t$")
box(ax, 0.36, 0.72, 0.28, 0.20, "Stroke\nsegmentation")
box(ax, 0.70, 0.72, 0.28, 0.20, "Quantile\ncalibration", fill=ACCENT)
arrow(ax, 0.30, 0.82, 0.36, 0.82)
arrow(ax, 0.64, 0.82, 0.70, 0.82)

ax.text(0.98, 0.655, "incoming samples only",
        ha="right", va="top", fontsize=5.2, style="italic", color=INK)

box(ax, 0.70, 0.40, 0.28, 0.18, "36 stroke\nfeatures")
arrow(ax, 0.84, 0.72, 0.84, 0.58)

box(ax, 0.36, 0.40, 0.28, 0.18, "Per-user\nclassifier")
arrow(ax, 0.70, 0.49, 0.64, 0.49)

box(ax, 0.02, 0.40, 0.28, 0.18, "Stroke score\n$s \\in [0,1]$")
arrow(ax, 0.36, 0.49, 0.30, 0.49)

box(ax, 0.02, 0.08, 0.44, 0.20, "Score fusion\nover $k$ strokes")
arrow(ax, 0.16, 0.40, 0.16, 0.28)

box(ax, 0.54, 0.08, 0.44, 0.20, "Equal error rate\nFAR $=$ FRR")
arrow(ax, 0.46, 0.18, 0.54, 0.18)

fig.savefig("/home/claude/pack/fig_pipeline.png", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ------------------------------------------------------- Figure: cross-device protocol
fig, ax = plt.subplots(figsize=(3.4, 2.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

ax.plot([0.515, 0.515], [0.05, 0.99], color=INK, linewidth=0.5,
        linestyle=(0, (2, 2.5)))
ax.text(0.25, 0.99, "Handset A (enrolment)", ha="center", va="top",
        fontsize=6.0, weight="bold", color=INK)
ax.text(0.79, 0.99, "Handset B (unfamiliar)", ha="center", va="top",
        fontsize=6.0, weight="bold", color=INK)

box(ax, 0.03, 0.76, 0.44, 0.13, "Strokes of $u$ on A", fontsize=6.0)
box(ax, 0.57, 0.76, 0.41, 0.13, "Strokes of $u$ on B", fontsize=6.0)

box(ax, 0.03, 0.52, 0.20, 0.13, "Enrol\nhalf", fontsize=6.0)
box(ax, 0.27, 0.52, 0.20, 0.13, "Held-out\nhalf", fontsize=6.0)
box(ax, 0.57, 0.52, 0.41, 0.13, "Calibration\nB $\\rightarrow$ A", fill=ACCENT, fontsize=6.0)

arrow(ax, 0.13, 0.76, 0.13, 0.65)
arrow(ax, 0.37, 0.76, 0.37, 0.65)
arrow(ax, 0.775, 0.76, 0.775, 0.65)

box(ax, 0.03, 0.20, 0.20, 0.13, "Template\nof $u$", fontsize=6.0)
arrow(ax, 0.13, 0.52, 0.13, 0.33)

# same-handset test: down from held-out, then left into the template
ax.plot([0.37, 0.37], [0.52, 0.295], color=INK, linewidth=0.7)
arrow(ax, 0.37, 0.295, 0.24, 0.295)
ax.text(0.395, 0.325, "same-handset test", ha="left", va="bottom",
        fontsize=5.4, color=INK)

# cross-handset test: down from calibration, then left into the template
ax.plot([0.775, 0.775], [0.52, 0.135], color=INK, linewidth=0.7,
        linestyle=(0, (2.5, 2)))
ax.plot([0.775, 0.24], [0.135, 0.135], color=INK, linewidth=0.7,
        linestyle=(0, (2.5, 2)))
arrow(ax, 0.24, 0.135, 0.185, 0.19)
ax.text(0.50, 0.165, "cross-handset test", ha="center", va="bottom",
        fontsize=5.4, color=INK)

ax.text(0.5, 0.01, "Quantile tables come from other participants, never from $u$",
        ha="center", va="bottom", fontsize=5.3, style="italic", color=INK)

fig.savefig("/home/claude/pack/fig_protocol.png", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("saved")
