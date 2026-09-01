import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "figure.dpi": 400, "axes.grid": True, "grid.alpha": 0.22,
                     "axes.spines.top": False, "axes.spines.right": False})
INK = "#1a1a1a"
OUT = "/home/claude/pack"

# ------------------------------------------------ Fig: handset screen dimensions
bio = [(6, 480, 800, 218), (7, 800, 1205, 198), (11, 480, 800, 235),
       (13, 720, 1280, 306), (14, 800, 1232, 150), (16, 480, 800, 213),
       (17, 480, 854, 160), (19, 480, 800, 235), (20, 800, 1205, 198),
       (21, 800, 1205, 213), (22, 800, 1205, 213)]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5),
                         gridspec_kw={"width_ratios": [1.25, 1]})

ax = axes[0]
seen, x0 = {}, 0.0
for dev, w, h, dpi in bio:
    key = (w, h)
    seen.setdefault(key, []).append(dev)
for i, ((w, h), devs) in enumerate(sorted(seen.items())):
    wmm, hmm = w / 200 * 25.4, h / 200 * 25.4
    ax.add_patch(Rectangle((x0, 0), wmm, hmm, facecolor="#eeeeee",
                           edgecolor=INK, linewidth=0.8))
    ax.text(x0 + wmm / 2, hmm + 4, f"{w}$\\times${h}", ha="center",
            va="bottom", fontsize=5.6)
    ax.text(x0 + wmm / 2, hmm / 2, "\n".join(str(d) for d in devs),
            ha="center", va="center", fontsize=5.4, color=INK)
    x0 += wmm + 9
ax.set_xlim(-4, x0); ax.set_ylim(0, 190)
ax.set_aspect("equal"); ax.grid(False)
ax.set_yticks([]); ax.set_xticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("BioIdent: screen sizes (labels are handset IDs)", fontsize=7)

ax = axes[1]
touch = [(0, 546, 536), (1, 532, 508), (2, 555, 536), (3, 504, 505), (4, 319, 538)]
ax.bar([str(d) for d, _, _ in touch], [w for _, w, _ in touch], width=0.38,
       label="max $x$", color="#4C72B0", align="edge")
ax.bar([str(d) for d, _, _ in touch], [h for _, _, h in touch], width=-0.38,
       label="max $y$", color="#C44E52", align="edge")
ax.set_xlabel("Touchalytics handset", fontsize=7)
ax.set_ylabel("observed extent (px)", fontsize=7)
ax.tick_params(labelsize=6)
ax.legend(fontsize=6)
ax.set_title("Touchalytics: observed touch extents", fontsize=7)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_devices.png", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ------------------------------------------------ Fig: normalisation, two panels
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))

# --- panel 1: equal-frequency binning
ax = axes[0]
a_levels = np.array([10, 20, 30, 40])
b_levels = np.array([5, 10, 15, 20, 25, 30, 35, 40])
ax.scatter(a_levels, np.full_like(a_levels, 1.0, dtype=float), s=26,
           color="#4C72B0", zorder=3, label="Device A (4 levels)")
ax.scatter(b_levels, np.full_like(b_levels, 0.72, dtype=float), s=26,
           color="#C44E52", marker="s", zorder=3, label="Device B (8 levels)")
edges = [0, 12.5, 22.5, 32.5, 45]
for e in edges[1:-1]:
    ax.axvline(e, color=INK, linewidth=0.6, linestyle=(0, (2.5, 2)))
for i in range(4):
    ax.text((edges[i] + edges[i + 1]) / 2, 0.36, f"bin {i+1}", ha="center",
            fontsize=5.8, color=INK)
ax.set_ylim(0.25, 1.20); ax.set_xlim(0, 45)
ax.set_yticks([]); ax.tick_params(labelsize=6)
ax.set_xlabel("raw feature value", fontsize=7)
ax.set_title("Equal-frequency binning", fontsize=7.5)
ax.legend(fontsize=5.6, loc="upper left")

# --- panel 2: percentile mapping
ax = axes[1]
rng = np.random.default_rng(0)
p = np.linspace(0.01, 0.99, 200)
test = 20 + 60 * p ** 1.4
enrol = 30 + 55 * p ** 0.75
ax.plot(test, p * 100, color="#C44E52", lw=1.4, label="test handset")
ax.plot(enrol, p * 100, color="#4C72B0", lw=1.4, label="enrolment handset")

pct = 70
v_test = float(np.interp(pct / 100, p, test))
v_enrol = float(np.interp(pct / 100, p, enrol))
ax.plot([v_test, v_test], [0, pct], color=INK, lw=0.7, ls=(0, (2, 2)))
ax.plot([v_test, v_enrol], [pct, pct], color=INK, lw=0.7, ls=(0, (2, 2)))
ax.annotate("", xy=(v_enrol, pct), xytext=(v_test, pct),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9))
ax.plot([v_enrol, v_enrol], [0, pct], color=INK, lw=0.7, ls=(0, (2, 2)))
ax.scatter([v_test, v_enrol], [pct, pct], s=18, color=INK, zorder=4)
ax.text(v_test, pct - 9, f"{v_test:.0f}", ha="center", fontsize=6, color="#C44E52")
ax.text(v_enrol, pct - 9, f"{v_enrol:.0f}", ha="center", fontsize=6, color="#4C72B0")
ax.text(2, pct + 2, f"{pct}th percentile", fontsize=5.8, color=INK)
ax.set_ylim(0, 104); ax.set_xlim(0, 95)
ax.tick_params(labelsize=6)
ax.set_xlabel("raw channel value", fontsize=7)
ax.set_ylabel("percentile", fontsize=7)
ax.set_title("Percentile mapping, test $\\rightarrow$ enrolment", fontsize=7.5)
ax.legend(fontsize=5.8, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_normalisation.png", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ------------------------------------------------ Fig: per-device ROC + fusion
curves = json.load(open(f"{OUT}/per_device_roc.json"))
tab = pd.read_csv(f"{OUT}/per_device_kfold.csv")
ttd = pd.read_csv(f"{OUT}/cross_device_time_to_decision.csv", index_col=0)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax = axes[0]
styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
for i, row in tab.iterrows():
    dev = str(int(row.device))
    fpr, tpr = curves[dev]
    ax.plot(fpr, tpr, lw=1.2, linestyle=styles[i % len(styles)],
            label=f"handset {dev} (EER {row.eer_pooled:.02f})")
ax.plot([0, 1], [0, 1], color=INK, lw=0.6, ls=":")
ax.set_xlabel("false accept rate", fontsize=7)
ax.set_ylabel("true accept rate", fontsize=7)
ax.set_title("Per-handset ROC, five-fold cross-validation", fontsize=7.5)
ax.tick_params(labelsize=6); ax.legend(fontsize=5.6, loc="lower right")

ax = axes[1]
ks = [int(c) for c in ttd.columns]
for cond, colour, marker, lab in (("same", "#55A868", "o", "same handset"),
                                  ("cal", "#4C72B0", "s", "different, calibrated"),
                                  ("uncal", "#C44E52", "^", "different, uncorrected")):
    ax.plot(ks, ttd.loc[cond].values, marker=marker, ms=3.5, lw=1.4,
            color=colour, label=lab)
ax.axhline(0.5, color=INK, ls="--", lw=0.8)
ax.text(20, 0.512, "chance", ha="right", fontsize=5.6)
ax.set_xlabel("strokes fused into one decision", fontsize=7)
ax.set_ylabel("median equal error rate", fontsize=7)
ax.set_title("Score fusion either side of the device boundary", fontsize=7.5)
ax.tick_params(labelsize=6); ax.legend(fontsize=5.8, loc="center left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_roc_fusion.png", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("saved three figures")
