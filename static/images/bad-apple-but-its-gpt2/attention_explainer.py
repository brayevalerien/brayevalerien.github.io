#!/usr/bin/env python3
"""Generate attention map explainer diagram for blog post."""

import matplotlib.pyplot as plt
import numpy as np

# Monokai Classic Pro
BG = "#272822"
FG = "#F8F8F2"
COMMENT = "#75715E"
GUTTER = "#3E3D32"
PRIMARY = "#AE81FF"  # bright magenta
GREEN = "#A6E22E"
CYAN = "#66D9EF"

fig, axes = plt.subplots(1, 3, figsize=(11, 4), facecolor=BG,
                          gridspec_kw={"width_ratios": [0.65, 0.1, 1]})

tokens = ["The", "cat", "sat", "on", "the", "mat"]
n = len(tokens)

# ── Left: tokens with attention arrows ──
ax = axes[0]
ax.set_facecolor(BG)
y_positions = np.linspace(0.9, 0.1, n)

for i, (tok, y) in enumerate(zip(tokens, y_positions)):
    color = CYAN if i % 2 == 0 else GREEN
    ax.text(0.6, y, tok, fontsize=14, fontweight="bold", color=color,
            ha="center", va="center", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=GUTTER, edgecolor=color, linewidth=1.2))

weights = [0.05, 0.3, 0.15, 0.1, 0.1, 0.3]
for i in range(n - 1):
    alpha = max(0.15, weights[i])
    lw = weights[i] * 6 + 0.5
    ax.annotate("", xy=(0.32, y_positions[i]), xytext=(0.32, y_positions[-1]),
                arrowprops=dict(arrowstyle="-|>", color=PRIMARY, lw=lw, alpha=alpha))

ax.text(0.17, y_positions[-1], "attends to\nprevious", fontsize=8, color=COMMENT,
        ha="center", va="center", fontstyle="italic")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ── Middle: arrow ──
ax = axes[1]
ax.set_facecolor(BG)
ax.text(0.5, 0.5, "→", fontsize=36, color=PRIMARY, ha="center", va="center", fontweight="bold")
ax.axis("off")

# ── Right: attention matrix ──
ax = axes[2]
ax.set_facecolor(BG)

np.random.seed(42)
attn = np.zeros((n, n))
for i in range(n):
    raw = np.random.randn(i + 1) * 2
    attn[i, :i+1] = np.exp(raw) / np.exp(raw).sum()

mask = np.triu(np.ones((n, n), dtype=bool), k=1)
attn[mask] = np.nan
cmap = plt.cm.magma.copy()
cmap.set_bad(color=BG)

ax.imshow(attn, cmap=cmap, aspect="equal", interpolation="nearest")

for i, tok in enumerate(tokens):
    color = CYAN if i % 2 == 0 else GREEN
    ax.text(i, -0.65, tok, fontsize=9, color=color, ha="center", va="center",
            fontfamily="monospace", fontweight="bold")
    ax.text(-0.8, i, tok, fontsize=9, color=color, ha="center", va="center",
            fontfamily="monospace", fontweight="bold")

for i in range(n + 1):
    ax.axhline(i - 0.5, color=GUTTER, linewidth=0.5)
    ax.axvline(i - 0.5, color=GUTTER, linewidth=0.5)

ax.text(n/2, n + 0.3, "keys (attended to)", fontsize=8, color=COMMENT, ha="center", fontstyle="italic")
ax.text(-1.6, n/2, "queries", fontsize=8, color=COMMENT, ha="center", va="center", rotation=90, fontstyle="italic")

ax.set_xlim(-0.5, n - 0.5)
ax.set_ylim(n - 0.5, -1.1)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

fig.tight_layout(w_pad=0.3)
fig.savefig(__file__.replace(".py", ".png"), dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
plt.close()
print("Saved!")
