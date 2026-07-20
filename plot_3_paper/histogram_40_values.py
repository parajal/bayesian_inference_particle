import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


values = np.array(
    [
        1.5456e01,
        1.7149e01,
        1.7453e01,
        2.1560e01,
        1.6598e01,
        1.4354e01,
        2.0060e01,
        1.6660e01,
        2.0331e01,
        1.8179e01,
        2.3802e01,
        1.8448e01,
        1.6052e01,
        1.6389e01,
        1.6111e01,
        1.8252e01,
        1.7960e01,
        1.5990e01,
        1.9692e01,
        1.8119e01,
        1.5340e01,
        1.5332e01,
        1.6182e01,
        1.9146e01,
        1.6297e01,
        1.4711e01,
        1.6169e01,
        1.7638e01,
        1.7575e01,
        1.4790e01,
        2.0669e01,
        1.6892e01,
        1.5134e01,
        1.4624e01,
        1.6614e01,
        2.3050e01,
        1.6994e01,
        1.7578e01,
        1.8347e01,
        1.5969e01,
    ]
)

bulk_g = 17.14
mean = values.mean()
stderr = values.std(ddof=1) / math.sqrt(values.size)
ci95_half_width = 2 * stderr

plt.rcParams.update(
    {
        "font.size": 35,
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.weight": "light",
        "lines.linewidth": 1.5,
        "figure.figsize": (10, 8),
        "legend.fontsize": 30,
        "axes.labelsize": 40,
        "legend.loc": "best",
        "lines.markersize": 8,
        "legend.frameon": True,
    }
)

output_dir = Path(__file__).resolve().parent

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

bins = np.histogram_bin_edges(values, bins="sturges")
ax.hist(
    values,
    bins=bins,
    color="#9FBAD6",
    edgecolor="black",
    linewidth=1.2,
    alpha=0.68,
    # label=r"Values",
    zorder=1,
)

# ax.axvspan(
#     mean - ci95_half_width,
#     mean + ci95_half_width,
#     facecolor="#F4A261",
#     edgecolor="#8C510A",
#     linewidth=1.4,
#     hatch="//",
#     alpha=0.42,
#     label=rf"95\% CI $=\pm {ci95_half_width:.2f}$",
#     zorder=2,
# )
ax.axvline(
    mean,
    color="#1B5E20",
    linestyle="-",
    linewidth=2.6,
    label=rf"$\bar{{G}}_{{\rm loc}}={mean:.2f}$",
    zorder=4,
)
ax.axvline(
    bulk_g,
    color="red",
    linestyle=":",
    linewidth=3.0,
    label=rf"$G_{{\rm bulk}}={bulk_g:.2f}$",
    zorder=4,
)

ax.set_xlabel(r"$G$")
ax.tick_params(axis="both", which="major", labelsize=25)
ax.grid(axis="y", alpha=0.22)
ax.legend(framealpha=0.93)

fig.savefig(output_dir / "values_histogram_40.png", dpi=300, bbox_inches="tight")
fig.savefig(output_dir / "values_histogram_40.pdf", bbox_inches="tight")

print(f"n = {values.size}")
print(f"mean = {mean:.6f}")
print(f"stderr = {stderr:.6f}")
print(f"95_ci_half_width = {ci95_half_width:.6f}")
print(f"bulk_g = {bulk_g:.6f}")
print(f"saved_png = {output_dir / 'values_histogram_40.png'}")
print(f"saved_pdf = {output_dir / 'values_histogram_40.pdf'}")
