"""Plot delta=0.1 l_bias sensitivity for 0 and 90 degree cases.

The plotted quantity follows the style of the delta=0.001 figure:
    sigma_bias / sigma_y
where sigma_y is taken as the realized std from the table.
The shaded band is approximated as mean +/- 1.96 posterior standard deviations
because the table provides posterior standard deviations rather than posterior
quantiles.

Run from the project root or anywhere else with:

    python paper_2/plot_delta01_lbias_angle_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "lbias_delta01_angle_sensitivity"

L_BIAS = np.array([0.25, 1.0, 3.0, 6.0, 12.0])
CI95_MULTIPLIER = 1.96

DATASETS = {
    r"$0^\circ$": {
        "realized_std": 0.078,
        "sigma_bias": np.array([0.0382, 0.0300, 0.0385, 0.0462, 0.0612]),
        "sigma_bias_std": np.array([2.75e-2, 2.64e-2, 3.62e-2, 4.54e-2, 6.41e-2]),
        "color": "#4bb3df",
    },
    r"$90^\circ$": {
        "realized_std": 0.104,
        "sigma_bias": np.array([0.449, 0.391, 0.529, 0.712, 0.986]),
        "sigma_bias_std": np.array([6.4e-2, 5.8e-2, 8.3e-2, 0.119, 0.168]),
        "color": "#ff6f7d",
    },
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",

        "font.size": 30,
        "axes.labelsize": 30,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "legend.fontsize": 30,
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.0,
    })


    fig, ax = plt.subplots(figsize=(8, 6.0), constrained_layout=True)

    for label, data in DATASETS.items():
        sigma_y = float(data["realized_std"])
        y = data["sigma_bias"] / sigma_y
        y_ci = CI95_MULTIPLIER * data["sigma_bias_std"] 
        lo = np.maximum(y - y_ci, 0.0)
        hi = y + y_ci
        color = data["color"]

        ax.fill_between(L_BIAS, lo, hi, color=color, alpha=0.25, linewidth=0.0)
        ax.plot(L_BIAS, y, color=color, label=label)

    ax.set_xscale("log", base=2)
    ax.set_xlim(0.25, 12.0)
    ax.set_xticks(L_BIAS)
    ax.set_xticklabels([f"{v:g}" for v in L_BIAS])
    ax.set_xlabel(r"$l_{\mathrm{bias}}$")
    ax.set_ylabel(r"$\sigma_{\mathrm{bias}}/\sigma_y$")
    # ax.set_title(r"$\delta/a = 0.1$")
    ax.grid(True, alpha=0.28, linewidth=0.7)
    ax.legend(loc="best", framealpha=0.95)

    png_path = OUTPUT_DIR / "delta01_lbias_angle_sensitivity.png"
    pdf_path = OUTPUT_DIR / "delta01_lbias_angle_sensitivity.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
