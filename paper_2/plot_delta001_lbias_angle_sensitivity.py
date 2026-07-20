"""Plot delta=0.001 l_bias sensitivity for 0 and 90 degree cases.

The numbers are taken from the two delta=0.001 tables:

    0 degree:  2% noise, prior on sigma_noise/bias = 0.56
    90 degree: 2% noise, prior on sigma_noise/bias = 0.33

The plotted quantity follows the style of the reference figure:
    sigma_bias / sigma_y
where sigma_y is taken as the realized std from the table.
The shaded band is approximated as mean +/- 1.96 posterior standard deviations
because the table provides posterior standard deviations rather than posterior
quantiles.

Run from the project root or anywhere else with:

    python paper_2/plot_delta001_lbias_angle_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "lbias_delta001_angle_sensitivity"

L_BIAS = np.array([0.25, 1.0, 3.0, 6.0, 12.0])
CI95_MULTIPLIER = 1.96

DATASETS = {
    r"$0^\circ$": {
        "realized_std": 0.039,
        "sigma_bias": np.array([0.0182, 0.0154, 0.0195, 0.0250, 0.0317]),
        "sigma_bias_std": np.array([1.34e-2, 1.33e-2, 1.844e-2, 2.42e-2, 3.33e-2]),
        "color": "#4bb3df",
    },
    r"$90^\circ$": {
        "realized_std": 0.05174,
        "sigma_bias": np.array([0.236, 0.262, 0.352, 0.467, 0.644]),
        "sigma_bias_std": np.array([1.58e-2, 2.16e-2, 4.08e-2, 5.50e-2, 7.72e-2]),
        "color": "#ff6f7d",
    },
}


def _format_lbias_tick(value: float) -> str:
    """Pretty tick labels for the l_bias values."""
    return f"{value:g}"


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
    ax.set_xticklabels([_format_lbias_tick(v) for v in L_BIAS])
    ax.set_xlabel(r"$l_{\mathrm{bias}}$")
    ax.set_ylabel(r"$\sigma_{\mathrm{bias}}/\sigma_y$")
    # ax.set_title(r"$\delta/a = 0.001$")
    ax.grid(True, alpha=0.28, linewidth=0.7)
    ax.legend(loc="best", framealpha=0.95)

    png_path = OUTPUT_DIR / "delta001_lbias_angle_sensitivity.png"
    pdf_path = OUTPUT_DIR / "delta001_lbias_angle_sensitivity.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
