"""Plot inferred sigma values against the upper bound of the l_bias prior.

The numbers are taken from the 0 degree and 90 degree l_bias-prior
sensitivity tables.
Run from anywhere with:

    python paper_2/plot_lbias_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "lbias_sensitivity"

# Prior intervals are (lower, upper). The plot uses only the upper value.
L_BIAS_UPPER = np.array([160.0, 80.0, 40.0, 20.0, 10.0, 5.0, 2.5, 1.25, 0.625, 0.3125])
DATASETS = {
    "0 degree": {
        "sigma_noise": np.array([0.0830, 0.0832, 0.0819, 0.0821, 0.0823, 0.0815, 0.0810, 0.0783, 0.0759, 0.0747]),
        "sigma_bias": np.array([0.1000, 0.0987, 0.0776, 0.0559, 0.0465, 0.0355, 0.0324, 0.0309, 0.0347, 0.0358]),
        "l_bias": np.array([84.83, 44.75, 21.65, 11.11, 5.43, 2.47, 1.33, 0.689, 0.40, 0.253]),
        "true_model_error": 1.25,
        "true_sigma_noise": 0.1028,
    },
    "90 degree": {
        "sigma_noise": np.array([0.049, 0.049, 0.051, 0.049, 0.050, 0.047, 0.048, 0.044, 0.047, 0.062]),
        "sigma_bias": np.array([2.16, 1.70, 1.28, 0.98, 0.758, 0.590, 0.468, 0.400, 0.380, 0.425]),
        "l_bias": np.array([64.76, 37.14, 20.03, 11.46, 6.40, 3.66, 2.04, 1.10, 0.579, 0.29]),
        "true_model_error": 0.99,
        "true_sigma_noise": 0.1366,
    },
}

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

def _safe_angle_name(angle: str) -> str:
    return angle.replace(" ", "").replace("degree", "deg")


def _plot_single_angle(angle: str, data: dict[str, np.ndarray | float]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    ax.plot(
        L_BIAS_UPPER,
        data["sigma_bias"],
        marker="o",
        ms=7,
        lw=2.8,
        color="tab:red",
        label=r"inferred $\sigma_{\mathrm{bias}}$",
    )
    ax.plot(
        L_BIAS_UPPER,
        data["sigma_noise"],
        marker="s",
        ms=7,
        lw=2.8,
        color="tab:blue",
        label=r"inferred $\sigma_{\mathrm{noise}}$",
    )

    ax.axhline(
        data["true_sigma_noise"],
        color="black",
        ls=":",
        lw=2.2,
        label=rf"true $\sigma_{{\mathrm{{noise}}}}$ = {data['true_sigma_noise']:g}",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(L_BIAS_UPPER.max(), L_BIAS_UPPER.min())
    ax.set_xticks(np.linspace(L_BIAS_UPPER.min(), L_BIAS_UPPER.max(), num=5))
    ax.set_ylabel(r"inferred $\sigma$")
    ax.set_xlabel(r"upper bound of $l_{\mathrm{bias}}$ prior")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    
    stem = f"lbias_upper_vs_sigmas_{_safe_angle_name(angle)}"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [pdf_path, png_path]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "axes.linewidth": 1.0,
        }
    )

    saved_paths: list[Path] = []
    for angle, data in DATASETS.items():
        saved_paths.extend(_plot_single_angle(angle, data))

    for path in saved_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
