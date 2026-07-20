"""
Domain-convergence plots for the multi-particle triperiodic Stokes runs.

Produces two figures:

  1. domain_conv_xt_plot.png      -- raw curves only (no correction):
       * velocity  U(t)  (column 5)
       * displacement x(t) (column 2)
     overlaid for every box size, with the analytical model.

  2. domain_conv_hasimoto.png     -- raw vs. Hasimoto-corrected (2x2):
       left column  = raw, right column = corrected by Q(L)
       top row      = velocity, bottom row = displacement.

Hasimoto correction (simple-cubic array, sphere radius a):

    U_corrected(t) = U(t) / Q(L),   x_corrected(t) = x(t) / Q(L)

    Q(L) = 1 - 2.8373*(a/L) + (4*pi/3)*(a/L)**3 + ...

Q(L) is constant for a given box, so it simply rescales the whole time
series. Dividing the finite-box result by Q(L) should recover the unbounded
(analytical) behaviour.

Data file columns (lx_*.out or lx-*.out):
    0:t  1:x  2:y  3:z  4:Ux  5:Uy  6:Uz  7:Wx  8:Wy  9:Wz
Analytical model columns (analytical_model.txt):
    0:t  1:x_model  2:vx_model
"""

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 25,
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.0,
})

BASE = Path(__file__).resolve().parent

# Column indices.
T_COL = 0      # time
X_COL = 1      # displacement x(t)
V_COL = 4      # velocity Ux(t)  (the 5th column)

# Analytical-model columns.
AM_T, AM_X, AM_V = 0, 1, 2

A_RADIUS = 1.0                       # sphere radius

LX_NAME_TO_VALUE = {
    "five": 5,
    "ten": 10,
    "twenty": 20,
    "thirty": 30,
    "fourty": 40,  # keep the spelling used by the data file
    "forty": 40,
    "sixty": 60,
}


def discover_boxes():
    """Find lx_<name>.out or lx-<L>.out files; return sorted {L: path}."""
    found = {}
    for p in [*BASE.glob("lx_*.out*"), *BASE.glob("lx-*.out*")]:
        stem = p.name.split(".")[0]
        try:
            token = stem.split("_", 1)[1] if "_" in stem else stem.split("-", 1)[1]
        except IndexError:
            continue
        try:
            L = int(token)
        except ValueError:
            L = LX_NAME_TO_VALUE.get(token.lower())
        if L is None:
            continue
        # prefer the shorter/canonical name if duplicates exist
        if L not in found or len(p.name) < len(found[L].name):
            found[L] = p
    return dict(sorted(found.items()))


BOX_PATHS = discover_boxes()
BOXES = list(BOX_PATHS)
DOMAIN_COLORS = {
    5: "#D55E00",   # vermillion
    10: "#0072B2",  # blue
    20: "#CC79A7",  # reddish purple
    40: "red",  # orange
    60: "#56B4E9",  # sky blue
}
_fallback = plt.cm.tab10(np.linspace(0, 1, max(len(BOXES), 1)))
COLORS = {L: DOMAIN_COLORS.get(L, _fallback[i]) for i, L in enumerate(BOXES)}


def hasimoto_Q(L, a=A_RADIUS):
    """Hasimoto drag factor for a simple-cubic array (leading terms)."""
    return 1.0 - 2.8373 * (a / L) + (4.0 * math.pi / 3.0) * (a / L) ** 3


def load_box(L):
    return np.loadtxt(BOX_PATHS[L], ndmin=2)


def load_analytical():
    path = BASE / "analytical_model.txt"
    return np.loadtxt(path, ndmin=2) if path.exists() else None


def plot_raw(am):
    """Figure 1: raw velocity and displacement, no correction."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for L in BOXES:
        d = load_box(L)
        c = COLORS[L]
        ax[0].plot(d[:, T_COL], d[:, V_COL], "-o", ms=3, color=c, alpha=0.9, label=f"lx-${L}$")
        ax[1].plot(d[:, T_COL], d[:, X_COL], "-o", ms=3, color=c, alpha=0.9, label=f"lx-${L}$")
    if am is not None:
        ax[0].plot(am[:, AM_T], am[:, AM_V], "--", lw=2.5, color="0.15",
                   alpha=0.85, label="analytical")
        ax[1].plot(am[:, AM_T], am[:, AM_X], "--", lw=2.5, color="0.15",
                   alpha=0.85, label="analytical")
    ax[0].set_title("Velocity (raw, col 5)")
    ax[1].set_title("Displacement (raw, col 2)")
    ax[0].set_ylabel("U(t)")
    ax[1].set_ylabel("x(t)")
    for a in ax:
        a.set_xlabel("t")
        a.grid(True, alpha=0.3)
        a.legend( ncols = 1)
    fig.suptitle("Domain convergence (no correction)", fontsize=14)
    out = BASE / "domain_conv_xt_plot.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"[saved] {out}")


def plot_corrected(am):
    """Figure 2: raw vs. Hasimoto-corrected displacement."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), constrained_layout = True)
    print("  L      Q(L)")
    for L in [L for L in BOXES if L in (5, 10, 20, 40, 60)]:
        d = load_box(L)
        t, x = d[:, T_COL], d[:, X_COL]
        q = hasimoto_Q(L)
        c = COLORS[L]
        print(f"{L:3d}  {q:.6f}")
        ax[0].plot(t, x, "-o", ms=3, color=c, alpha=0.9, label=f"$L = {L}$")
        ax[1].plot(t, x / q, "-o", ms=3, color=c, alpha=0.9, label=f"$L = {L}$")

    if am is not None:
        for a in ax:
            a.plot(am[:, AM_T], am[:, AM_X], "--", lw=2.5, color="0.15",
                   alpha=0.85, label="SAM")

    ax[0].set_ylabel("$x(t)$")
    ax[1].set_ylabel("$x(t)/Q(L)$")
    for a in ax:
        a.set_xlabel("$t$")
        a.set_xlim(0, 0.5)
        a.set_ylim(0, 0.3)
        a.set_xticks([0, 0.25, 0.5])
        a.set_yticks([0, 0.15, 0.3])
        a.grid(True, alpha=0.3)
        a.legend( ncols = 2)
    # fig.subplots_adjust(wspace=0.2)
    out = BASE / "domain_conv_hasimoto.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"[saved] {out}")


def main():
    am = load_analytical()
    plot_raw(am)
    plot_corrected(am)


if __name__ == "__main__":
    main()
