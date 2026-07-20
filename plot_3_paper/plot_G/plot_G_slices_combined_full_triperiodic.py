"""
G(x) z-slices for full_triperiodic/flow0000.vtk, all three planes in ONE figure
sharing a single horizontal colorbar (coolwarm) that spans the three panels.

Field: G(x) = Gphi = G_0 * phi(x)  (Gphi/phi = 9 is the uniform background).
Each panel is (8,6); constrained_layout keeps them equal-sized and aligned.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

VTK = r"C:/Users/paraj/Documents/full_triperiodic/flow0000.vtk"
NPTS = 42675
Z_SLICES = [-2.5, 0.0, 2.5]
NG = 240
LIM = 5.0
CMAP = "coolwarm"

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


def parse():
    d = open(VTK, "rb").read()

    def al(kw, extra=False):
        i = d.find(kw); j = d.find(b"\n", i) + 1
        if extra: j = d.find(b"\n", j) + 1
        return j

    i = d.find(b"POINTS 42675 float")
    pts = np.frombuffer(d, ">f4", NPTS * 3, d.find(b"\n", i) + 1).reshape(-1, 3).astype(float)
    gphi = np.frombuffer(d, ">f4", NPTS, al(b"SCALARS Gphi", True)).astype(float)
    return pts, gphi


def main():
    pts, G = parse()
    lin = LinearNDInterpolator(pts, G)
    nn = NearestNDInterpolator(pts, G)

    gx = np.linspace(-LIM, LIM, NG)
    X, Y = np.meshgrid(gx, gx)

    vmin, vmax = float(G.min()), float(G.max())
    levels = np.linspace(vmin, vmax, 60)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    # Three equally sized (8, 6) panels in one row, laid out automatically.
    fig, axes = plt.subplots(
        1, 3, figsize=(24, 6), constrained_layout=True
    )

    cf = None
    for ax, z in zip(axes, Z_SLICES):
        Z = np.full_like(X, z)
        Gs = lin(X, Y, Z)
        m = np.isnan(Gs)
        if m.any():
            Gs[m] = nn(X[m], Y[m], Z[m])
        cf = ax.contourf(
            X, Y, Gs,
            levels=levels,
            cmap=CMAP,
            norm=norm,
            extend="neither",
            antialiased=False,
        )
        ax.set_title(r"$z = %+.1f$" % z)
        ax.set_xlabel(r"$x$")
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks([-4, -2, 0, 2, 4])
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel(r"$y$")
    for ax in axes[1:]:
        ax.set_yticklabels([])

    # single shared HORIZONTAL colorbar spanning the 3 panels; flat (no caps).
    ticks = np.linspace(vmin, vmax, 8)
    sm = cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cb = fig.colorbar(
        sm, ax=axes, orientation="horizontal", location="bottom",
        ticks=ticks, fraction=0.05, pad=0.04, aspect=60,
    )
    cb.ax.set_xticklabels([r"$%.2f$" % t for t in ticks])
    cb.set_label(r"$G$", labelpad=12)

    out = r"C:/Users/paraj/Documents/mulit-particle/plot_G/G_slices_combined.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("[saved] %s   (G in [%.2f, %.2f], cmap=%s)" % (out, vmin, vmax, CMAP))


if __name__ == "__main__":
    main()