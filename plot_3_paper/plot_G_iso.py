"""
Matplotlib 3D isosurface plot of G(x).

This version:
- Keeps the visible 3D axes/grid.
- Uses translucent isosurfaces.
- Uses the coolwarm colormap.
- Reduces 3D tick-label overlap.
- Uses fewer axis ticks: [-5, 0, 5].
- Keeps the colorbar large enough for paper figures.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


# ------------------------------------------------------------
# Domain and file paths
# ------------------------------------------------------------
L = 10.0
HALF_L = L / 2.0

CACHE = Path(
    r"C:/Users/paraj/AppData/Local/Temp/claude/"
    r"C--Users-paraj-Documents-mulit-particle/"
    r"2e3a4312-71cf-423e-9f87-daeda2fcb6d9/scratchpad/Gvol_96.npy"
)

OUTDIR = Path(__file__).resolve().parent


# ------------------------------------------------------------
# Plot controls
# ------------------------------------------------------------
# Dataset approximately ranges from 10.28 to 33.57.
# Levels outside this range are skipped automatically.
LEVELS = [11, 14, 16, 18, 20, 23, 26, 30]

# One alpha value for each level.
# Lower alpha = more transparent.
# One alpha value for each level. Lower alpha = more transparent.
ALPHAS = [0.03, 0.04, 0.06, 0.07, 0.08, 0.1, 0.2, 0.30]

CMAP = "coolwarm"
VMIN, VMAX = 10.0, 34.0


# ------------------------------------------------------------
# Matplotlib styling
# ------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 25,
    "axes.labelsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    "lines.linewidth": 1.5,
    "figure.figsize": (8, 6),
})


def add_isosurface(ax, V, level, alpha, spacing, cmap, norm):
    """
    Extract and plot one isosurface of the scalar field V.
    """

    if not (V.min() < level < V.max()):
        print(f"[skip] level {level:g} outside field range")
        return

    verts, faces, _, _ = measure.marching_cubes(
        V,
        level=float(level),
        step_size=1,
    )

    # Convert voxel coordinates to physical coordinates in [-L/2, L/2]
    verts = -HALF_L + verts * spacing

    mesh = Poly3DCollection(
        verts[faces],
        alpha=alpha,
        linewidths=0.0,
        rasterized=True,
    )

    mesh.set_facecolor(cmap(norm(level)))
    mesh.set_edgecolor("none")

    ax.add_collection3d(mesh)


def format_3d_axes(ax):
    """
    Format 3D axes to avoid overlapping x/y/z labels and tick labels.
    """

    ax.set_xlim(-HALF_L, HALF_L)
    ax.set_ylim(-HALF_L, HALF_L)
    ax.set_zlim(-HALF_L, HALF_L)
    ax.set_box_aspect((1, 1, 1))

    # Use fewer ticks to avoid clutter in 3D.
    ticks = [-3, 0, 3]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    # Smaller 3D tick labels reduce overlap at the cube corners.
    ax.tick_params(axis="x",  pad=1)
    ax.tick_params(axis="y", pad=1)
    ax.tick_params(axis="z", pad=1)

    # Axis labels
    ax.set_xlabel(r"$x$", labelpad=6)
    ax.set_ylabel(r"$y$", labelpad=6)
    ax.set_zlabel(r"$z$", labelpad=6)

    # Orthographic projection looks cleaner for paper figures.
    ax.set_proj_type("ortho")

    # This view separates the front-corner tick labels better.
    ax.view_init(elev=22, azim=-50)

    # Light grid/pane style
    ax.grid(True)

    ax.xaxis.pane.set_alpha(0.08)
    ax.yaxis.pane.set_alpha(0.08)
    ax.zaxis.pane.set_alpha(0.08)


def add_colorbar(fig, ax, cmap, norm):
    """
    Add colorbar with paper-style label and tick sizes.
    """

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cb = fig.colorbar(
        sm,
        ax=ax,
        fraction=0.065,
        pad=0.10,
        aspect=16,
        ticks=[10, 15, 20, 25, 30, 34],
    )

    cb.set_label(r"$G$", labelpad=10)
    cb.ax.tick_params(labelsize=20, width=0.5, length=1)

    return cb


def main():
    V = np.load(CACHE)

    n = V.shape[0]
    spacing = L / (n - 1)

    print(f"G range: {V.min():.3f} to {V.max():.3f}")
    print(f"Grid size: {V.shape}")

    norm = mpl.colors.Normalize(vmin=VMIN, vmax=VMAX)
    cmap = plt.get_cmap(CMAP)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for level, alpha in zip(LEVELS, ALPHAS):
        add_isosurface(
            ax=ax,
            V=V,
            level=level,
            alpha=alpha,
            spacing=spacing,
            cmap=cmap,
            norm=norm,
        )

    format_3d_axes(ax)
    add_colorbar(fig, ax, cmap, norm)

    out_pdf = OUTDIR / "G_isosurface_clean_axes.pdf"
    out_png = OUTDIR / "G_isosurface_clean_axes.png"

    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    plt.close(fig)

    print("[saved]", out_pdf)
    print("[saved]", out_png)


if __name__ == "__main__":
    main()