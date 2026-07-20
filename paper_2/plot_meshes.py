"""Render the 2D and 3D VTK meshes used in paper_2.

Default outputs:
    mesh_plots/mesh_3d.pdf
    mesh_plots/mesh_2d.pdf
    mesh_plots/meshes_side_by_side.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import LineCollection


HERE = Path(__file__).resolve().parent
DEFAULT_3D_MESH = HERE / "3d-mesh" / "mesh.vtk"
DEFAULT_2D_MESH = HERE / "project_saramito" / "mesh.vtk"
DEFAULT_OUTDIR = HERE / "mesh_plots"
FACE_COLOR = "#c9c6ba"
EDGE_COLOR = "#101a9a"
EDGE_LINE_WIDTH = 0.7
BACKGROUND_COLOR = "white"


def _mesh_summary(mesh: pv.DataSet) -> str:
    bounds = ", ".join(f"{v:.4g}" for v in mesh.bounds)
    return (
        f"{mesh.n_points:,} points, {mesh.n_cells:,} cells\n"
        f"bounds: ({bounds})"
    )


def _surface_geometry(mesh: pv.DataSet, *, top_surface_only: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return projected points and triangular faces for a flat mesh plot."""
    surface = mesh.extract_surface(algorithm="dataset_surface")
    points = surface.points
    faces = surface.faces

    triangles: list[np.ndarray] = []
    index = 0
    while index < len(faces):
        n_vertices = int(faces[index])
        vertex_ids = faces[index + 1 : index + 1 + n_vertices]
        if n_vertices == 3:
            triangles.append(vertex_ids)
        elif n_vertices > 3:
            for i in range(1, n_vertices - 1):
                triangles.append(np.array([vertex_ids[0], vertex_ids[i], vertex_ids[i + 1]]))
        index += n_vertices + 1

    if not triangles:
        raise ValueError("No triangular surface faces were found in the mesh.")

    face_ids = np.asarray(triangles, dtype=int)
    if top_surface_only:
        z = points[:, 2]
        z_max = z.max()
        atol = max(1.0e-9, 1.0e-8 * max(1.0, abs(float(z_max))))
        keep = np.all(np.isclose(z[face_ids], z_max, atol=atol), axis=1)
        face_ids = face_ids[keep]
        if len(face_ids) == 0:
            raise ValueError("No top-surface triangles were found in the 3D mesh.")

    return points[:, :2], face_ids


def _geometry_bounds(points: np.ndarray, face_ids: np.ndarray) -> tuple[float, float, float, float]:
    xy = points[np.unique(face_ids.ravel())]
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    return float(x_min), float(x_max), float(y_min), float(y_max)


def _figure_size(points: np.ndarray, face_ids: np.ndarray, *, height: float = 6.5) -> tuple[float, float]:
    x_min, x_max, y_min, y_max = _geometry_bounds(points, face_ids)
    width = max(x_max - x_min, 1.0e-12)
    mesh_height = max(y_max - y_min, 1.0e-12)
    return height * width / mesh_height, height


def _unique_edge_segments(points: np.ndarray, face_ids: np.ndarray) -> np.ndarray:
    edge_ids = np.vstack(
        [
            face_ids[:, [0, 1]],
            face_ids[:, [1, 2]],
            face_ids[:, [2, 0]],
        ]
    )
    edge_ids = np.sort(edge_ids, axis=1)
    edge_ids = np.unique(edge_ids, axis=0)
    return points[edge_ids]


def _draw_projected_mesh(
    ax: plt.Axes,
    points: np.ndarray,
    face_ids: np.ndarray,
    *,
    title: str | None = None,
    summary: str | None = None,
    annotate: bool = False,
) -> None:
    collection = LineCollection(
        _unique_edge_segments(points, face_ids),
        colors=EDGE_COLOR,
        linewidths=EDGE_LINE_WIDTH,
        antialiaseds=True,
    )
    ax.add_collection(collection)

    x_min, x_max, y_min, y_max = _geometry_bounds(points, face_ids)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0)
    ax.set_axis_off()
    ax.set_facecolor(FACE_COLOR)

    if annotate and title is not None:
        ax.text(0.015, 0.985, title, transform=ax.transAxes, va="top", color="black", fontsize=12)
    if annotate and summary is not None:
        ax.text(0.015, 0.015, summary, transform=ax.transAxes, va="bottom", color="black", fontsize=8)


def _save_figure(fig: plt.Figure, output_path: Path, *, show: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)


def _trim_uniform_border(img: np.ndarray, *, tol: float = 1.0e-3) -> np.ndarray:
    """Crop the surrounding (near-white) background so the block fills the frame."""
    rgb = img[..., :3]
    content = np.any(rgb < (1.0 - tol), axis=-1)  # non-white pixels
    if not content.any():
        return img
    rows = np.where(content.any(axis=1))[0]
    cols = np.where(content.any(axis=0))[0]
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _save_png_as_pdf(png_path: Path, pdf_path: Path) -> None:
    """Wrap a rendered PNG in a tightly cropped PDF page."""
    img = _trim_uniform_border(plt.imread(png_path))
    height, width = img.shape[:2]
    dpi = 300
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_axis_off()
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _save_3d_perspective(
    mesh: pv.DataSet,
    output_path: Path,
    *,
    show: bool,
) -> None:
    """Render the 3D mesh as a block with front and side faces visible."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")

    surface = mesh.extract_surface(algorithm="dataset_surface")
    plotter = pv.Plotter(
        off_screen=not show,
        window_size=(1200, 1200),
        border=False,
    )
    plotter.set_background(BACKGROUND_COLOR)
    plotter.add_mesh(
        surface,
        color=FACE_COLOR,
        show_edges=True,
        edge_color=EDGE_COLOR,
        line_width=1.15,
        lighting=False,
        smooth_shading=False,
    )
    plotter.add_mesh(
        surface.extract_feature_edges(
            boundary_edges=True,
            feature_edges=True,
            manifold_edges=False,
            non_manifold_edges=False,
            feature_angle=20,
        ),
        color=EDGE_COLOR,
        line_width=1.6,
        lighting=False,
    )

    center = np.array(mesh.center)
    span = float(mesh.length)
    camera_position = (
        center + np.array([0.55 * span, -0.35 * span, 1.70 * span]),
        center,
        (0.0, 1.0, 0.0),
    )
    plotter.camera_position = camera_position
    plotter.camera.SetViewAngle(31.0)
    plotter.camera.zoom(0.95)
    plotter.disable_anti_aliasing()
    plotter.show(screenshot=str(png_path), auto_close=not show)

    if output_path.suffix.lower() == ".pdf":
        _save_png_as_pdf(png_path, output_path)


def _save_single(
    mesh: pv.DataSet,
    output_path: Path,
    title: str,
    *,
    is_3d: bool,
    show: bool,
    annotate: bool,
) -> None:
    if is_3d:
        _save_3d_perspective(mesh, output_path, show=show)
        return

    points, face_ids = _surface_geometry(mesh, top_surface_only=is_3d)
    fig, ax = plt.subplots(figsize=_figure_size(points, face_ids), dpi=300)
    fig.patch.set_facecolor(FACE_COLOR)
    _draw_projected_mesh(
        ax,
        points,
        face_ids,
        title=title,
        summary=_mesh_summary(mesh),
        annotate=annotate,
    )
    _save_figure(fig, output_path, show=show)


def _save_side_by_side(
    mesh_3d: pv.DataSet,
    mesh_2d: pv.DataSet,
    output_path: Path,
    *,
    show: bool,
    annotate: bool,
) -> None:
    points_3d, face_ids_3d = _surface_geometry(mesh_3d, top_surface_only=True)
    points_2d, face_ids_2d = _surface_geometry(mesh_2d, top_surface_only=False)
    width_3d, height_3d = _figure_size(points_3d, face_ids_3d)
    width_2d, height_2d = _figure_size(points_2d, face_ids_2d)
    height = max(height_3d, height_2d)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(height * (width_3d / height_3d + width_2d / height_2d), height),
        dpi=300,
        gridspec_kw={"width_ratios": [width_3d / height_3d, width_2d / height_2d]},
    )
    fig.patch.set_facecolor(FACE_COLOR)
    _draw_projected_mesh(
        axes[0],
        points_3d,
        face_ids_3d,
        title="3D mesh: paper_2/3d-mesh/mesh.vtk",
        summary=_mesh_summary(mesh_3d),
        annotate=annotate,
    )
    _draw_projected_mesh(
        axes[1],
        points_2d,
        face_ids_2d,
        title="2D mesh: paper_2/project_saramito/mesh.vtk",
        summary=_mesh_summary(mesh_2d),
        annotate=annotate,
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.015)
    _save_figure(fig, output_path, show=show)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the paper_2 VTK meshes.")
    parser.add_argument("--mesh-3d", type=Path, default=DEFAULT_3D_MESH)
    parser.add_argument("--mesh-2d", type=Path, default=DEFAULT_2D_MESH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the Matplotlib figures after saving the PDF files.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Add titles and mesh summaries to the rendered images.",
    )
    args = parser.parse_args()

    mesh_3d_path = args.mesh_3d.resolve()
    mesh_2d_path = args.mesh_2d.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    mesh_3d = pv.read(mesh_3d_path)
    mesh_2d = pv.read(mesh_2d_path)

    out_3d = outdir / "mesh_3d.pdf"
    out_2d = outdir / "mesh_2d.pdf"
    out_both = outdir / "meshes_side_by_side.pdf"

    _save_single(
        mesh_3d,
        out_3d,
        f"3D mesh: {mesh_3d_path.name}",
        is_3d=True,
        show=args.show,
        annotate=args.annotate,
    )
    _save_single(
        mesh_2d,
        out_2d,
        f"2D mesh: {mesh_2d_path.name}",
        is_3d=False,
        show=args.show,
        annotate=args.annotate,
    )
    _save_side_by_side(mesh_3d, mesh_2d, out_both, show=args.show, annotate=args.annotate)

    print("Saved mesh plots:")
    print(f"  3D:        {out_3d}")
    print(f"  2D:        {out_2d}")
    print(f"  side-by-side: {out_both}")


if __name__ == "__main__":
    main()
