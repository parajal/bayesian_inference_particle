"""
Data-loading mixin: ``load_data`` and all its local helpers.
"""
import os
from typing import Callable, Optional
import numpy as np


def _recenter_displacement(values: np.ndarray) -> np.ndarray:
    """Convert an absolute displacement-like series to displacement from its initial value."""
    values = np.asarray(values, dtype=float)
    return values - values[0]


def _resolve_data_file(fname: str, data_dir: str) -> str:
    """Resolve ``fname`` to an existing file (relative paths are looked up in the data folder)."""
    path = fname if os.path.isabs(fname) else os.path.join(data_dir, fname)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(
        f"File not found: {fname}\n"
        "Searched in data folder:\n"
        f"  - {path}"
    )


def _thin_data(
    t: np.ndarray,
    x: np.ndarray | None,
    y: np.ndarray | None,
    thin_factor: int,
    extras: list[np.ndarray | None],
    info: Callable[[str], None],
):
    """Keep every ``thin_factor``-th sample of each series (no-op when factor <= 1)."""
    if thin_factor <= 1:
        return t, x, y, extras

    t = t[::thin_factor]
    x = x[::thin_factor] if x is not None else None
    y = y[::thin_factor] if y is not None else None
    extras = [item[::thin_factor] if item is not None else None for item in extras]
    info(f"  Thinning (factor={thin_factor}): kept {len(t)} points")

    return t, x, y, extras


class DataIO:
    """Dataset loading, thinning, and noise injection."""

    @staticmethod
    def _is_perpendicular_angle(angle: float, tol: float = 1e-6) -> bool:
        """Return True if angle corresponds to perpendicular (≈90°) direction."""
        if angle is None:
            return False
        angle_mod = angle % 180.0
        return abs(angle_mod - 90.0) < tol

    def _max_fitted_displacement(self) -> float:
        peak = 0.0

        for d in self.datasets:
            for name in ("x", "y"):
                values = d.get(f"{name}_clean")
                if values is None:
                    values = d.get(name)
                if values is None:
                    continue

                values = np.asarray(values, dtype=float)
                values = values[np.isfinite(values)]

                if values.size:
                    peak = max(peak, np.max(np.abs(values)))

        return float(peak)

    def _sigma_priors_from_data(self, info: Callable[[str], None]) -> None:
        """Set beta from 1/beta = 10% of the maximum fitted displacement."""
        peak = self._max_fitted_displacement()
        target_mean = 0.10 * peak
        beta = 1.0 / target_mean
        self.beta = float(beta)
        self.sigma_noise_prior = float(beta)
        self.sigma_bias_prior = float(beta)

        info(
            "Automatic sigma prior scale: "
            f"max fitted displacement={peak:.6g}, "
            f"10% max displacement={target_mean:.6g}, "
            f"beta=1/(10% max displacement)={beta:.6g}"
        )

    def _parse_columns(self, data, ncols, base, t_unload, L_data):
        """Extract the observed displacement for the active model.

        Perpendicular models observe the y-displacement; all others observe x.
        Only the observed series is kept (the other is ``None``). Returns
        ``t, x, y`` plus a dict of model-specific dataset fields and a flag for
        whether perpendicular data must supply a y column.
        """
        t = data[:, 0]
        x = None
        y = None
        extra: dict = {}
        require_perp_y = self._is_perpendicular()

        if require_perp_y:
            # Perpendicular (90°): observed y-displacement is column 2, recentered.
            y = _recenter_displacement(data[:, 2])
        else:
            x = data[:, 1]

        if self.material_model == "viscoelastic":
            extra["t_unload"] = t_unload
        elif self.model == "newtonian_particle_particle":
            extra["L"] = L_data

        return t, x, y, extra, require_perp_y

    def load_data(
        self,
        *filenames: str,
        force: Optional[float] = None,
        t_unload: Optional[float] = None,
        append: bool = False,
        seed: Optional[int] = 42,
        thin_factor: Optional[int] = None,
    ) -> None:

        info = print
        self._t_unload_eff = self.t_unload
        t_unload = self.t_unload if t_unload is None else t_unload

        thin = int(thin_factor) if thin_factor is not None else int(self.thin_factor)

        if not append:
            self.datasets.clear()
            self.sigma_noise_true = None

        # Resolve the applied force and split it along the loading angle.
        force = force if force is not None else self.force
        angle = float(getattr(self, "theta", 0.0))
        theta_rad = np.deg2rad(angle)
        Fx = force * np.cos(theta_rad)
        Fy = force * np.sin(theta_rad)

        # Viscoelastic datasets require a finite unloading time.
        if self.material_model == "viscoelastic":
            if t_unload is None:
                raise ValueError("t_unload must be specified for viscoelastic datasets")
            t_unload = float(t_unload)
            if not np.isfinite(t_unload):
                raise ValueError("t_unload must be finite")

        n_existing = len(self.datasets)
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
        )

        for fname in filenames:
            fpath = _resolve_data_file(fname, data_dir)
            try:
                data = np.loadtxt(fpath, comments="#")
            except Exception as e:
                raise ValueError(f"Failed to load numeric data from '{fpath}': {e}")

            if data.ndim == 1:
                data = np.atleast_2d(data)

            ncols = data.shape[1]

            base = os.path.basename(fpath)

            # Hasimoto domain length is only needed by the particle-particle model.
            L_data = None
            if self.model == "newtonian_particle_particle":
                L_data = self.L
                if L_data is None or not np.isfinite(L_data) or L_data <= 0.0:
                    raise ValueError(
                        "particle_particle requires a positive L in "
                        "InferenceProcedure(..., L=...)."
                    )
                info(f"  Hasimoto domain length: L/a = {L_data:g}")

            t, x, y, extra, require_perp_y = self._parse_columns(
                data, ncols, base, t_unload, L_data
            )

            t, x, y, _ = _thin_data(t, x, y, thin, [], info)

            if require_perp_y and self.model.endswith("_perp") and y is None:
                raise ValueError(
                    f"Perpendicular (90°) data requires a y-displacement column in '{base}'."
                )

            self.datasets.append(
                dict(
                    t=t, x=x, y=y,
                    F=force, Fx=Fx, Fy=Fy, angle=angle,
                    filename=fname, thin_factor=thin,
                    **extra,
                )
            )
            info(
                f"Loaded {base}: angle={angle:.1f}°, Fx={Fx:.3e}, Fy={Fy:.3e}, "
                f"n_points={len(t)}"
            )

        # Keep a pristine copy of each new series before any noise is added.
        new_datasets = self.datasets[n_existing:]
        for d in new_datasets:
            if d.get("x") is not None:
                d["x_clean"] = d["x"].copy()
            if d.get("y") is not None:
                d["y_clean"] = d["y"].copy()

        use_percent = getattr(self, "sigma_noise_percent", None) is not None
        if use_percent and new_datasets:
            peak = 0.0
            for d in new_datasets:
                obs = d.get("y") if self._is_perpendicular() else d.get("x")
                if obs is not None and len(obs) > 0:
                    peak = max(peak, float(np.max(np.abs(obs))))

            noise_level = (self.sigma_noise_percent / 100.0) * peak
            self.sigma_noise_added = noise_level
            if noise_level > 0:
                rng = np.random.default_rng(seed)
                for d in new_datasets:
                    n = len(d["t"])
                    tmp = rng.normal(0, noise_level, n)
                    d["noise_added"] = tmp.copy()
                    if d.get("x") is not None:
                        d["x"] = d["x"] + tmp
                    if d.get("y") is not None:
                        d["y"] = d["y"] + tmp

                all_noise = np.concatenate(
                    [d["noise_added"] for d in self.datasets if "noise_added" in d]
                )
                self.sigma_noise_true = float(
                    np.std(all_noise, ddof=1 if all_noise.size > 1 else 0)
                )
                info(
                    f"Added experimental noise: {self.sigma_noise_percent:.4g}% of "
                    f"peak |data|={peak:.4g} -> sigma_noise_added={noise_level:.4g}, "
                    f"realized sigma_noise={self.sigma_noise_true:.4g} ")
            else:
                self.sigma_noise_true = 0.0
        elif new_datasets:
            self.sigma_noise_added = 0.0
            self.sigma_noise_true = None

        self._sigma_priors_from_data(info)

        info(f"sigma_noise prior: Exp(beta={self.beta:.4g})")

        if self._bias_is_inferred():
            info(f"sigma_bias prior: Exp(beta={self.beta:.4g})")
            info(f"\nl_bias fixed at {self.l_bias:.4g}")
