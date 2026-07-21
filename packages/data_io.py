"""Data-loading mixin: ``load_data`` and its helpers."""

import os
import numpy as np

def _recenter(values):
    """Shift a displacement series to start from zero."""
    return values - values[0]

def _max_disp(values) -> float:
    """Maximum displacement in a series."""
    return float(np.max(np.abs(values)))

def _thin(values, factor):
    """Keep every ``factor``-th sample."""
    if values is None:
        return None
    return values[::factor]

def _realized_sigma(values) -> float:
    """Sample standard deviation of the finite values."""
    return float(np.std(values, ddof=1))

class DataIO:
    """Load one dataset, thin it, add experimental noise, and set priors."""

    def _parse_columns(self, data, use_y):
        t = np.asarray(data[:, 0], dtype=float)
        x = y = None

        if self.is_perpendicular():
            y = _recenter(data[:, 2])
            components = ("y",)
        else:
            x = np.asarray(data[:, 1], dtype=float)
            components = ("x",)
            if use_y and self.material_model == "viscoelastic" and self.boundary_model == "bounded":
                y = _recenter(data[:, 2])
                components = ("x", "y")

        return t, x, y, components

    def _add_noise(self, d, seed=None) -> None:
        rng = np.random.default_rng(seed)
        all_noise = []
        d["sigma_noise_target"] = {}
        d["sigma_noise_realized"] = {}
        d["max_displacement"] = {}

        for component in d["fit_components"]:
            clean = d[component]
            target_sigma = self.sigma_noise_percent / 100.0 * _max_disp(clean)
            noise = rng.normal(0.0, target_sigma, len(clean))

            d[component] = clean + noise
            d[f"{component}_noise"] = noise
            d["max_displacement"][component] = _max_disp(clean)
            d["sigma_noise_target"][component] = target_sigma
            d["sigma_noise_realized"][component] = _realized_sigma(noise)
            all_noise.append(noise)

        pooled = np.concatenate(all_noise)
        d["sigma_noise_realized_pooled"] = _realized_sigma(pooled)

        self.sigma_noise_added = d["sigma_noise_target"]
        self.sigma_noise_true = d["sigma_noise_realized_pooled"]

    def _sigma_priors_from_data(self, d) -> None:
        peaks = [_max_disp(d[component]) for component in d["fit_components"]]
        peak = max(peaks)
        self.beta = 1.0 / (0.10 * peak)
        self.sigma_noise_prior = self.beta
        self.sigma_bias_prior = self.beta
        d["sigma_prior_beta"] = self.beta

    def load_data(self, filename: str) -> None:
        """Load, thin, noise, store, and report one dataset."""
        self._t_unload_eff = self.t_unload
        if self.material_model == "viscoelastic":
            if self.t_unload is None or not np.isfinite(float(self.t_unload)):
                raise ValueError("t_unload must be finite for viscoelastic datasets.")

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, filename)
        data = np.atleast_2d(np.loadtxt(path))

        Fx = self.force * np.cos(np.deg2rad(self.theta))
        Fy = self.force * np.sin(np.deg2rad(self.theta))

        thin_factor = int(getattr(self, "thin_factor", 1) or 1)
        if thin_factor < 1:
            raise ValueError("thin_factor must be a positive integer.")

        t, x, y, components = self._parse_columns(data, self.use_y)
        t, x, y = (_thin(series, thin_factor) for series in (t, x, y))

        d = {
            "t": t,
            "x": x,
            "y": y,
            "F": self.force,
            "Fx": Fx,
            "Fy": Fy,
            "angle": self.theta,
            "fit_components": components,
        }

        if self.material_model == "viscoelastic":
            d["t_unload"] = self.t_unload
        elif self.model == "newtonian_particle_particle":
            if self.L is None or not np.isfinite(self.L) or self.L <= 0.0:
                raise ValueError(
                    "particle_particle requires a positive L in "
                    "InferenceProcedure(..., L=...)."
                )
            d["L"] = self.L

        self._add_noise(d, seed=getattr(self, "seed", None))
        self._sigma_priors_from_data(d)

        self.data = d

        print(
            f"Loaded {os.path.basename(path)}: angle={self.theta:.1f}\u00b0, "
            f"Fx={Fx:.3e}, Fy={Fy:.3e}, "
            f"components={','.join(components)}, n_points={len(t)}"
        )
        for component in components:
            maximum = d["max_displacement"][component]
            added = d["sigma_noise_target"][component]
            realized = d["sigma_noise_realized"][component]
            print(f"maximum displacement ({component}): {maximum:.4g}")
            print(
                f"sigma_noise added ({component}): {added:.4g} "
                f"({self.sigma_noise_percent}% of maximum displacement)"
            )
            print(f"realized experimental noise ({component}): {realized:.4g}")
        print(f"sigma_noise prior: Exp(beta={self.beta:.4g})")
        if self._bias_is_inferred():
            print(f"sigma_bias prior: Exp(beta={self.beta:.4g})")
            print(f"\nl_bias fixed at {self.l_bias:.4g}")
