"""Main Bayesian inference procedure."""

from __future__ import annotations

import numpy as np

from .data_io import DataIO
from .mcmc import Diagnostics
from .forward_models import ForwardModels
from .likelihood import Likelihood
from .plotting import Plotting
from .priors import Priors
from .transforms import Transforms
from .wall_corrections import WallCorrections


class InferenceProcedure(
    WallCorrections,
    ForwardModels,
    DataIO,
    Transforms,
    Priors,
    Likelihood,
    Diagnostics,
    Plotting,
):
    """Compose data loading, forward models, priors, likelihood, MCMC, and plots."""

    def __init__(
        self,
        force: float,
        a: float = 1.0,
        theta: float = 45.0,
        material_model: str = "newtonian",
        boundary_model: str = "bounded",
        delta0: float | None = None,
        L: float | None = None,
        t_unload: float | None = 0.2,
        eta_bounds: tuple[float, float] = (0.01, 100.0),
        delta_bounds: tuple[float, float] = (0.001, 100.0),
        eta1_bounds: tuple[float, float] = (0.01, 100.0),
        eta2_bounds: tuple[float, float] = (0.01, 100.0),
        lambda_bounds: tuple[float, float] = (0.01, 100.0),
        sigma_noise_percent: float | None = None,
        l_bias: float = 1.0,
        sigma_bias: float | str | None = "infer",
        burn_fraction: float = 0.3,
        nsteps: int = 5000,
        nwalkers: int | str = "auto",
        thin_factor: int = 1,
    ) -> None:
        theta = float(theta) % 180.0
        self.material_model = str(material_model).strip().lower()
        self.boundary_model = str(boundary_model).strip().lower()
        self.model = f"{self.material_model}_{self.boundary_model}"
        if self.boundary_model == "bounded" and np.isclose(theta, 90.0):
            self.model += "_perp"

        self.force = float(force)
        self.a = float(a)
        self.theta = theta
        self.delta0 = None if delta0 is None else float(delta0)
        self.L = None if L is None else float(L)
        self.t_unload = t_unload
        self._t_unload_eff = t_unload

        self.bounds = {
            "eta": eta_bounds,
            "delta": delta_bounds,
            "eta1": eta1_bounds,
            "eta2": eta2_bounds,
            "lambda_": lambda_bounds,
        }

        self.datasets = []
        self.sampler = None
        self.samples = None

        self.sigma_noise_percent = (
            None if sigma_noise_percent is None else float(sigma_noise_percent)
        )
        self.sigma_bias = sigma_bias
        self.l_bias = float(l_bias)

        self.burn_fraction = float(burn_fraction)
        self.nsteps = int(nsteps)
        self.thin_factor = int(thin_factor)

        self.ndim = self._get_ndim()
        self.nwalkers = 2 * self.ndim if nwalkers == "auto" else int(nwalkers)

    def _get_ndim(self) -> int:
        return len(self._get_parameter_bounds()) + 1 + int(self._bias_is_inferred())

    def _get_parameter_bounds(self) -> list[tuple[float, float]]:
        b = self.bounds
        return list(
            {   "newtonian_unbounded": [b["eta"]],
                "newtonian_bounded": [b["eta"], b["delta"]],
                "newtonian_bounded_perp": [b["eta"], b["delta"]],
                "newtonian_particle_particle": [b["eta"]],
                "viscoelastic_unbounded": [b["eta1"], b["eta2"], b["lambda_"]],
                "viscoelastic_bounded": [b["eta1"], b["eta2"], b["lambda_"], b["delta"]],
                "viscoelastic_bounded_perp": [b["eta1"], b["eta2"], b["lambda_"], b["delta"]],
            }[self.model]
        )

    def _get_base_physical_ndim(self) -> int:
        return len(self._get_parameter_bounds())

    def _get_parameter_labels(self, latex: bool = True) -> list[str]:
        labels = {
            "newtonian_unbounded": ["eta_s"],
            "newtonian_bounded": ["eta_s", "delta"],
            "newtonian_bounded_perp": ["eta_s", "delta"],
            "newtonian_particle_particle": ["eta_s"],
            "viscoelastic_unbounded": ["eta_s", "eta_p_eff", "lambda"],
            "viscoelastic_bounded": ["eta_s", "eta_p_eff", "lambda", "delta"],
            "viscoelastic_bounded_perp": ["eta_s", "eta_p_eff", "lambda", "delta"],
        }[self.model]

        labels = labels + ["sigma_noise"]
        if self._bias_is_inferred():
            labels.append("sigma_bias")

        if not latex:
            return labels

        latex_labels = {
            "eta_s": r"$\eta_{\mathrm{s}}$",
            "eta_p_eff": r"$\eta_{\mathrm{p}}$",
            "lambda": r"$\lambda$",
            "delta": r"$\delta$",
            "sigma_noise": r"$\sigma_{\mathrm{noise}}$",
            "sigma_bias": r"$\sigma_{\mathrm{bias}}$",
        }
        return [latex_labels[name] for name in labels]

    def _bias_is_inferred(self) -> bool:
        return self.sigma_bias == "infer"

    def _bias_is_disabled(self) -> bool:
        return self.sigma_bias is None

    def _get_delta0(self, theta: np.ndarray | None = None) -> float | None:
        if self.boundary_model != "bounded":
            return None
        if theta is not None:
            theta = np.asarray(theta, dtype=float)
            idx = 1 if self.material_model == "newtonian" else 3
            if theta.size > idx:
                return float(theta[idx])
        if self.delta0 is not None:
            return self.delta0
        raise ValueError("bounded model requires delta0 in theta or InferenceProcedure(..., delta0=...).")

    def _get_fixed_bias(self) -> float | None:
        if isinstance(self.sigma_bias, (int, float)):
            return float(self.sigma_bias)
        return None

    def _extract_noise_bias(self, theta: np.ndarray) -> tuple[float, float | None]:
        theta = np.asarray(theta, dtype=float)
        idx = len(self._get_parameter_bounds())
        sigma_noise = float(theta[idx])
        idx += 1
        sigma_bias = float(theta[idx]) if self._bias_is_inferred() else self._get_fixed_bias()
        return sigma_noise, sigma_bias


__all__ = ["InferenceProcedure"]
