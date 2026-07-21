"""Main Bayesian inference procedure."""

from __future__ import annotations

import numpy as np

from .data_io import DataIO
from .sampler import Sampler
from .forward_models import ForwardModels
from .likelihood import Likelihood
from .plotting import Plotting
from .priors import Priors
from .logtransforms import Transforms
from .wall_corrections import WallCorrections


_MODEL_PARAMETERS = {
    "newtonian_unbounded": ["eta_s"],
    "newtonian_bounded": ["eta_s", "delta0"],
    "newtonian_bounded_perp": ["eta_s", "delta0"],
    "viscoelastic_unbounded": ["eta_s", "eta_p", "lambda_"],
    "viscoelastic_bounded": ["eta_s", "eta_p", "lambda_", "delta0"],
    "viscoelastic_bounded_perp": ["eta_s", "eta_p", "lambda_", "delta0"],
}

_LATEX_LABELS = {
    "eta_s": r"$\eta_{\mathrm{s}}$",
    "eta_p": r"$\eta_{\mathrm{p}}$",
    "lambda_": r"$\lambda$",
    "delta0": r"$\delta_0$",
    "sigma_noise": r"$\sigma_{\mathrm{noise}}$",
    "sigma_bias": r"$\sigma_{\mathrm{bias}}$",
}


class InferenceProcedure(
    WallCorrections,
    ForwardModels,
    DataIO,
    Transforms,
    Priors,
    Likelihood,
    Sampler,
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
        t_unload: float | None = 0.2,
        eta_s_bounds: tuple[float, float] = (0.01, 100.0),
        eta_p_bounds: tuple[float, float] = (0.01, 100.0),
        lambda_bounds: tuple[float, float] = (0.01, 100.0),
        delta0_bounds: tuple[float, float] = (0.001, 100.0),
        sigma_noise_percent: float | None = None,
        seed: int | None = None,
        use_y: bool = False,
        l_bias: float = 1.0,
        sigma_bias: float | str | None = "infer",
        burn_fraction: float = 0.3,
        nsteps: int = 5000,
        nwalkers: int | str = "auto",
        thin_factor: int = 1,
    ) -> None:
        """Configure the inference problem.

        Parameters
        ----------
        force : float
            Applied force magnitude.
        a : float, optional
            Bead radius.
        theta : float, optional
            Pull angle in degrees (90 deg is perpendicular to the wall).
        material_model : {"newtonian", "viscoelastic"}, optional
            Rheological model.
        boundary_model : {"bounded", "unbounded"}, optional
            Whether wall corrections are applied.
        delta0 : float or None, optional
            Fixed initial wall gap when it is not inferred.
        t_unload : float or None, optional
            Load-removal time for viscoelastic datasets.
        eta_s_bounds, eta_p_bounds, lambda_bounds, delta0_bounds : tuple of float, optional
            ``(low, high)`` log-uniform prior bounds for each material
            parameter; both entries must be positive.
        sigma_noise_percent : float or None, optional
            Noise level as a percentage of peak displacement (used by data loading).
        seed : int or None, optional
            Seed for the synthetic measurement noise added in ``load_data``.
        use_y : bool, optional
            Also fit ``y`` in the parallel bounded viscoelastic case.
        l_bias : float, optional
            Correlation length of the model-discrepancy kernel.
        sigma_bias : float, "infer", or None, optional
            Discrepancy scale: a fixed value, ``"infer"`` to sample it, or
            ``None`` to disable it.
        burn_fraction : float, optional
            Fraction of each chain discarded as burn-in.
        nsteps : int, optional
            MCMC steps per walker.
        nwalkers : int or "auto", optional
            Walker count; ``"auto"`` uses ``2 * ndim``.
        thin_factor : int, optional
            Keep every ``thin_factor``-th sample when loading data.

        Returns
        -------
        None
        """
        self.theta = float(theta)
        self.material_model = str(material_model).strip().lower()
        self.boundary_model = str(boundary_model).strip().lower()
        self.model = f"{self.material_model}_{self.boundary_model}"
        if self.boundary_model == "bounded" and self.is_perpendicular():
            self.model += "_perp"

        self.force = float(force)
        self.a = float(a)
        self.delta0 = None if delta0 is None else float(delta0)
        self.t_unload = t_unload
        self._t_unload_eff = t_unload

        self.bounds = {
            "eta_s": eta_s_bounds,
            "eta_p": eta_p_bounds,
            "lambda_": lambda_bounds,
            "delta0": delta0_bounds,
        }
        for name, (low, high) in self.bounds.items():
            if not 0.0 < low < high:
                raise ValueError(f"{name}_bounds must satisfy 0 < low < high.")

        self.data = None
        self.sampler = None
        self.samples = None

        self.sigma_noise_percent = (
            None if sigma_noise_percent is None else float(sigma_noise_percent)
        )
        self.seed = seed
        self.use_y = bool(use_y)
        self.sigma_bias = sigma_bias
        self.l_bias = float(l_bias)

        self.burn_fraction = float(burn_fraction)
        self.nsteps = int(nsteps)
        self.thin_factor = int(thin_factor)

        self.ndim = self._get_ndim()
        self.nwalkers = 2 * self.ndim if nwalkers == "auto" else int(nwalkers)

    def _get_parameter_names(self) -> list[str]:
        """Material parameter names for the active model, in sampling order.

        Returns
        -------
        list of str
            Names shared by ``self.bounds``, ``_LATEX_LABELS``, and the
            forward-model signatures.

        Raises
        ------
        ValueError
            If ``self.model`` is not a recognised model string.
        """
        try:
            return _MODEL_PARAMETERS[self.model]
        except KeyError:
            raise ValueError(f"unknown model '{self.model}'.")

    def _get_ndim(self) -> int:
        """Total sampler dimension: material parameters + noise + optional bias.

        Returns
        -------
        int
            Number of sampled coordinates.
        """
        return len(self._get_parameter_bounds()) + 1 + int(self._bias_is_inferred())

    def _get_parameter_bounds(self) -> list[tuple[float, float]]:
        """Prior bounds for the active model's material parameters.

        Returns
        -------
        list of tuple of float
            ``(low, high)`` bounds in parameter order.
        """
        return [self.bounds[name] for name in self._get_parameter_names()]

    def _get_parameter_labels(self, latex: bool = True) -> list[str]:
        """Parameter labels for the active model, including noise and bias.

        Parameters
        ----------
        latex : bool, optional
            If ``True``, return LaTeX-formatted labels; otherwise plain names.

        Returns
        -------
        list of str
            One label per sampled parameter.
        """
        names = self._get_parameter_names() + ["sigma_noise"]
        if self._bias_is_inferred():
            names.append("sigma_bias")
        return [_LATEX_LABELS[name] for name in names] if latex else names

    def _bias_is_inferred(self) -> bool:
        """Whether ``sigma_bias`` is sampled.

        Returns
        -------
        bool
            ``True`` if ``sigma_bias == "infer"``.
        """
        return self.sigma_bias == "infer"

    def _bias_is_disabled(self) -> bool:
        """Whether the model-discrepancy term is turned off.

        Returns
        -------
        bool
            ``True`` if ``sigma_bias is None``.
        """
        return self.sigma_bias is None

    def is_perpendicular(self) -> bool:
        """Whether loading is perpendicular to the wall (``theta == 90 deg``).

        Returns
        -------
        bool
            ``True`` when only ``y(t)`` moves.
        """
        return abs(float(self.theta) - 90.0) < 1e-6

    def _get_delta0(self, theta: np.ndarray | None = None) -> float | None:
        """Resolve the initial wall gap ``delta0``.

        Prefers ``delta0`` carried in ``theta`` (when it is inferred), then the
        fixed ``self.delta0``.

        Parameters
        ----------
        theta : numpy.ndarray or None, optional
            Physical parameters that may include ``delta0``.

        Returns
        -------
        float or None
            The wall gap, or ``None`` for unbounded models.

        Raises
        ------
        ValueError
            If a bounded model has no ``delta0`` available.
        """
        if self.boundary_model != "bounded":
            return None
        if theta is not None:
            theta = np.asarray(theta, dtype=float)
            idx = self._get_parameter_names().index("delta0")
            if theta.size > idx:
                return float(theta[idx])
        if self.delta0 is not None:
            return self.delta0
        raise ValueError(
            "bounded model requires delta0 in theta or "
            "InferenceProcedure(..., delta0=...)."
        )

    def _get_fixed_bias(self) -> float | None:
        """Return ``sigma_bias`` when it is a fixed number.

        Returns
        -------
        float or None
            The fixed value, or ``None`` if bias is inferred or disabled.
        """
        if isinstance(self.sigma_bias, (int, float)):
            return float(self.sigma_bias)
        return None

    def _extract_noise_bias(self, theta: np.ndarray) -> tuple[float, float | None]:
        """Split noise and bias scales out of a physical parameter vector.

        Parameters
        ----------
        theta : numpy.ndarray
            Physical parameters, ordered as material parameters, then
            ``sigma_noise``, then ``sigma_bias`` (if inferred).

        Returns
        -------
        sigma_noise : float
            Measurement-noise scale.
        sigma_bias : float or None
            Discrepancy scale (inferred value, fixed value, or ``None``).
        """
        theta = np.asarray(theta, dtype=float)
        idx = len(self._get_parameter_bounds())
        sigma_noise = float(theta[idx])
        sigma_bias = (
            float(theta[idx + 1]) if self._bias_is_inferred() else self._get_fixed_bias()
        )
        return sigma_noise, sigma_bias


__all__ = ["InferenceProcedure"]