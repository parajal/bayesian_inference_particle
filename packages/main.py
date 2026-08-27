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
    WallCorrections, ForwardModels, DataIO, Transforms,
    Priors, Likelihood, Sampler, Plotting,
):

    def __init__(
        self, force, a=1.0, theta=45.0,
        material_model="newtonian", boundary_model="bounded",
        delta0=None, t_unload=0.2, theta_true=None,
        hasimoto_corr=False, L=None,
        eta_s_bounds=(0.01, 100.0), eta_p_bounds=(0.01, 100.0),
        lambda_bounds=(0.01, 100.0), delta0_bounds=(1e-3, 100.0),
        sigma_noise_percent=2.0, seed=42, use_y=False,
        l_bias=1.0, sigma_bias="infer", burn_fraction=0.3,
        nsteps=10000, nwalkers="auto", thin_factor=1,
        rtol=1e-7, atol=1e-9, n_brenner_terms=100,
    ):
        self.theta = float(theta)
        self.material_model = material_model.lower()
        self.boundary_model = boundary_model.lower()

        self.model = f"{self.material_model}_{self.boundary_model}"
        if self.boundary_model == "bounded" and self.is_perpendicular():
            self.model += "_perp"

        self.force = float(force)
        self.a = float(a)
        self.delta0 = None if delta0 is None else float(delta0)
        self.t_unload = t_unload
        self._t_unload_eff = t_unload
        self.theta_true = None if theta_true is None else list(theta_true)
        self.hasimoto_corr = bool(hasimoto_corr)
        self.L = None if L is None else float(L)

        self.bounds = {
            "eta_s": eta_s_bounds, "eta_p": eta_p_bounds,
            "lambda_": lambda_bounds, "delta0": delta0_bounds,
        }
        for name, (lo, hi) in self.bounds.items():
            if not 0 < lo < hi:
                raise ValueError(f"{name}_bounds must satisfy 0 < low < high.")

        self.data = self.sampler = self.samples = None

        self.sigma_noise_percent = sigma_noise_percent
        self.seed = seed
        self.use_y = bool(use_y)
        self.sigma_bias = sigma_bias
        self.l_bias = float(l_bias)
        self.burn_fraction = float(burn_fraction)
        self.nsteps = int(nsteps)
        self.thin_factor = int(thin_factor)

        self.ndim = self._get_ndim()
        self.nwalkers = 2 * self.ndim if nwalkers == "auto" else int(nwalkers)

        self.rtol = rtol
        self.atol = atol
        self.n_brenner_terms = n_brenner_terms

    def _get_parameter_names(self):
        try:
            return _MODEL_PARAMETERS[self.model]
        except KeyError:
            raise ValueError(f"unknown model '{self.model}'")

    def _get_parameter_bounds(self):
        return [self.bounds[p] for p in self._get_parameter_names()]

    def _get_ndim(self):
        return len(self._get_parameter_bounds()) + 1 + int(self._bias_is_inferred())

    def _get_parameter_labels(self, latex=True):
        names = self._get_parameter_names() + ["sigma_noise"]
        if self._bias_is_inferred():
            names.append("sigma_bias")
        return [_LATEX_LABELS[n] for n in names] if latex else names

    def _bias_is_inferred(self):
        return self.sigma_bias == "infer"

    def is_perpendicular(self):
        return abs(self.theta - 90.0) < 1e-6

    def _get_delta0(self, theta=None):
        if self.boundary_model != "bounded":
            return None

        if theta is not None:
            idx = self._get_parameter_names().index("delta0")
            theta = np.asarray(theta)
            if len(theta) > idx:
                return float(theta[idx])

        if self.delta0 is not None:
            return self.delta0

        raise ValueError("bounded model requires delta0.")

    def _extract_noise_bias(self, theta):
        idx = len(self._get_parameter_bounds())
        sigma_noise = float(theta[idx])
        sigma_bias = float(theta[idx + 1]) if self._bias_is_inferred() else None
        return sigma_noise, sigma_bias