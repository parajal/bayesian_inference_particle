"""Data-loading mixin: ``load_data`` and its helpers."""

import os
import numpy as np


def _recenter(values):
    """Shift a displacement series so it starts at zero.

    Parameters
    ----------
    values : array_like
        Displacement samples.

    Returns
    -------
    numpy.ndarray
        values with its first element subtracted.
    """
    return values - values[0]


def _max_disp(values):
    """Return the largest absolute displacement in a series.

    Parameters
    ----------
    values : array_like
        Displacement samples.

    Returns
    -------
    float
        max(abs(values)).
    """
    return float(np.max(np.abs(values)))


def _realized_sigma(values):
    """Sample standard deviation (``ddof=1``) of a series.

    Parameters
    ----------
    values : array_like
        Samples.

    Returns
    -------
    float
        The sample standard deviation.
    """
    return float(np.std(values, ddof=1))


class DataIO:
    """Load one dataset, thin it, add experimental noise, and set priors."""

    def _parse_columns(self, data, use_y):
        """Split a raw data array into time and displacement components.

        Parameters
        ----------
        data : numpy.ndarray
            Array with columns ``[t, x, y]``.
        use_y : bool
            Whether to also fit the ``y`` component (parallel bounded
            viscoelastic case only).

        Returns
        -------
        t : numpy.ndarray
            Time column.
        x, y : numpy.ndarray or None
            Displacement components; the unused one is ``None``.
        components : tuple of str
            Names of the components to fit.
        """
        t = np.asarray(data[:, 0], dtype=float)
        x = y = None

        if self.is_perpendicular():
            y = _recenter(data[:, 2])
            return t, x, y, ("y",)

        x = np.asarray(data[:, 1], dtype=float)
        fit_y = (use_y and self.material_model == "viscoelastic"
                 and self.boundary_model == "bounded")
        if fit_y:
            y = _recenter(data[:, 2])
            return t, x, y, ("x", "y")
        return t, x, y, ("x",)

    def _add_noise(self, d, seed=None):
        """Add Gaussian experimental noise to each fitted component in place.

        Noise per component is drawn from ``N(0, target_sigma)`` where
        ``target_sigma`` is ``sigma_noise_percent`` of that component's peak
        displacement. Realized and target sigmas are recorded on ``d``. The
        noise-free series is kept as ``d["<component>_clean"]``, so the model
        can be compared against the original FOM data.

        Parameters
        ----------
        d : dict
            Dataset dict; mutated in place.
        seed : int or None, optional
            Seed for the random generator.

        Returns
        -------
        None
        """
        rng = np.random.default_rng(seed)
        d["sigma_noise_target"] = {}
        d["sigma_noise_realized"] = {}
        d["max_displacement"] = {}
        all_noise = []

        for component in d["fit_components"]:
            clean = d[component]
            target_sigma = self.sigma_noise_percent / 100.0 * _max_disp(clean)
            noise = rng.normal(0.0, target_sigma, len(clean))

            d[component] = clean + noise
            d[f"{component}_clean"] = clean
            d[f"{component}_noise"] = noise
            d["max_displacement"][component] = _max_disp(clean)
            d["sigma_noise_target"][component] = target_sigma
            d["sigma_noise_realized"][component] = _realized_sigma(noise)
            all_noise.append(noise)

        d["sigma_noise_realized_pooled"] = _realized_sigma(np.concatenate(all_noise))
        self.sigma_noise_added = d["sigma_noise_target"]
        self.sigma_noise_true = d["sigma_noise_realized_pooled"]

    def _sigma_priors_from_data(self, d):
        """Set exponential noise/bias priors from the dataset's peak displacement.

        The prior rate ``beta`` is ``1 / (0.10 * peak)``, where ``peak`` is the
        largest peak displacement across fitted components.

        Parameters
        ----------
        d : dict
            Dataset dict; ``sigma_prior_beta`` is added in place.

        Returns
        -------
        None
        """
        peak = max(_max_disp(d[component]) for component in d["fit_components"])
        self.beta = 1.0 / (0.10 * peak)
        self.sigma_noise_prior = self.beta
        self.sigma_bias_prior = self.beta
        d["sigma_prior_beta"] = self.beta

    def load_data(self, filename):
        """Load, thin, noise, store, and report one dataset.

        Reads a whitespace-delimited file relative to the project root, thins it
        by ``self.thin_factor``, adds experimental noise, derives sigma priors,
        and stores the result on ``self.data``.

        Parameters
        ----------
        filename : str
            Path to the data file, relative to the project root (two levels
            above this module).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``t_unload`` is not finite for a viscoelastic dataset, or if
            ``thin_factor`` is not a positive integer.
        """
        self._t_unload_eff = self.t_unload
        if self.material_model == "viscoelastic":
            if self.t_unload is None or not np.isfinite(float(self.t_unload)):
                raise ValueError("t_unload must be finite for viscoelastic datasets.")

        if self.thin_factor < 1:
            raise ValueError("thin_factor must be a positive integer.")

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, filename)
        data = np.atleast_2d(np.loadtxt(path))

        Fx = self.force * np.cos(np.deg2rad(self.theta))
        Fy = self.force * np.sin(np.deg2rad(self.theta))

        t, x, y, components = self._parse_columns(data, self.use_y)
        t, x, y = (None if s is None else s[::self.thin_factor] for s in (t, x, y))

        d = {
            "t": t, "x": x, "y": y,
            "F": self.force, "Fx": Fx, "Fy": Fy,
            "angle": self.theta, "fit_components": components,
        }
        if self.material_model == "viscoelastic":
            d["t_unload"] = self.t_unload

        self._add_noise(d, seed=self.seed)
        self._sigma_priors_from_data(d)
        self.data = d

        self._report(path, d, components)

    def _report(self, path, d, components):
        """Print a summary of the loaded dataset.

        Parameters
        ----------
        path : str
            Full path of the file that was loaded.
        d : dict
            The populated dataset dict.
        components : tuple of str
            Names of the fitted components.

        Returns
        -------
        None
        """
        print(
            f"Loaded {os.path.basename(path)}: angle={self.theta:.1f}\u00b0, "
            f"Fx={d['Fx']:.3e}, Fy={d['Fy']:.3e}, "
            f"components={','.join(components)}, n_points={len(d['t'])}"
        )
        for component in components:
            print(f"maximum displacement ({component}): "
                  f"{d['max_displacement'][component]:.4g}")
            print(f"sigma_noise added ({component}): "
                  f"{d['sigma_noise_target'][component]:.4g} "
                  f"({self.sigma_noise_percent}% of maximum displacement)")
            print(f"realized experimental noise ({component}): "
                  f"{d['sigma_noise_realized'][component]:.4g}")
        print(f"sigma_noise prior: Exp(beta={self.beta:.4g})")
        if self._bias_is_inferred():
            print(f"sigma_bias prior: Exp(beta={self.beta:.4g})")
            print(f"\nl_bias fixed at {self.l_bias:.4g}")