"""Joint inference over several trajectories sharing one parameter set."""

from contextlib import contextmanager

import numpy as np

from .main import InferenceProcedure

_PER_DATASET = ("force", "theta", "boundary_model", "t_unload")


class JointInferenceProcedure(InferenceProcedure):
    """Fit several trajectories that share eta_s, eta_p, lambda_, delta0."""

    def load_datasets(self, items):
        """Load several datasets, each with its own experimental settings.

        Parameters
        ----------
        items : sequence of dict
            One dict per dataset with key ``"filename"``, plus any of
            ``"force"``, ``"theta"``, ``"boundary_model"`` and ``"t_unload"``
            to override the constructor defaults.

        Returns
        -------
        list of dict
            The loaded datasets, also stored on ``self.datasets``.
        """
        defaults = {name: getattr(self, name) for name in _PER_DATASET}
        self.datasets = []

        for item in items:
            config = {**defaults, **{k: v for k, v in item.items() if k != "filename"}}
            with self._configured(config):
                self._t_unload_eff = config["t_unload"]
                self.load_data(item["filename"])
            self.data["config"] = config
            self.datasets.append(self.data)

        # Shared noise/bias priors from the largest peak across all datasets, so
        # they are not set by whichever file happened to be read last.
        peak = max(
            np.max(np.abs(d[component]))
            for d in self.datasets
            for component in d["fit_components"]
        )
        self.beta = 1.0 / (0.10 * peak)
        self.sigma_noise_prior = self.sigma_bias_prior = self.beta

        self.data = self.datasets[0]
        print(f"\nJoint fit over {len(self.datasets)} datasets:")
        for d in self.datasets:
            c = d["config"]
            print(f"   F={c['force']:.4g}  theta={c['theta']}  "
                  f"t_unload={c['t_unload']}  {c['boundary_model']}  "
                  f"n={len(d['t'])}  fit={','.join(d['fit_components'])}")
        return self.datasets

    @contextmanager
    def _configured(self, config):
        """Temporarily apply one dataset's experimental settings."""
        saved = {name: getattr(self, name) for name in _PER_DATASET}
        try:
            for name in _PER_DATASET:
                setattr(self, name, config[name])
            yield
        finally:
            for name, value in saved.items():
                setattr(self, name, value)

    def select(self, index):
        """Make one dataset current, for the per-dataset plotting methods.

        Parameters
        ----------
        index : int
            Position in ``self.datasets``.

        Returns
        -------
        JointInferenceProcedure
            ``self``, so calls can be chained.
        """
        self.data = self.datasets[index]
        config = self.data["config"]
        for name in _PER_DATASET:
            setattr(self, name, config[name])
        self._t_unload_eff = config["t_unload"]
        return self

    def log_likelihood(self, phi):
        """Total log-likelihood, summed over the loaded datasets.

        Parameters
        ----------
        phi : numpy.ndarray
            Parameters in sampler space.

        Returns
        -------
        float
            Summed log-likelihood, or ``-inf`` if any dataset is non-finite.
        """
        current, total = self.data, 0.0
        try:
            for d in self.datasets:
                self.data = d
                with self._configured(d["config"]):
                    self._t_unload_eff = d["config"]["t_unload"]
                    value = super().log_likelihood(phi)
                if not np.isfinite(value):
                    return -np.inf
                total += value
        finally:
            self.data = current
        return float(total)


__all__ = ["JointInferenceProcedure"]
