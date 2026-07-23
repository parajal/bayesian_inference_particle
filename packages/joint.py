"""Joint inference over several trajectories sharing one parameter set."""

from contextlib import contextmanager
import numpy as np
from .main import InferenceProcedure

_PER_DATASET = ("force", "theta", "boundary_model", "t_unload", "hasimoto_corr", "L")

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

class JointInferenceProcedure(InferenceProcedure):
    """Fit several trajectories that share eta_s, eta_p, lambda_, delta0."""

    def load_datasets(self, items):
        """Load several datasets, each with its own experimental settings.

        Parameters
        ----------
        items : sequence of dict
            One dict per dataset with key ``"filename"``, plus any of
            ``"force"``, ``"theta"``, ``"boundary_model"``, ``"t_unload"``,
            ``"hasimoto_corr"``, ``"L"`` and ``"thin_factor"`` to override the
            constructor defaults. Use ``"L"`` with ``"hasimoto_corr": True`` to
            give each periodic box its own Hasimoto correction, and
            ``"thin_factor"`` to thin dense files more than sparse ones.

        Returns
        -------
        list of dict
            The loaded datasets, also stored on ``self.datasets``.
        """
        defaults = {name: getattr(self, name) for name in _PER_DATASET}
        self.datasets = []

        for item in items:
            overrides = {k: v for k, v in item.items() if k != "filename"}
            thin_factor = int(overrides.pop("thin_factor", self.thin_factor))
            config = {**defaults, **overrides}
            with self._configured(config):
                self._t_unload_eff = config["t_unload"]
                saved_thin, self.thin_factor = self.thin_factor, thin_factor
                try:
                    self.load_data(item["filename"])
                finally:
                    self.thin_factor = saved_thin
            self.data["config"] = {**config, "thin_factor": thin_factor}
            self.datasets.append(self.data)

        # Shared noise/bias priors from the largest peak across all datasets, so
        # they are not set by whichever file happened to be read last.
        peak = max(
            d["max_displacement"][component]
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
                  f"thin={c['thin_factor']}  n={len(d['t'])}  "
                  f"fit={','.join(d['fit_components'])}")
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

    def _dataset_labels(self):
        """Legend labels naming whichever settings differ across datasets.

        Returns
        -------
        list of str
            One label per dataset; falls back to ``"dataset i"`` when the
            configs are identical.
        """
        configs = [d["config"] for d in self.datasets]
        varying = [k for k in ("force", "theta", "t_unload", "L", "boundary_model")
                   if len({c[k] for c in configs}) > 1]

        def fmt(config):
            parts = []
            for k in varying:
                v = config[k]
                if k == "force":
                    parts.append(rf"$F = {v / np.pi:g}\pi$")
                elif k == "theta":
                    parts.append(rf"$\theta = {v:g}^\circ$")
                elif k == "t_unload":
                    parts.append(rf"$t_0 = {v:g}$")
                elif k == "L":
                    parts.append(rf"$L = {v:g}$")
                else:
                    parts.append(str(v))
            return ", ".join(parts)

        return [fmt(c) or f"dataset {i}" for i, c in enumerate(configs)]

    def plot_data(self, theta_true=None) -> None:
        """Overlay every dataset's data (FOM points) and model (SAM line).

        One figure per fitted component; each dataset gets its own colour.

        Parameters
        ----------
        theta_true : sequence of float, optional
            Physical parameters at which the model curves are drawn. Defaults to
            the ``theta_true`` set on the model.

        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt

        theta_true = self._resolve_theta_true(theta_true)
        if theta_true is None:
            raise ValueError("plot_data needs theta_true (pass it or set it on the model).")
        labels = self._dataset_labels()
        lw = 3 if self.material_model == "newtonian" else 2

        for component in self._all_components():
            fig, ax = plt.subplots(figsize=(8, 6))
            t_all, y_all = [], []
            for i, _ in enumerate(self.datasets):
                self.select(i)
                d = self.data
                obs = d.get(component)
                if obs is None:
                    continue
                color = _COLORS[i % len(_COLORS)]
                t_model = (np.linspace(0.0, max(d["t"]), 400)
                           if self.model == "viscoelastic_bounded" else d["t"])
                model = self._model_component(theta_true, t_model, d, component=component)
                ax.scatter(d["t"], obs, color=color, s=30, marker="o",
                           edgecolors="black", linewidths=0.5, zorder=3)
                ax.plot(t_model, model, color=color, lw=lw, zorder=2, label=labels[i])
                t_all.append(d["t"])
                y_all.append(np.concatenate([np.ravel(obs), np.ravel(model)]))

            ax.set(xlabel=r"$t$", ylabel=rf"${component}(t)$")
            if y_all:
                self._set_axes(ax, np.concatenate(t_all), np.concatenate(y_all),
                               key=component, store=True)
            ax.grid(alpha=0.3)
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                      framealpha=0.95, borderaxespad=0.0)
            self._save_current_figure(f"data_vs_noise_{component}"
                                      if len(self._all_components()) > 1 else "data_vs_noise")
            plt.show()

    def plot_posterior_predictive(self, n_sigma=1.96, nsamples_pred=5000,
                                  logx=False, logy=False,
                                  condition_discrepancy=False) -> None:
        """Overlay the posterior predictive band of every dataset in one figure.

        One figure per fitted component; each dataset gets its own colour, with
        its data as points and its predictive band shaded.

        Parameters
        ----------
        n_sigma, nsamples_pred, condition_discrepancy
            As in :meth:`Plotting.plot_posterior_predictive`.
        logx, logy : bool, optional
            Use logarithmic axes.

        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt

        if self.samples is None:
            raise RuntimeError("Run MCMC first.")
        labels = self._dataset_labels()
        self.pp_diagnostics = {}

        for component in self._all_components():
            fig, ax = plt.subplots(figsize=(8, 6))
            t_all, y_all, drawn = [], [], False
            print(f"\nPosterior-predictive coverage ({component}):")
            for i, _ in enumerate(self.datasets):
                self.select(i)
                summary = self._predictive_summary(
                    component, n_sigma, nsamples_pred, condition_discrepancy)
                if summary is None:
                    continue
                t, obs, mean_pred, pred_lo, pred_hi, diag = summary
                color = _COLORS[i % len(_COLORS)]
                ax.fill_between(t, pred_lo, pred_hi, color=color, alpha=0.25)
                ax.plot(t, mean_pred, color=color, lw=1.5, zorder=4, label=labels[i])
                ax.scatter(t, obs, color="black", s=10, zorder=5, marker="o",
                           edgecolors="black", linewidths=0.5, alpha=0.8)
                t_all.append(t)
                y_all.append(np.concatenate([np.ravel(obs), np.ravel(pred_hi)]))
                self.pp_diagnostics[f"{labels[i]}:{component}"] = diag
                print(f"  {labels[i]:<22}{diag['coverage']:>8.1%}"
                      f"{diag['rms_z']:>10.2f}{diag['max_abs_z']:>10.2f}")
                drawn = True

            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            if drawn and not (logx or logy):
                self._set_axes(ax, np.concatenate(t_all), np.concatenate(y_all),
                               key=component)
            ax.set_xlabel("$t$")
            ax.set_ylabel(rf"${component}(t)$")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
                      framealpha=0.9, borderaxespad=0.0)
            self._save_current_figure(f"posterior_predictive_{component}"
                                      if len(self._all_components()) > 1
                                      else "posterior_predictive")
            plt.show()

    def _all_components(self):
        """Union of fitted components across all datasets, x before y."""
        seen = {c for d in self.datasets for c in self._fit_components(d)}
        return [c for c in ("x", "y") if c in seen]


__all__ = ["JointInferenceProcedure"]
