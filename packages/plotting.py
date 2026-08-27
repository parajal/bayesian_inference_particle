"""Figures: data, model error, priors, corner, traces, posterior predictive."""

import os
import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 30,
        "axes.labelsize": 30,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "legend.fontsize": 28,
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.0,
    }
)

def _correlated_normal(rng, cov, scale):

    cov = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    lam, U = np.linalg.eigh(cov)
    z = rng.standard_normal(len(cov))
    return scale * (U @ (np.sqrt(np.clip(lam, 0.0, None)) * z))


class _BandWithLineHandler(HandlerBase):
    """Legend handler drawing a shaded band with a centre line."""

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        band = Rectangle(
            (0, 0),
            handlebox.width,
            handlebox.height * 0.6,
            facecolor="steelblue",
            edgecolor="none",
            alpha=0.45,
            transform=handlebox.get_transform(),
        )
        line = Line2D(
            [0, handlebox.width],
            [handlebox.height * 0.3, handlebox.height * 0.3],
            color="blue",
            linewidth=2,
            transform=handlebox.get_transform(),
        )
        handlebox.add_artist(band)
        handlebox.add_artist(line)
        return [band, line]


class Plotting:
    """Diagnostic and results figures, saved as PDFs under ``plots/``."""

    @staticmethod
    def _figure_name(base_name, component, components):
        """Suffix a figure name with the component when more than one is fitted."""
        return base_name if len(components) == 1 else f"{base_name}_{component}"

    def _set_axes(self, ax, x, y, n_ticks=3, y_max=None, key=None, store=False):

        x_max = float(np.max(x))
        ax.set_xlim(0.0, x_max)
        ax.set_xticks(np.linspace(0.0, x_max, n_ticks))

        cache = self.__dict__.setdefault("_yaxis_cache", {})
        if key is not None and not store and key in cache:
            y_lo, y_hi, y_ticks = cache[key]
        else:
            y_hi = (float(np.max(y)) if y_max is None else float(y_max)) * 1.1
            y_lo, y_ticks = 0.0, np.linspace(0.0, y_hi, n_ticks)
            if key is not None:
                cache[key] = (y_lo, y_hi, y_ticks)

        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(y_ticks)

    def _save_current_figure(self, name: str) -> None:
        output_dir = os.path.abspath("plots")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{name}.pdf")
        plt.gcf().savefig(path, bbox_inches="tight")
        print(f"Saved figure: {path}")

    def _resolve_theta_true(self, theta_true):

        if isinstance(theta_true, (bool, np.bool_)):
            theta_true = None
        if theta_true is None:
            theta_true = getattr(self, "theta_true", None)
        return None if theta_true is None else list(theta_true)

    def plot_data(self, theta_true=None) -> None:

        theta_true = self._resolve_theta_true(theta_true)
        if theta_true is None:
            raise ValueError("plot_data needs theta_true (pass it or set it on the model).")
        components = self._fit_components()
        lw = 3 if self.material_model == "newtonian" else 2
        d = self.data

        for component in components:
            obs = d.get(component)
            if obs is None:
                continue
            t_model = np.linspace(0.0, max(d["t"]), 400) if self.model == "viscoelastic_bounded" else d["t"]
            model = self._model_component(theta_true, t_model, d, component=component)

            _, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(d["t"], obs, color="black", s=30, marker="o", edgecolors="black",
                       linewidths=0.5, zorder=3, label="FOM + noise")
            ax.plot(t_model, model, color="red", lw=lw, zorder=2, label="SAM")

            ax.set(xlabel=r"$t$", ylabel=rf"${component}(t)$")
            self._set_axes(ax, d["t"], np.concatenate([np.ravel(obs), np.ravel(model)]),
                           key=component, store=True)
            ax.grid(alpha=0.3)
            ax.legend(loc="best", framealpha=0.95)
            self._save_current_figure(
                self._figure_name("data_vs_noise", component, components)
            )
            plt.show()

    def plot_corner(self, theta_true=None, log_scale=False,
                    label_size=40, physical_only=True) -> None:

        if self.samples is None:
            raise RuntimeError("Run MCMC first.")
        theta_true = self._resolve_theta_true(theta_true)
        names = self._get_parameter_names()
        n_phys = len(names)
        ndim = n_phys if physical_only else self.samples.shape[1]
        samples = self.samples[:, :ndim].astype(float).copy()
        labels = list(self._get_parameter_labels()[:ndim])

        # Show material parameters in log10 if requested; hyperparameters stay physical.
        if log_scale:
            samples[:, :n_phys] = np.log10(samples[:, :n_phys])
            labels[:n_phys] = [rf"$\log_{{10}}\,{lab.strip('$')}$" for lab in labels[:n_phys]]

        truths = [None] * ndim
        if theta_true is not None:
            vals = np.atleast_1d(theta_true).astype(float)
            for i in range(min(n_phys, len(vals))):
                if np.isfinite(vals[i]) and vals[i] > 0:
                    truths[i] = float(np.log10(vals[i]) if log_scale else vals[i])

        # Ground truth for the noise scale is the realized sigma; sigma_bias has none.
        if not physical_only:
            hyper_names = self._get_parameter_labels(latex=False)
            sigma_true = getattr(self, "sigma_noise_true", None)
            if sigma_true is not None and "sigma_noise" in hyper_names:
                truths[hyper_names.index("sigma_noise")] = float(sigma_true)

        fig = plt.figure(figsize=(8 * ndim, 8 * ndim))
        corner.corner(
            samples,
            fig=fig,
            labels=labels,
            label_kwargs=dict(fontsize=label_size) if label_size else None,
            levels=(0.68, 0.95),
            color="black",
            hist_kwargs=dict(histtype="step", linewidth=2.0, density=True, color="black"),
            data_kwargs=dict(ms=1.5, alpha=0.2, color="gray"),
        )

        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            ax = axes[i, i]
            ylim = ax.get_ylim()   # keep the posterior histogram's framing

            # Widen the column so a ground truth outside the posterior stays visible.
            xlo, xhi = ax.get_xlim()
            if truths[i] is not None:
                pad = 0.05 * (xhi - xlo)
                xlo = min(xlo, truths[i] - pad)
                xhi = max(xhi, truths[i] + pad)
                for j in range(i, ndim):
                    axes[j, i].set_xlim(xlo, xhi)

            # Log-uniform prior overlay, drawn only for the material parameters.
            if i < n_phys:
                lo, hi = self.bounds[names[i]]
                if log_scale:
                    a, b = np.log10(lo), np.log10(hi)
                    ax.plot([a, b], [1.0 / (b - a)] * 2, color="red", lw=1.5)
                else:
                    xs = np.linspace(xlo, xhi, 400)
                    dens = np.where((xs >= lo) & (xs <= hi), 1.0 / (xs * np.log(hi / lo)), 0.0)
                    ax.plot(xs, dens, color="red", lw=1.5)

            ci = np.percentile(samples[:, i], [2.5, 97.5])
            ax.axvline(ci[0], color="black", ls=":", lw=1.5)
            ax.axvline(ci[1], color="black", ls=":", lw=1.5)
            if truths[i] is not None:
                ax.axvline(truths[i], color="blue", ls=":", lw=2.0, label="ground truth")
                ax.legend(loc="best")
            ax.set_ylim(ylim)

        base = "corner" if log_scale else "corner_physical"
        self._save_current_figure(base if physical_only else base + "_all")
        plt.show()

    def plot_trace(self) -> None:
 
        chain = self._to_physical(self.sampler.get_chain())
        nsteps, nwalkers, ndim = chain.shape
        burn = int(self.burn_fraction * nsteps)
        labels = self._get_parameter_labels()[:ndim]

        _, axes = plt.subplots(ndim, 1, figsize=(16, 6 * ndim), squeeze=False)
        axes = axes.ravel()
        cmap = plt.get_cmap("tab20" if nwalkers <= 20 else "hsv")
        colors = [cmap(i / nwalkers) for i in range(nwalkers)]
        for idx, (ax, label) in enumerate(zip(axes, labels)):
            for w in range(nwalkers):
                ax.plot(chain[:, w, idx], color=colors[w], lw=1.0, alpha=0.8)
            ax.axvline(burn, color="red", linestyle="--", lw=2.0, alpha=0.7)
            mean_val = np.mean(chain[burn:, :, idx])
            ax.axhline(mean_val, color="black", linestyle="--", lw=2.0, alpha=0.7,
                       label=f"{mean_val:.2e}")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
        axes[-1].set_xlabel("MCMC step")
        self._save_current_figure("trace")
        plt.show()

    def plot_prior(self, n: int = 5000, coordinate: str = "log10") -> None:

        coordinate = coordinate.lower()
        bounds = self._get_parameter_bounds()
        n_phys = len(bounds)
        labels = self._get_parameter_labels()[:n_phys]

        sigmas = [(self.sigma_noise_prior, r"$\sigma_{\mathrm{noise}}$")]
        if self._bias_is_inferred():
            sigmas.append((self.sigma_bias_prior, r"$\sigma_{\mathrm{bias}}$"))

        _, axes = plt.subplots(
            1, n_phys + len(sigmas),
            figsize=(8 * (n_phys + len(sigmas)), 6),
            squeeze=False,
        )
        axes = axes.ravel()

        for ax, (lo, hi), label in zip(axes, bounds, labels):
            a, b = np.log10(lo), np.log10(hi)

            if coordinate == "log10":
                x = np.linspace(a, b, n)
                p = np.full(n, 1.0 / (b - a))
                label = rf"$\log_{{10}}({label[1:-1]})$"
            else:
                x = np.logspace(a, b, n)
                p = 1.0 / (x * np.log(10.0) * (b - a))
                ax.set_xscale("log")

            ax.plot(x, p, lw=2)
            ax.set_xlabel(label)

        for ax, (rate, label) in zip(axes[n_phys:], sigmas):
            x = np.linspace(0.0, 5.0 / rate, n)
            ax.plot(x, rate * np.exp(-rate * x), lw=2)
            ax.set_xlabel(label)

        axes[0].set_ylabel("Prior density")

        for ax in axes:
            ax.grid(alpha=0.3)

        self._save_current_figure("prior")
        plt.show()

    def plot_posterior_predictive(
        self,
        nsamples_pred: int = 5000,
        logx: bool = False,
        logy: bool = False,
    ) -> None:

        if self.samples is None:
            raise RuntimeError("Run MCMC first.")
        if self.data is None:
            raise RuntimeError("No data loaded.")

        components = self._fit_components()

        for component in components:
            summary = self._predictive_summary(component, nsamples_pred)
            if summary is None:
                continue
            t, obs, mean_pred, pred_lo, pred_hi = summary

            _, ax = plt.subplots(figsize=(8, 6))
            ax.fill_between(t, pred_lo, pred_hi, color="steelblue", alpha=0.25)
            ax.plot(t, mean_pred, color="steelblue", lw=1.5, zorder=4)
            ax.scatter(t, obs, color="black", s=10, zorder=5, marker="o",
                       edgecolors="black", linewidths=0.5, alpha=0.8)

            if logx:
                ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")

            ax.set_xlabel("$t$")
            ax.set_ylabel(rf"${component}(t)$")
            ax.grid(True, alpha=0.3)
            custom = Line2D([0], [0], label="Posterior predictive")
            h, lbl = ax.get_legend_handles_labels()
            h.append(custom)
            lbl.append("Posterior predictive")
            ax.legend(h, lbl, loc="best", framealpha=0.9,
                      handler_map={custom: _BandWithLineHandler()})

            self._save_current_figure(
                self._figure_name("posterior_predictive", component, components)
            )
            plt.show()

    def _predictive_summary(self, component, nsamples_pred=5000):

        d = self.data
        obs = d.get(component)
        if obs is None:
            return None
        obs = np.asarray(obs, dtype=float)
        t = d["t"]

        rng = np.random.default_rng(0)
        n_draws = min(nsamples_pred, len(self.samples))
        sel = rng.choice(len(self.samples), size=n_draws, replace=False)
        theta_draws = self.samples[sel]
        labels = self._get_parameter_labels(latex=False)
        sn_draws = np.abs(self.samples[sel, labels.index("sigma_noise")])

        if self._bias_is_inferred():
            sb_draws = np.abs(self.samples[sel, labels.index("sigma_bias")])
        else:
            sb_draws = np.zeros(n_draws)
        l_bias = float(self.l_bias or 1.0)

        def _prior_discrepancy_draw(sb):
            corr = self._bias_correlation_matrix(t, l_bias)
            return _correlated_normal(rng, corr, sb)

        X_pred = np.array(
            [self._model_component(th, t, d, component=component) for th in theta_draws]
        )
        Y_rep = np.empty_like(X_pred, dtype=float)
        for k in range(n_draws):
            g, sn, sb = X_pred[k], float(sn_draws[k]), float(sb_draws[k])
            delta = np.zeros_like(g) if sb <= 0.0 else _prior_discrepancy_draw(sb)
            Y_rep[k] = g + delta + rng.standard_normal(len(g)) * sn

        mean_pred = X_pred.mean(0)
        pred_lo, pred_hi = np.percentile(Y_rep, [2.5, 97.5], axis=0)
        return t, obs, mean_pred, pred_lo, pred_hi

    def plot_results(self, physical_only: bool = True) -> None:

        self.plot_posterior_predictive()
        self.plot_corner(physical_only=physical_only)
        self.plot_trace()
