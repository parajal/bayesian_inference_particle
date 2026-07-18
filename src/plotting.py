import os
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import corner

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 30,
        "axes.labelsize": 30,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "legend.fontsize": 30,
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.0,
    }
)

class _BandWithLineHandler(HandlerBase):

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

    # Every model currently supported by plot_posterior_predictive.
    _ALL_MODELS = (
        "newtonian_unbounded",
        "newtonian_bounded",
        "newtonian_bounded_perp",
        "newtonian_particle_particle",
        "viscoelastic_bounded",
        "viscoelastic_unbounded",
        "viscoelastic_bounded_perp",
    )

    def _make_output_dir(self):
        self.output_dir = os.path.abspath("plots")
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir

    def _save_current_figure(self, name: str):
        if not hasattr(self, "output_dir"):
            self._make_output_dir()
        path = os.path.join(self.output_dir, f"{name}.pdf")
        plt.gcf().savefig(path, bbox_inches="tight")
        print(f"Saved figure: {path}")
    
    def _model_at(self, theta, t, d):
        """Forward model at physical parameters theta on time grid t for one dataset."""
        if self.material_model == "newtonian":
            return self.model_newtonian(theta[0], t, d["F"], self._get_delta0(theta), L=d.get("L"))
        return self.model_viscoelastic(
            theta[0], theta[1], theta[2], t, d["F"],
            self._get_delta0(theta), t_unload=d.get("t_unload"),
        )

    def plot_data(self, theta_true: Tuple[float, ...]) -> None:
        """Plot each dataset (FOM points) against the forward model at theta_true (SAM line)."""
        if not self.datasets:
            raise RuntimeError("No data loaded.")
        if self.model not in self._ALL_MODELS:
            raise ValueError(f"Invalid model in plot_data: {self.model!r}")
        self._theta_true = tuple(theta_true) if theta_true is not None else None

        _, ax = plt.subplots(figsize=(8, 6))
        multi = len(self.datasets) > 1
        lw = 3 if self.material_model == "newtonian" else 2
        handles, force_labels = [], []

        for i, d in enumerate(self.datasets):
            use_y = self._use_y_displacement(d)
            obs = d["y"] if use_y else d["x"]
            if obs is None:
                raise ValueError("Perpendicular data requires y-displacement.")
            # Bounded creep draws the model on a fine fixed grid; all others on the data grid.
            t_model = np.linspace(0.0, 0.5, 400) if self.model == "viscoelastic_bounded" else d["t"]
            model = self._model_at(theta_true, t_model, d)
            color = plt.cm.tab10.colors[i % 10] if multi else "red"

            ax.scatter(d["t"], obs, color="black", s=30, marker="o", edgecolors="black",
                       linewidths=0.5, zorder=3, label="FOM" if i == 0 else None)
            (line,) = ax.plot(t_model, model, color=color, lw=lw, zorder=2,
                              label=None if multi else "SAM")
            if multi:
                handles.append(line)
                force_labels.append(rf"$F = {float(d['F']) / np.pi:g}\pi$")

        is_perp = any(self._use_y_displacement(d) for d in self.datasets)
        ax.set(xlabel=r"$t$", ylabel=r"$y(t)$" if is_perp else r"$x(t)$")
        ax.grid(alpha=0.3)
        if multi:
            ax.legend(handles, force_labels, loc="lower right", framealpha=0.95)
        else:
            ax.legend(loc="best", framealpha=0.95)
        self._save_current_figure("data_vs_noise")
        plt.show()

    def plot_model_error(self, theta_true: Tuple[float, ...]) -> None:
        """Plot the absolute error |FOM - SAM| at theta_true over time, per dataset."""
        if not self.datasets:
            raise RuntimeError("No data loaded.")
        if self.model not in self._ALL_MODELS:
            raise ValueError(f"Invalid model in plot_model_error: {self.model!r}")

        _, ax = plt.subplots(figsize=(8, 6))
        multi = len(self.datasets) > 1
        all_errors = []

        for i, d in enumerate(self.datasets):
            use_y = self._use_y_displacement(d)
            obs = d["y"] if use_y else d["x"]
            if obs is None:
                raise ValueError("Perpendicular data requires y-displacement.")
            # Evaluate on the data grid so it aligns point-by-point with the FOM data.
            model = self._model_at(theta_true, d["t"], d)
            error = np.abs(np.asarray(obs, dtype=float) - model)
            all_errors.append(error)
            color = plt.cm.tab10.colors[i % 10] if multi else "red"
            label = rf"$F = {float(d['F']) / np.pi:g}\pi$" if multi else None
            ax.plot(d["t"], error, color=color, lw=2, marker="o", ms=4, label=label)
            if multi:
                print(f"  dataset[{i}]: max |error| = {error.max():.6g}, mean |error| = {error.mean():.6g}")

        is_perp = any(self._use_y_displacement(d) for d in self.datasets)
        obs_symbol = "y" if is_perp else "x"
        ax.set(xlabel=r"$t$", ylabel=rf"$|{obs_symbol}_{{\mathrm{{FOM}}}} - {obs_symbol}_{{\mathrm{{SAM}}}}|$")
        ax.grid(alpha=0.3)
        if multi:
            ax.legend(loc="best", framealpha=0.95)
        self._save_current_figure("model_error")
        plt.show()

        errors = np.concatenate(all_errors)
        label = "Overall model error" if multi else "Model error"
        print(f"{label}: max |error| = {errors.max():.6g}, mean |error| = {errors.mean():.6g}")

    def plot_corner(self, theta_true=None) -> None:
        """Corner plot of every inferred parameter in physical coordinates."""
        if isinstance(theta_true, (bool, np.bool_)):
            theta_true = None
        if self.samples is None:
            raise RuntimeError("Run MCMC first.")
        ndim = self.samples.shape[1]
        labels = self._get_parameter_labels()[:ndim]

        truths = None
        if theta_true is not None:
            vals = np.atleast_1d(theta_true).astype(float)
            truths = [float(vals[i]) if i < len(vals) and np.isfinite(vals[i]) else None
                      for i in range(ndim)]

        fig = plt.figure(figsize=(8 * ndim, 8 * ndim))
        corner.corner(
            self.samples,
            fig=fig,
            labels=labels,
            levels=(0.68, 0.95),
            color="black",
            truths=truths,
            truth_color="blue",
            hist_kwargs=dict(histtype="step", linewidth=2.0, density=True, color="black"),
            data_kwargs=dict(ms=1.5, alpha=0.2, color="gray"),
        )
        self._save_current_figure("corner")
        plt.show()

    def plot_trace(self) -> None:
        chain = self._to_physical(self.sampler.get_chain())
        nsteps, nwalkers, ndim = chain.shape
        burn = int(self.burn_fraction * nsteps)
        labels = self._get_parameter_labels()[:ndim]

        _, axes = plt.subplots(ndim, 1, figsize=(16, 6 * ndim))
        if ndim == 1:
            axes = [axes]
        cmap = plt.cm.get_cmap("tab20" if nwalkers <= 20 else "hsv")
        colors = [cmap(i / nwalkers) for i in range(nwalkers)]
        for ax, idx, label in zip(axes, range(ndim), labels):
            for w in range(nwalkers):
                ax.plot(chain[:, w, idx], color=colors[w], lw=1.0, alpha=0.8)
            ax.axvline(burn, color="red", linestyle="--", lw=2.0, alpha=0.7)
            mean_val = np.mean(chain[burn:, :, idx])
            ax.axhline(mean_val, color="black", linestyle="--", lw=2.0, alpha=0.7, label=f"{mean_val:.2e}")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
        axes[-1].set_xlabel("MCMC step")
        self._save_current_figure("trace")
        plt.show()

    def plot_prior(self, n: int = 5000, coordinate: str = "log10") -> None:
        coordinate = coordinate.lower()

        labels = {"newtonian": [r"$\eta_s$"],
                    "viscoelastic": [r"$\eta_s$", r"$\eta_p$", r"$\lambda$"],
        }[self.material_model]

        if self.boundary_model == "bounded":
            labels.append(r"$\delta_0$")

        sigmas = [(self.sigma_noise_prior, r"$\sigma_{\mathrm{noise}}$")]
        if self._bias_is_inferred():
            sigmas.append((self.sigma_bias_prior, r"$\sigma_{\mathrm{bias}}$"))

        bounds = list(self._get_parameter_bounds())[:len(labels)]
        n_phys = len(labels)
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
        theta_true=None,
        n_sigma: float = 1.96,
        nsamples_pred: int = 5000,
        logx: bool = False,
        logy: bool = False,
        condition_discrepancy: bool = False,
    ) -> None:
        if self.samples is None:
            raise RuntimeError("Run MCMC first.")
        if not self.datasets:
            raise RuntimeError("No data loaded.")
        if isinstance(theta_true, (int, float, np.number)):
            if n_sigma != 1.96 and nsamples_pred == 5000:
                nsamples_pred = int(n_sigma)
            n_sigma = float(theta_true)
            theta_true = None

        rng = np.random.default_rng(0)
        n_draws = min(nsamples_pred, len(self.samples))
        sel = rng.choice(len(self.samples), size=n_draws, replace=False)
        theta_draws = self.samples[sel]
        labels = self._get_parameter_labels(latex=False)
        sn_draws = np.abs(self.samples[sel, labels.index("sigma_noise")])
        if self._bias_is_inferred() and "sigma_bias" in labels:
            sb_draws = np.abs(self.samples[sel, labels.index("sigma_bias")])
        elif self._bias_is_disabled():
            sb_draws = np.zeros(n_draws)
        else:
            sb_draws = np.full(n_draws, abs(float(self._get_fixed_bias())))
        l_bias_draws = np.full(n_draws, float(self.l_bias or 1.0))

        def _prior_discrepancy_draw(t, sb, l_bias):
            n = len(t)
            if sb <= 0.0:
                return np.zeros(n, dtype=float)
            Kcorr = np.asarray(self._bias_correlation_matrix(t, l_bias), dtype=float)
            Kcorr = 0.5 * (Kcorr + Kcorr.T)
            lam, U = np.linalg.eigh(Kcorr)
            lam = np.clip(lam, 0.0, None)
            z = rng.standard_normal(n)
            return sb * (U @ (np.sqrt(lam) * z))

        def _conditional_discrepancy_draw(t, residual, sn, sb, l_bias):
            n = len(t)
            if sb <= 0.0:
                zeros = np.zeros(n, dtype=float)
                return (zeros, zeros)
            Kcorr = np.asarray(self._bias_correlation_matrix(t, l_bias), dtype=float)
            Kcorr = 0.5 * (Kcorr + Kcorr.T)
            A = sb * sb * Kcorr
            Sigma = A + sn * sn * np.eye(n)
            Sigma = 0.5 * (Sigma + Sigma.T)
            try:
                L = np.linalg.cholesky(Sigma)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, residual))
                solve_A = np.linalg.solve(L.T, np.linalg.solve(L, A))
            except np.linalg.LinAlgError:
                Sigma_inv = np.linalg.pinv(Sigma, hermitian=True)
                alpha = Sigma_inv @ residual
                solve_A = Sigma_inv @ A
            mean = A @ alpha
            cov = A - A @ solve_A
            cov = 0.5 * (cov + cov.T)
            lam, U = np.linalg.eigh(cov)
            lam = np.clip(lam, 0.0, None)
            z = rng.standard_normal(n)
            draw = mean + U @ (np.sqrt(lam) * z)
            return (draw, mean)

        def _predictive_from_draws(X_pred, obs, d):
            from scipy.stats import norm

            t = d["t"]
            include_discrepancy = not self._bias_is_disabled()
            latent_mean = np.empty_like(X_pred, dtype=float)
            Y_rep = np.empty_like(X_pred, dtype=float)
            for k in range(n_draws):
                g = X_pred[k]
                sn = float(sn_draws[k])
                sb = float(sb_draws[k])
                if include_discrepancy:
                    if condition_discrepancy:
                        residual = np.asarray(obs, dtype=float) - g
                        delta, delta_mean = _conditional_discrepancy_draw(t, residual, sn, sb, float(l_bias_draws[k]))
                    else:
                        delta = _prior_discrepancy_draw(t, sb, float(l_bias_draws[k]))
                        delta_mean = np.zeros_like(g, dtype=float)
                else:
                    delta = np.zeros_like(g, dtype=float)
                    delta_mean = delta
                latent_mean[k] = g + delta_mean
                obs_sd = max(sn, 0.0)
                if obs_sd > 0.0:
                    Y_rep[k] = g + delta + rng.standard_normal(len(g)) * obs_sd
                else:
                    Y_rep[k] = g + delta
            tail = float(norm.cdf(n_sigma))
            q_lo, q_hi = (100.0 * (1.0 - tail), 100.0 * tail)
            mean_pred = latent_mean.mean(0)
            pred_lo = np.percentile(Y_rep, q_lo, axis=0)
            pred_hi = np.percentile(Y_rep, q_hi, axis=0)
            sigma_total = np.maximum(np.std(Y_rep, axis=0), 1e-08)
            return (mean_pred, pred_lo, pred_hi, sigma_total)

        if self.model not in self._ALL_MODELS:
            raise ValueError(f"Invalid model in plot_posterior_predictive: {self.model!r}")

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10.colors
        multi = len(self.datasets) > 1
        _pp_diag = []
        for _di, d in enumerate(self.datasets):
            t = d["t"]
            use_y = self._use_y_displacement(d)
            X_pred = np.array([self._model_at(th, d["t"], d) for th in theta_draws])
            x_obs = d["y"] if use_y else d["x"]
            mean_pred, pred_lo, pred_hi, sigma_total = _predictive_from_draws(X_pred, x_obs, d)
            _pp_diag.append((
                f"dataset[{_di}]",
                (x_obs - mean_pred) / sigma_total,
                float(np.mean((x_obs >= pred_lo) & (x_obs <= pred_hi))),
            ))
            col = colors[_di % len(colors)] if multi else "steelblue"
            ax.fill_between(t, pred_lo, pred_hi, color=col, alpha=0.25)
            ax.plot(t, mean_pred, color=col, lw=1.5, zorder=4)
            ax.scatter(t, x_obs, color="black", s=10, zorder=5, marker="o",
                       edgecolors="black", linewidths=0.5, alpha=0.8)

        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        is_perp_plot = any(self._use_y_displacement(d) for d in self.datasets)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$y(t)$" if is_perp_plot else "$x(t)$")
        ax.grid(True, alpha=0.3)
        custom = Line2D([0], [0], label="Posterior predictive")
        h, lbl = ax.get_legend_handles_labels()
        h.append(custom)
        lbl.append("Posterior predictive")
        ax.legend(h, lbl, loc="best", framealpha=0.9, handler_map={custom: _BandWithLineHandler()})
        from scipy.stats import norm as _norm

        nominal_cov = float(2.0 * _norm.cdf(n_sigma) - 1.0)
        self.pp_diagnostics = {}
        if _pp_diag:
            if condition_discrepancy:
                print("\nConditional discrepancy reconstruction coverage (observed grid):")
            else:
                print("\nPosterior-predictive adequacy (replicated data):")
            print(f"  Nominal central coverage of band: {nominal_cov:6.1%}")
            print("  " + "-" * 64)
            print(f"  {'series':<16}{'cover':>8}{'RMS z':>10}{'max|z|':>10}{'mean z':>10}")
            for label, zres, cover in _pp_diag:
                zres = np.asarray(zres, dtype=float)
                rms_z = float(np.sqrt(np.mean(zres**2)))
                max_z = float(np.max(np.abs(zres)))
                mean_z = float(np.mean(zres))
                print(f"  {label:<16}{cover:>8.1%}{rms_z:>10.2f}{max_z:>10.2f}{mean_z:>10.2f}")
                self.pp_diagnostics[label] = dict(coverage=cover, rms_z=rms_z, max_abs_z=max_z, mean_z=mean_z, nominal_coverage=nominal_cov)
    
        self._save_current_figure("posterior_predictive")
        plt.show()

    def plot_results(self) -> None:
        self.plot_posterior_predictive()
        self.plot_corner()
        self.plot_trace()