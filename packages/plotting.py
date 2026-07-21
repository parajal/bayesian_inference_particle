"""Figures: data, model error, priors, corner, traces, posterior predictive."""

import os
import shutil

import corner
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import norm


def latex_available() -> bool:
    """Whether matplotlib can render text through a working LaTeX install.

    A ``latex`` executable alone is not enough, since matplotlib also needs
    packages such as ``type1ec.sty``, so a short string is actually rendered.

    Returns
    -------
    bool
        ``True`` if LaTeX rendering succeeds.
    """
    if shutil.which("latex") is None:
        return False
    try:
        from matplotlib.texmanager import TexManager

        TexManager().make_dvi(r"$x$", 10)
        return True
    except Exception:
        return False


def use_latex(enabled: bool = True) -> None:
    """Turn LaTeX text rendering on or off for subsequent figures.

    Parameters
    ----------
    enabled : bool, optional
        Set ``text.usetex``. Enabling it without a working LaTeX install makes
        plotting fail, so check :func:`latex_available` first.

    Returns
    -------
    None
    """
    plt.rcParams["text.usetex"] = bool(enabled)


# LaTeX gives the nicest labels but is not installed everywhere, so fall back to
# matplotlib's own mathtext rather than failing on import.
plt.rcParams.update(
    {
        "text.usetex": latex_available(),
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

def _correlated_normal(rng, cov, scale):
    """Draw a zero-mean Gaussian vector with covariance ``scale**2 * cov``.

    The covariance is symmetrised and its eigenvalues clipped at zero, so
    numerically indefinite inputs are handled gracefully.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator.
    cov : numpy.ndarray
        Covariance (or correlation) matrix.
    scale : float
        Multiplies the drawn vector.

    Returns
    -------
    numpy.ndarray
        A single draw of length ``len(cov)``.
    """
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

    def _save_current_figure(self, name: str) -> None:
        """Save the current figure to ``plots/<name>.pdf``."""
        output_dir = os.path.abspath("plots")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{name}.pdf")
        plt.gcf().savefig(path, bbox_inches="tight")
        print(f"Saved figure: {path}")

    def plot_data(self, theta_true) -> None:
        """Plot the observed data (FOM) against the forward model at ``theta_true`` (SAM).

        Parameters
        ----------
        theta_true : sequence of float
            Physical parameters at which the model curve is drawn.

        Returns
        -------
        None
        """
        components = self._fit_components(self.data)
        lw = 3 if self.material_model == "newtonian" else 2
        d = self.data

        for component in components:
            obs = d.get(component)
            if obs is None:
                continue
            # Bounded creep draws the model on a fine fixed grid; all others on the data grid.
            t_model = np.linspace(0.0, max(d["t"]), 400) if self.model == "viscoelastic_bounded" else d["t"]
            model = self._model_component(theta_true, t_model, d, component=component)

            _, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(d["t"], obs, color="black", s=30, marker="o", edgecolors="black",
                       linewidths=0.5, zorder=3, label="FOM")
            ax.plot(t_model, model, color="red", lw=lw, zorder=2, label="SAM")

            ax.set(xlabel=r"$t$", ylabel=rf"${component}(t)$")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", framealpha=0.95)
            self._save_current_figure(
                self._figure_name("data_vs_noise", component, components)
            )
            plt.show()

    def plot_model_error(self, theta_true) -> None:
        """Plot the absolute model error ``|FOM - SAM|`` over time, per component.

        This is the pure discrepancy between the two models, so it uses the
        noise-free FOM series rather than the noisy observations. The printed
        relative L2 error is ``||FOM - SAM||_2 / ||FOM||_2``.

        Parameters
        ----------
        theta_true : sequence of float
            Physical parameters at which the model is evaluated.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If no dataset has been loaded.
        """
        if self.data is None:
            raise RuntimeError("No data loaded.")

        components = self._fit_components(self.data)
        d = self.data

        for component in components:
            fom = d.get(f"{component}_clean", d.get(component))
            if fom is None:
                continue
            fom = np.asarray(fom, dtype=float)
            # Evaluate on the data grid so it aligns point-by-point with the FOM data.
            model = self._model_component(theta_true, d["t"], d, component=component)
            error = np.abs(fom - model)

            _, ax = plt.subplots(figsize=(8, 6))
            ax.plot(d["t"], error, color="red", lw=2, marker="o", ms=4)
            ax.set(
                xlabel=r"$t$",
                ylabel=rf"$|{component}_{{\mathrm{{FOM}}}} - {component}_{{\mathrm{{SAM}}}}|$",
            )
            ax.grid(alpha=0.3)
            self._save_current_figure(
                self._figure_name("model_error", component, components)
            )
            plt.show()

            fom_norm = np.linalg.norm(fom)
            relative_l2 = np.linalg.norm(error) / fom_norm if fom_norm > 0 else np.nan
            print(f"{component} model error: max |error| = {error.max():.6g}, "
                  f"mean |error| = {error.mean():.6g}, "
                  f"relative L2 error = {relative_l2:.6g}")

    def plot_corner(self, theta_true=None) -> None:
        """Corner plot of every inferred parameter in physical coordinates.

        Parameters
        ----------
        theta_true : sequence of float, optional
            Reference values marked on the panels.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the sampler has not been run.
        """
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
        """Trace plot per parameter, with the burn-in cut and posterior mean marked.

        Returns
        -------
        None
        """
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
        """Plot the marginal prior density of every sampled parameter.

        Parameters
        ----------
        n : int, optional
            Number of points used to draw each curve.
        coordinate : {"log10", "physical"}, optional
            Whether the log-uniform material priors are drawn against
            ``log10(parameter)`` or the parameter itself on a log axis.

        Returns
        -------
        None
        """
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
        n_sigma: float = 1.96,
        nsamples_pred: int = 5000,
        logx: bool = False,
        logy: bool = False,
        condition_discrepancy: bool = False,
    ) -> None:
        """Plot the posterior predictive band against the observed data.

        For each posterior draw the forward model is evaluated, a model
        discrepancy is added (drawn from its prior, or conditioned on the
        residual when ``condition_discrepancy`` is set), and measurement noise is
        added to form a replicated dataset. Coverage diagnostics are printed and
        stored on ``self.pp_diagnostics``.

        Parameters
        ----------
        n_sigma : float, optional
            Half-width of the band in standard normal deviates; 1.96 gives a
            central 95% band.
        nsamples_pred : int, optional
            Maximum number of posterior draws used.
        logx, logy : bool, optional
            Use logarithmic axes.
        condition_discrepancy : bool, optional
            Condition the discrepancy on the observed residual instead of
            drawing it from its prior.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the sampler has not been run or no dataset is loaded.
        """
        if self.samples is None:
            raise RuntimeError("Run MCMC first.")
        if self.data is None:
            raise RuntimeError("No data loaded.")

        rng = np.random.default_rng(0)
        n_draws = min(nsamples_pred, len(self.samples))
        sel = rng.choice(len(self.samples), size=n_draws, replace=False)
        theta_draws = self.samples[sel]
        labels = self._get_parameter_labels(latex=False)
        sn_draws = np.abs(self.samples[sel, labels.index("sigma_noise")])

        use_bias = not self._bias_is_disabled()
        if self._bias_is_inferred():
            sb_draws = np.abs(self.samples[sel, labels.index("sigma_bias")])
        elif use_bias:
            sb_draws = np.full(n_draws, abs(float(self._get_fixed_bias())))
        else:
            sb_draws = np.zeros(n_draws)
        l_bias = float(self.l_bias or 1.0)

        def _prior_discrepancy_draw(t, sb):
            """Draw a discrepancy from its prior, ``N(0, sb**2 K)``."""
            corr = self._bias_correlation_matrix(t, l_bias)
            return _correlated_normal(rng, corr, sb)

        def _conditional_discrepancy_draw(t, residual, sn, sb):
            """Draw the discrepancy conditioned on ``residual``; also return its mean."""
            n = len(t)
            A = sb * sb * self._bias_correlation_matrix(t, l_bias)
            Sigma = 0.5 * (A + A.T) + sn * sn * np.eye(n)
            try:
                L = np.linalg.cholesky(Sigma)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, residual))
                solve_A = np.linalg.solve(L.T, np.linalg.solve(L, A))
            except np.linalg.LinAlgError:
                Sigma_inv = np.linalg.pinv(Sigma, hermitian=True)
                alpha = Sigma_inv @ residual
                solve_A = Sigma_inv @ A
            mean = A @ alpha
            return mean + _correlated_normal(rng, A - A @ solve_A, 1.0), mean

        def _predictive_from_draws(X_pred, obs, t):
            """Replicate datasets from the posterior draws and summarise them."""
            latent_mean = np.empty_like(X_pred, dtype=float)
            Y_rep = np.empty_like(X_pred, dtype=float)
            for k in range(n_draws):
                g, sn, sb = X_pred[k], float(sn_draws[k]), float(sb_draws[k])
                if not use_bias or sb <= 0.0:
                    delta = delta_mean = np.zeros_like(g)
                elif condition_discrepancy:
                    delta, delta_mean = _conditional_discrepancy_draw(
                        t, np.asarray(obs, dtype=float) - g, sn, sb
                    )
                else:
                    delta = _prior_discrepancy_draw(t, sb)
                    delta_mean = np.zeros_like(g)
                latent_mean[k] = g + delta_mean
                Y_rep[k] = g + delta + rng.standard_normal(len(g)) * sn

            tail = 100.0 * float(norm.cdf(n_sigma))
            return (
                latent_mean.mean(0),
                np.percentile(Y_rep, 100.0 - tail, axis=0),
                np.percentile(Y_rep, tail, axis=0),
                np.maximum(np.std(Y_rep, axis=0), 1e-8),
            )

        nominal_cov = float(2.0 * norm.cdf(n_sigma) - 1.0)
        self.pp_diagnostics = {}
        components = self._fit_components(self.data)
        d = self.data
        t = d["t"]

        for component in components:
            obs = d.get(component)
            if obs is None:
                continue
            X_pred = np.array(
                [self._model_component(th, t, d, component=component) for th in theta_draws]
            )
            mean_pred, pred_lo, pred_hi, sigma_total = _predictive_from_draws(X_pred, obs, t)
            zres = (obs - mean_pred) / sigma_total
            cover = float(np.mean((obs >= pred_lo) & (obs <= pred_hi)))

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

            diagnostics = dict(
                coverage=cover,
                rms_z=float(np.sqrt(np.mean(zres**2))),
                max_abs_z=float(np.max(np.abs(zres))),
                mean_z=float(np.mean(zres)),
                nominal_coverage=nominal_cov,
            )
            self.pp_diagnostics[component] = diagnostics
            print(f"  {component:<18}{cover:>8.1%}{diagnostics['rms_z']:>10.2f}"
                  f"{diagnostics['max_abs_z']:>10.2f}{diagnostics['mean_z']:>10.2f}")

            self._save_current_figure(
                self._figure_name("posterior_predictive", component, components)
            )
            plt.show()

    def plot_results(self) -> None:
        """Draw the posterior predictive, corner, and trace figures.

        Returns
        -------
        None
        """
        self.plot_posterior_predictive()
        self.plot_corner()
        self.plot_trace()
