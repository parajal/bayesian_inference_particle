# Bayesian inference for particle-trajectory rheology

Infers rheological parameter from the trajectory of a forced spherical particle,
using MCMC over a Gaussian likelihood with an explicit model-discrepancy term.

Given a measured displacement series `x(t)` (or `y(t)`) for a particle set in
motion by a constant force, the code recovers the viscosity of the surrounding 
fluid, and for viscoelastic media the polymer viscosity and relaxation time.
Displacement recorded near a plane wall are corrected for hydrodynamic wall effects, and the
wall gap itself can be inferred alongside the material parameters.

Requires Python 3.10 or newer. Figure labels are rendered with LaTeX when a
working LaTeX installation is detected.

## Quick start

```python
from packages import InferenceProcedure

model = InferenceProcedure(
    force=12 * 3.14159,        # applied force magnitude
    material_model="newtonian",  # "newtonian" or "viscoelastic"
    boundary_model="bounded",    # "bounded" or "unbounded"
    theta=45,                    # angle in degrees; 90 is wall-normal
    delta0=0.1,                  # initial wall gap
    sigma_noise_percent=2.0,     # synthetic noise, as % of peak displacement
    seed=0,                      # makes that noise reproducible
    nsteps=10000,
)

model.load_data("data/newtonian/angle-45/delta-0.1-45.txt")
model.run_mcmc(warmup=True)
model.plot_results()
```

`run_mcmc` returns post-burn-in samples in physical units and prints posterior
means, standard deviations, and Gelman-Rubin R-hat per parameter.

## Models

The model is selected by `material_model` and `boundary_model`; a pull angle of
90 degrees selects the wall-normal variant automatically.

| `material_model` | `boundary_model` | Inferred parameters |
| --- | --- | --- |
| `newtonian` | `unbounded` | `eta_s` |
| `newtonian` | `bounded` | `eta_s`, `delta0` |
| `viscoelastic` | `unbounded` | `eta_s`, `eta_p`, `lambda_` |
| `viscoelastic` | `bounded` | `eta_s`, `eta_p`, `lambda_`, `delta0` |

Here `eta_s` is the solvent viscosity, `eta_p` the polymer viscosity, `lambda_`
the polymer relaxation time, and `delta0` the initial particle-wall gap. Every run
also infers a measurement-noise scale `sigma_noise` by default and 
model-discrepancy scale `sigma_bias` (if inferred by sigma_bias = "infer").

The Newtonian model is a constant-force motion; bounded cases are integrated
forward in time because the wall corrections depend on the evolving gap. The
viscoelastic model is creep followed by recovery once the load is
removed at `t_unload`, solved in closed form when unbounded and by `solve_ivp`
when bounded.

Wall corrections use the Zeng interpolant for motion parallel to the wall and
Brenner's series for motion perpendicular to it. Brenner's series is evaluated
once on a logarithmic grid and interpolated thereafter, which keeps the MCMC
inner loop cheap.

## Bayesian inference procedure

Material parameters get log-uniform priors over the bounds passed to
`InferenceProcedure` (`eta_s_bounds`, `eta_p_bounds`, `lambda_bounds`,
`delta0_bounds`). The noise and discrepancy scales get exponential priors whose
rate is set from the peak displacement of the loaded dataset.

Observation error is the sum of independent measurement noise and a correlated
model discrepancy, giving the covariance

```
Sigma = sigma_noise^2 I + sigma_bias^2 K,   K_ij = exp(-|t_i - t_j| / l_bias)
```

so the discrepancy has correlation length `l_bias`. 
Set `sigma_bias` to a number to fix it, `"infer"` to sample it, or
`None` to drop the discrepancy term and assume the model is exact.

Sampling uses `emcee` with a mixture of stretch, differential-evolution, and
Gaussian moves. Walkers start from prior draws; with `warmup=True` a short
preliminary chain runs first and walkers restart near its highest-posterior
samples.

## Figures

| Method | Output |
| --- | --- |
| `plot_data(theta_true)` | Observed data against the model at `theta_true` |
| `plot_model_error(theta_true)` | Absolute model error, noise-free |
| `plot_prior()` | Marginal prior density per parameter |
| `plot_corner(theta_true)` | Posterior corner plot |
| `plot_trace()` | Per-parameter chains with burn-in marked |
| `plot_posterior_predictive()` | Posterior Predictive band |
| `plot_results()` | Posterior predictive, corner, and trace together |

Figures are written as PDFs to `plots/` relative to the working directory.

LaTeX rendering is enabled automatically when available. To control it
explicitly:

## Repository layout

```
packages/     inference code (see below)
data/         example trajectories, by material and pull angle
examples/     notebooks, one per model configuration
```

Inside `packages/`, `main.py` defines `InferenceProcedure`, which composes:
 `data_io` (loading, thinning, synthetic noise), `forward_models`,
`wall_corrections`, `logtransforms` (sampler/physical coordinate maps),
`priors`, `likelihood`, `sampler`, and `plotting`.

## Reproducibility

`seed` fixes the synthetic measurement noise added at load time, and
`run_mcmc(random_state=...)` fixes walker initialisation, so a full run is
reproducible end to end.

