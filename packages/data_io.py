import os
import numpy as np

def hasimoto_Q(L):
    return 1 - 2.8373 / L + (4 * np.pi / 3) / L**3

class DataIO:
    """Load data, optionally add noise, and set priors."""

    def _fit_components(self):
        if self.is_perpendicular():
            return ("y",)
        if (self.use_y and self.material_model == "viscoelastic"
                and self.boundary_model == "bounded"):
            return ("x", "y")
        return ("x",)

    def load_data(self, filename, prior_fraction=0.1):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data = np.atleast_2d(np.loadtxt(os.path.join(root, filename)))
        components = self._fit_components()
        time = data[::self.thin_factor, 0]

        dataset = {
            "t": time,
            "x": data[::self.thin_factor, 1] if "x" in components else None,
            "y": (data[:, 2] - data[0, 2])[::self.thin_factor] if "y" in components else None,
            "fit_components": components,
            "F": self.force,
            "Fx": self.force * np.cos(np.deg2rad(self.theta)),
            "Fy": self.force * np.sin(np.deg2rad(self.theta)),
        }

        if self.material_model == "viscoelastic":
            dataset["t_unload"] = self.t_unload

        if self.hasimoto_corr:
            drag = hasimoto_Q(self.L)
            for comp in components:
                dataset[comp] /= drag
            dataset["hasimoto_Q"] = drag

        rng = np.random.default_rng(self.seed)
        max_disps, noises = [], []
        for comp in components:
            disp = dataset[comp]
            max_disp = np.abs(disp).max()
            sigma_noise = (self.sigma_noise_percent / 100) * max_disp
            noise = rng.normal(0, sigma_noise, len(disp))
            dataset[comp] = disp + noise
            max_disps.append(max_disp)
            noises.append(noise)

        dataset["max_displacement"] = dict(zip(components, max_disps))
        self.beta = 1 / (prior_fraction * max(max_disps))
        self.sigma_noise_prior = self.sigma_bias_prior = self.beta
        self.sigma_realized = np.concatenate(noises).std(ddof=1)
        dataset["sigma_prior_beta"] = self.beta
        self.data = dataset

        print(f"Loaded {os.path.basename(filename)} (angle={self.theta:.1f}, "
              f"n={len(time)}, components={','.join(components)})")
        print(f"realized noise = {self.sigma_realized:.5f}")