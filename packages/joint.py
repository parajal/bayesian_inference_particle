from contextlib import contextmanager
import numpy as np

from .main import InferenceProcedure

_PER_DATASET = ("force", "theta", "boundary_model", "t_unload", "hasimoto_corr", "L")

class JointInferenceProcedure(InferenceProcedure):

    def load_datasets(self, items):
        defaults = {k: getattr(self, k) for k in _PER_DATASET}
        self.datasets = []

        for item in items:
            cfg = {**defaults, **{k: v for k, v in item.items() if k != "filename"}}
            thin = int(cfg.pop("thin_factor", self.thin_factor))

            with self._configured(cfg):
                old = self.thin_factor
                self.thin_factor = thin
                try:
                    self.load_data(item["filename"])
                finally:
                    self.thin_factor = old

            self.data["config"] = {**cfg, "thin_factor": thin}
            self.datasets.append(self.data)

        peak = max(d["max_displacement"][c] for d in self.datasets for c in d["fit_components"])
        self.beta = 1 / (0.1 * peak)
        self.sigma_noise_prior = self.sigma_bias_prior = self.beta
        self.data = self.datasets[0]

        print(f"\nJoint fit over {len(self.datasets)} datasets")
        return self.datasets

    @contextmanager
    def _configured(self, config):
        saved = {k: getattr(self, k) for k in _PER_DATASET}
        try:
            for k, v in config.items():
                setattr(self, k, v)
            yield
        finally:
            for k, v in saved.items():
                setattr(self, k, v)

    def select(self, i):
        self.data = self.datasets[i]
        for k in _PER_DATASET:
            setattr(self, k, self.data["config"][k])
        self._t_unload_eff = self.data["config"]["t_unload"]
        return self

    def log_likelihood(self, phi):
        current = self.data
        total = 0.0
        try:
            for d in self.datasets:
                self.data = d
                with self._configured(d["config"]):
                    value = super().log_likelihood(phi)
                if not np.isfinite(value):
                    return -np.inf
                total += value
        finally:
            self.data = current

        return float(total)

    def _dataset_labels(self):
        configs = [d["config"] for d in self.datasets]
        keys = ("force", "theta", "t_unload", "L", "boundary_model")
        varying = [k for k in keys if len({c[k] for c in configs}) > 1]

        labels = []
        for i, c in enumerate(configs):
            parts = []
            if "force" in varying:
                parts.append(rf"$F={c['force']/np.pi:g}\pi$")
            if "theta" in varying:
                parts.append(rf"$\theta={c['theta']:g}^\circ$")
            if "t_unload" in varying:
                parts.append(rf"$t_0={c['t_unload']:g}$")
            if "L" in varying:
                parts.append(rf"$L={c['L']:g}$")
            if "boundary_model" in varying:
                parts.append(c["boundary_model"])
            labels.append(", ".join(parts) or f"dataset {i}")
        return labels

    def _all_components(self):
        return [c for c in ("x", "y") if any(c in d["fit_components"] for d in self.datasets)]


__all__ = ["JointInferenceProcedure"]