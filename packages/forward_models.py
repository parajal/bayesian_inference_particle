"""Forward models mapping material parameters to bead displacement.
"""
 
import numpy as np
from scipy.integrate import solve_ivp
 
 
class ForwardModels:
    """Newtonian and viscoelastic forward models for bead displacement."""
 
    # ----------------------------------------------------------------- helpers
    def _drag(self, eta):
        """Stokes drag coefficient ``6*pi*a*eta``.
 
        Parameters
        ----------
        eta : float
            Viscosity.
 
        Returns
        -------
        float
            The drag coefficient for radius ``self.a``.
        """
        return 6.0 * np.pi * self.a * eta
 
    def _force_components(self, F):
        """Resolve a scalar force into Cartesian components along ``self.theta``.
 
        Parameters
        ----------
        F : float
            Force magnitude.
 
        Returns
        -------
        Fx, Fy : float
            Components of ``F`` at angle ``self.theta`` (degrees).
        """
        theta = np.deg2rad(float(self.theta))
        return F * np.cos(theta), F * np.sin(theta)
 
    def _default_component(self, component):
        """Choose which displacement component to return.
 
        Parameters
        ----------
        component : str or None
            Explicit component (``"x"`` or ``"y"``), or ``None`` to pick from
            the pull geometry.
 
        Returns
        -------
        str
            ``"y"`` for perpendicular pulls, otherwise ``"x"`` (or the
            lower-cased explicit value).
        """
        if component is not None:
            return str(component).lower()
        return "y" if self.is_perpendicular() else "x"
 
    @staticmethod
    def _solve_creep_recovery(ode_eqn, y0, t, t_unload, load_args, recovery_args):
        """Integrate an ODE through a loading phase then a recovery phase.
 
        Parameters
        ----------
        ode_eqn : callable
            Right-hand side ``f(t, state, *args)`` passed to ``solve_ivp``.
        y0 : array_like
            Initial state at ``t[0]``.
        t : array_like
            Output time grid; the split point is ``t_unload``.
        t_unload : float
            Time at which the load is removed.
        load_args, recovery_args : tuple
            Extra arguments to ``ode_eqn`` during each phase.
 
        Returns
        -------
        numpy.ndarray or None
            State array of shape ``(n_states, len(t))``, or ``None`` if either
            integration fails.
        """
        t = np.asarray(t, dtype=float)
        y0 = np.asarray(y0, dtype=float)
 
        load = solve_ivp(
            ode_eqn, (t[0], t_unload), y0,
            t_eval=np.append(t[t < t_unload], t_unload),
            rtol=1e-7, atol=1e-9, args=load_args,
        )
        if not load.success:
            return None
 
        recovery = solve_ivp(
            ode_eqn, (t_unload, t[-1]), y0=load.y[:, -1],
            t_eval=t[t >= t_unload],
            rtol=1e-7, atol=1e-9, args=recovery_args,
        )
        if not recovery.success:
            return None
 
        result = np.empty((y0.size, t.size))
        result[:, t < t_unload] = load.y[:, :-1]
        result[:, t >= t_unload] = recovery.y
        return result
 
    def model_newtonian(self, eta, t, F, delta0=None):
        """Newtonian bead displacement under a constant force.
 
        Parameters
        ----------
        eta : float
            Fluid viscosity.
        t : array_like
            Time grid.
        F : float
            Applied force magnitude.
        delta0 : float, optional
            Initial wall gap for the bounded model. Defaults to ``self.delta0``.
 
        Returns
        -------
        numpy.ndarray
            Displacement of the selected component on ``t``.
        """
        t = np.asarray(t, dtype=float)
        component = self._default_component(None)
        drag = self._drag(eta)
        Fx, Fy = self._force_components(F)
 
        # Unbounded fluid: linear drift, no wall correction.
        if self.boundary_model == "unbounded":
            F_dir = Fy if component == "y" else Fx
            return F_dir * (t - t[0]) / drag
 
        # Bounded fluid: forward-Euler because the wall factors depend on y.
        delta0 = self.delta0 if delta0 is None else delta0
        x = np.zeros_like(t)
        y = np.zeros_like(t)
        for i in range(t.size - 1):
            f_parallel, f_perp = self._wall_factors(delta0 + y[i])
            dt = t[i + 1] - t[i]
            x[i + 1] = x[i] + dt * Fx / (drag * f_parallel)
            y[i + 1] = y[i] + dt * Fy / (drag * f_perp)
 
        return y if component == "y" else x
 
    def model_viscoelastic(self, eta_s, eta_p, lambda_, t, F,
                           delta0=None, t_unload=None, component=None):
        """Viscoelastic bead displacement (Jeffreys-type creep and recovery).
 
        Parameters
        ----------
        eta_s, eta_p : float
            Solvent and polymer viscosities.
        lambda_ : float
            Polymer relaxation time.
        t : array_like
            Time grid.
        F : float
            Applied force magnitude.
        delta0 : float, optional
            Initial wall gap for the bounded model. Defaults to ``self.delta0``.
        t_unload : float, optional
            Load-removal time. Defaults to ``self._t_unload_eff``.
        component : str, optional
            ``"x"`` or ``"y"``; defaults to the pull geometry.
 
        Returns
        -------
        numpy.ndarray or None
            Displacement of the selected component, or ``None`` if a bounded
            integration fails.
 
        Raises
        ------
        ValueError
            If ``t_unload`` is not finite, or the bounded model is missing a
            positive ``delta0``.
        """
        t = np.asarray(t, dtype=float)
        component = self._default_component(component)
        Fx, Fy = self._force_components(F)
 
        t_unload = self._t_unload_eff if t_unload is None else t_unload
        if t_unload is None or not np.isfinite(float(t_unload)):
            raise ValueError("t_unload must be finite for viscoelastic models.")
        t_unload = float(t_unload)
 
        if self.boundary_model == "unbounded":
            return self._viscoelastic_unbounded(
                eta_s, eta_p, lambda_, t, Fx, Fy, t_unload, component
            )
 
        delta0 = self.delta0 if delta0 is None else delta0
        if delta0 is None or not np.isfinite(delta0) or delta0 <= 0.0:
            raise ValueError("bounded viscoelastic model requires a positive delta0.")
 
        if self.is_perpendicular():
            return self._viscoelastic_bounded_perp(
                eta_s, eta_p, lambda_, t, Fy, delta0, t_unload, component
            )
        return self._viscoelastic_bounded_full(
            eta_s, eta_p, lambda_, t, Fx, Fy, delta0, t_unload, component
        )
 
    def _viscoelastic_unbounded(self, eta_s, eta_p, lambda_, t, Fx, Fy,
                                t_unload, component):
        """Closed-form viscoelastic creep/recovery with no wall correction.
 
        Parameters
        ----------
        eta_s, eta_p : float
            Solvent and polymer viscosities.
        lambda_ : float
            Polymer relaxation time.
        t : numpy.ndarray
            Time grid.
        Fx, Fy : float
            Force components.
        t_unload : float
            Load-removal time.
        component : str
            ``"x"`` or ``"y"``.
 
        Returns
        -------
        numpy.ndarray
            Displacement of the selected component on ``t``.
        """
        eta_0 = eta_s + eta_p
        drag_0 = self._drag(eta_0)
        eff_tau = lambda_ * eta_s / eta_0
 
        F_dir = Fy if component == "y" else Fx
        v_steady = F_dir / drag_0
        prefactor = F_dir * lambda_ * eta_p / (drag_0 * eta_0)
 
        load_time = t - t[0]
        end_load = max(0.0, t_unload - t[0])
        recovery_time = np.maximum(t - t_unload, 0.0)
 
        x_load = v_steady * load_time + prefactor * (1.0 - np.exp(-load_time / eff_tau))
        x_at_unload = (v_steady * end_load
                       + prefactor * (1.0 - np.exp(-end_load / eff_tau)))
        recoil = prefactor * (1.0 - np.exp(-end_load / eff_tau))
        x_recovery = x_at_unload - recoil * (1.0 - np.exp(-recovery_time / eff_tau))
 
        return np.where(t <= t_unload, x_load, x_recovery)
 
    def _viscoelastic_bounded_perp(self, eta_s, eta_p, lambda_, t, Fy,
                                   delta0, t_unload, component):
        """Bounded viscoelastic model for a perpendicular pull.
 
        Parameters
        ----------
        eta_s, eta_p : float
            Solvent and polymer viscosities.
        lambda_ : float
            Polymer relaxation time.
        t : numpy.ndarray
            Time grid.
        Fy : float
            Perpendicular force component.
        delta0 : float
            Initial wall gap.
        t_unload : float
            Load-removal time.
        component : str
            ``"x"`` returns zeros; ``"y"`` returns the integrated displacement.
 
        Returns
        -------
        numpy.ndarray or None
            Displacement on ``t``, or ``None`` if integration fails.
        """
        if component == "x":
            return np.zeros_like(t)
 
        solvent_drag = self._drag(eta_s)
        polymer_drag = self._drag(eta_p)
 
        def ode_eqn(time, state, force_y):
            y, Fpy = state
            f_perp = float(self.brenner_perpendicular(delta0 + y))
            dydt = (force_y / f_perp - Fpy) / solvent_drag
            dFpydt = (-Fpy + polymer_drag * dydt) / lambda_
            return [dydt, dFpydt]
 
        sol = self._solve_creep_recovery(
            ode_eqn, [0.0, 0.0], t, t_unload, (Fy,), (0.0,)
        )
        return None if sol is None else sol[0]
 
    def _viscoelastic_bounded_full(self, eta_s, eta_p, lambda_, t, Fx, Fy,
                                   delta0, t_unload, component):
        """Bounded viscoelastic model for full 2-D motion.
 
        Parameters
        ----------
        eta_s, eta_p : float
            Solvent and polymer viscosities.
        lambda_ : float
            Polymer relaxation time.
        t : numpy.ndarray
            Time grid.
        Fx, Fy : float
            Force components.
        delta0 : float
            Initial wall gap.
        t_unload : float
            Load-removal time.
        component : str
            Which component to return, ``"x"`` or ``"y"``.
 
        Returns
        -------
        numpy.ndarray or None
            Displacement of the selected component, or ``None`` if integration
            fails.
        """
        solvent_drag = self._drag(eta_s)
        polymer_drag = self._drag(eta_p)
 
        def ode_eqn(time, state, force_x, force_y):
            x, y, Fpx, Fpy = state
            f_parallel, f_perp = self._wall_factors(delta0 + y)
            dxdt   = (force_x - Fpx) / (f_parallel * solvent_drag)
            dydt   = (force_y - Fpy) / (f_perp * solvent_drag)
            dFpxdt = (-Fpx + f_parallel * polymer_drag * dxdt) / lambda_
            dFpydt = (-Fpy + f_perp * polymer_drag * dydt) / lambda_
            return [dxdt, dydt, dFpxdt, dFpydt]
 
        sol = self._solve_creep_recovery(
            ode_eqn, [0.0, 0.0, 0.0, 0.0], t, t_unload, (Fx, Fy), (0.0, 0.0)
        )
        if sol is None:
            return None
        return sol[1] if component == "y" else sol[0]
 
    def _model_component(self, theta, t, d, component=None):
        """Evaluate one displacement component for a dataset.
 
        Parameters
        ----------
        theta : sequence of float
            Physical parameters: ``[eta_s]`` for the Newtonian model, or
            ``[eta_s, eta_p, lambda_]`` for the viscoelastic model.
        t : array_like
            Time grid.
        d : dict
            Dataset providing ``"F"`` and, for viscoelastic models, ``"t_unload"``.
        component : str, optional
            ``"x"`` or ``"y"``; defaults to the pull geometry.
 
        Returns
        -------
        numpy.ndarray or None
            Model displacement for the selected component.
        """
        if self.material_model == "newtonian":
            return self.model_newtonian(theta[0], t, d["F"], self._get_delta0(theta))
        return self.model_viscoelastic(
            theta[0], theta[1], theta[2], t, d["F"],
            delta0=self._get_delta0(theta),
            t_unload=d.get("t_unload"),
            component=component,
        )