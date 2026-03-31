import numpy as np

from scipy.optimize import fsolve, brentq
from scipy.integrate import quad
from scipy.stats import norm
from scipy.integrate import solve_ivp

import copy

from typing import List, Callable, Dict, Any

from SModel import SModel  # e.g., for absorption

# Define constants.
H = 6.62606957e-34  # Planck"s constant [m^2.kg/s]
C = 2.99792458e8    # Speed of light in a vacuum [m/s]
KB =  1.3806488e-23 # Boltzmann constant [m^2.kg/s^2/K]
R = 8.3144621       # Universal gas constant [J/mol/K]
NA = 6.0221409e23   # Avogadro's number [-]
PI = np.pi

"""
HTModel: A class containing a heat transfer model for LII. 

This class defines the TiRe-LII heat transfer model, designed for evaluating heat 
transfer properties based on specified material and experimental conditions. 

Usage:
  - `htmodel = HTModel(prop, x, t)` creates a heat transfer model using the
    specified properties (`prop`), variables of interest (`x`), and time vector (`t`).
  - `htmodel.evaluate(...)` evaluates the heat transfer model for the specified
    parameters.

Parameters:
- `prop`: Default material and experimental properties.
- `x`: List of variables of interest (QoI).
- `t`: Time vector for model evaluation.
- `opts`: Dictionary of options controlling the heat transfer model.

Author: Timothy Sipkens, 2015 (original MATLAB code)
"""

class HTModel:
    prop: Dict[str, Any] = {}  # Default material and experimental properties
    t: List[float] = []  # Time vector
    x: List[str] = ['dp0']  # QoI variable names

    # Function handles for heat transfer processes
    dTdt: Callable = None  # Rate of temperature change
    dmdt: Callable = None  # Rate of mass change
    dXdt: Callable = None  # Rate of annealed fraction change

    # Heat transfer model options
    opts: Dict[str, Any] = {}

    def __init__(self, prop, x=['dp0'], t=np.array([0]), **kwargs):
        self.prop = prop
        self.x = x if isinstance(x, list) else [x]
        self.t = t
        self.opts = {
            'cond': 'free-molecular',  # conduction model
            'cond_sphere': 'primary',  # heat transfer from equivalent sphere ('eq-sphere') or primary particle ('primary')
            'vap': 'free-molecular',  # vaporization model
            'vap_ann': False,   # whether the vaporization model is coupled with annealing species
            'rad': 'none',  # radiation model
            'abs': 'none',  # absorption model
            'ann': 'none',  # annealing model
            'polydispersity': 0,  # incorporate polydispersity
            'deMethod': 'RK45'  # ODE solver method
        }
        self.opts.update(kwargs)  # Parse additional options

        print(self)

    def __repr__(self):
        lines = []
        lines.append('\r' +'\033[32m' + 'HTModel > opts:' + '\033[0m')
        lines.append(f" \033[34m Conduction\033[0m → {self.opts['cond']} (from {self.opts['cond_sphere']})")
        lines.append(f" \033[34m Vaporization\033[0m → {self.opts['vap']}")
        lines.append(f" \033[34m Absorption\033[0m → {self.opts['abs']}" + (" (Gaussian)" if self.opts['abs'] == 'include' else ""))
        lines.append(f" \033[34m Radiation\033[0m → {self.opts['rad']}")
        lines.append(f" \033[34m Annealing\033[0m → {self.opts['ann']}")
        lines.append(f" \033[34m ODE Method\033[0m → {self.opts['deMethod']}")
        lines.append('\r')
        return "\n".join(lines)


    def evaluate(self, x: List[float]):
        prop = self.prop
        for ii in range(len(self.x)):
            setattr(prop, self.x[ii], np.asarray(x[ii]))
        return self.de_solve(prop, np.array([prop.dp0]))

    def de_solve(self, prop=None, dp0=None, t=None):
        
        # If no t, then inherit from instance of class.
        if t is None: t = self.t
        
        # If no prop, then inherit from instance of class.
        if prop is None: prop = self.prop
        else: self.prop = prop
        
        # Same for dp0.
        if dp0 is None: dp0 = np.array([prop.dp0])
        Nd = len(dp0)  # number of size classes to consider in solver

        # Initial temperature.
        Ti = prop.Ti * np.ones(Nd)  # initial temperature, [K]

        # Initial mass
        if not hasattr(prop, 'rho0'):
            prop.rho0 = prop.rho(prop.Tg)
        mass_conv = 1e21  # converts mass to attogram (ag)
        mpi = (prop.rho0 * (dp0 * 1e-9) ** 3 * (np.pi / 6)) * mass_conv  # initial mass, [ag]

        # Initial annealed fraction.
        if hasattr(prop, 'Xi'):
            Xi = np.asarray(prop.Xi) * np.ones_like(Ti)
        else:
            Xi = np.array([1]) * np.ones_like(Ti)
        
        # Starting point exception (allows for gap between the laser pulse and measurements).
        if t[0] > 0.1:  # allows for initial condition at t=0 instead of first entry in time vector
            t = np.concatenate(([0], t))
            opts_tadd = 1
        else: opts_tadd = 0

        # Define the system of ODEs
        def dydt(t, y):
            # Extract quantities.
            T = y[:Nd]
            m = np.abs(y[Nd:2*Nd]) / mass_conv
            X = y[2*Nd:3*Nd] if len(y) > 2 * Nd else None

            # If no annealing.
            if self.opts['ann'] == 'none':
                dTdt = self.dTdt(t, T, m)
                dmdt = self.dmdt(t, T, m)
                return np.concatenate([dTdt * 1e-9, dmdt * mass_conv * 1e-9])
            
            if self.opts['ann'] != 'none':
                dTdt = self.dTdt(t, T, m, X)
                dmdt = self.dmdt(t, T, m, X)
                dXdt = self.dXdt(t, T, m, X)
                dXdt[X > 1] = 0  # stop when fully annealed
                return np.concatenate([dTdt * 1e-9, dmdt * mass_conv * 1e-9, dXdt * 1e-9])

        # Initial state
        if self.opts['ann'] != 'none':
            yi = np.concatenate([Ti, mpi, Xi])
        else:
            yi = np.concatenate([Ti, mpi])
        # else:  elif self.opts['vap'] != 'none':
        #     yi = np.concatenate([Ti])

        # Solve the ODE
        if self.opts['deMethod'] in ['BDF', 'RK45']:  # specifics of ODE solver call
            if self.opts['deMethod'] in ['RK45']:
                sol = solve_ivp(dydt, (t[0], t[-1]), yi, t_eval=t, method='RK45', max_step=(t[1]-t[0]))

            else:
                sol = solve_ivp(dydt, (t[0], t[-1]), yi, t_eval=t, method='BDF', max_step=(t[1]-t[0]))
            
            To = sol.y[:Nd, :]
            mpo = sol.y[Nd:2*Nd, :] / mass_conv

            if self.opts['ann'] == 'none':
                Xo = Xi[0] * np.ones_like(To)
            else:
                Xo = np.clip(sol.y[2*Nd:3*Nd, :], 0, 1)

        elif self.opts['deMethod'] == 'Euler':
            dt = 0.2
            t_eval = np.arange(t[0], t[-1], dt)
            T_eval = np.zeros(len(t_eval))
            m_eval = np.zeros(len(t_eval))
            X_eval = np.zeros(len(t_eval))
            
            T_eval[0] = yi[0]
            m_eval[0] = yi[Nd]
            X_eval[0] = yi[2*Nd]

            for ii in range(1, len(t_eval)):
                dydt_ii = dydt(t_eval[ii], [T_eval[ii-1], m_eval[ii-1], X_eval[ii-1]])
                T_eval[ii] = T_eval[ii-1] + dydt_ii[0] * dt
                m_eval[ii] = m_eval[ii-1] + dydt_ii[1] * dt
                X_eval[ii] = X_eval[ii-1] + dydt_ii[2] * dt

            To = np.interp(t, t_eval, T_eval)
            mpo = np.interp(t, t_eval, m_eval) / mass_conv
            Xo = np.interp(t, t_eval, X_eval)

        else:
            print('deMethod not available.')

        # Post-process results.
        # Remove added time, if necessary.
        if opts_tadd == 1:
            To = To[1:]
            mpo = mpo[1:]
            Xo = Xo[1:]

        dpo = ((6 * mpo) / (prop.rho(To) * np.pi)) ** (1 / 3) * 1e9  # calculate diameter over time
        # mpo = mpo / np.expand_dims(mpo[:,0], 1)  # would normalize the particle mass

        return To, dpo, mpo, Xo


    def dp(self, mp, T):
        """
        Function for nanoparticle diameter as a function of mass and temperature.
        Allow for contraction / expansion of the particles and change in mass.
        Helper function for the d_dt methods below.
        """
        return np.maximum(1e9 * (6 * mp / (np.pi * self.prop.rho(T))) ** (1./3), 0.0)  # Output in nm

    # Mass component of the ODE.
    def dmdt(self, t, T, mp, X=1.):
        return -self.J_vap(self.prop, T, self.dp(mp, T), X)

    # Temperature component of the ODE.
    def dTdt(self, t, T, mp, X=1.):
        # Start building.
        dTdt = np.zeros_like(t) * np.zeros_like(T) * np.zeros_like(mp) * np.zeros_like(X)
        prop = self.prop
        
        # Conduction model
        if self.opts.get('cond', 'default') != 'none':
            dTdt = dTdt - self.q_cond(prop, T, self.dp(mp, T))[0]

        # Vaporation model
        if self.opts.get('vap', 'default') != 'none':
            dTdt = dTdt - self.q_vap(prop, T, self.dp(mp, T), X)[0]

        # Radiative model
        if self.opts.get('rad', 'none') != 'none':
            dTdt = dTdt - self.q_rad(prop, T, self.dp(mp, T))

        # Absorption model
        abs_option = self.opts.get('abs', 'none')
        if abs_option != 'none':
            if self.opts.get('ann', 'none') == 'none':
                dTdt = dTdt + self.q_abs(prop, t, self.dp(mp, T))[0]
            else:
                dTdt = dTdt + self.q_abs(prop, t, self.dp(mp, T), X)[0]
    
        # Annealing model
        ann_option = self.opts.get('ann', 'none')
        if ann_option != 'none':
            dTdt = dTdt + self.q_ann(prop, T, t, self.dp(mp, T), X)[0]
        
        # Finalize dTdt expression
        dTdt = dTdt / (prop.cp(T) * mp)

        return dTdt

    # Phase change/annealing component of the ODE.
    def dXdt(self, t, T, mp, X):
        # Start building.
        dXdt = np.zeros_like(t * T * mp * X)

        # Pass to annealing model.
        ann_option = self.opts.get('ann', 'none')
        if ann_option != 'none':
            dXdt = self.q_ann(self.prop, T, t, self.dp(mp, T), X)[1]  # get second ouput

            # Accommodate vaporization of only species 1 - X.
            if self.opts['vap_ann']:
                dXdt = dXdt - X * self.dmdt(t, T, mp, X) / mp

        return dXdt


    # Heat transfer submodels
    def q_cond(self, prop, T, dp, model=None, sphere=None):
        """
        Computes the rate of conduction energy loss from the nanoparticle.

        Parameters:
        - self: Instance of the heat transfer model.
        - prop: Properties of the material and gas.
        - T: Vector of nanoparticle temperatures [K].
        - dp: Nanoparticle diameter [nm].
        - model: Optional conduction model specification (default: self.opts['cond']).
        - sphere: Heat transfer from what sphere (default: self.opts['cond_sphere']).

        Returns:
        - q: Rate of conductive losses [W].
        - Kn: Knudsen number (optional).
        """
        if model is None:
            model = self.opts['cond']

        if sphere is None:
            sphere = self.opts['cond_sphere']

        # Convert dp to meters for SI units
        dp = np.array(dp) * 1e-9

        # Convert to equivalent sphere of aggregate, if relevant.
        if sphere == 'eq-sphere':
            if hasattr(prop, 'Rg'):
                Np = prop.kf * (prop.Rg * 1e-9 / (dp/2)) ** prop.Df
            else:
                Np = prop.Np
            dp = dp * (Np / prop.fa) ** (1 / (2 * prop.eps_a))

        # Evaluate the relevant model by calling subfunctions.
        if model in {'free-molecular', 'fm'}:
            q = self.qc_fm(prop, T, dp, prop.Tg)[0]

        elif model == 'continuum':
            q = self.qc_cont(prop, T, dp, prop.Tg)

        elif model in {'transition', 'fuchs', 'mccoy-cha'}:
            q = self.qc_tr(prop, T, dp, prop.Tg, model)

        # Compute Knudsen number if requested as output.
        Kn = None
        if hasattr(prop, 'mu'):
            Kn = self.get_mfp(prop, T) / (dp / 2)

        # Convert to equivalent sphere of aggregate, if relevant.
        if sphere == 'eq-sphere':
            q = q / Np  # convert back to a per primary rate

        return q, Kn
    
    def gamma_r(self, prop, T):
        return (prop.gamma1(T) + 1)/(prop.gamma1(T) - 1)  # builds gamma ratio

    def qc_fm(self, prop, T, dp, Tg):
        """
        Free molecular conduction.

        Parameters:
        - prop: Properties of the material and gas.
        - T: Nanoparticle temperature [K].
        - dp: Nanoparticle diameter [m].
        - Tg: Gas temperature [K].

        Returns:
        - q: Rate of free molecular conduction [W].

        prop.gamma_r is pre-computed when initializing the HTModel class. 
        """
        ct = np.sqrt(8 * KB * Tg / (np.pi * prop.mg))  # average gas speed
        q = prop.alpha * prop.Pg * ct * np.pi * dp ** 2 / (8 * Tg) * self.gamma_r(prop, T) * (T - Tg)
        return q, ct
    
    def kg_star_DT(self, prop, T, Tg):
        """Integrate conductivity over temperature range."""
        T = np.atleast_1d(T)
        return np.array([quad(prop.k, Tg, Ti)[0] for Ti in T])

    def qc_cont(self, prop, T, dp, Tg):
        """
        Continuum regime conduction.

        Parameters:
        - prop: Properties of the material and gas.
        - T: Nanoparticle temperature [K].
        - dp: Nanoparticle diameter [m].
        - Tg: Gas temperature [K].

        Returns:
        - q: Rate of continuum conduction [W].
        """
        return 2 * np.pi * dp * self.kg_star_DT(prop, T, Tg)
    
    def qc_tr(self, prop, T, dp, Tg, model=None):
        """
        Transition regime conduction by Fuchs method.

        Parameters:
        - prop: Properties of the material and gas.
        - T: Nanoparticle temperature [K].
        - dp: Nanoparticle diameter [m].
        - Tg: Gas temperature [K].

        Returns:
        - q: Rate of conduction [W].
        """
        
        if model is None:
            model = self.opts['cond']

        if 'mccoy-cha' in model:
            f = (9 * prop.gamma1(Tg) - 5) / 4
            G = 8 * f / (prop.alpha * (prop.gamma1(Tg) + 1))  # combined, equivalent to G in Michelsen et al. (2015), Eq. (14)

            # Eq. (32) from Liu et al. (2006).
            q = 2 * np.pi * dp**2 * self.kg_star_DT(prop, T, Tg) \
                / (dp + self.get_mfp(prop, Tg, model='mccoy-cha') * G)

        else:  # transition, fuchs
            q = []
            T = np.array(T)
            if T.size == 1:
                T = np.full_like(dp, T)
            
            for Ti, dpi in zip(T, dp):
                def residual(T_delta):
                    d_delta = dpi + 2 * self.get_mfp(prop, T_delta)  # boundary between FM and cont. regimes
                    return np.log(np.abs(self.qc_fm(prop, Ti, dpi, T_delta)[0] / self.qc_cont(prop, T_delta, d_delta, Tg)))

                eps = 1e-12
                T_delta = brentq(residual, np.minimum(Tg, Ti) + eps, np.maximum(Tg, Ti) - eps)
                # T_delta = fsolve(residual, 0.99 * Tg, xtol=1e-12)[0]  # old solver
                q.append(self.qc_fm(prop, Ti, dpi, T_delta)[0])

        q = np.array(q)
        return q

    @staticmethod
    def get_mfp(prop, T, model=None):
        """
        Computes the Maxwell mean free path of the gas.

        Parameters:
        - prop: Properties of the material and gas.
        - T: Gas temperature [K].

        Returns:
        - lambda: Maxwell mean free path [m].
        """
        rho = prop.mg * prop.Pg / (KB * T)

        if model=='mccoy-cha' or not hasattr(prop, 'mu'):  # McCoy and Cha
            f = (9 * prop.gamma1(T) - 5) / 4
            lambda_mfp = prop.k(T) / (prop.Pg * f) * (prop.gamma1(T) - 1) * np.sqrt(T * np.pi * prop.mg / (2 * KB))

        elif hasattr(prop, 'mu'):
            lambda_mfp = prop.mu(T) / (rho * np.sqrt(2 * KB * T / (np.pi * prop.mg)))

        return lambda_mfp


    def q_vap(self, prop, T, dp, X=1.):
        """
        Computes the rate of vaporation or sublimation energy loss from the nanoparticle.

        Parameters:
        - self: Instance of the heat transfer model.
        - prop: Properties of the material and gas.
        - T: Vector of nanoparticle temperatures [K].
        - dp: Nanoparticle diameter [nm].
        - X: Annealed fraction [-]

        Returns:
        - q: Rate of evaporative/sublimative losses [W].
        - J: Vapor flux [kg/s].
        - hv: Latent heat of vaporization/sublimation [J/kg].
        - pv: Vapor pressure [Pa].
        """

        # --- Consider case of multiple vaporizing species. ---
        if hasattr(prop, 'vapor_species'):
            J = q = hv = pv = np.zeros_like(T)

            # Loop through and create namespace with required variables.
            for ii in range(len(prop.vapor_species)):
                prop_ii = prop.vapor_species[ii]  # get subset of properties
                vap = self.q_vap(prop_ii, T, dp, X)
                q = q + vap[0]
                J = J + vap[1]
                hv = hv + vap[2]
                pv = pv + vap[3]
            
            return q, J, hv, pv
        # ------------------------------------------------------

        dp = np.array(dp) * 1e-9  # Convert dp to meters for SI units

        if hasattr(prop, 'gamma'):
            if prop.gamma is None:
                prop.gamma = props.eq_tolman

        if not hasattr(prop, 'alpham'):
            prop.alpham = None
            
        if prop.alpham is None:
            alpham = 1
        else:
            alpham = prop.alpham(T)
        
        # Evaluate local copies of vapor properties.
        hv = prop.hv(T)
        pv = prop.pv(T, dp, prop.hv, X)
        mv = prop.mv(T)

        cv = np.sqrt(np.maximum(8 * KB * T / (np.pi * mv), 0))  # Molecular speed [m/s], max(,0) prevents warnings
        nv = alpham * pv / (KB * T)  # Vapor number flux [m^-3]

        J = mv * nv * cv / 4 * np.pi * dp**2

        q = hv * J

        return q, J, hv, pv
        

    def J_vap(self, prop, T, dp, X=1.):
        """
        Simple bridging function to just output J.
        """
        if self.opts['vap'] == 'none':
            J = np.zeros_like(T)
        else:
            _, J, _, _ = self.q_vap(prop, T, dp, X)
        return J

    def q_rad(self, prop, T, dp):
        # Placeholder for radiation evaluation
        pass

    def q_abs(self, prop, t, dp, X=None):
        """
        Computes the rate of laser energy input into the nanoparticle.

        Parameters:
        - htmodel: Heat transfer model containing options and properties.
        - prop: Dictionary of material and gas properties.
        - t: Time [ns].
        - dp: Nanoparticle diameter [nm].
        - X: Optional, auxiliary variable for material state. Defaults to 1.

        Returns:
        - q: Rate of laser energy uptake by the nanoparticle [W].
        - Cabs: Absorption cross-section [m²].
        - f: Laser profile as a function of time.
        """
        if X is None:
            X = np.ones_like(dp)

        if not hasattr(prop, 'tlm'):
            prop.tlm = 0  # default, center laser pulse at t = 0

        # Convert units to SI
        dp = np.array(dp) * 1e-9  # Convert to meters
        tlp = prop.tlp * 1e-9  # Convert pulse duration to seconds
        tlm = prop.tlm * 1e-9  # Convert pulse midpoint to seconds
        t = np.array(t) * 1e-9  # Convert time to seconds
        F1 = prop.F0 * 100**2  # Convert laser fluence from J/cm² to J/m²

        # Evaluate absorption cross-section
        Cabs = SModel.cabs(dp * 1e9, prop.l_laser, prop, X=X)  # < NOTE: could also add model as input here

        # Define laser profile based on `self.opts.abs`
        abs_option = self.opts.get('abs', 'none')
        if abs_option in {'tophat', 'square'}:  # Square laser profile
            f = lambda t: F1 * (np.heaviside(t - (tlm - tlp / 2), 1) -
                                np.heaviside(t - (tlm + tlp / 2), 1)) / tlp

        elif abs_option in {'gaussian', 'normal', 'include'}:  # Gaussian profile
            sigma = tlp / (2 * np.sqrt(2 * np.log(2)))  # FWHM to standard deviation
            f = lambda t: F1 * norm.pdf(t, loc=tlm, scale=sigma)

        elif abs_option == 'lognormal':  # Lognormal profile
            Sk = 0.9282  # Skewness parameter for lognormal fit
            f_A = np.cbrt(Sk**2 + np.sqrt(Sk**4 + 4 * Sk**2) + 2) / np.cbrt(2)
            f_s = np.sqrt(np.log(f_A + 1 / f_A - 1))
            f_m = (np.log(tlp / (2 * np.sqrt(2 * np.log(2))) /
                        np.sqrt(np.exp(f_s**2) - 1)) -
                0.5 * f_s**2)
            f = lambda t: F1 * np.exp(-(np.log(t + np.exp(f_m)) - f_m)**2 / (2 * f_s**2)) / \
                (t + np.exp(f_m)) / (np.sqrt(2 * np.pi) * f_s)
            
        elif abs_option == 'custom':
            f = lambda t: prop.f(t) * F1  # expects function

        else:
            raise ValueError(f"Unknown laser profile option: {abs_option}")

        # Calculate rate of laser energy uptake
        q = Cabs * f(t)

        return q, Cabs, f


    def q_ann(self, prop, T, t, dp, X):
        """
        Wrapper function for annealing rates.
        
        Parameters:
        htmodel : object (not used in this function but kept for compatibility)
        prop : object, must contain attributes 'R', 'rho' (method), and 'M'
        T : float, temperature in Kelvin
        dp : float or np.array, particle diameter in nm
        X : float or np.array, fraction of particle
        
        Returns:
        q : float or np.array, annealing rate
        dXdt : float or np.array, rate of change of fraction of particle
        """
        dp = dp * 1e-9  # Convert to meters (SI units)

        ann_option = self.opts.get('ann', 'none')
        if ann_option in ['include', 'volumetric',  'michelsen']:
            q, dXdt = self.q_ann_volumetric(prop, T, dp, X)
        elif ann_option in ['sipkens', 'concentric']:
            q, dXdt = self.q_ann_concentric(prop, T, dp, X)
        elif ann_option == 'photo':
            q, dXdt = self.q_ann_photo(prop, T, t, dp, X)
        else:
            q, dXdt = np.ones_like(T), np.ones_like(T)
        
        return q, dXdt
    
    @staticmethod
    def arrhenius(A, E, T):
        with np.errstate(over='ignore'):  # can overflow, that is okay
            return A * np.exp(-E / (R * T))


    def q_ann_volumetric(self, prop, T, dp, X):
        """
        Implementation of volumetric annealing rate calculation following Michelsen (2003).
        
        Parameters:
        htmodel : object (not used in this function but kept for compatibility)
        prop : object, must contain attributes 'R', 'rho' (method), and 'M'
        T : float, temperature in Kelvin
        dp : float or np.array, particle diameter in nm
        X : float or np.array, fraction of particle
        
        Returns:
        q : float or np.array, annealing rate
        dXdt : float or np.array, rate of change of fraction of particle

        Notes:
        1. Arrhenius for various transformations in the material
        are contained in prop.A and prop.E, as arrays.

        2. Equations are adjusted, such that dXdt is phrased simply in terms of X.
        This makes the volumetric formulation more explicitly.
        """

        # Compute annealing and heat transfer rates.
        q = np.zeros_like(T)
        dXdt = np.zeros_like(T)
        
        for ii in range(len(prop.A)):
            k = self.arrhenius(prop.A[ii], prop.E[ii], T)
            q = q + prop.DH[ii] * k

            # Rearrange Eq. (38) and take derivative.
            dXdt = dXdt + k * (1 - X)

        # Modifier for dissociation.
        if hasattr(prop, 'Edis'):
            k = self.arrhenius(prop.Adis, prop.Edis, T)
            dXdt = dXdt - k * X / 2 / prop.Xd
        
        # Incorporate pre-factor for heat transfer.
        Np = (dp**3 * np.pi * prop.rho(T)) / (6 * prop.M)  # moles of atoms in nanoparticle
        fd = (1 - X) * prop.Xd  # fraction of "atoms" that are defects
        q = Np * fd * q
        
        return q, dXdt

    def q_ann_concentric(self, prop, T, dp, X):
        """
        Rate of annealing based on the simplified model of Sipkens (2019).
        
        Parameters:
        htmodel : object (not used in this function but kept for compatibility)
        prop : object, must contain attributes 'R', 'rho' (method), 'M', and optionally 'E', 'k0'
        T : float, temperature in Kelvin
        dp : float or np.array, particle diameter in nm
        X : float or np.array, fraction of particle
        
        Returns:
        q : float or np.array, rate of annealing
        dXdt : float or np.array, rate of change of fraction of particle
        """        
        dd_dt = 2 * self.arrhenius(prop.A, prop.E, T)
        
        # Compute rate of change of fraction of particle
        X = np.maximum(X, 0.)  # bound X
        X = np.minimum(X, 1.)
        dXdt = 3 * (1 - X) ** (2/3) / dp * dd_dt
        
        # Compute annealing rate q
        Np = dp**3 * np.pi * prop.rho(T) / (6 * prop.M)  # moles of atoms in nanoparticle
        q = prop.DH * dXdt * Np
        
        return q, dXdt

    def q_ann_photo(self, prop, T, t, dp, X):
        """
        Rate of annealing including photoannealing.
        
        Parameters:
        htmodel : object (not used in this function but kept for compatibility)
        prop : object, must contain attributes 'R', 'rho' (method), 'M', and optionally 'E', 'k0'
        T : float, temperature in Kelvin
        dp : float or np.array, particle diameter in nm
        X : float or np.array, fraction of particle
        
        Returns:
        q : float or np.array, rate of annealing
        dXdt : float or np.array, rate of change of fraction of particle
        """
        # Set default values if 'E' and 'k0' are not attributes of prop
        t = np.array(t) * 1e-9  # Convert time to seconds
        _, _, f = self.q_abs(prop, t, dp, X)
        
        dd_dt = 2 * prop.A0 * np.exp(-prop.E / (R * T)) + \
            2 * prop.Apho * prop.Em(prop.l_laser, dp, X) * f(t) / (prop.l_laser * 1e-9)
        
        DH = 0  # multiply by percentage of defects existing
        
        # Compute rate of change of fraction of particle
        X = np.maximum(X, 0.)  # bound X
        X = np.minimum(X, 1.)
        dXdt = 3 * (1 - X) ** (2/3) / dp * dd_dt
        
        # Compute annealing rate q
        q = (DH * dXdt * dp**3 * np.pi * prop.rho(T)) / (6 * prop.M)
        
        return q, dXdt
