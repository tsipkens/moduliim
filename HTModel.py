import numpy as np

from scipy.optimize import fsolve
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
        self.prop = copy.copy(prop)
        self.x = x if isinstance(x, list) else [x]
        self.t = t
        self.opts = {
            'cond': 'free-molecular',  # Conduction model
            'cond_sphere': 'primary',  # Heat transfer from equivalent sphere ('eq-sphere') or primary particle ('primary')
            'evap': 'free-molecular',  # Evaporation model
            'rad': 'none',  # Radiation model
            'abs': 'none',  # Absorption model
            'ann': 'none',  # Annealing model
            'polydispersity': 0,  # Incorporate polydispersity
            'deMethod': 'default'  # ODE solver method
        }
        self.opts.update(kwargs)  # Parse additional options

        self._print_properties()

    def _print_properties(self):
        print('\r' +'\033[32m' + 'HTModel > opts:' + '\033[0m')
        print(f" \033[34m Conduction\033[0m → {self.opts['cond']} (from {self.opts['cond_sphere']})")
        print(f" \033[34m Evaporation\033[0m → {self.opts['evap']}")
        print(f" \033[34m Absorption\033[0m → {self.opts['abs']}" + (" (Gaussian)" if self.opts['abs'] == 'include' else ""))
        print(f" \033[34m Radiation\033[0m → {self.opts['rad']}")
        print(f" \033[34m Annealing\033[0m → {self.opts['ann']}")
        print(' ')


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

        # Initial conditions
        Ti = prop.Ti * np.ones(Nd)  # initial temperature, [K]

        # Initial mass
        if not hasattr(prop, 'rho0'):
            prop.rho0 = prop.rho(prop.Tg)

        mass_conv = 1e21  # converts mass to attogram (ag)
        mpi = (prop.rho0 * (dp0 * 1e-9) ** 3 * (np.pi / 6)) * mass_conv  # initial mass, [ag]

        if hasattr(prop, 'Xi'):
            Xi = np.asarray(prop.Xi)
        else:
            Xi = np.array([1])

        # Starting point exception
        if t[0] > 0.1:  # allows for initial condition at t=0 instead of first entry in time vector
            t = np.concatenate(([0], t))
            opts_tadd = 1
        else:
            opts_tadd = 0

        # Define the system of ODEs
        def dydt(t, y):
            T = y[:Nd]
            m = np.abs(y[Nd:2*Nd]) / mass_conv
            X = y[2*Nd:3*Nd] if len(y) > 2 * Nd else None

            if self.opts['ann'] == 'none':
                dTdt = self.dTdt(t, T, m)
                dmdt = self.dmdt(t, T, m)
                return np.concatenate([dTdt * 1e-9, dmdt * mass_conv * 1e-9])
            else:
                dTdt = self.dTdt(t, T, m, X)
                dmdt = self.dmdt(t, T, m, X)
                dXdt = self.dXdt(t, T, m, X)
                return np.concatenate([dTdt * 1e-9, dmdt * mass_conv * 1e-9, dXdt * 1e-9])

        # Initial state
        yi = np.concatenate([Ti, mpi, Xi] if self.opts['ann'] != 'none' else [Ti, mpi])

        # Solve the ODE
        if self.opts['deMethod'] in ['default']:  # specifics of ODE solver call
            
            sol = solve_ivp(dydt, (t[0], t[-1]), yi, t_eval=t, method='BDF')  # use BDF method as good for stiff ODEs

            Tout = np.maximum(sol.y[:Nd, :], prop.Tg)
            mpo = sol.y[Nd:2*Nd, :] / mass_conv

            if self.opts['ann'] == 'none':
                Xo = Xi * np.ones_like(Tout)
            else:
                Xo = sol.y[2*Nd:3*Nd, :]

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

            Tout = np.interp(t, t_eval, T_eval)
            mpo = np.interp(t, t_eval, m_eval) / mass_conv
            Xo = np.interp(t, t_eval, X_eval)

        else:
            print('deMethod not available.')

        # Post-process results
        if opts_tadd == 1:
            Tout = Tout[1:]
            mpo = mpo[1:]
            Xo = Xo[1:]

        dpo = ((6 * mpo) / (prop.rho(Tout) * np.pi)) ** (1 / 3) * 1e9  # calculate diameter over time
        # mpo = mpo / np.expand_dims(mpo[:,0], 1)  # output relative change in particle mass over time

        return Tout, dpo, mpo, Xo


    def dp(self, mp, T):
        """
        Function for nanoparticle diameter as a function of mass and temperature.
        Helper function for the d_dt methods below.
        """
        return 1e9 * (6 * mp / (np.pi * self.prop.rho(T))) ** (1./3)  # Output in nm

    # Mass component of the ODE.
    def dmdt(self, t, T, mp, X=1.):
        return -self.J_evap(self.prop, T, self.dp(mp, T))

    # Temperature component of the ODE.
    def dTdt(self, t, T, mp, X=1.):
        # Start building.
        dTdt = np.zeros_like(t) * np.zeros_like(T) * np.zeros_like(mp) * np.zeros_like(X)
        prop = self.prop
        
        # Conduction model
        if self.opts.get('cond', 'default') != 'none':
            dTdt = dTdt - self.q_cond(prop, T, self.dp(mp, T))

        # Evaporation model
        evap_option = self.opts.get('evap', 'default')
        if evap_option == 'mult':
            dTdt = dTdt - self.q_evapm(prop, T, self.dp(mp, T))[0]
        elif evap_option != 'none':
            dTdt = dTdt - self.q_evap(prop, T, self.dp(mp, T))[0]

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
            q = self.qc_fm(prop, T, dp, prop.Tg)

        elif model == 'continuum':
            q = self.qc_cont(prop, T, dp, prop.Tg)

        elif model in {'transition', 'fuchs'}:
            q = self.qc_tr(prop, T, dp, prop.Tg)

        # Compute Knudsen number if requested as output.
        Kn = None
        if hasattr(prop, 'mu'):
            Kn = self.get_mfp(prop, T) / (dp / 2)

        # Convert to equivalent sphere of aggregate, if relevant.
        if sphere == 'eq-sphere':
            q = q / Np  # convert back to a per primary rate

        return (q, Kn) if Kn is not None else q

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
        """
        alpha = np.clip(prop.alpha, 0, 1)
        q = ((alpha * prop.Pg * prop.ct() * np.pi * (dp ** 2) / (8 * Tg)) *
            prop.gamma2(T) * (T - Tg))
        return q

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
        def conductivity(T):
            return prop.k(T)

        q = 2 * np.pi * dp * np.array([quad(conductivity, Tg, Ti)[0] for Ti in T])
        return q
    
    def qc_tr(self, prop, T, dp, Tg):
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
        q = []
        T = np.array(T)
        if T.size == 1:
            T = np.full_like(dp, T)
        
        for Ti, dpi in zip(T, dp):
            def residual(T_delta):
                return self.q_fm(prop, Ti, dpi, T_delta) - self.q_cont(prop, T_delta, dpi + 2 * self.get_mfp(prop, T_delta), Tg)

            T_delta = fsolve(residual, [Tg, Ti])[0]
            q.append(self.q_fm(prop, Ti, dpi, T_delta))

        q = np.array(q)
        return q


    def get_mfp(self, prop, T):
        """
        Computes the Maxwell mean free path of the gas.

        Parameters:
        - prop: Properties of the material and gas.
        - T: Gas temperature [K].

        Returns:
        - lambda: Maxwell mean free path [m].
        """
        rho = prop.mg * prop.Pg / (KB * prop.Tg)

        if hasattr(prop, 'mu'):
            lambda_mfp = prop.mu(T) / (rho * np.sqrt(2 * KB * prop.Tg / (np.pi * prop.mg)))

        else:  # McCoy and Cha
            lambda_mfp = prop.k(T) / (prop.Pg * prop.f) * (prop.gamma1(T) - 1) / (rho * np.sqrt(2 * KB * prop.Tg / (np.pi * prop.mg)))

        return lambda_mfp


    def q_evap(self, prop, T, dp):
        """
        Computes the rate of evaporation or sublimation energy loss from the nanoparticle.

        Parameters:
        - self: Instance of the heat transfer model.
        - prop: Properties of the material and gas.
        - T: Vector of nanoparticle temperatures [K].
        - dp: Nanoparticle diameter [nm].

        Returns:
        - q: Rate of evaporative/sublimative losses [W].
        - J: Vapor flux [kg/s].
        - hv: Latent heat of vaporization/sublimation [J/kg].
        - pv: Vapor pressure [Pa].
        """
        dp = np.array(dp) * 1e-9  # Convert dp to meters for SI units
        prop = self.prop

        if hasattr(prop, 'gamma'):
            if prop.gamma is None:
                prop.gamma = props.eq_tolman

        if not hasattr(prop, 'alpham'):
            prop.alpham = None
            
        if prop.alpham is None:
            prop.alpham = lambda T: 1

        hv = prop.hv(T)
        pv = prop.pv(T, dp, hv)
        mv = prop.mv(T) if callable(prop.mv) else prop.mv

        cv = np.sqrt(np.maximum(8 * KB * T / (np.pi * mv), 0))  # Molecular speed [m/s], max(,0) prevents warnings
        nv = prop.alpham(T) * pv / (KB * T)  # Vapor number flux [m^-3]

        J = mv * nv * cv / 4 * np.pi * dp**2
        q = hv * J

        return q, J, hv, pv
        

    def J_evap(self, prop, T, dp):
        """
        Simple bridging function to just output J.
        """
        _, J, _, _ = self.q_evap(prop, T, dp)
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
        if ann_option in ['include', 'michelsen']:
            q, dXdt = self.q_ann_mich(prop, T, dp, X)
        elif ann_option == 'sipkens':
            q, dXdt = self.q_ann_sip(prop, T, dp, X)
        elif ann_option == 'photo':
            q, dXdt = self.q_ann_photo(prop, T, t, dp, X)
        else:
            q, dXdt = 0, 0
        
        return q, dXdt

    def q_ann_mich(self, prop, T, dp, X):
        """
        Implementation of annealing rate calculation from Michelsen (2003).
        
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
        Na = 6.0221409e23  # Avogadro's number
        Np = (dp**3 * np.pi * prop.rho(T)) / (6 * prop.M) * Na  # number of atoms in nanoparticle
        Xd = 0.01  # Initial defect density
        
        def k_fun(A, E):
            with np.errstate(over='ignore'):  # can overflow, that is okay
                return A * np.exp(-E / (R * T))
        
        # Arrhenius for various transformations in the material.
        A_dis, E_dis = 1e18, 9.6e5
        k_dis = k_fun(A_dis, E_dis)  # dissociation
        A_int, E_int = 1e8, 8.3e4
        k_int = k_fun(A_int, E_int)  # interstitial movement
        A_vac, E_vac = 1.5e17, 6.7e5
        k_vac = k_fun(A_vac, E_vac)  # vacancy movement
        
        DH_int, DH_vac = -1.9e4, -1.4e5  # Energy changes
        Nd = (1 - X) * (Xd * Np)  # Number of defects
        
        q = -(DH_int * k_int + DH_vac * k_vac) * Nd / Na

        # Rearrange Eq. (38) and take derivative.
        dXdt = -1 / (Xd * Np) * (X * Np / 2 * k_dis - (k_int + k_vac) * Nd)
        
        return q, dXdt

    def q_ann_sip(self, prop, T, dp, X):
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
        # Set default values if 'E' and 'k0' are not attributes of prop
        E = getattr(prop, 'E', 4e5)  # default: 5e5 from Newell, J. Appl. Polymer Sci., 1996
        A0 = getattr(prop, 'A0', 1e5)  # default: 4e12 form Michelsen et al., 2003 prev. 1e5, changed again from 3e-8 * 2e13 / 5
        
        dd_dt = 2 * A0 * np.exp(-E / (R * T))
        
        DH = 0  # multiply by percentage of defects existing
        
        # Compute rate of change of fraction of particle
        X = np.maximum(X, 0.)  # bound X
        X = np.minimum(X, 1.)
        dXdt = 3 * (1 - X) ** (2/3) / dp * dd_dt
        
        # Compute annealing rate q
        q = (DH * dXdt * dp**3 * np.pi * prop.rho(T)) / (6 * prop.M)
        
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
