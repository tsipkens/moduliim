
# A simple set of functions to approximate temperature curves with laser fluence. 
# 
# Based on Sipkens and Daun, # "Defining regimes and analytical expressions 
# for fluence curves in pulsed laser heating of aerosolized nanoparticles." 
# Optics Express 28.3 (2020): 3301-3316.s


import numpy as np
from scipy.special import lambertw
from scipy.optimize import minimize


H = 6.62606957e-34  # Planck"s constant [m^2.kg/s]
C = 2.99792458e8    # Speed of light in a vacuum [m/s]
KB =  1.3806488e-23 # Boltzmann constant [m^2.kg/s^2/K]
R = 8.3144621       # Universal gas constant [J/mol/K]
PHI = H * C / KB    # Planck's constant factor


def get_prop(prop, Teval=2500):
    """
    GET_PROP: Formats property dictionary for input to other functions.
    
    This sets up the proper format for the property dictionary used in evaluating
    functions. Units can be altered from those given, but must be consistent 
    so that non-dimensional quantities remain valid (units must cancel out).

    Parameters
    ----------
    prop : Prop, optional
        Instance of property class to be used in estimation. 

    Returns
    -------
    dict
        Dictionary of physical properties with derived parameters and helper functions.
    """

    # -- Vaporization properties ------------------ #
    prop.mv = prop.M * 1.660538782e-24  # mass of vapor, kg
    prop.Rs = R / prop.M  # specific gas constant, m^2/(s^2*K)
    prop.hvb = prop.hvb / prop.M  # latent heat of vaporization, J/kg

    # -- Clausius-Clapeyron (C-C) equation -------- #
    prop.A = np.exp(np.log(prop.Pref) + prop.hvb / prop.Rs / prop.Tb)  # Pa
    
    prop.add('pv', 'lambda self, T: self.A * np.exp(-self.hvref / self.Rs / T)')  # Pa

    # Convert density and specific heat to values.
    # prop.rho = prop.rho(Teval)
    # prop.cp = prop.cp(Teval)

    prop.C1 = (
        KB * np.pi / (9 * prop.Rs * prop.hvb * prop.mv)
        * (prop.rho * prop.cp / prop.A) ** 2
    )  # As defined in Ref. [1]

    return prop


def get_ref(prop):
    """
    Calculates fluence regime transition temperature and fluence.

    This function takes a property object/dictionary and outputs the reference or
    transition fluence (Fref) and temperature (Tref).

    Parameters
    ----------
    prop : object or dict
        Must contain the following attributes/keys:
        - hvb   : latent heat of vaporization (J/kg)
        - Rs    : specific gas constant (J/(kg*K))
        - C1    : dimensionless constant
        - dp    : particle diameter (m)
        - Tg    : gas temperature (K)
        - tlp   : laser pulse length (s)
        - l_laser : laser wavelength or path length (m)
        - rho   : density (kg/m^3)
        - cp    : specific heat capacity (J/(kg*K))
        - Eml   : some material constant (same units as numerator)

    Returns
    -------
    Tref : float
        Reference (transition) temperature in Kelvin.
    Fref : float
        Reference (transition) fluence in J/cm^2.

    Notes
    -----
    Translated from MATLAB's `get_ref` function.
    The MATLAB `fminsearch` call is replaced by `scipy.optimize.minimize`
    using the Nelder-Mead method.
    """
    # Helper to access either dict or object
    get = lambda k: prop[k] if isinstance(prop, dict) else getattr(prop, k)

    # Function to minimize
    def min_fun(Tref):
        return np.linalg.norm(
            2 * get('hvb') / get('Rs') +
            Tref * lambertw(
                -get('C1') * (get('dp') * (Tref - get('Tg')) / get('tlp')) ** 2,
                k=-1
            ).real
        )

    # Optimization to find reference temperature
    T0 = 3000.0  # starting guess, K
    res = minimize(min_fun, T0, method='Nelder-Mead')
    Tref = res.x[0]

    # Reference fluence from reference temperature
    Fref = (
        get('l_laser') * get('rho') * get('cp') * (Tref - get('Tg'))
        / (6 * np.pi * get('Eml')) / 10000.0  # convert to J/cm^2
    )

    return Tref, Fref



def gen_peak_fun(prop, opts='default', n=-10, Tref=None, Fref=None):
    """
    Generates three functions describing the peak temperature as a function of laser fluence.

    This function produces:
        (1) An interpolated function between the high-fluence and low-fluence regimes.
        (2) The high-fluence regime expression.
        (3) The low-fluence regime expression.

    Parameters
    ----------
    prop : object
        Object containing material/experiment properties, such as:
        - hvb : float
        - Rs : float
        - C1 : float
        - dp : float
        - Tg : float
        - tlp : float
    opts : str, optional
        Output mode:
        - 'dimless' or 'dimensionless' : 
            Returns functions with dimensionless fluence (F1) 
            producing dimensionless temperature (relative to Tref, Tg).
        - 'mix' :
            Returns functions with dimensionless fluence (F1) producing temperature in K.
        - 'default' :
            Returns functions with fluence in J/cm² producing temperature in K.
        Default is 'default'.
    n : float, optional
        Power in the interpolation function. Default is -10.
    Tref : float, optional
        Reference temperature (K). If None, will be obtained from get_ref(prop).
    Fref : float, optional
        Reference fluence (J/cm²). If None, will be obtained from get_ref(prop).

    Returns
    -------
    T_fun : callable
        Interpolated temperature function.
    T_high : callable
        High-fluence regime temperature function.
    T_low : callable
        Low-fluence regime temperature function.

    Notes
    -----
    - The high-fluence expression uses the -1 branch of the Lambert W function.
    - By default, input fluence is expected in J/cm² for 'default' mode.
    """

    # Get default reference values if not provided
    if Tref is None or Fref is None:
        Tref, Fref = get_ref(prop)

    # Define high-fluence expression (dimensionless fluence F1)
    def T_high_dl(F1):
        val = -prop.C1 * (prop.dp * (Tref - prop.Tg) / prop.tlp * F1) ** 2
        return np.real(
            -2 * prop.hvb / prop.Rs / lambertw(val, k=-1)
        )

    # Low-fluence expression (dimensionless)
    def T_low_dl(F1):
        return F1 * (Tref - prop.Tg) + prop.Tg

    # Interpolated function (dimensionless)
    def T_fun_dl(F1):
        return (T_high_dl(F1) ** n + T_low_dl(F1) ** n) ** (1.0 / n)

    # Select output format
    if opts in ('dimless', 'dimensionless'):
        T_fun = lambda F1: (T_fun_dl(F1) - prop.Tg) / (Tref - prop.Tg)
        T_high = lambda F1: (T_high_dl(F1) - prop.Tg) / (Tref - prop.Tg)
        T_low = lambda F1: (T_low_dl(F1) - prop.Tg) / (Tref - prop.Tg)
    elif opts == 'mix':
        T_fun = T_fun_dl
        T_high = T_high_dl
        T_low = T_low_dl
    else:  # default: fluence in J/cm² → dimensionless via Fref
        T_fun = lambda F0: T_fun_dl(F0 / Fref)
        T_high = lambda F0: T_high_dl(F0 / Fref)
        T_low = lambda F0: T_low_dl(F0 / Fref)

    return T_fun, T_high, T_low


def estimatej(F0, prop, l):
    """
    Estimate the peak incandescence curve.

    Parameters
    ----------
    F0 : array_like
        Fluence values to evaluate (J/cm² or appropriate units).
    prop : object
        Properties object with at least:
            - hvb : Latent heat of vaporization (J/kg)
            - cp  : Specific heat capacity (J/kg·K)
    l : float
        Wavelength (m).

    Returns
    -------
    J : ndarray
        Peak incandescence curve accounting for evaporation.
    DM : ndarray
        Mass loss factor (0 to 1).
    J0 : ndarray
        Incandescence curve without evaporation losses.

    Notes
    -----
    Translated from MATLAB code by Timothy Sipkens, 2021-06-16.
    """
    phi = 1.44e-2  # constant from original MATLAB code

    # Generate peak temperature functions
    T, _, Tlow = gen_peak_fun(prop)

    # Extra energy to evaporation [K]
    DT = Tlow(F0) - T(F0)

    # Mass loss factor (clamped to >= 0)
    DM = np.maximum(1 - (DT / prop.hvb) * prop.cp, 0)

    # Peak incandescence accounting for evaporation
    J = DM / (np.exp(phi / T(F0) / l) - 1)

    # Incandescence without evaporation
    J0 = 1 / (np.exp(phi / T(F0) / l) - 1)

    return J, DM, J0
