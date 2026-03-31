import numpy as np

from tqdm import tqdm

from scipy.sparse import spdiags
from scipy.optimize import minimize

# Define constants.
H = 6.62606957e-34  # Planck"s constant [m^2.kg/s]
C = 2.99792458e8    # Speed of light in a vacuum [m/s]
KB =  1.3806488e-23 # Boltzmann constant [m^2.kg/s^2/K]
R = 8.3144621       # Universal gas constant [J/mol/K]
PHI = H * C / KB    # Planck's constant factor

class SModel:
    """
    Class for the spectroscopic model, which enables pyrometry calculations.

    Parameters:
    -----------
    prop : object
        Material properties.
    x : list
        Variable names, quantities of interest (QoIs).
    t : array-like
        Time points.
    l : array-like
        Wavelengths.
    **kwargs : dict
        Additional options for configuration.
    """

    def __init__(self, prop, x=['dp0'], t=np.array([0]), lam=np.array([442,716]), **kwargs):
        self.prop = prop  # material properties
        self.lam = np.array(lam)  # wavelengths
        self.the = np.array([])  # angles (for scattering)
        self.t = np.array(t)  # time vector
        self.T = None  # temperature, function handle
        self.J = None  # incandescence
        self.htmodel = None  # Embedded heat transfer model

        self.x = x  # variable names, quantities of interest (QoIs)

        self.data_sc = None  # used to scale Planck's law for stability

        # Options for multicolor solver and pyrometry
        self.opts = {
            'multicolor_solver': 'default',  # indicates which multicolor solver to use
            'pyrometry': 'ratio',  # indicates how to handle pyrometry
            'model': 'rdg-fa',  # default spectroscopic model (rdg-fa, Mie)
        }
        self.opts.update(kwargs)  # update with additional options if provided

        # Scale Planck's law for stability (closer to 1).
        # Uses dp = 30 nm for data scaling.
        T_sc = np.array([[3000]])  # Temperature for scaling/stability [K]
        self.data_sc = self.blackbody(T_sc, 1064) / (1064e-9)

        print(self)

    def __repr__(self):
        lines = []
        lines.append('\r' +'\033[32m' + 'SModel > opts:' + '\033[0m')
        for key, value in self.opts.items():
            lines.append(f"  \033[34m{key}\033[0m → {value}")
        return "\n".join(lines)


    def inverse(self, J, X=None):
        """
        Use default method to evaluate inverse model, converting incandescence to temperature. 
        """
        if X is None:  # if no annealing, set to ones of same size as T
            X = np.array([1])

        if self.opts['pyrometry'] == 'ratio':
            return self.pyrometry_ratio(J[:,:,0], J[:,:,1], X=X)[0]
        else:
            return self.spectral_fit(J)
        
    def forward(self, T, prop=None, dp0=None, X=None, mp=None):
        """
        Use default method to evaluate forward model, converting temperature to incandescence. 
        """
        if X is None:  # if no annealing, set to ones of same size as T
            X = np.ones_like(T)

        if prop is None:  # if no prop, inherit from class instance
            prop = self.prop

        if dp0 is None:  # if no size, inherit from prop
            dp0 = prop.dp0

        if mp is None:
            mp = np.ones_like(T)  # if no mass, assume unit mass everywhere

        mp = np.expand_dims(mp, [2])  # expand for lambda dimension
        Cabs = self.cabs(dp0, self.lam, prop, self.opts['model'], np.expand_dims(X, 2))
        Ib = self.blackbody(T, self.lam)

        return mp * Cabs * Ib


    @staticmethod
    def cabs(d, lam, prop, model='rdg-fa', X=None):
        """
        Calculate the absorption cross section.

        Parameters:
        -----------
        d : ndarray
            Particle diameter(s) in nm.
        lam : ndarray
            Wabelength(s) in nm.
        prop : object
            Material properties.
        X : ndarray
            Annealed fraction (optional)
        """
        if X is None:  # if no annealing, set to ones of same size as T
            X = np.array([1])

        # Evaluate cross section.
        if 'rdg-fa' in model or 'rayleigh' in model:  # RDG-FA (volumetric), same as Rayleigh except for Npp
            if hasattr(prop, 'm'):
                Em = ((prop.m**2 - 1) / (prop.m**2 + 2)).imag
            else:
                Em = prop.Em(lam, d, X)
            Cabs = np.pi ** 2 * (d * 1e-9) ** 3 / (lam * 1e-9) * Em

        elif 'mie' in model:  # Mie absorption
            import miepython as mie  # import mie library
            qext, qsca, _, _ = mie.efficiencies(prop['m'], d, lam)
            Cabs = (qext - qsca) * (np.pi * d ** 2 / 4)  # multiple eff. by cross section

        else:
            Cabs = None  # unsupported model

        # Modify absorption using an AAE.
        if 'aae' in model:
            Cabs = Cabs * (lam / prop.lam_ref) ** (-prop.aae + 1)  # apply AAE scaling

        return Cabs

    @staticmethod
    def blackbody(T, lam):
        """
        Calculate blackbody radiation intensity at a given temperature and wavelength.
        
        Parameters:
        -----------
        T : float
            Temperature in Kelvin.
        lam : float
            Wavelength in nanometers.
        
        Returns:
        --------
        float
            Blackbody radiation intensity.
        """
        lam = np.expand_dims(lam * 1e-9, [0,1])  # Convert nm to meters
        T = np.expand_dims(T, 2)

        # Planck's law
        numerator = 2 * H * C**2 / lam**5
        denominator = np.exp(H * C / (lam * KB * T)) - 1
        return numerator / denominator
    

    def pyrometry_ratio(self, J1, J2, Emr=None, idx=None, X=None):
        """
        Evaluate temperature by two-color pyrometry.

        Parameters:
        -----------
        J1 : ndarray
            Incandescence at the first wavelength.
        J2 : ndarray
            Incandescence at the second wavelength.
        Emr : float, optional
            Ratio of the absorption function at the two wavelengths. If excluded,
            the value is extracted from self.prop.
        idx : list of int, optional
            Indices specifying which wavelengths to use for pyrometry.

        Returns:
        --------
        To : ndarray
            Calculated temperature.
        Co : ndarray
            Scaling constant.
        """
        lam = self.lam  # Local copy of wavelengths

        # Check the number of wavelengths
        if len(lam) > 2:
            if idx is None:
                raise ValueError(
                    "More than two wavelengths in SModel. Provide indices (idx) for pyrometry."
                )
            if len(idx) != 2:
                raise ValueError("Invalid indices: idx must have exactly two values.")
            lam = lam[idx]

        # Handle Emr input
        if Emr is None:
            if hasattr(self.prop, 'Emr'):  # then use Emr directly (allows for overwriting fo Em)
                Emr = self.prop.Emr(lam[0], lam[1], self.prop.dp0)  # evaluate function
            else:  # use Em function at the two wavelengths
                Emr = self.prop.Em(lam[0], self.prop.dp0, X) / self.prop.Em(lam[1], self.prop.dp0, X)

        # Ratio of incandescence
        # Pre-allocation and only evaluate select (avoids error messages)
        Jr = np.empty_like(J1)
        Jr[:] = np.nan
        isEval = np.logical_and(J1 > 0, J2 > 0)  # if both signals are non-zero
        Jr[isEval] = J1[isEval] / J2[isEval]

        # Basic ratio calculation
        To = (PHI * (1 / (lam[1] * 1e-9) - 1 / (lam[0] * 1e-9))) / np.log(
            Jr * ((lam[0] / lam[1])**6) / Emr
        )

        # Calculate scaling constant.
        Co = J1 / (self.blackbody(To, lam[0])[:,:,0])  # not currently absolute scaling

        return To, Co
    

    def spectral_fit(self, J):
        """
        Spectral fitting, sequential.

        Parameters:
        -----------
        J : ndarray
            Incandescence data of shape (ntime, nshots, nwavelengths).

        Returns:
        --------
        To : ndarray
            Calculated temperatures.
        Co : ndarray
            Scaling constants.
        s_T : ndarray
            Standard deviations of temperatures.
        out : dict
            Additional output data (currently empty).
        """
        ntime, nshots, _ = J.shape
        s = np.std(J, axis=1) / np.sqrt(J.shape[1])  # standard error
        prop = self.prop

        # Ensure `C_J` exists in `prop`
        if not hasattr(prop, 'C_J') or prop.C_J is None:
            prop.C_J = 1

        lam = self.lam
        s_T = np.zeros((ntime, nshots))
        s_C = np.zeros((ntime, nshots))

        T0, _ = self.pyrometry_ratio(J[:,:,0], J[:,:,1], Emr=None)

        # Define the model based on options
        if self.opts['multicolor_solver'] == "default":
            x0 = [1e3, 0]
            def bb1(x):
                return (10 ** x[1]) * self.blackbody([[x[0]]], lam) * (
                    prop.Em(lam, prop.dp0, 1) / (lam * 1e-9)
                )
        elif self.opts['multicolor_solver'] == "constC":
            x0 = [1e3]
            def bb1(x):
                return prop.C_J * self.blackbody([[x[0]]], lam) * (
                    prop.Em(lam, prop.dp0, 1) / (lam * 1e-9)
                )
        else:
            raise ValueError(f"Unknown multicolor option: {self.opts['multicolor_solver']}")

        beta = np.zeros((ntime, nshots, len(x0)))
        resid = np.zeros((ntime, nshots))

        print("Calculating temperatures:")
        for ii in tqdm(range(ntime)):  # Time loop
            for jj in range(nshots):  # Shot loop
                if self.opts['multicolor_solver'] == "constC":
                    x0 = [T0[ii, jj]]
                else:
                    x0 = [T0[ii, jj], 0]
                    C0 = np.log10(J[ii,jj,0] / bb1(x0)[:,:,0])[0]
                    x0 = np.hstack((T0[ii, jj], C0))
                
                data = J[ii, jj, :]
                data_std = s[ii, :]
                data_std[data_std == 0] = 1e-3  # Avoid zero errors
                nn = len(data)

                # Diagonal matrix for weighting
                data_Li = spdiags(1.0 / data_std, 0, nn, nn).toarray()

                # Likelihood function
                def likelihood(x):
                    return np.sum(
                        (np.log(data) - np.log(bb1(x)))**2
                    )

                # Minimize likelihood
                result = minimize(likelihood, x0, method='BFGS')
                mle = result.x
                jacobian = result.jac
                
                # Restore constant to full values
                # mle[1] = 10 ** (mle[1] / tmp[0] * data[0])
                # G_T = np.linalg.inv(jacobian.T @ jacobian)

                beta[ii, jj, :] = mle
                # s_T[ii, jj] = np.sqrt(G_T[0, 0])

        # Assign outputs based on multicolor option
        if self.opts['multicolor_solver'] in {"priorC", "default", "priorT"}:
            Co = beta[:, :, 1]
        elif self.opts['multicolor_solver'] == "constC":
            Co = np.full((ntime, nshots), prop.C_J)
        else:
            Co = None

        To = beta[:, :, 0]
        out = {}

        return To



    @staticmethod
    def csca(d, lam, the, prop, model='rdg-fa', X=None, Rg=None):
        """
        Calculate the absorption cross section.

        Parameters:
        -----------
        d : ndarray
            Particle diameter(s) in nm.
        lam : ndarray
            Wabelength(s) in nm.
        prop : object
            Material properties.
        X : ndarray
            Annealed fraction (optional)
        """
        k = 2 * np.pi / lam  # wavenumber
        q = 2 * k * np.sin(the / 2)  # scattering vector
        x = np.pi * d / lam  # size parameter

        if X is None:  # if no annealing, set to ones of same size as T
            X = np.array([1])

        # Evaluate cross section.
        if 'rdg-fa' in model or 'rayleigh' in model:  # RDG-FA (volumetric), same as Rayleigh save for Npp
            C = 1.0
            S = (q * Rg < 1) * (1 - (q * Rg) ** 2 / 3) + \
                  (q * Rg >= 1) * C * (q * Rg) ** (-prop.Df)  # structure factor

            if hasattr(prop, 'm'):
                Fm = np.abs((prop.m**2 - 1) / (prop.m**2 + 2)) ** 2
            else:
                Fm = prop.Fm(lam, d, X)
                
            Npp = prop.kf * (Rg / (d / 2)) ** prop.Df  # number of primaries
            dCsca = k ** 4 * (d / 2) ** 6 * Fm * S * Npp ** 2

        elif 'mie' in model:  # Mie absorption
            import miepython as mie  # import mie library
            # _, qsca, _, _ = mie.efficiencies(prop['m'], d, lam)
            # Csca = qsca * (np.pi * d ** 2 / 4)  # multiple eff. by cross section
            
            S1, S2 = mie.mie_S1_S2(prop.m, x, np.cos(the))

            # Differential scattering cross section [m^2/sr]
            dCsca = (np.abs(S1)**2 + np.abs(S2)**2) / (2 * k**2)

        else:
            dCsca = None  # unsupported model

        return dCsca
    