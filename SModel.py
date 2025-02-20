import numpy as np

from tqdm import tqdm

from scipy.sparse import spdiags
from scipy.optimize import minimize

# Define constants.
H = 6.62606957e-34  # Planck"s constant [m^2.kg/s]
C = 2.99792458e8    # Speed of light in a vacuum [m/s]
KB =  1.3806488e-23 # Boltzmann constant [m^2.kg/s^2/K]
R = 8.3144621       # Universal gas constant [J/mol/K]

class SModel:
    """
    Class for the spectroscopic model and pyrometry calculations.

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
        self.prop = prop  # Material properties
        self.lam = np.array(lam)  # Wavelengths
        self.t = np.array(t)  # Time points
        self.T = None  # Temperature, function handle
        self.J = None  # Incandescence
        self.htmodel = None  # Embedded heat transfer model

        self.x = x  # Variable names, quantities of interest (QoIs)

        self.data_sc = None  # Used to scale Planck's law for stability

        # Options for multicolor solver and pyrometry
        self.opts = {
            "multicolor_solver": "default",  # Indicates which multicolor solver to use
            "pyrometry": "ratio"  # Indicates how to handle pyrometry
        }
        self.opts.update(kwargs)  # Update with additional options if provided

        # Scale Planck's law for stability (closer to 1).
        # Uses dp = 30 nm for data scaling.
        T_sc = np.array([[3000]])  # Temperature for scaling/stability [K]
        self.data_sc = self.blackbody(T_sc, 1064) / (1064e-9)

        self._print_properties()

    def _print_properties(self):
        print('SModel > opts:')
        print(f"  Pyrometry: {self.opts['pyrometry']}")
        print(f"  Multicolor solver: {self.opts['multicolor_solver']}")


    def inverse(self, J):
        """
        Use default method to evaluate inverse model, converting incandescence to temperature. 
        """
        if self.opts['pyrometry'] == 'ratio':
            return self.pyrometry_ratio(J[:,:,0], J[:,:,1], Emr=None)  # uses Emr from prop
        else:
            return self.spectral_fit(J)
        
    def forward(self, T):
        """
        Use default method to evaluate forward model, converting temperature to incandescence. 
        """
        return self.blackbody(T.T, self.lam) / np.expand_dims(self.lam, [0,1])


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
    

    def pyrometry_ratio(self, J1, J2, Emr=None, idx=None):
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
        s_T : ndarray
            Standard deviation of temperature.
        out : dict
            Additional outputs, including `s_C` (std of scaling constant) and `r_TC` (correlation).
        """
        l = self.lam  # Local copy of wavelengths

        # Check the number of wavelengths
        if len(l) > 2:
            if idx is None:
                raise ValueError(
                    "More than two wavelengths in SModel. Provide indices (idx) for pyrometry."
                )
            if len(idx) != 2:
                raise ValueError("Invalid indices: idx must have exactly two values.")
            l = l[idx]

        # Handle Emr input
        if Emr is None:
            Emr = self.prop.Emr(l[0], l[1], self.prop.dp0)  # Ratio of Em at two wavelengths

        # Ratio of incandescence
        # Pre-allocation and only evaluate select (avoids error messages)
        Jr = np.empty_like(J1)
        Jr[:] = np.nan
        isEval = np.logical_and(J1 > 0, J2 > 0)
        Jr[isEval] = J1[isEval] / J2[isEval]

        # Basic ratio calculation
        PHI = 0.0143877696  # Planck's constant factor
        To = (PHI * (1 / (l[1] * 1e-9) - 1 / (l[0] * 1e-9))) / np.log(
            Jr * ((l[0] / l[1])**6) / Emr
        )
        To = np.real(To)  # Avoid imaginary values

        return To
    

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
        s = np.std(J, axis=1) / np.sqrt(J.shape[1])  # Standard error
        prop = self.prop

        # Ensure `C_J` exists in `prop`
        if not hasattr(prop, 'C_J') or prop.C_J is None:
            prop.C_J = 1

        lam = self.lam
        s_T = np.zeros((ntime, nshots))
        s_C = np.zeros((ntime, nshots))

        T0 = self.pyrometry_ratio(J[:,:,0], J[:,:,1], Emr=None)

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
