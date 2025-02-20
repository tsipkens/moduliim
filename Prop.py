import numpy as np
import yaml

# Define constants.
H = 6.62606957e-34  # Planck"s constant [m^2.kg/s]
C = 2.99792458e8    # Speed of light in a vacuum [m/s]
KB =  1.3806488e-23 # Boltzmann constant [m^2.kg/s^2/K]
R = 8.3144621       # Universal gas constant [J/mol/K]
PI = np.pi

def load_yaml(fns):
    """
    Load a single or list of YAML files.
    """
    if not type(fns) == list:
        fns = [fns]

    prop = {}  # initialize empty distionary
    for fn in fns:
        with open(fn) as stream:
            try:
                prop.update(yaml.safe_load(stream))  # append new properties
            except yaml.YAMLError as exc:
                print(exc)

    return prop

# Dynamically load configuration file.
def Prop(prop):

    if type(prop) == str or type(prop) == list:
        prop = load_yaml(prop)

    class Prop0:
        def __init__(self):
            pass

        def iif(self, cond, a, b):
            """
            If function for writing inline conditional statements.
            AUTHOR: Timothy Sipkens, 2020-12-27
            """
            a = np.asarray(a)
            b = np.asarray(b)
            cond = np.asarray(cond)
            out = b
            out[cond] = a[cond]
            return out
        
        def eq_claus_clap(self, T, dp, hv):
            """
            Evaluate the Clausius-Clapeyron equation.
            """
            return np.exp(self.C - self.hvb * 1e6 / self.Rs / T)

        def eq_kelvin(self, T, dp, hv):
            """
            Evaluate the Kelvin equation.
            """
            pv0 = self.eq_claus_clap(self, T, dp, hv)  # Clausius-Clapeyron equation
            return pv0 * np.exp((4 * self.gamma(dp, T, prop)) / \
                (dp * self.rho(T) * self.Rs * T))  # Evaluate the Kelvin Eqn.
        
        def eq_antione(self, T, dp, hv):
            """
            Evaluate the Antione equation.
            """
            return np.exp(self.C - self.C1 / (T + self.C2))

        def eq_mu(self, T):
            """
            Returns the dynamic viscosity of a gas in units of Ns/m^2.  
            AUTHOR: Kyle Daun, 2020-12-17
            MODIFIED: Timothy Sipkens
            """
            mu = (T<1000) * (np.exp(self.coeffs[1,1] *np.log(T) + self.coeffs[1,2] / T + \
                    self.coeffs(1,3) / T ** 2 + self.coeffs[1,4])) + \
                (T>=1000) * (np.exp(self.coeffs[2,1] * np.log(T) + self.coeffs[2,2] / T + \
                    self.coeffs[2,3] / T ** 2 + self.coeffs[2,4]))
            return mu * 1e-7

        
    def add(Prop0, prop):
        """
        A function to dynamically add attributes to the class.
        """
        for key in prop.keys():
            try:
                setattr(Prop0, key, eval(prop[key]))
            except:
                setattr(Prop0, key, prop[key])
        return Prop0
    
    Prop0 = add(Prop0, prop)  # add properties to class

    return Prop0()  # create instance and return
