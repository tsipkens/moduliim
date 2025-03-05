import numpy as np
import yaml

from types import MethodType

from pprint import pprint

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

class Prop:
    def __init__(self, prop):
        prop = load_yaml(prop)  # load yaml file, returns dictionary
        for key in prop.keys():
            self.add(key, prop[key])

    def add(self, key, value):
        try:
            fun = eval(value)
            if callable(fun):  # then add as a bound method
                setattr(self, key, MethodType(fun, self))
                setattr(self, key + "_fun", value)  # save text version in prop
            else:
                setattr(self, key, eval(value))  # add as an attribute directly
        except:
            if callable(value):
                setattr(self, key, MethodType(value, self))
            else:
                setattr(self, key, value)  # add value directly

    def show(self):
        v = vars(self).copy()
        keys = vars(self).keys()
        
        # Flag duplicates.
        todelete = []
        for key in keys:
            if key + '_fun' in keys:
                todelete.append(key)

        # Now delete duplicates. 
        for key in todelete:
            v[key] = v[key + '_fun']  # move text over
            del v[key + '_fun']  # delete text

        pprint(v)


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
        pv0 = self.eq_claus_clap(T, dp, hv)  # Clausius-Clapeyron equation
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
