import numpy as np
import yaml

import copy

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

    prop = {}  # initialize empty dictionary
    for fn in fns:
        with open(fn) as stream:
            try:
                prop.update(yaml.safe_load(stream))  # append new properties
            except yaml.YAMLError as exc:
                print(exc)

    return prop

class Prop:
    def __init__(self, fns=[]):
        prop = load_yaml(fns)  # load yaml file, returns dictionary

        # Import constants.
        self.H = 6.62606957e-34  # Planck"s constant [m^2.kg/s]
        self.C = 2.99792458e8    # Speed of light in a vacuum [m/s]
        self.KB =  1.3806488e-23 # Boltzmann constant [m^2.kg/s^2/K]
        self.R = 8.3144621       # Universal gas constant [J/mol/K]
        self.PI = np.pi

        for key in prop.keys():
            self.add(key, prop[key])

    def copy(self):
        return copy.copy(self)

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

    # Override __repr__ so Jupyter uses it
    def __repr__(self):
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

        lines = []
        lines.append('\r' +'\033[32m' + 'Prop:' + '\033[0m')
        for key, value in v.items():
            lines.append(f"  \033[34m{key}\033[0m → {value}")
        lines.append(' ')

        return "\n".join(lines)
    
    def show(self):
        print(self.__repr__())


    def iif(self, cond, a, b):
        """
        If function for writing inline conditional statements.
        AUTHOR: Timothy Sipkens, 2020-12-27
        """
        a = np.asarray(a) * np.ones_like(cond)
        b = np.asarray(b) * np.ones_like(cond)
        cond = np.asarray(cond)
        out = b
        out[cond] = a[cond]
        return out
    
    def eq_claus_clap(self, T, dp, hv):
        """
        Evaluate the Clausius-Clapeyron equation.
        """
        if not hasattr(self, 'Tref'):
            self.Tref = self.Tb  # then boiling temperature 'Tb' was used
        if not hasattr(self, 'Rs'):
            self.Rs = self.R / self.M  # specific gas constant
        if not hasattr(self, 'hvb'):
            self.hvb = hv(self.Tref) / self.M / 1e6
        if not hasattr(self, 'Pref'):  # assume atmospheric pressure reference
            self.Pref = 101325
        if not hasattr(self, 'Ccc'):
            self.Ccc = np.log(self.Pref) + (self.hvb*1e6) / self.Rs / self.Tref

        return np.exp(self.Ccc - self.hvb * 1e6 / self.Rs / T)

    def eq_kelvin(self, T, dp, hv):
        """
        Evaluate the Kelvin equation.
        """
        pv0 = self.eq_claus_clap(T, dp, hv)  # Clausius-Clapeyron equation
        return pv0 * np.exp((4 * self.gamma(dp, T)) / \
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
        mu = (T<1000) * (np.exp(self.coeffs[0,0] *np.log(T) + self.coeffs[0,1] / T + \
                self.coeffs[0,2] / T ** 2 + self.coeffs[0,3])) + \
            (T>=1000) * (np.exp(self.coeffs[1,0] * np.log(T) + self.coeffs[1,1] / T + \
                self.coeffs[1,2] / T ** 2 + self.coeffs[1,3]))
        return mu * 1e-7