import numpy as np

# Import specific functions to help YAML shorthand.
from numpy import exp, polyval

import yaml

import copy

import ast
from functools import partial


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


# Define two classes that handle bound lambdas and their display.
class LambdaWrapper:
    def __init__(self, func_str, instance):
        self.func_str = func_str
        func = eval(func_str)            # lambda self, T: ...
        self._callable = partial(func, instance)  # bind instance once

    def __call__(self, *args, **kwargs):
        return self._callable(*args, **kwargs)

    def __repr__(self):
        return self.func_str

    def add_args(self, expected_args, default_value="None"):
        """
        Add extra arguments to the lambda until it has all of the arguments from expected_args.
        The new arguments will have default values.
        """

        # Get current arguments of the lambda funciton.
        node = ast.parse(self.func_str, mode='eval')
        if isinstance(node.body, ast.Lambda):
            args = [arg.arg for arg in node.body.args.args]
        else:
            raise ValueError("Validated property is not a lambda expression.")

        # Return if already enough arguments.
        if len(expected_args) == len(args):
            return  # already has enough arguments

        # Add arguments until the correct number are present.
        new_args = [f"{expected_args[ii-len(args)+1]}={default_value}" for ii in range(len(args), len(expected_args)+1)]
        # Insert before the colon in the original lambda
        lambda_body = self.func_str.split(":", 1)[1]  # body after colon
        new_lambda = f"lambda {', '.join(args + new_args)}: {lambda_body}"
        self.func_str = new_lambda
        self.func = eval(new_lambda)


class Prop:
    def __init__(self, fns=[]):
        prop = load_yaml(fns)  # load yaml file, returns dictionary

        # Import constants.
        self.H = H    # Planck"s constant [m^2.kg/s]
        self.C = C    # Speed of light in a vacuum [m/s]
        self.KB = KB  # Boltzmann constant [m^2.kg/s^2/K]
        self.R = R    # Universal gas constant [J/mol/K]
        self.PI = np.pi

        for key in prop.keys():
            self.add(key, prop[key])

        self.validate()

    def copy(self):
        return copy.copy(self)

    def add(self, key, value):
        
        # If list, convert entries to floats.
        def try_float(x):
            try:
                return float(x)
            except (ValueError, TypeError):
                return x  # leave strings that are not numbers as-is
            
        if isinstance(value, list):
            value = [try_float(x) for x in value]
            setattr(self, key, np.array(value))  # convert list to array for computations

        elif isinstance(value, str) and value.strip().startswith("lambda"):
            # Add as LambdaWrapper descriptor on the class and bind method.
            setattr(self, key, LambdaWrapper(value, self))

        else:
            setattr(self, key, try_float(value))  # directly assign value
    
    def validate(self):
        """
        Validates the function inputs and modifies them if necessary.
        """
        patterns = load_yaml('yaml\\validator.yaml')

        # Loop through properties in file and validate.
        for key in patterns.keys():
            self.__getattribute__(key).add_args(patterns[key])
                

    # Override __repr__ so Jupyter uses it
    def __repr__(self):
        lines = []
        lines.append('\r' + '\033[32m' + 'Prop:' + '\033[0m')

        # First, show instance attributes
        for attr, val in self.__dict__.items():
            if isinstance(val, LambdaWrapper):
                val_repr = repr(val)
            elif callable(val):
                val_repr = "<bound method>"
            else:
                val_repr = repr(val)
            lines.append(f"  \033[34m{attr}\033[0m → {val_repr}")

        # Then show LambdaWrapper descriptors on the class
        for attr, val in self.__class__.__dict__.items():
            if isinstance(val, LambdaWrapper):
                # Access the LambdaWrapper itself, not the bound lambda
                lines.append(f"  \033[34m{attr}\033[0m → {val}")

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
    
    def promote(self, key, idx):
        """
        Move an indexed value from a list of dictionaries in prop to inherent attributes of prop.
        """

        # Get dictionary specified by index and key arguments.
        retrieved_dict = self.__getattribute__(key)[idx]

        # Copy prop and delete the corresponding key.
        prop = self.copy()
        delattr(prop, key)
        
        for key, value in retrieved_dict.items():  # loop through new keys
            prop.add(key, value)  # add each key
            
        return prop

    
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