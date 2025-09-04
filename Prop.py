import numpy as np

# Import specific functions to help YAML shorthand.
# Not used directly but used in eval calls.
from numpy import exp, log, polyval

import types
import yaml
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
    
    # Parse prop inputs.
    for k, v in prop.items():
        prop[k] = parse_value(v)

    return prop

def parse_value(v):
        """
        Normalize a value from YAML or constants:
        - Lists of numbers -> numpy arrays
        - Strings that can be converted to floats -> float
        - Leave other strings as-is
        """
        if isinstance(v, list):
            try:
                return np.array([float(x) for x in v])
            except (ValueError, TypeError):
                return np.array(v)  # leave as array of strings if cannot convert
        elif isinstance(v, (int, float)):
            return float(v)
        elif isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return v  # leave string as-is
        else:
            return v


# Define two classes that handle bound lambdas and their display.
class LambdaWrapper:
    def __init__(self, func_str, instance):
        self.func_str = func_str
        self.instance = instance

        # Parse and bind once and cache.
        self._ast = ast.parse(func_str, mode="eval")
        func = eval(func_str)
        self._callable = partial(func, instance)

    def __call__(self, *args, **kwargs):
        return self._callable(*args, **kwargs)

    def __repr__(self):
        return self.func_str

    def add_args(self, expected_args, default_value="None"):
        """
        Add extra arguments with default values until the lambda has all
        of the arguments in expected_args. Keeps instance binding intact.

        Used to add arguments as optional during function validation.
        """

        # Parse the existing lambda
        node = self._ast
        if not isinstance(node.body, ast.Lambda):
            raise ValueError("Validated property is not a lambda expression.")

        current_args = [arg.arg for arg in node.body.args.args]

        # Drop 'self' if present (already bound)
        if current_args and current_args[0] == "self":
            current_args = current_args[1:]

        # Nothing to do if already complete
        if len(current_args) >= len(expected_args):  # minus self
            return

        # Build new arg list (exclude self)
        needed = expected_args[len(current_args):]  # skip self
        new_args = [f"{name}={default_value}" for name in needed]

        # Rebuild lambda string
        lambda_body = self.func_str.split(":", 1)[1]
        new_lambda = f"lambda self, {', '.join(current_args + new_args)}: {lambda_body}"

        # Store new string and rebind
        self.func_str = new_lambda
        func = eval(new_lambda)
        self._callable = partial(func, self.instance)



class Prop:
    __slots__ = ("_store", )  # smaller memory footprint
    
    def __init__(self, fns=[]):
        self._store = {}
        
        if type(fns) == dict:  # then parse dictionary into Prop
            self._store.update(fns)
        elif fns:
            self._store.update(load_yaml(fns))

        # Universal constants
        # self._store.update({
        #     "H": H, "C": C, "KB": KB, "R": R, "PI": PI
        # })

        self.validate()

        # Special case where multiple vapor species are specified.
        # Then also parse each species, get LambdaWrappers, and validate. 
        if hasattr(self, 'vapor_species'):
            for ii in range(len(self.vapor_species)):
                for k, v in self.vapor_species[ii].items():  # apply parse_values
                    self.vapor_species[ii][k] = parse_value(v)
                self.vapor_species[ii] = Prop(self.vapor_species[ii])  # use class to parse


    def __getattr__(self, key):
        try:
            val = self._store[key]
        except KeyError:
            raise AttributeError(f"{key} not found")
        # Wrap lambda strings lazily
        if isinstance(val, str) and val.strip().startswith("lambda"):
            lw = LambdaWrapper(val, self)
            self._store[key] = lw
            return lw
        return val

    def __setattr__(self, key, value):
        if key == "_store":
            object.__setattr__(self, key, value)
        else:
            self._store[key] = value

    def copy(self):
        """
        Return a shallow copy of this Prop instance.
        LambdaWrapper objects are re-bound to the new instance
        to prevent circular references.
        """
        new = Prop()
        for k, v in self._store.items():
            if isinstance(v, LambdaWrapper):
                new._store[k] = LambdaWrapper(v.func_str, new)  # add functions
            else:
                new._store[k] = v  # add other values

        return new

    def to_dict(self):
        """Convert instance of Prop to a simple dictionary."""
        return self._store.copy()
    
    def view(self):
        """
        Return a lightweight SimpleNamespace where:
        - Constants are converted to numbers or numpy arrays.
        - Lambda strings are converted to normal Python functions.
        - Nested lambdas referencing other lambdas work via the namespace.
        """
        ns = types.SimpleNamespace()

        # Step 1: copy constants (convert lists to np.array)
        for k, v in self._store.items():
            if not (isinstance(v, str) and v.strip().startswith("lambda")):
                setattr(ns, k, np.array(v) if isinstance(v, list) else v)

        # Step 2: convert all lambdas to normal functions
        # Bind them so that 'self' inside the lambda refers to ns
        for k, v in self._store.items():
            if isinstance(v, str) and v.strip().startswith("lambda"):
                func = eval(compile(v, "<string>", "eval"))  # lambda self, ...
                
                # Wrap to capture 'ns' as self
                def make_func(f):
                    return lambda *args, **kwargs: f(ns, *args, **kwargs)
                
                setattr(ns, k, make_func(func))

        return ns
    
    def validate(self):
        """
        Validates the function inputs and modifies them if necessary.
        See LambdaWrapper's add_args above for more details.
        """
        patterns = load_yaml("yaml\\validator.yaml")  # list of function arguments
        for key, expected_args in patterns.items():  # loop through properties to validate
            if key in self._store:
                fn = self.__getattr__(key)
                if not type(fn) == LambdaWrapper:  # if not function, make function
                    fn = LambdaWrapper(f'lambda self: {fn}', self)
                fn.add_args(expected_args)  # add necessary arguments to match pattern
                self.__setattr__(key, fn)   # add back updated function

    # Override __repr__ so Jupyter uses it
    def __repr__(self):
        lines = []
        lines.append('\r' + '\033[32m' + 'Prop:' + '\033[0m')

        # First, show instance attributes
        for attr, val in self._store.items():
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


    # --- PHYSICAL EQUATIONS ACCESSIBLE TO PROPS ---
    def iif(self, cond, a, b):
        """Inline if function for writing inline conditional statements."""
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
            self.Rs = R / self.Mv  # specific gas constant
        if not hasattr(self, 'hvb'):
            self.hvb = hv(self.Tref) / self.Mv / 1e6
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

    def eq_mu(self, T, coeffs=None):
        """
        Returns the dynamic viscosity of a gas in units of Ns/m^2.  
        AUTHOR: Kyle Daun, 2020-12-17
        MODIFIED: Timothy Sipkens
        """
        if coeffs == None:
            coeffs = self.coeffs

        mu = (T<1000) * (np.exp(coeffs[0,0] *np.log(T) + coeffs[0,1] / T + \
                coeffs[0,2] / T ** 2 + coeffs[0,3])) + \
            (T>=1000) * (np.exp(coeffs[1,0] * np.log(T) + coeffs[1,1] / T + \
                coeffs[1,2] / T ** 2 + coeffs[1,3]))
        return mu * 1e-7