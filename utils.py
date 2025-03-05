import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def update_prop(obj, x=None):
    """
    Update the material property structure using an x.

    Parameters:
        obj: Object with `prop` and `x` attributes.
        x: List or array of new values to update `prop`.

    Returns:
        obj: Updated object.
        prop: Updated property structure.
    """
    prop = obj.prop

    if x is not None:  # update x values
        if len(x) < len(obj.x):
            raise ValueError("Error: QoIs parameter size mismatch.")
        elif len(x) > len(obj.x):
            print("Warning: QoIs parameter size mismatch.")
        
        for ii in range(len(obj.x)):
            setattr(prop, obj.x[ii], x[ii])  # Update prop attributes

    obj.prop = prop
    return obj, prop


def plot_sweep(data, d, t):
    df = pd.DataFrame(data)  # Transpose to align columns with series
    df.columns = [f"{d[ii]}" for ii in range(len(d))]  # Name columns as "Series 1", "Series 2", ...
    df['t'] = t  # Add an index column
    df = pd.melt(df, id_vars='t', var_name='d', value_name='T')

    sns.lineplot(df, x='t', y='T', hue='d', palette='rocket')


def add_noise(s, scale=1, gam=0):
    s = np.random.poisson(s * scale).astype(np.float32)
    s = s + np.random.normal(0, gam * np.ones_like(s))
    s = s / scale
    sig = np.sqrt(np.maximum(gam ** 2 + s * scale, 0))
    return s, sig


def textdone():
    print('\r' +'\033[32m' + '^ DONE!' + '\033[0m' + '\n')

