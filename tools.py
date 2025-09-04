import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm  # matplotlib
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



def plot_sweep(x, y, c=None, cmap='rocket', usage=0.9, **kwargs):
    """
    Sweep through a colormap when generating a line plot.
    usage : number between 0 and 1 that determines the amount of the colormap to use
    """

    # Get overlapping dimension (handles if data is transposed).
    dim = np.where(np.asarray(np.shape(y)) != len(x))[0][0]
    if dim == 0:
        y = y.T

    n = np.shape(y)[1]  # number of series
    cm = get_cmap(cmap)  # get colormap

    # Get colors for lines from colormap.
    if c is None:
        c = np.linspace(0, 1, n)
    cm = cm((c - np.min(c)) / (np.max(c) - np.min(c)) * usage)

    # Finally, plot using loop.
    for ii in range(n):
        plt.plot(x, y[:,ii], color=cm[ii], label=str(c[ii]), **kwargs)


def plot(x, y, c=None, cmap='rocket', usage=1.0, **kwargs):
    """
    Sweep through a colormap when generating a plot.
    usage : number between 0 and 1 that determines the amount of the colormap to use
    """

    # Get overlapping dimension (handles if data is transposed).
    dim = np.where(np.asarray(np.shape(y)) != len(x))[0][0]
    if dim == 0:
        y = y.T

    n = np.shape(y)[1]  # number of series
    cm = get_cmap(cmap)  # get colormap

    # Get colors for lines from colormap.
    if c is None:
        c = np.linspace(0, 1, n)
    cm = cm((c - np.min(c)) / (np.max(c) - np.min(c)) * usage)

    # Finally, plot using loop.
    for ii in range(n):
        plt.plot(x, y[:,ii], color=cm[ii], label=str(c[ii]), **kwargs)


def add_noise(s, scale=1, gam=0):
    s = s.astype(np.float32)  # convert to float32

    # Add Poisson noise.
    flag_large = s * scale > 1e3  # flag when Gaussian is appropriate  (avoid lam too large issues)
    s[~flag_large] = np.random.poisson(s[~flag_large] * scale)  # for small values use Poisson noise
    s[flag_large] = np.random.normal(s[flag_large] * scale, np.sqrt(s[flag_large] * scale))  # for large values, approx. as Gaussian
    
    s = s + np.random.normal(0, gam * np.ones_like(s))
    s = s / scale
    sig = np.sqrt(np.maximum(gam ** 2 + s * scale, 0))
    return s, sig


def textdone():
    print('\r' +'\033[32m' + '^ DONE!' + '\033[0m' + '\n')


def get_cmap(spec):
    try:
        palette = sns.color_palette(spec, as_cmap=True)  # for seaborn

    except:
        try:
            import cmasher as cmr  # cmasher (only load if get this fas)
            palette = cm.get_cmap(spec)  # for cmasher

        except:
            try:
                palette = cm.get_cmap('cmr.' + spec)  # for standard matplotlib

            except:
                print('Colormap not found!')
                palette = None

    return palette