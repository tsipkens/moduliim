import numpy as np

import scipy.optimize as op

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from Prop import Prop
from HTModel import HTModel
from SModel import SModel
import utils

prop = Prop(['yaml/air.yaml', 'yaml/C_liu.yaml'])
prop.Tg = 300
prop.Pg = 101325
prop.Ti = 4100

t = np.linspace(0, 1000, 500)

htm = HTModel(prop, ['dp0', 'alpha'], t=t) # , abs='include')


# Evaluate temporal decays.
# htm.dTdt(0, 3000, 0.1, 1)

d = np.array([60])
To, _, _, _ = htm.de_solve(prop, d)


# Spectroscopic model.
prop.dp0 = 30
sm = SModel(prop)

J0 = 1e-1 * sm.forward(To)
J, sig = utils.add_noise(J0, scale=0.0001, gam=1e-15)
J[J < 1e2] = np.nan

T1 = sm.inverse(J)
# T1 = sm.pyrometry_ratio(J[:,:,0], J[:,:,1])
# T2 = sm.spectral_fit(J)



def fun(x):
    # resid = np.squeeze(T1.T - htm.de_solve(prop, np.array([x[0]]))[0])
    resid = np.squeeze(T1.T - htm.evaluate(x)[0])
    return resid[~np.isnan(resid)]
x0 = 1.2 * np.array([60, prop.alpha])
res = op.least_squares(fun, x0=x0)
x1 = res['x']

Jac = res['jac']
Gam = np.linalg.pinv(Jac.T @ Jac)
sx = np.sqrt(np.diag(Gam))
R = Gam / np.outer(sx, sx)


plt.plot(t, To.T, 'k--')
plt.scatter(t, T1, s=1, c='r')
plt.plot(t, np.squeeze(htm.evaluate(x1)[0]), 'g-')
plt.yscale('log')
plt.ylim([1000, 4500])
plt.show()
