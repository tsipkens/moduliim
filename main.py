import numpy as np

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
prop.Ti = 3000

t = np.linspace(0, 2500, 500)

htm = HTModel(prop, ['dp0'], t=t) # , abs='include')


# Evaluate heat transfer model components.
T = np.expand_dims(np.linspace(500, 3500, 100), 1).T
d = np.expand_dims(np.linspace(10, 20, 3), 1)

qc = htm.q_cond(prop, T, np.asarray([[12], [5]]))
plt.semilogy(np.squeeze(T), qc.T)

qe, _, _, _ = htm.q_evap(prop, T, np.asarray([[12], [5]]))
plt.semilogy(np.squeeze(T), qe.T)

plt.show()


# Evaluate temporal decays.
# htm.dTdt(0, 3000, 0.1, 1)

d = np.arange(15, 91, 15)
out = htm.de_solve(prop, d)


utils.plot_sweep(out[0].T, d, t)


# Spectroscopic model.
prop.dp0 = 30
sm = SModel(prop)

J0 = 1e-1 * sm.blackbody(out[0].T, sm.lam) / np.expand_dims(sm.lam, [0,1])
J = 1e1 * utils.add_noise(J0, scale=1.)

T1 = sm.pyrometry_ratio(J[:,:,0], J[:,:,1])

T2 = sm.spectral_fit(J)

utils.plot_sweep(T2, d, t)
