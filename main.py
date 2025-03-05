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
prop.tlp = 10

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
To, dpo, mpo, _ = htm.de_solve(prop, d)


utils.plot_sweep(To.T, d, t)
plt.show()


# Spectroscopic model.
prop.dp0 = 30
sm = SModel(prop)

J0 = 1e-1 * sm.forward(To)
J, sig = utils.add_noise(J0, scale=0.1, gam=1e-15)
J[J < 1e2] = np.nan

T1 = sm.inverse(J)
# T1 = sm.pyrometry_ratio(J[:,:,0], J[:,:,1])
# T2 = sm.spectral_fit(J)

utils.plot_sweep(T1, d, t)
plt.show()
