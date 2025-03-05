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
prop.Ti = prop.Tg
prop.dp0 = 30

# Absorption info.
prop.tlp = 30
prop.tlm = 0
prop.F0 = 0.1
prop.l_laser = 1064

t = np.linspace(-100, 300, 500)

htm = HTModel(prop, ['dp0'], t=t, abs='include')


# Evaluate temporal decays.
# htm.dTdt(0, 3000, 0.1, 1)

F0_vec = np.linspace(0, 1.0, 41)
To = np.zeros((len(F0_vec), len(t)))
mpo = np.zeros((len(F0_vec), len(t)))
for ii in range(len(F0_vec)):
    prop.F0 = F0_vec[ii]
    To[ii,:], _, mpo[ii,:], _ = htm.de_solve(prop, np.array([30.]))


utils.plot_sweep(To.T, F0_vec, t)
plt.legend([],[], frameon=False)
plt.show()

plt.plot(F0_vec, np.max(To, axis=1), '-')
plt.show()


# Spectroscopic model.
prop.dp0 = 30
sm = SModel(prop, lam=np.array([442]))

Jo = sm.forward(To)

plt.plot(F0_vec, np.max(Jo[:,:,0], axis=0), '-')
plt.plot(F0_vec, np.max(mpo.T * Jo[:,:,0], axis=0), '-')
plt.ylabel('Peak incandescence')
plt.show()

plt.plot(F0_vec, Jo[:,:,0][125,:], '-')
plt.plot(F0_vec, (mpo[:,:].T * Jo[:,:,0])[125,:], '-')
plt.ylabel('Incandescence at peak laser power')
plt.show()
