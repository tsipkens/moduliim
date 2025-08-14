import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from Prop import Prop
from HTModel import HTModel
from SModel import SModel
import tools

prop = Prop(['yaml/air.yaml', 'yaml/C_sipkens.yaml'])
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
To = np.zeros((len(t), len(F0_vec)))
mpo = np.zeros((len(t), len(F0_vec)))
print('Running fluence sweep:')
for ii in tqdm(range(len(F0_vec))):
    prop.F0 = F0_vec[ii]
    To[:,ii], _, mpo[:,ii], _ = htm.de_solve(prop, np.array([30.]))


tools.plot_sweep(To, F0_vec, t)
plt.legend([],[], frameon=False)
plt.show()

plt.plot(F0_vec, np.max(To, axis=0), '-')
plt.show()


# Spectroscopic model.
prop.dp0 = 30
sm = SModel(prop, lam=np.array([442, 716]))

Jo = sm.forward(To)

plt.plot(F0_vec, np.max(Jo[:,:,0], axis=0), '-')
plt.plot(F0_vec, np.max(mpo * Jo[:,:,0], axis=0), '-')
plt.ylabel('Peak incandescence')
plt.show()

plt.plot(F0_vec, Jo[:,:,0][125,:], '-')
plt.plot(F0_vec, (mpo[:,:] * Jo[:,:,0])[125,:], '-')
plt.ylabel('Incandescence at peak laser power')
plt.show()




# -- Simplified fluence approach --
import numpy as np
import matplotlib.pyplot as plt
import fluence

import importlib
importlib.reload(fluence)


prop2 = fluence.get_prop(Prop(['yaml/C_simple.yaml']))

Tref, Fref = fluence.get_ref(prop2)

T_fun, T_high, T_low = fluence.gen_peak_fun(prop2, -10)


# --- Generate plot of peak temperature curve ---
F0_vec = np.linspace(np.finfo(float).eps, 3 * Fref, 450)  # fluence to evaluate funs
T_vec = T_fun(F0_vec)  # predicted peak temperature curve

plt.figure(1)
plt.plot(F0_vec, T_vec, 'k', linewidth=1.2, label="Overall fluence curve")
plt.plot(F0_vec, T_low(F0_vec), '-', label="Low-fluence regime")
plt.plot(F0_vec, T_high(F0_vec), '-', label="High-fluence regime")
plt.xlim([0, 3 * Fref])
plt.ylim([prop.Tg, 1.2 * Tref])
plt.legend()
plt.xlabel("Fluence (J/cm²)")
plt.ylabel("Peak Temperature (K)")

# --- Generate an approximate incandescence curve ---
F0_vec2 = np.linspace(np.finfo(float).eps, 6 * Fref, 550)  # extended fluence range
J, DM, J0 = fluence.estimatej(F0_vec2, prop2, 500e-9)

plt.figure(3)
plt.plot(F0_vec2, J, 'k', linewidth=1.2, label="J")
plt.plot(F0_vec2, J0, label="J0")
plt.plot(F0_vec2, DM * np.max(J), label="DM × max(J)")
plt.plot(F0_vec2, np.cbrt(DM) * np.max(J), label="DM^(1/3) × max(J)")
plt.ylim([0, 1.5 * np.max(J)])
plt.legend()
plt.xlabel("Fluence (J/cm²)")
plt.ylabel("Incandescence (a.u.)")

plt.show()