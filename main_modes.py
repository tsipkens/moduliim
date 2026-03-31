import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from Prop import Prop
from HTModel import HTModel
from SModel import SModel
import tools

prop = Prop(['yaml/air.yaml', 'yaml/C_liu25.yaml'])
prop.Tg = 300
prop.Pg = 101325
prop.Ti = 300

# Absorption info.
prop.tlp = 30
prop.tlm = 0
prop.F0 = 0.3  # laser fluence
prop.l_laser = 1064

t = np.linspace(-100, 500, 500)

htm = HTModel(prop, ['dp0'], t=t, abs='gaussian')


# Evaluate heat transfer model components.
# T = np.expand_dims(np.linspace(500, 3500, 100), 1).T
# d = np.expand_dims(np.linspace(10, 20, 3), 1)

# qc = htm.q_cond(prop, T, np.asarray([[12], [5]]))
# plt.semilogy(np.squeeze(T), qc.T)

# qe, _, _, _ = htm.q_evap(prop, T, np.asarray([[12], [5]]))
# plt.semilogy(np.squeeze(T), qe.T)

# plt.show()


# Assign diameter and evaluate HT model.
d = np.array([30])
To, dpo, mpo, _ = htm.de_solve(prop, d)



# --- Generate main plot ---
plt.figure(figsize=(10, 5))

# Plot for temperature.
plt.subplot(1, 2, 1)
plt.plot(t, To.T)
plt.xlabel('Temperature [K]')
plt.ylabel('Time [ns]')


# Plot for modes.
ax2a = plt.subplot(1, 2, 2)
normalizer = (np.pi * d  ** 3 / 6) * prop.rho(To) * prop.cp(To)

qa, _, _ = htm.q_abs(prop, t, d)
ax2a.plot(t, qa.T * 1e9)

qc, _ = htm.q_cond(prop, To, d)
ax2a.plot(t, qc.T * 1e9)

qe, _, _, _ = htm.q_evap(prop, To, d)
ax2a.plot(t, qe.T * 1e9)

ax2a.set_yscale('log')
ax2a.set_ylim([1e-3, 1e6])
ax2a.set_ylabel('q [nW]')
ax2a.set_xlabel('t [ns]')

def forward(x):
    return np.mean(normalizer) * x / 1e9  # example: natural log

def inverse(x):
    return x / 1e9 / np.mean(normalizer)

# Create the secondary axis (top x-axis)
ax2b = ax2a.secondary_yaxis('right', functions=(forward, inverse))
ax2b.set_xlabel('dT/dt [K/s]')

plt.show()


# Spectroscopic model.
prop.dp0 = 30
sm = SModel(prop)

J = sm.forward(To)



# Example data
data = {
    't': t,
    'T': To[0,:],
    'J442': J[0,:,0],
    'J716': J[0,:,1],
    'abs': qa,
    'cond': qc[0,:],
    'evap': qe[0,:]
}
df = pd.DataFrame(data)

# Write to Excel
df.to_excel('data_out\\htmodes.xlsx', index=False)

