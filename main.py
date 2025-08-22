import numpy as np
import matplotlib.pyplot as plt

from Prop import Prop
from HTModel import HTModel
from SModel import SModel
import tools

prop = Prop(['yaml/air.yaml', 'yaml/C_liu25.yaml'])
prop.Tg = 300
prop.Pg = 101325
prop.Ti = 3000
prop.tlp = 10

t = np.linspace(0, 2500, 500)

htm = HTModel(prop, ['dp0'], t=t)


# Evaluate temporal decays.
# htm.dTdt(0, 3000, 0.1, 1)

d = np.arange(15, 91, 15)
To, dpo, mpo, _ = htm.de_solve(prop, d)


tools.plot_sweep(To.T, d, t)
plt.show()

# Spectroscopic model.
prop.dp0 = 30
sm = SModel(prop)

sc = 1e13
J0 = sc * sm.forward(To)
J, sig = tools.add_noise(J0, scale=1, gam=1e-20)
J[J < 5] = np.nan

T1 = sm.inverse(J)
# T1 = sm.pyrometry_ratio(J[:,:,0], J[:,:,1])
# T2 = sm.spectral_fit(J)

tools.plot_sweep(T1.T, d, t)
plt.show()
