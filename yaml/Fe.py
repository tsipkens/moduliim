import numpy as np
from Prop import Prop

def Fe(prop=None, **kwargs):

    """
    Thermophysical and optical properties of iron. 
    Can mix-and-match various properties.
    """

    # -- Parse inputs ------------------------------------------------------
    if prop is None:
        prop = Prop()

    opts = {}
    opts['rho'] = kwargs.get('rho', 'hixson').lower()
    opts['cp'] = kwargs.get('cp', 'mixed').lower()
    opts['hv'] = kwargs.get('hv', 'meyra').lower()  # to be changed to Watson
    opts['pv'] = kwargs.get('pv', 'kelvin-cc').lower()
    opts['Em'] = kwargs.get('Em', 'emr1.1').lower()

    # -- Sensible energy properties ----------------------------------------
    prop.phi = f"{prop.H}*{prop.C}/{prop.KB}"

    prop.M = 0.055847
    prop.Tm = 1811  # (Desai, 1986)

    # Density rho
    if opts['rho'] in ['hixson']:
        prop.Arho = 8171
        prop.Brho = -0.64985
        prop.add('rho', "lambda self, T: self.Brho*T + self.Arho")
    elif opts['rho'] == 'basinski':
        prop.rho_data = load_mat('rho_Fe_Basinski.mat')['rho_data']
        prop.rho_gi = gridded_interpolant(prop.rho_data[:,0], prop.rho_data[:,1])
        prop.add('rho', "lambda self, T: self.rho_gi(T)")
    elif opts['rho'] == 'mills':
        func_str = (
            "lambda self, T: self.iif((T>=0)&(T<293), 7871.6*(T/T), "
            "self.iif((T>=293)&(T<1184), 7874/(1+3*14.5e-6*(T-273-20)), "
            "self.iif((T>=1184)&(T<1667), 7650-0.51*(T-273-911), "
            "self.iif((T>=1667)&(T<1811), 7355-0.42*(T-273-1394), 7030-0.86*(T-273-1538))))))"
        )
        prop.add('rho', func_str)
    elif opts['rho'] == 'noslope':
        prop.Arho = 6350
        prop.Brho = 0
        prop.add('rho', "lambda self, T: self.Brho*T + self.Arho")
    elif opts['rho'] == 'constant':
        prop.Arho = 6350
        prop.add('rho', "lambda self, T: self.Arho")

    # Heat capacity cp
    if opts['cp'] in ['default', 'mixed']:
        prop.Ccp = 1
        prop.Dcp = 1
        func_str = (
            "lambda self, T: self.Ccp * (self.iif(T>=self.Tm, 46.632, "
            "self.iif(T>=1667, -12.38+3.161e-2*T, 17.64+1.232e-2*T))) / self.M"
        )
        prop.add('cp', func_str)
    elif opts['cp'] == 'desai':
        prop.Ccp = 1
        prop.Dcp = 1
        func_str = (
            "lambda self, T: (self.iif(T>=self.Tm, self.Ccp*46.632, "
            "self.iif(T>=1667, self.Ccp*40.368+self.Dcp*3.2194e-2*(T-1667), "
            "self.Ccp*33.803+self.Dcp*9.1605e-3*(T-1181))))/self.M"
        )
        prop.add('cp', func_str)
    elif opts['cp'] == 'noslope':
        prop.Ccp = 1
        prop.Dcp = 0
        func_str = (
            "lambda self, T: self.Ccp * (self.iif(T>=self.Tm, 46.632, "
            "self.iif(T>=1667, -12.38+3.161e-2*T, 17.64+1.232e-2*T))) / self.M"
        )
        prop.add('cp', func_str)
    elif opts['cp'] == 'constant':
        prop.Ccp = 1
        prop.add('cp', f"lambda self, T: {prop.Ccp*46.632/prop.M}")

    # Conduction properties
    prop.alpha = 0.23
    prop.Tg = 298
    prop.Pg = 101325
    prop.add('ct', "lambda self: (8*self.KB*self.Tg/3.141592653589793/self.mg)**0.5")

    # Evaporation properties
    prop.mv = prop.M * 1.660538782e-24
    prop.Rs = prop.R / prop.M
    prop.Tb = 3134
    prop.hvb = 340e3 / prop.M / 1e6

    # Latent heat hv
    prop.Tcr = 9340
    prop.n = 0.38
    if opts['hv'] in ['default', 'watson']:
        prop.add('hv', "lambda self, T: self.eq_watson(T)")
    elif opts['hv'] == 'roman':
        prop.beta = 0.371
        func_str = (
            "lambda self, T: (self.hvb*1e6)*exp((self.n-self.beta)*((T-self.Tb)/(self.Tcr-self.Tb)))*"
            "(((self.Tcr-T)/(self.Tcr-self.Tb))**self.n)"
        )
        prop.add('hv', func_str)
    elif opts['hv'] == 'meyra':
        func_str = (
            "lambda self, T: (self.hvb*1e6) * (((self.Tcr-T)/(self.Tcr-self.Tb))**"
            "((self.n**2)*((self.Tcr-T)/(self.Tcr-self.Tb))+self.n))"
        )
        prop.add('hv', func_str)
    elif opts['hv'] == 'constant':
        prop.add('hv', f"lambda self, T: {prop.hvb*1e6}")

    # Surface tension & vapor pressure
    prop.gamma0 = 1.865
    prop.Pref = 101325
    prop.Ccc = np.log(prop.Pref) + (prop.hvb*1e6)/prop.Rs/prop.Tb

    # Vapor pressure options
    if opts['pv'] in ['default', 'kelvin-cc']:
        prop.add('gamma', "lambda self, dp,T: self.gamma0")
        prop.add('pv', "lambda self, T,dp,hv: self.eq_kelvin(T,dp,hv)")
    elif opts['pv'] == 'tolman-cc':
        prop.delta = 0.126
        prop.add('gamma', "lambda self, dp,T: self.eq_tolman(dp,T)")
        prop.add('gammaT', "lambda self, T: self.gamma0")
        prop.add('pv', "lambda self, T,dp,hv: self.eq_kelvin(T,dp,hv)")
    elif opts['pv'] == 'cc':
        prop.add('pv', "lambda self, T,dp,hv: self.eq_claus_clap(T,dp,hv)")
    elif opts['pv'] == 'cc-alt':
        prop.C1 = prop.hvb*1e6/prop.Rs
        prop.add('pv', "lambda self, T,dp,hv: exp(self.C-self.C1/T) * exp(4*self.gamma0/(dp*self.rho(T)*self.Rs*T))")
    elif opts['pv'] == 'antoine-alt':
        prop.C1 = prop.hvb*1e6/prop.Rs
        prop.C2 = 1
        prop.add('pv', "lambda self, T,dp,hv: exp(self.C-self.C1/(T+self.C2)) * exp(4*self.gamma0/(dp*self.rho(T)*self.Rs*T))")
    elif opts['pv'] == 'rankine-kirchoff-alt':
        prop.C1 = prop.hvb*1e6/prop.Rs
        prop.C2 = 1
        prop.add('pv', "lambda self, T,dp,hv: exp(self.C-self.C1/T + self.C2*log(T)) * exp(4*self.gamma0/(dp*self.rho(T)*self.Rs*T))")
    elif opts['pv'] == 'nernst-alt':
        prop.C1 = prop.hvb*1e6/prop.Rs
        prop.C2 = prop.C3 = 1
        prop.add('pv', "lambda self, T,dp,hv: exp(self.C-self.C1/T + self.C2*log(T) + self.C3*T) * exp(4*self.gamma0/(dp*self.rho(T)*self.Rs*T))")
    elif opts['pv'] == 'crc-alt':
        prop.C1 = prop.hvb*1e6/prop.Rs
        prop.C2 = prop.C3 = prop.C4 = 1
        prop.add('pv', "lambda self, T,dp,hv: exp(self.C-self.C1/T + self.C2*log(T) + self.C3/T**3) * exp(4*self.gamma0/(dp*self.rho(T)*self.Rs*T))")
    elif opts['pv'] == 'kelvin-antoine':
        prop.add('gamma', "lambda self, dp,T: self.gamma0")
        prop.add('pv', "lambda self, T,dp,hv: self.Antoine(T,dp,hv) * exp(4*self.gamma(dp,T)/(dp*self.rho(T)*self.Rs*T))")

    # Optical properties
    if opts['Em'] in ['default', 'emr1.1']:
        prop.CEmr = 1
        prop.add('Emr', "lambda self, l1,l2,dp: self.CEmr*1.1")
        prop.add('Em', "lambda self, l,dp,X: (l-716)/(442-716)*(self.Emr(442,716,dp)-1)+1")
    elif opts['Em'] == 'drude':
        prop.omega_p = 6.78e17
        prop.tau = 1.69e-19
        prop.Em_data = load_mat('Em_Fe_Drude.mat')['Em_data']
        prop.Em_gi = gridded_interpolant(prop.Em_data[:,0], prop.Em_data[:,1])
        prop.add('Em', "lambda self, l,dp,X: self.Em_gi(l)")
        prop.CEmr = 1
        prop.add('Emr', "lambda self, l1,l2,dp: self.CEmr*self.Em(l1,dp)/self.Em(l2,dp)")
    elif opts['Em'] == 'krishnan':
        func_poly = "[3.9751e-13,-1.6904e-9,2.7217e-6,-0.0020557,0.69807]"
        prop.CEmr = 1
        prop.add('Em', f"lambda self, l,dp,X: polyval({func_poly}, l)*ones_like(dp)")
        prop.add('Emr', "lambda self, l1,l2,dp: self.CEmr*self.Em(l1,dp)/self.Em(l2,dp)")
    elif opts['Em'] == 'shvarev':
        prop.Em_data = load_mat('Em_Fe_Shvarev.mat')['Em_data']
        prop.Em_gi = gridded_interpolant(prop.Em_data[:,0], prop.Em_data[:,1])
        prop.add('Em', "lambda self, l,dp,X: self.Em_gi(l)")
        prop.CEmr = 1
        prop.add('Emr', "lambda self, l1,l2,dp: self.CEmr*self.Em(l1,dp)/self.Em(l2,dp)")
    elif opts['Em'] == 'mie-krishnan':
        Krishnan = load_mat('prop.Fe_opt_prop.mat')['Krishnan']
        n_gi = gridded_interpolant(Krishnan.l, Krishnan.n)
        k_gi = gridded_interpolant(Krishnan.l, Krishnan.k)
        prop.add('Em', "lambda self, l,dp,X: self.get_Mie_solution(n_gi,k_gi,l,dp)")
        prop.CEmr = 1
        prop.add('Emr', "lambda self, l1,l2,dp: self.CEmr*self.Em(l1,dp)/self.Em(l2,dp)")

    # Particle size
    prop.dp0 = 20
    prop.sigma = 0

    prop.opts = opts
    return prop


prop = Fe()