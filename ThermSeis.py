import numpy as np
from Triforce.mathPlus import logQuad

'''
Next version of class for elastic/anelastic model to replace OceanSeis.py
1. Change density calculation to TherModel model instead of SeisModel
2. add RhoType parameter in OceanSeisRitz.__init__(). 
    RhoType == 'raw':           Using previous formula in raw paper
    RhoType == 'corrected':     Using corrected version from Erratum.pdf, but this result in very low Vs
    RhoType == 'from_thermal':  Using density from thermal model. This rho is closer to 'raw' instead 
                                of 'corrected'
'''

class TherModel():
    def __init__(self,**kwargs) -> None:
        self.zdeps = kwargs.get('zdeps',None)       # in km
        self.T     = kwargs.get('T',self._calT())   # in K
        self.P     = self._calP()                   # in Pa
        self.rho   = self._calRho()                 # in kg/m^3
        pass
    def _calP(self,rho=3.4e3):
        if self.zdeps is None:
            return None
        else:
            # change 3.2 to 3.4, the difference is comparable to 4Ma vs 4.1Ma at 30km
            return rho*9.8*self.zdeps*1000   # 3.424 in https://doi.org/10.1029/2004JB002965
    def _calT(self):
        return None
    def _calRho(self,rho0=3.42e3,P0=0.6e9,T0=500+273.15,alpha=4.4e-5,kappa=6.12e-12):
        P,T = self.P,self.T
        if P is None or T is None:
            return None
        else:
            return rho0*(1-alpha*(T-T0))*(1+kappa*(P-P0))
    def _calRho_depracated(self,rho0=3.43e3,P0=6.5e9,T0=1200+273.15):
        kappa,alpha = 1e-11,3e-5 # Pa^-1,K^-1
        P = self._calP()
        T = self._calT()
        rho = rho0*(1+kappa*(P-P0)-alpha*(T-T0))
        return rho

class SeisModel():
    def __init__(self,therModel=None,**kwargs) -> None:
        if therModel is not None:
            self.fromThermal(therModel)
        else:
            self.zdeps = None
            self.vs    = None
    def fromThermal(self,therModel):
        self._therMod = therModel
        pass

class HSCM(TherModel):
    def __init__(self, age, zdeps=np.linspace(0,200,200),rho0=3.43e3,Tp=1325,kappa=1e-6) -> None:
        self.age   = age
        self.zdeps = zdeps
        self.P     = self._calP()
        self.T     = self._calT(Tp,kappa)
        self.rho   = self._calRho(rho0)
    def _calT(self,Tp=1325,kappa=1e-6):
        def calTm(T0,Tp,Da,kappa,age):
            def f(z): return erf(z*1e3/(2*np.sqrt(age*365*24*3600*1*(kappa/1e-6))))
            def g(z):
                dz = 0.001;fz = f(z);fz_d = f(z+dz);dfz = (fz_d-fz)/dz + 1e-10
                return fz/dfz-z-(Tp-T0)/Da
            z0,z1 = 0,400
            while z1-z0 > 0.01:
                z2 = (z1+z0)/2
                if g(z2) < 0:
                    z0 = z2
                else:
                    z1 = z2
            # print((Da*z1 + Tp-T0)/f(z1) + T0,(Da*z0 + Tp-T0)/f(z0) + T0)
            Tm = (Da*z1 + Tp-T0)/f(z1) + T0
            z_adiaBegin = z0
            return Tm,z_adiaBegin
        C2K = 273.15
        zdeps = self.zdeps
        age   = self.age
        ''' temperature calculated from half space cooling model, topography change ignored''' 
        from scipy.special import erf
        T0 = 0; Da=0.4; T_adiabatic = Tp + zdeps*Da
        Tm,z_adiaBegin = calTm(T0,Tp,Da,kappa,age)

        theta = erf(zdeps*1e3/(2*np.sqrt(age*365*24*3600*1*(kappa/1e-6))))# suppose kappa=1e-6
        T = (Tm-T0)*theta+T0
        try:
            # adiaBegin = np.where(np.diff(T)/np.diff(zdeps) < Da)[0][0]
            adiaBegin = np.where(zdeps > z_adiaBegin)[0][0]
            if adiaBegin == 0:
                T = T_adiabatic
            else:
                T[adiaBegin:] = T_adiabatic[adiaBegin:]
        except:
            pass
        # return Tp*np.ones(self.zdeps.shape)
        return T+C2K

class OceanSeisRitz(SeisModel):  # https://doi.org/10.1111/j.1365-246X.2004.02254.x 
    def __init__(self, therModel=None, **kwargs) -> None:
        self.X          = kwargs.get('X',0.1)
        self.ws         = kwargs.get('ws',[0.75,0.21,0.035,0,0.005]) #https://doi.org/10.1016/0012-821X(84)90076-1
        self.RhoType    = kwargs.get('RhoType','raw')
        self.elasticParas = {
            'Olivine':{'rho0':3.222e3,'rho_X':1.182e3,
                        'K0':129,'K_T':-16e-3,'K_P':4.2,'K_X':0,
                        'mu0':82,'mu_T':-14e-3,'mu_P':1.4,'mu_X':-30,
                        'alpha0':0.2010e-4,'alpha1':0.1390e-7,'alpha2':0.1627e-2,'alpha3':-0.3380},
            'Orthopyroxene':{'rho0':3.198e3,'rho_X':0.804e3,
                        'K0':111,'K_T':-12e-3,'K_P':6.0,'K_X':-10,
                        'mu0':81,'mu_T':-11e-3,'mu_P':2.0,'mu_X':-29,
                        'alpha0':0.3871e-4,'alpha1':0.0446e-7,'alpha2':0.0343e-2,'alpha3':-1.7278},
            'Clinopyroxene':{'rho0':3.280e3,'rho_X':0.377e3,
                        'K0':105,'K_T':-13e-3,'K_P':6.2,'K_X':13,
                        'mu0':67,'mu_T':-10e-3,'mu_P':1.7,'mu_X':-6,
                        'alpha0':0.3206e-4,'alpha1':0.0811e-7,'alpha2':0.1347e-2,'alpha3':-1.8167},
            'Spinel':{'rho0':3.578e3,'rho_X':0.702e3,
                        'K0':198,'K_T':-28e-3,'K_P':5.7,'K_X':12,
                        'mu0':108,'mu_T':-12e-3,'mu_P':0.8,'mu_X':-24,
                        'alpha0':0.6969e-4,'alpha1':-0.0108e-7,'alpha2':-3.0799e-2,'alpha3':5.0395},
            'Garnet':{'rho0':3.565e3,'rho_X':0.758e3,
                        'K0':173,'K_T':-21e-3,'K_P':4.9,'K_X':7,
                        'mu0':92,'mu_T':-10e-3,'mu_P':1.4,'mu_X':-7,
                        'alpha0':0.0991e-4,'alpha1':0.1165e-7,'alpha2':1.0624e-2,'alpha3':-2.5000}
        }
        super().__init__(therModel, **kwargs)
    
    def _pt2vs(self,P,T): # P in GPa, T in K
        ws,X = self.ws,self.X
        RhoType = self.RhoType
        therRho = self._therMod.rho

        def TPX2_rho_mu_K_pure(T,P,X,d):
            T0 = 273.15
            P0 = 101.325e-6
            alpha = d['alpha0'] + d['alpha1']*T + d['alpha2']*T**(-1) + d['alpha3']*T**(-2)
            if RhoType == 'raw':
                rho0X = d['rho0']*d['rho_X']/1e3
            else:
                rho0X = d['rho0'] + X * d['rho_X']
            mu  = d['mu0']  + (T-T0)*d['mu_T']  + (P-P0)*d['mu_P']  + X*d['mu_X']
            K   = d['K0']   + (T-T0)*d['K_T']   + (P-P0)*d['K_P']   + X*d['K_X']
            rho = rho0X*(1-alpha*(T-T0)+(P-P0)/K)
            if RhoType == 'from_thermal':
                rho = therRho
            return mu,K,rho

        def TPX2_rho_mu_K(T,P,X,ds,ws):
            # ws.sum() = 1
            mus,Ks,rhos = np.zeros((len(T),len(ds))),np.zeros((len(T),len(ds))),np.zeros((len(T),len(ds)))
            for i,d in enumerate(ds):
                mus[:,i],Ks[:,i],rhos[:,i] = TPX2_rho_mu_K_pure(T,P,X,d)
            rho = (ws*rhos).sum(axis=1)
            mu  = 1/2 * ((ws*mus).sum(axis=1) + 1/((ws/mus).sum(axis=1)))
            K   = 1/2 * ((ws*Ks).sum(axis=1)  + 1/((ws/Ks).sum(axis=1)))
            return mu,K,rho
        
        mu,K,rho = TPX2_rho_mu_K(T,P,X,list(self.elasticParas.values()),ws)
        # plt.figure(figsize=[5.5,8])
        # plt.plot(rho,self.zdeps)
        K,mu = K*1e9,mu*1e9
        vp,vs = np.sqrt((K+4/3*mu)/rho),np.sqrt(mu/rho)
        self._rho = rho
        self._mu  = mu
        return vs
    def fromThermal(self,therMod):
        self._therMod = therMod
        self.zdeps = therMod.zdeps
        self.vs    = self._pt2vs(therMod.P/1e9,therMod.T)

class OceanSeisRuan(SeisModel):  # https://doi.org/10.1016/j.epsl.2018.05.035
    def __init__(self,therModel=None,damp=True,YaTaJu=False,period=50) -> None:
        if therModel is not None:
            self.fromThermal(therModel,damp,YaTaJu,period)
        else:
            self.zdeps = None
            self.vs    = None
    def fromThermal(self, therModel,damp=True,YaTaJu=False,period=50):
        super().fromThermal(therModel)
        Ju = 1/(72.45-0.01094*(therModel.T-273.15)+1.75*therModel.P/1e9)*1e-9
        if YaTaJu:
            Ju = 1/(72.45-0.01094*(therModel.T-273.15)+1.987*therModel.P/1e9)*1e-9
        J1,J2,Tn = self._calQ_ruan(therModel.T,therModel.P,period=period,damp=damp,verbose=True)
        self.zdeps = therModel.zdeps
        self.vs    = 1/np.sqrt(therModel.rho*Ju*J1)
        # if np.any(Tn>1):
        #     self.vs[Tn>1] = np.clip(self.vs[Tn>1] * (1-(Tn[Tn>1]-1)/0.001 * 0.078),a_min=0,a_max=10)
        self.vs_no_anelastic = 1/np.sqrt(therModel.rho*Ju)
        self.qs    = J1/J2
        self._Tn = Tn
    @staticmethod
    def _calQ_ruan(T,P,period,damp=True,verbose=False):
        ''' calculate quality factor follow Ruan+(2018) 
        T: temperature in K
        P: pressure in Pa
        period: seismic wave period in second
        '''
        from scipy.special import erf
        
        def calTn(T,P): # solidus given pressure and temperature
            P = P/1e9
            if damp is True:
                Tm = -5.1*P**2 + 92.5*P + 1120.6 + 273.15
            elif damp is False:
                Tm = -5.1*P**2 + 132.9*P + 1120.6 + 273.15
            else:
                Tm = damp
            return T/Tm
        def calTauM(T,P): # Maxwell time for viscous relaxation
            def A_eta(Tn):
                gamma = 5
                Tn_eta = 0.94
                minuslamphi = 0
                Aeta = np.zeros(Tn.shape)
                for i in range(len(Tn)):
                    if Tn[i]<Tn_eta:
                        Aeta[i] = 1
                    elif Tn[i] < 1:
                        Aeta[i] = np.exp( -(Tn[i]-Tn_eta)/(Tn[i]-Tn[i]*Tn_eta)*np.log(gamma) )
                    else:
                        Aeta[i] = 1/gamma*np.exp(minuslamphi)
                return Aeta
            E = 4.625e5
            R = 8.314
            V = 7.913e-6
            etaR = 6.22e21
            TR = 1200+273.15
            PR = 1.5e9

            mu_U = (72.45-0.01094*(T-273.15)+1.75*P*1e-9)*1e9
            eta = etaR * np.exp(E/R*(1/T-1/TR)) * np.exp(V/R*(P/T-PR/TR)) * A_eta(calTn(T,P))
            tauM = eta/mu_U
            return tauM
        
        def A_P(Tn):
            ap = np.zeros(Tn.shape)
            for i in range(len(Tn)):
                if Tn[i] < 0.91:
                    ap[i] = 0.01
                elif Tn[i] < 0.96:
                    ap[i] = 0.01+0.4*(Tn[i]-0.91)
                elif Tn[i] < 1:
                    ap[i] = 0.03
                else:
                    ap[i] = 0.03+0
            return ap
        def sig_P(Tn):
            sigp = np.zeros(Tn.shape)
            for i in range(len(Tn)):
                if Tn[i]<0.92:
                    sigp[i] = 4
                elif Tn[i] < 1:
                    sigp[i] = 4+37.5*(Tn[i]-0.92)
                else:
                    sigp[i] = 7
            return sigp

        A_B = 0.664
        tau_np = 6e-5
        alpha = 0.38

        tau_M = calTauM(T,P)
        tau_ns = period/(2*np.pi*tau_M)

        J1b = A_B*(tau_ns**alpha)/alpha
        J1p = np.sqrt(2*np.pi)/2*A_P(calTn(T,P))*sig_P(calTn(T,P))*(1-erf(np.log(tau_np/tau_ns)/(np.sqrt(2)*sig_P(calTn(T,P)))))
        J2b = np.pi/2* A_B*(tau_ns**alpha)
        J2p = np.pi/2* (A_P(calTn(T,P))*np.exp(-((np.log(tau_np/tau_ns)/(np.sqrt(2)*sig_P(calTn(T,P))))**2)))
        J2e = tau_ns
        J1 = 1+J1b+J1p
        J2 = J2b+J2p+J2e
        if verbose:
            return J1,J2,calTn(T,P)
        else:
            return J1,J2

class OceanSeisJack(SeisModel):  # https://doi.org/10.1016/j.pepi.2010.09.005
    '''
    Jackson & Faul, 2010: https://doi.org/10.1016/j.pepi.2010.09.005
    Migrated from William Shinevar's matlab function by Ayu, 20220603
    '''
    def __init__(self,therModel=None,gs=1e-3,period=1) -> None:
        if therModel is not None:
            self.fromThermal(therModel,gs,period)
        else:
            self.zdeps = None
            self.vs    = None
    def fromThermal(self,therMod,gs=1e-3,period=1):
        self._therMod = therMod
        self.zdeps = therMod.zdeps
        J1,J2,_ = self.creep10(therMod.T,gs,therMod.P,omega=np.pi*2/period)
        Ju = 1/(66.5-0.0136*(therMod.T-273.15-900)+1.8*(therMod.P/1e9-0.2))*1e-9
        self.vs = 1/np.sqrt(therMod.rho*Ju*J1)
        self.qs = J1/J2
        self.vs_no_anelastic = 1/np.sqrt(therMod.rho*Ju)
        
    @staticmethod
    def creep10(T,gs,pres,omega):
        try:
            len(T);Te = np.array(T)
        except:
            Te = np.array([T])

        Tr = 1173; iTr = 1/Tr   # reference temperature in K
        Pr = 0.2e9; PT = Pr/Tr  # reference pressure in Pa
        gsr = 1.34e-5           # reference grain size in m

        tauLo,tauHo,tauMo = 1E-3,1E7,3.02E7     #reference relaxation time

        deltaB = 1.04           # background relaxation strength
        alpha = 0.274           # background frequency exponent

        ma = 1.31               # anelastic grain size exponent
        mv = 3                  # viscous grain size exponent
        EB = 3.6E5              # activation energy for background and peak
        AV = 1E-5               # activation volume
        R = 8.314
        AVR  = AV/R; ER = EB/R; gr=gs/gsr

        # peak parameters:
        tauPo = 3.98E-4;        # reference peak relaxation time,
        deltaP = 0.057;         # peak relaxation strength pref (orig 0.057)
        sig = 4;                # peak width (orig 4)
        cp = deltaP*((2*np.pi)**(-0.5))/sig;       # peak integration const.

        # relaxation times eqs. 9 and 10
        taut = np.exp((ER)*(1/Te-iTr))*np.exp(AVR*((pres/Te)-PT))
        tauH = tauHo*(gr**ma)*taut
        tauL = tauLo*(gr**ma)*taut
        tauP = tauPo*(gr**ma)*taut
        tauM = tauMo*(gr**mv)*taut
        # initialize arrays
        # sT = size(Te)
        on  = np.ones(Te.shape)
        ij1 = np.zeros(Te.shape); ij2 = np.zeros(Te.shape)
        ip1 = np.zeros(Te.shape); ip2 = np.zeros(Te.shape)

        # from scipy import integrate
        def J1anel(tau):
            return tau**(alpha-1) / (1+(omega*tau)**2)
        def J2anel(tau):
            return tau**(alpha) / (1+(omega*tau)**2)
        def J1p(tauP):
            def _J1p(tau):
                return (1/tau)*np.exp(-0.5*(np.log(tau/tauP)/sig)**2)/(1+(omega*tau)**2)
            return _J1p
        def J2p(tauP):
            def _J2p(tau):
                return np.exp(-0.5*(np.log(tau/tauP)/sig)**2)/(1+(omega*tau)**2)
            return _J2p
            
        ij1 = np.array([logQuad(J1anel,l,h) for l,h in zip(tauL,tauH)])
        ij2 = np.array([logQuad(J2anel,l,h) for l,h in zip(tauL,tauH)])
        ip1 = np.array([logQuad(J1p(p),0,h) for p,h in zip(tauP,tauH)])
        ip2 = np.array([logQuad(J2p(p),0,h) for p,h in zip(tauP,tauH)])

        Jb1 = alpha*deltaB*ij1/(tauH**alpha-tauL**alpha)
        Jb2 = omega*alpha*deltaB*ij2/(tauH**alpha-tauL**alpha)
        Jp1 = cp*ip1
        Jp2 = cp*omega*ip2

        J1 = on + Jb1 + Jp1
        J2 = (Jb2 + Jp2) + 1/(omega*tauM)
        fM = 1/tauM

        # to test: J1,J2,fM = creep10(1000+273.15,1e-3,2e9,2*np.pi/1)

        return J1,J2,fM

class OceanSeisPM13(SeisModel): # http://dx.doi.org/10.1016/j.epsl.2013.08.022
    def __init__(self,therModel=None,period=1) -> None:
        if therModel is not None:
            self.fromThermal(therModel,period)
        else:
            self.zdeps = None
            self.vs    = None
    def fromThermal(self, therModel,period=1):
        Ju = 1/(72.66-0.00871*(therModel.T)+2.04*therModel.P/1e9)*1e-9

        E = 402.9e3 #J/mol
        Va = 7.81e-6 #m^3/mol
        R = 8.314 # J/(mol*K)
        Pr = 1.5e9 #Pa
        Tr = 1473 #K
        eta0 = np.power(10,22.38)
        aStar = np.exp((E+Pr*Va)/(R*Tr) - (E+therModel.P*Va)/(R*therModel.T))
        eta = eta0/aStar
        
        tauM = Ju*eta
        fPrime = tauM*1/period

        poly1d = np.poly1d([3.9461e-9,-3.4761e-7,9.9473e-6,-5.7175e-5,-2.3616e-3,0.054332,0.55097])
        F = poly1d(np.log(fPrime))
        F[fPrime > 1e13] = 1


        J1=Ju/F
        vs = 1/np.sqrt(therModel.rho*J1)
        self.zdeps = therModel.zdeps
        self.vs    = vs


def behn2009Shear(freq,d,T,P,coh=100):
    '''
        Behn+ 2009: https://doi.org/10.1016/j.epsl.2009.03.014
        modified from William Shinevar's matlab function by Ayu, 20220603

        frequency is frequency of seismic waves (Hz).
        d is the estimated grain size of the mantle in meters
        t is the temperature in C
        p is the pressure in GPa
    '''

    T=T+273.1;      # convert to K
    pqref=1.09;     # reference grain size exponenet
    pq=1            # grain size exponent
    Tqref=1265      # ?C, reference temperature
    dqref=1.24e-5   # m, reference grain size
    Eqref=505e3     # J/mol, reference activation energy
    Vqref=1.2e-5    # reference activation volume m^3/mol
    Bo=1.28e8       # prefactor for Q for omega=0.122 s^-1
    Eq=420e3        # activation energy
    Vq=1.2e-5       # activation volume
    cohref=50       # H/10^6 Si
    R=8.314
    Pqref=300e6     # reference pressure of 300 MPa
    rq=1.2
    alpha=0.27

    B=Bo*dqref**(pq-pqref)*(coh/cohref)**rq*np.exp(((Eq+Pqref*Vq)-(Eqref+Pqref*Vqref))/R/Tqref)
    Qinv=(B*d**(-1*pq)/freq*np.exp(-(Eq+P*1e9*Vq)/R/T))**alpha;#anelastic factor
    F=(1/np.tan(np.pi*alpha/2))/2
    shearFactor=(1-F*Qinv)**2
    return Qinv,shearFactor

# Qinv,shearFactor = behn2009Shear(1,1e-3,1000,2,100)
# J1,J2,_ = seisJack.creep10(1000+273.15,1e-3,2e9,2*np.pi/1)



if __name__ == '__main__':
    from Triforce.pltHead import *
    
    therMod = HSCM(4)
    mod1 = OceanSeisRitz(therMod,RhoType='raw')
    mod2 = OceanSeisRitz(therMod,RhoType='corrected')
    mod3 = OceanSeisRitz(therMod,RhoType='from_thermal')

    plt.figure(figsize=(5.5,8))
    plt.plot(mod1.vs,mod1.zdeps)
    plt.plot(mod2.vs,mod2.zdeps)
    plt.plot(mod3.vs,mod3.zdeps)
    plt.gca().invert_yaxis()
    mod4 = OceanSeisRitz(HSCM(3.5),RhoType='raw')
    plt.plot(mod4.vs,mod4.zdeps)


    plt.figure(figsize=(5.5,8))
    plt.plot(mod1._rho,mod1.zdeps)
    plt.plot(mod2._rho,mod2.zdeps)
    plt.plot(mod3._rho,mod3.zdeps)
    plt.gca().invert_yaxis()

    plt.figure(figsize=(5.5,8))
    plt.plot(mod3.vs-mod1.vs,mod1.zdeps)
    plt.gca().invert_yaxis()