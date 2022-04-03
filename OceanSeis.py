# ver = 0.01
import numpy as np


class TherModel():
    def __init__(self) -> None:
        self.zdeps = None
        self.P     = None
        self.T     = None
        pass

class HSCM(TherModel):
    def __init__(self, age, zdeps=np.linspace(0,200,200),rho0=3.43e3,Tp=1325) -> None:
        self.age   = age
        self.zdeps = zdeps
        self.P     = self._calP()
        self.T     = self._calT(Tp)
        self.rho    = self._calRho(rho0)
    def _calT(self,Tp=1325):
        C2K = 273.15
        zdeps = self.zdeps
        age   = self.age
        ''' temperature calculated from half space cooling model, topography change ignored''' 
        from scipy.special import erf
        T0 = 0; Da=0.4
        T_adiabatic = Tp + zdeps*Da
        theta = erf(zdeps*1e3/(2*np.sqrt(age*365*24*3600*1)))# suppose kappa=1e-6
        thetaD = np.diff(theta)/np.diff(zdeps)
        Tm_search = np.linspace(1250,1450,201)
        diffT_same_gradient = np.array( [ ((Tm-T0)*theta+T0-T_adiabatic)[np.argmin(abs((Tm-T0)*thetaD-Da))]
                                        for Tm in Tm_search] )    
        Tm = Tm_search[np.argmin(abs(diffT_same_gradient))]
        T = (Tm-T0)*theta+T0
        try:
            adiaBegin = np.where(np.diff(T)/np.diff(zdeps) < 0.4)[0][0]
            T[adiaBegin:] = T_adiabatic[adiaBegin:]
        except:
            pass
        return T+C2K
    def _calP(self,rho=3.4e3):  # in Pa
        # change 3.2 to 3.4, the difference is comparable to 4Ma vs 4.1Ma at 30km
        return rho*9.8*self.zdeps*1000   # 3.424 in https://doi.org/10.1029/2004JB002965
    def _calRho(self,rho0=3.43e3,P0=6.5e9,T0=1200+273.15):
        kappa,alpha = 1e-11,3e-5 # Pa^-1,K^-1
        # P = rho0*9.8*self.zdeps*1000
        P = self._calP()
        T = self._calT()
        rho = rho0*(1+kappa*(P-P0)-alpha*(T-T0))
        return rho

class seisModel():
    def __init__(self,therModel=None) -> None:
        self.zdeps = None
        self.vs    = None
        if therModel is not None:
            self.loadThermal(therModel)
    def loadThermal(self,therModel):
        pass





class OceanSeisBass(seisModel): # https://doi.org/10.1029/RF002p0045 https://doi.org/10.1016/j.pepi.2010.09.005
    def loadThermal(self, therModel):
        Ju = 1/(66.5-0.0136*(therModel.T-273.15-900)+1.8*(therModel.P/1e9-0.2))*1e-9
        vs = 1/np.sqrt(therModel.rho*Ju)
        self.zdeps = therModel.zdeps
        self.vs    = vs
class OceanSeisYaTa(seisModel):  # https://doi.org/10.1002/2016JB013316
    ''' unrelaxed only!!! '''
    def loadThermal(self, therModel):
        Ju = 1/(72.45-0.01094*(therModel.T-273.15)+1.987*therModel.P/1e9)*1e-9
        vs = 1/np.sqrt(therModel.rho*Ju)
        self.zdeps = therModel.zdeps
        self.vs    = vs
class OceanSeisStix(seisModel): # https://doi.org/10.1029/2004JB002965
    def loadThermal(self, therModel):
        vs = 4.77+0.038*therModel.zdeps/29.80-0.000378*(therModel.T-300)
        self.zdeps = therModel.zdeps
        self.vs    = vs*1000





class OceanSeisRitz(seisModel):  # https://doi.org/10.1111/j.1365-246X.2004.02254.x 
    def __init__(self,therModel=None) -> None:
        self.zdeps = None
        self.vs    = None
        self.elasticParas = {
            'Olivine':{'rho0':3.222e3,'rho_X':1.182,
                        'K0':129,'K_T':-16e-3,'K_P':4.2,'K_X':0,
                        'mu0':82,'mu_T':-14e-3,'mu_P':1.4,'mu_X':-30,
                        'alpha0':0.2010e-4,'alpha1':0.1390e-7,'alpha2':0.1627e-2,'alpha3':-0.3380},
            'Orthopyroxene':{'rho0':3.198e3,'rho_X':0.804,
                        'K0':111,'K_T':-12e-3,'K_P':6.0,'K_X':-10,
                        'mu0':81,'mu_T':-11e-3,'mu_P':2.0,'mu_X':-29,
                        'alpha0':0.3871e-4,'alpha1':0.0446e-7,'alpha2':0.0343e-2,'alpha3':-1.7278},
            'Clinopyroxene':{'rho0':3.280e3,'rho_X':0.377,
                        'K0':105,'K_T':-13e-3,'K_P':6.2,'K_X':13,
                        'mu0':67,'mu_T':-10e-3,'mu_P':1.7,'mu_X':-6,
                        'alpha0':0.3206e-4,'alpha1':0.0811e-7,'alpha2':0.1347e-2,'alpha3':-1.8167},
            'Spinel':{'rho0':3.578e3,'rho_X':0.702,
                        'K0':198,'K_T':-28e-3,'K_P':5.7,'K_X':12,
                        'mu0':108,'mu_T':-12e-3,'mu_P':0.8,'mu_X':-24,
                        'alpha0':0.6969e-4,'alpha1':-0.0108e-7,'alpha2':-3.0799e-2,'alpha3':5.0395},
            'Garnet':{'rho0':3.565e3,'rho_X':0.758,
                        'K0':173,'K_T':-21e-3,'K_P':4.9,'K_X':7,
                        'mu0':92,'mu_T':-10e-3,'mu_P':1.4,'mu_X':-7,
                        'alpha0':0.0991e-4,'alpha1':0.1165e-7,'alpha2':1.0624e-2,'alpha3':-2.5000}
        }
        if therModel is not None:
            self.loadThermal(therModel)
    def _pt2vs(self,P,T): # P in GPa, T in K
        def TPX2_rho_mu_K_pure(T,P,X,d):
            T0 = 273.15
            P0 = 101.325e-6
            alpha = d['alpha0'] + d['alpha1']*T + d['alpha2']*T**(-1) + d['alpha3']*T**(-2)
            rho0X = d['rho0'] * d['rho_X']
            mu  = d['mu0']  + (T-T0)*d['mu_T']  + (P-P0)*d['mu_P']  + X*d['mu_X']
            K   = d['K0']   + (T-T0)*d['K_T']   + (P-P0)*d['K_P']   + X*d['K_X']
            rho = rho0X*(1-alpha*(T-T0)+(P-P0)/K)
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

        X = 0.1; ws = [0.75,0.21,0.035,0,0.005]  # https://doi.org/10.1016/0012-821X(84)90076-1
        mu,K,rho = TPX2_rho_mu_K(T,P,X,list(self.elasticParas.values()),ws)
        # plt.figure(figsize=[5.5,8])
        # plt.plot(rho,self.zdeps)
        K,mu = K*1e9,mu*1e9
        vp,vs = np.sqrt(K+4/3*mu/rho),np.sqrt(mu/rho)
        self._rho = rho
        self._mu  = mu
        return vs

    def loadThermal(self,therMod):
        self.zdeps = therMod.zdeps
        self.vs    = self._pt2vs(therMod.P/1e9,therMod.T)

class OceanSeisRuan(seisModel):  # https://doi.org/10.1016/j.epsl.2018.05.035
    def __init__(self,therModel=None,damp=True,YaTaJu=False,period=50) -> None:
        self.zdeps = None
        self.vs    = None
        if therModel is not None:
            self.loadThermal(therModel,damp,YaTaJu,period)
    def loadThermal(self, therModel,damp=True,YaTaJu=False,period=50):
        Ju = 1/(72.45-0.01094*(therModel.T-273.15)+1.75*therModel.P/1e9)*1e-9
        if YaTaJu:
            Ju = 1/(72.45-0.01094*(therModel.T-273.15)+1.987*therModel.P/1e9)*1e-9
        J1,J2 = self._calQ_ruan(therModel.T,therModel.P,period=period,damp=damp)
        self.zdeps = therModel.zdeps
        self.vs    = 1/np.sqrt(therModel.rho*Ju*J1)
        self.vs_unrelaxed = 1/np.sqrt(therModel.rho*Ju)
        self.qs    = J1/J2
    @staticmethod
    def _calQ_ruan(T,P,period,damp=True):
        ''' calculate quality factor follow Ruan+(2018) 
        T: temperature in K
        P: pressure in Pa
        period: seismic wave period in second
        '''
        from scipy.special import erf
        def calTn(T,P): # solidus given pressure and temperature
            P = P/1e9
            if damp:
                Tm = -5.1*P**2 + 92.5*P + 1120.6 + 273.15
            else:
                Tm = -5.1*P**2 + 132.9*P + 1120.6 + 273.15
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

        return J1,J2





if __name__ == '__main__':
    from Triforce.pltHead import *

    zdeps = np.linspace(0,200,1000)
    therMod = HSCM(4,zdeps)
    seisModRitz = OceanSeisRitz(therMod)
    seisModRuan = OceanSeisRuan(therMod)
    seisModYaTa = OceanSeisRuan(therMod,YaTaJu=True)
    seisModBass = OceanSeisBass(therMod)
    seisModStix = OceanSeisStix(therMod)

    plt.figure(figsize=[5.5,8])
    plt.plot(seisModRitz.vs,zdeps,label='Rtizwoller')
    plt.plot(seisModRuan.vs,zdeps,label='Ruan')
    plt.plot(seisModRuan.vs_unrelaxed,zdeps,label='Ruan Unrelaxed')
    plt.plot(seisModYaTa.vs,zdeps,label='YaTa')
    plt.plot(seisModYaTa.vs_unrelaxed,zdeps,label='YaTa Unrelaxed')
    plt.legend()
    plt.gca().invert_yaxis()