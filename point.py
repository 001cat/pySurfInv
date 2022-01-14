import os,time,glob,random,yaml,scipy.signal
import numpy as np
import multiprocessing as mp
from copy import deepcopy
from scipy import interpolate
from geographiclib.geodesic import Geodesic
from Triforce.pltHead import *
from Triforce.utils import GeoMap, savetxt
from Triforce.obspyPlus import randString
from Triforce.customPlot import cvcpt,rbcpt
import sys; sys.path.append('../')
import pySurfInv.fast_surf as fast_surf

def plotLayer(h,v,axes=None,label=None,**kwargs):
    if axes is None:
        plt.figure(figsize=[5,7])
        axes = plt.axes()
    else:
        plt.axes(axes)
    hNew = np.insert(np.repeat(np.cumsum(h),2)[:-1],0,0)
    vNew = np.repeat(v,2)
    axes.plot(vNew,hNew,label=label,**kwargs)
    if not axes.yaxis_inverted():
        axes.invert_yaxis()
    return axes
def plotGrid(zdepth,v,fig=None,ax=None,label=None,**kwargs):
    if ax is None:
        if fig is None:
            fig = plt.figure(figsize=[5,7])
        else:
            plt.figure(fig.number)
    else:
        plt.axes(ax)
        fig = plt.gcf()
    plt.plot(v,zdepth,label=label,**kwargs)
    ax = ax or fig.axes[0]
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    return fig
def monoIncrease(a,eps=np.finfo(float).eps):
    return np.all(np.diff(a)>=0)
def randString(N):
    import random,string
    ''' Return a random string '''
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(N)])

def calT_HSCM(zdeps,age,z0=0):
    ''' temperature calculated from half space cooling model, topography change ignored''' 
    from scipy.special import erf
    T0 = 273.15; Tm = T0+1350
    T = (Tm-T0)*erf((zdeps-z0)*1e3/(2*np.sqrt(age*365*24*3600*1)))+T0 # suppose kappa=1e-6
    adiaBegin = np.where(np.diff(T)/np.diff(zdeps) < 0.4)[0][0]
    T[adiaBegin:] = T[adiaBegin]+(zdeps[adiaBegin:]-zdeps[adiaBegin])*0.4
    return T
def calP(zdeps):
    return 3.2e3*9.8*zdeps*1000

def calMantleQ(deps,vpvs,period=1,age=4.0):
    """ get Q value for mantle layer, follw Eq(4) from Ye (2013)
        Calculate Q value for 1 sec period, Q doesn't change a lot with period, age in unit of Ma
    """
    A = 30 # A value in Ye 2013 eq(4)
    temps = calT_HSCM(deps,max(1e-3,age))
    press = calP(deps)
    qs = A * (2*np.pi*1/period)**0.1 * np.exp(0.1*(2.5e5+press*1e-5)/(8.314472*temps))
    qp = 1./(4./3.*vpvs**(-2) / qs + (1-4./3.*vpvs**(-2))/57823.)
    return qs,qp

def calMantleQ_Ruan(deps,vpvs,period=1,age=4.0):
    def calQ_Ruan(T,P,period,damp=False):
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

        return J1/J2
    A = 30 # A value in Ye 2013 eq(4)
    temps = calT_HSCM(deps,max(1e-3,age))
    press = calP(deps)
    qs = calQ_Ruan(temps,press,period,damp=True)
    qp = 1./(4./3.*vpvs**(-2) / qs + (1-4./3.*vpvs**(-2))/57823.)
    # qs = np.zeros(deps.shape)
    # qs[deps>=125] = 120
    # qs[deps<=50] = 150
    # qs[(deps<125)*(deps>50)] = 50
    return qs,qp
def calCrustQ(vpvs):
    qs = 350
    qp = 1./(4./3.*(vpvs)**(-2) / qs + (1-4./3.*(vpvs)**(-2))/57823.)
    return qs,qp

class BsplBasis(object):
    ''' calculating cubic b-spline basis functions '''
    def __init__(self,z,n,deg=None,alpha=2.,eps=np.finfo(float).eps) -> None:
        self.n,self.nBasis,self.deg,self.alpha,self.eps = len(z),n,deg,alpha,eps
        if deg is None:
            deg = 3 + (n>=4)
        x = np.zeros(n+deg)
        x[:deg-1]	= -eps*np.ones(deg-1)
        x[deg-1]	= 0.
        x[deg:n]	= np.power(alpha,range(n-deg)) * (alpha-1) / (np.power(alpha,n-deg+1)-1)
        x[n]		= 1.
        x[n+1:]		= (1+eps)*np.ones(deg-1)
        x = z[0] + x*(z[-1]-z[0]) 
        bs0 = np.zeros((z.size,x.size-1))
        bs1 = bs0.copy()
        for i in range(bs0.shape[1]):
            bs0[ (z>=x[i]) * (z<x[i+1]),i] = 1
        for irun in range(deg-1):
            for i in range(bs0.shape[1]-irun-1):
                bs1[:,i] = 0
                if x[i+irun+1]-x[i] != 0:
                    bs1[:,i] += bs0[:,i]*(z-x[i])/(x[i+irun+1]-x[i])
                if x[i+1+irun+1]-x[i+1] != 0:
                    bs1[:,i] += bs0[:,i+1]*(x[i+1+irun+1]-z)/(x[i+1+irun+1]-x[i+1])
            bs0 = bs1.copy()
        bs = bs1[:,:n].copy()
        self.basis = bs.T
    def __mul__(self,coef):
        return np.dot(coef,self.basis)
    def plot(self):
        plt.figure();plt.plot(np.linspace(0,1,self.n),self.basis.T)

def randVar2List(v):
    if v.vmin is None:
        return [float(v),'fixed']
    else:
        return [float(v),v.vmin,v.vmax,v.step]
def list2RandVar(l):
    if type(l[1]) == str:
        try:
            v,vtype,lim,step = l
        except ValueError:
            v,vtype = l
        if vtype == 'rel':
            vmin,vmax = v*(1-lim/100),v*(1+lim/100)
        elif vtype == 'abs':
            vmin,vmax = v-lim,v+lim
            vmin = max(0,vmin)                                          # avoid negative parameters
        elif vtype == 'fixed' or vtype == 'default' or vtype == 'total':
            vmin,vmax,step = None,None,None
        else:
            raise ValueError()
    else:
        v,vmin,vmax,step = l
    return RandVar(v,vmin,vmax,step)
def dictR2L(dIn):
    if type(dIn) is dict:
        dOut = {}
        for k,v in dIn.items():
            dOut[k] = dictR2L(v)
    elif type(dIn) in (list,np.ndarray):
        dOut = []
        for v in dIn:
            dOut.append(dictR2L(v))
    elif type(dIn) is RandVar:
        dOut = randVar2List(dIn)
    else:
        dOut = deepcopy(dIn)
    return dOut
def dictL2R(dIn):
    if type(dIn) is dict:
        dOut = {}
        for k,v in dIn.items():
            dOut[k] = dictL2R(v)
    elif type(dIn) in (list,np.ndarray) and type(dIn[0]) is list:
        dOut = []
        for v in dIn:
            dOut.append(dictL2R(v))
    elif type(dIn) in (list,np.ndarray) and type(dIn[0]) is not list:
        dOut = list2RandVar(dIn)
    else:
        dOut = deepcopy(dIn)
    return dOut

class Setting(dict):
    ''' Parameter setting to construct Model1D '''
    def load(self,setting):
        self.clear()
        if type(setting) not in (Setting,dict):
            with open(setting, 'r') as f:
                setting = yaml.load(f,Loader=yaml.FullLoader)
        for k,v in setting.items():
            self[k] = deepcopy(v)
    @property
    def info(self):
        return self['Info']
    def _updateWater(self,topo):
        waterDepth = max(-topo,0)
        if waterDepth <= 0:
            if 'water' in self.keys():
                self.pop('water')
        else:
            if 'water' in self.keys():
                self['water']['h'][0] = waterDepth
            else:
                print('Warning: topo<0 found! Please add water layer in setting file!!')
    def _updateSediment(self,sedthk):
        self['sediment']['h'][0] = sedthk
    def _updateCrust(self,crsthk):
        self['crust']['h'][0] = crsthk
    def updateInfo(self,topo=None,sedthk=None,crsthk=None,lithoAge=None):
        if topo is not None:
            self.info['topo'] = topo
            self._updateWater(topo)
        if sedthk is not None:
            self.info['sedthk'] = sedthk
            self._updateSediment(sedthk)
        if crsthk is not None:
            self.info['crsthk'] = crsthk
            self._updateCrust(crsthk)
        if lithoAge is not None:
            self.info['lithoAge'] = lithoAge
    # def _updateVars(self,newParas):  #for test only
    #     i = 0
    #     for ltype in self.keys():
    #         if ltype == 'Info':
    #             continue
    #         for k in self[ltype].keys():
    #             if type(self[ltype][k]) is str:
    #                 continue
    #             if type(self[ltype][k][0]) is list:
    #                 for j in range(len(self[ltype][k])):
    #                     if self[ltype][k][j][1] in ('abs','rel'):
    #                         self[ltype][k][j][0] = newParas[i];i+=1
    #             else:
    #                 if self[ltype][k][1] in ('abs','rel'): 
    #                     self[ltype][k][0] = newParas[i];i+=1
    def copy(self):
        return deepcopy(self)

class RandVar(float):
    ''' Random variables, varies between vmin and vmax. It could be gaussian perturbed 
    with sigma=step, uniformly reset or reset to the center of vmin and vmax'''
    def __new__(cls,v,vmin=None,vmax=None,step=None):
        return super().__new__(cls,v)
    def __init__(self,v,vmin=None,vmax=None,step=None) -> None:
        self.vmin,self.vmax,self.step = vmin,vmax,step
    def reset(self,resetype='center'):
        if self.vmin == None:
            vNew = float(self)
        elif resetype == 'center':
            vNew = (self.vmin+self.vmax)/2
        elif resetype == 'uniform':
            vNew = random.uniform(self.vmin,self.vmax)
        else:
            raise ValueError('Unknown reset type! Could only be uniform or center(default)')
        return RandVar(vNew,self.vmin,self.vmax,self.step)
    def perturb(self,avoidLargeStep=True):
        if self.vmin == None:
            return RandVar(float(self),self.vmin,self.vmax,self.step)
        if avoidLargeStep:
            step = min((self.vmax-self.vmin)/2,self.step)
        for i in range(1000):
            vNew = random.gauss(self,step)
            if vNew < self.vmax and vNew > self.vmin:
                break
            if i == 999:
                print(f'No valid perturb, uniform reset instead! '+
                      f'{self} {self.vmin} {self.vmax} {self.step}')
                return self.reset('uniform')
        return RandVar(vNew,self.vmin,self.vmax,self.step)
    def show(self):
        print(f'v={self} vmax={self.vmax} vmin={self.vmin} step={self.step}')

class SurfLayer(object):
    ''' Horizontal layers used to generate Model1D '''
    def __init__(self,layerType,settingDict) -> None:
        self.type,self.mtype = layerType,settingDict['mtype']
        self.stype = '' if 'stype' not in settingDict.keys() else settingDict['stype']
        self.paraDict = {}
        if self.mtype == 'water':
            self.paraDict['h'] = dictL2R(settingDict['h'])
            self.paraDict['vp'] = dictL2R(settingDict['vp'])
        elif self.mtype == 'grid':
            self.paraDict['zdeps'] = np.array([RandVar(vs) for vs in settingDict['zdeps']])
            self.paraDict['vs'] = np.array([RandVar(vs) for vs in settingDict['vs']])
            self.paraDict['vpvs'] = dictL2R(settingDict['vpvs'])
        else:
            self.paraDict['h'] = dictL2R(settingDict['h'])
            self.paraDict['vs'] = dictL2R(settingDict['vs'])
            self.paraDict['vpvs'] = dictL2R(settingDict['vpvs'])
    @property
    def H(self):
        if self.mtype == 'grid':
            return self.paraDict['zdeps'][-1] - self.paraDict['zdeps'][0]
        else:
            return self.paraDict['h']
    @property
    def vs(self):
        return self.paraDict['vs']
    @property
    def vp(self):
        return self.paraDict['vp']
    @property
    def vpvs(self):
        return self.paraDict['vpvs']
    @property
    def nFine(self):
        if self.mtype == 'water':
            nFine = 1
        elif self.mtype == 'constant':
            nFine = 1 if type(self.vs) is not list else len(self.vs)
        elif self.mtype == 'linear':
            nFine = min(20,int(self.H/1.0)) if self.H > 10 else max(int(self.H/1),2)
        elif self.mtype == 'Bspline':
            if self.H >= 150:
                nFine = 60
            elif self.H > 60:
                nFine = 30
            elif self.H > 20:
                nFine = 15
            elif self.H > 10:
                nFine = 10
            else:
                nFine = 5
        elif self.mtype == 'grid':
            nFine = len(self.paraDict['zdeps'])-1
        return nFine
    @property
    def nRandVar(self):
        n = 0
        for v in self.paraDict.values():
            if type(v) is not list:
                n += (v.vmin is not None)
            else:
                for vsub in v:
                    n += (vsub.vmin is not None)
        return n
    def bspl(self,z,nBasis,deg):
        if hasattr(self,'_bspl') and (nBasis == self._bspl.nBasis) and \
           (deg == self._bspl.deg) and (len(z) == self._bspl.n):
           pass
        else:
            self._bspl = BsplBasis(z,nBasis,deg)
        return self._bspl
    def updateVars(self,vars):
        if self.nRandVar != len(vars):
            raise ValueError('Variables to be loaded is incompatible with model!')
        i = 0
        for k in self.paraDict.keys():
            if type(self.paraDict[k]) is not list:
                if self.paraDict[k].vmin is not None:
                    self.paraDict[k] = RandVar(vars[i],self.paraDict[k].vmin,
                                            self.paraDict[k].vmax,self.paraDict[k].step)
                    i += 1
            else:
                for j in range(len(self.paraDict[k])):
                    if self.paraDict[k][j].vmin is not None:
                        self.paraDict[k][j] = RandVar(vars[i],self.paraDict[k][j].vmin,
                                                self.paraDict[k][j].vmax,self.paraDict[k][j].step)
                        i += 1
    def _vsProfileGrid(self):
        if self.mtype == 'water':
            return np.array([0]*(self.nFine+1))
        elif self.mtype == 'constant':
            return np.array([self.vs]*(self.nFine+1))
        elif self.mtype == 'linear':
            return np.linspace(self.vs[0],self.vs[1],self.nFine+1)
        elif self.mtype == 'Bspline':
            z = np.linspace(0, self.H, self.nFine+1)
            nBasis = len(self.vs)
            deg = 3 + (nBasis>=4)
            return self.bspl(z,nBasis,deg) * self.vs
        elif self.mtype == 'grid':
            return self.paraDict['vs']
        else:
            raise ValueError('Not supported mtype.')
    def _hProfile(self):
        if self.mtype == 'grid':
            return self.paraDict['zdeps'][1:]-self.paraDict['zdeps'][:-1]
        else:
            return np.array([self.H/self.nFine]*self.nFine)
    def _vsProfile(self):
        vsGrid = self._vsProfileGrid()
        return (vsGrid[:-1]+vsGrid[1:])/2
    def _vsProfile_Leon(self):       # interp using n instead of n+1, which should be worse
        if self.mtype == 'water':
            return np.array([0]*self.nFine)
        elif self.mtype == 'constant':
            return np.array([self.vs]*self.nFine)
        elif self.mtype == 'linear':
            tmp = np.linspace(self.vs[0],self.vs[1],self.nFine)
            return tmp
        elif self.mtype == 'Bspline':
            z = np.linspace(0, self.H, self.nFine)
            nBasis = len(self.vs)
            deg = 3 + (nBasis>=4)
            tmp = self.bspl(z,nBasis,deg) * self.vs
            return tmp
        else:
            raise ValueError('Not supported mtype.')
    def genProfile(self):
        vs = self._vsProfile()
        h  = self._hProfile()
        if self.type+self.stype == 'water':
            vp  = [self.vp]     *self.nFine
            rho = [1.027]       *self.nFine
            qs  = [10000.]      *self.nFine
            qp  = [57822.]      *self.nFine
        elif self.type+self.stype == 'sediment':
            vpvs = 2.0 if not hasattr(self,'vpvs') else self.vpvs
            # vp   = vs*vpvs
            vp   = vs*1.23 + 1.28   # marine sediments and rocks, Hamilton 1979
            rho  = 0.541 + 0.3601*vp
            qs  = [80.]   *self.nFine
            qp  = [160.]  *self.nFine
        elif self.type+self.stype == 'sedimentland':
            vpvs = self.vpvs
            vp   = vs*vpvs
            rho  = 0.541 + 0.3601*vp
            qs  = [80.]   *self.nFine
            qp  = [160.]  *self.nFine
        elif self.type+self.stype == 'crust':
            vpvs = 1.8 if not hasattr(self,'vpvs') else self.vpvs
            vp   = vs*vpvs
            rho  = 0.541 + 0.3601*vp
            qs  = [600.]   *self.nFine
            qp  = [1400.]  *self.nFine
        elif self.type+self.stype == 'mantle':
            vpvs = 1.76 if not hasattr(self,'vpvs') else self.vpvs
            vp = vs*vpvs
            rho = 3.4268+(vs-4.5)/4.5
            # rho  = 0.541 + 0.3601*vp    ## from Hongda
            qs  = [150.]   *self.nFine
            qp  = [1400.]  *self.nFine
        else:
            raise ValueError('Valid type+subtype')
        # rho = np.array(rho)
        # rho[np.array(vp) > 7.5]       = 3.35
        return np.array([h,vs,vp,rho,qs,qp])
    def genProfileGrid(self):
        vs = self._vsProfileGrid()
        zdepth = np.insert(self._hProfile().cumsum(),0,0)
        ltype = [self.type]*(self.nFine+1)
        return [zdepth,vs,ltype]
    def _perturb(self):
        for paraKey in self.paraDict.keys():
            if type(self.paraDict[paraKey]) is not list:
                self.paraDict[paraKey] = self.paraDict[paraKey].perturb()
            else:
                self.paraDict[paraKey] = [v.perturb() for v in self.paraDict[paraKey]]
    def perturb(self):
        newLayer = self.copy()
        newLayer._perturb()
        return newLayer
    def _reset(self,resetype='center'):
        for paraKey in self.paraDict.keys():
            if type(self.paraDict[paraKey]) is not list:
                self.paraDict[paraKey] = self.paraDict[paraKey].reset(resetype=resetype)
            else:
                self.paraDict[paraKey] = [v.reset(resetype=resetype) for v in self.paraDict[paraKey]]
    def reset(self,resetype='center'):
        newLayer = self.copy()
        newLayer._reset(resetype=resetype)
        return newLayer
    def copy(self):
        return deepcopy(self)

def _calForward(inProfile,wavetype='Ray',periods=[5,10,20,40,60,80]):
    if wavetype == 'Ray':
        ilvry = 2
    elif wavetype == 'Love':
        ilvry = 1
    else:
        raise ValueError('Wrong surface wave type: %s!' % wavetype)
    
    ind = np.where(inProfile[0]>1e-3)[0]    #1e-5 gives wrong dispersion in Daruk
    h,Vs,Vp,rho,qs,qp = inProfile[:,ind]
    qsinv			= 1./qs
    nper			= len(periods)
    per 			= np.zeros(200, dtype=np.float64)
    per[:nper]		= periods[:]
    nlay			= h.size
    (ur0,ul0,cr0,cl0)       = fast_surf.fast_surf(nlay, ilvry, Vp, Vs, rho, h, qsinv, per, nper)
    
    if np.any(cr0[:nper]<0.01):
        return None
    return cr0[:nper]
    # phDisp = SurfDisp(period,cr0[:nper],wtype=wavetype,ctype='Phase')
    # grDisp = SurfDisp(period,ur0[:nper],wtype=wavetype,ctype='Group')
    # return (phDisp,grDisp) 
class Model1D(object):
    ''' One demensional isotropic model '''
    def __init__(self,layerList=[],info={}) -> None:
        self.layers = layerList
        self.info   = info
    @property
    def altitudeH(self):
        altitudeH = 0 if 'topo' not in self.info.keys() else max(self.info['topo'],0)
        return altitudeH
    @property
    def totalH(self):
        return np.sum([layer.H for layer in self.layers])
    def _ltypes(self,refLayer=False):
        types = []
        for layer in self.layers:
            types += [layer.type]*layer.nFine
        if refLayer is True:
            refLayer = self.refLayer
            types += [refLayer.type]*refLayer.nFine
        return np.array(types)
    @property
    def refLayer(self):
        profile = np.concatenate([layer.genProfile() for layer in self.layers],axis=1)
        vs0,h0 = profile[1][-1],profile[0][-1]
        return SurfLayer('mantle',{'mtype':'linear','h':[300,'fixed'],
                                'vs':[[vs0,'fixed'],[vs0+0.35/200*300,'fixed']],
                                'vpvs':1.75}) # 0.35km/s per 200km from AK135
    def loadSetting(self,settingYML):
        setting = Setting(); setting.load(settingYML)
        self.info = setting['Info']; setting.pop('Info')
        if list(setting.values())[-1]['mtype'] != 'grid':
            if list(setting.values())[-1]['h'][1] == 'total':
                Hs = [setting[ltype]['h'][0] for ltype in setting.keys()]
                setting[list(setting.keys())[-1]]['h'] = [Hs[-1]+self.altitudeH-np.sum(Hs[:-1]),'total']
            else:
                raise ValueError('Thickness of last layer should be labeled as total!')
        self.layers = [SurfLayer(ltype,setting[ltype]) for ltype in setting.keys()]
    def toSetting(self):
        z0 = 0 if 'topo' not in self.info.keys() else max(0,self.info['topo'])
        setting = Setting()
        for layer in self.layers:
            setting[layer.type] = {'mtype':layer.mtype,'stype':layer.stype}
            setting[layer.type].update(dictR2L(layer.paraDict))
        setting[layer.type]['h'] = [self.totalH-z0,'total']
        setting['Info'] = self.info
        return deepcopy(setting)
    def setFinalH(self,totalH):
        finalH = totalH - np.sum([l.paraDict['h'] for l in self.layers[:-1]])
        self.layers[-1].paraDict['h'] = RandVar(finalH)
    def updateVars(self,vars):
        totalH = self.totalH
        i = 0
        for l in self.layers:
            l.updateVars(vars[i:i+l.nRandVar])
            i += l.nRandVar
        if i != len(vars):
            raise ValueError('Variables to be loaded is incompatible with model!')
        self.setFinalH(totalH)
    def perturb(self):
        i = 0
        while i < 100:
            newMod = Model1D([layer.perturb() for layer in self.layers],self.info)
            newMod.setFinalH(self.totalH)
            i += 1
            if newMod.isgood():
                return newMod
        print('Warning: no good perturbation found, return uniform reset instead')
        return self.reset('uniform')
    def reset(self,resetype='center'):
        i = 0
        while i < 1000:
            newMod = Model1D([layer.reset(resetype) for layer in self.layers],self.info)
            newMod.setFinalH(self.totalH)
            i += 1
            if newMod.isgood():
                return newMod
        print(f'Error: no good reset:{resetype} found!!')
    def genProfile(self,refLayer=False): # h,vs,vp,rho,qs,qp
        profile = np.concatenate([layer.genProfile() for layer in self.layers],axis=1)
        typeLst = self._ltypes(refLayer=refLayer)
        if refLayer is True:
            profile = np.concatenate((profile,self.refLayer.genProfile()),axis=1)
        if 'lithoAge' in self.info.keys() and self.info['lithoAge'] is not None: # assume mantle is the last layer
            typeLst = np.array(typeLst)
            tmp = profile.copy()
            h = tmp[0];deps = h.cumsum()
            vpvsC = 1.8  if not hasattr(self.layers[-2],'vpvs') else self.layers[-2].vpvs
            vpvsM = 1.76 if not hasattr(self.layers[-1],'vpvs') else self.layers[-1].vpvs
            qsCrust = 350
            qpCrust = 1./(4./3.*(vpvsC)**(-2) / qsCrust + (1-4./3.*(vpvsC)**(-2))/57823.)
            qsCrust,qpCrust   = calCrustQ(vpvsC)
            qsMantle,qpMantle = calMantleQ(deps[typeLst=='mantle'],vpvsM,age=self.info['lithoAge'])
            if 'Qmodel' in self.info.keys() and self.info['Qmodel'] == 'Ruan2018':
                qsMantle,qpMantle = calMantleQ_Ruan(deps[typeLst=='mantle'],vpvsM,age=self.info['lithoAge'])
            tmp[4,typeLst=='crust'] = qsCrust
            tmp[5,typeLst=='crust'] = qpCrust
            tmp[4,typeLst=='mantle'] = qsMantle[:]
            tmp[5,typeLst=='mantle'] = qpMantle[:]
            try:
                if self.info['specialHighQs'] is True:
                    z1 = h[typeLst!='mantle'].sum()
                    tmp[4,(typeLst=='mantle') * (deps<z1+40)] = 5000
            except:
                pass
            return tmp
        else:
            return profile
    def genProfileGrid(self): # h,vs only
        zdepth,vs,ltype = [],[],[]
        try:
            z0 = -max(self.info['topo'],0)
        except:
            z0 = 0
        zOffset = z0
        for layer in self.layers:
            a,b,c = layer.genProfileGrid()
            zdepth.extend(a+zOffset);vs.extend(b);ltype.extend(c)
            zOffset = zdepth[-1]
        return zdepth,vs,ltype
    def forward(self,periods=[5,10,20,40,60,80]):
        refLayer = False if 'refLayer' not in self.info.keys() else self.info['refLayer']
        pred = _calForward(self.genProfile(refLayer=refLayer),wavetype='Ray',periods=periods)
        if pred is None:
            print(f'Warning: Forward not complete! Model listed below:')
            self.show()
        return pred
    def isgood(self):
        ltypes = self._ltypes()
        h,vs,vp,rho,qs,qp = self.genProfile()
        vsSediment = vs[ltypes=='sediment']
        vsCrust = vs[ltypes=='crust']
        vsMantle = vs[ltypes=='mantle']

        # Vs in sediment > 0.2; from Lili's code
        if np.any(vsSediment<0.2):
            return False
        # Vs jump between layer is positive, contraint (5) in 4.2 of Shen et al., 2012
        for i in np.where((ltypes[1:]!=ltypes[:-1]))[0]:
            if vs[i+1] < vs[i]:
                return False
        # # All Vs < 4.9km/sec, contraint (6) in 4.2 of Shen et al., 2012
        # if np.any(vs > 4.9):
        #     return False
        # velocity in the sediment must increase with depth
        if not monoIncrease(vs[ltypes=='sediment']):
            return False
        # velocity in the crust must increase with depth
        if not monoIncrease(vs[ltypes=='crust']): 
            return False
        # negative velocity gradient below moho for ocean

        if 'NegSlopeBelowCrust' in self.info.keys() and self.info['NegSlopeBelowCrust'] is False:
            pass
        elif 'noNegSlopeBelowCrust' in self.info.keys() and self.info['noNegSlopeBelowCrust'] is True: 
            #deprecated, just for compatibility 
            pass
        elif not vsMantle[1]<vsMantle[0]:
            return False
        # # vs in crust < 4.3
        # if np.any(vsCrust > 4.3):
        #     return False
        # # Vs at first fine layer in mantle is between 4.0 and 4.6
        # if vsMantle[0] < 4.0 or vsMantle[0] > 4.6:
        #     return False
        # # Vs at last fine layer in mantle > 4.3 
        # if vsMantle[-1] < 4.3:
        #     return False
        # # Vs > 4.0 below 80km
        # if np.any(vsMantle < 4.0):
        #     return False
        # change in mantle < 15%
        if (vsMantle.max() - vsMantle.min()) > 0.15*vsMantle.mean():
            return False
        # Oscillation Limit
        osciLim = 0.1*vsMantle.mean()
        indLocMax = scipy.signal.argrelmax(vsMantle)[0]
        indLocMin = scipy.signal.argrelmin(vsMantle)[0]
        if len(indLocMax) + len(indLocMin) > 1:
            indLoc = np.sort(np.append(indLocMax,indLocMin))
            osci = abs(np.diff(vsMantle[indLoc]))
            if len(np.where(osci > osciLim)[0]) >= 1:   # origin >= 1
                return False

        # temporary only
        # if vsMantle[-1] - vsMantle[-2] < 0:
        #     return False

        return True
    def plotProfile(self,type='vs',**kwargs):
        h,vs,vp,rho,qs,qp = self.genProfile()
        if type == 'vs':
            fig = plotLayer(h,vs,**kwargs);plt.title('Vs')
        else:
            print('To be added...')
        return fig
    def plotProfileGrid(self,type='vs',ax=None,**kwargs):
        zdepth,vs,ltype = self.genProfileGrid()
        if type == 'vs':
            fig = plotGrid(zdepth,vs,ax=ax,**kwargs);plt.title('Vs')
        else:
            print('To be added...')
        return fig
    def _paras(self):
        return self.paras(fullInfo=False)
    def paras(self,fullInfo=False):
        paras = []
        for l in self.layers:
            for k in l.paraDict.keys():
                if type(l.paraDict[k]) == list:
                    for v in l.paraDict[k]:
                        paras.append([v,l.type,k])
                else:
                    v = l.paraDict[k]
                    paras.append([v,l.type,k])
        if not fullInfo:
            paras = [a[0] for a in paras if a[0].vmin is not None]
        else:
            paras = [a for a in paras if a[0].vmin is not None]
        return paras
    def copy(self):
        return deepcopy(self)
    def write(self,index,target,addInfo):
        addInfo.extend(self._paras())
        target[index] = addInfo
        pass
    def show(self):
        for layer in self.layers:
            print(layer.type)
            print(layer.paraDict)

def accept(L0,L1):
    if L0 == 0:
        return True
    return random.random() > (L0-L1)/L0
class Point(object):
    def __init__(self,settingYML=None,periods=[],vels=[],uncers=[],
                      sedthk=None,crsthk=None,topo=None,lithoAge=None) -> None:
        setting = Setting(); setting.load(settingYML)
        setting.updateInfo(sedthk=sedthk,topo=topo,crsthk=crsthk,lithoAge=lithoAge)
        self.initMod = Model1D(); self.initMod.loadSetting(setting)
        self.obs = {'T':periods,'c':vels,'uncer':uncers}    # Rayleigh wave, phase velocity only
    def misfit(self,model=None):
        if model is None:
            model = self.initMod
        T = self.obs['T']
        cP = model.forward(periods=T)
        if cP is None:
            return 88888,0
        cO = self.obs['c']
        uncer = self.obs['uncer']
        N = len(T)
        chiSqr = (((cO - cP)/uncer)**2).sum()
        misfit = np.sqrt(chiSqr/N)
        chiSqr =  chiSqr if chiSqr < 50 else np.sqrt(chiSqr*50.) 
        L = np.exp(-0.5 * chiSqr)
        return misfit,L
    def MCinv(self,outdir='MCtest',id='test',runN=50000,step4uwalk=1000,init=True,
              seed=None,verbose=False,priori=False):
        debug = False
        random.seed(seed)
        timeStamp = time.time()
        mcTrack = [0]*runN
        for i in range(runN):
            if i % step4uwalk == 0:
                if init:
                    mod0 = self.initMod.copy();init=False
                else:
                    mod0 = self.initMod.reset('uniform')
                    if verbose == True:
                        print(f'{i+1}/{runN} Time cost:{time.time()-timeStamp:.2f} ')
                misfit0,L0 = self.misfit(mod0)
                mod0.write(i,mcTrack,[misfit0,L0,1])
                # modSeed = mod0
            else:
                mod1 = mod0.perturb()   # Check mod0 or self.initMod
                # mod1 = modSeed.perturb()
                if debug:
                    plt.figure()
                    T = self.obs['T']
                    plt.plot(T,self.obs['c'],'--')
                    plt.plot(T,mod0.forward(periods=T))
                    plt.plot(T,mod1.forward(periods=T))
                if priori:
                    mod1.write(i,mcTrack,[0,1,1])
                    mod0 = mod1
                    continue
                misfit1,L1 = self.misfit(mod1)
                if accept(L0,L1):
                    mod1.write(i,mcTrack,[misfit1,L1,1])
                    mod0,misfit0,L0 = mod1,misfit1,L1
                else:
                    mod1.write(i,mcTrack,[misfit1,L1,0])
                if debug and L0>0.01:
                    debug = False if input() == 'Y' else True
                    plt.close()
        mcTrack = np.array(mcTrack)
        os.makedirs(outdir,exist_ok=True)
        np.savez_compressed(f'{outdir}/{id}.npz',mcTrack=mcTrack,
                            setting=dict(self.initMod.toSetting()),obs=self.obs)
        if verbose == 'mp':
            print(f'Step {id.split("_")[1]} Time cost:{time.time()-timeStamp:.2f} ')
    def MCinvMP(self,outdir='MCtest',id='test',runN=50000,step4uwalk=1000,nprocess=12,seed=None,priori=False):
        tmpDir = 'MCtmp'+randString(10)
        random.seed(seed); seed = random.random()
        argInLst = [ [tmpDir,f'tmp_{i:03d}_{id}',step4uwalk,step4uwalk,i==0,seed+i,'mp',priori]
                     for i in range(runN//step4uwalk)]
        timeStamp = time.time()
        pool = mp.Pool(processes=nprocess)
        pool.starmap(self.MCinv, argInLst)
        pool.close()
        pool.join()

        subMCLst = []
        for argIn in argInLst:
            tmp = np.load(f'{tmpDir}/{argIn[1]}.npz',allow_pickle=True)
            subMC,_,_ = tmp['mcTrack'],tmp['setting'][()],tmp['obs'][()]
            subMCLst.append(subMC)
        os.system(f'rm -r {tmpDir}')
        mcTrack = np.concatenate(subMCLst,axis=0)
        os.makedirs(outdir,exist_ok=True)
        np.savez_compressed(f'{outdir}/{id}.npz',mcTrack=mcTrack,
                            setting=dict(self.initMod.toSetting()),obs=self.obs)

        print(f'Time cost:{time.time()-timeStamp:.2f} ')
        
class PostPoint(Point):
    def __init__(self,npzMC=None,npzPriori=None):
        if npzMC is not None:
            tmp = np.load(npzMC,allow_pickle=True)
            self.MC,setting,self.obs = tmp['mcTrack'],tmp['setting'][()],tmp['obs'][()]
            self.initMod = Model1D(); self.initMod.loadSetting(setting)
            
            self.N       = self.MC.shape[0]
            self.misfits = self.MC[:,0]
            self.Ls      = self.MC[:,1]
            self.accepts = self.MC[:,2]
            self.MCparas = self.MC[:,3:]

            indMin = np.nanargmin(self.misfits)
            self.minMod         = self.initMod.copy()
            self.minMod.updateVars(self.MCparas[indMin])
            self.minMod.L       = self.Ls[indMin]
            self.minMod.misfit  = self.misfits[indMin]

            self.thres  = max(self.minMod.misfit*2, self.minMod.misfit+0.5)
            self.accFinal = (self.misfits < self.thres)

            # bspl = BsplBasis(np.linspace(0,1,1000),5,4)
            # roughBasis = (bspl.basis[:,:-2] + bspl.basis[:,2:] - 2*bspl.basis[:,1:-1])
            # avgParas = np.mean(self.MCparas[self.accFinal,:],axis=0)
            # biaParas = self.MCparas[:,-5:] - avgParas[-5:]
            # rough = np.array([abs(np.dot(biaParas[ind],roughBasis)).sum() for ind in range(self.N)])
            # self.accFinal = (self.misfits < self.thres) * (rough<=0.004)
            # print(f'rejected by roughness:{(1-self.accFinal.sum()/(self.misfits < self.thres).sum())*100:.2f}%')
            

            self.avgMod         = self.initMod.copy()
            self.avgMod.updateVars(np.mean(self.MCparas[self.accFinal,:],axis=0))
            self.avgMod.misfit,self.avgMod.L = self.misfit(model=self.avgMod)
    
        if npzPriori is not None:
            tmp = np.load(npzPriori,allow_pickle=True)['mcTrack']
            self.Priparas = tmp[:,3:]
    def loadpyMCinv(self,dsetFile,id,invDir,priDir=None):
        from MCinv.ocean_surf_dbase import invhdf5
        setting_Hongda_pyMCinv = {'water': {'type': 'water',
                                    'h':  [1,'fixed'],
                                    'vp': [1.475,'fixed']},
                                  'sediment': {'type': 'constant',
                                    'h':  [2, 'abs', 1.0, 0.1],
                                    'vs': [1.0, 'abs', 1.0, 0.01],
                                    'vpvs': [2, 'fixed']},
                                  'crust': {'type': 'linear',
                                    'h': [7, 'abs',0.001,0.001],
                                    'vs': [[3.25, 'abs', 0.001, 0.001],
                                           [3.94, 'abs', 0.001, 0.001]],
                                    'vpvs': [1.8, 'fixed']},
                                  'mantle': {'type': 'Bspline',
                                    'h': [200, 'total'],
                                    'vs': [[4.4, 'rel', 10, 0.02],
                                           [4.0, 'rel', 10, 0.02],
                                           [4.3, 'rel', 10, 0.02],
                                           [4.5, 'rel', 5, 0.02]],
                                    'vpvs': [1.76, 'fixed']},
                                  'Info':{'label':'Hongda-2021Summer'}}

        dset = invhdf5(dsetFile,'r')
        topo = dset[id].attrs['topo']
        sedthk = dset[id].attrs['sedi_thk']
        lithoAge = dset[id].attrs['litho_age']
        setting = Setting(setting_Hongda_pyMCinv)
        setting.updateInfo(topo=topo,sedthk=sedthk,lithoAge=lithoAge)
        self.initMod = Model1D()
        self.initMod.loadSetting(setting_Hongda_pyMCinv)

        T,pvelp,pvelo,uncer = np.loadtxt(f'{invDir}/{id}_0.ph.disp').T
        self.obs = {'T':T,'c':pvelo,'uncer':uncer}

        inarr = np.load(f'{invDir}/mc_inv.{id}.npz')
        invdata    = inarr['arr_0']
        disppre_ph = inarr['arr_1']
        disppre_gr = inarr['arr_2']
        rfpre      = inarr['arr_3']

        self.N        = invdata.shape[0]
        self.accepts  = invdata[:,0]
        iaccept       = invdata[:,1]
        paraval       = invdata[:,2:11]
        self.Ls       = invdata[:,11]
        self.misfits  = invdata[:,12]

        # vsed,vcrust1,vcrust2,vmantle1,vmantle2,vmantle3,vmantle4,hsed,hcrust
        colOrder = np.array([7,0,8,1,2,3,4,5,6])
        self.MCparas = paraval[:,colOrder]



        indMin = np.nanargmin(self.misfits)
        self.minMod         = self.initMod.copy()
        self.minMod.updateVars(self.MCparas[indMin])
        self.minMod.L       = self.Ls[indMin]
        self.minMod.misfit  = self.misfits[indMin]

        self.thres  = max(self.minMod.misfit*2, self.minMod.misfit+0.5)
        self.accFinal = (self.misfits < self.thres)

        self.avgMod         = self.initMod.copy()
        self.avgMod.updateVars(np.mean(self.MCparas[self.accFinal,:],axis=0))
        self.avgMod.misfit,self.avgMod.L = self.misfit(model=self.avgMod)

        if priDir is not None:
            inarr = np.load(f'{priDir}/mc_inv.{id}.npz')
            invdata    = inarr['arr_0']
            paraval       = invdata[:,2:11]
            self.Priparas = paraval[:,colOrder]
    def plotDisp(self,ax=None):
        T,vel,uncer = self.obs['T'],self.obs['c'],\
                      self.obs['uncer']
        if ax is None:
            plt.figure()
        else:
            plt.axes(ax)
        mod = self.avgMod.copy()
        indFinAcc = np.where(self.accFinal)[0]
        for _ in range(min(len(indFinAcc),500)):
            i = random.choice(indFinAcc)
            mod.updateVars(self.MCparas[i,:])
            plt.plot(T,mod.forward(T),color='grey',lw=0.1)
        plt.errorbar(T,vel,uncer,ls='None',color='k',capsize=3,capthick=2,elinewidth=2,label='Observation')
        plt.plot(T,self.initMod.forward(T),label='Initial')
        plt.plot(T,self.avgMod.forward(T),label='Avg accepted')
        plt.plot(T,self.minMod.forward(T),label='Min misfit')
        plt.legend()
        plt.title('Dispersion')
    def plotDistrib(self,inds='all'):
        if inds == 'all':
            inds = range(len(self.initMod._paras()))
        for ind in inds:
            plt.figure()
            y = self.Priparas[:,ind]
            _,bin_edges = np.histogram(y,bins=30)
            y = self.MCparas[self.accFinal,ind]
            plt.hist(y,bins=bin_edges,weights=np.ones_like(y)/float(len(y)))
            y = self.Priparas[:,ind]
            plt.hist(y,bins=bin_edges,weights=np.ones_like(y)/float(len(y)),
                        fill=False,ec='k',rwidth=1.0)
            plt.title(f'N = {self.accFinal.sum()}/{len(self.accFinal)}')
    def plotVsProfile(self,allAccepted=False):
        fig = self.initMod.plotProfile(label='Initial')
        mod = self.avgMod.copy()
        indFinAcc = np.where(self.accFinal)[0]
        for i in range(min(len(indFinAcc),(self.N if allAccepted else 2000))):
            ind = indFinAcc[i] if allAccepted else random.choice(indFinAcc)
            mod.updateVars(self.MCparas[ind,:])
            mod.plotProfile(fig=fig,color='grey',lw=0.1)
        self.avgMod.plotProfile(fig=fig,label='Avg')
        self.minMod.plotProfile(fig=fig,label='Min')
        plt.xlim(3.8,4.8)
        plt.legend()
        return fig
    def plotVsProfileGrid(self,allAccepted=False,ax=None):
        fig = self.initMod.plotProfileGrid(label='Initial',ax=ax)
        if ax is None:
            fig.set_figheight(8.4);fig.set_figwidth(5)
        mod = self.avgMod.copy()
        indFinAcc = np.where(self.accFinal)[0]
        for i in range(min(len(indFinAcc),(self.N if allAccepted else 2000))):
            ind = indFinAcc[i] if allAccepted else random.choice(indFinAcc)
            mod.updateVars(self.MCparas[ind,:])
            mod.plotProfileGrid(fig=fig,color='grey',ax=ax,lw=0.1)
        self.avgMod.plotProfileGrid(fig=fig,label='Avg',ax=ax)
        self.minMod.plotProfileGrid(fig=fig,label='Min',ax=ax)
        plt.xlim(3.0,4.8)
        plt.legend()
        return fig
    def plotVsProfileStd(self):
        indFinAcc = np.where(self.accFinal)[0]
        zdeps = np.linspace(0,199,300)
        allVs = np.zeros([len(zdeps),len(indFinAcc)])

        mod = self.avgMod.copy()
        for i,ind in enumerate(indFinAcc):
            mod.updateVars(self.MCparas[ind,:])
            profile = ProfileGrid(mod.genProfileGrid())
            allVs[:,i] = profile.value(zdeps)
        std = allVs.std(axis=1)

        fig = self.initMod.plotProfileGrid(label='Initial',alpha=0.1)
        fig.set_figheight(8.4);fig.set_figwidth(5)
        avgProfile = ProfileGrid(self.avgMod.genProfileGrid()).value(zdeps)
        plt.fill_betweenx(zdeps,avgProfile+std,avgProfile-std,facecolor='grey',alpha=0.4)
        self.avgMod.plotProfileGrid(fig=fig,label='Avg')
        # self.minMod.plotProfileGrid(fig=fig,label='Min')
        plt.xlim(3.0,4.8)
        plt.legend()
    def plotCheck(self):
        plt.figure()
        ksquare = self.misfits**2*len(self.obs['T'])
        plt.plot(ksquare)
        ind = np.where(self.accepts>0.1)[0]
        plt.plot(ind,ksquare[ind],'or')
        plt.plot([0,self.N],[self.thres**2*len(self.obs['T'])]*2,'--g')
        pass
        ''' plot likelihood '''
        # plt.figure()
        # plt.plot(self.Ls)
        # I = self.accepts==1
        # plt.plot(np.arange(self.N)[I],self.Ls[I])

        ''' plot misfit '''
        # plt.figure()
        # plt.plot(self.misfits)
        # I = self.accepts==1
        # plt.plot(np.arange(self.N)[I],self.misfits[I])

class ProfileGrid():
    def __init__(self,profileIn) -> None:
        # zdepth,vs,ltype = profileIn
        self.zdepth = np.array(profileIn[0])
        self.vs     = np.array(profileIn[1])
        # self.layers = np.array(layers.copy())
        # self.Hs     = np.array(Hs.copy())
        # self.z0     = -max(topo,0)
        self.ltype  = np.array(profileIn[2])
    def _type(self,z):
        tmp = np.where(self.zdepth >= z)
        if len(tmp) == 0:
            raise ValueError()
        return self.ltype[tmp[0][0]]
    def value(self,z):
        return np.interp(z,self.zdepth,self.vs,left=np.nan,right=np.nan)
    def type(self,z):
        try:
            return np.array([self._type(i) for i in z])
        except:
            return self._type(z)
    def resample(self,layerDict):
        Z,Vs = [],[]
        for l,n in layerDict.items():
            if l not in self.ltype:
                Z.extend([np.nan for _ in range(n)])
                Vs.extend([np.nan for _ in range(n)])
                continue
            z1,vs1 = self.zdepth[self.ltype==l],self.vs[self.ltype==l]
            z2 = np.linspace(z1[0],z1[-1],n)
            vs2 = np.interp(z2,z1,vs1)
            Z.extend(z2);Vs.extend(vs2)
        return np.array(Z),np.array(Vs)
    def moho(self):
        return self.zdepth[np.where(self.ltype == 'crust')[0][-1]]


def mapSmooth(lons,lats,z,tension=0.0, width=50.):
    lons = lons.round(decimals=4)
    lats = lats.round(decimals=4)
    tmpFname = f'tmp{randString(10)}'
    XX,YY = np.meshgrid(lons,lats)
    dlon,dlat = lons[1]-lons[0],lats[1]-lats[0]
    savetxt(f'{tmpFname}.xyz',XX.flatten(),YY.flatten(),z.flatten())
    with open(f'{tmpFname}.bash','w+') as f:
        REG     = f'-R{lons[0]:.2f}/{lons[-1]:.2f}/{lats[0]:.2f}/{lats[-1]:.2f}'
        f.writelines(f'gmt gmtset MAP_FRAME_TYPE fancy \n')
        f.writelines(f'gmt surface {tmpFname}.xyz -T{tension} -G{tmpFname}.grd -I{dlon:.2f}/{dlat:.2f} {REG} \n')
        f.writelines(f'gmt grdfilter {tmpFname}.grd -D4 -Fg{width} -G{tmpFname}_Smooth.grd {REG} \n')
    os.system(f'bash {tmpFname}.bash')
    from netCDF4 import Dataset
    with Dataset(f'{tmpFname}_Smooth.grd') as dset:
        zSmooth = dset['z'][()]
    os.system(f'rm {tmpFname}* gmt.conf gmt.history')
    return zSmooth
class Model3D(object):
    ''' to avoid bugs in gmt smooth, start/end of lons/lats should be integer '''
    def __init__(self,lons=[],lats=[]) -> None:
        self.lons = np.array(lons)
        self.lats = np.array(lats)
        self.mods       = [ [None]*len(lons) for _ in range(len(lats))]
        self._mods_init = [ [None]*len(lons) for _ in range(len(lats))]
        self._profiles  = [ [None]*len(lons) for _ in range(len(lats))]
        self.misfits    = [ [None]*len(lons) for _ in range(len(lats))]
        self.disps      = [ [None]*len(lons) for _ in range(len(lats))]
    @property
    def dlon(self):
        return self.lons[1] - self.lons[0]
    @property
    def dlat(self):
        return self.lats[1] - self.lats[0]
    @property
    def XX(self):
        return np.meshgrid(self.lons,self.lats)[0]
    @property
    def YY(self):
        return np.meshgrid(self.lons,self.lats)[1]
    def _findInd(self,lon,lat):
        j = np.where(abs(self.lons-lon)<=self.dlon/4)[0][0]
        i = np.where(abs(self.lats-lat)<=self.dlat/4)[0][0]
        return i,j
    @property
    def mask(self):
        m,n = len(self.lats),len(self.lons)
        mask = np.ones((m,n),dtype=bool)
        for i in range(m):
            for j in range(n):
                mask[i,j] = (self._profiles[i][j] is None)
        return mask

    def loadInvDir(self,invDir='example-Cascadia'):
        ptlons,ptlats = [],[]
        if len(self.lons) == 0:
            try: # check format and initialize
                for npzfile in glob.glob(f'{invDir}/*.npz'):
                    ptlon,ptlat = npzfile.split('/')[-1][:-4].split('_')[:]
                    ptlons.append(ptlon); ptlats.append(ptlat)
                ptlons=np.array([float(a) for a in set(ptlons)]); ptlons.sort(); dlon = min(np.diff(ptlons))
                ptlats=np.array([float(a) for a in set(ptlats)]); ptlats.sort(); dlat = min(np.diff(ptlats))
                lons = np.arange(np.floor(ptlons[0]),np.ceil(ptlons[-1])+dlon/2,dlon)
                lats = np.arange(np.floor(ptlats[0]),np.ceil(ptlats[-1])+dlat/2,dlat)
                self.__init__(lons,lats)
            except:
                raise TypeError('Could not take lat/lon, please make sure the format is invDir/lon_lat.npz')
        for npzfile in glob.glob(f'{invDir}/*.npz'):
            ptlon,ptlat = npzfile.split('/')[-1][:-4].split('_')[:]
            ptlon,ptlat = float(ptlon),float(ptlat)
            self.addInvPoint(ptlon,ptlat,PostPoint(npzfile))
    def addInvPoint(self,lon,lat,postpoint:PostPoint):
        print(f'Add point {lon:.1f}_{lat:.1f}')
        i,j = self._findInd(lon,lat)
        self.mods[i][j]     = postpoint.avgMod.copy()
        self._mods_init[i][j] = postpoint.initMod.copy()
        self._profiles[i][j] = ProfileGrid(postpoint.avgMod.genProfileGrid())
        self.misfits[i][j]  = postpoint.avgMod.misfit
        self.disps[i][j]    = {'T':postpoint.obs['T'],
                               'pvelo':postpoint.obs['c'],
                               'pvelp':postpoint.avgMod.forward(postpoint.obs['T']),
                               'uncer':postpoint.obs['uncer']}

    def vsProfile(self,z,lat,lon):
        lon = lon + 360*(lon < 0)
        if (lon-self.lons[0]) * (lon-self.lons[-1]) > 0:
            # raise ValueError('Longitude is out of range!')
            return np.nan
        if (lat-self.lats[0]) * (lat-self.lats[-1]) > 0:
            # raise ValueError('Latitude is out of range!')
            return np.nan
        i = np.where(self.lons-lon>=0)[0][0]
        j = np.where(self.lats-lat>=0)[0][0]
        try:
            p0 = self._profiles[j-1,i-1].value(z)
            p1 = self._profiles[j,i-1].value(z)
            p2 = self._profiles[j-1,i].value(z)
            p3 = self._profiles[j,i].value(z)
            Dx = self.lons[i] - self.lons[i-1]
            Dy = self.lats[j] - self.lats[j-1]
            dx = lon - self.lons[i-1]
            dy = lat - self.lats[j-1]
            p = p0+(p1-p0)*dy/Dy+(p2-p0)*dx/Dx+(p0+p3-p1-p2)*dx*dy/Dx/Dy
            return p
        except AttributeError:
            return np.nan*np.ones(z.shape)
    def topo(self,lat,lon):
        lon = lon + 360*(lon < 0)
        if (lon-self.lons[0]) * (lon-self.lons[-1]) > 0:
            # raise ValueError('Longitude is out of range!')
            return np.nan
        if (lat-self.lats[0]) * (lat-self.lats[-1]) > 0:
            # raise ValueError('Latitude is out of range!')
            return np.nan
        i = np.where(self.lons-lon>=0)[0][0]
        j = np.where(self.lats-lat>=0)[0][0]
        try:
            p0 = self.mods[j-1,i-1].info['topo']
            p1 = self.mods[j,i-1].info['topo']
            p2 = self.mods[j-1,i].info['topo']
            p3 = self.mods[j,i].info['topo']
            Dx = self.lons[i] - self.lons[i-1]
            Dy = self.lats[j] - self.lats[j-1]
            dx = lon - self.lons[i-1]
            dy = lat - self.lats[j-1]
            p = p0+(p1-p0)*dy/Dy+(p2-p0)*dx/Dx+(p0+p3-p1-p2)*dx*dy/Dx/Dy
            return p
        except AttributeError:
            return np.nan
    def moho(self,lat,lon):
        lon = lon + 360*(lon < 0)
        if (lon-self.lons[0]) * (lon-self.lons[-1]) > 0:
            # raise ValueError('Longitude is out of range!')
            return np.nan
        if (lat-self.lats[0]) * (lat-self.lats[-1]) > 0:
            # raise ValueError('Latitude is out of range!')
            return np.nan
        i = np.where(self.lons-lon>=0)[0][0]
        j = np.where(self.lats-lat>=0)[0][0]
        try:
            p0 = self._profiles[j-1,i-1].moho()
            p1 = self._profiles[j,i-1].moho()
            p2 = self._profiles[j-1,i].moho()
            p3 = self._profiles[j,i].moho()
            Dx = self.lons[i] - self.lons[i-1]
            Dy = self.lats[j] - self.lats[j-1]
            dx = lon - self.lons[i-1]
            dy = lat - self.lats[j-1]
            p = p0+(p1-p0)*dy/Dy+(p2-p0)*dx/Dx+(p0+p3-p1-p2)*dx*dy/Dx/Dy
            return p
        except AttributeError:
            return np.nan

    def smoothGrid(self,width=50):
        ''' To combine and smooth areas with different model settings '''
        m,n = len(self.lats),len(self.lons)
        layerDict = {'water':2,'sediment':6,'crust':30,'mantle':200}
        ltypes = []
        for k,v in layerDict.items():
            ltypes.extend([k]*v)
        l = np.sum(list(layerDict.values()))
        zMat = np.zeros((m,n,l))
        vsMat = np.zeros((m,n,l))
        for i in range(m):
            for j in range(n):
                if not self.mask[i,j]:
                    z,vs = self._profiles[i][j].resample(layerDict)
                else:
                    z,vs = np.nan,np.nan
                zMat[i,j,:] = z
                vsMat[i,j,:] = vs
        zMatSmooth = zMat.copy()
        vsMatSmooth = vsMat.copy()
        print('smoothing')
        for k in range(l):
            print(f'{k+1}/{l}')
            zMatSmooth[:,:,k] = mapSmooth(self.lons,self.lats,zMat[:,:,k],width=width)
            vsMatSmooth[:,:,k] = mapSmooth(self.lons,self.lats,vsMat[:,:,k],width=width)
        for i in range(m):
            for j in range(n):
                if not self.mask[i,j]:
                    self._profiles[i][j] = ProfileGrid((zMatSmooth[i,j,:],vsMatSmooth[i,j,:],
                                                        ltypes))
    def smoothGrid_OLD(self,width=50):
        ''' To combine and smooth areas with different model settings '''
        def updateArray(a,b):
            if len(set(a)) != len(a) or len(set(b)) != len(b):
                raise ValueError(f'Error: repeat element found: {a} | {b}!!')
            a,b = list(a),list(b)
            i,j = 0,0
            while i <= len(a) and j<len(b):
                if b[j] not in a[i:]:
                    a.insert(i,b[j])
                    i += 1
                else:
                    i += a[i:].index(b[j])+1
                j+=1
            return np.array(a)
        lDict = {'water':1,'sediment':5,'crust':30,'mantle':200}
        m,n = len(self.lats),len(self.lons)
        allLayerTypes = np.array([])
        for i in range(m):
            for j in range(n):
                if self.mods[i][j] is not None:
                    allLayerTypes = updateArray(allLayerTypes,[l.type for l in self.mods[i][j].layers])
        nFines = np.array([lDict[layerType] for layerType in allLayerTypes])
        zMat   = np.ones((nFines.sum(),m,n))
        vsMat  = np.ones((nFines.sum(),m,n))
        ltypeSmooth = []
        for layerType in allLayerTypes:
            ltypeSmooth.extend([layerType]*lDict[layerType])
        def reSampleProfile(inProfileGrid):
            z,vs,ltype = inProfileGrid
            zNew,vsNew = np.zeros(nFines.sum()),np.zeros(nFines.sum())
            ltypeNew = []
            k,kNew = 0,0
            for layerType,nFine in zip(allLayerTypes,nFines):
                dk,dkNew = (np.array(ltype) == layerType).sum(),nFine
                zNew[kNew:kNew+dkNew] = np.linspace(z[k],z[k+max(0,dk-1)],dkNew)
                # print(z[k:k+dk])
                # print(zNew[kNew:kNew+dkNew])
                vsNew[kNew:kNew+dkNew] = np.interp(zNew[kNew:kNew+dkNew],z[k:k+max(dk,1)],vs[k:k+max(dk,1)])
                ltypeNew.extend([layerType]*dkNew)
                k += dk
                kNew += dkNew
            return zNew,vsNew,ltypeNew
        print('reSample')
        for i in range(m):
            for j in range(n):
                if self.mods[i][j] is not None:
                    z,vs,_ = reSampleProfile(self.mods[i][j].genProfileGrid())
                else:
                    z,vs,_ = np.nan*np.ones(nFines.sum()),np.nan*np.ones(nFines.sum()),np.nan*np.ones(nFines.sum())
                zMat[:,i,j] = z[:]
                vsMat[:,i,j] = vs[:]
        zMatSmooth = zMat.copy()
        vsMatSmooth = vsMat.copy()
        print('smoothing')
        for k in range(nFines.sum()):
            print(f'{k+1}/{nFines.sum()}')
            zMatSmooth[k,:,:] = mapSmooth(self.lons,self.lats,zMat[k,:,:],width=width)
            vsMatSmooth[k,:,:] = mapSmooth(self.lons,self.lats,vsMat[k,:,:],width=width)
        for i in range(m):
            for j in range(n):
                if not self.mask[i,j]:
                    self._profiles[i][j] = ProfileGrid((zMatSmooth[:,i,j],vsMatSmooth[:,i,j],
                                                        ltypeSmooth))
    def smooth(self,width=50):
        m,n = len(self.lats),len(self.lons)
        paras = [ [None]*n for _ in range(m)]
        mask = self.mask
        Nparas = len(self.mods[np.where(~mask)[0][0]][np.where(~mask)[1][0]].paras())
        for i in range(m):
            for j in range(n):
                if not mask[i,j]:
                    paras[i][j] = self.mods[i][j].paras()
                else:
                    paras[i][j] = [np.nan]*Nparas
        paras = np.array(paras)
        for i in range(paras.shape[-1]):
            paras[:,:,i] = mapSmooth(self.lons,self.lats,paras[:,:,i],width=width)
        for i in range(m):
            for j in range(n):
                if not mask[i,j]:
                    self.mods[i][j].updateVars(paras[i][j])
                    self._profiles[i][j] = ProfileGrid(self.mods[i][j].genProfileGrid())

    def write(self,fname):
        np.savez_compressed(fname,lons=self.lons,lats=self.lats,profiles=self._profiles,
                            misfits=self.misfits,disps=self.disps,
                            mods=self.mods,modsInit=self._mods_init)
    def load(self,fname):
        tmp = np.load(fname,allow_pickle=True)
        self.lons = tmp['lons'][()]
        self.lats = tmp['lats'][()]
        self._profiles  = tmp['profiles'][()]
        self.misfits    = tmp['misfits'][()]
        self.disps      = tmp['disps'][()]
        self.mods       = tmp['mods'][()]
        self._mods_init = tmp['modsInit'][()]


    ''' figure plotting '''
    def mapview(self,depth):
        mask = self.mask
        vsMap = np.ma.masked_array(np.zeros(mask.shape),mask=mask)
        m,n = len(self.lats),len(self.lons)
        for i in range(m):
            for j in range(n):
                if not mask[i,j]:
                    vsMap[i,j] = self._profiles[i][j].value(depth)
        return vsMap
    def section(self,lon1,lat1,lon2,lat2):
        # lon1,lat1,lon2,lat2 = -131+360,46,-125+360,43.8
        geoDict = Geodesic.WGS84.Inverse(lat1,lon1,lat2,lon2)
        # lats,lons = [],[]
        x = np.linspace(0,geoDict['s12'],301)/1000
        y = np.linspace(0,200-0.01,201)
        z = np.zeros((len(y),len(x)))
        moho = np.zeros(len(x))
        topo = np.zeros(len(x))
        for i,d in enumerate(x*1000):
            tmp = Geodesic.WGS84.Direct(lat1,lon1,geoDict['azi1'],d)
            # lats.append(tmp['lat2']);lons.append(tmp['lon2'])
            z[:,i] = self.vsProfile(y,tmp['lat2'],tmp['lon2'])
            moho[i]= self.moho(tmp['lat2'],tmp['lon2'])
            topo[i]= self.topo(tmp['lat2'],tmp['lon2'])
        z = np.ma.masked_array(z,np.isnan(z))
        if abs(lon1-lon2)<0.01:
            x = np.linspace(lat1,lat2,301)
        elif abs(lat1-lat2)<0.01:
            x = np.linspace(lon1,lon2,301)
        XX,YY = np.meshgrid(x,y)
        return XX,YY,z,moho,topo
    def plotSection_OLD(self,lon1,lat1,lon2,lat2,vmin=4.1,vmax=4.4,cmap=cvcpt,maxD=200,shading='gouraud'):
        XX,YY,Z,moho = self.section(lon1,lat1,lon2,lat2)
        plt.figure(figsize=[8,4.8])
        # f = interpolate.interp2d(XX[0,:],YY[:,0],Z,kind='cubic')
        # newX = np.linspace(XX[0,0],XX[0,-1],300)
        # newY = np.linspace(YY[0,0],YY[-1,0],300)
        # newZ = f(newX,newY)
        # XX,YY = np.meshgrid(newX,newY)
        plt.pcolormesh(XX,YY,Z,shading=shading,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.plot(XX[0,:],moho,'k',lw=4)
        plt.ylim(0,maxD)
        plt.colorbar(orientation='horizontal',fraction=0.1,aspect=50,pad=0.08)
        plt.gca().invert_yaxis()

    def plotSection(self,lon1,lat1,lon2,lat2,vCrust=[3.0,4.0],vMantle=[4.0,4.5],cmap=cvcpt,
                    maxD=200,shading='gouraud',title=None):
        from Triforce.customPlot import addAxes,addCAxes
        XX,YY,Z,moho,topo = self.section(lon1,lat1,lon2,lat2)
        mask = ~(YY <= np.tile(moho,(YY.shape[0],1)))
        Z_crust = np.ma.masked_array(Z,mask=mask)
        fig = plt.figure(figsize=[12,5])
        ax2 = addAxes([0,0.18,1,0.65])
        bbox = ax2.get_position()
        x,y,w,h = bbox.x0,bbox.y0,bbox.width,bbox.height
        ax1 = plt.axes([x,y+h+0.05*h,w,h/6],sharex=ax2)
        ax1.axes.xaxis.set_visible(False)
        cax1 = addCAxes(ax2,location='bottom',size=0.05,pad=0.06)
        cax2 = addCAxes(ax2,location='bottom',size=0.05,pad=0.16)

        topo_plot = topo.copy()
        topo_plot[topo_plot<0] /= 2
        topo_min,topo_max = np.nanmin(topo)*1000,np.nanmax(topo)*1000
        ax1.plot(XX[0,:], topo_plot*1000., 'k', lw=3)
        ax1.fill_between(XX[0,:], -10000, topo_plot*1000., facecolor='grey')
        ax1.fill_between(XX[0,:], 0, topo_plot*1000., where=topo_plot<0, facecolor='#d4f1f9')
        ax1.set_yticks([-2000/2,0,1000])
        ax1.set_yticklabels(['-2000','0','1000'])
        ax1.set_ylim(np.nanmin(topo_plot)*1000-100, np.nanmax(topo_plot)*1000.+300.)
        # ax1.plot(XX[0,:],np.zeros(XX.shape[1]),'--k',lw=0.5)
        
        

        ax1.set_title(title)

        plt.axes(ax2)
        plt.pcolormesh(XX,YY,Z,shading=shading,cmap=cmap,vmin=vMantle[0],vmax=vMantle[1])
        plt.colorbar(cax=cax2,orientation='horizontal')
        plt.pcolormesh(XX,YY,Z_crust,shading=shading,cmap=cmap,vmin=vCrust[0],vmax=vCrust[1])
        plt.colorbar(cax=cax1,orientation='horizontal')
        plt.plot(XX[0,:],moho,'k',lw=4)
        plt.ylim(0,maxD)
        plt.gca().invert_yaxis()

        return fig,ax1,ax2

    def _plotBasemap(self,loc='Cascadia',ax=None):
        from Triforce.customPlot import plotLocalBase
        if loc=='Cascadia':
            minlon,maxlon,minlat,maxlat,dlon,dlat = -132,-121,39,50,2,3
        elif loc=='auto':
            minlon,maxlon,minlat,maxlat = self.lons[0],self.lons[-1],self.lats[0],self.lats[-1]
            dlat,dlon = (maxlat-minlat)//5,(maxlon-minlon)//3
        else:
            minlon,maxlon,minlat,maxlat,dlon,dlat = loc
        fig,m = plotLocalBase(minlon,maxlon,minlat,maxlat,dlat=dlat,dlon=dlon,resolution='l',ax=ax)
        m.readshapefile('/home/ayu/Projects/Cascadia/Models/Plates/PB2002_boundaries','PB2002_boundaries',
            linewidth=2.0,color='orange')
        return fig,m
    def plotMapView(self,mapTerm,loc='Cascadia',vmin=4.1,vmax=4.4,cmap=None):
        fig,m = self._plotBasemap(loc=loc)
        if mapTerm == 'misfit':
            cmap = plt.cm.gnuplot_r if cmap is None else cmap
            norm = mpl.colors.BoundaryNorm(np.linspace(0.5,3,6), cmap.N)
            misfits = np.array(self.misfits)
            misfits[misfits==None] = np.nan
            misfits = np.ma.masked_array(misfits.astype(float),mask=self.mask,fill_value=0)
            # misfits -= 0.10
            m.pcolormesh(self.XX-360*(self.XX[0,0]>180),self.YY,misfits,shading='gouraud',cmap=cmap,latlon=True,norm=norm)
            plt.title(f'Misfit')
            plt.colorbar(location='bottom',fraction=0.012,aspect=50)
        else:
            if cmap is None:
                cmap = cvcpt
            vsMap = self.mapview(mapTerm)
            m.pcolormesh(self.XX-360*(self.XX[0,0]>180),self.YY,vsMap,shading='gouraud',cmap=cmap,vmin=vmin,vmax=vmax,latlon=True)
            plt.title(f'Depth: {mapTerm} km')
            plt.colorbar(location='bottom',fraction=0.012,aspect=50)
        return fig,m

    def checkLayerThick(self):
        mask = self.mask
        hCrust      =  np.ma.masked_array(np.zeros(mask.shape),mask=mask)
        hCrust0     =  np.ma.masked_array(np.zeros(mask.shape),mask=mask)
        hCrustBias  =  np.ma.masked_array(np.zeros(mask.shape),mask=mask)
        hSed        =  np.ma.masked_array(np.zeros(mask.shape),mask=mask)
        hSed0       =  np.ma.masked_array(np.zeros(mask.shape),mask=mask)
        hSedBias    =  np.ma.masked_array(np.zeros(mask.shape),mask=mask)
        m,n = len(self.lats),len(self.lons)
        for i in range(m):
            for j in range(n):
                if not mask[i,j]:
                    mod,mod0 = self.mods[i][j],self._mods_init[i][j]
                    indCrust = [l.type for l in mod.layers].index('crust')
                    indSed   = [l.type for l in mod.layers].index('sediment')
                    hCrust[i,j]     = mod.layers[indCrust].H
                    hCrust0[i,j]    = mod0.layers[indCrust].H
                    hSed[i,j]       = mod.layers[indSed].H
                    hSed0[i,j]      = mod0.layers[indSed].H
                    hCrustBias[i,j] = (hCrust[i,j] - hCrust0[i,j])/(hCrust0[i,j])*100
                    if mod.info['label'][:6] == 'Hongda' and hSed0[i,j]<1.0:
                        hSedBias[i,j]   = (hSed[i,j]-hSed0[i,j])/1.0*100
                    else:
                        hSedBias[i,j]   = (hSed[i,j]-hSed0[i,j])/hSed0[i,j]*100

        fig,m = self._plotBasemap(loc='auto')
        m.pcolormesh(self.XX-360,self.YY,hCrust,shading='gouraud',cmap=cvcpt,latlon=True)
        plt.title(f'Crust Thickness')
        plt.colorbar(location='bottom',fraction=0.012,aspect=50)
        plt.savefig('CrustH.png')

        fig,m = self._plotBasemap(loc='auto')
        m.pcolormesh(self.XX-360,self.YY,hCrustBias,shading='gouraud',cmap=cvcpt,latlon=True)
        plt.title(f'Crust Thickness Difference from Initial Model (%)')
        plt.colorbar(location='bottom',fraction=0.012,aspect=50)
        plt.clim(-25,25)
        plt.savefig('CrustH-Change.png')

        fig,m = self._plotBasemap(loc='auto')
        m.pcolormesh(self.XX-360,self.YY,hSed,shading='gouraud',cmap=cvcpt,latlon=True)
        plt.title(f'Sediment Thickness')
        plt.colorbar(location='bottom',fraction=0.012,aspect=50)
        plt.savefig('SedimentH.png')

        fig,m = self._plotBasemap(loc='auto')
        m.pcolormesh(self.XX-360,self.YY,hSedBias,shading='gouraud',cmap=cvcpt,latlon=True)
        plt.title(f'Sediment Thickness Difference from Initial Model (%)')
        plt.colorbar(location='bottom',fraction=0.012,aspect=50)
        plt.clim(-100,100)
        plt.savefig('SedimentH-Change.png')
    def checkPhaseVelocity(self,pers='all',savefig=False):
        from Triforce.customPlot import addCAxes
        vminmax = {
                '010s':(3.0,4.0),
                '012s':(3.0,4.0),
                '014s':(3.0,4.0),
                '016s':(3.0,4.0),
                '018s':(3.1,4.0),
                '020s':(3.3,4.0),
                '022s':(3.3,4.0),
                '024s':(3.3,4.0),
                '026s':(3.3,4.0),
                '028s':(3.3,4.0),
                '030s':(3.4,4.0),
                '032s':(3.5,4.0),
                '036s':(3.6,4.0),
                '040s':(3.6,4.0),
                '050s':(3.6,4.0),
                '060s':(3.7,4.0),
                '070s':(3.7,4.0),
                '080s':(3.8,4.05)
                }
        if pers == 'all':
            pers = []
            for disp in self.disps.reshape(-1):
                if disp is not None:
                    pers.extend(disp['T'])
            pers = list(set(pers)); pers.sort()
        disps = {} 
        for iper,per in enumerate(pers):
            print(per)
            Tstr = f'{int(per):03d}s'
            vmin,vmax = vminmax[Tstr] if Tstr in vminmax.keys() else (None,None)
            m,n = self.XX.shape
            pvelo = np.ma.masked_array(np.zeros(self.XX.shape),mask=self.mask)
            pvelp = np.ma.masked_array(np.zeros(self.XX.shape),mask=self.mask)
            uncer = np.ma.masked_array(np.zeros(self.XX.shape),mask=self.mask)
            for i in range(m):
                for j in range(n):
                    if self.mask[i,j]:
                        continue
                    disp = self.disps[i][j]
                    ind = disp['T'].index(int(Tstr[:-1]))
                    pvelo[i,j] = disp['pvelo'][ind]
                    pvelp[i,j] = disp['pvelp'][ind]
                    uncer[i,j] = disp['uncer'][ind]
            disps[per] = {'pvelo':pvelo,'pvelp':pvelp}

            fig, axes = plt.subplots(1,3,figsize=[12,4.8])
            plt.subplots_adjust(wspace=0.25,hspace=0.3,left=0.08,right=0.92,bottom=0.15)
            _,m1 = self._plotBasemap(loc='auto',ax=axes[0])
            _,m2 = self._plotBasemap(loc='auto',ax=axes[1])
            _,m3 = self._plotBasemap(loc='auto',ax=axes[2])

            XX,YY = self.XX-360,self.YY
            im = m1.pcolormesh(XX,YY,pvelo,latlon=True,cmap=cvcpt,shading='gouraud',ax=axes[0],vmin=vmin,vmax=vmax)
            cax = addCAxes(axes[0],location='bottom',size=0.03,pad=0.20);plt.colorbar(im,cax=cax,orientation='horizontal')
            im = m2.pcolormesh(XX,YY,pvelp,latlon=True,cmap=cvcpt,shading='gouraud',ax=axes[1],vmin=vmin,vmax=vmax)
            cax = addCAxes(axes[1],location='bottom',size=0.03,pad=0.20);plt.colorbar(im,cax=cax,orientation='horizontal')
            im = m3.pcolormesh(XX,YY,(pvelp-pvelo)/uncer,latlon=True,cmap=cvcpt,shading='gouraud',ax=axes[2],vmin=-3,vmax=3)
            cax = addCAxes(axes[2],location='bottom',size=0.03,pad=0.20);plt.colorbar(im,cax=cax,orientation='horizontal')
            axes[0].set_title(f'Observation T={int(per):02d}s')
            axes[1].set_title(f'Prediction T={int(per):02d}s')
            axes[2].set_title(f'Pred-Obs (normed by uncer)')

            if savefig:
                plt.savefig(f'PhaseVel-{int(per):02d}s.png')
                plt.close()

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)


if __name__ == '__main__':
    pass
    # # test RandVar
    # a = RandVar(3.4,3,4,0.2); a.show()
    # b = a.perturb(); b.show()
    # c = a.reset('uniform'); c.show()

    # B-spline test
    # bspl = BsplBasis(np.linspace(0,200,100),5,4);bspl.plot()
    # plt.figure();plt.plot(np.linspace(0,1,bspl.n),bspl*[4.2,4.3,3.9,4.6,4.5])

    # Setting test
    # setting=Setting(); setting.load('setting-Hongda.yml')
    # setting.updateInfo(topo=-10,sedthk=1)
    # setting._updateVars([6,5,4,3,2,1])

    # SurfLayer test
    # lay1 = SurfLayer('mantle',setting['mantle']);print(lay1.vs)
    # lay2 = lay1.perturb();print(lay2.vs)
    # lay3 = lay2.reset('uniform');print(lay3.vs)
    # print(lay1.paraDict)

    # Model1D test
    # mod1 = Model1D()
    # mod1.loadSetting('setting-Hongda.yml');mod1.show()
    # print(dict(mod1.toSetting()))
    # h,vs,vp,rho,qs,qp = mod1.genProfile()
    # mod2 = mod1.perturb(); mod2.show()
    # mod3 = mod2.reset('uniform'); mod3.show()
    # plt.figure();plt.plot(np.linspace(5,80,100),mod1.forward(np.linspace(5,80,100)))
    # print(mod1.paras())
    # print(mod1.isgood())

    # inversion test
    random.seed(36)
    periods = np.array([10,12,14,16,18,20,24,28,32,36,40,50,60,70,80])
    setting = Setting(); setting.load('setting-Hongda.yml'); setting.updateInfo(topo=-4,sedthk=0.6)
    mod1 = Model1D(); mod1.loadSetting(setting); mod2 = mod1.reset('uniform') # set true model
    mod2 = mod1.copy()
    for _ in range(20):
        mod2 = mod2.perturb()
    p = Point(setting,topo=-4,sedthk=0.6,
              periods=periods,vels=mod2.forward(periods),uncers=[0.01]*len(periods))
    p.MCinvMP(runN=50000,step4uwalk=1000,nprocess=26)
    p.MCinvMP('MCtest_priori',runN=50000,step4uwalk=1000,nprocess=26,priori=True)
    postp = PostPoint('MCtest/test.npz','MCtest_priori/test.npz')
    postp.plotDisp()
    fig = postp.plotVsProfileGrid()
    mod2.plotProfileGrid(fig=fig,label='True')
    postp.plotDistrib()
    # timeStamp = time.time()
    # for _ in range(10000):
    #     postp.avgMod.forward(periods)
    # print(f'{time.time()-timeStamp:.3f}s')

    # inversion test real data
    # id     = '233.0_43.2'
    # T = [ 10.,  12.,  14.,  16.,  18.,  20.,  22.,  24.,  26.,  28.,  30.,
    #     32.,  36.,  40.,  50.,  60.,  70.,  80.]
    # vels = [ 3.66170009,  3.72857888,  3.75951126,  3.76266499,  3.77191581,
    #     3.7685344 ,  3.77129248,  3.77428902,  3.77529921,  3.78802274,
    #     3.79433664,  3.80568807,  3.82146285,  3.8505667 ,  3.84643676,
    #     3.87612961,  3.91444643,  3.96543979]
    # uncers = [ 0.01754326,  0.01164089,  0.00903466,  0.00797875,  0.00716722,
    #     0.00713235,  0.00744013,  0.00770071,  0.00797466,  0.00956988,
    #     0.01142398,  0.00890576,  0.00949308,  0.01012225,  0.01201   ,
    #     0.01743369,  0.01614607,  0.01649115]
    # topo = -3.068602323532039
    # sedthk = 0.22750000655653985
    # p = Point('setting-Hongda.yml',topo=-4,sedthk=0.6,periods=T,vels= vels,uncers=uncers)
    # p.MCinvMP(id=id,runN=50000,step4uwalk=1000,nprocess=26)









    # invModel = Model3D()
    # invModel.loadInvDir('example-Cascadia/inv')
    # invModel.write('example-Cascadia/inv3D.npz')
    # invModel.load('example-Cascadia/inv3D.npz')
    # invModel.smooth(width=80)
    # invModel.write('example-Cascadia/inv3D-smooth.npz')

    # invModel = Model3D()
    # invModel.load('example-Cascadia/inv3D-smooth.npz')

    # invModel.plotSection(-131+360,47,-125+360,47) # I-I'
    # invModel.plotSection(-131+360,46,-125+360,46) # J-J'
    # invModel.plotSection(-131+360,45,-125+360,45) # K-K'
    # invModel.plotSection(-131+360,42,-125+360,42) # L-L'

    # invModel.plotMapView(14,vmin=4.1,vmax=4.8)
    # invModel.plotMapView(20,vmin=4.1,vmax=4.75)
    # invModel.plotMapView(30,vmin=4.1,vmax=4.7)
    # invModel.plotMapView(50,vmin=4.0,vmax=4.6)
    # invModel.plotMapView(100,vmin=4.08,vmax=4.48)
    # invModel.plotMapView(150,vmin=4.15,vmax=4.45)

    # invModel.plotMapView('misfit',vmin=4.1,vmax=4.8)


