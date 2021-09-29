from multiprocessing import Value
from os import times
import os,time
import multiprocessing as mp
import random,yaml,copy,time
from numpy.random.mtrand import f
import scipy.signal
import numpy as np
from Triforce.pltHead import *
import sys; sys.path.append('../')
import pySurfInv.fast_surf as fast_surf

def plotLayer(h,v,fig=None,label=None):
    if fig is None:
        fig = plt.figure(figsize=[5,7])
    else:
        plt.figure(fig.number)
    hNew = np.insert(np.repeat(np.cumsum(h),2)[:-1],0,0)
    vNew = np.repeat(v,2)
    plt.plot(vNew,hNew,label=label)
    return fig
def monoIncrease(a,eps=np.finfo(float).eps):
    return np.all(np.diff(a)>=0)
def randString(N):
    import random,string
    ''' Return a random string '''
    return ''.join([random.choice(string.ascii_letters + string.digits) for i in range(N)])
def calMantleQ(deps,vpvs,period=1,age=4.0):
    """ get Q value for mantle layer, follw Eq(4) from Ye (2013)
        Calculate Q value for 20 sec period, Q doesn't change a lot with period
    """
    from scipy.special import erf
    Tm = 1315. # deep mantle temperature in celsius
    A = 30. # A value in Ye 2013 eq(4) 
    temps = Tm*erf(500*deps/np.sqrt(age*365*24*3600))+273.15 # depth dependent mantle temperature in Kelvin
    qs = A * (2*np.pi*1/period)**0.1 * np.exp(0.1*(2.5e5+3.2*9.8*deps*10)/(8.314472*temps))
    qp = 1./(4./3.*vpvs**(-2) / qs + (1-4./3.*vpvs**(-2))/57823.)
    return qs,qp
def calCrustQ(vpvs):
    qs = 350
    qp = 1./(4./3.*(vpvs)**(-2) / qs + (1-4./3.*(vpvs)**(-2))/57823.)
    return qs,qp

class ezDict(object):
    def __init__(self,d) -> None:
        self.dict = dict(d)
    def keys(self):
        return self.dict.keys()
    def values(self):
        return self.dict.values()
    def updataValue(self,path,v):
        a = self.dict
        for k in path[:-1]:
            a = a[k]
        a[path[-1]] = v
    def insert(self,ind,key,value):
        newDict = {}
        oldKeys = list(self.keys())
        oldVals = list(self.values())
        for i in len(self.keys()):
            if i == ind:
                newDict[key] = value
            newDict[oldKeys[i]] = oldVals[i]
        self.dict = newDict
    def delete(self,key):
        newDict = {}
        oldKeys = list(self.keys())
        oldVals = list(self.values())
        for oldKey,oldVal in zip(oldKeys,oldVals):
            if oldKey != key:
                newDict[oldKey] = oldVal
        self.dict = newDict
class settingDict(dict):
    def load(self,setting):
        self.clear()
        if type(setting) not in (dict,settingDict):
            with open(setting, 'r') as f:
                setting = yaml.load(f,Loader=yaml.FullLoader)
        setting = copy.deepcopy(setting)
        for k in setting:
            self[k] = setting[k]
    def updateVariable(self,newParas):
        i = 0
        for ltype in self.keys():
            for k in self[ltype].keys():
                if type(self[ltype][k]) is str:
                    continue
                if type(self[ltype][k][0]) is list:
                    for j in range(len(self[ltype][k])):
                        if self[ltype][k][j][1] in ('abs','rel'):
                            self[ltype][k][j][0] = newParas[i];i+=1
                else:
                    if self[ltype][k][1] in ('abs','rel'): 
                        self[ltype][k][0] = newParas[i];i+=1
    def updateInfo(self,infoDict):
        if infoDict is None:
            return -1
        ezdict = ezDict(self)
        try:
            waterDepth = max(-infoDict['topo'],0)
            if waterDepth > 0:
                if 'water' in self.keys():
                    ezdict.updataValue(['water','h',0],-infoDict['topo'])
                else:
                    ezdict.insert(0,'water',{'type':'water','h':[-infoDict['topo'],'fixed'],
                                'vp':[1.475,'fixed']})
            else:
                if 'water' in ezdict.keys():
                    ezdict.delete('water')
        except KeyError:
            pass
        try:
            ezdict.updataValue(['sediment','h',0],infoDict['sedthk'])
        except KeyError:
            pass
        self.load(ezdict.dict)
    def copy(self):
        return copy.deepcopy(self)
class bsplBasis(object):
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

class SurfDisp():
    '''Class to store surface dispersion curve

        Attributes:
            wtype:  wave type - Ray/Love
            ctype:  velocity type - Phase or Group
            vel:    velocity value
            per:    periods
        '''
    def __init__(self,per=[],vel=[],uncer=None,wtype='Ray',ctype='Phase'):
        self.wtype = wtype
        self.ctype = ctype
        self.per = np.array(per)
        self.vel = np.array(vel)
        if uncer is None:
            self.uncer = np.zeros(len(per))
        else:
            self.uncer = np.array(uncer)
    @property
    def N(self):
        '''Number of periods'''
        return len(self.per)
    def setDisp(self,per=[],vel=[],uncer=None,wtype=None,ctype=None):
        if wtype is None:
            wtype = self.wtype
        if ctype is None:
            ctype = self.ctype
        self.__init__(per=per,vel=vel,uncer=uncer,wtype=wtype,ctype=ctype)
    def __repr__(self):
        return '%s disp(%s): N of period = %d' % (self.wtype,self.ctype,self.N)
    def __str__(self):
        return '%s disp(%s): N of period = %d\nPeriod   = %s\nVelocity = %s' \
            % (self.wtype,self.ctype,self.N,self.per,self.vel)

class randomWalkFloat(float):
    def __new__(cls,v,vmin=None,vmax=None,step=None):
        return super().__new__(cls,v)
    def __init__(self,v,vmin=None,vmax=None,step=None) -> None:
        self.vmin,self.vmax,self.step = vmin,vmax,step
    def reset(self,resetype='origin'):
        if self.vmin == None:
            vNew = float(self)
        elif resetype == 'origin':
            vNew = (self.vmin+self.vmax)/2
        elif resetype == 'uniform':
            vNew = random.uniform(self.vmin,self.vmax)
        else:
            raise ValueError('Unknown reset type!')
        return randomWalkFloat(vNew,self.vmin,self.vmax,self.step)
    def perturb(self):
        if self.vmin == None:
            return randomWalkFloat(float(self),self.vmin,self.vmax,self.step)
        for i in range(1000):
            vNew = random.gauss(self,self.step)
            if vNew < self.vmax and vNew > self.vmin:
                break
            if i == 999:
                print(f'No valid perturb, uniform reset instead! '+
                      f'{self} {self.vmin} {self.vmax} {self.step}')
                return self.reset('uniform')
        return randomWalkFloat(vNew,self.vmin,self.vmax,self.step)
    def updateValue(self,v):
        return randomWalkFloat(v,vmin=self.vmin,vmax=self.vmax,step=self.step)

def genRWFList(inList):
    outList = []
    if type(inList[0]) is not list:
         inList = [inList]
    for i in range(len(inList)):
        try:
            v,vtype,lim,step = inList[i]
        except ValueError:
            v,vtype = inList[i]
        if vtype == 'rel':
            vmin,vmax = v*(1-lim/100),v*(1+lim/100)
        elif vtype == 'abs':
            vmin,vmax = v-lim,v+lim
            vmin = max(0,vmin)
        elif vtype == 'fixed' or vtype == 'default' or vtype == 'total':
            vmin,vmax,step = None,None,0
        else:
            raise ValueError()
        outList.append(randomWalkFloat(v,vmin,vmax,step))
    if len(outList) == 1:
        outList = outList[0]
    return outList
class surfLayer(object):
    def __init__(self,layerType,settingDict) -> None:
        self.type,self.mtype = layerType,settingDict['type']
        self.paraDict = {}
        self.paraDict['H'] = genRWFList(settingDict['h'])
        if self.mtype == 'water':
            self.paraDict['vp'] = genRWFList(settingDict['vp'])
        else:
            self.paraDict['vs'] = genRWFList(settingDict['vs'])
            self.paraDict['vpvs'] = genRWFList(settingDict['vpvs'])
    @property
    def H(self):
        return self.paraDict['H']
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
        return nFine
    def bspl(self,z,nBasis,deg):
        if hasattr(self,'_bspl') and (nBasis == self._bspl.nBasis) and \
           (deg == self._bspl.deg) and (len(z) == self._bspl.n):
           pass
        else:
            self._bspl = bsplBasis(z,nBasis,deg)
        return self._bspl
    def _hProfile(self):
        return np.array([self.H/self.nFine]*self.nFine)
    def _vsProfile(self):
        vsGrid = self._vsProfileGrid()
        return (vsGrid[:-1]+vsGrid[1:])/2
        if self.mtype == 'water':
            return np.array([0]*self.nFine)
        elif self.mtype == 'constant':
            return np.array([self.vs]*self.nFine)
        elif self.mtype == 'linear':
            tmp = np.linspace(self.vs[0],self.vs[1],self.nFine+1)
            return np.array((tmp[:-1]+tmp[1:])/2)
        elif self.mtype == 'Bspline':
            z = np.linspace(0, self.H, self.nFine+1)
            nBasis = len(self.vs)
            deg = 3 + (nBasis>=4)
            tmp = self.bspl(z,nBasis,deg) * self.vs
            return np.array((tmp[:-1]+tmp[1:])/2)
    # def _vsProfile(self):       # interp using n instead of n+1, which should be worse
    #     if self.mtype == 'water':
    #         return np.array([0]*self.nFine)
    #     elif self.mtype == 'constant':
    #         return np.array([self.vs]*self.nFine)
    #     elif self.mtype == 'linear':
    #         tmp = np.linspace(self.vs[0],self.vs[1],self.nFine)
    #         return tmp
    #     elif self.mtype == 'Bspline':
    #         z = np.linspace(0, self.H, self.nFine)
    #         nBasis = len(self.vs)
    #         deg = 3 + (nBasis>=4)
    #         tmp = self.bspl(z,nBasis,deg) * self.vs
    #         return tmp
    def genProfile(self):
        vs = self._vsProfile()
        h  = self._hProfile()
        if self.type == 'water':
            vp  = [self.vp]     *self.nFine
            rho = [1.027]       *self.nFine
            qs  = [10000.]      *self.nFine
            qp  = [57822.]      *self.nFine
        elif self.type == 'sediment':
            vpvs = 2.0 if not hasattr(self,'vpvs') else self.vpvs
            # vp   = vs*vpvs
            vp   = vs*1.23 + 1.28   # marine sediments and rocks, Hamilton 1979
            rho  = 0.541 + 0.3601*vp
            qs  = [80.]   *self.nFine
            qp  = [160.]  *self.nFine
        elif self.type == 'crust':
            vpvs = 1.8 if not hasattr(self,'vpvs') else self.vpvs
            vp   = vs*vpvs
            rho  = 0.541 + 0.3601*vp
            qs  = [600.]   *self.nFine
            qp  = [1400.]  *self.nFine
        elif self.type == 'mantle':
            vpvs = 1.76 if not hasattr(self,'vpvs') else self.vpvs
            vp = vs*vpvs
            rho = 3.4268+(vs-4.5)/4.5
            # rho  = 0.541 + 0.3601*vp    ## from Hongda
            qs  = [150.]   *self.nFine
            qp  = [1400.]  *self.nFine
        # rho = np.array(rho)
        # rho[np.array(vp) > 7.5]       = 3.35
        return np.array([h,vs,vp,rho,qs,qp])
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
    def _reset(self,resetype='origin'):
        for paraKey in self.paraDict.keys():
            if type(self.paraDict[paraKey]) is not list:
                self.paraDict[paraKey] = self.paraDict[paraKey].reset(resetype=resetype)
            else:
                self.paraDict[paraKey] = [v.reset(resetype=resetype) for v in self.paraDict[paraKey]]
    def reset(self,resetype='origin'):
        newLayer = self.copy()
        newLayer._reset(resetype=resetype)
        return newLayer
    def plotProfile(self,type='vs'):
        h,vs,vp,rho,qs,qp = self.genProfile()
        if type == 'vs':
            plotLayer(h,vs);plt.title('Vs')
        else:
            print('To be added...')
    def copy(self):
        return copy.deepcopy(self)

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
class model1D(object):
    def __init__(self,layerList=[],info={}) -> None:
        self.layers = layerList
        from copy import deepcopy
        self.info = deepcopy(info)
    @property
    def totalH(self):
        return np.sum([layer.H for layer in self.layers])
    def loadSetting(self,settingYML=None):
        if settingYML is None:
            return -1
        setting = settingDict(); setting.load(settingYML)
        if list(setting.values())[-1]['h'][1] == 'total':
            Hs = [setting[ltype]['h'][0] for ltype in setting.keys()]
            setting[list(setting.keys())[-1]]['h'] = [Hs[-1]-np.sum(Hs[:-1]),'total']
        else:
            raise ValueError()
        self.layers = [surfLayer(ltype,setting[ltype]) for ltype in setting.keys()]
    def loadFromMC(self,MCpara):
        H = self.totalH
        i = 0
        for l in self.layers:
            p = l.paraDict
            for k in p.keys():
                if type(p[k]) is not list:
                    if p[k].vmin is not None:
                        p[k] = randomWalkFloat(MCpara[i],p[k].vmin,p[k].vmax,p[k].step)
                        i+=1
                else:
                    for j in range(len(p[k])):
                        if p[k][j].vmin is not None:
                            p[k][j] = randomWalkFloat(MCpara[i],p[k][j].vmin,p[k][j].vmax,p[k][j].step)
                            i+=1
        l.paraDict['H'] = randomWalkFloat(H-np.sum([layer.H for layer in self.layers[:-1]]))
    def setFinalH(self,totalH):
        finalH = totalH - np.sum([l.paraDict['H'] for l in self.layers[:-1]])
        self.layers[-1].paraDict['H'] = randomWalkFloat(finalH)
    def perturb(self):
        i = 0
        while i < 100:
            newMod = model1D([layer.perturb() for layer in self.layers],self.info)
            newMod.setFinalH(self.totalH)
            i += 1
            if newMod.isgood():
                return newMod
        print('Warning: no good perturbation found, return uniform reset instead')
        return self.reset('uniform')
    def reset(self,resetype='origin'):
        i = 0
        while i < 1000:
            newMod = model1D([layer.reset(resetype) for layer in self.layers],self.info)
            newMod.setFinalH(self.totalH)
            i += 1
            if newMod.isgood():
                return newMod
        print(f'Error: no good reset:{resetype} found!!')
    def genProfile(self): # h,vs,vp,rho,qs,qp
        if 'lithoAge' in self.info.keys(): # assume mantle is the last layer
            typeLst = []
            for layer in self.layers:
                typeLst += [layer.type]*layer.nFine
            typeLst = np.array(typeLst)
            tmp = np.concatenate([layer.genProfile() for layer in self.layers],axis=1)
            h = tmp[0];deps = h.cumsum()
            vpvsC = 1.8  if not hasattr(self.layers[-2],'vpvs') else self.layers[-2].vpvs
            vpvsM = 1.76 if not hasattr(self.layers[-1],'vpvs') else self.layers[-1].vpvs
            qsCrust = 350
            qpCrust = 1./(4./3.*(vpvsC)**(-2) / qsCrust + (1-4./3.*(vpvsC)**(-2))/57823.)
            qsCrust,qpCrust   = calCrustQ(vpvsC)
            qsMantle,qpMantle = calMantleQ(deps[typeLst=='mantle'],vpvsM,age=self.info['lithoAge'])
            tmp[4,typeLst=='crust'] = qsCrust
            tmp[5,typeLst=='crust'] = qpCrust
            tmp[4,typeLst=='mantle'] = qsMantle[:]
            tmp[5,typeLst=='mantle'] = qpMantle[:]
            return tmp
        else:
            return np.concatenate([layer.genProfile() for layer in self.layers],axis=1)
    def plotProfile(self,type='vs',**kwargs):
        h,vs,vp,rho,qs,qp = self.genProfile()
        if type == 'vs':
            fig = plotLayer(h,vs,**kwargs);plt.title('Vs')
        else:
            print('To be added...')
        return fig
    def forward(self,periods=[5,10,20,40,60,80]):
        pred = _calForward(self.genProfile(),wavetype='Ray',periods=periods)
        if pred is None:
            print(f'Warning: Forward not complete! Model listed below:')
            self.show()
        return pred
    def copy(self):
        return copy.deepcopy(self)
    def isgood(self):
        ltypes = []
        for layer in self.layers:
            ltypes.extend([layer.type]*layer.nFine)
        ltypes = np.array(ltypes)
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
        if not vsMantle[1]<vsMantle[0]: 
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
        return True
    def _paras(self):
        paras = []
        for l in self.layers:
            for k in l.paraDict.keys():
                if type(l.paraDict[k]) == list:
                    paras.extend(l.paraDict[k])
                else:
                    paras.append(l.paraDict[k])
        paras = [v for v in paras if v.vmin is not None]
        return paras
    def paras(self,fullInfo=False):
        if not fullInfo:
            return self._paras()
        paras = []
        for l in self.layers:
            for k in l.paraDict.keys():
                if type(l.paraDict[k]) == list:
                    for v in l.paraDict[k]:
                        paras.append([v,l.type,k])
                else:
                    v = l.paraDict[k]
                    paras.append([v,l.type,k])
        paras = [a for a in paras if a[0].vmin is not None]
        return paras
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
    def __init__(self,settingYML=None,infoDict=None,periods=[],vels=[],uncers=[]) -> None:
        self.setting = settingDict(); self.setting.load(settingYML)
        self.setting.updateInfo(infoDict)
        self.initMod = model1D(info=infoDict)
        self.initMod.loadSetting(self.setting)
        self.obs = {}
        self.obs['RayPhase'] = {'T':periods,'c':vels,'uncer':uncers}
    def updateSetting(self,newVars):
        if type(newVars) == settingDict:
            self.setting = newVars
        else:
            self.setting.updateVariable(newVars)
        self.initMod.loadSetting(self.setting)
    def misfit(self,model=None):
        if model is None:
            model = self.initMod
        T = self.obs['RayPhase']['T']
        cP = model.forward(periods=T)
        if cP is None:
            return 88888,0
        cO = self.obs['RayPhase']['c']
        uncer = self.obs['RayPhase']['uncer']
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
                    T = self.obs['RayPhase']['T']
                    plt.plot(T,self.obs['RayPhase']['c'],'--')
                    plt.plot(T,mod0.forward(periods=T))
                    plt.plot(T,mod1.forward(periods=T))
                if priori:
                    mod1.write(i,mcTrack,[0,1,1])
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
        np.savez_compressed(f'{outdir}/{id}.npz',mcTrack=mcTrack,initMod=self.initMod,obs=self.obs)
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
            subMC,_,_ = tmp['mcTrack'],tmp['initMod'][()],tmp['obs'][()]
            subMCLst.append(subMC)
        os.system(f'rm -r {tmpDir}')
        mcTrack = np.concatenate(subMCLst,axis=0)
        os.makedirs(outdir,exist_ok=True)
        np.savez_compressed(f'{outdir}/{id}.npz',mcTrack=mcTrack,initMod=self.initMod,obs=self.obs)

        print(f'Time cost:{time.time()-timeStamp:.2f} ')
        
class PostPoint(Point):
    def __init__(self,npzMC=None,npzPriori=None):
        if npzMC is not None:
            tmp = np.load(npzMC,allow_pickle=True)
            self.MC,self.initMod,self.obs = tmp['mcTrack'],tmp['initMod'][()],tmp['obs'][()]
            self.N       = self.MC.shape[0]
            self.misfits = self.MC[:,0]
            self.Ls      = self.MC[:,1]
            self.accepts = self.MC[:,2]
            self.MCparas = self.MC[:,3:]

            indMin = np.nanargmin(self.misfits)
            self.minMod         = self.initMod.copy()
            self.minMod.loadFromMC(self.MCparas[indMin])
            self.minMod.L       = self.Ls[indMin]
            self.minMod.misfit  = self.misfits[indMin]

            self.thres  = max(self.minMod.misfit*2, self.minMod.misfit+0.5)
            self.accFinal = (self.misfits < self.thres)

            self.avgMod         = self.initMod.copy()
            self.avgMod.loadFromMC(np.mean(self.MCparas[self.accFinal,:],axis=0))
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
                                    'vpvs': [1.76, 'fixed']}}

        dset = invhdf5(dsetFile,'r')
        topo = dset[id].attrs['topo']
        sedthk = dset[id].attrs['sedi_thk']
        lithoAge = dset[id].attrs['litho_age']
        infoDict = {'topo':topo,'sedthk':sedthk,'lithoAge':lithoAge}
        setting = settingDict(setting_Hongda_pyMCinv)
        setting.updateInfo(infoDict)
        self.initMod = model1D(info=infoDict)
        self.initMod.loadSetting(setting_Hongda_pyMCinv)

        T,pvelp,pvelo,uncer = np.loadtxt(f'{invDir}/{id}_0.ph.disp').T
        self.obs = {}
        self.obs['RayPhase'] = {'T':T,'c':pvelo,'uncer':uncer}

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
        self.minMod.loadFromMC(self.MCparas[indMin])
        self.minMod.L       = self.Ls[indMin]
        self.minMod.misfit  = self.misfits[indMin]

        self.thres  = max(self.minMod.misfit*2, self.minMod.misfit+0.5)
        self.accFinal = (self.misfits < self.thres)

        self.avgMod         = self.initMod.copy()
        self.avgMod.loadFromMC(np.mean(self.MCparas[self.accFinal,:],axis=0))
        self.avgMod.misfit,self.avgMod.L = self.misfit(model=self.avgMod)

        if priDir is not None:
            inarr = np.load(f'{priDir}/mc_inv.{id}.npz')
            invdata    = inarr['arr_0']
            paraval       = invdata[:,2:11]
            self.Priparas = paraval[:,colOrder]
    def plotDisp(self):
        T,vel,uncer = self.obs['RayPhase']['T'],self.obs['RayPhase']['c'],\
                      self.obs['RayPhase']['uncer']
        plt.figure()
        plt.errorbar(T,vel,uncer,ls='None',capsize=3,label='Observation')
        plt.plot(T,self.avgMod.forward(T),label='Avg accepted')
        plt.plot(T,self.minMod.forward(T),label='Min misfit')
        plt.plot(T,self.initMod.forward(T),label='Initial')
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
    def plotVsProfile(self):
        fig = self.initMod.plotProfile(label='Initial')
        self.avgMod.plotProfile(fig=fig,label='Avg')
        self.minMod.plotProfile(fig=fig,label='Min')
        plt.xlim(3.8,4.8)
        plt.legend()
    def plotCheck(self):
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


if __name__ == '__main__':
    pass
    # a = randomWalkFloat(3.4,3,4,0.2)

    # bspl = bsplBasis(np.linspace(0,200,100),5,4)
    # plt.plot(bspl.z,bspl.basis.T)

    # setting=settingDict(); setting.load('setting-Hongda.yml')
    # setting.updateInfo({'topo':2,'sedthk':1})
    # setting.updateVariable([8,8,8,8,8,8])

    # setting=settingDict(); setting.load('setting-Hongda.yml')
    # lay1 = surfLayer('sediment',setting['sediment'])
    # print(lay1.genProfile())
    # print(lay1.vs)
    # lay2 = lay1.perturb()
    # print(lay2.vs)
    # lay3 = lay2.reset('uniform')
    # print(lay3.vs)

    # mod1 = model1D()
    # mod1.loadSetting('setting-Hongda.yml')
    # h,vs,vp,rho,qs,qp = mod1.genProfile()
    # mod1.plotProfile()
    # mod2 = mod1.perturb(); mod2.plotProfile()
    # mod3 = mod2.reset('uniform'); mod3.plotProfile()
    # plt.figure();plt.plot(np.linspace(5,80,100),mod1.forward(np.linspace(5,80,100)))
    # print(mod1.paras())
    # print(mod1.isgood())

    # with open('setting-Hongda.yml', 'r') as f:
    #     setting = ezDict(yaml.load(f,Loader=yaml.FullLoader))
    # # setting.updataValue(['water','h',0],30)
    # # setting.updataValue(['water','type'],'test')
    # infoDict = {'topo':-10,'sedthk':20}
    # setting.updateInfo(infoDict)

    # random.seed(36)
    # setting = settingDict(); setting.load('setting-Hongda.yml')
    # setting.updateInfo({'topo':2,'sedthk':1})
    # mod1 = model1D(); mod1.loadSetting(setting)
    # mod2 = mod1.reset('uniform')
    # periods = np.array([10,12,14,16,18,20,24,28,32,36,40,50,60,70,80])
    # p = Point('setting-Hongda.yml',{'topo':2,'sedthk':1},
    #           periods,mod2.forward(periods),[0.01]*len(periods))
    # p.updateSetting((np.array(mod1.paras())+np.array(mod2.paras()))/2)
    # setting = p.setting.copy()
    # setting['sediment']['h'][2] = min(setting['sediment']['h'][0],setting['sediment']['h'][2])
    # p.updateSetting(setting)
    # # p.MCinvMP(runN=50000,step4uwalk=1000,nprocess=26)
    # # p.MCinvMP('MCtest_priori',runN=50000,step4uwalk=1000,nprocess=26,priori=True)
    # pN = PostPoint('MCtest/test.npz','MCtest_priori/test.npz')
    # pN.plotCheck()

    # dsetFile = '/home/ayu/Projects/Cascadia/Tasks/runInv/test_surf_thesis_dbase_finalMar.h5'
    # invDir = '/home/ayu/Projects/Cascadia/Tasks/runInv/inv_thesis_vs_finalMar'
    # priDir = '/home/ayu/Projects/Cascadia/Tasks/runInv/inv_thesis_vs_finalMar_priori'
    # id     = '233.0_43.2'
    # pHD = PostPoint()
    # pHD.loadpyMCinv(dsetFile,id,invDir,priDir)
    # # pHD.plotDisp()
    # # pHD.plotDistrib()
    # # pHD.plotVsProfile()
    # pHD.initMod.show()

    id     = '233.0_43.2'
    T = [ 10.,  12.,  14.,  16.,  18.,  20.,  22.,  24.,  26.,  28.,  30.,
        32.,  36.,  40.,  50.,  60.,  70.,  80.]
    vel = [ 3.66170009,  3.72857888,  3.75951126,  3.76266499,  3.77191581,
        3.7685344 ,  3.77129248,  3.77428902,  3.77529921,  3.78802274,
        3.79433664,  3.80568807,  3.82146285,  3.8505667 ,  3.84643676,
        3.87612961,  3.91444643,  3.96543979]
    uncer = [ 0.01754326,  0.01164089,  0.00903466,  0.00797875,  0.00716722,
        0.00713235,  0.00744013,  0.00770071,  0.00797466,  0.00956988,
        0.01142398,  0.00890576,  0.00949308,  0.01012225,  0.01201   ,
        0.01743369,  0.01614607,  0.01649115]
    topo = -3.068602323532039
    sedthk = 0.22750000655653985
    p = Point('setting-Hongda.yml',{'topo':topo,'sedthk':sedthk}, T,vel,uncer)
