import numpy as np
from copy import deepcopy

class BsplBasis(object):
    ''' calculating cubic b-spline basis functions '''
    def __init__(self,z,n,deg=None,alpha=2.,eps=np.finfo(float).eps) -> None:
        self.n,self.nBasis,self.deg,self.alpha,self.eps = len(z),n,deg,alpha,eps
        if self.nBasis == 1:
            self.basis = np.ones((1,self.n))
            return
        if self.nBasis == 2:
            self.basis = np.ones((2,self.n))
            self.basis[0,:] = np.linspace(1,0,self.n)
            self.basis[1,:] = np.linspace(0,1,self.n)
            return
        if deg is None:
            deg = 3 + (n>=4)
        x = np.zeros(n+deg)
        x[:deg-1]	= -eps*np.ones(deg-1)
        x[deg-1]	= 0.
        x[deg:n]	= np.power(alpha,range(n-deg)) * (alpha-1) / (np.power(alpha,n-deg+1)-1)
        x[n]		= 1.
        x[n+1:]		= (1+eps)*np.ones(deg-1)
        x = z[0] + x*(z[-1]-z[0]) 
        bs0 = np.zeros((len(z),len(x)-1))
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
        if self.nBasis == 1:
            coef = np.array([coef])
        return np.dot(coef,self.basis)
    def plot(self):
        import matplotlib.pyplot as plt
        plt.figure();plt.plot(np.linspace(0,1,self.n),self.basis.T)


class SeisLayer():
    def __init__(self,parm={},prop={}) -> None:
        self.parm = parm
        self.prop = {'Group':None,'LayerName':None}
        self.prop.update(prop)
    def seisPropGrids(self,**kwargs):
        return None,None,None,None,None,None
    def seisPropLayers(self,**kwargs):
        z,vs,vp,rho,qs,qp = self.seisPropGrids(**kwargs)
        h = np.diff(z)
        vs  = (vs[1:] + vs[:-1])/2
        vp  = (vp[1:] + vp[:-1])/2
        rho = (rho[1:]+ rho[:-1])/2
        qs  = (qs[1:] + qs[:-1])/2
        qp  = (qp[1:] + qp[:-1])/2
        return h,vs,vp,rho,qs,qp
    def _perturb(self,reset=False):
        from pySurfInv.utils import _dictIterModifier
        from pySurfInv.brownian import BrownianVar
        def checker(v):
            return isinstance(v,BrownianVar)
        if reset:
            def modifier(v):
                return v.reset()
        else:
            def modifier(v):
                return v.move()
        newLayer = self.copy()
        newLayer.parm = _dictIterModifier(self.parm,checker,modifier)
        return newLayer
    def _reset(self):
        return self._perturb(reset=True)
    def copy(self):
        return deepcopy(self)

class PureLayer(SeisLayer):
    def __init__(self, parm={}, prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'PureLayer'})

    def seisPropLayers(self,**kwargs):
        h,vs,vp,rho,qs,qp = self.parm['h'],self.parm['vs'],self.parm['vp'],\
                            self.parm['rho'],self.parm['qs'],self.parm['qp']
        return np.array(h),np.array(vs),np.array(vp),np.array(rho),np.array(qs),np.array(qp)
    def H(self,**kwargs):
        return self.parm['h'].sum()

class PureGrid(SeisLayer):
    def __init__(self, parm={}, prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'PureGrid'})
    def seisPropGrids(self,**kwargs):
        z,vs,vp,rho,qs,qp = self.parm['z'],self.parm['vs'],self.parm['vp'],\
                            self.parm['rho'],self.parm['qs'],self.parm['qp']
        return np.array(z),np.array(vs),np.array(vp),np.array(rho),np.array(qs),np.array(qp)
    def H(self, **kwargs):
        return self.parm['z'][-1] - self.parm['z'][0]




class SeisLayerVs(SeisLayer):
    # kwargs['layersAbove'] = [z,vs,vp,rho,qs,qp,grp,layerName]
    def seisPropGrids(self,**kwargs):
        N   = self._nFineLayers(**kwargs)
        z   = np.linspace(0, self._calH(**kwargs), N+1)
        vs = self._calVs(z,**kwargs)
        vp,rho,qs,qp = self._calOthers(z,vs,**kwargs)
        return z,vs,vp,rho,qs,qp
    def _calH(self,**kwargs):
        if 'BottomDepth' in self.parm:
            z0 = kwargs['layersAbove'][0][-1]
            H = self.parm['BottomDepth'] - z0
        else:
            H = self.parm['H']
        return H
    def _nFineLayers(self,**kwargs):
        pass
    def _calVs(self,z,**kwargs):
        pass
    def _calOthers(self,z,vs,**kwargs):
        pass
    def _bspl(self,z,nBasis,deg=None):
        deg = 3 + (nBasis>=4) if deg is None else deg
        if hasattr(self,'_bspl_') and (nBasis == self._bspl_.nBasis) and \
           (deg == self._bspl_.deg) and (len(z) == self._bspl_.n):
           pass
        else:
            self._bspl_ = BsplBasis(z,nBasis,deg)
        return self._bspl_

class Sediment(SeisLayerVs):
    def __init__(self,parm={},prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'LandSediment','Group':'sediment'})
    def _nFineLayers(self,**kwargs):
        return 1
    def _calVs(self,z,**kwargs):
        if isinstance(self.parm['Vs'],list):
            return np.linspace(self.parm['Vs'][0],self.parm['Vs'][1],len(z))
        else:
            return np.array([self.parm['Vs']] * len(z))
    def _calOthers(self, z, vs, **kwargs):
        vp  = vs*2.0
        rho = 1.22679 + 1.53201*vs - 0.83668*vs*vs + 0.20673*vs**3 - 0.01656*vs**4
        qs  = np.array( [80]  * len(z) )
        qp  = np.array( [160] * len(z) )
        return vp,rho,qs,qp
    
class Crust(SeisLayerVs):
    def __init__(self,parm={},prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'LandCrust','Group':'crust'})
    def _nFineLayers(self,**kwargs):
        H = self._calH(**kwargs)
        if H >= 150:
            N = 60
        elif H > 60:
            N = 30
        elif H > 20:
            N = 15
        elif H > 10:
            N = 10
        else:
            N = 5
        return N
    def _calVs(self, z, **kwargs):
        nBasis = len(self.parm['Vs'])
        if self.parm.get('Gauss',False) is False:
            return self._bspl(z,nBasis) * self.parm['Vs']
        else:
            from Triforce.mathPlus import gaussFun
            nBasis = len(self.parm['Vs'])
            vs0 = self._bspl(z,nBasis) * self.parm['Vs']
            vs1 = gaussFun(self.parm['Gauss'][0],self.parm['Gauss'][1],self.parm['Gauss'][2],z)
            return vs0+vs1
    def _calOthers(self, z, vs, **kwargs):
        vp  = vs*1.80
        rho = 1.22679 + 1.53201*vs - 0.83668*vs*vs + 0.20673*vs**3 - 0.01656*vs**4
        qs  = np.array( [600]  * len(z) )
        qp  = np.array( [1400] * len(z) )
        return vp,rho,qs,qp

class OceanWater(SeisLayerVs):
    def __init__(self,parm={},prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'OceanWater','Group':'water'})
        self.parm['Vs'] = 0
    def seisPropGrids(self,**kwargs):
        N   = 1
        z   = np.array([0,self._calH(**kwargs)])
        vs  = np.array([0]     * (N+1))
        vp  = np.array([1.475] * (N+1))
        rho = np.array([1.027] * (N+1))
        qs  = np.array([10000] * (N+1) )
        qp  = np.array([57822] * (N+1) )
        return z,vs,vp,rho,qs,qp

class OceanSediment(SeisLayerVs):
    def __init__(self,parm={},prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'OceanSediment','Group':'sediment'})
    def _nFineLayers(self,**kwargs):
        return 1
    def _calVs(self,z,**kwargs):
        return np.array([self.parm['Vs']] * len(z))
    def _calOthers(self, z, vs, **kwargs):
        vp  = vs*1.23 + 1.28
        rho = 0.541 + 0.3601*vp
        qs  = np.array( [80]  * len(z) )
        qp  = np.array( [160] * len(z) )
        return vp,rho,qs,qp

class OceanCrust(SeisLayerVs):
    def __init__(self,parm={},prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'OceanCrust','Group':'crust'})
    def _nFineLayers(self,**kwargs):
        return min(max(int(round(self._calH(**kwargs)/2)),2),10)
    def _calVs(self, z, **kwargs):
        try:
            return np.linspace(self.parm['Vs'][0],self.parm['Vs'][1],len(z))
        except:
            return np.array([self.parm['Vs']] * len(z))
    def _calOthers(self, z, vs, **kwargs):
        vp  = vs*1.8
        rho = 0.541 + 0.3601*vp
        qs  = np.array( [350]  * len(z) )
        qp  = np.array( [1400] * len(z) )
        return vp,rho,qs,qp

class OceanMantle(SeisLayerVs):
    def __init__(self,parm={},prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'OceanMantle','Group':'mantle'})
    def _nFineLayers(self,**kwargs):
        H = self._calH(**kwargs)
        if H >= 150:
            N = 60
        elif H > 60:
            N = 30
        elif H > 20:
            N = 15
        elif H > 10:
            N = 10
        else:
            N = 5
        return N
    def _calVs(self, z, **kwargs):
        nBasis = len(self.parm['Vs'])
        deg    = self.parm.get('deg',None)
        return self._bspl(z,nBasis,deg) * self.parm['Vs']
    def _calOthers(self, z, vs, **kwargs):
        vp  = vs*1.76
        rho = 3.4268+(vs-4.5)/4.5
        qs  = np.array( [150.]  * len(z) )
        qp  = np.array( [1400.] * len(z) )
        return vp,rho,qs,qp

class ReferenceMantle(OceanMantle):
    def __init__(self,parm={},prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'ReferenceMantle','Group':'mantle'})
    def _nFineLayers(self,**kwargs):
        return 20
    def _calVs(self, z, **kwargs):
        layersAbove = kwargs['layersAbove']
        vs0 = layersAbove[1][-1] # z,vs,vp,rho,qs,qp,grp,layerName
        return np.linspace(vs0,vs0+(z[-1]-z[0])*self.parm['Slope'],len(z))
    def _calOthers(self, z, vs, **kwargs):
        vp,rho,qs,qp = super()._calOthers(z,vs,**kwargs)
        layersAbove = kwargs['layersAbove']
        vp0 = layersAbove[2][-1];  vp = vp0 + (vp-vp[0])
        rho0 = layersAbove[3][-1]; rho = rho0 + (rho-rho[0])
        qs0 = layersAbove[4][-1];  qs = qs0 + (qs-qs[0])
        qp0 = layersAbove[5][-1];  qp = qp0 + (qp-qp[0])
        return vp,rho,qs,qp
        


# Juan de Fuca Specified
class OceanSedimentCascadia(OceanSediment):
    def __init__(self, parm={}, prop={}) -> None:
        super().__init__(parm, prop)
        self.prop.update({'LayerName':'OceanSedimentCascadia','Group':'sediment'})
    def _calVs(self, z, **kwargs):
        vs = (0.02*self._calH(**kwargs)**2+1.27*self._calH(**kwargs)+0.29*0.1)/(self._calH(**kwargs)+0.29)
        return np.array([vs]* len(z))

class OceanMantleHybrid(OceanMantle):
    def __init__(self, parm={}, prop={}) -> None:
        super().__init__(parm, prop)
        self.prop.update({'LayerName':'OceanMantleHybrid','Group':'mantle'})

    def _calVs(self, z, **kwargs):
        layersAbove = kwargs['layersAbove']

        def getCrustH():
            h = np.diff(layersAbove[0]); grp = np.array(layersAbove[6][:-1]) # z,vs,vp,rho,qs,qp,grp,layerName
            if np.diff(np.insert(grp=='crust',[0,len(grp)],False)).sum() != 2:# check if there are two seperated layers of crust
                raise ValueError(f'In {self.__class__}: more than 1 crust layer found!')
            h,grp = h[h>0.01],grp[h>0.01]
            crustH = np.sum(h[grp=='crust'])
            return crustH
        def meltStart(age):
            therMod = HSCM(age=age)
            P = therMod.P/1e9
            solidus = -5.1*P**2 + 92.5*P + 1120.6 + 273.15  # damp solidus
            try:
                return therMod.zdeps[therMod.T > 0.92*solidus][0]
            except:
                return therMod.zdeps[-1]
        def merge2(x,y1,y2,xL,xH):
            vss = list(y1[x<xL]) + list(y2[x>xH])
            xs  = list(x[x<xL])  + list(x[x>xH])
            from scipy.interpolate import CubicSpline
            return CubicSpline(xs, vss)(x)

        from pySurfInv.ThermSeis import OceanSeisRitz,OceanSeisRuan,HSCM
        crustH = getCrustH()
        nBasis = len(self.parm['Vs']) + 1
        Tp = self.parm.get('Tp',1325)

        conversionModel = self.parm.get('Conversion','Ritzwoller')
        if conversionModel == 'Yamauchi':
            seisMod = OceanSeisRuan(HSCM(age=max(1e-3,self.parm['ThermAge']),zdeps=crustH+z,Tp=Tp),
                                    period=1)
        elif conversionModel == 'Ritzwoller':
            seisMod = OceanSeisRitz(HSCM(age=max(1e-3,self.parm['ThermAge']),zdeps=crustH+z,Tp=Tp))
        else:
            raise ValueError(f'Invalid convertion model: {conversionModel}')

        zMelt = meltStart(max(1e-3,self.parm['ThermAge']))-crustH
        vs = merge2(z,seisMod.vs,self._bspl(z,nBasis)*np.array([0]+list(self.parm['Vs']))+seisMod.vs,
                   xL=zMelt,xH=(zMelt+crustH)*1.7-crustH)
        self._debug_zMelt = zMelt

        return vs

    def _calOthers(self, z, vs, **kwargs):
        layersAbove,modelInfo = kwargs['layersAbove'],kwargs['modelInfo']

        Qage = modelInfo.get('lithoAge',None) if modelInfo.get('lithoAgeQ',False) else None
        z0 = layersAbove[0][-1]  # z,vs,vp,rho,qs,qp,grp,layerName
        period = modelInfo.get('period',1)

        from pySurfInv.ThermSeis import OceanSeisRuan,HSCM
        Qage = self.parm['ThermAge'] if Qage is None else Qage
        seisMod = OceanSeisRuan(HSCM(age=max(1e-3,Qage),zdeps=z0+z),period=period)

        vp,rho,qs,qp = super()._calOthers(z, vs, **kwargs)
        qs = seisMod.qs;qs[qs>5000] = 5000
        return vp,rho,qs,qp



''' unverified layers 

# class OceanSediment_Prism(OceanSediment):
#     def __init__(self,parm,prop={}) -> None:
#         super().__init__(parm,prop)
#         self.prop.update({'LayerName':'OceanSediment_Prism','Group':'sediment'})
#     def _nFineLayers(self,**kwargs):
#         return  min(max(int(round(self._calH(**kwargs)/2)),2),10)
#     def _calVs(self, z, **kwargs):
#         return np.linspace(0, self._calH(**kwargs), len(z))


# class OceanCrust_Prism(OceanCrust):
#     def __init__(self,parm,prop={}) -> None:
#         super().__init__(parm,prop)
#         self.prop.update({'LayerName':'OceanCrust_Prism','Group':'crust'})
#     def _nFineLayers(self,**kwargs):
#         H = self._calH(**kwargs)
#         if H >= 150:
#             N = 60
#         elif H > 60:
#             N = 30
#         elif H > 20:
#             N = 15
#         elif H > 10:
#             N = 10
#         else:
#             N = 5
#         return N
#     def _calVs(self, z, **kwargs):
#         nBasis = len(self.parm['Vs'])
#         return self._bspl(z,nBasis) * self.parm['Vs']


# class Prism(Crust):
#     def __init__(self,parm={},prop={}) -> None:
#         super().__init__(parm,prop)
#         self.prop.update({'LayerName':'Prism','Group':'prism'})
#     def _calVs(self, z, **kwargs):
#         if len(self.parm['Vs'])>2:
#             nBasis = len(self.parm['Vs'])
#             return self._bspl(z,nBasis) * self.parm['Vs']
#         else:
#             return np.linspace(self.parm['Vs'][0],self.parm['Vs'][1],len(z))

# class SubductionPlateCrust(OceanCrust):
#     def __init__(self,parm={},prop={}) -> None:
#         super().__init__(parm,prop)
#         self.prop.update({'LayerName':'SubductionPlateCrust','Group':'crust'})
#     def _calVs(self, z, **kwargs):
#         if isinstance(self.parm['Vs'],list):
#             return np.linspace(self.parm['Vs'][0],self.parm['Vs'][1],len(z))
#         else:
#             return np.array([self.parm['Vs']] * len(z))

# class SubductionPlateCrustLowVs(SubductionPlateCrust):
#     def __init__(self,parm={},prop={}) -> None:
#         super().__init__(parm,prop)
#         self.prop.update({'LayerName':'SubductionPlateCrustLowVs','Group':'crust'})
#     def _calOthers(self, z, vs, **kwargs):
#         vp  = vs*2.45
#         rho = 0.541 + 0.3601*vp
#         qs  = np.array( [350]  * len(z) )
#         qp  = np.array( [1400] * len(z) )
#         return vp,rho,qs,qp
        

# class SubductionPlateMantle(OceanMantle):
#     def __init__(self,parm={},prop={}) -> None:
#         super().__init__(parm,prop)
#         self.prop.update({'LayerName':'SubductionPlateMantle','Group':'mantle'})
#     def _calVs(self, z, **kwargs):
#         if isinstance(self.parm['Vs'],list):
#             return np.linspace(self.parm['Vs'][0],self.parm['Vs'][1],len(z))
#         else:
#             return np.array([self.parm['Vs']] * len(z))

# class SubductionPlateMantleParabola(OceanMantle):
#     def __init__(self,parm={},prop={}) -> None:
#         super().__init__(parm,prop)
#         self.prop.update({'LayerName':'SubductionPlateMantleParabola','Group':'mantle'})
#     def _calVs(self, z, **kwargs):
#         vs0,vs1 = self.parm['Vs']
#         H = z[-1]-z[0]; A = -(vs1-vs0)*4/(H**2)
#         return A*(z-z[0]-H/2)**2+vs1

# class OceanMantleHighNBspl(OceanMantle):
#     def __init__(self,parm={},prop={}) -> None:
#         super().__init__(parm,prop)
#         self.prop.update({'LayerName':'OceanMantleHighNBspl','Group':'mantle'})
#     def _bspl(self,z,nBasis):
#         deg = nBasis-1 if nBasis > 3 else 3
#         if hasattr(self,'_bspl_') and (nBasis == self._bspl_.nBasis) and \
#            (deg == self._bspl_.deg) and (len(z) == self._bspl_.n):
#            pass
#         else:
#             self._bspl_ = BsplBasis(z,nBasis,deg)
#         return self._bspl_

# class OceanMantleSerpentineTop(OceanMantle):
#     def __init__(self,parm={},prop={}) -> None:
#         super().__init__(parm,prop)
#         self.prop.update({'LayerName':'OceanMantleSerpentineTop','Group':'mantle'})
#     def _calOthers(self, z, vs, **kwargs):
#         # vp  = vs*1.76
#         vp = np.clip(vs,4.4,None)*1.76
#         rho = 3.4268+(vs-4.5)/4.5
#         qs  = np.array( [150.]  * len(z) )
#         qp  = np.array( [1400.] * len(z) )
#         return vp,rho,qs,qp



# class OceanMantleGaussian(OceanMantle):
#     def __init__(self,parm={},prop={}) -> None:
#         super().__init__(parm,prop)
#         self.prop.update({'LayerName':'OceanMantleGaussian','Group':'mantle'})
#     def _calVs(self, z, **kwargs):
#         from Triforce.mathPlus import gaussFun
#         nBasis = len(self.parm['Vs'])
#         vs0 = self._bspl(z,nBasis) * self.parm['Vs']
#         vs1 = gaussFun(self.parm['Gauss'][0],self.parm['Gauss'][1],self.parm['Gauss'][2],z)
#         return vs0+vs1

# class OceanMantleBoxCar(OceanMantle):
    def __init__(self,parm={},prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'OceanMantleBoxCar','Group':'mantle'})
    @staticmethod
    def _subdivideInt_positive(N,segLs):
        if len(segLs) > N:
            raise ValueError()
        segLs = np.asarray(segLs)
        def _subdivideInt(N,segLs,noResidual=True):
            if len(segLs) > N:
                raise ValueError()
            segLs = np.asarray(segLs)
            totalL = sum(segLs)
            Ns  = np.zeros(len(segLs),dtype=int)
            res = np.zeros(len(segLs))
            for i in range(len(segLs)):
                tmp = segLs[i]/totalL*N
                Ns[i]  = int(max(1,np.floor(tmp)))
                res[i] = tmp - Ns[i]
            if noResidual:
                for i in np.argsort(res)[::-1][:N-sum(Ns)]:
                    Ns[i]+=1
            return Ns
        Ns = _subdivideInt(N,segLs,noResidual=False)
        Ns[Ns==0] = 1
        if sum(Ns) <= N:
            res = segLs/sum(segLs)*N-Ns
            for i in np.argsort(res)[::-1][:N-sum(Ns)]:
                Ns[i]+=1
        else:
            Ns[Ns>1] -= _subdivideInt(sum(Ns)-N,segLs[Ns>1],noResidual=True)
        return Ns
    def _nFineLayers(self, **kwargs):
            N = super()._nFineLayers(**kwargs)
            return self._subdivideInt_positive(N,[
                self.parm['BoxCar'][0], 
                self.parm['BoxCar'][1]-self.parm['BoxCar'][0],
                self._calH(**kwargs) - self.parm['BoxCar'][1]
                ])
    def seisPropGrids(self,**kwargs):
        N1,N2,N3   = self._nFineLayers(**kwargs)
        # from IPython import embed; embed()
        z = np.concatenate((
            np.linspace(0,self.parm['BoxCar'][0],N1+1),
            np.linspace(self.parm['BoxCar'][0],self.parm['BoxCar'][1],N2+1),
            np.linspace(self.parm['BoxCar'][1],self._calH(**kwargs),N3+1)
        ))
        # z   = np.linspace(0, self._calH(**kwargs), N+1)
        vs = self._calVs(z,**kwargs)
        vp,rho,qs,qp = self._calOthers(z,vs,**kwargs)
        return z,vs,vp,rho,qs,qp
    def _calVs(self, z, **kwargs):
        nBasis = len(self.parm['Vs'])
        vs0 = self._bspl(z,nBasis) * self.parm['Vs']
        N1,N2,N3   = self._nFineLayers(**kwargs)
        vs1 = np.zeros(vs0.shape)
        vs1[N1+1:N1+N2+2] = self.parm['BoxCar'][2]
        return vs0+vs1

'''

layerClassDict = {
    'PureLayer'                         : PureLayer,
    'PureGrid'                          : PureGrid,

    'Sediment'                          : Sediment,
    'Crust'                             : Crust,
    'Mantle'                            : OceanMantle,

    'OceanWater'                        : OceanWater,
    'OceanSediment'                     : OceanSediment,
    'OceanCrust'                        : OceanCrust,
    'OceanMantle'                       : OceanMantle,
    'ReferenceMantle'                   : ReferenceMantle,

    # Juan de Fuca plate
    'OceanSedimentCascadia'             : OceanSedimentCascadia,
    'OceanMantleHybrid'                 : OceanMantleHybrid
}


def buildSeisLayer(parm:dict,layerClass:type[SeisLayer],BrownianConvert=True) -> SeisLayer:
    # more details about the usage of type[SeisLayer]: https://adamj.eu/tech/2021/05/16/python-type-hints-return-class-not-instance/
    if BrownianConvert:
        from pySurfInv.brownian import BrownianVar,BrownianVarMC
        from pySurfInv.utils import _dictIterModifier
        def isNumeric(v):
            try:
                float(v);return True
            except:
                return False
        def toBrownian(v):
            if v[1] in ('fixed','total'):
                return v[0]
            elif v[1] in ('abs','abs_pos','rel','rel_pos'):
                return BrownianVarMC(v[0],ref=v[0],type=v[1],width=v[2],step=v[3])
            elif isNumeric(v[1]):
                return BrownianVar(v[0],v[1],v[2],v[3])
            else:
                raise ValueError(f'Error: Wrong checker??? v={v}')
        def isBrownian(v):
            if type(v) is list:
                if len(v)>=2 and v[1] in ('fixed','total','abs','abs_pos','rel','rel_pos'):
                    return True
                elif len(v) == 4 and isNumeric(v[1]):
                    return True
            return False
        parm = _dictIterModifier(parm,isBrownian,toBrownian)
    try:
        seisLayer = layerClass(parm)
    except Exception as e:  # old version
        raise ValueError(f'Error: Can not load seisLayer {layerClass} {e}')
    return seisLayer
    
if __name__ == '__main__':
    from pySurfInv.brownian import BrownianVar
    a = buildSeisLayer({'H':10,'Vs':[3.2,3.7]},OceanCrust)
    b = buildSeisLayer({'H':10,'Vs':[
        BrownianVar(3.2,3.1,3.3,0.02),
        BrownianVar(3.7,3.5,3.9,0.02)
        ]},OceanCrust)
    c = b.copy()

    a = buildSeisLayer({'H':200,'Vs':[-0.3,-0.2,-0.1],'ThermAge':4},OceanMantleHybrid)
    z,vs,vp,rho,qs,qp = a.seisPropGrids(topDepth=7,hCrust=7)
    from pySurfInv.utils import plotGrid
    plotGrid(z,vs)

    from pySurfInv.utils import plotGrid
    b = buildSeisLayer({'H':200,'Vs':[4.4,4.1,4.2,4.4,4.5],'ThermAge':4},OceanMantle)
    z1,_,_,_,qs1,_ = b.seisPropGrids(topDepth=7)
    fig = plotGrid(z1,qs1)



