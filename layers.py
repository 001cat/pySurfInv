import numpy as np
from copy import deepcopy

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
        import matplotlib.pyplot as plt
        plt.figure();plt.plot(np.linspace(0,1,self.n),self.basis.T)


class SeisLayer():
    def __init__(self,parm,prop={}) -> None:
        self.parm = parm
        self.prop = {'Group':None,'LayerName':None}
        self.prop.update(prop)
        # self.group  = None
        # self._SeisLayerID = None
        # self._tmpInfo = {}
    def seisPropGrids(self,**kwargs):
        return None,None,None,None,None,None
    def seisPropLayers(self,**kwargs):
        z,vs,vp,rho,qs,qp = self.seisPropGrids(kwargs)
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
            return type(v)==BrownianVar
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
    def __init__(self, parm, prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'PureGrid'})

    def seisPropLayers(self,**kwargs):
        h,vs,vp,rho,qs,qp = self.parm['h'],self.parm['vs'],self.parm['vp'],\
                            self.parm['rho'],self.parm['qs'],self.parm['qp']
        return np.array(h),np.array(vs),np.array(vp),np.array(rho),np.array(qs),np.array(qp)

class PureGrid(SeisLayer):
    def __init__(self, parm, prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'PureGrid'})
    def seisPropGrids(self,**kwargs):
        z,vs,vp,rho,qs,qp = self.parm['z'],self.parm['vs'],self.parm['vp'],\
                            self.parm['rho'],self.parm['qs'],self.parm['qp']
        return np.array(z),np.array(vs),np.array(vp),np.array(rho),np.array(qs),np.array(qp)

class OceanWater(SeisLayer):
    def __init__(self,parm,prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'PureGrid','Group':'water'})
        self.parm['Vs'] = 0
    def seisPropGrids(self,**kwargs):
        N   = 1
        z   = np.array([0,self.parm['H']])
        vs  = np.array([0]     * (N+1))
        vp  = np.array([1.475] * (N+1))
        rho = np.array([1.027] * (N+1))
        qs  = np.array([10000] * (N+1) )
        qp  = np.array([57822] * (N+1) )
        return z,vs,vp,rho,qs,qp



class SeisLayerVs(SeisLayer):
    def seisPropGrids(self,**kwargs):
        N   = self._nFineLayers()
        z   = np.linspace(0, self.parm['H'], N+1)
        vs = self._calVs(z,**kwargs)
        vp,rho,qs,qp = self._calOthers(z,vs,**kwargs)
        return z,vs,vp,rho,qs,qp
    def _nFineLayers(self):
        pass
    def _calVs(self,z,**kwargs):
        pass
    def _calOthers(self,z,vs,**kwargs):
        pass
    def _bspl(self,z,nBasis):
        # nBasis = len(self.parm['Vs'])
        deg = 3 + (nBasis>=4)
        if hasattr(self,'_bspl_') and (nBasis == self._bspl_.nBasis) and \
           (deg == self._bspl_.deg) and (len(z) == self._bspl_.n):
           pass
        else:
            self._bspl_ = BsplBasis(z,nBasis,deg)
        return self._bspl_

class OceanSediment(SeisLayerVs):
    def __init__(self,parm,prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'OceanSediment','Group':'sediment'})
    def _nFineLayers(self):
        return 1
    def _calVs(self,z,**kwargs):
        return np.array([self.parm['Vs']] * len(z))
    def _calOthers(self, z, vs, **kwargs):
        vp  = vs*1.23 + 1.28
        rho = 0.541 + 0.3601*vp
        qs  = np.array( [80]  * len(z) )
        qp  = np.array( [160] * len(z) )
        return vp,rho,qs,qp

class OceanSediment_Prism(OceanSediment):
    def __init__(self,parm,prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'OceanSediment_Prism','Group':'sediment'})
    def _nFineLayers(self):
        return  min(max(int(round(self.parm['H']/2)),2),10)
    def _calVs(self, z, **kwargs):
        return np.linspace(0, self.parm['H'], len(z))

class OceanCrust(SeisLayerVs):
    def __init__(self,parm,prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'OceanCrust','Group':'crust'})
    def _nFineLayers(self):
        return min(max(int(round(self.parm['H']/2)),2),10)
    def _calVs(self, z, **kwargs):
        return np.linspace(self.parm['Vs'][0],self.parm['Vs'][1],len(z))
    def _calOthers(self, z, vs, **kwargs):
        vp  = vs*1.8
        rho = 0.541 + 0.3601*vp
        qs  = np.array( [350]  * len(z) )
        qp  = np.array( [1400] * len(z) )
        return vp,rho,qs,qp

class OceanCrust_Prism(OceanCrust):
    def __init__(self,parm,prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'OceanCrust_Prism','Group':'crust'})
    def _nFineLayers(self):
        H = self.parm['H']
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
        return self._bspl(z,nBasis) * self.parm['Vs']

class OceanMantle(SeisLayerVs):
    def __init__(self,parm,prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'OceanMantle','Group':'mantle'})
    def _nFineLayers(self):
        H = self.parm['H']
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
        return self._bspl(z,nBasis) * self.parm['Vs']
    def _calOthers(self, z, vs, **kwargs):
        vp  = vs*1.76
        rho = 3.4268+(vs-4.5)/4.5
        qs  = np.array( [150.]  * len(z) )
        qp  = np.array( [1400.] * len(z) )
        return vp,rho,qs,qp

class LandSediment(SeisLayerVs):
    def __init__(self,parm,prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'LandSediment','Group':'sediment'})
    def _nFineLayers(self):
        return 1
    def _calVs(self,z,**kwargs):
        return np.array([self.parm['Vs']] * len(z))
    def _calOthers(self, z, vs, **kwargs):
        vp  = vs*2.0
        rho = 1.22679 + 1.53201*vs - 0.83668*vs*vs + 0.20673*vs**3 - 0.01656*vs**4
        qs  = np.array( [80]  * len(z) )
        qp  = np.array( [160] * len(z) )
        return vp,rho,qs,qp
    
class LandCrust(SeisLayerVs):
    def __init__(self,parm,prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'LandCrust','Group':'crust'})
    def _nFineLayers(self):
        H = self.parm['H']
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
        return self._bspl(z,nBasis) * self.parm['Vs']
    def _calOthers(self, z, vs, **kwargs):
        vp  = vs*1.80
        rho = 1.22679 + 1.53201*vs - 0.83668*vs*vs + 0.20673*vs**3 - 0.01656*vs**4
        qs  = np.array( [600]  * len(z) )
        qp  = np.array( [1400] * len(z) )
        return vp,rho,qs,qp

class ReferenceMantle(OceanMantle):
    def __init__(self,parm,prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'ReferenceMantle','Group':'mantle'})
    def _nFineLayers(self):
        return 20
    def _calVs(self, z, **kwargs):
        return np.linspace(self.parm['Vs'][0],self.parm['Vs'][1],len(z))


# Cascadia Specified
class OceanSediment_Cascadia(OceanSediment):
    def _calVs(self, z, **kwargs):
        vs = (0.02*self.parm['H']**2+1.27*self.parm['H']+0.29*0.1)/(self.parm['H']+0.29)
        return np.array([vs]* len(z))

class OceanMantle_CascadiaQ(OceanMantle):
    def _calOthers(self, z, vs, z0=None, age=None, **kwargs):
        vp,rho,qs,qp = super()._calOthers(z, vs, **kwargs)
        from pySurfInv.OceanSeis import OceanSeisRuan,HSCM
        seisMod = OceanSeisRuan(HSCM(age=max(1e-3,age),zdeps=z0+z))
        qs = seisMod.qs;qs[qs>5000] = 5000
        return vp,rho,qs,qp
# class Seis_Mantle_Bspline_Ocean_CascadiaQ(Seis_Mantle_Bspline_Ocean):
#     @property
#     def z0(self):
#         return None if 'z0' not in self._tmpInfo.keys() else self._tmpInfo['z0']
#     @property
#     def age(self):
#         return None if 'age' not in self._tmpInfo.keys() else self._tmpInfo['age']
#     @property
#     def seisPropGrids(self):
#         z,vs,vp,rho,qs,qp = super().seisPropGrids

#         from pySurfInv.OceanSeis import OceanSeisRuan,HSCM
#         seisMod = OceanSeisRuan(HSCM(age=max(1e-3,self.age),zdeps=self.z0+z))
#         qs = seisMod.qs;qs[qs>5000] = 5000
#         return z,vs,vp,rho,qs,qp

class Seis_Mantle_Therm_Bspline_Mix_Ocean(SeisLayer):
    def __init__(self, parm) -> None:
        super().__init__(parm)
        self.group = 'mantle'
    @property
    def z0(self):
        return None if 'z0' not in self._tmpInfo.keys() else self._tmpInfo['z0']
    @property
    def hCrust(self):
        return None if 'hCrust' not in self._tmpInfo.keys() else self._tmpInfo['hCrust']
    def bspl(self,z):
        nBasis = len(self.parm['Vs'])+1
        deg = 3 + (nBasis>=4)
        if hasattr(self,'_bspl') and (nBasis == self._bspl.nBasis) and \
        (deg == self._bspl.deg) and (len(z) == self._bspl.n):
            pass
        else:
            self._bspl = BsplBasis(z,nBasis,deg)
        return self._bspl
    
    @property
    def seisPropGrids(self):
        def getN(H):
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
        from pySurfInv.OceanSeis import OceanSeisRitz,OceanSeisRuan,HSCM
        N   = getN(self.parm['H'])
        z   = np.linspace(0, self.parm['H'], N+1)
        seisMod = OceanSeisRitz(HSCM(age=max(1e-3,self.parm['Age']),zdeps=self.hCrust+z))
        def merge(x,y1,y2,xL=None,xK=None,xH=None,s=1):
            def transformD(x,xL=None,xK=None,xH=None):
                d = np.zeros(x.shape)
                xL = min(x) if xL is None else xL
                xH = max(x) if xH is None else xH
                xK = (xH+xL)/2 if xK is None else xK
                d[x<xK]  = (((x[x<xK]-xK)/(xL-xK)))  *(-np.pi/2)
                d[x>=xK] = (((x[x>=xK]-xK)/(xH-xK)))  *(np.pi/2)
                I = (d <= -np.pi/2); d[I] = -np.pi/2
                I = (d >=  np.pi/2); d[I] =  np.pi/2
                return d
            d = transformD(x,xL,xK,xH)
            l = np.tan(d)
            l = l/s
            l[l<-25] = -25
            lam = 1/(1+np.exp(-l))
            return y1*(1-lam)+y2*lam
        def meltStart(age):
            therMod = HSCM(age=age)
            P = therMod.T/1e9
            solidus = -5.1*P**2 + 92.5*P + 1120.6 + 273.15
            try:
                return therMod.zdeps[therMod.T > 1.0*solidus][0]
            except:
                return therMod.zdeps[-1]
        # vs = merge(z,seisMod.vs/1000,self.bspl(z)*np.array([0]+list(self.parm['Vs']))+seisMod.vs/1000,
        #            xL=10,xK=20,xH=40,s=1/3)
        zMelt = meltStart(max(1e-3,self.parm['Age']))-self.hCrust
        vs = merge(z,seisMod.vs/1000,self.bspl(z)*np.array([0]+list(self.parm['Vs']))+seisMod.vs/1000,
                   xL=zMelt,xK=zMelt+10,xH=zMelt+30,s=1/3)
        vp  = vs*1.76
        rho = 3.4268+(vs-4.5)/4.5
        qs  = np.array( [150.]  * (N+1) )
        qp  = np.array( [1400.] * (N+1) )

        seisMod = OceanSeisRuan(HSCM(age=max(1e-3,self.parm['Age']),zdeps=self.z0+z))
        qs = seisMod.qs;qs[qs>5000] = 5000
        return z,vs,vp,rho,qs,qp


typeDict = {
        'PureLayer'                         : PureLayer,
        'PureGrid'                          : PureGrid,
        'OceanWater'                        : OceanWater,
        'OceanSediment'                     : OceanSediment,
        'OceanSediment_Prism'               : OceanSediment_Prism,
        'OceanCrust'                        : OceanCrust,
        'OceanCrust_Prism'                  : OceanCrust_Prism,
        'OceanMantle'                       : OceanMantle,
        'LandSediment'                      : LandSediment,
        'LandCrust'                         : LandCrust,
        'ReferenceMantle'                   : ReferenceMantle,
        # For Cascadia
        'OceanSediment_Cascadia'            : OceanSediment_Cascadia,
        'OceanMantle_CascadiaQ'             : OceanMantle_CascadiaQ,
        'Mantle_Therm_Bspline_Mix'          : Seis_Mantle_Therm_Bspline_Mix_Ocean
    }
oldTypeDict = { # to convert previous layer notes to new layer type id, type_mtype_stype: new type ID
        'water_water_'              : 'OceanWater',
        'sediment_constant_'        : 'OceanSediment',
        'sediment_linear_'          : 'OceanSediment_Prism',
        'crust_linear_'             : 'OceanCrust',
        'mantle_Bspline_'           : 'OceanMantle', # OceanMantle_CascadiaQ if Qmodel: Ruan2018, see models.loadSetting.dictReshape
        'crust_Bspline_'            : 'OceanCrust_Prism',
        'sediment_linear_land'      : 'LandSediment'
        # 'crust_Bspline_land'      : 'Crust_Bspline_Land'
    }

def buildSeisLayer(parm,typeID):
    try:
        seisLayer = typeDict[typeID](parm)
        return seisLayer
    except:
        raise ValueError()
    
if __name__ == '__main__':
    # from brownian import BrownianVar
    # a = buildSeisLayer({'H':10,'Vs':[3.2,3.7]},'OceanCrust')
    # b = buildSeisLayer({'H':10,'Vs':[
    #     BrownianVar(3.2,3.1,3.3,0.02),
    #     BrownianVar(3.7,3.5,3.9,0.02)
    #     ]},'OceanCrust')
    # c = b.copy()

    # a = buildSeisLayer({'H':200,'Vs':[0,0,0],'Age':4},'Mantle_Therm_Bspline_Mix')
    # a.z0,a.hCrust = 7,7
    # z,vs,vp,rho,qs,qp = a.seisPropGrids
    # from pySurfInv.utils import plotGrid
    # plotGrid(z,vs)

    from pySurfInv.utils import plotGrid
    b = buildSeisLayer({'H':200,'Vs':[4.4,4.1,4.2,4.4,4.5]},'OceanMantle_CascadiaQ')
    b.z0 = 7; b.age = 4
    z1,_,_,_,qs1,_ = b.seisPropGrids
    b.test = True
    z2,_,_,_,qs2,_ = b.seisPropGrids
    fig = plotGrid(z1,qs1)
    plotGrid(z2,qs2,fig=fig)



