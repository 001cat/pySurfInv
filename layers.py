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
    def H(self,**kwargs):
        if self.parm.get('BottomDepth',None) == None:
            H = self.parm['H']
        else:
            H = self.parm['BottomDepth'] - kwargs['topDepth']
        return H

class PureLayer(SeisLayer):
    def __init__(self, parm, prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'PureGrid'})

    def seisPropLayers(self,**kwargs):
        h,vs,vp,rho,qs,qp = self.parm['h'],self.parm['vs'],self.parm['vp'],\
                            self.parm['rho'],self.parm['qs'],self.parm['qp']
        return np.array(h),np.array(vs),np.array(vp),np.array(rho),np.array(qs),np.array(qp)
    def H(self,**kwargs):
        return self.parm['h'].sum()

class PureGrid(SeisLayer):
    def __init__(self, parm, prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'PureGrid'})
    def seisPropGrids(self,**kwargs):
        z,vs,vp,rho,qs,qp = self.parm['z'],self.parm['vs'],self.parm['vp'],\
                            self.parm['rho'],self.parm['qs'],self.parm['qp']
        return np.array(z),np.array(vs),np.array(vp),np.array(rho),np.array(qs),np.array(qp)
    def H(self, **kwargs):
        return self.parm['z'][-1] - self.parm['z'][0]

class OceanWater(SeisLayer):
    def __init__(self,parm,prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'OceanWater','Group':'water'})
        self.parm['Vs'] = 0
    def seisPropGrids(self,**kwargs):
        N   = 1
        z   = np.array([0,self.H(**kwargs)])
        vs  = np.array([0]     * (N+1))
        vp  = np.array([1.475] * (N+1))
        rho = np.array([1.027] * (N+1))
        qs  = np.array([10000] * (N+1) )
        qp  = np.array([57822] * (N+1) )
        return z,vs,vp,rho,qs,qp



class SeisLayerVs(SeisLayer):
    def seisPropGrids(self,**kwargs):
        N   = self._nFineLayers(**kwargs)
        z   = np.linspace(0, self.H(**kwargs), N+1)
        vs = self._calVs(z,**kwargs)
        vp,rho,qs,qp = self._calOthers(z,vs,**kwargs)
        return z,vs,vp,rho,qs,qp
    def _nFineLayers(self,**kwargs):
        pass
    def _calVs(self,z,**kwargs):
        pass
    def _calOthers(self,z,vs,**kwargs):
        pass
    def _bspl(self,z,nBasis):
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

class OceanSediment_Prism(OceanSediment):
    def __init__(self,parm,prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'OceanSediment_Prism','Group':'sediment'})
    def _nFineLayers(self,**kwargs):
        return  min(max(int(round(self.H(**kwargs)/2)),2),10)
    def _calVs(self, z, **kwargs):
        return np.linspace(0, self.H(**kwargs), len(z))

class OceanCrust(SeisLayerVs):
    def __init__(self,parm,prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'OceanCrust','Group':'crust'})
    def _nFineLayers(self,**kwargs):
        return min(max(int(round(self.H(**kwargs)/2)),2),10)
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
    def _nFineLayers(self,**kwargs):
        H = self.H(**kwargs)
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
    def _nFineLayers(self,**kwargs):
        H = self.H(**kwargs)
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
    def _nFineLayers(self,**kwargs):
        return 1
    def _calVs(self,z,**kwargs):
        return np.linspace(self.parm['Vs'][0],self.parm['Vs'][1],len(z))
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
    def _nFineLayers(self,**kwargs):
        H = self.H(**kwargs)
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
    def _nFineLayers(self,**kwargs):
        return 20
    def _calVs(self, z, **kwargs):
        return np.linspace(self.parm['Vs'][0],self.parm['Vs'][1],len(z))


# Cascadia Specified
class OceanSediment_Cascadia(OceanSediment):
    def __init__(self, parm, prop={}) -> None:
        super().__init__(parm, prop)
        self.prop.update({'LayerName':'OceanSediment_Cascadia','Group':'sediment'})
    def _calVs(self, z, **kwargs):
        vs = (0.02*self.H(**kwargs)**2+1.27*self.H(**kwargs)+0.29*0.1)/(self.H(**kwargs)+0.29)
        return np.array([vs]* len(z))

class OceanMantle_CascadiaQ(OceanMantle):
    def __init__(self, parm, prop={}) -> None:
        super().__init__(parm, prop)
        self.prop.update({'LayerName':'OceanMantle_CascadiaQ','Group':'mantle'})
    def _calOthers(self, z, vs, topDepth=None, period=1, Qage=None, **kwargs):
        vp,rho,qs,qp = super()._calOthers(z, vs, **kwargs)
        from pySurfInv.OceanSeis import OceanSeisRuan,HSCM
        Qage = self.parm['ThermAge'] if Qage is None else Qage
        seisMod = OceanSeisRuan(HSCM(age=max(1e-3,Qage),zdeps=topDepth+z),period=period)
        qs = seisMod.qs;qs[qs>5000] = 5000
        return vp,rho,qs,qp

class OceanMantle_CascadiaQ_20220305SingleLayerClass(OceanMantle):
    def __init__(self, parm, prop={}) -> None:
        super().__init__(parm, prop)
        self.prop.update({'LayerName':'OceanMantle_CascadiaQ_20220305SingleLayerClass','Group':'mantle'})
    def _calOthers(self, z, vs, topDepth=None, **kwargs):
        vp,rho,qs,qp = super()._calOthers(z, vs, **kwargs)
        from pySurfInv.OceanSeis import OceanSeisRuan,HSCM
        seisMod = OceanSeisRuan(HSCM(age=max(1e-3,self.parm['ThermAge']),zdeps=topDepth+z),period=1)
        qs = seisMod.qs
        return vp,rho,qs,qp

class OceanCrust_Therm(OceanCrust):
    def __init__(self,parm,prop={}) -> None:
        super().__init__(parm,prop)
        self.prop.update({'LayerName':'OceanCrust_Therm','Group':'crust'})
    def _calVs(self, z, Tage, **kwargs):
        from pySurfInv.OceanSeis import HSCM
        mod0 = HSCM(4.0); T0_bottom = np.interp(self.parm['H'],mod0.zdeps,mod0.T)
        mod  = HSCM(age=max(1e-3,Tage)); T_bottom = np.interp(self.parm['H'],mod.zdeps,mod.T)
        v1 = self.parm['Vs'][0]
        v2 = self.parm['Vs'][1] - max(0,(T_bottom-T0_bottom)*0.000378) # assume self.parm['Vs'][1] is for age >= 4 Ma
        return np.linspace(v1,v2,len(z))

class OceanMantle_ThermBsplineHybrid(OceanMantle_CascadiaQ):
    def __init__(self, parm, prop={}) -> None:
        super().__init__(parm, prop)
        self.prop.update({'LayerName':'OceanMantle_ThermBsplineHybrid','Group':'mantle'})
    def _calVs(self, z, topDepth, hCrust, **kwargs):
        nBasis = len(self.parm['Vs']) + 1
        from pySurfInv.OceanSeis import OceanSeisRitz,OceanSeisRuan,HSCM
        Tp = self.parm.get('Tp',1325)
        seisID = self.parm.get('SeisMod','Yamauchi')
        if seisID == 'Yamauchi':
            seisMod = OceanSeisRuan(HSCM(age=max(1e-3,self.parm['ThermAge']),zdeps=hCrust+z,Tp=Tp),
                                    damp=True,YaTaJu=True,period=1)
        elif seisID == 'Ritzwoller':
            seisMod = OceanSeisRitz(HSCM(age=max(1e-3,self.parm['ThermAge']),zdeps=hCrust+z,Tp=Tp))
        else:
            raise ValueError(f'Invalid SeisMod: {seisID}')

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
            d[d>1.53] = 1.53; d[d<-1.53] = -1.53            # for speeding up
            l = np.tan(d)
            l = l/s
            l[l<-25] = -25
            lam = 1/(1+np.exp(-l))
            return y1*(1-lam)+y2*lam
        def merge2(x,y1,y2,xL,xH):
            vss = list(y1[x<xL]) + list(y2[x>xH])
            xs  = list(x[x<xL])  + list(x[x>xH])
            from scipy.interpolate import CubicSpline
            return CubicSpline(xs, vss)(x)

        def meltStart(age):
            therMod = HSCM(age=age)
            P = therMod.P/1e9
            solidus = -5.1*P**2 + 92.5*P + 1120.6 + 273.15  # damp solidus
            try:
                return therMod.zdeps[therMod.T > 0.92*solidus][0]
            except:
                return therMod.zdeps[-1]
        # vs = merge(z,seisMod.vs/1000,self.bspl(z)*np.array([0]+list(self.parm['Vs']))+seisMod.vs/1000,
        #            xL=10,xK=20,xH=40,s=1/3)
        zMelt = meltStart(max(1e-3,self.parm['ThermAge']))-hCrust
        # vs = merge(z,seisMod.vs/1000,self._bspl(z,nBasis)*np.array([0]+list(self.parm['Vs']))+seisMod.vs/1000,
        #            xL=zMelt,xK=zMelt+10,xH=zMelt+30,s=1/3)
        vs = merge2(z,seisMod.vs/1000,self._bspl(z,nBasis)*np.array([0]+list(self.parm['Vs']))+seisMod.vs/1000,
                   xL=zMelt,xH=(zMelt+hCrust)*1.7-hCrust)
                   
        self._debug_zMelt = zMelt
        return vs

class OceanMantle_ThermBsplineHybridConstQ(OceanMantle_ThermBsplineHybrid):
    def _calOthers(self, z, vs, topDepth=None, period=1, **kwargs):
        vp,rho,_,_ = super()._calOthers(z, vs, topDepth=topDepth, period=period, **kwargs)
        qs  = np.array( [self.parm.get('Qs',150)]  * len(z) )
        qp  = np.array( [1400.] * len(z) )
        return vp,rho,qs,qp

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
        'OceanCrust_Therm'                  : OceanCrust_Therm
        'OceanSediment_Cascadia'            : OceanSediment_Cascadia,
        'OceanMantle_CascadiaQ'             : OceanMantle_CascadiaQ,
        'OceanMantle_CascadiaQ_compatible'  : OceanMantle_CascadiaQ_20220305SingleLayerClass,
        'OceanMantle_ThermBsplineHybrid'    : OceanMantle_ThermBsplineHybrid,
        'OceanMantle_ThermBsplineHybridConstQ': OceanMantle_ThermBsplineHybridConstQ
    }
oldTypeDict = { # to convert previous layer notes to new layer type id, type_mtype_stype: new type ID
        'water_water_'              : 'OceanWater',
        'sediment_constant_'        : 'OceanSediment',
        'sediment_linear_'          : 'OceanSediment_Prism',
        'crust_linear_'             : 'OceanCrust',
        'mantle_Bspline_'           : 'OceanMantle',
        'mantle_Bspline_Ruan'       : 'OceanMantle_CascadiaQ_compatible',
        'crust_Bspline_'            : 'OceanCrust_Prism',
        'sediment_linear_land'      : 'LandSediment'
        # 'crust_Bspline_land'      : 'Crust_Bspline_Land'
    }

def buildSeisLayer(parm:dict,typeID,BrownianConvert=True) -> SeisLayer:
    if BrownianConvert:
        from pySurfInv.brownian import BrownianVar
        from pySurfInv.utils import _dictIterModifier
        def isNumeric(v):
            try:
                float(v);return True
            except:
                return False
        def toBrownian(v):
            if v[1] in ('fixed','total'):
                return v[0]
            elif v[1] == 'abs':
                return BrownianVar(v[0],v[0]-v[2],v[0]+v[2],v[3])
            elif v[1] == 'abs_pos':
                return BrownianVar(v[0],max(v[0]-v[2],0),v[0]+v[2],v[3])
            elif v[1] == 'rel':
                return BrownianVar(v[0],v[0]*(1-v[2]/100),v[0]*(1+v[2]/100),v[3])
            elif v[1] == 'rel_pos':
                return BrownianVar(v[0],max(v[0]*(1-v[2]/100),0),v[0]*(1+v[2]/100),v[3])
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
        seisLayer = typeDict[typeID](parm)
    except:  # old version
        try:
            mtype,stype = parm.get('mtype',''),parm.get('stype','')
            typeID = oldTypeDict['_'.join([typeID,mtype,stype])]
            parmNew = {}
            for k,v in parm.items():
                parmNew[k[0].upper()+k[1:]] = v
            seisLayer = typeDict[typeID](parmNew)
        except Exception as e:
            raise ValueError(f'Error: Can not load seisLayer {typeID} {e}')
    return seisLayer
    
if __name__ == '__main__':
    from pySurfInv.brownian import BrownianVar
    a = buildSeisLayer({'H':10,'Vs':[3.2,3.7]},'OceanCrust')
    b = buildSeisLayer({'H':10,'Vs':[
        BrownianVar(3.2,3.1,3.3,0.02),
        BrownianVar(3.7,3.5,3.9,0.02)
        ]},'OceanCrust')
    c = b.copy()

    a = buildSeisLayer({'H':200,'Vs':[-0.3,-0.2,-0.1],'ThermAge':4},'OceanMantle_ThermBsplineHybrid')
    z,vs,vp,rho,qs,qp = a.seisPropGrids(topDepth=7,hCrust=7)
    from pySurfInv.utils import plotGrid
    plotGrid(z,vs)

    from pySurfInv.utils import plotGrid
    b = buildSeisLayer({'H':200,'Vs':[4.4,4.1,4.2,4.4,4.5],'ThermAge':4},'OceanMantle_CascadiaQ')
    z1,_,_,_,qs1,_ = b.seisPropGrids(topDepth=7)
    fig = plotGrid(z1,qs1)



