import glob
import numpy as np
from copy import deepcopy
from Triforce.pltHead import *
from pySurfInv.layers import buildSeisLayer
from pySurfInv.utils import plotLayer,plotGrid,_dictIterModifier
from pySurfInv.brownian import BrownianVar

def _calForward(inProfile,wavetype='Ray',periods=[5,10,20,40,60,80]):
    import pySurfInv.fast_surf as fast_surf
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





class Model1D():
    def __init__(self,layers=[],info=None) -> None:
        self._layers = layers
        self.info   = info
    def loadYML(self,ymlFile,localInfo={}):
        if type(ymlFile) is not dict:
            import yaml
            with open(ymlFile, 'r') as f:
                ymlFile = yaml.load(f,Loader=yaml.FullLoader)
        ymlD = deepcopy(ymlFile)
        self.info = ymlD.pop('Info')

        # to be compatible
        if self.info.get('Qmodel',None) == 'Ruan2018':
            for k,v in ymlD.items():
                if '_'.join((k,v['mtype'],v.get('stype',''))) == 'mantle_Bspline_':
                    v['stype'] = 'Ruan'
                    v['ThermAge'] = self.info['lithoAge']
        for k,v in ymlD.items():
            for par in v.keys():
                try:
                    if par.upper() == 'H' and v[par][1].lower() == 'total':
                        v['BottomDepth'] = v.pop(par)[0]
                        break
                except:
                    continue
        layersD = self._loadLocalInfo(ymlD,localInfo)
        self._layers = [buildSeisLayer(parm,typeID) for typeID,parm in layersD.items()]

    def _loadLocalInfo(self,layersD,localInfo):
        self.info.update(localInfo)
        '''
        To Be Specified in Child Class: how local information modifies layers
        '''
        return layersD

    def toYML(self):
        def checker(v):
            return isinstance(v,BrownianVar)
        def modifier(v):
            return [v.v,v.vmin,v.vmax,v.step]
        ymlDict = {}
        for layer in self.layers:
            ymlDict[layer.prop['LayerName']] = _dictIterModifier(layer.parm,checker,modifier)
        ymlDict['Info'] = self.info
        return deepcopy(ymlDict)

    def seisPropGrids(self,refLayer=False,_layerName=False):
        layers = self.layers.copy(); layers += [self._refLayer.copy()] if refLayer else []
        z0 = -max(self.info.get('topo',0),0)
        z,vs,vp,rho,qs,qp,grp,layerName = [],[],[],[],[],[],[],[]
        for layer in layers:
            z1,vs1,vp1,rho1,qs1,qp1 = layer.seisPropGrids(
                layerInfo =[z,vs,vp,rho,qs,qp,grp,layerName],
                modelInfo = self.info)
            if z1[-1]-z1[0] < 0.01:
                continue
            z += list(z1+z0)
            vs += list(vs1); vp += list(vp1); rho += list(rho1); qs += list(qs1); qp += list(qp1)
            grp         += [layer.prop['Group']]*len(z1)
            layerName   += [layer.prop['LayerName']]*len(z1)
            z0 = z[-1]
        if _layerName:
            return np.array(z),np.array(vs),np.array(vp),np.array(rho),\
                   np.array(qs),np.array(qp),grp,layerName
        else:
            return np.array(z),np.array(vs),np.array(vp),np.array(rho),\
                   np.array(qs),np.array(qp),grp


    def seisPropGrids_deprecated(self,refLayer=False):
        z0 = -max(self.info.get('topo',0),0)
        z,vs,vp,rho,qs,qp,grp = [],[],[],[],[],[],[]
        for layer in self.layers:
            z1,vs1,vp1,rho1,qs1,qp1 = layer.seisPropGrids(topDepth=z0)
            z += list(z1+z0)
            vs += list(vs1); vp += list(vp1); rho += list(rho1); qs += list(qs1); qp += list(qp1)
            grp += [layer.prop['Group']]*len(z1)
            z0 = z[-1]
        if refLayer:
            refLayer = self._refLayer.copy()
            refLayer.parm['Vs'][0] += vs[-1]
            refLayer.parm['Vs'][1] += vs[-1]
            z1,vs1,vp1,rho1,qs1,qp1 = refLayer.seisPropGrids(topDepth=z0)
            vs1 += [vs[-1]-vs1[0]]; vp1 += [vp[-1]-vp1[0]]; rho1 += [rho[-1]-rho1[0]]
            qs1 += [qs[-1]-qs1[0]]; qp1 += [qp[-1]-qp1[0]]
            z += list(z1+z0)
            vs += list(vs1); vp += list(vp1); rho += list(rho1); qs += list(qs1); qp += list(qp1)
            grp += [refLayer.prop['Group']]*len(z1)
        return np.array(z),np.array(vs),np.array(vp),np.array(rho),np.array(qs),np.array(qp),grp
    def seisPropLayers(self,refLayer=False):
        z,vs,vp,rho,qs,qp,grp = self.seisPropGrids(refLayer)
        h = np.diff(z)
        vs  = (vs[1:] + vs[:-1])/2
        vp  = (vp[1:] + vp[:-1])/2
        rho = (rho[1:]+ rho[:-1])/2
        qs  = (qs[1:] + qs[:-1])/2
        qp  = (qp[1:] + qp[:-1])/2
        grp = grp[:-1]
        return h[h>0.01],vs[h>0.01],vp[h>0.01],rho[h>0.01],qs[h>0.01],qp[h>0.01],list(np.array(grp)[h>0.01])
    def forward(self,periods=[5,10,20,40,60,80]):
        refLayer = self.info.get('refLayer',False)
        pred = _calForward(np.array(self.seisPropLayers(refLayer=refLayer)[:-1]),wavetype='Ray',periods=periods)
        if pred is None:
            print(f'Warning: Forward not complete! Model listed below:')
            self.show()
        return pred

    def value(self,zdeps,type='vs'):
        if type != 'vs': 
            raise ValueError('Error: only support vs, others to be added...')
        z,vs,vp,rho,qs,qp,grp = self.seisPropGrids()
        return np.interp(zdeps,z,vs,left=np.nan,right=np.nan)
    def moho(self):
        z,vs,vp,rho,qs,qp,grp = self.seisPropGrids()
        return z[grp.index('mantle')]
    def show(self):
        for layer in self.layers:
            print(layer.prop['Group'])
            print(layer.parm)
    
    def plotProfile(self,type='vs',**kwargs):
        h,vs,vp,rho,qs,qp,_ = self.seisPropLayers()
        if type == 'vs':
            ax = plotLayer(h,vs,**kwargs);plt.title('Vs')
        else:
            print('To be added...')
        return ax
    def plotProfileGrid(self,type='vs',ax=None,**kwargs):
        z,vs,vp,rho,qs,qp,_ = self.seisPropGrids(refLayer=False)
        if type == 'vs':
            ax = plotGrid(z,vs,ax=ax,**kwargs);plt.title('Vs')
        else:
            print('To be added...')
        return ax

    @property
    def _refLayer(self):
        return buildSeisLayer({'H':300,'Slope':0.35/200},'ReferenceMantle')
    @property
    def layers(self):
        return self._layers

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

class Model1D_Puregird(Model1D):
    def __init__(self, inProfiles, info=None) -> None:
        from pySurfInv.layers import PureGrid
        parm = {}
        parm['z'],parm['vs'],parm['vp'],\
        parm['rho'],parm['qs'],parm['qp'],\
        grps = inProfiles
        self._layers = []
        for grp in list(dict.fromkeys(grps)):
            I = np.array(grps) == grp
            parmLayer = {}
            for k,v in parm.items():
                parmLayer[k] = v[I]
            self._layers.append(PureGrid(parmLayer,prop={'Group':grp}))
        self.info = info
    def loadYML(self, ymlFile, localInfo={}):
        raise AttributeError('"Model1D_Puregird" object has no method "loadYML"')
    def seisPropGrids_deprecated(self,refLayer=False):
        z,vs,vp,rho,qs,qp,grp = [],[],[],[],[],[],[]
        for layer in self.layers:
            z1,vs1,vp1,rho1,qs1,qp1 = layer.seisPropGrids()
            z += list(z1)
            vs += list(vs1); vp += list(vp1); rho += list(rho1); qs += list(qs1); qp += list(qp1)
            grp += [layer.prop['Group']]*len(z1)
        if refLayer:
            refLayer = self._refLayer.copy()
            refLayer.parm['Vs'][0] += vs[-1]
            refLayer.parm['Vs'][1] += vs[-1]
            z1,vs1,vp1,rho1,qs1,qp1 = refLayer.seisPropGrids()
            vs1 += [vs[-1]-vs1[0]]; vp1 += [vp[-1]-vp1[0]]; rho1 += [rho[-1]-rho1[0]]
            qs1 += [qs[-1]-qs1[0]]; qp1 += [qp[-1]-qp1[0]]
            z += list(z1)
            vs += list(vs1); vp += list(vp1); rho += list(rho1); qs += list(qs1); qp += list(qp1)
            grp += [refLayer.prop['Group']]*len(z1)
        return np.array(z),np.array(vs),np.array(vp),np.array(rho),np.array(qs),np.array(qp),grp
    @property
    def layers(self):
        return self._layers

class Model1D_MCinv(Model1D):
    def _loadMC(self,mc): 
        mc_ind = 0
        for layer in self.layers:
            for k,v in layer.parm.items():
                if isinstance(v,BrownianVar):
                    layer.parm[k] = v._setValue(mc[mc_ind]);mc_ind += 1
                elif type(v) == list:
                    for i in range(len(v)):
                        if isinstance(v[i],BrownianVar):
                            v[i] = v[i]._setValue(mc[mc_ind]);mc_ind += 1
                    layer.parm[k] = v
    def _brownians(self,numberOnly=True):
        brownians = []
        for layer in self.layers:
            for k,v in layer.parm.items():
                if type(v) is list:
                    for e in v:
                        if isinstance(e,BrownianVar):
                            brownians.append([float(e),layer.prop['Group'],k])
                else:
                    if isinstance(v,BrownianVar):
                        brownians.append([float(v),layer.prop['Group'],k])
        if numberOnly:
            brownians = [v[0] for v in brownians]
        return brownians
    def _dump(self,index,target,preInfo=[]):
        preInfo.extend(self._brownians())
        target[index] = preInfo

    def perturb(self,isgood=None,verbose=False):
        if isgood is None:
            def isgood(model):
                return model.isgood()
        for i in range(1000):
            newModel = self.copy()
            newModel._layers = [l._perturb() for l in self.layers]
            if isgood(newModel):
                if verbose:
                    print(f'Perturb at {i}')
                return newModel
        return self.reset(isgood=isgood,verbose=verbose)
    def reset(self,isgood=None,verbose=False):
        if isgood is None:
            def isgood(model):
                return model.isgood()
        for i in range(10000):
            newModel = self.copy()
            newModel._layers = [l._reset() for l in self.layers]
            if isgood(newModel):
                if verbose:
                    print(f'Reset at {i}')
                return newModel
        print(f'Cound not find a good model through reset after {i+1} iter.')
        self.show()
        raise RuntimeError(f'Error: Cound not find a good model through reset.')
    def isgood(self):
        '''
        To Be Specified in Child Class: prioris (prior constrains)
        '''
        return True

class Model1D_Cascadia_Oceanic(Model1D_MCinv):
    def _loadLocalInfo(self, layersD, localInfo):
        super()._loadLocalInfo(layersD, localInfo)
        layersD = deepcopy(layersD)
        layersK = list(layersD.keys())
        grps = [buildSeisLayer(parm,typeID).prop['Group'] for typeID,parm in layersD.items()]

        topo = localInfo.get('topo',self.info.get('topo',0))
        waterH = max(-topo,0)
        if waterH > 0 and 'water' in grps:
            try:
                layersD[layersK[grps.index('water')]]['H'][0] = waterH
            except:
                layersD[layersK[grps.index('water')]]['H'] = waterH
        elif waterH == 0 and 'water' in grps:
            del layersD[layersK[grps.index('water')]]

        if 'sedthk' in localInfo.keys():
            try:
                layersD[layersK[grps.index('sediment')]]['H'][0] = localInfo['sedthk']
            except:
                layersD[layersK[grps.index('sediment')]]['H'] = localInfo['sedthk']
        
        if 'crsthk' in localInfo.keys():
            try:
                layersD[layersK[grps.index('crust')]]['H'][0] = localInfo['crsthk']
            except:
                layersD[layersK[grps.index('crust')]]['H'] = localInfo['crsthk']
        
        if 'lithoAge' in localInfo.keys():
            try:
                layersD[layersK[grps.index('mantle')]]['ThermAge'][0] = localInfo['lithoAge']
                if layersD[layersK[grps.index('mantle')]]['ThermAge'][1] in ('rel_pos','rel') and \
                   layersD[layersK[grps.index('mantle')]]['ThermAge'][0] < 2 :
                    layersD[layersK[grps.index('mantle')]]['ThermAge'][1] = 'abs_pos'
                    layersD[layersK[grps.index('mantle')]]['ThermAge'][2] = 2*layersD[layersK[grps.index('mantle')]]['ThermAge'][2]/100
            except:
                layersD[layersK[grps.index('mantle')]]['ThermAge'] = localInfo['lithoAge']


        if 'mantleInitParmVs' in localInfo.keys():
            for i,vs in enumerate(localInfo['mantleInitParmVs']):
                try:
                    layersD[layersK[grps.index('mantle')]]['Vs'][i][0] = vs
                except:
                    layersD[layersK[grps.index('mantle')]]['Vs'][i] = vs
            self.info.pop('mantleInitParmVs')
        
        return layersD

    def seisPropGrids_deprecated(self,refLayer=False):
        period = self.info.get('period',1)
        Qage = self.info.get('lithoAge',None) if self.info.get('lithoAgeQ',False) else None
        z0 = -max(self.info.get('topo',0),0)
        hCrust = np.sum([l.H() if l.prop['Group'] == 'crust' else 0 for l in self.layers])
        z,vs,vp,rho,qs,qp,grp = [],[],[],[],[],[],[]
        for layer in self.layers:
            z1,vs1,vp1,rho1,qs1,qp1 = layer.seisPropGrids(
                topDepth=z0,hCrust=hCrust,period=period,Qage=Qage)
            z += list(z1+z0)
            vs += list(vs1); vp += list(vp1); rho += list(rho1); qs += list(qs1); qp += list(qp1)
            grp += [layer.prop['Group']]*len(z1)
            z0 = z[-1]
        if refLayer:
            refLayer = self._refLayer.copy()
            refLayer.parm['Vs'][0] += vs[-1]
            refLayer.parm['Vs'][1] += vs[-1]
            z1,vs1,vp1,rho1,qs1,qp1 = refLayer.seisPropGrids(topDepth=z0)
            vs1 += [vs[-1]-vs1[0]]; vp1 += [vp[-1]-vp1[0]]; rho1 += [rho[-1]-rho1[0]]
            qs1 += [qs[-1]-qs1[0]]; qp1 += [qp[-1]-qp1[0]]
            z += list(z1+z0)
            vs += list(vs1); vp += list(vp1); rho += list(rho1); qs += list(qs1); qp += list(qp1)
            grp += [refLayer.prop['Group']]*len(z1)
        return np.array(z),np.array(vs),np.array(vp),np.array(rho),np.array(qs),np.array(qp),grp
    
    def isgood(self,verbose=False):
        import scipy.signal
        def monoIncrease(a,eps=np.finfo(float).eps):
            return np.all(np.diff(a)>=0)
        
        h,vs,vp,rho,qs,qp,grp = self.seisPropLayers();grp = np.array(grp)
        vsSediment = vs[grp=='sediment']
        vsCrust = vs[grp=='crust']
        vsMantle = vs[grp=='mantle']

        ''' 
        Vs in sediment > 0.2; from Lili's code 
        '''
        if np.any(vsSediment<0.2):
            return False

        '''
        Vs jump between group is positive, contraint (5) in 4.2 of Shen et al., 2012
        '''
        for i in np.where((grp[1:]!=grp[:-1]))[0]:
            if vs[i+1] < vs[i]:
                return False

        '''
        All Vs < 4.9km/sec, contraint (6) in 4.2 of Shen et al., 2012
        '''
        # if np.any(vs > 4.9):
        #     return False

        '''
        Velocity in sediment and crust layers must increase with depth
        '''
        if not monoIncrease(vs[grp=='sediment']):
            return False
        if not monoIncrease(vs[grp=='crust']): 
            return False

        '''
        Negative velocity gradient below moho, works if
        'NegSlopeBelowCrust'=True in setting['Info']
        '''
        if self.info.get('NegSlopeBelowCrust',False) is False:
            pass
        elif self.info.get('noNegSlopeBelowCrust',True) is True:
            pass
        elif not vsMantle[1]<vsMantle[0]:
            return False
    
        '''
        Vs in crust < 4.3
        '''
        # if np.any(vsCrust > 4.3):
        #     return False

        '''
        Vs at first fine layer in mantle is between 4.0 and 4.6
        '''
        # if vsMantle[0] < 4.0 or vsMantle[0] > 4.6:
        #     return False

        ''' 
        Vs at last fine layer in mantle > 4.3 
        '''
        # if vsMantle[-1] < 4.3:
        #     return False

        '''
        Vs > 4.0 below 80km
        '''
        # if np.any(vsMantle < 4.0):
        #     return False

        '''
        Change in mantle < 15%, works if 'LargeMantleChange' = False, made by Ayu
        '''
        if self.info.get('LargeMantleChange',True) is True:
            pass
        else:
            if (vsMantle.max() - vsMantle.min()) > 0.15*vsMantle.mean():
                return False

        '''
        Oscillation Limit, the difference between nearby local extrema < 10% of mean vs in mantle
        '''
        osciLim = 0.1*vsMantle.mean()
        indLocMax = scipy.signal.argrelmax(vsMantle)[0]
        indLocMin = scipy.signal.argrelmin(vsMantle)[0]
        if len(indLocMax) + len(indLocMin) > 1:
            indLoc = np.sort(np.append(indLocMax,indLocMin))
            osci = abs(np.diff(vsMantle[indLoc]))
            if len(np.where(osci > osciLim)[0]) >= 1:   # origin >= 1
                return False

        # new constrains based on grids instead of layers
        z,vs,_,_,_,_,grp = self.seisPropGrids()
        iMantle = np.array(grp) == 'mantle'
        z,vs = z[iMantle],vs[iMantle]

        '''
        No local maximum
        '''
        if self.layers[-1].prop['LayerName'] == 'OceanMantle_ThermBsplineHybrid':
            if len(scipy.signal.argrelmax(vs)[0]) > 0:
                if verbose:
                    print('Debug: shallow local maximum found')
                return False

        '''
        No extreme velocity decrease below moho
        '''
        if self.layers[-1].prop['LayerName'] == 'OceanMantle_ThermBsplineHybrid':
            slope = np.diff(vs)/np.diff(z)
            if slope.min() < slope[0]*1.5:  # should be between 1.5 to 2.2
                return False
                
        '''
        osci limitation using cwt
        '''
        # if self.layers[-1].prop['LayerName'] == 'OceanMantle_ThermBsplineHybrid':
        cwtWidth = 30//(z[1]-z[0])
        cwt = scipy.signal.cwt(vs - np.interp(z,[z[0],z[-1]],[vs[0],vs[-1]]),
                            scipy.signal.ricker,[cwtWidth])[0]
        indLocMax = scipy.signal.argrelmax(cwt)[0]
        indLocMin = scipy.signal.argrelmin(cwt)[0]
        indLoc = np.sort(np.append(indLocMax,indLocMin))
        osci = abs(np.diff(cwt[indLoc]))
        if np.any(osci > 0.3):
            return False

        '''
        velocity increase at bottom
        '''
        if (vs[-1]-vs[-2])/(z[-1]-z[-2]) <= 0:
            return False

        # temporary only
        # if len(indLocMin) > 1:
        #     return False

        return True

class Model1D_Cascadia_Prism(Model1D_MCinv):
    def _loadLocalInfo(self, layersD, localInfo):
        super()._loadLocalInfo(layersD, localInfo)
        layersD = deepcopy(layersD)
        layersGrp = {buildSeisLayer(parm,typeID).prop['Group']:typeID for typeID,parm in layersD.items()}

        # water layer thickness
        waterH = max(-localInfo.get('topo',self.info.get('topo',0)),0)
        if 'water' in layersGrp.keys():
            if waterH > 0:
                layersD[layersGrp['water']]['H'] = waterH
            elif waterH == 0:
                del layersD[layersGrp['water']]
        
        # sediment thickness 
        if 'sedthk' in localInfo.keys():
            try:
                layersD[layersGrp['sediment']]['H'][0] = localInfo['sedthk']
            except:
                layersD[layersGrp['sediment']]['H'] = localInfo['sedthk']
        
        # crust thickness 
        if 'prismthk' in localInfo.keys():
            try:
                layersD[layersGrp['prism']]['H'][0] = localInfo['prismthk']
            except:
                layersD[layersGrp['prism']]['H'] = localInfo['prismthk']
        
        return layersD
    def isgood(self,verbose=False):
        import scipy.signal
        def monoIncrease(a,eps=np.finfo(float).eps):
            return np.all(np.diff(a)>=0)

        z,vs,_,_,_,_,grp,layerName = self.seisPropGrids(_layerName=True);grp = np.array(grp)
        vsMantle    = vs[grp=='mantle']
        vsSediment  = vs[grp=='sediment']
        vsPrism     = vs[grp=='prism']
        vsCrust     = vs[grp=='crust']
        zMantle     = z[grp=='mantle']

        '''
        Vs jump between group is positive, contraint (5) in 4.2 of Shen et al., 2012
        '''
        for i in np.where((grp[1:]!=grp[:-1]))[0]:
            if vs[i+1] < vs[i]:
                return False

        '''
        All Vs < 4.9km/sec, contraint (6) in 4.2 of Shen et al., 2012
        '''
        if np.any(vs > 4.9):
            return False

        '''
        Velocity in sediment and crust layers must increase with depth
        '''
        if not monoIncrease(vs[grp=='sediment']):
            return False
        if not monoIncrease(vs[grp=='crust']): 
            return False
        if not monoIncrease(vsPrism): 
            return False

        '''
        Negative velocity gradient below moho
        '''
        # if vsMantle[1]>vsMantle[0]:
        #     return False

        '''
        mantle plate get higher Vs
        '''
        if np.any( np.array(layerName) == 'SubductionPlateMantle'):
            ind = [l.prop['LayerName'] for l in self.layers].index('SubductionPlateMantle')
            if self.layers[ind].parm['Vs'][0] > self.layers[ind].parm['Vs'][1]:
                return False


        '''
        No local maximum in last layer
        '''
        if self.info.get('mantleLocalMax',True) is False:
            vsDeeperMantle = vs[np.array(layerName)==layerName[-1]]
            indLocMax = scipy.signal.argrelmax(vsDeeperMantle)[0]
            if len(indLocMax) > 0:
                return False

        '''
        tmp
        '''
        # tmp1,tmp2 = np.array(layerName),vs.copy()
        # vs_last_layer = tmp2[tmp1==tmp1[-1]]
        # tmp1,tmp2 = tmp1[tmp1!=tmp1[-1]],tmp2[tmp1!=tmp1[-1]]
        # vs_second_last_layer = tmp2[tmp1==tmp1[-1]]
        # if vs_second_last_layer[-1]<vs_last_layer[0]:
        #     return False



        '''
        Vs in crust < 4.3
        '''
        if np.any(vsCrust > 4.3):
            return False


        '''
        velocity increase at bottom
        '''
        if (vsMantle[-1]-vsMantle[-2])/(zMantle[-1]-zMantle[-2]) <= 0:
            return False

        # temporary only
        # if len(indLocMin) > 1:
        #     return False

        return True
 
class Model1D_Cascadia_Continental(Model1D_MCinv):
    def _loadLocalInfo(self, layersD, localInfo):
        super()._loadLocalInfo(layersD, localInfo)
        layersD = deepcopy(layersD)
        layersK = list(layersD.keys())
        grps = [buildSeisLayer(parm,typeID).prop['Group'] for typeID,parm in layersD.items()]

        topo = localInfo.get('topo',self.info.get('topo',0))
        waterH = max(-topo,0)
        if waterH > 0 and 'water' in grps:
            try:
                layersD[layersK[grps.index('water')]]['H'][0] = waterH
            except:
                layersD[layersK[grps.index('water')]]['H'] = waterH
        elif waterH == 0 and 'water' in grps:
            del layersD[layersK[grps.index('water')]]

        if 'sedthk' in localInfo.keys():
            try:
                layersD[layersK[grps.index('sediment')]]['H'][0] = localInfo['sedthk']
            except:
                layersD[layersK[grps.index('sediment')]]['H'] = localInfo['sedthk']
        
        if 'crsthk' in localInfo.keys():
            try:
                layersD[layersK[grps.index('crust')]]['H'][0] = localInfo['crsthk']
            except:
                layersD[layersK[grps.index('crust')]]['H'] = localInfo['crsthk']
        
        return layersD
    def isgood(self,verbose=False):
        import scipy.signal
        def monoIncrease(a,eps=np.finfo(float).eps):
            return np.all(np.diff(a)>=0)

        z,vs,_,_,_,_,grp,layerName = self.seisPropGrids(_layerName=True);grp = np.array(grp)
        vsMantle    = vs[grp=='mantle']
        vsSediment  = vs[grp=='sediment']
        vsCrust     = vs[grp=='crust']
        zMantle     = z[grp=='mantle']

        '''
        Vs jump between group is positive, contraint (5) in 4.2 of Shen et al., 2012
        '''
        for i in np.where((grp[1:]!=grp[:-1]))[0]:
            if vs[i+1] < vs[i]:
                return False

        '''
        All Vs < 4.9km/sec, contraint (6) in 4.2 of Shen et al., 2012
        '''
        if np.any(vs > 4.9):
            return False

        '''
        Velocity in sediment and crust layers must increase with depth
        '''
        if not monoIncrease(vs[grp=='sediment']):
            return False
        if not monoIncrease(vs[grp=='crust']): 
            return False

        '''
        Negative velocity gradient below moho
        '''
        # if vsMantle[1]>vsMantle[0]:
        #     return False

        '''
        No local maximum in last layer
        '''
        if self.info.get('mantleLocalMax',True) is False:
            vsDeeperMantle = vs[np.array(layerName)==layerName[-1]]
            indLocMax = scipy.signal.argrelmax(vsDeeperMantle)[0]
            if len(indLocMax) > 0:
                return False

        '''
        Vs in crust < 4.3
        '''
        # if np.any(vsCrust > 4.3):
        #     return False

        '''
        Vs at first fine layer in mantle is between 4.0 and 4.6
        '''
        # if vsMantle[0] < 4.0 or vsMantle[0] > 4.6:
        #     return False

        ''' 
        Vs at last fine layer in mantle > 4.3 
        '''
        # if vsMantle[-1] < 4.3:
        #     return False

        '''
        Vs > 4.0 below 80km
        '''
        # if np.any(vsMantle < 4.0):
        #     return False

        '''
        Change in mantle < 15%, made by Ayu
        '''
        # if (vsMantle.max() - vsMantle.min()) > 0.15*vsMantle.mean():
        #     return False

        '''
        Oscillation Limit, the difference between nearby local extrema < 10% of mean vs in mantle
        '''
        # osciLim = 0.1*vsMantle.mean()
        # indLocMax = scipy.signal.argrelmax(vsMantle)[0]
        # indLocMin = scipy.signal.argrelmin(vsMantle)[0]
        # if len(indLocMax) + len(indLocMin) > 1:
        #     indLoc = np.sort(np.append(indLocMax,indLocMin))
        #     osci = abs(np.diff(vsMantle[indLoc]))
        #     if len(np.where(osci > osciLim)[0]) >= 1:   # origin >= 1
        #         return False

        '''
        velocity increase at bottom
        '''
        if (vsMantle[-1]-vsMantle[-2])/(zMantle[-1]-zMantle[-2]) <= 0:
            return False

        # temporary only
        # if len(indLocMin) > 1:
        #     return False

        return True


def buildModel1D(ymlFile,localInfo={},default='Cascadia_Oceanic') -> Model1D:
    modelTypeDict = {
        'General'                           : Model1D,
        'MCInv'                             : Model1D_MCinv,
        'Cascadia_Oceanic'                  : Model1D_Cascadia_Oceanic,
        'Cascadia_Prism'                    : Model1D_Cascadia_Prism,
        'Cascadia_Continental'              : Model1D_Cascadia_Continental
    }
    def loadRawDict(ymlFile):
        if ymlFile is None:
            return None
        if type(ymlFile) is dict:
            rawDict = ymlFile
        else:
            import yaml
            with open(ymlFile, 'r') as f:
                rawDict = yaml.load(f,Loader=yaml.FullLoader)
        return rawDict
    def initModel(rawDict):
        if rawDict is None:
            return None
        modelType = rawDict['Info'].get('ModelType',default)
        try:
            mod = modelTypeDict[modelType]()
            mod.loadYML(rawDict,localInfo)
        except Exception as e:
            print(e)
            raise ValueError(f'Error: ModelType {modelType} not supported!')
        return mod

    return initModel(loadRawDict(ymlFile))
    

if __name__ == '__main__':
    mod = Model1D_Cascadia_Oceanic()
    mod.loadYML('cascadia-ocean.yml',{'topo':-2,'sedthk':0.5,'lithoAge':4.0})
    print(mod.forward())
    # mod.plotProfileGrid();mod.show()
    # mod = mod.perturb().perturb().perturb()
    # mod.plotProfileGrid(ax=plt.gca());mod.show()
    # mod = mod.reset()
    # mod.plotProfileGrid(ax=plt.gca());mod.show()
    # print(mod.toYML())

    # print(mod._brownians(numberOnly=True))
    # print(mod.forward())
    # mod.info['refLayer'] = False
    # print(mod.forward())
