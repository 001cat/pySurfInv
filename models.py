import numpy as np
from copy import deepcopy
from Triforce.pltHead import *
from pySurfInv.layers import buildSeisLayer, layerClassDict as layerClassDefault
from pySurfInv.utils import plotLayer,plotGrid,_dictIterModifier
from pySurfInv.brownian import BrownianVar

def monoIncrease(a,eps=np.finfo(float).eps):
    return np.all(np.diff(a)>=eps)

def _calForward(inProfile,wavetype='Ray',periods=[5,10,20,40,60,80],debug=False):
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
        if debug:
            print(cr0[:nper])
        return None
    return cr0[:nper]


class Model1D():
    def __init__(self,layers=[],info=None) -> None:
        self._layers = layers
        self.info   = info

    # load and save settings
    def loadYML(self,ymlFile,localInfo={},layerClassCustom={}):
        layerClassDict = layerClassDefault.copy();layerClassDict.update(layerClassCustom)
        if type(ymlFile) is not dict:
            import yaml
            with open(ymlFile, 'r') as f:
                ymlFile = yaml.load(f,Loader=yaml.FullLoader)
        ymlDict = deepcopy(ymlFile)
        self.info = ymlDict.pop('Info')
        localDict = self._loadLocalInfo(ymlDict,localInfo,layerClassDict)
        self._layers = [buildSeisLayer(parm,layerClassDict[typeID]) for typeID,parm in localDict.items()]

    def _loadLocalInfo(self,ymlDict,localInfo,layerClassDict):
        self.info.update(localInfo)
        '''
        To Be Specified in Child Class: how local information modifies layers
        '''
        return ymlDict

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

    # offer structure information 
    def seisPropGrids(self,refLayer=False,_layerName=False,hLowerLimit=0.01):
        layers = self.layers.copy(); layers += [self._refLayer.copy()] if refLayer else []
        z0 = -max(self.info.get('topo',0),0)
        z,vs,vp,rho,qs,qp,grp,layerName = [],[],[],[],[],[],[],[]
        for layer in layers:
            z1,vs1,vp1,rho1,qs1,qp1 = layer.seisPropGrids(
                layersAbove =[z,vs,vp,rho,qs,qp,grp,layerName],
                modelInfo = self.info)
            if z1[-1]-z1[0] < hLowerLimit:
                continue
            z += list(z1+z0); vs += list(vs1); vp += list(vp1); rho += list(rho1); qs += list(qs1); qp += list(qp1)
            grp         += [layer.prop['Group']]*len(z1)
            layerName   += [layer.prop['LayerName']]*len(z1)
            z0 = z[-1]
        if _layerName:
            return np.array(z),np.array(vs),np.array(vp),np.array(rho),\
                   np.array(qs),np.array(qp),grp,layerName
        else:
            return np.array(z),np.array(vs),np.array(vp),np.array(rho),\
                   np.array(qs),np.array(qp),grp

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

    def value(self,zdeps,type='vs'):
        if type != 'vs': 
            raise ValueError('Error: only support vs, others to be added...')
        z,vs,vp,rho,qs,qp,grp = self.seisPropGrids()
        return np.interp(zdeps,z,vs,left=np.nan,right=np.nan)

    def moho(self):
        z,vs,vp,rho,qs,qp,grp = self.seisPropGrids()
        return z[grp.index('mantle')]

    # offer predition 
    def forward(self,periods=[5,10,20,40,60,80]):
        refLayer = self.info.get('refLayer',False)
        pred = _calForward(np.array(self.seisPropLayers(refLayer=refLayer)[:-1]),wavetype='Ray',periods=periods)
        if pred is None:
            print(f'Warning: Forward not complete! Model listed below:')
            self.show()
        return pred

    # present structure information
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

    # miscellaneous
    def _getLayer(self,layerName): # using _layers instead of layers, for parameter modification only
        try:
            ind = [l.prop['LayerName'] for l in self._layers].index(layerName)
            return self._layers[ind]
        except:
            return None
    @property
    def _refLayer(self):
        return buildSeisLayer({'H':300,'Slope':0.35/200},layerClassDefault['ReferenceMantle'])
    @property
    def layers(self):
        return self._layers

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

class PureGird(Model1D):
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
                if k == 'z':
                    parmLayer[k] = parmLayer[k]-parmLayer[k][0]
            self._layers.append(PureGrid(parmLayer,prop={'Group':grp}))
        self.info = info
    def loadYML(self, ymlFile, localInfo={}):
        raise AttributeError('"Model1D_Puregird" object has no method "loadYML"')
    @property
    def layers(self):
        return self._layers

class PureLayer(Model1D):
    pass


class MCinv(Model1D):
    # used in MC inversion
    def perturb(self,isgood=None,verbose=False):
        if isgood is None:
            def isgood(model):
                return model.isgood()
        for i in range(1000):
            newModel = self.copy()
            newModel._layers = [l._perturb() for l in self.layers]
            if verbose:
                print(newModel._brownians())
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

    # easy access in MC post-processing
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

    # miscellaneous for MC inversion
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


''' Cascadia specified '''
class CascadiaPrism(MCinv):
    def _loadLocalInfo(self, layersD, localInfo, layerClassDict):
        super()._loadLocalInfo(ymlDict, localInfo, layerClassDict)
        localDict = deepcopy(ymlDict)
        grp2layer = {layerClassDict[lType]().prop['Group']:lType for lType in ymlDict.keys()}
        if len(ymlDict) != len(grp2layer):
            raise ValueError('Please check if repeated group exists!')

        # water layer thickness
        topo = localInfo.get('topo',self.info.get('topo',0))
        waterH = max(-topo,0)
        if waterH > 0 and 'water' in grp2layer:
            try:
                localDict[grp2layer['water']]['H'][0] = waterH
            except:
                localDict[grp2layer['water']]['H'] = waterH
        elif waterH == 0 and 'water' in grp2layer:
            del localDict[grp2layer['water']]
        
        # sediment thickness 
        if 'sedthk' in localInfo.keys():
            try:
                localDict[grp2layer['sediment']]['H'][0] = localInfo['sedthk']
            except:
                localDict[grp2layer['sediment']]['H'] = localInfo['sedthk']
        
        # prism thickness 
        if 'prismthk' in localInfo.keys():
            try:
                localDict[grp2layer['prism']]['H'][0] = localInfo['prismthk']
            except:
                localDict[grp2layer['prism']]['H'] = localInfo['prismthk']

        return localDict
    def isgood(self,verbose=False):
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

        # '''
        # Negative velocity gradient below moho
        # '''
        # if self.info.get('negSlopeMoho',False) is True:
        #     if vsMantle[1]>vsMantle[0]:
        #         return False

        # '''
        # mantle plate get higher Vs
        # '''
        # if np.any( np.array(layerName) == 'SubductionPlateMantle'):
        #     ind = [l.prop['LayerName'] for l in self.layers].index('SubductionPlateMantle')
        #     if isinstance(self.layers[ind].parm['Vs'],list):
        #         if self.layers[ind].parm['Vs'][0] > self.layers[ind].parm['Vs'][1]:
        #             return False

        # '''
        # mantle Vs 100km
        # '''
        # vsLessThan = self.info.get('mantleVs100LessThan',False)
        # if vsLessThan is not False:
        #     if vs[(z<150) * (z>50)].mean() >= vsLessThan:
        #         return False

        '''
        Vs in crust < 4.3
        '''
        # if np.any(vsCrust > 4.3):
        #     return False


        '''
        velocity increase at bottom
        '''
        if (vsMantle[-1]-vsMantle[-2])/(zMantle[-1]-zMantle[-2]) <= 0:
            return False

        return True
 
class CascadiaContinent(MCinv):
    def _loadLocalInfo(self, ymlDict, localInfo, layerClassDict):
        super()._loadLocalInfo(ymlDict, localInfo, layerClassDict)
        localDict = deepcopy(ymlDict)
        grp2layer = {layerClassDict[lType]().prop['Group']:lType for lType in ymlDict.keys()}
        if len(ymlDict) != len(grp2layer):
            raise ValueError('Please check if repeated group exists!')

        # sediment thickness
        if 'sedthk' in localInfo.keys():
            try:
                localDict[grp2layer['sediment']]['H'][0] = localInfo['sedthk']
            except:
                localDict[grp2layer['sediment']]['H'] = localInfo['sedthk']
        
        # crust thickness
        if 'crsthk' in localInfo.keys():
            try:
                localDict[grp2layer['crust']]['H'][0] = localInfo['crsthk']
            except:
                localDict[grp2layer['crust']]['H'] = localInfo['crsthk']
        
        return localDict
    def isgood(self,verbose=False):
        import scipy.signal

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
        # if self.info.get('mantleLocalMax',True) is False:
        #     vsDeeperMantle = vs[np.array(layerName)==layerName[-1]]
        #     indLocMax = scipy.signal.argrelmax(vsDeeperMantle)[0]
        #     if len(indLocMax) > 0:
        #         return False

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
        # if (vsMantle[-1]-vsMantle[-2])/(zMantle[-1]-zMantle[-2]) <= 0:
        #     return False

        '''
        criterion for OceanMantle_Gaussian layer
        1. velocity above slab can not be larger than slab
        2. without gaussian part, the oscillation should be small
        '''
        # from pySurfInv.layers import OceanMantle_Gaussian
        # if isinstance(self.layers[-1],OceanMantle_Gaussian):
        #     A,mu,sig = self.layers[-1].parm['Gauss']
        #     ztmp = zMantle-zMantle[0]
        #     indAbove = ztmp < mu - 2*sig
        #     indIn    = abs(ztmp-mu) <= 2*sig
        #     indBelow = ztmp > mu + 2*sig
        #     try:
        #         if np.any(vsMantle[indAbove] > max(vsMantle[indIn])):
        #             return False
        #     except Exception as e:
        #         # print(e)
        #         pass

        #     # if np.mean(vsMantle[indBelow]) < np.mean(vsMantle[indAbove]):
        #     #     return False

        #     from Triforce.mathPlus import gaussFun
        #     vstmp = vsMantle - gaussFun(A,mu,sig,ztmp)
        #     osciLim = 0.03*vstmp.mean()
        #     # osciLim = sig
        #     indLocMax = scipy.signal.argrelmax(vstmp)[0]
        #     indLocMin = scipy.signal.argrelmin(vstmp)[0]
        #     if len(indLocMax) + len(indLocMin) > 1:
        #         indLoc = np.sort(np.append(indLocMax,indLocMin))
        #         osci = abs(np.diff(vstmp[indLoc]))
        #         if len(np.where(osci > osciLim)[0]) >= 1:   # origin >= 1
        #             return False

        #     # from Triforce.mathPlus import gaussFun
        #     # vsMantle - gaussFun(A,mu,sig,z)
        #     # cwtWidth = 30//(z[1]-z[0])
        #     # cwt = scipy.signal.cwt(vsMantle - np.interp(z,[z[0],z[-1]],[vsMantle[0],vsMantle[-1]]),
        #     #                     scipy.signal.ricker,[cwtWidth])[0]
        #     # indLocMax = scipy.signal.argrelmax(cwt)[0]
        #     # indLocMin = scipy.signal.argrelmin(cwt)[0]
        #     # indLoc = np.sort(np.append(indLocMax,indLocMin))
        #     # osci = abs(np.diff(cwt[indLoc]))
        #     # if np.any(osci > 0.3):
        #     #     return False
        return True

class CascadiaOcean(MCinv):
    def _loadLocalInfo(self, ymlDict, localInfo, layerClassDict):
        ymlDict = super()._loadLocalInfo(ymlDict, localInfo, layerClassDict)
        localDict = deepcopy(ymlDict)
        grp2layer = {layerClassDict[lType]().prop['Group']:lType for lType in ymlDict.keys()}
        if len(ymlDict) != len(grp2layer):
            raise ValueError('Please check if repeated group exists!')

        # water layer thickness
        topo = localInfo.get('topo',self.info.get('topo',0))
        waterH = max(-topo,0)
        if waterH > 0 and 'water' in grp2layer:
            try:
                localDict[grp2layer['water']]['H'][0] = waterH
            except:
                localDict[grp2layer['water']]['H'] = waterH
        elif waterH == 0 and 'water' in grp2layer:
            del localDict[grp2layer['water']]

        # sediment thickness
        if 'sedthk' in localInfo.keys():
            try:
                localDict[grp2layer['sediment']]['H'][0] = localInfo['sedthk']
            except:
                localDict[grp2layer['sediment']]['H'] = localInfo['sedthk']
        
        # plate age
        if 'lithoAge' in localInfo.keys():
            try:
                localDict[grp2layer['mantle']]['ThermAge'][0] = localInfo['lithoAge']
                if localDict[grp2layer['mantle']]['ThermAge'][1] in ('rel_pos','rel') and localDict[grp2layer['mantle']]['ThermAge'][0] < 2 :
                    localDict[grp2layer['mantle']]['ThermAge'][1] = 'abs_pos'
                    localDict[grp2layer['mantle']]['ThermAge'][2] = 2*localDict[grp2layer['mantle']]['ThermAge'][2]/100
            except:
                localDict[grp2layer['mantle']]['ThermAge'] = localInfo['lithoAge']

        # reference mantle Vs
        if 'mantleInitParmVs' in localInfo.keys():
            for i,vs in enumerate(localInfo['mantleInitParmVs']):
                try:
                    localDict[grp2layer['mantle']]['Vs'][i][0] = vs
                except:
                    localDict[grp2layer['mantle']]['Vs'][i] = vs
            self.info.pop('mantleInitParmVs')

        return localDict
    def isgood(self,verbose=False):
        if not super().isgood():
            return False
        
        z,vs,_,_,_,_,grp = self.seisPropGrids(refLayer=False)
        indS = np.array(grp)=='sediment'
        indC = np.array(grp)=='crust'
        indM = np.array(grp)=='mantle'
        vsS,vsC,vsM = vs[indS],vs[indC],vs[indM]

        ''' Vs in sediment > 0.2; from Lili's code '''
        if np.any(vsS<0.2):
            return False

        ''' Vs jump between group is positive, contraint (5) in 4.2 of Shen et al., 2012 '''
        for i in np.where((grp[1:]!=grp[:-1]))[0]:
            if vs[i+1] < vs[i]:
                return False

        ''' Velocity in sediment and crust layers must increase with depth '''
        if not monoIncrease(vs[grp=='sediment']):
            return False
        if not monoIncrease(vs[grp=='crust']): 
            return False

        ''' velocity increase at bottom '''
        if (vs[-1]-vs[-2])/(z[-1]-z[-2]) <= 0:
            return False
        
        ''' Oscillation Limit, the difference between nearby local extrema < 10% of mean vs in mantle '''
        import scipy.signal
        osciLim = 0.1*vsM.mean()
        indLocMax = scipy.signal.argrelmax(vsM)[0]
        indLocMin = scipy.signal.argrelmin(vsM)[0]
        if len(indLocMax) + len(indLocMin) > 1:
            indLoc = np.sort(np.append(indLocMax,indLocMin))
            osci = abs(np.diff(vsM[indLoc]))
            if len(np.where(osci > osciLim)[0]) >= 1:   # origin >= 1
                return False
        
        ################
        # For hybrid parameterization only
        zM = z[indM]
        ''' No local maximum in the mantle layer '''
        if len(scipy.signal.argrelmax(vsM)[0]) > 0:
            if verbose:
                print('Debug: shallow local maximum found')
            return False
        
        ''' No extreme velocity decrease below moho '''
        slope = np.diff(vsM)/np.diff(zM)
        if slope.min() < slope[0]*1.5:  # should be between 1.5 to 2.2
            return False
            
        ''' osci limitation using cwt '''
        cwtWidth = 30//(zM[1]-zM[0])
        cwt = scipy.signal.cwt(vsM - np.interp(zM,[zM[0],zM[-1]],[vsM[0],vsM[-1]]),
                            scipy.signal.ricker,[cwtWidth])[0]
        indLocMax = scipy.signal.argrelmax(cwt)[0]
        indLocMin = scipy.signal.argrelmin(cwt)[0]
        indLoc = np.sort(np.append(indLocMax,indLocMin))
        osci = abs(np.diff(cwt[indLoc]))
        if np.any(osci > 0.3):
            return False
        ################

        ################
        # constraints used previously
        '''
        All Vs < 4.9km/sec, contraint (6) in 4.2 of Shen et al., 2012
        '''
        # if np.any(vs > 4.9):
        #     return False

        '''
        Vs in crust < 4.3
        '''
        # if np.any(vsC > 4.3):
        #     return False

        '''
        Vs at the mantle top is between 4.0 and 4.6
        '''
        # if vsM[0] < 4.0 or vsM[0] > 4.6:
        #     return False

        ''' 
        Vs at the bottom of the mantle layer in mantle > 4.3 
        '''
        # if vsM[-1] < 4.3:
        #     return False

        '''
        Vs > 4.0 below 80km
        '''
        # if np.any(vsMantle < 4.0):
        #     return False

        '''
        Limit Vs variation within mantle, made by Ayu
        '''
        # if (vsM.max() - vsM.min()) > 0.15*vsM.mean():
        #     return False
        ################


        return True



def buildModel1D(ymlFile,localInfo={},modelTypeCustom=None,layerClassCustom={}) -> Model1D:
    modelTypeDict = {
        'General'                           : Model1D,
        'MCInv'                             : MCinv,
        'CascadiaOcean'                     : CascadiaOcean,
        'CascadiaPrism'                     : CascadiaPrism,
        'CascadiaContinent'                 : CascadiaContinent
    }
    import yaml
    if isinstance(ymlFile,dict):
        ymlDict = ymlFile
    else:
        with open(ymlFile, 'r') as f:
            ymlDict = yaml.load(f,Loader=yaml.FullLoader)
    try:
        if modelTypeCustom:
            mod = modelTypeCustom()
            mod.loadYML(ymlDict,localInfo,layerClassCustom)
        else:
            mod = modelTypeDict[ymlDict['Info'].get('modelType','General')]()
            mod.loadYML(ymlDict,localInfo)
    except Exception as e:
        raise e
    return mod

if __name__ == '__main__':
    from pySurfInv.layers import OceanMantleThermBsplineHybrid as OceanMantleHybrid

    class ModelNew(CascadiaOcean):
        def _loadLocalInfo(self, ymlDict, localInfo, layerClassDict):
            return super()._loadLocalInfo(ymlDict, localInfo, layerClassDict)
        def isgood(self,verbose=False):
            if not super().isgood():
                return False
            
    class LayerNew(OceanMantleHybrid):
        def __init__(self, parm={}, prop={}) -> None:
            super().__init__(parm, prop)
            self.prop.update({'LayerName':'LayerNew','Group':'mantle'})

    ymlDict = {
        'OceanWater': {'H':2},
        'OceanSedimentCascadia': {'H':[1,'rel_pos',100,0.1]},
        'OceanCrust': {'H':7,'Vs':[3.25,3.94]},
        'LayerNew': {
            'Vs':[[0.2, 'rel', 10, 0.02],
                [0.1, 'rel', 10, 0.02],
                [-0.1, 'rel', 10, 0.02],
                [-0.2, 'rel', 5, 0.02]
                ],
            'ThermAge': [4,'rel_pos',200,0.4],
            'BottomDepth':200,
            'Conversion': 'Ritzwoller'
        },
        'Info':{
            'modelType':'ModelNew',
            'period': 10,
            'refLayer':True,
            'lithoAgeQ':True
        }
    }

    mod = buildModel1D(ymlDict,{'topo':-2,'sedthk':0.5,'lithoAge':4.0},
            modelTypeCustom=ModelNew,
            layerClassCustom={'LayerNew':LayerNew})
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
