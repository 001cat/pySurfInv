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

def isNumeric(v):
    try:
        float(v);return True
    except:
        return False

def loadSetting(ymlFile='setting-Hongda.yml',localInfo={}):
    import yaml
    from pySurfInv.layers import typeDict,oldTypeDict

    def dictConvert(inDict):
        from pySurfInv.utils import _dictIterModifier
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
        return _dictIterModifier(inDict,isBrownian,toBrownian)

    def dictReshape(inDict):
        layers = {}
        for k,v in inDict.items():
            if k == 'Info':
                info = v
            elif k in typeDict.keys():      # yml file for point2
                layers[k] = v
            else:                           # for previous yml file
                if 'Qmodel' in inDict['Info'].keys() and inDict['Info']['Qmodel'] == 'Ruan2018':
                    oldTypeDict['mantle_Bspline_'] = 'Mantle_Bspline_Ocean_CascadiaQ'
                mtype = '' if 'mtype' not in v.keys() else v['mtype']
                stype = '' if 'stype' not in v.keys() else v['stype']
                typeID = oldTypeDict['_'.join([k,mtype,stype])]
                if typeID == 'Water_Ocean':
                    layers[typeID] = {'H':v['h'],'Vs':0}
                else:
                    layers[typeID] = {'H':v['h'],'Vs':v['vs']}
        return layers,info
    def mergeLocalInfo(rawDict,info,localInfo):
        from copy import deepcopy
        mergedDict = deepcopy(rawDict)

        keys  = [k for k in rawDict.keys()]
        types = [k.lower().split('_')[0] for k in keys]

        if 'topo' in localInfo.keys():
            info['topo'] = localInfo['topo']
        # if 'topo' in info.keys():
            waterH = max(-info['topo'],0)
            if waterH > 0 and 'water' in types:
                try:
                    mergedDict[keys[types.index('water')]]['H'][0] = waterH
                except:
                    mergedDict[keys[types.index('water')]]['H'] = waterH
            elif waterH == 0 and 'water' in types:
                del mergedDict[keys[types.index('water')]]

        if 'sedthk' in localInfo.keys():
            info['sedthk'] = localInfo['sedthk']
        # if 'sedthk' in info.keys():
            try:
                mergedDict[keys[types.index('sediment')]]['H'][0] = info['sedthk']
            except:
                mergedDict[keys[types.index('sediment')]]['H'] = info['sedthk']
        
        if 'crsthk' in localInfo.keys():
            info['crsthk'] = localInfo['crsthk']
        # if 'crsthk' in info.keys():
            try:
                mergedDict[keys[types.index('crust')]]['H'][0] = info['crsthk']
            except:
                mergedDict[keys[types.index('crust')]]['H'] = info['crsthk']
        
        if 'lithoAge' in localInfo.keys():
            info['lithoAge'] = localInfo['lithoAge']
            try:
                mergedDict[keys[types.index('mantle')]]['Age'][0] = info['lithoAge']
            except:
                mergedDict[keys[types.index('mantle')]]['Age'] = info['lithoAge']


        return mergedDict,info
    
    if type(ymlFile) is dict:
        rawDict = ymlFile
    else:
        with open(ymlFile, 'r') as f:
            rawDict = yaml.load(f,Loader=yaml.FullLoader)

    rawDict,info = dictReshape(rawDict)
    mergedDict,info = mergeLocalInfo(rawDict,info,localInfo)
    layerDict = dictConvert(mergedDict)

    # if 'total' used in last layer's H
    try:
        if list(rawDict.values())[-1]['H'][1].lower() == 'total':
            info['TotalH'] = list(rawDict.values())[-1]['H'][0]
    except:
        pass

    return layerDict,info

def buildModel1D(ymlFile,localInfo={}):
    if type(ymlFile) is dict:
        rawDict = ymlFile
    else:
        import yaml
        with open(ymlFile, 'r') as f:
            rawDict = yaml.load(f,Loader=yaml.FullLoader)
    modelType = rawDict['Info'].get('ModelType','General_Model')
    if modelType == 'General_Model':
        mod = Model1D()
        mod.loadYML(rawDict,localInfo)
        return mod
    if modelType == 'Cascadia_Oceanic':
        mod = Model1D_Cascadia_Oceanic()
        mod.loadYML(rawDict,localInfo)
        return mod
    else:
        raise ValueError(f'Error: ModelType {modelType} not supported!')



class Model1D():
    def __init__(self,layers=[],info=None) -> None:
        self._layers = layers
        self.info   = info
    def loadYML(self,ymlFile,localInfo={}):
        layerDict,info = loadSetting(ymlFile,localInfo)
        self.info = info
        self._layers = [buildSeisLayer(parm,typeID) for typeID,parm in layerDict.items()]
    def toYML(self):
        def checker(v):
            return type(v) == BrownianVar
        def modifier(v):
            return [v.v,v.vmin,v.vmax,v.step]
        ymlDict = {}
        for layer in self.layers:
            ymlDict[layer._SeisLayerID] = _dictIterModifier(layer.parm,checker,modifier)
        ymlDict['Info'] = self.info
        return deepcopy(ymlDict)
    def _loadMC(self,mc): 
        mc_ind = 0
        for layer in self.layers:
            for k,v in layer.parm.items():
                if type(v) == BrownianVar:
                    layer.parm[k] = v._setValue(mc[mc_ind]);mc_ind += 1
                elif type(v) == list:
                    for i in range(len(v)):
                        if type(v[i]) == BrownianVar:
                            v[i] = v[i]._setValue(mc[mc_ind]);mc_ind += 1
                    layer.parm[k] = v
    def _brownians(self,numberOnly=True):
        
        brownians = []
        for layer in self.layers:
            for k,v in layer.parm.items():
                if type(v) is list:
                    for e in v:
                        if type(e) is BrownianVar:
                            brownians.append([float(e),layer.group,k])
                else:
                    if type(v) is BrownianVar:
                        brownians.append([float(v),layer.group,k])
        if numberOnly:
            brownians = [v[0] for v in brownians]
        return brownians
    def _dump(self,index,target,preInfo=[]):
        preInfo.extend(self._brownians())
        target[index] = preInfo

    def seisPropGrids(self,refLayer=False):
        z,vs,vp,rho,qs,qp,grp = [],[],[],[],[],[],[]
        for layer in self.layers:
            z1,vs1,vp1,rho1,qs1,qp1 = layer.seisPropGrids
            z += list(z1)
            vs += list(vs1); vp += list(vp1); rho += list(rho1); qs += list(qs1); qp += list(qp1)
            grp += [layer.group]*len(z1)
        if refLayer:
            refLayer = self._refLayer.copy()
            refLayer.parm['Vs'][0] += vs[-1]
            refLayer.parm['Vs'][1] += vs[-1]
            z1,vs1,vp1,rho1,qs1,qp1 = refLayer.seisPropGrids
            vs1 += [vs[-1]-vs1[0]]; vp1 += [vp[-1]-vp1[0]]; rho1 += [rho[-1]-rho1[0]]
            qs1 += [qs[-1]-qs1[0]]; qp1 += [qp[-1]-qp1[0]]
            z += list(z1)
            vs += list(vs1); vp += list(vp1); rho += list(rho1); qs += list(qs1); qp += list(qp1)
            grp += [refLayer.group]*len(z1)
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

    def perturb(self,isgood=None):
        if isgood is None:
            def isgood(model):
                return model.isgood()
        for i in range(1000):
            newModel = self.copy()
            newModel._layers = [l._perturb() for l in self.layers]
            if isgood(newModel):
                return newModel
        return self.reset()
    def reset(self,isgood=None):
        if isgood is None:
            def isgood(model):
                return model.isgood()
        for i in range(1000):
            newModel = newModel = self.copy()
            newModel._layers = [l._reset() for l in self.layers]
            if isgood(newModel):
                return newModel
        self.show()
        raise RuntimeError(f'Error: Cound not find a good model through reset.')
    def isgood(self):
        return True

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
            print(layer.group)
            print(layer.parm)
    
    def plotProfile(self,type='vs',**kwargs):
        h,vs,vp,rho,qs,qp,_ = self.seisPropLayers()
        if type == 'vs':
            fig = plotLayer(h,vs,**kwargs);plt.title('Vs')
        else:
            print('To be added...')
        return fig
    def plotProfileGrid(self,type='vs',ax=None,**kwargs):
        z,vs,vp,rho,qs,qp,_ = self.seisPropGrids(refLayer=False)
        if type == 'vs':
            fig = plotGrid(z,vs,ax=ax,**kwargs);plt.title('Vs')
        else:
            print('To be added...')
        return fig

    @property
    def _refLayer(self):
        return buildSeisLayer({'H':300,'Vs':[0,0.35/200*300]},'Mantle_Reference')
    @property
    def layers(self):
        if 'TotalH' in self.info.keys():
            aboveSeaLevel = max(self.info.get('topo',0),0)
            self._layers[-1].parm['H'] = self.info['TotalH'] + aboveSeaLevel - \
                                         np.sum([l.parm['H'] for l in self._layers[:-1]])
        return self._layers

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)


class Model1D_Puregird(Model1D):
    def __init__(self, inProfiles, info=None) -> None:
        from pySurfInv.layers import Seis_Puregrid
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
            self._layers.append(Seis_Puregrid(parmLayer,grp))
        self.info = info
    @property
    def layers(self):
        return self._layers
    def seisPropGrids(self,refLayer=False):
        z,vs,vp,rho,qs,qp,grp = [],[],[],[],[],[],[]
        for layer in self.layers:
            z1,vs1,vp1,rho1,qs1,qp1 = layer.seisPropGrids
            z += list(z1)
            vs += list(vs1); vp += list(vp1); rho += list(rho1); qs += list(qs1); qp += list(qp1)
            grp += [layer.group]*len(z1)
        if refLayer:
            refLayer = self._refLayer.copy()
            refLayer.parm['Vs'][0] += vs[-1]
            refLayer.parm['Vs'][1] += vs[-1]
            z1,vs1,vp1,rho1,qs1,qp1 = refLayer.seisPropGrids
            vs1 += [vs[-1]-vs1[0]]; vp1 += [vp[-1]-vp1[0]]; rho1 += [rho[-1]-rho1[0]]
            qs1 += [qs[-1]-qs1[0]]; qp1 += [qp[-1]-qp1[0]]
            z += list(z1)
            vs += list(vs1); vp += list(vp1); rho += list(rho1); qs += list(qs1); qp += list(qp1)
            grp += [refLayer.group]*len(z1)
        return np.array(z),np.array(vs),np.array(vp),np.array(rho),np.array(qs),np.array(qp),grp


class Model1D_Cascadia_Oceanic(Model1D):
    def seisPropGrids(self,refLayer=False):
        z0 = -max(self.info['topo'],0)
        hCrust = np.sum([l.parm['H'] if l.group == 'crust' else 0 for l in self.layers])
        z,vs,vp,rho,qs,qp,grp = [],[],[],[],[],[],[]
        for layer in self.layers:
            layer._tmpInfo = {'z0':z0,'age':self.info['lithoAge'],'hCrust':hCrust}
            z1,vs1,vp1,rho1,qs1,qp1 = layer.seisPropGrids
            z += list(z1+z0)
            vs += list(vs1); vp += list(vp1); rho += list(rho1); qs += list(qs1); qp += list(qp1)
            grp += [layer.group]*len(z1)
            z0 = z[-1]
        if refLayer:
            refLayer = self._refLayer.copy()
            refLayer.parm['Vs'][0] += vs[-1]
            refLayer.parm['Vs'][1] += vs[-1]
            z1,vs1,vp1,rho1,qs1,qp1 = refLayer.seisPropGrids
            vs1 += [vs[-1]-vs1[0]]; vp1 += [vp[-1]-vp1[0]]; rho1 += [rho[-1]-rho1[0]]
            qs1 += [qs[-1]-qs1[0]]; qp1 += [qp[-1]-qp1[0]]
            z += list(z1+z0)
            vs += list(vs1); vp += list(vp1); rho += list(rho1); qs += list(qs1); qp += list(qp1)
            grp += [refLayer.group]*len(z1)
        return np.array(z),np.array(vs),np.array(vp),np.array(rho),np.array(qs),np.array(qp),grp
    def isgood(self):
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

        '''
        No local maximum above 75km
        '''
        # if np.any(h[grp=='mantle'][indLocMax]<75):
        #     print('Debug: shallow local maximum found')
        #     return False

        # temporary only
        # if len(indLocMin) > 1:
        #     return False

        return True






if __name__ == '__main__':
    mod = Model1D()
    mod.loadYML('setting-Hongda.yml',{'topo':-2,'sedthk':0.5,'lithoAge':4.0})
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

    print(mod._brownians())
    mod._loadMC([0.6,1.1,4.2,4.1,4.4,4.5])
    print(mod._brownians())

    ymlDict = mod.toYML()
    modN = Model1D()
    modN.loadYML(ymlDict)

    modN.show()
    mod.show()

    modN.info
    mod.info
