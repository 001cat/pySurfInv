import random,time,os,tqdm
import numpy as np
import multiprocessing as mp
from pySurfInv.models import buildModel1D
from Triforce.pltHead import *
from Triforce.obspyPlus import randString



class Point(object):
    def __init__(self,setting=None,localInfo={},periods=[],vels=[],uncers=[]):
        self.initMod = buildModel1D(setting,localInfo)
        self.obs = {'T':periods,'c':vels,'uncer':uncers} # Rayleigh wave, phase velocity only
        self.pid  = 'test'
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
        return misfit,chiSqr,L
    def MCinv(self,outdir='MCtest',pid=None,runN=50000,step4uwalk=1000,init=True,
              seed=None,verbose=False,priori=False,isgood=None):
        def accept(chiSqr0,chiSqr1):
            if chiSqr1 < chiSqr0: # avoid overflow
                return True
            return random.random() > 1-np.exp(-(chiSqr1-chiSqr0)/2) # (L0-L1)/L0
        if isgood is None:
            def isgood(model):
                return model.isgood()
        debug = False
        random.seed(seed)
        pid = self.pid if pid is None else pid
        timeStamp = time.time()
        mcTrack = [0]*runN
        for i in range(runN):
            if i % step4uwalk == 0:
                if init:
                    mod0 = self.initMod.copy();init=False
                    if not isgood(mod0):
                        mod0 = mod0.perturb(isgood)
                else:
                    mod0 = self.initMod.reset()
                    if verbose == True:
                        print(f'{i+1}/{runN} Time cost:{time.time()-timeStamp:.2f} ')
                misfit0,chiSqr0,L0 = self.misfit(mod0)
                mod0._dump(i,mcTrack,[misfit0,L0,1])
            else:
                mod1 = mod0.perturb(isgood)
                if debug:
                    plt.figure()
                    T = self.obs['T']
                    plt.plot(T,self.obs['c'],'--')
                    plt.plot(T,mod0.forward(periods=T))
                    plt.plot(T,mod1.forward(periods=T))
                if priori:
                    mod1._dump(i,mcTrack,[0,1,1])
                    mod0 = mod1
                    continue
                misfit1,chiSqr1,L1 = self.misfit(mod1)
                if accept(chiSqr0,chiSqr1):
                    mod1._dump(i,mcTrack,[misfit1,L1,1])
                    mod0,misfit0,chiSqr0,L0 = mod1,misfit1,chiSqr1,L1
                else:
                    mod1._dump(i,mcTrack,[misfit1,L1,0])
                if debug and L0>0.01:
                    debug = False if input() == 'Y' else True
                    plt.close()
        mcTrack = np.array(mcTrack)
        os.makedirs(outdir,exist_ok=True)
        np.savez_compressed(f'{outdir}/{pid}.npz',mcTrack=mcTrack,
                            setting=dict(self.initMod.toYML()),obs=self.obs,pid=pid)
        if verbose == 'mp':
            print(f'Step {pid.split("_")[1]} Time cost:{time.time()-timeStamp:.2f} ')
    def MCinvMP(self,outdir='MCtest',pid=None,runN=50000,step4uwalk=1000,nprocess=12,seed=None,priori=False,isgood=None):
        if priori and outdir.split('_')[-1] != 'priori':
            outdir = '_'.join((outdir,'priori'))
        tmpDir = 'MCtmp'+randString(10)
        random.seed(seed); seed = random.random()
        pid = self.pid if pid is None else pid

        print(f'Running MC inversion: {pid}')

        argInLst = [ [tmpDir,f'tmp_{i:03d}_{pid}',step4uwalk,step4uwalk,i==0,seed+i,'mp',priori,isgood]
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
        np.savez_compressed(f'{outdir}/{pid}.npz',mcTrack=mcTrack,
                            setting=dict(self.initMod.toYML()),obs=self.obs,pid=pid)

        print(f'Time cost:{time.time()-timeStamp:.2f} ')
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)
class PointCascadia(Point):
    def misfit(self,model=None):
        if model is None:
            model = self.initMod
        T = np.array(self.obs['T'])
        cP = model.forward(periods=T)
        if cP is None:
            return 88888,0
        cO = self.obs['c']
        uncer = self.obs['uncer']

        N = len(T)
        # chiSqr = (((cO - cP)/uncer)**2).sum()

        bias = (cO - cP)/uncer
        bias1 = bias[T<=40]
        bias2 = bias[T>40]
        chiSqr = ((bias1**2).mean() + (bias2**2).mean())/2*N

        misfit = np.sqrt(chiSqr/N)
        chiSqr =  chiSqr if chiSqr < 50 else np.sqrt(chiSqr*50.) 
        L = np.exp(-0.5 * chiSqr)
        return misfit,chiSqr,L

class PostPoint(Point):
    def __init__(self,npzMC=None,npzPriori=None):
        if npzMC is not None:
            tmp = np.load(npzMC,allow_pickle=True)
            self.MC,setting,self.obs = tmp['mcTrack'],tmp['setting'][()],tmp['obs'][()]
            self.initMod = buildModel1D(setting)
            # try:
            #     self.initMod = buildModel1D(setting)
            # except:
            #     from pySurfInv.models import Model1D
            #     self.initMod = Model1D(); self.initMod.loadYML(setting)
                
            self.N       = self.MC.shape[0]
            self.misfits = self.MC[:,0]
            self.Ls      = self.MC[:,1]
            self.accepts = self.MC[:,2]
            self.MCparas = self.MC[:,3:]

            indMin = np.nanargmin(self.misfits)
            self.minMod         = self.initMod.copy()
            self.minMod._loadMC(self.MCparas[indMin])
            self.minMod.L       = self.Ls[indMin]
            self.minMod.misfit  = self.misfits[indMin]

            self.thres  = max(self.minMod.misfit*2, self.minMod.misfit+0.5)
            self.accFinal = (self.misfits < self.thres)

            self.avgMod         = self.initMod.copy()
            self.avgMod._loadMC(np.mean(self.MCparas[self.accFinal,:],axis=0))
            
            self.avgMod.misfit,_,self.avgMod.L = self.misfit(model=self.avgMod)
    
        if npzPriori is not None:
            tmp = np.load(npzPriori,allow_pickle=True)['mcTrack']
            self.MCparas_pri = tmp[:,3:]
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
        self.initMod = buildModel1D(setting_Hongda_pyMCinv,
                            {'topo':topo,'sedthk':sedthk,'lithoAge':lithoAge})

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
        self.minMod._loadMC(self.MCparas[indMin])
        self.minMod.L       = self.Ls[indMin]
        self.minMod.misfit  = self.misfits[indMin]

        self.thres  = max(self.minMod.misfit*2, self.minMod.misfit+0.5)
        self.accFinal = (self.misfits < self.thres)

        self.avgMod         = self.initMod.copy()
        self.avgMod._loadMC(np.mean(self.MCparas[self.accFinal,:],axis=0))
        self.avgMod.misfit,self.avgMod.L = self.misfit(model=self.avgMod)

        if priDir is not None:
            inarr = np.load(f'{priDir}/mc_inv.{id}.npz')
            invdata    = inarr['arr_0']
            paraval       = invdata[:,2:11]
            self.Priparas = paraval[:,colOrder]
    def plotDisp(self,ax=None,ensemble=True):
        T,vel,uncer = self.obs['T'],self.obs['c'],\
                      self.obs['uncer']
        if ax is None:
            plt.figure()
        else:
            plt.axes(ax)
        mod = self.avgMod.copy()
        indFinAcc = np.where(self.accFinal)[0]
        if ensemble:
            for _ in range(min(len(indFinAcc),500)):
                i = random.choice(indFinAcc)
                mod._loadMC(self.MCparas[i,:])
                plt.plot(T,mod.forward(T),color='grey',lw=0.1)
        plt.errorbar(T,vel,uncer,ls='None',color='k',capsize=3,capthick=2,elinewidth=2,label='Observation')
        plt.plot(T,self.initMod.forward(T),label='Initial')
        plt.plot(T,self.avgMod.forward(T),label='Avg accepted')
        plt.plot(T,self.minMod.forward(T),label='Min misfit')
        plt.legend()
        plt.title('Dispersion')
        return plt.gcf(),plt.gca()
    def plotDistrib(self,inds='all',zdeps=None):
        def loadMC(mod,mc):
            mod._loadMC(mc)
            return mod.copy()
        if zdeps is not None:
            mod = self.initMod.copy()
            accMods = [loadMC(mod,mc) for mc in self.MCparas[self.accFinal]]
            accVs   = np.array([mod.value(zdeps) for mod in tqdm.tqdm(accMods)])
            priMods = [loadMC(mod,mc) for mc in self.MCparas_pri[:]]
            priVs   = np.array([mod.value(zdeps) for mod in tqdm.tqdm(priMods)])
            for i,z in enumerate(zdeps):
                plt.figure()
                _,bin_edges = np.histogram(priVs[:,i],bins=30)
                y = accVs[:,i]
                plt.hist(y,bins=bin_edges,weights=np.ones_like(y)/float(len(y)))
                y = priVs[:,i]
                plt.hist(y,bins=bin_edges,weights=np.ones_like(y)/float(len(y)),
                            fill=False,ec='k',rwidth=1.0)
                plt.title(f'Hist of Vs at {z} km')
            return
        else:
            if inds == 'all':
                inds = range(len(self.initMod._brownians()))
            for ind in inds:
                plt.figure()
                y = self.MCparas_pri[:,ind]
                _,bin_edges = np.histogram(y,bins=30)
                y = self.MCparas[self.accFinal,ind]
                plt.hist(y,bins=bin_edges,weights=np.ones_like(y)/float(len(y)))
                y = self.MCparas_pri[:,ind]
                plt.hist(y,bins=bin_edges,weights=np.ones_like(y)/float(len(y)),
                            fill=False,ec='k',rwidth=1.0)
                plt.title(f'N = {self.accFinal.sum()}/{len(self.accFinal)}')
    def plotVsProfile(self,allAccepted=False):
        ax = self.initMod.plotProfile(label='Initial')
        mod = self.avgMod.copy()
        indFinAcc = np.where(self.accFinal)[0]
        for i in range(min(len(indFinAcc),(self.N if allAccepted else 2000))):
            ind = indFinAcc[i] if allAccepted else random.choice(indFinAcc)
            mod._loadMC(self.MCparas[ind,:])
            mod.plotProfile(ax=ax,color='grey',lw=0.1)
        self.avgMod.plotProfile(ax=ax,label='Avg')
        self.minMod.plotProfile(ax=ax,label='Min')
        plt.xlim(3.8,4.8)
        plt.legend()
        return ax
    def plotVsProfileGrid(self,allAccepted=False,ax=None):
        ax = self.initMod.plotProfileGrid(label='Initial',ax=ax)
        # if ax is None:
        #     fig.set_figheight(8.4);fig.set_figwidth(5)
        mod = self.avgMod.copy()
        indFinAcc = np.where(self.accFinal)[0]
        for i in range(min(len(indFinAcc),(self.N if allAccepted else 500))):
            ind = indFinAcc[i] if allAccepted else random.choice(indFinAcc)
            mod._loadMC(self.MCparas[ind,:])
            mod.plotProfileGrid(color='grey',ax=ax,lw=0.1)
        self.avgMod.plotProfileGrid(label='Avg',ax=ax)
        self.minMod.plotProfileGrid(label='Min',ax=ax)
        plt.xlim(3.0,4.8)
        plt.legend()
        return ax
    
    # in testing
    def _check(self,step4uwalk=1000):
        from scipy.ndimage import uniform_filter1d
        iStart,iEnd = 0,step4uwalk
        localThres,decayRates,localAccRates = [],[],[]
        localAccContRates = []
        # rates,localMins,localThres = [],[],[]
        while iEnd <= len(self.misfits):
            Ntail = step4uwalk//2
            misfits = self.misfits[iStart:iEnd]
            thres = max(misfits.min()*2,misfits.min()+0.5)
            localAccI = misfits[-Ntail:]<thres
            tmp = uniform_filter1d(misfits[-Ntail:][localAccI],31)
            decayRate = max(0,-np.polyfit(np.arange(len(tmp)),tmp,1)[0]*Ntail)
            ''' assume the decay continues if we add Ntail more model sampling '''
            minmisfit_New = tmp.mean()-decayRate*1.5
            thres_New = max(minmisfit_New*2,minmisfit_New+0.5)
            # the ratio of models still accepted after more sampling applied
            localAccContRate = (misfits[-Ntail:]<thres_New).sum()/localAccI.sum()  
            # print(f'Step:{iStart//step4uwalk}: {localAccI.sum()} {decayRate} {misfits.min()}')
            decayRates.append(decayRate);localAccRates.append(localAccI.sum()/Ntail)
            localThres.append(thres);localAccContRates.append(localAccContRate)
            iStart += step4uwalk; iEnd += step4uwalk
        plt.figure()
        plt.scatter(range(len(localThres)),localThres,s=100,c=localAccRates,
                    norm=mpl.colors.BoundaryNorm([0,0.01,0.05,0.1,0.2,0.3,1],256))
        plt.colorbar()
        plt.figure()
        sc = plt.scatter(range(len(localThres)),localThres,s=100,c=localAccContRates,
                         norm=mpl.colors.BoundaryNorm([0,0.2,0.4,0.6,0.8,1.0],256))
                         #s=np.clip(localAccRates,0,0.5)**2*400,
        # plt.legend(*sc.legend_elements("sizes", num=6))
        plt.colorbar()
    def _check_deprecated(self,step4uwalk=1000,stepLens=[]):
        from scipy.ndimage.filters import uniform_filter1d
        iStart,iEnd = 0,step4uwalk
        rates,localMins,localThres = [],[],[]
        while iEnd <= len(self.misfits):
            misfits = self.misfits[iStart:iEnd]
            thres = max(misfits.min()*2,misfits.min()+0.5)
            localAcc = misfits[step4uwalk//2:]<thres
            tmp = uniform_filter1d(misfits[step4uwalk//2:][localAcc],31)
            rate = max(0,-np.polyfit(np.arange(len(tmp)),tmp,1)[0]*(step4uwalk//2))
            print(f'Step:{iStart//step4uwalk}: {localAcc.sum()} {rate} {misfits.min()}')
            rates.append(rate);localMins.append(misfits.min());localThres.append(thres)
            iStart += step4uwalk; iEnd += step4uwalk

        rates = np.array(rates); localMins = np.array(localMins); localThres = np.array(localThres)
        print((rates > localThres-localMins).sum())
    def _mod_generator(self,N=None,isRandom=True,accFinal=True):
        inds_pool = np.where(self.accFinal)[0] if accFinal else range(self.N)
        N = self.N if N is None else N
        if isRandom:
            inds = [random.choice(inds_pool) for _ in range(N)]
        else:
            inds = inds_pool[:N]

        mod = self.initMod.copy()
        for ind in inds:
            mod._loadMC(self.MCparas[ind,:])
            yield mod

    # not used now
    def plotVsProfileStd(self):
        indFinAcc = np.where(self.accFinal)[0]
        zdeps = np.linspace(0,199,300)
        allVs = np.zeros([len(zdeps),len(indFinAcc)])

        mod = self.avgMod.copy()
        for i,ind in enumerate(indFinAcc):
            mod._loadMC(self.MCparas[ind,:])
            profile =  (mod.genProfileGrid())
            allVs[:,i] = profile.value(zdeps)
        std = allVs.std(axis=1)

        fig = self.initMod.plotProfileGrid(label='Initial',alpha=0.1)
        fig.set_figheight(8.4);fig.set_figwidth(5)
        avgProfile = self.avgMod.value(zdeps)
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

        pass

# Testing
def synthetic_test():
    random.seed(36)
    periods = np.array([10,12,14,16,18,20,24,28,32,36,40,50,60,70,80])
    mod1 = buildModel1D('cascadia-ocean.yml',{'topo':-4,'sedthk':0.6,'lithoAge':3})
    mod2 = mod1.copy()
    for _ in range(20):
        mod2 = mod2.perturb()
    p = Point('cascadia-ocean.yml',{'topo':-4,'lithoAge':3},periods=periods,
              vels=mod2.forward(periods),uncers=[0.01]*len(periods))
    p.MCinvMP(runN=50000,step4uwalk=1000,nprocess=26)
    p.MCinvMP('MCtest_priori',runN=50000,step4uwalk=1000,nprocess=26,priori=True)

    postp = PostPoint('MCtest/test.npz','MCtest_priori/test.npz')
    postp.plotDisp()
    postp.plotDistrib()
    fig = postp.plotVsProfileGrid()
    mod2.plotProfileGrid(fig=fig,label='True',lineStyle='--')
    plt.legend()

def realdata_test():
    from netCDF4 import Dataset
    from Triforce.utils import GeoMap
    with Dataset('example-Cascadia/infos/ETOPO_Cascadia_smoothed.grd') as dset:
        topo = GeoMap(dset['lon'][()],dset['lat'][()],dset['z'][()]/1000)
    with Dataset('example-Cascadia/infos/sedthick_world_v2.grd') as dset:
        sedthkOce = GeoMap(dset['x'][()],dset['y'][()],dset['z'][()]/1000)
    with Dataset('example-Cascadia/infos/age_JdF_model_0.01.grd') as dset:
        lithoAge = GeoMap(dset['x'][()],dset['y'][()],dset['z'][()])
    topoDict = {
        '232.0_46.0':topo.value(232.0,46.0),
        '233.0_46.0':topo.value(233.0,46.0),
        '234.0_46.0':topo.value(234.0,46.0)
    }
    sedthkDict = {
        '232.0_46.0':0.8, #0.396
        '233.0_46.0':sedthkOce.value(233.0,46.0),
        '234.0_46.0':sedthkOce.value(234.0,46.0)
    }
    lithoAgeDict = {
        '232.0_46.0':lithoAge.value(232.0,46.0),
        '233.0_46.0':5, #6.73
        '234.0_46.0':15 #8.10
    }


    for pid in ['232.0_46.0','233.0_46.0','234.0_46.0']:
        postp = PostPoint(f'/work2/ayu/Cascadia/Works/invA0_2022_Jan10/OceanInv/{pid}.npz')
        pers,vels,uncers = postp.obs['T'],postp.obs['c'],postp.obs['uncer']
        if pid == '233.0_46.0':
            pers,vels,uncers = pers[:-1],vels[:-1],uncers[:-1]
        p = Point('cascadia-ocean.yml',{
            'topo':topoDict[pid],
            'lithoAge':lithoAgeDict[pid],
            'sedthk':sedthkDict[pid]},
                periods=pers,vels=vels,uncers=uncers)
        if pid == '233.0_46.0':
            from pySurfInv.brownian import BrownianVar
            p.initMod._layers[-1].parm['Vs'][1] = BrownianVar(-0.2,-0.6,0.2,0.02)
        p.pid = pid
        p.MCinvMP(runN=17000,step4uwalk=1000,nprocess=17)
        p.MCinvMP('MCtest_priori',runN=17000,step4uwalk=1000,nprocess=17,priori=True)

    fig = plt.figure(figsize=[5,8.4])
    for pid in ['232.0_46.0','233.0_46.0','234.0_46.0']:
        postp = PostPoint(f'MCtest/{pid}.npz',f'MCtest_priori/{pid}.npz')
        postp.avgMod.plotProfileGrid(fig=fig,label=pid)
    plt.legend()
    plt.xlim(4.0,4.8)

    pid = '232.0_46.0'
    postp = PostPoint(f'MCtest/{pid}.npz',f'MCtest_priori/{pid}.npz')
    postp.plotVsProfileGrid()
    postp.plotDistrib([0,-1])
    postp.plotDisp()

if __name__ == '__main__':
    pass