import random,time,os,tqdm
import numpy as np
import multiprocessing as mp
from pySurfInv.models import buildModel1D
from Triforce.pltHead import *
from Triforce.obspyPlus import randString

class Point(object):
    def __init__(self,setting=None,localInfo={},modelTypeCustom=None,layerClassCustom={},
                 periods=[],vels=[],uncers=[]):
        self.initMod = buildModel1D(setting,localInfo,modelTypeCustom=modelTypeCustom,
                                    layerClassCustom=layerClassCustom)
        self.obs = {'T':periods,'c':vels,'uncer':uncers} # Rayleigh wave, phase velocity only
        self.pid  = 'test'
    def misfit(self,model=None):
        if model is None:
            model = self.initMod
        T = self.obs['T']
        cP = model.forward(periods=T)
        if cP is None:
            return 88888,88888,0
        cO = self.obs['c']
        if not np.ma.isMaskedArray(cO):
            cO = np.ma.masked_array(cO)
        uncer = self.obs['uncer']
        N = cO.count()
        chiSqr = (((cO - cP)/uncer)**2).sum()
        misfit = np.sqrt(chiSqr/N)
        chiSqr =  chiSqr if chiSqr < 50 else np.sqrt(chiSqr*50.) 
        L = np.exp(-0.5 * chiSqr)
        return misfit,chiSqr,L
    def MCinv(self,outdir='MCtest',pid=None,runN=50000,chainL=1000,init=True,
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
            if i % chainL == 0:
                if init:
                    mod0 = self.initMod.copy();init=False
                    if not isgood(mod0):
                        mod0 = mod0.perturb(isgood,verbose=verbose=='perturb')
                else:
                    mod0 = self.initMod.reset()
                    if verbose == True:
                        print(f'{i+1}/{runN} Time cost:{time.time()-timeStamp:.2f} ')
                misfit0,chiSqr0,L0 = self.misfit(mod0)
                mod0._dump(i,mcTrack,[misfit0,L0,1])
            else:
                mod1 = mod0.perturb(isgood,verbose=verbose=='perturb')
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
                            setting=dict(self.initMod.toYML()),obs=self.obs,invMeta={
                                'pid':pid, 'chainL':chainL
                                })
        if verbose == 'mp':
            print(f'Step {pid.split("_")[1]} Time cost:{time.time()-timeStamp:.2f} ')
        else:
            return mod1
    def MCinvMP(self,outdir='MCtest',pid=None,runN=50000,chainL=1000,nprocess=12,seed=42,priori=False,isgood=None,
                verbose=True):
        if priori and outdir.split('_')[-1] != 'priori':
            outdir = '_'.join((outdir,'priori'))
        random.seed(None); tmpDir = 'MCtmp'+randString(10)
        random.seed(seed); seed = random.random()
        pid = self.pid if pid is None else pid

        if verbose:
            print(f'Running MC inversion: {pid}')

        argInLst = [ [tmpDir,f'tmp_{i:03d}_{pid}',chainL,chainL,i==0,seed+i,
                      'mp' if verbose else False,priori,isgood] for i in range(runN//chainL)]
        timeStamp = time.time()
        pool = mp.Pool(processes=nprocess)
        pool.starmap(self.MCinv, argInLst)
        pool.close()
        pool.join()
        
        # while (time.time() - timeStamp) < waitingForSaving:
        #     time.sleep(0.5)

        subMCLst = []
        for argIn in argInLst:
            tmp = np.load(f'{tmpDir}/{argIn[1]}.npz',allow_pickle=True)
            subMC,_,_ = tmp['mcTrack'],tmp['setting'][()],tmp['obs'][()]
            subMCLst.append(subMC)
        os.system(f'rm -r {tmpDir}')
        mcTrack = np.concatenate(subMCLst,axis=0)
        os.makedirs(outdir,exist_ok=True)
        np.savez_compressed(f'{outdir}/{pid}.npz',mcTrack=mcTrack,
                            setting=dict(self.initMod.toYML()),obs=self.obs,invMeta={
                                'pid':pid, 'chainL':chainL
                                })
        if verbose:
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
            return 88888,88888,0
        cO = self.obs['c']
        if not np.ma.isMaskedArray(cO):
            cO = np.ma.masked_array(cO)
        uncer = self.obs['uncer']

        N = cO.count()
        # chiSqr = (((cO - cP)/uncer)**2).sum()
        bias = (cO - cP)/uncer
        bias1 = bias[T<=40]
        bias2 = bias[T>40]
        if not np.all(bias1.mask) and not np.all(bias2.mask):
            chiSqr = ((bias1**2).mean() + (bias2**2).mean())/2*N
        elif np.all(bias1.mask) and not np.all(bias2.mask):
            chiSqr = (bias2**2).mean()*N
        elif not np.all(bias1.mask) and np.all(bias2.mask):
            chiSqr = (bias1**2).mean()*N
        else:
            raise ValueError('All observations are masked???')

        misfit = np.sqrt(chiSqr/N)
        chiSqr =  chiSqr if chiSqr < 50 else np.sqrt(chiSqr*50.) 
        L = np.exp(-0.5 * chiSqr)
        return misfit,chiSqr,L

class PostPoint(Point):
    def __init__(self,npzMC=None,npzPriori=None,
                 modelTypeCustom=None,layerClassCustom={},
                 trueMarkovChain=True):
        if npzMC is not None:
            tmp = np.load(npzMC,allow_pickle=True)
            self.MC,setting,self.obs = tmp['mcTrack'],tmp['setting'][()],tmp['obs'][()]
            self.invMeta = tmp['invMeta'][()]
            self.initMod = buildModel1D(setting,modelTypeCustom=modelTypeCustom,
                                        layerClassCustom=layerClassCustom)
                
            self.N       = self.MC.shape[0]
            self.misfits = self.MC[:,0]
            self.Ls      = self.MC[:,1]
            self.accepts = self.MC[:,2]
            self.MCparas = self.MC[:,3:]
            self.MCparas_pri = None

            if trueMarkovChain:
                for i in range(self.N):
                    if self.accepts[i]:
                        iAcc = i
                    else:
                        self.MCparas[i,:] = self.MCparas[iAcc,:]

            indMin = np.nanargmin(self.misfits)
            self.minMod         = self.initMod.copy()
            self.minMod._loadMC(self.MCparas[indMin])
            self.minMod.L       = self.Ls[indMin]
            self.minMod.misfit  = self.misfits[indMin]

            self.thres  = self._thres(self.minMod.misfit)
            self.accFinal = (self.misfits < self.thres)

            self.avgMod         = self.initMod.copy()
            self.avgMod._loadMC(np.mean(self.MCparas[self.accFinal,:],axis=0))
            
            self.avgMod.misfit,_,self.avgMod.L = self.misfit(model=self.avgMod)

        if npzPriori is not None:
            tmp = np.load(npzPriori,allow_pickle=True)['mcTrack']
            self.MCparas_pri = tmp[:,3:]
    @staticmethod
    def _thres(minMisfit):
        return max(minMisfit*2, minMisfit+0.5)

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
    def plotVsProfileShaded(self):
        indFinAcc = np.where(self.accFinal)[0]
        zdeps = np.linspace(0,200,200)
        # zdeps = postp.avgMod.seisPropGrids()[0]
        allVs = np.zeros([len(zdeps),len(indFinAcc)])

        mod = self.avgMod.copy()
        for i,ind in enumerate(indFinAcc):
            mod._loadMC(self.MCparas[ind,:])
            allVs[:,i] = mod.value(zdeps)
        std = allVs.std(axis=1)

        ax = self.initMod.plotProfileGrid(label='Initial',alpha=0.2)
        plt.axes(ax); fig = plt.gcf()
        fig.set_figheight(8.4);fig.set_figwidth(5)
        avgProfile = self.avgMod.value(zdeps)
        plt.fill_betweenx(zdeps,avgProfile+std,avgProfile-std,facecolor='grey',alpha=0.6)
        self.avgMod.plotProfileGrid(ax=ax,label='Avg')
        plt.xlim(3.0,4.8)
        plt.legend()

    def _check_distribution(self,indVars='all',zdeps=None):
        def loadMC(mod,mc):
            mod._loadMC(mc)
            return mod.copy()
        if zdeps is not None:
            mod = self.initMod.copy()
            accMods = [loadMC(mod,mc) for mc in self.MCparas[self.accFinal]]
            accYs   = np.array([mod.value(zdeps) for mod in tqdm.tqdm(accMods)]).T
            if self.MCparas_pri is not None:
                priMods = [loadMC(mod,mc) for mc in self.MCparas_pri[:]]
                priYs   = np.array([mod.value(zdeps) for mod in tqdm.tqdm(priMods)]).T
            titles = [f'Hist of Vs at {z} km' for z in zdeps]
        else:
            indVars = range(len(self.initMod._brownians())) if indVars == 'all' else indVars
            accYs = [self.MCparas[self.accFinal,ind] for ind in indVars]
            if self.MCparas_pri is not None:
                priYs = [self.MCparas_pri[:,ind] for ind in indVars]
            titles = [f'Parameter index {ind}: {self.accFinal.sum()}/{len(self.accFinal)}' for ind in indVars]

        for i,title in enumerate(titles):
            plt.figure()
            if self.MCparas_pri is not None:
                _,bin_edges = np.histogram(priYs[i],bins=30)
                plt.hist(accYs[i],bins=bin_edges,weights=np.ones_like(accYs[i])/float(len(accYs[i])),
                            fill=True,ec='k',rwidth=0.8)
                plt.hist(priYs[i],bins=bin_edges,weights=np.ones_like(priYs[i])/float(len(priYs[i])),
                            fill=False,ec='k',rwidth=1.0)
            else:
                plt.hist(accYs[i],bins=30)
            plt.title(title)
    def _check_convergency(self,indVars='all',showVarsSpace=False):
        chainL = self.invMeta['chainL']
        indVars = np.arange(len(self.initMod._brownians())) if indVars == 'all' else indVars
        chainLTests = [int(l) for l in np.linspace(chainL/10,chainL,20)]
        yMean = np.zeros([len(indVars),len(chainLTests)])
        yStd  = np.zeros([len(indVars),len(chainLTests)])
        def indChainLTest(chainLTest):
            N = len(self.misfits); iStart = 0
            indices = np.zeros(N,dtype=bool)
            while iStart < N:
                indices[iStart:iStart+chainLTest] = True
                iStart += chainL
            return indices
        for j,chainLTest in enumerate(chainLTests):
            indSteps = indChainLTest(chainLTest)
            thres = self._thres(self.misfits[indSteps].min())
            accInd = (self.misfits<thres) * indSteps
            for i,indVar in enumerate(indVars):
                yMean[i,j] = np.mean(self.MCparas[accInd,indVar])
                yStd[i,j]  = np.std(self.MCparas[accInd,indVar])
        
        varLabels = [f'{i}: {b[1]}-{b[2]}' for i,b in enumerate(self.initMod._brownians(False))]
        varMins   = [b[0].vmin for b in self.initMod._brownians(False)]
        varMaxs   = [b[0].vmax for b in self.initMod._brownians(False)]

        plt.figure()
        for i,indVar in enumerate(indVars):
            ymin,ymax = varMins[indVar],varMaxs[indVar]
            plt.plot(chainLTests,yMean[i],label=varLabels[indVar])
            if showVarsSpace:
                plt.fill_between(chainLTests,ymin,ymax,alpha=0.1)

        plt.legend(); plt.title('Mean')

        plt.figure()
        for i,indVar in enumerate(indVars):
            plt.plot(chainLTests,yStd[i],label=varLabels[indVar])
        plt.legend(); plt.title('Standard Deviation')
    def _check_history(self,yType='ksquare'):
        plt.figure()
        if yType == 'ksquare':
            y = self.misfits**2*len(self.obs['T'])
            thres = self.thres**2*len(self.obs['T'])
        elif yType == 'likelihood':
            y = self.Ls; thres = None
        elif yType == 'misfit':
            y = self.misfits; thres = self.thres
        else:
            raise ValueError(f'Unsupported type of y: {yType}')
        plt.plot(y)
        ind = np.where(self.accepts.astype(bool))[0]
        plt.plot(ind,y[ind],'or')
        if thres:
            plt.plot([0,self.N],[thres,thres],'--g')
 
    # Unused
    def _model1D_generator(self,N=None,isRandom=True,accFinal=True):
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


class PostPointCascadia(PostPoint):
    misfit = PointCascadia.misfit

if __name__ == '__main__':
    setting = {
        'OceanWater'            : {'H':2},
        'OceanSedimentCascadia' : {'H':[1,'rel_pos',100,0.1]},
        'OceanCrust'            : {'H':7, 'Vs':[3.25, 3.94]},
        'OceanMantleHybrid'     : {'BottomDepth':200, 
                                   'Conversion':'Ritzwoller',
                                   'ThermAge':[4,'rel_pos',200,0.4],
                                   'Vs': [[0, 'abs', 0.4, 0.01],
                                          [0, 'abs', 0.4, 0.01],
                                          [0, 'abs', 0.4, 0.01],
                                          [0, 'abs', 0.2, 0.01]]
                                   },
        'Info':{
            'modelType' : 'CascadiaOcean',
            'period'    : 10,
            'refLayer'  : True,
            'lithoAgeQ' : True
        }
    }

    p = PointCascadia(setting,localInfo={
        'topo':-2.567706,
        'lithoAge':0.6,
        'sedthk':0.019,
        'mantleInitParmVs':[-0.3426920324186606,-0.1863907997418917,
                            -0.1882828662382096,-0.05648363217566826]
        },
        periods = [10,12,14,16,18,20,22,24,26,28,30,32,36,40,50,60,70,80],
        vels    = [3.5724066175576223, 3.6222019289297043, 3.6520621581430763, 3.6588731735179367,
                   3.673255450218663,  3.683443600610537,  3.6844591498161896, 3.689993791502759,
                   3.6935745493241487, 3.696092260762209,  3.707185398688356,  3.7148258328900985,
                   3.7209668755498257, 3.7486729577980427, 3.7706463827824748, 3.82144353111797,
                   3.8603954933518914, 3.9030011211762767],
        uncers  = [0.006550350458769691, 0.005, 0.005, 0.005,
                   0.005, 0.005, 0.005, 0.005, 
                   0.005, 0.005, 0.005, 0.005499996722895128, 
                   0.00751713560920708, 0.007910350806141024, 0.007711019920661203, 0.010152973423528881,
                   0.01062776863809981, 0.015829560954127662]
        )
    p.MCinvMP(f'test',pid='test',runN=24000,chainL=800,nprocess=20)
    p.MCinvMP(f'test_priori',pid='test',runN=24000,chainL=800,nprocess=20,priori=True)

    # postp = PostPointCascadia('test/test.npz','test_priori/test.npz')
    # postp.plotDisp()
    # postp.plotVsProfileGrid()
    # postp.plotVsProfileShaded()
    # postp._check_distribution([1])
    # postp._check_convergency(indVars=[1,3])
    # postp._check_history()

    pass