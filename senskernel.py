import os,tempfile
import numpy as np
import pandas as pd
from Triforce.pltHead import *
from pySurfInv.models import Model1D

PREMcsv = os.path.dirname(__file__)+'/senskernel-1.0/Ayu/PREM_senskernel.csv'

class SensKernel():
    def __init__(self,model=None,wtype='R',Tmin=20,Tmax=100,Tstep=10,endmode=0,dz=2) -> None:
        if model is None:
            self.model = pd.read_csv(PREMcsv)
        elif type(model) == str:
            self.model = pd.read_csv(model)
        elif type(model) == pd.core.frame.DataFrame:
            self.model = model.copy()
        else:
            raise ValueError(f'Wrong model input: {model}')
        
        # print(self.model)
        
        if wtype == 'R':
            nCol = 3    # sen-Vs sen-Vp sen-rho
            self.xtype = ['Vs','Vp','Rho']
        elif wtype == 'L':
            nCol = 2    # sen-Vs sen-rho
            self.xtype = ['Vs','Rho']
        else:
            raise ValueError('Wrong surface wave type!')

        self.zdeps = np.arange(0,self.model['H'].sum(),dz)
        self.wtype = wtype
        self.periods = range(Tmin,Tmax+Tstep//2,Tstep)
        self.kernel_phv = np.zeros((endmode+1,nCol,len(self.periods),len(self.zdeps)))*np.nan
        self.kernel_grv = np.zeros((endmode+1,nCol,len(self.periods),len(self.zdeps)))*np.nan

        binDir = os.path.dirname(os.path.realpath(__file__))+'/senskernel-1.0/bin'
        with tempfile.TemporaryDirectory() as tmpDir:
            modPath = f'{tmpDir}/example.mod'
            outPath = f'{tmpDir}/example'
            wType   = self.wtype
            startMode = 0
            endMode = endmode
            startT  = Tmin
            endT    = Tmax
            dT      = Tstep


            self.model.to_csv(modPath,sep=' ',header=False,index=False)
            os.system(f'{binDir}/SURF_PERTURB {modPath} {outPath} {wType} {startMode} {endMode} '+ 
                      f'{startT} {endT} {dT} -s {dz} -a -f -p 1.0')
            os.system(f'{binDir}/PHV_SENS_KERNEL {modPath} {outPath} {wType} {outPath}.phv')

            os.system(f'{binDir}/SURF_PERTURB {modPath} {outPath}- {wType} {startMode} {endMode} '+ 
                      f'{startT} {endT} {dT} -s {dz} -a -f -p 0.99')
            os.system(f'{binDir}/PHV_SENS_KERNEL {modPath} {outPath}- {wType} {outPath}-.phv')
            os.system(f'{binDir}/SURF_PERTURB {modPath} {outPath}+ {wType} {startMode} {endMode} '+ 
                      f'{startT} {endT} {dT} -s {dz} -a -f -p 1.01')
            os.system(f'{binDir}/PHV_SENS_KERNEL {modPath} {outPath}+ {wType} {outPath}+.phv')
            os.system(f'{binDir}/GRV_SENS_KERNEL {outPath} {endMode} {wType}')

            for mode in range(startMode,endMode+1):
                for iper,per in enumerate(self.periods):
                    for kernel,ytype in zip([self.kernel_phv,self.kernel_grv],['phv','grv']):
                        with open(f'{tmpDir}/example.{ytype}.{wtype}_{mode}_{per}','r') as f:
                            data = []
                            for l in f.readlines():
                                data.append([float(s) for s in l.split()[1:nCol+1]])
                            data = np.array(data)
                            nRow = min(len(self.zdeps),data.shape[0])
                            for iCol in range(nCol):
                                kernel[mode,iCol,iper,:nRow] = data[:nRow,iCol]
    def plot(self,mode=0,per=None,ytype='phv',xtype='Vs'):
        if ytype == 'phv':
            kernel = self.kernel_phv
        elif ytype == 'grv':
            kernel = self.kernel_grv
        else:
            raise ValueError()
        ixtype = self.xtype.index(xtype)
        
        fig,axes = plt.subplots(1,1,figsize=[6,8])
        for iper,per in enumerate(self.periods):
            plt.plot(kernel[mode,ixtype,iper,:],self.zdeps,label=f'{per}s')
        plt.gca().invert_yaxis()
        plt.legend()

class sensModel():
    def __init__(self,df) -> None:
        self._df = df.copy()
        self.H = df['H']
        self.Vs = df['Vs']
        self.Grp = df.get('Grp',None)
    @property
    def Vp(self):
        return self._df.get('Vp',self._convert()[0])
    @property
    def Rho(self):
        return self._df.get('Rho',self._convert()[1])
    @property
    def Qs(self):
        return self._df.get('Qs',self._convert()[2])
        pass
    def _convert(self):
        if self.Grp is None:
            return None,None,None
        Vp,Rho,Qs = np.zeros(len(self.H)),np.zeros(len(self.H)),np.zeros(len(self.H))
        for i,grp in enumerate(self.Grp):
            if grp == 'water':
                Vp[i]  = 1.475
                Rho[i] = 1.027
                Qs[i]  = 10000
            elif grp == 'sediment':
                Vp[i]  = self.Vs[i]*1.23 + 1.28
                Rho[i] = 0.541 + 0.3601*Vp[i]
                Qs[i]  = 80
            elif grp == 'crust':
                Vp[i]  = self.Vs[i]*1.8
                Rho[i] = 0.541 + 0.3601*Vp[i]
                Qs[i]  = 350
            elif grp == 'mantle':
                Vp[i]  = self.Vs[i]*1.76
                Rho[i] = 3.4268+(self.Vs[i]-4.5)/4.5
                Qs[i]  = 150
        return Vp,Rho,Qs
    def copy(self):
        from copy import deepcopy
        return deepcopy(self)
class SensKernelPert():
    def __init__(self,model=None,wtype='R',Tmin=20,Tmax=100,Tstep=10,dz=2) -> None:
        if model is None:
            self.df = pd.read_csv(PREMcsv)
        elif type(model) == str:
            self.df = pd.read_csv(model)
        elif type(model) == pd.core.frame.DataFrame:
            self.df = model.copy()
        else:
            raise ValueError(f'Wrong model input: {model}')
        self.model = sensModel(self.df)
        # self.fineModel = self._fineModel(dz)
        self.wtype = wtype
        self.periods = range(Tmin,Tmax+Tstep//2,Tstep)

        self.kernel = {}
        self.kernel['Vs'] = np.zeros((len(self.periods),len(self.model.H)))
        for i in range(len(self.model.H)):
            v0 = self._forward()
            vL = self._forward(self._perturb(i,pert=0.999))
            vH = self._forward(self._perturb(i,pert=1.001))
            self.kernel['Vs'][:,i] = (vH-vL)/0.2/self.model.H[i]

        if 'Vp' in self.df.keys():
            self.kernel['Vp'] = np.zeros((len(self.periods),len(self.model.H)))
            for i in range(len(self.model.H)):
                v0 = self._forward()
                vL = self._forward(self._perturb(i,pert=0.999,xtype='Vp'))
                vH = self._forward(self._perturb(i,pert=1.001,xtype='Vp'))
                self.kernel['Vp'][:,i] = (vH-vL)/0.2/self.model.H[i]

    def _perturb(self,ilayer,pert=1.0,xtype='Vs') -> pd.core.frame.DataFrame:
        model = self.model.copy()
        if xtype == 'Vs':
            model.Vs[ilayer] *= pert
        else:
            model._df[xtype][ilayer] *= pert
        return model
    # def _fineModel(self,dz):
    #     fineModel = []
    #     for _,row in self.model.iterrows():
    #         n = int(row['H']//dz)
    #         v = row.values.copy(); v[0] /= n
    #         fineModel.extend([v]*n)
    #     return pd.DataFrame(fineModel, columns=self.model.columns)

    def _forward(self,model=None):
        import pySurfInv.fast_surf as fast_surf
        model = self.model if model is None else model

        ilvry = {'R':2,'L':1}[self.wtype]
        h,Vs,Vp,rho,qs = model.H,model.Vs,model.Vp,model.Rho,model.Qs
        I = np.where(h>1e-3)[0]
        h,Vs,Vp,rho,qs = h[I],Vs[I],Vp[I],rho[I],qs[I]
        qsinv			= 1./qs
        nper			= len(self.periods)
        per 			= np.zeros(200, dtype=np.float64)
        per[:nper]		= self.periods[:]
        nlay			= h.size
        (ur0,ul0,cr0,cl0) = fast_surf.fast_surf(nlay, ilvry, Vp, Vs, rho, h, qsinv, per, nper)
    
        if np.any(cr0[:nper]<0.01):
            return None
        return cr0[:nper]
    def plot(self,per=None,ytype='phv',xtype='Vs'):
        if ytype == 'phv':
            kernel = self.kernel
        # elif ytype == 'grv':
        #     kernel = self.kernel_grv
        else:
            raise ValueError()
        
        plt.subplots(1,1,figsize=[6,8])
        zdeps = self.model.H.cumsum() - self.model.H/2
        for iper,per in enumerate(self.periods):
            plt.plot(kernel[xtype][iper,:],zdeps,label=f'{per}s')
        plt.gca().invert_yaxis()
        plt.legend()


def plotKernel(mod:Model1D,Tmin=10,Tmax=80,Tstep=10,dz=1):
    H,Vs,Vp,Rho,Qs,_,_ = mod.seisPropLayers()
    df = pd.DataFrame.from_dict({'H':H,'Vp':Vp,'Vs':Vs,'Rho':Rho,'Qs':Qs})
    sens = SensKernel(model=df,Tmin=Tmin,Tmax=Tmax,Tstep=Tstep,dz=dz)
    sens.plot()
    return sens


if __name__ == '__main__':
    # inFile = '/home/ayu/Projects/Cascadia/Tasks/senKernel/sensKernel/PREM.model'
    # df = pd.DataFrame(np.loadtxt(inFile))
    # df.columns = ["H", "Vp", "Vs", "Rho",'Qs']
    # df.to_csv('PREM_senskernel.csv',index=False)

    # s0 = SensKernel()
    # s0.plot()
    # s.plot(xtype='Vp')
    # s.plot(xtype='Rho')

    s1 = SensKernelPert()
    s1.plot()
    s1.plot(xtype='Vp')

    # df = pd.read_csv('PREM_senskernel.csv')
    # df['Grp'] = ['crust']*6 + ['mantle']*(len(df)-6)
    # df = df.drop(columns=['Vp','Rho','Qs'])
    # s2 = SensKernelPert(model=df)

