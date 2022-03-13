import numpy as np
from pySurfInv.point import Point,PostPoint
from pySurfInv.OceanSeis import OceanSeisRitz,OceanSeisRuan,HSCM
from Triforce.pltHead import *

plt.figure()
seisMod0 = OceanSeisRitz(HSCM(5))
for age in [3,4,5,6,7]:
    seisMod = OceanSeisRitz(HSCM(age))
    plt.plot(seisMod.zdeps,seisMod.vs-seisMod0.vs,label=f'{age}Ma')
plt.xlim(10,20)

# cascadia, lon = 233.0, lat = 46.0
obs = {
    'T': [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 
          28.0, 30.0, 32.0, 36.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    'c': [3.6577, 3.7798, 3.8266, 3.8448, 3.8588, 3.8681, 3.8766, 
          3.8870, 3.8933, 3.8932, 3.8992, 3.8983, 3.9045, 3.9050, 
          3.9071, 3.9452, 3.9669, 3.9717],
    'uncer': [0.0141, 0.0060, 0.0049, 0.0043, 0.0039, 0.0041, 0.0043, 
              0.0048, 0.0051, 0.0057, 0.0061, 0.0067, 0.0081, 0.0100, 
              0.0133, 0.0168, 0.0260, 0.0305]
}

localInfo = {'topo':-2.7753, 'lithoAge':6.7330, 'sedthk':1.9960}

p = Point('cascadia-ocean.yml',localInfo,periods=obs['T'],vels=obs['c'],uncers=obs['uncer'])
p.initMod._layers[-1].parm['Age'] = float(p.initMod._layers[-1].parm['Age'])
p.MCinvMP(pid='testDeltaAge',runN=50000,step4uwalk=1000,nprocess=17)
postp = PostPoint('MCtest/testDeltaAge.npz')

plt.figure()
z = np.linspace(5,30,200)
vs0 = postp.initMod.value(z+localInfo['sedthk']+(-localInfo['topo']))
plt.plot(z,vs0-vs0)
vs1 = postp.avgMod.value(z+localInfo['sedthk']+(-localInfo['topo']))
plt.plot(z,vs1-vs0)
plt.xlim(10,20)


p = Point('cascadia-ocean.yml',localInfo,periods=obs['T'],vels=obs['c'],uncers=obs['uncer'])
p.MCinvMP(pid='testEstimateAge',runN=50000,step4uwalk=1000,nprocess=17)
postp = PostPoint('MCtest/testEstimateAge.npz')
plt.figure();plt.scatter(postp.MCparas[postp.accFinal,-1],postp.MCparas[postp.accFinal,2])


vs15Table = {'age':np.linspace(1,20,1000)}
vs15Table['vs'] = np.array([ OceanSeisRitz(HSCM(age,np.linspace(12,18,20))).vs.mean()/1000 
                             for age in vs15Table['age']])
ageEstimate = np.zeros(postp.MCparas.shape[0])
import tqdm
for i,param in tqdm.tqdm(enumerate(postp.MCparas)):
    mod = postp.initMod.copy()
    mod._loadMC(param)
    vs15 = mod.value(np.linspace(12,18,20)+localInfo['sedthk']+(-localInfo['topo'])).mean()
    ageEstimate[i] = np.interp(vs15,vs15Table['vs'],vs15Table['age'])
plt.figure();plt.scatter(postp.MCparas[postp.accFinal,-1],ageEstimate[postp.accFinal])
plt.figure();plt.hist(ageEstimate[postp.accFinal],bins=40)

postp.plotVsProfileGrid()
mod = postp.initMod.copy()
mod._layers[-1].parm['Age'] = 5.5
z,vs,_,_,_,_,_ = mod.seisPropGrids()
plt.plot(vs,z)
