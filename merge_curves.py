import numpy as np
from pySurfInv.point import Point,PostPoint
from pySurfInv.utils import plotGrid

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

postp = PostPoint('MCtest/testDeltaAge.npz')
zdeps = np.linspace(12,200,200)
vs0 = postp.initMod.value(zdeps=zdeps)
vs1 = postp.avgMod.value(zdeps=zdeps)
fig = plotGrid(zdeps,vs0)
plotGrid(zdeps,vs1,fig=fig)
vsM = merge(zdeps,vs0,vs1,xL=20,xK=30,xH=50,s=1/3)
plotGrid(zdeps,vsM,fig=fig)
plotGrid(zdeps,vsM)
from Triforce.pltHead import *
plt.xlim(4,4.8)
