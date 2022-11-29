import numpy as np
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic
from Triforce.utils import GeoGrid,GeoMap
from pySurfInv.utils import plotGrid,plotLayer
from copy import deepcopy

class Model1D_Exchange():
    def __init__(self,parm,info={}) -> None:
        if 'h' in parm:
            self.type = 'layer'
        if 'z' in parm:
            self.type = 'grid'
        self.parm     = deepcopy(parm)
        self.info     = deepcopy(info)
    def propGrids(self,parName='pySurfInv'):
        if parName == 'pySurfInv':
            raise ValueError('Not ready yet!')
        if self.type == 'grid':
            return (self.parm['z'],self.parm[parName])
        elif self.type == 'layer':
            h = self.parm['h']
            z = np.zeros(2*len(h)) + self.info.get('z0',0)
            v = np.zeros(len(h)*2)
            z[1::2] += h.cumsum() 
            z[2::2] += h[:-1].cumsum()
            v[::2]   = self.parm[parName][:]
            v[1::2]  = self.parm[parName][:]
        return z,v
    def propLayers(self,parName='pySurfInv'):
        if parName == 'pySurfInv':
            raise ValueError('Not ready yet!')
        if self.type == 'layer':
            return (self.parm['h'],self.parm[parName])
        elif self.type == 'grid':
            h = np.diff(self.parm['z'])
            v = (self.parm[parName][1:] + self.parm[parName][:-1])/2
            return (h,v)
    def value(self,zdeps,parName='vs'):
        z,v = self.propGrids(parName)
        return np.interp(zdeps,z,v,left=np.nan,right=np.nan)

    def plotLayers(self,parName='vs',ax=None,**kwargs):
        h,v = self.propLayers(parName)
        ax = plotLayer(h,v,ax=ax,**kwargs);plt.title(parName)
        return ax
    def plotGrids(self,parName='vs',ax=None,**kwargs):
        z,v = self.propGrids(parName)
        ax = plotGrid(z,v,ax=ax,**kwargs);plt.title(parName)
        return ax

    def copy(self):
        return deepcopy(self)

class Model3D_Exchange():
    def __init__(self,fname=None,lons=[],lats=[]) -> None:
        if fname:
            self.load(fname)
        else:
            self.grid = GeoGrid(lons,lats)
            self.mods = [[None for _ in range(len(self.lons))] for _ in range(len(self.lats))]
    def addMod(self,lon,lat,mod):
        i,j = self.grid._findInd(lon,lat)
        self.mods[i][j] = mod.copy()
    def getMod(self,lon,lat,parName,zdeps=None):
        def _get_z_v(mod,zdeps):
            if zdeps is None:
                return mod.propGrids(parName)
            else:
                return zdeps,mod.value(zdeps,parName)
            
        ind = self.grid._findInd_linear_interp(lon,lat)
        if ind is None:
            return np.nan
        if len(ind) == 2:
            i,j = ind
            try:
                mod = self.mods[i][j]
                z,v = _get_z_v(mod,zdeps)
                return Model1D_Exchange({'z':z,parName:v})
            except KeyError:
                return np.nan
        try:
            i,j,dx,dy,Dx,Dy = ind
            z0,v0 = _get_z_v(self.mods[i-1][j-1],zdeps)
            z1,v1 = _get_z_v(self.mods[i][j-1],zdeps)
            z2,v2 = _get_z_v(self.mods[i-1][j],zdeps)
            z3,v3 = _get_z_v(self.mods[i][j],zdeps)
            z = z0+(z1-z0)*dy/Dy+(z2-z0)*dx/Dx+(z0+z3-z1-z2)*dx*dy/Dx/Dy
            v = v0+(v1-v0)*dy/Dy+(v2-v0)*dx/Dx+(v0+v3-v1-v2)*dx*dy/Dx/Dy
            return Model1D_Exchange({'z':z,parName:v})
        except KeyError:
            return np.nan
    def copy(self):
        return deepcopy(self)


    def getMap(self,z,parName):
        grid = self.grid.copy()
        v = np.nan*np.ones(grid.XX.shape)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if self.mods[i][j]:
                    v[i,j] = self.mods[i][j].value(z,parName)
        return GeoMap(grid.lons,grid.lats,v)

    def getSection(self,lat1,lon1,lat2,lon2,parName,y=np.linspace(0,200-0.01,201),xtype='auto'):
        geoDict = Geodesic.WGS84.Inverse(lat1,lon1,lat2,lon2)
        x = np.linspace(0,geoDict['s12'],301)/1000
        z = np.zeros((len(y),len(x)))
        for i,d in enumerate(x*1000):
            tmp = Geodesic.WGS84.Direct(lat1,lon1,geoDict['azi1'],d)
            z[:,i] = self.getMod(tmp['lon2'],tmp['lat2'],parName,y).parm[parName]
        z = np.ma.masked_array(z,np.isnan(z))
        if xtype == 'lat' or (xtype == 'auto' and abs(lon1-lon2)<0.01):
            x = np.linspace(lat1,lat2,301)
        elif xtype == 'lon' or (xtype == 'auto' and abs(lat1-lat2)<0.01):
            x = np.linspace(lon1,lon2,301)
        elif xtype in ('km','auto'):
            pass
        else:
            raise ValueError(f'Wrong xtype: {xtype}')
        XX,YY = np.meshgrid(x,y)
        return XX,YY,z
    

    @property
    def lons(self):
        return self.grid.lons
    @property
    def lats(self):
        return self.grid.lats

    def save(self,fname):
        m,n = len(self.mods),len(self.mods[0])
        mods = [[None for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                mod = self.mods[i][j]
                mods[i][j] = mod if mod is None else [mod.parm,mod.info]
        np.savez_compressed(fname,model={'lons':self.lons,'lats':self.lats,'mods':mods})

    def load(self,fname):
        mod3D = np.load(fname,allow_pickle=True)['model'][()]
        lons,lats,mods = mod3D['lons'],mod3D['lats'],mod3D['mods']
        self.grid = GeoGrid(lons,lats)
        m,n = len(mods),len(mods[0])
        self.mods = np.array([[None for _ in range(n)] for _ in range(m)])
        for i in range(m):
            for j in range(n):
                mod = mods[i][j]
                self.mods[i][j] = mod if mod is None else Model1D_Exchange(mod[0],mod[1])



if __name__ == '__main__':
    pass

    # from tqdm import tqdm
    # from Triforce.pltHead import *
    # mod3D = Model3D_Exchange(lons=np.arange(-136.0,-118.0+0.1+1,0.2),lats=np.arange(36,54+0.1,0.2))
    # mat = [[[] for _ in range(mod3D.grid.XX.shape[1])] for _ in range(mod3D.grid.XX.shape[0])] 
    # with open('/home/ayu/Projects/Cascadia-Land/Tasks/OtherStudies/bell2016/jgrb51815-sup-0002-supplementary.txt','r') as f:
    #     for l in tqdm(f.readlines()):
    #         strs = l.split()
    #         if len(strs) == 2:
    #             lon,lat = float(strs[0]),float(strs[1])
    #             i,j = mod3D.grid._findInd(lon,lat)
    #         if len(strs) == 4:
    #             mat[i][j].append([float(s) for s in strs])
    # for i,lat in enumerate(mod3D.grid.lats):
    #     for j,lon in enumerate(mod3D.grid.lons):
    #         if mat[i][j]:
    #             h,rho,vp,vs = np.array(mat[i][j]).T
    #             mod = Model1D_Exchange({'h':h,'rho':rho,'vp':vp,'vs':vs})
    #             mod3D.addMod(lon,lat,mod)
            
    # mod3D.save('test.npz')
    # mod3D = Model3D_Exchange('test.npz')

    # vsMap = mod3D.getMap(60,'vs')
    # from Triforce.customPlot import plotBasemap_Cascadia,cvcpt
    # _,m = plotBasemap_Cascadia()
    # m.pcolormesh(vsMap.XX,vsMap.YY,vsMap.z,shading='gouraud',latlon=True,cmap=cvcpt)
    # m.drawcoastlines();plt.colorbar()


    # XX,YY,Z = mod3D.getSection(45,-130,45,-122,'vs')
    # plt.figure()
    # plt.pcolormesh(XX,YY,Z,shading='gouraud',cmap=cvcpt,vmin=4.0,vmax=4.5)
    # plt.gca().invert_yaxis()
    # plt.colorbar()
