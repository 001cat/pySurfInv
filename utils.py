import numpy as np
from copy import deepcopy
from netCDF4 import Dataset
from geographiclib.geodesic import Geodesic
from Triforce.utils import GeoGrid,GeoMap
from Triforce.pltHead import *


def _dictIterModifier(d,checker,modifier):
    if type(d) is dict:
        dOut = {}
        for k,v in d.items():
            if checker(v):
                dOut[k] = modifier(v)
            elif type(v) in (dict,list):
                dOut[k] = _dictIterModifier(v,checker,modifier)
            else:
                dOut[k] = v
    elif type(d) is list:
        dOut = []
        for v in d:
            if checker(v):
                dOut.append(modifier(v))
            elif type(v) in (dict,list):
                dOut.append(_dictIterModifier(v,checker,modifier))
            else:
                dOut.append(v)
    else:
        dOut = d
    return dOut

def plotLayer(h,v,fig=None,ax=None,label=None,**kwargs):
    if ax is None:
        fig = plt.figure(figsize=[5,7]);ax = plt.gca()
    else:
        plt.sca(ax); fig = plt.gcf()
    hNew = np.insert(np.repeat(np.cumsum(h),2)[:-1],0,0)
    vNew = np.repeat(v,2)
    ax.plot(vNew,hNew,label=label,**kwargs)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    return ax

def plotGrid(zdepth,v,fig=None,ax=None,label=None,**kwargs):
    if ax is None:
        fig = plt.figure(figsize=[5,7]);ax = plt.gca()
    else:
        plt.sca(ax); fig = plt.gcf()
    plt.plot(v,zdepth,label=label,**kwargs)
    ax = ax or fig.axes[0]
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    return ax


## not in use
def plotCascadiaSlab(lon1,lat1,lon2,lat2):
    from geographiclib.geodesic import Geodesic
    with Dataset('/home/ayu/Projects/Cascadia/Models/Slab2_Cascadia/cas_slab2_dep_02.24.18.grd') as dset:
        slabDep = GeoMap(dset['x'][()],dset['y'][()],-dset['z'][()])
    with Dataset('/home/ayu/Projects/Cascadia/Models/Slab2_Cascadia/cas_slab2_thk_02.24.18.grd') as dset:
        slabThk = GeoMap(dset['x'][()],dset['y'][()],dset['z'][()])
    with Dataset('/home/ayu/Projects/Cascadia/Models/Slab2_Cascadia/cas_slab2_dip_02.24.18.grd') as dset:
        slabDip = GeoMap(dset['x'][()],dset['y'][()],dset['z'][()])
    geoDict = Geodesic.WGS84.Inverse(lat1,lon1,lat2,lon2)
    slabU = []
    xslabD,slabD = [],[]
    x = np.linspace(0,geoDict['s12'],301)/1000
    for i,d in enumerate(x*1000):
        tmp = Geodesic.WGS84.Direct(lat1,lon1,geoDict['azi1'],d)
        slabU.append(slabDep.value(tmp['lon2'],tmp['lat2']))
        theta = slabDip.value(tmp['lon2'],tmp['lat2'])/180*np.pi
        H = slabThk.value(tmp['lon2'],tmp['lat2'])
        xslabD.append(d/1000-H*np.sin(theta))
        slabD.append(slabDep.value(tmp['lon2'],tmp['lat2'])+H*np.cos(theta))
    if abs(lon1-lon2)<0.01:
        xslabD = np.interp(xslabD,x,np.linspace(lat1,lat2,301))
        x = np.linspace(lat1,lat2,301)
    elif abs(lat1-lat2)<0.01:
        xslabD = np.interp(xslabD,x,np.linspace(lon1,lon2,301))
        x = np.linspace(lon1,lon2,301)
    plt.plot(x,slabU,'w',lw=4)
    plt.plot(xslabD,slabD,'w',lw=4)

def plotCascadiaTrenchCoast(lon1,lat1,lon2,lat2):
    lon1 = lon1-360*(lon1>180)
    lon2 = lon2-360*(lon2>180)
    from shapely.geometry import LineString
    from geographiclib.geodesic import Geodesic 
    a = LineString(np.loadtxt('prism.csv',delimiter=','))
    b = LineString([(lon1,lat1),(lon2,lat2)])
    p1,p2 = a.intersection(b).geoms
    if p1.x > p2.x:
        p1,p2 = p2,p1
    for p in [p1,p2]:
        if abs(lon1-lon2)<0.01:
            x = p.xy[1][0]
        elif abs(lat1-lat2)<0.01:
            x = p.xy[0][0]+360
        else:
            x = Geodesic.WGS84.Inverse(lat1,lon1,p.xy[1][0],p.xy[0][0])['s12']/1000
        plt.plot([x,x],[0,200],'--',c='r',lw=0.5)
    
def plotVolcanos(lon1,lat1,lon2,lat2):
    pass

def plotMORLocation(lon1,lat1,lon2,lat2):
    lon1 = lon1-360*(lon1>180)
    lon2 = lon2-360*(lon2>180)
    import geopandas as gpd
    from shapely.geometry import LineString
    df = gpd.read_file('/home/ayu/Projects/Cascadia/Models/Plates/GeoJSON/PB2002_boundaries.json')
    mor = df.loc[df['Name']=='PA-JF'].geometry.values[0]
    p = mor.intersection(LineString([(lon1,lat1),(lon2,lat2)]))

    # get x coordinate in ax, only lat, only lon or using distance
    from geographiclib.geodesic import Geodesic 
    if abs(lon1-lon2)<0.01:
        x = p.xy[1][0]
    elif abs(lat1-lat2)<0.01:
        x = p.xy[0][0]+360
    else:
        x = Geodesic.WGS84.Inverse(lat1,lon1,p.xy[1][0],p.xy[0][0])['s12']/1000
    plt.plot(x,0,'^',markersize=10,markerfacecolor='r',clip_on=False,zorder=100)

def plotCascadiaSlab4Map(m,levels=[60,75,90,120,150]):
    with Dataset('/home/ayu/Projects/Cascadia/Models/Slab2_Cascadia/cas_slab2_dep_02.24.18.grd') as dset:
        slabDep = GeoMap(dset['x'][()],dset['y'][()],-dset['z'][()])
    XX,YY = np.meshgrid(slabDep.lons-360,slabDep.lats)
    cs = m.contour(XX,YY,slabDep.z,latlon=True,levels=levels,colors='white',linewidths=2)
    plt.clabel(cs, fontsize=9, inline=True,colors='k')

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
    import yaml
    from brownian import BrownianVar
    def checker(v):
        return type(v) is list and len(v)>=2 \
            and v[1] in ('fixed','total','abs','rel')
    def modifier(v):
        if v[1] in ('fixed','total'):
            return v[0]
        elif v[1] == 'abs':
            return BrownianVar(v[0],v[0]-v[2],v[0]+v[2],v[3])
        elif v[1] == 'rel':
            return BrownianVar(v[0],v[0]*(1-v[2]/100),v[0]*(1+v[2]/100),v[3])
        else:
            raise ValueError(f'Error: Wrong checker??? v={v}')
    with open('setting-Hongda.yml', 'r') as f:
        d = _dictIterModifier(yaml.load(f,Loader=yaml.FullLoader),checker,modifier)
